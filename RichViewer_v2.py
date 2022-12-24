import streamlit as st
import warnings, matplotlib, requests, OpenDartReader, sys, math 
import pandas as pd
import numpy as np
from matplotlib import font_manager, rc
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mplfinance.original_flavor import candlestick2_ohlc
from tqdm import tqdm
from bs4 import BeautifulSoup
from IPython.core.display import display, HTML
from datetime import date, datetime, timedelta
from pykrx import stock
from pandas_datareader import data as pdr
warnings.filterwarnings(action='ignore')
pd.set_option('display.max_columns',None)
pd.set_option('mode.chained_assignment',None) # 경고 끄기
plt.rc('font',family='malgun gothic')
matplotlib.rcParams['axes.unicode_minus'] = False # 그래프 - 폰트 깨짐 해결
rc('font',family=font_manager.FontProperties(fname=r'c:\Windows\Fonts\malgun.ttf').get_name())
display(HTML("<style>.container {width:95% !important;}</style>"))
from plotly.subplots import make_subplots
from plotly import tools

#=============================================================================
#=== matplotlib 한글 깨짐 해결
import matplotlib
from matplotlib import font_manager, rc

plt.rc('font',family='LGSmHaTR')

matplotlib.rcParams['axes.unicode_minus'] = False # 그래프 - 폰트 깨짐 해결
#=============================================================================

#rc('font',family=font_manager.FontProperties(fname=r'c:\Windows\Fonts\LGSmHaTR.ttf').get_name())

#******************************************************************************
#*                            class: FutureUI                                 *
#******************************************************************************
def krxInfo():
#==========================================================================================================================================
#================================================= 1. KRX Information =====================================================================
#==========================================================================================================================================
	def stockCodeDigit(data): # 종목 리스트의 코드 자리수 6자리 변환
		return '0'*(6-len(str(data)))+str(data)

	def signalIndicator(df,factor,buy,sell): # Sell, buy check
		df['trade'] = np.nan
		if buy >= sell:
			df['trade'].mask(df[factor]>buy,'buy',inplace=True)
			df['trade'].mask(df[factor]<sell,'zero',inplace=True)
		else:
			df['trade'].mask(df[factor]<buy,'buy',inplace=True)
			df['trade'].mask(df[factor]>sell,'zero',inplace=True)
		df['trade'].fillna(method='ffill',inplace=True)
		df['trade'].fillna('zero',inplace=True)
		return df['trade']

	def positionDirection(df):
		df['position'] = ''
		df['position'].mask((df['trade'].shift(1)=='zero')&(df['trade']=='zero'),'zz',inplace=True)
		df['position'].mask((df['trade'].shift(1)=='zero')&(df['trade']=='buy'),'zl',inplace=True)
		df['position'].mask((df['trade'].shift(1)=='buy')&(df['trade']=='zero'),'lz',inplace=True)
		df['position'].mask((df['trade'].shift(1)=='buy')&(df['trade']=='buy'),'ll',inplace=True)
		df['position_chart'] = 0
		df['position_chart'].mask(df['trade']=='buy',1,inplace=True)
		return df['position']

	def signalEvaluate(df,cost):
		df['signal_price'] = np.nan
		df['signal_price'].mask(df['position']=='zl',df.iloc[:,0],inplace=True)
		df['signal_price'].mask(df['position']=='lz',df.iloc[:,0],inplace=True)
		record = df[['position','signal_price']].dropna()
		record['rtn'] = 1
		record['rtn'].mask(record['position']=='lz',(record['signal_price']*(1-cost))/record['signal_price'].shift(1),inplace=True)
		record['acc_rtn'] = record['rtn'].cumprod()
		df['signal_price'].mask(df['position']=='ll',df.iloc[:,0],inplace=True)
		df['rtn'] = record['rtn']
		df['rtn'].fillna(1,inplace=True)
		df['daily_rtn'] = 1
		df['daily_rtn'].mask(df['position']=='ll',df['signal_price']/df['signal_price'].shift(1),inplace=True)
		df['daily_rtn'].mask(df['position']=='lz',(df['signal_price']*(1-cost))/df['signal_price'].shift(1),inplace=True)
		df['daily_rtn'].fillna(1,inplace=True)
		df['acc_rtn'] = df['daily_rtn'].cumprod()
		df['acc_rtn_dp'] = round((df['acc_rtn']-1)*100,2)
		df['mdd'] = round(df['acc_rtn']/df['acc_rtn'].cummax(),4)
		df['bm_mdd'] = round(df.iloc[:,0]/df.iloc[:,0].cummax(),4)
		df.drop(columns='signal_price',inplace=True)
		return df

	def bandtoSignal(df,buy,sell): # 모멘템 (A,B), 평균회귀 (D,B)
		df['trade'] = np.nan
		if 'close' in df.columns:
			if buy == 'A':
				df['trade'].mask(df['close']>df['ub'],'buy',inplace=True)
			elif buy == 'B':
				df['trade'].mask((df['ub']>df['close'])&(df['close']>df['center']),'buy',inplace=True)
			elif buy == 'C':
				df['trade'].mask((df['center']>df['close'])&(df['close']>df['lb']),'buy',inplace=True)
			elif buy == 'D':
				df['trade'].mask((df['lb']>df['close']),'buy',inplace=True)
			if sell == 'A':
				df['trade'].mask(df['close']>df['ub'],'zero',inplace=True)
			elif sell == 'B':
				df['trade'].mask((df['ub']>df['close'])&(df['close']>df['center']),'zero',inplace=True)
			elif sell == 'C':
				df['trade'].mask((df['center']>df['close'])&(df['close']>df['lb']),'zero',inplace=True)
			elif sell == 'D':
				df['trade'].mask((df['lb']>df['close']),'zero',inplace=True)
		else:
			if buy == 'A':
				df['trade'].mask(df['종가']>df['ub'],'buy',inplace=True)
			elif buy == 'B':
				df['trade'].mask((df['ub']>df['종가'])&(df['종가']>df['center']),'buy',inplace=True)
			elif buy == 'C':
				df['trade'].mask((df['center']>df['종가'])&(df['종가']>df['lb']),'buy',inplace=True)
			elif buy == 'D':
				df['trade'].mask((df['lb']>df['종가']),'buy',inplace=True)
			if sell == 'A':
				df['trade'].mask(df['종가']>df['ub'],'zero',inplace=True)
			elif sell == 'B':
				df['trade'].mask((df['ub']>df['종가'])&(df['종가']>df['center']),'zero',inplace=True)
			elif sell == 'C':
				df['trade'].mask((df['center']>df['종가'])&(df['종가']>df['lb']),'zero',inplace=True)
			elif sell == 'D':
				df['trade'].mask((df['lb']>df['종가']),'zero',inplace=True)
		df['trade'].fillna(method='ffill',inplace=True)
		df['trade'].fillna('zero',inplace=True)
		return df['trade']

	def signalCombine(df,*cond):
		for i in cond:
			df['trade'].mask((df['trade']=='buy') & (df[i]=='buy'),'buy',inplace=True)
			df['trade'].mask((df['trade']=='zero') | (df[i]=='zero'),'zero',inplace=True)
		return df

	def funcMovingAverage(df):
		maList = ['MA03','MA05','MA10','MA20','MA60','MA120','MA224','MA240','MA360','MA480']
		maDate = [maElement.replace('MA','') for maElement in maList]
		maDate = [int(maElement) for maElement in maDate]
		# Dataframe에 들어온 데이터의 갯수가 Mov_avg_date 어느 영역에 위치하는지 순번 체크
		maCnt = np.searchsorted(maDate,len(df),'left') 
		if len(df) < maDate[0]:
			pass
		else:
			for i in range(maCnt):
				df[maList[i]] = round(df['종가'].rolling(maDate[i]).mean())
		df = df.replace(np.nan,0) # 데이터 중 Nan을 0으로 변환
		for i in range(maCnt):
			df[maList[i]] = df[maList[i]].astype(int)
		return df

	def funcRsi(df,info):
		# 일정 기간동안의 상승폭과 하락폭에대한 상대비율을 시장의 과매수/과매도로 표현 (14일 데이터, 70 과열, 30 침체)
		pd.options.mode.chained_assignment = None
		df.fillna(method='ffill',inplace=True) # 들어온 데이터의 구멍을 메꿔준다
		if len(df) > info[0]:
			df['diff'] = df.iloc[:,0].diff() # 일별 가격차이 계산
			df['au'] = df['diff'].where(df['diff']>0, 0).rolling(info[0]).mean()
			df['ad'] = df['diff'].where(df['diff']<0, 0).rolling(info[0]).mean().abs()
			for i in range(info[0]+1,len(df)):
				df['au'][i] = (df['au'][i-1]*(info[0]-1) + df['diff'].where(df['diff']>0,0)[i])/info[0]
				df['ad'][i] = (df['ad'][i-1]*(info[0]-1) + df['diff'].where(df['diff']<0,0).abs()[i])/info[0]
			df['rsi'] = round(df['au']/(df['au']+df['ad'])*100,2)
			del df['au'], df['ad'], df['diff']
		signalIndicator(df,factor='rsi',buy=rsiInfo[1],sell=rsiInfo[2]) # buy/sell 값 적용..
		positionDirection(df)
		df.rename(columns = {'position_chart':'positionRsi'},inplace=True)
		del df['trade'], df['position']
		return df

	def funcMfi(df): # 'close','diff','open','high','low','volume'
		# 주식의 가격/거래량을 사용하여 주식이 과잉 매도인지 과잉 매수인지를 식별
		# 거래량이 가중된 RSI, 20 이하일 경우에 과잉 매도 (매수 시그널), MFI 값이 80 이상일 경우에는 과잉 매수
		if 'close' in df.columns:
			df['PB'] = (df['close']-df['low'])/(df['high']-df['low'])
			df['TP'] = (df['high']+df['low']+df['close'])/3
			df['PMF'], df['NMF'] = 0, 0
			for i in range(len(df['close'])-1):
				if df.TP.values[i] < df.TP.values[i+1]:
					df.PMF.values[i+1] = df.TP.values[i+1]*df.volume.values[i+1]
					df.NMF.values[i+1] = 0
				else:
					df.NMF.values[i+1] = df.TP.values[i+1]*df.volume.values[i+1]
					df.PMF.values[i+1] = 0
		else:
			df['PB'] = (df['종가']-df['저가'])/(df['고가']-df['저가'])
			df['TP'] = (df['고가']+df['저가']+df['종가'])/3
			df['PMF'], df['NMF'] = 0, 0
			for i in range(len(df['종가'])-1):
				if df.TP.values[i] < df.TP.values[i+1]:
					df.PMF.values[i+1] = df.TP.values[i+1]*df.거래량.values[i+1]
					df.NMF.values[i+1] = 0
				else:
					df.NMF.values[i+1] = df.TP.values[i+1]*df.거래량.values[i+1]
					df.PMF.values[i+1] = 0
		df['MFR'] = (df.PMF.rolling(window=10).sum()/df.NMF.rolling(window=10).sum())
		df['MFI10'] = round(100-100/(1+df['MFR']),2)
		del df['PB'], df['TP'], df['PMF'], df['NMF'], df['MFR']
		return df

	def funcStochastic(df,info):
		# 최근 X일간 주가 범위 중 현재 주가가 얼마나 높이 있는지 확인 (이동평균)
		# 평균회귀 - Slow k가 20보다 낮을 때 매수, 80보다 높을때 매도
		# 모멘템 - Slow k와 Slow D의 차이를 계산하여 양수면 매수, 음수면 매도
		if 'close' in df.columns:
			df['fast_k'] = round((df['close']-df['low'].rolling(info[0]).min())/(df['high'].rolling(info[0]).max()-df['low'].rolling(info[0]).min())*100,2)
		else:
			df['fast_k'] = round((df['종가']-df['저가'].rolling(info[0]).min())/(df['고가'].rolling(info[0]).max()-df['저가'].rolling(info[0]).min())*100,2)
		df['slow_k'] = round(df['fast_k'].rolling(info[1]).mean(),2)
		df['slow_d'] = round(df['slow_k'].rolling(info[2]).mean(),2)
		signalIndicator(df,factor='slow_k',buy=info[3],sell=info[4]) # -----> buy/sell 값 적용..
		positionDirection(df)
		signalEvaluate(df,info[5])
		df.rename(columns = {'slow_k':'stochastic_slow_k'},inplace=True)
		df.rename(columns = {'slow_d':'stochastic_slow_d'},inplace=True)
		df.rename(columns = {'acc_rtn_dp':'stochastic_acc_rtn_dp'},inplace=True)
		df.rename(columns = {'position_chart':'stochastic_position'},inplace=True)
		del df['fast_k'], df['trade'], df['position'], df['rtn'], df['daily_rtn'], df['acc_rtn'], df['mdd'], df['bm_mdd']
		return df

	def funcEnvelop(df,info):
		# 모멘템 - envelop가 상향 돌파 시 매수, envelop 안으로 들어오면 매도
		# 평균회귀 - envelop 하단 매수하여 상단에 올라오면 매도
		if 'close' in df.columns:
			df['center'] = df['close'].rolling(info[0]).mean()
		else:
			df['center'] = df['종가'].rolling(info[0]).mean()
		df['ub'] = df['center']*(1+info[1])
		df['lb'] = df['center']*(1-info[1])
		bandtoSignal(df,buy='D',sell='B')
		positionDirection(df)
		signalEvaluate(df,info[2])
		df.rename(columns = {'center':'envelop_center'},inplace=True)
		df.rename(columns = {'ub':'envelop_ub'},inplace=True)
		df.rename(columns = {'lb':'envelop_lb'},inplace=True)
		df.rename(columns = {'acc_rtn_dp':'envelop_acc_rtn_dp'},inplace=True)
		df.rename(columns = {'position_chart':'envelop_position'},inplace=True)
		del df['trade'], df['position'], df['rtn'], df['daily_rtn'], df['acc_rtn'], df['mdd'], df['bm_mdd']
		return df

	def funcBollinger(df,info):
		# 이동평균 + 변동성 결합 (주가가 변동성 대비 어느 위치) -> 모멘템 (밴드상단), 평균회귀 (밴드하단)
		if 'close' in df.columns:
			df['center'] = round(df['close'].rolling(info[0]).mean(),2)
			df['sigma'] = round(df['close'].rolling(info[0]).std(),2)
		else:
			df['center'] = round(df['종가'].rolling(info[0]).mean(),2)
			df['sigma'] = round(df['종가'].rolling(info[0]).std(),2)
		df['ub'] = round(df['center'] + info[1]*df['sigma'],2)
		df['lb'] = round(df['center'] - info[1]*df['sigma'],2)
		df['s1'] = bandtoSignal(df,buy='A',sell='B')
		df['s2'] = bandtoSignal(df,buy='D',sell='B')
		signalCombine(df,'s1','s2')
		positionDirection(df)
		signalEvaluate(df,info[2])
		df.rename(columns = {'center':'bollinger_center'},inplace=True)
		df.rename(columns = {'ub':'bollinger_ub'},inplace=True)
		df.rename(columns = {'lb':'bollinger_lb'},inplace=True)
		df.rename(columns = {'acc_rtn_dp':'bollinger_acc_rtn_dp'},inplace=True)
		df.rename(columns = {'position_chart':'bollinger_position'},inplace=True)
		del df['trade'],df['sigma'], df['position'], df['rtn'], df['daily_rtn'], df['acc_rtn'], df['mdd'], df['bm_mdd'], df['s1'], df['s2']
		return df

	def funcMacd(df,info):
		# 이동평균수렴확산지수, 12일간의 단기 이동평균에서 26일간의 장기 이평 차이 (양수=매수, 음수=매도)
		if 'close' in df.columns:
			df['ema_short'] = df['close'].ewm(span=info[0]).mean()
			df['ema_long'] = df['close'].ewm(span=info[1]).mean()
		else:
			df['ema_short'] = df['종가'].ewm(span=info[0]).mean()
			df['ema_long'] = df['종가'].ewm(span=info[1]).mean()
		df['macd'] = round(df['ema_short']-df['ema_long'],2)
		df['macd_signal'] = round(df['macd'].ewm(span=info[2]).mean(),2)
		df['macd_oscillator'] = round(df['macd']-df['macd_signal'],2)
		signalIndicator(df,factor='macd',buy=info[3],sell=info[4]) # -----> buy/sell 값 적용..
		positionDirection(df)
		df.rename(columns = {'position_chart':'macd_position'},inplace=True)
		del df['ema_short'], df['ema_long'], df['trade'], df['position']
		signalIndicator(df,factor='macd_oscillator',buy=info[3],sell=info[4]) # -----> buy/sell 값 적용.
		positionDirection(df)
		df.rename(columns = {'position_chart':'macd_oscillator_position'},inplace=True)
		del df['trade'], df['position']
		return df

	def funcIchBalance(df,info): # 일목균형표
		# 상승추세:후행스팬>주가>구름대&기준선, 하락추세: 후행스팬<주가<구름대&기준선
		# Calculate conversion line
		if 'close' in df.columns:
			high_20 = df['high'].rolling(info[0]).max()
			low_20 = df['low'].rolling(info[0]).min()
			df['Ich_conv_line'] = (high_20+low_20)/2
			# Calculate based line
			high_60 = df['high'].rolling(info[1]).max()
			low_60 = df['low'].rolling(info[1]).min()
			df['Ich_base_line'] = (high_60+low_60)/2
			# Calculate leading span A
			df['Ich_lead_spanA'] = ((df.Ich_conv_line+df.Ich_base_line)/2).shift(info[3])
			# Calculate leading span B
			high_120 = df['high'].rolling(info[2]).max()
			low_120 = df['high'].rolling(info[2]).min()
			df['Ich_lead_spanB'] = ((high_120+low_120)/2).shift(info[2])
			# Calculate lagging span
			df['Ich_lagging_span'] = df['close'].shift(-info[3])
		else:
			high_20 = df['고가'].rolling(info[0]).max()
			low_20 = df['저가'].rolling(info[0]).min()
			df['Ich_conv_line'] = (high_20+low_20)/2
			# Calculate based line
			high_60 = df['고가'].rolling(info[1]).max()
			low_60 = df['저가'].rolling(info[1]).min()
			df['Ich_base_line'] = (high_60+low_60)/2
			# Calculate leading span A
			df['Ich_lead_spanA'] = ((df.Ich_conv_line+df.Ich_base_line)/2).shift(info[3])
			# Calculate leading span B
			high_120 = df['고가'].rolling(info[2]).max()
			low_120 = df['고가'].rolling(info[2]).min()
			df['Ich_lead_spanB'] = ((high_120+low_120)/2).shift(info[2])
			# Calculate lagging span
			df['Ich_lagging_span'] = df['종가'].shift(-info[3])
		return df

	def funcSupport(df,current,num1,num2): # 지지선 찾기
		if 'close' in df.columns:
			for i in range(current-num1+1,current+1):
				if df['low'][i] > df['low'][i-1]:
					return 0
			for i in range(current+1,current+num2+1):
				if df['low'][i] < df['low'][i-1]:
					return 0
		else:
			for i in range(current-num1+1,current+1):
				if df['저가'][i] > df['저가'][i-1]:
					return 0
			for i in range(current+1,current+num2+1):
				if df['저가'][i] < df['저가'][i-1]:
					return 0
		return 1 

	def funcResistance(df,current,num1,num2): # 지항선 찾기
		if 'close' in df.columns:
			for i in range(current-num1+1,current+1):
				if df['high'][i] > df['high'][i-1]:
					return 0
			for i in range(current+1,current+num2+1):
				if df['high'][i] < df['high'][i-1]:
					return 0
		else:
			for i in range(current-num1+1,current+1):
				if df['고가'][i] > df['고가'][i-1]:
					return 0
			for i in range(current+1,current+num2+1):
				if df['고가'][i] < df['고가'][i-1]:
					return 0
		return 1

	def funcPriceBound(data):
		while True:
			if len(data) == 1:
				priceSupRes = data; break
			else:
				if not 'i' in locals():
					i, j = 0, 1
				if len(data) == i:
					break
				if (data[i]*0.96 <= data[i+j] <= data[i]*1.04):    
					if len(data) == (i+j+1):
						if not 'priceSupRes' in locals():
							priceSupRes = round(np.mean(data[i:i+j+1]),0); break
						else:
							priceSupRes = np.append(priceSupRes,round(np.mean(data[i:i+j+1]),0)); break
					else:
						j += 1
				else:
					if not 'priceSupRes' in locals():
						priceSupRes = round(np.mean(data[i:i+j]),0)
					else:
						priceSupRes = np.append(priceSupRes,round(np.mean(data[i:i+j]),0))
					i += j
					j = 1
					if len(data) == i+1:
						if not 'priceSupRes' in locals():
							priceSupRes = round(np.mean(data[i:i+j]),0); break
						else:
							priceSupRes = np.append(priceSupRes,round(np.mean(data[i:i+j]),0)); break
		return priceSupRes

	def funcSupportResistance(df): # 저항/지지선의 공통영역 찾기, 비슷한 영역은 평균 적용
		if 'close' in df.columns:
			df = df[df['volume'] != 0]
		else:
			df = df[df['거래량'] != 0]
		df.reset_index(drop=True,inplace=True)
		if len(df) == 0:
			supResDf = 0
		else:
			supResLen = [3,2]
			for row in range(supResLen[0],len(df)-supResLen[1]):
				if not 'supResDfTemp' in locals():
					supResDfTemp = []
				if funcSupport(df,row,supResLen[0],supResLen[1]):
					if 'close' in df.columns:
						supResDfTemp.append((row,df['low'][row],1))
					else:
						supResDfTemp.append((row,df['저가'][row],1))
				if funcResistance(df,row,supResLen[0],supResLen[1]):
					if 'close' in df.columns:
						supResDfTemp.append((row,df['high'][row],2))
					else:
						supResDfTemp.append((row,df['고가'][row],2))
			plotList1 = [x[1] for x in supResDfTemp if x[2]==1]
			plotList2 = [x[1] for x in supResDfTemp if x[2]==2]
			plotList1.sort()
			plotList2.sort()
			for i in range(1,len(plotList1)):
				if len(plotList1) <= i:
					break
				if abs(plotList1[i]-plotList1[i-1])/plotList1[i] <= 0.02:
					plotList1.pop(i)
			for i in range(1,len(plotList2)):
				if len(plotList2) <= i:
					break
				if abs(plotList2[i]-plotList2[i-1])/plotList2[i] <= 0.02:
					plotList2.pop(i)
			
			stock_df_temp = plotList1 + plotList2
			stock_df_temp = sorted(stock_df_temp)
			supResDf = funcPriceBound(stock_df_temp)
		return supResDf

	def funcVolumeProfile(df,num=10): # 사용자가 지정한 기간에 대해 평균단가와 거래량을 통해 매물대 분석 
		if 'close' in df.columns:
			df['avg_cost'] = round(sum([df.close,df.open,df.high,df.low])/4)
			sortValueDf = df.sort_values(by='avg_cost',ascending=True)
			totalVolumeDf = sortValueDf['volume'].sum()
			sortValueDf['vol_percent'] = sortValueDf['volume']/totalVolumeDf*100
		else:
			df['avg_cost'] = round(sum([df.종가,df.시가,df.고가,df.저가])/4)
			sortValueDf = df.sort_values(by='avg_cost',ascending=True)
			totalVolumeDf = sortValueDf['거래량'].sum()
			sortValueDf['vol_percent'] = sortValueDf['거래량']/totalVolumeDf*100
		countDf,dividerBins = np.histogram(sortValueDf['avg_cost'],bins=num)
		dividerBins = dividerBins.round(0)
		volDf = pd.DataFrame(np.zeros(shape=(num,1),dtype=np.int8))
		volDf.rename(columns = {0:'bin_len'},inplace=True)
		volDf['bin_len'] = volDf['bin_len'].astype(float)
		for i in range(len(dividerBins)-1):
			if i == 0:
				volTempDf = sortValueDf[sortValueDf['avg_cost'] >= dividerBins[i]]
			else:
				volTempDf = sortValueDf[sortValueDf['avg_cost'] > dividerBins[i]]
			volTempDf = volTempDf[volTempDf['avg_cost'] <= dividerBins[i+1]]
			volTempSumDf = volTempDf['vol_percent'].sum().round(2)
			volDf['bin_len'][i] = volTempSumDf
		return dividerBins, volDf

	def stockIndices(dateInfo,rsiInfo,stocsaticInfo,envelopInfo,bollingerInfo,macdInfo,IchBalInfo):
		# 코스피: 1001 코스피, 1002 코스피 대형주, 1003 코스피 중형주, 1004 코스피 소형주, 1028 코스피 200, 1034 코스피 100, 1035 코스피 50
		ksIndices = stock.get_index_ohlcv_by_date(str(date.today()-timedelta(days=dateInfo[1]*5)).replace('-',''),str(date.today()-timedelta(days=dateInfo[0])).replace('-',''),"1001")
		# ksIndices = pdr.get_data_yahoo("^KS11",date.today()-timedelta(days=dateInfo[1]),date.today()-timedelta(days=(dateInfo[0]))) # 코스피 지수
		# ksIndices = ksIndices.rename(columns= {'High':'고가','Low':'저가','Open':'시가','Close':'종가','Volume':'거래대금'})
		ksIndices[['고가','저가','시가','종가']] = round(ksIndices[['고가','저가','시가','종가']],2)
		ksIndices['거래량[만]'] = round(ksIndices['거래량']/10000,2)
		ksIndices['거래대금[억]'] = round(ksIndices['거래대금']/100000000,2)
		ksIndices['상장시가총액[조]'] = round(ksIndices['상장시가총액']/1000000000000,2)
	#     ksIndices.drop(['거래량','거래대금','상장시가총액'],axis=1,inplace=True)
		ksIndices = funcMovingAverage(ksIndices)
		ksIndices = funcRsi(ksIndices,rsiInfo)
		ksIndices = funcMfi(ksIndices)
		ksIndices = funcStochastic(ksIndices,stocsaticInfo)
		ksIndices = funcEnvelop(ksIndices,envelopInfo)
		ksIndices = funcBollinger(ksIndices,bollingerInfo)
		ksIndices = funcMacd(ksIndices,macdInfo)
		ksIndices = funcIchBalance(ksIndices,IchBalInfo)
		ksSupRes = funcSupportResistance(ksIndices)
		ksDividerBins, ksVolDf = funcVolumeProfile(ksIndices,num=10)

		# 코스닥: 2001 코스닥, 2002 코스닥 대형주, 2003 코스닥 중형주, 2004 코스닥 소형주, 2203 코스닥 150
		kdIndices = stock.get_index_ohlcv_by_date(str(date.today()-timedelta(days=dateInfo[1]*5)).replace('-',''),str(date.today()-timedelta(days=dateInfo[0])).replace('-',''),"2001")
		# kdIndices = pdr.get_data_yahoo("^KQ11",date.today()-timedelta(days=Date_info[1]),date.today()-timedelta(days=(add_date+Date_info[0]))) # 코스닥 지수
		# kdIndices = kdIndices.rename(columns= {'High':'고가','Low':'저가','Open':'시가','Close':'종가','Volume':'거래금액'})
		kdIndices[['고가','저가','시가','종가']] = round(kdIndices[['고가','저가','시가','종가']],2)
		kdIndices['거래량[만]'] = round(kdIndices['거래량']/10000,2)
		kdIndices['거래대금[억]'] = round(kdIndices['거래대금']/100000000,2)
		kdIndices['상장시가총액[조]'] = round(kdIndices['상장시가총액']/1000000000000,2)
	#     kdIndices.drop(['거래량','거래대금','상장시가총액'],axis=1,inplace=True)
		kdIndices = funcMovingAverage(kdIndices)
		kdIndices = funcRsi(kdIndices,rsiInfo)
		kdIndices = funcMfi(kdIndices)
		kdIndices = funcStochastic(kdIndices,stocsaticInfo)
		kdIndices = funcEnvelop(kdIndices,envelopInfo)
		kdIndices = funcBollinger(kdIndices,bollingerInfo)
		kdIndices = funcMacd(kdIndices,macdInfo)
		kdIndices = funcIchBalance(kdIndices,IchBalInfo)
		kdSupRes = funcSupportResistance(kdIndices)
		kdDividerBins, kdVolDf = funcVolumeProfile(kdIndices,num=10)

	#     diffDate = date.today()-datetime.date(kdIndices.index[len(kdIndices)-dateInfo[2]])
		# stockGovnSum = stock.get_market_net_purchases_of_equities_by_ticker(str(date.today()-(diffDate+timedelta(days=(1+dateInfo[0])))),str(date.today()-(timedelta(days=(dateInfo[0])))),market="ALL",investor="기관합계")
		# stockFornSum = stock.get_market_net_purchases_of_equities_by_ticker(str(date.today()-(diffDate+timedelta(days=(1+dateInfo[0])))),str(date.today()-(timedelta(days=(dateInfo[0])))),market="ALL",investor="외국인")       
		return ksIndices, kdIndices, ksSupRes, kdSupRes, ksDividerBins, ksVolDf, kdDividerBins, kdVolDf#stockGovnSum, stockFornSum #, diffDate

	userSelectMode = 'Pullback' # Pullback, Leading, ExcessiveFall ...
	currentTime = datetime.now().strftime("%H:%M:%S")
	dateInfo = [0, 700, 30]
	dateInfo[0] = 1 if currentTime <= '16:00:00' else 0
	rsiInfo = [14, 30, 70] # Date_len, buy, sell
	macdInfo = [12, 26, 9, 0, 0] # Short date, long date, span signal, buy, sell
	envelopInfo = [50, 0.05, 0.025] # Date_len, spread rate, cost_interest
	bollingerInfo = [20, 2, 0.025] # Date_len, scale, cost_interest
	stochasticInfo = [14, 3, 3, 20, 80, 0.025] # Date_len, slow-k, slow-d, buy, sell, cost_interest
	IchBalInfo = [20, 60, 120, 30]

	ksIndices, kdIndices, ksSupRes, kdSupRes, ksDividerBins, ksVolDf, kdDividerBins, kdVolDf = stockIndices(dateInfo,rsiInfo,stochasticInfo,envelopInfo,bollingerInfo,macdInfo,IchBalInfo)
	indicesInfo = pd.concat([pd.DataFrame(ksSupRes),pd.DataFrame(ksDividerBins),pd.DataFrame(ksVolDf),pd.DataFrame(kdSupRes),pd.DataFrame(kdDividerBins),pd.DataFrame(kdVolDf)],axis=1)
	indicesInfo.rename = ['ksSupRes','ksDividerBins','ksVolDf','kdSupRes','kdDividerBins','kdVolDf']
	indicesInfo.columns = ['ksSupRes','ksDividerBins','ksVolDf','kdSupRes','kdDividerBins','kdVolDf']
	# indicesInfo = indicesInfo.fillna(-999)

	#csv 파일 저장
	ksIndices.to_csv('ksIndices.csv', index=True, encoding='utf-8-sig')
	kdIndices.to_csv('kdIndices.csv', index=True, encoding='utf-8-sig')
	indicesInfo.to_csv('indicesInfo.csv', index=True, encoding='utf-8-sig')

	#Debug
	st.write("KOSPI", ksIndices.head())
	st.write("KOSDAQ", kdIndices.head())
	st.write("Info", indicesInfo.head())


@st.cache(allow_output_mutation=True)
def readCSV(filename):
	data = pd.read_csv(filename)
	return data


#******************************************************************************
#*                              Streamlit Main                                *
#******************************************************************************
st.set_page_config(layout="wide")

#sidebar radio button
menu = st.sidebar.radio(
  "",
  ('M0_CSV Generation', 'M1_Stock Chart')
)

#******************************************************************************
#========================= Menu #0: CSV Generation ====================
#******************************************************************************  
if menu == 'M0_CSV Generation':

  # Data
  krxInfo()  

  #Debug
  #st.write(ksIndices.head())
  st.header("CSV Data Generation Completed!")


#******************************************************************************
#========================= Menu #1: Plotly Chart ====================
#******************************************************************************  
elif menu == 'M1_Stock Chart':

	#==========================================================================================================================================
	#================================================= 2. Plotly Bar Chart ====================================================================
	#==========================================================================================================================================
	# Reference1: https://plainenglish.io/blog/a-simple-guide-to-plotly-for-plotting-financial-chart-54986c996682
	# Reference2: 
	# Data
	ksIndices = readCSV('ksIndices.csv') 
	kdIndices = readCSV('kdIndices.csv') 
	indicesInfo = readCSV('indicesInfo.csv') 
    
	col1, col2 = st.columns(2)
	
	#================================================== KOSPI Chart =======================================================================================
	with col1:
		st.markdown("#### KOSPI")
		dates = pd.to_datetime(ksIndices['날짜']).dt.date

		fig = make_subplots(rows=3, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.4,0.2,0.2])
		#=== Candlestick
		fig.add_trace(go.Candlestick(x=dates,
		  open=ksIndices['시가'],
		  high=ksIndices['고가'],
		  low=ksIndices['저가'],
		  close=ksIndices['종가'],
		  showlegend=False,
		  increasing_line_color='red',
          decreasing_line_color='blue',
          #xaxis_rangeslider_visible=False,
          #xaxis=dict(type="category")),
          ), row=1, col=1)
		  
		fig.update_layout(xaxis_rangeslider_visible=False)
		
		#=== Category 변수는 사용하면 안됨. Autoscaling 기능이 불구가 됨
		#fig.update_layout(xaxis=dict(type="category"))  
		#fig.update_layout(height=1200)

		#2nd 거래량 차트
		fig.add_trace(go.Bar(x=dates, y=ksIndices['거래량[만]'], name='거래량[만]', marker_color='brown'), row=2, col=1)		
		
		#3rd-1 RSI
		fig.add_trace(go.Scatter(x=dates, y=ksIndices['rsi'], line=dict(color='red', width=1.5), name='RSI'), row=3, col=1)	
		
		#3rd-2 MACD
		fig.add_trace(go.Scatter(x=dates, y=ksIndices['macd'], line=dict(color='blue', width=1.5), name='macd'), row=3, col=1)			
		
		fig.update_layout(height=700)
		fig.update_layout(margin=go.layout.Margin(
				l=20, #left margin
				r=20, #right margin
				b=20, #bottom margin
				t=20  #top margin
			))	


		#=== 이빨 빠진 Date 구간 붙이기 ([Ref] https://plainenglish.io/blog/a-simple-guide-to-plotly-for-plotting-financial-chart-54986c996682)
		# removing all empty dates
		# build complete timeline from start date to end date
		dt_all = pd.date_range(start=pd.to_datetime(ksIndices['날짜']).iloc[0],end=pd.to_datetime(ksIndices['날짜']).iloc[-1])
		# retrieve the dates that ARE in the original datset
		dt_obs = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(ksIndices['날짜'])]
		# define dates with missing values
		dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
		fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

		#=== 이평선 Chart 추가하기		
		ksIndices['MA05'] = ksIndices['MA05'].replace(0, np.nan)
		fig.add_trace(go.Scatter(x=dates, y=ksIndices['MA05'], opacity=0.7, line=dict(color='green', width=1.5), name='MA05'), row=1, col=1)			
		ksIndices['MA20'] = ksIndices['MA20'].replace(0, np.nan)
		fig.add_trace(go.Scatter(x=dates, y=ksIndices['MA20'], opacity=0.7, line=dict(color='orange', width=1.5), name='MA20'), row=1, col=1)			
		ksIndices['MA60'] = ksIndices['MA60'].replace(0, np.nan)
		fig.add_trace(go.Scatter(x=dates, y=ksIndices['MA60'], opacity=0.7, line=dict(color='red', width=1.5), name='MA60'), row=1, col=1)			
		ksIndices['MA120'] = ksIndices['MA120'].replace(0, np.nan)
		fig.add_trace(go.Scatter(x=dates, y=ksIndices['MA120'], opacity=0.7, line=dict(color='blue', width=1.5), name='MA120'), row=1, col=1)			
		ksIndices['MA224'] = ksIndices['MA224'].replace(0, np.nan)
		fig.add_trace(go.Scatter(x=dates, y=ksIndices['MA224'], opacity=0.7, line=dict(color='purple', width=1.5), name='MA224'), row=1, col=1)			
		ksIndices['MA240'] = ksIndices['MA240'].replace(0, np.nan)
		fig.add_trace(go.Scatter(x=dates, y=ksIndices['MA240'], opacity=0.7, line=dict(color='magenta', width=1.5), name='MA240'), row=1, col=1)		
		ksIndices['MA360'] = ksIndices['MA360'].replace(0, np.nan)
		fig.add_trace(go.Scatter(x=dates, y=ksIndices['MA360'], opacity=0.7, line=dict(color='black', width=1.5), name='MA360'), row=1, col=1)		
		
		#=== 볼린저 밴드 추가하기
		fig.add_trace(go.Scatter(x=dates, y=ksIndices['bollinger_ub'], opacity=0.7, line=dict(color='red', width=1.5), name='bollinger_ub'), row=1, col=1)
		fig.add_trace(go.Scatter(x=dates, y=ksIndices['bollinger_center'], opacity=0.7, line=dict(color='green', width=1.5), name='bollinger_center'), row=1, col=1)
		fig.add_trace(go.Scatter(x=dates, y=ksIndices['bollinger_lb'], opacity=0.7, line=dict(color='blue', width=1.5), name='bollinger_lb'), row=1, col=1)

		#=== Envelop 추가하기
		fig.add_trace(go.Scatter(x=dates, y=ksIndices['envelop_ub'], opacity=0.7, line=dict(color='red', width=1.5), name='envelop_ub'), row=1, col=1)
		fig.add_trace(go.Scatter(x=dates, y=ksIndices['envelop_center'], opacity=0.7, line=dict(color='green', width=1.5), name='envelop_center'), row=1, col=1)
		fig.add_trace(go.Scatter(x=dates, y=ksIndices['envelop_lb'], opacity=0.7, line=dict(color='blue', width=1.5), name='envelop_lb'), row=1, col=1)		
		
		#=== 저항 지지선
		
		#=== 매물대
		
		
		st.plotly_chart(fig, theme="streamlit", use_container_width=True)

	#======================================================================================================================================================
	#================================================== KOSDAQ Chart ======================================================================================
	with col2:
		st.markdown("#### KOSDAQ")
		dates = pd.to_datetime(kdIndices['날짜']).dt.date

		fig = make_subplots(rows=3, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.4,0.2,0.2])
		#=== Candlestick
		fig.add_trace(go.Candlestick(x=dates,
		  open=kdIndices['시가'],
		  high=kdIndices['고가'],
		  low=kdIndices['저가'],
		  close=kdIndices['종가'],
		  showlegend=False,
		  increasing_line_color='red',
          decreasing_line_color='blue',
          #xaxis_rangeslider_visible=False,
          #xaxis=dict(type="category")),
          ), row=1, col=1)
		  
		fig.update_layout(xaxis_rangeslider_visible=False)
		
		#=== Category 변수는 사용하면 안됨. Autoscaling 기능이 불구가 됨
		#fig.update_layout(xaxis=dict(type="category"))  
		#fig.update_layout(height=1200)

		#2nd 거래량 차트
		fig.add_trace(go.Bar(x=dates, y=kdIndices['거래량[만]'], name='거래량[만]', marker_color='brown'), row=2, col=1)		
		
		#3rd-1 RSI
		fig.add_trace(go.Scatter(x=dates, y=kdIndices['rsi'], line=dict(color='red', width=1.5), name='RSI'), row=3, col=1)	
		
		#3rd-2 MACD
		fig.add_trace(go.Scatter(x=dates, y=kdIndices['macd'], line=dict(color='blue', width=1.5), name='macd'), row=3, col=1)			
		
		fig.update_layout(height=700)
		fig.update_layout(margin=go.layout.Margin(
				l=20, #left margin
				r=20, #right margin
				b=20, #bottom margin
				t=20  #top margin
			))	


		#=== 이빨 빠진 Date 구간 붙이기 ([Ref] https://plainenglish.io/blog/a-simple-guide-to-plotly-for-plotting-financial-chart-54986c996682)
		# removing all empty dates
		# build complete timeline from start date to end date
		dt_all = pd.date_range(start=pd.to_datetime(kdIndices['날짜']).iloc[0],end=pd.to_datetime(kdIndices['날짜']).iloc[-1])
		# retrieve the dates that ARE in the original datset
		dt_obs = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(kdIndices['날짜'])]
		# define dates with missing values
		dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if not d in dt_obs]
		fig.update_xaxes(rangebreaks=[dict(values=dt_breaks)])

		#=== 이평선 Chart 추가하기		
		kdIndices['MA05'] = kdIndices['MA05'].replace(0, np.nan)
		fig.add_trace(go.Scatter(x=dates, y=kdIndices['MA05'], opacity=0.7, line=dict(color='green', width=1.5), name='MA05'), row=1, col=1)			
		kdIndices['MA20'] = kdIndices['MA20'].replace(0, np.nan)
		fig.add_trace(go.Scatter(x=dates, y=kdIndices['MA20'], opacity=0.7, line=dict(color='orange', width=1.5), name='MA20'), row=1, col=1)			
		kdIndices['MA60'] = kdIndices['MA60'].replace(0, np.nan)
		fig.add_trace(go.Scatter(x=dates, y=kdIndices['MA60'], opacity=0.7, line=dict(color='red', width=1.5), name='MA60'), row=1, col=1)			
		kdIndices['MA120'] = kdIndices['MA120'].replace(0, np.nan)
		fig.add_trace(go.Scatter(x=dates, y=kdIndices['MA120'], opacity=0.7, line=dict(color='blue', width=1.5), name='MA120'), row=1, col=1)			
		kdIndices['MA224'] = kdIndices['MA224'].replace(0, np.nan)
		fig.add_trace(go.Scatter(x=dates, y=kdIndices['MA224'], opacity=0.7, line=dict(color='purple', width=1.5), name='MA224'), row=1, col=1)			
		kdIndices['MA240'] = kdIndices['MA240'].replace(0, np.nan)
		fig.add_trace(go.Scatter(x=dates, y=kdIndices['MA240'], opacity=0.7, line=dict(color='magenta', width=1.5), name='MA240'), row=1, col=1)		
		kdIndices['MA360'] = kdIndices['MA360'].replace(0, np.nan)
		fig.add_trace(go.Scatter(x=dates, y=kdIndices['MA360'], opacity=0.7, line=dict(color='black', width=1.5), name='MA360'), row=1, col=1)		
		
		#=== 볼린저 밴드 추가하기
		fig.add_trace(go.Scatter(x=dates, y=kdIndices['bollinger_ub'], opacity=0.7, line=dict(color='red', width=1.5), name='bollinger_ub'), row=1, col=1)
		fig.add_trace(go.Scatter(x=dates, y=kdIndices['bollinger_center'], opacity=0.7, line=dict(color='green', width=1.5), name='bollinger_center'), row=1, col=1)
		fig.add_trace(go.Scatter(x=dates, y=kdIndices['bollinger_lb'], opacity=0.7, line=dict(color='blue', width=1.5), name='bollinger_lb'), row=1, col=1)

		#=== Envelop 추가하기
		fig.add_trace(go.Scatter(x=dates, y=kdIndices['envelop_ub'], opacity=0.7, line=dict(color='red', width=1.5), name='envelop_ub'), row=1, col=1)
		fig.add_trace(go.Scatter(x=dates, y=kdIndices['envelop_center'], opacity=0.7, line=dict(color='green', width=1.5), name='envelop_center'), row=1, col=1)
		fig.add_trace(go.Scatter(x=dates, y=kdIndices['envelop_lb'], opacity=0.7, line=dict(color='blue', width=1.5), name='envelop_lb'), row=1, col=1)		
		
		#=== 저항 지지선
		
		#=== 매물대
		
		
		st.plotly_chart(fig, theme="streamlit", use_container_width=True)		
		
		
		
	#======================================================================================================================================================
		
		
		



