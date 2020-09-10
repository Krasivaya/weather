#!/usr/bin/env python

import numpy as np
import pandas as pd
import pylab as pl
import numpy.ma as ma
from datetime import datetime, timedelta

# set today's information
today = int((datetime.utcnow()-timedelta(days=1)).strftime('%j'))+1

# get data as csv into pandas
url = "http://data.rcc-acis.org/StnData?sid=kokc&sdate=por&edate=por&elems=4&output=csv"
df  = pd.read_csv(url,header=0,names=['date','precip'])
df["date"] = pd.to_datetime(df["date"])
df.set_index(df["date"],inplace=True)
df.replace('T',0,inplace=True)

# get first year
ayear = df.index.year[0]
byear = df.index.year[-1]

# group by day and year
pv = pd.pivot_table(df, index=df.index.dayofyear, columns=df.index.year,
                    values='precip', aggfunc='sum')

# mv to numpy array
dd = pv.to_numpy(dtype='float')
nd = dd.shape[0]
ny = dd.shape[1]

# climo cumulative sum
c = dd[:,:-1]
np.cumsum(ma.masked_invalid(dd[:,:-1]),axis=0,out=c)

# current year cumulative sum 
n = dd[:today,-1]
np.cumsum(ma.masked_invalid(dd[:today,-1]),axis=0,out=n)

# compute climo mean and std
cavg = np.mean(c,axis=1)
cstd = np.std(c,axis=1)

# find years with min and max
pmin = np.where(c[-1,:] == np.min(c[-1,1:-1]))[0][0]
pmax = np.where(c[-1,:] == np.max(c[-1,:-1]))[0][0]

# create the figure
fig = pl.figure(figsize=(12,6))
pl.style.use('dark_background')

# plot historical curves
for j in range(ny-1):
    pl.plot(c[:,j],color='gray',alpha=0.25,lw=3)

# plot average, min, max, and current year
pl.plot(cavg,color='k',label='climo',lw=3)
pl.plot(c[:,pmin],color='tab:blue',alpha=0.5,label=str(pmin+ayear),lw=3)
pl.plot(c[:,pmax],color='tab:blue',alpha=0.5,label=str(pmax+ayear),lw=3)
pl.plot(n,color='tab:red',lw=3)

# label label +/- standard deviation
pl.text(nd+2,(cavg+cstd)[-1]-0.25,'+1$\sigma$: %0.2f\"'%(cavg+cstd)[-1],fontsize=12)
pl.text(nd+2,(cavg-cstd)[-1]-0.25,'-1$\sigma$: %0.2f\"'%(cavg-cstd)[-1],fontsize=12)

# label average, min, max, and current year
pl.text(nd+2, cavg[-1] - 0.25,'Avg: %0.2f\"'%(cavg[-1]),fontsize=12)
pl.text(nd+2, c[:,pmin][-1]-2,'Min: %0.2f\"'%(c[:,pmin][-1])+'\n('+str(pmin+ayear)+')',fontsize=12)
pl.text(nd+2, c[:,pmax][-1]-2,'Max: %0.2f\"'%(c[:,pmax][-1])+'\n('+str(pmax+ayear)+')',fontsize=12)
pl.text(today+2,n[today-1]-0.1,'%0.2f\"'%(n[today-1]),color='white',fontsize=10,alpha=1.,fontweight='bold',bbox=dict(facecolor='tab:red', alpha=0.5))

# highlight +/- standard deviation range
pl.fill_between(np.arange(nd),cavg-cstd,cavg+cstd,color='white',zorder=2)

# customize figure
pl.title('Oklahoma City Precipitation %d-%d'%(ayear,byear),fontsize=14)
pl.ylabel("Cumulative Precipitation [in]",fontsize=14)
pl.xlim(0,nd)
pl.ylim(0,60)
em = np.array([0,31,30,31,30,31,30,31,31,30,31,30,31])
lab = np.array([])
for i in range(len(em)-1):
    lab = np.append(lab,np.sum(em[0:i+1]) + 0.5*em[i+1] )
pl.xticks(lab,['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],fontsize=14)
pl.yticks(fontsize=14)
ems = np.cumsum(em)
for i,j in enumerate(ems[::2]):
    pl.axvspan(j,j+em[i+1],color='darkgrey',alpha=0.3)
pl.text(nd+2,0,'by:\n@sciencegibbs',fontsize='10')
pl.savefig("figures/precip_kokc.png",dpi=300,bbox_inches="tight")
