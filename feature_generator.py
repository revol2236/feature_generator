import numpy as np
import math as ms
import os



class feature_generator():
    def __init__(self):
        #define at text file in original code. but just for test variable in this system
        self.initialization()
        self.nsample=[0]*(self.ntask+1)
        self.read_para_a()
        self.ngroup=[[0]*self.ntask for _ in range(1000)]
        self.isconvex=[[0]*self.ntask for _ in range(1000)]
        self.pfdim=[[0]*self.ndimtype for _ in range((self.nsf+self.nvf))]
        self.read_para_b()
        self.prop_y=[0]*sum(self.nsample)
        self.psfeat=[[0]*self.nsf for _ in range(sum(self.nsample))]
        self.pfname=['']*(self.nsf+self.nvf)
        self.res=[0]*sum(self.nsample)
        self.read_data()
        self.psfeat=np.ndarray.tolist(np.transpose(np.array(self.psfeat)))
        self.icontinue=1
        self.tocontinue='FC'


        #######################################feature_generator

        # self.funit=100
        # self.maxfval=1e5
        # self.sis_on=False
        # self.f_select = []
        # self.name_select=[]
        # self.ftag_select=[]
        # self.foutsave=True
        # self.ntot=0
        # self.nthis=0
        # self.fout=[]
        # self.name_out=[]
        # self.lastop_out=[]
        # self.complexity_out=[]
        # self.dim_out=[]
        # self.icomb=0
        #self.score_select=[[0]*2*self.subs_sis]*2
        self.score_select=[[0.906108704402843,  0.901128081550113,0.890064001214225,  0.914001838777997,  0.912264141449710,0.905684518988813], [1.00000000000000,   1.00000000000000,1.00000000000000,   1.00000000000000,   1.00000000000000,1.00000000000000]]
        #setting variable
        self.nreject=0
        self.nselect=3
        self.fout=[]
        self.trainy_c=[ -4.481707500000010E-002,5.149135000000005E-002,-6.465814650000001E-002,0.178032550500000,-0.120048678300000]
        self.tag=[1.00100000000000,1.00200000000000,1.00300000000000,1.00400000000000,1.00500000000000]

    def feature_contruction(self):
        trainy=[0]*sum(self.nsample)
        trainy_c=[0]*sum(self.nsample)
        self.mpirank_default=0
        i=max(1000,self.nsf+self.nvf)
        feat_in1=[[0]*sum(self.nsample) for _ in range(i)]
        name_in1=['']*i
        lastop_in1=['']*i
        dim_in1=[[0]*self.ndimtype for _ in range(i)]
        complexity_in1=[0]*i
        fout=[['']*sum(self.nsample) for _ in range(i)]
        self.name_out=['']*i
        self.lastop_out=[0]*i
        self.complexity_out=[0]*i
        self.dim_out=[[0]*self.ndimtype for _ in range(i)]
        j= 2*self.subs_sis
        self.f_select=[[0]*sum(self.nsample) for _ in range(j)]
        self.ftag_select=[0]*j
        self.socre_select=[[0]*j for _ in range(2)]
        self.name_select=['']*j

        dim_in1[:self.nsf+self.nvf]=np.ndarray.tolist(np.reshape(np.array(self.pfdim),[len(dim_in1[:self.nsf+self.nvf][:]),len(dim_in1[:self.nsf+self.nvf][:][0])]))
        self.sis_on=False
        nvf_new=self.nvf
        feat_in1[0:self.nsf]=self.psfeat[0:self.nsf][:]
        name_in1[:self.nsf+self.nvf]=self.pfname[:self.nsf+self.nvf]
        trainy=self.prop_y

        [self.reject,self.nreject]=self.reject_list()

        for j in range(1,self.ntask+1):
            mm1=sum(self.nsample[:j])
            mm2=sum(self.nsample[:j+1])
            trainy_c[mm1:mm2]=[tmp_i-sum(trainy[mm1:mm2])/(mm2-mm1) for tmp_i in trainy[mm1:mm2]]

        self.nselect=0
        self.nf=0
        self.foutsave=True
        lastop_in1=['']*i
        complexity_in1=[0]*i
        i=self.nsf+self.nvf
        j=0

        self.combine(feat_in1[0:i][:],name_in1[0:i],lastop_in1[0:i],complexity_in1[0:i],dim_in1[0:i][:]\
                     ,feat_in1[0:i][:],name_in1[0:i],lastop_in1[0:i],complexity_in1[0:i],dim_in1[0:i][:],'NO',j)
        ntot=self.nsf
        nthis=ntot
        ###############develop part


    def read_data(self):
        train_dat=open('./train.dat','r')
        line=train_dat.readline()
        if self.ptype==1:
            line=(line.split())
            for tmp_i in range(len(line)-2):
                self.pfname[tmp_i]=line[tmp_i+2].strip()
        else:
            line = line.split()
            for tmp_i in range(len(line) - 2):
                self.pfname[tmp_i] = line[tmp_i + 1].strip()

        for i in range(sum(self.nsample)):
            line=train_dat.readline()
            line=line.split()
            if self.ptype==1:
                self.prop_y[i]=float(line[1].strip())
                self.psfeat[i]=[float(tmp_i) for tmp_i in line[2:]]
            else:
                self.prop_y[tmp_i] = 0.0
                self.psfeat[i][:] = line[1 + tmp_i]


    def initialization(self):
        self.restart=False
        self.ndimtype=0
        self.nsf=1
        self.nvf=0
        self.ptype=1
        self.ntask=1
        self.vfsize=0
        self.vf2sf='sum'
        self.width=1e-6
        self.task_weighting=1

        self.maxcomplexity=10
        self.npf_must=0
        self.maxfval_lb=1e-8
        self.maxfval_ub=1e5
        self.rung=1
        self.subs_sis=1
        self.opset=''
        self.method='L0'
        self.metric='RMSE'
        self.fit_intercept=True
        self.desc_dim=1
        self.nm_output=100
        self.init_var=['restart','nsf','nvf','ptype','ntask','vfsize','vf2sf','width','task_weighting' \
                       ,'maxcomplexity','npf_must','maxfval_lb','maxfval_ub','rung','subs_sis','opset','method','metric' \
                       ,'fit_intercept','desc_dim','nm_output','nsample','dimclass']

    def strType(self,xstr):
        try:
            int(xstr)
            return 'int'
        except:
            try:
                float(xstr)
                return 'float'
            except:
                try:
                    complex(xstr)
                    return 'complex'
                except:
                    return 'str'

    def read_para_a(self):
        sisso_in=open('./SISSO.in','r')
        for tmp_read in sisso_in:
            name_value=tmp_read.split('=')
            if(name_value[0] in self.init_var):
                tmp_name='self.'+name_value[0]
                if name_value[0]=='nsample':
                    self.nsample[-1]=int(name_value[1])
                    continue

                if name_value[0] == 'dimclass':
                    k=-1
                    while True:
                        k = name_value[1][k + 1:].find('(') + k+1
                        self.ndimtype = self.ndimtype + 1
                        if name_value[1][k+1:].find('(')>0:
                            continue
                        else:
                            exec ("%s = %s" % (tmp_name, name_value[1]))
                            break
                    continue

                if self.strType(name_value[1])=='float':
                    exec ("%s = %f" % (tmp_name, float(name_value[1])))
                elif self.strType(name_value[1])=='int':
                    exec ("%s = %d" % (tmp_name, int(name_value[1])))
                else:
                    exec ("%s = %s" % (tmp_name, name_value[1]))
        sisso_in.close()

    def read_para_b(self):
        nsample=0
        nsgroup=0
        isconvex=1
        for ll in range(self.ndimtype):
            i=self.dimclass.find('(')
            j=self.dimclass.find(':')
            kk=self.dimclass.find(')')
            if (i>-1 and j>-1):
                k=int(self.dimclass[i+1:j])
                l=int(self.dimclass[j+1:kk])
                for tmp_i in range(k-1,l):
                    self.pfdim[tmp_i][ll]=1.0
                self.dimclass=self.dimclass.replace(self.dimclass[:kk+1],'')



    def isscalar(self,fname):
        isscalar_r =False
        fname=str(fname)
        if (self.nvf == 0):
            isscalar_r =True
        elif (fname.find('v_') == 0):
            isscalar_r =True
        return isscalar_r

    def sis_score(self,feat,yyy):
        xmean = []
        xnorm = []
        tmp = []
        score = [0] * 2
        sdfeat = [0] * max(self.nsample)
        npfeat = np.array(feat)
        for j in range(0, self.ntask):
            mm1 = self.nsample[j]
            mm2 = self.nsample[j + 1]
            xmean.append(sum(feat[mm1:mm2]) / self.nsample[j + 1])
            sdfeat[mm1:mm2] = [i - xmean[j] for i in feat[mm1:mm2]]
            xnorm.append(np.sqrt(sum([i ** 2 for i in sdfeat[mm1:mm2]])))
            if (xnorm[j] > 1e-50):
                sdfeat[mm1:mm2] = [i / xnorm[j] for i in sdfeat[mm1:mm2]]
            tmp.append(abs(sum([i * k for i, k in zip(sdfeat[mm1:mm2], yyy[mm1:mm2])])))
        score[0] = np.sqrt(sum([(i ** 2) / self.ntask for i in tmp]))
        score[0] = score[0] / np.sqrt(sum([(i ** 2) / self.ntask for i in yyy]))
        score[1] = 1

        return score

    def isgoodf(self,feat,name_feat,lastop,comp,dimens,nf):
        if self.goodf(feat,name_feat,dimens,comp):
            nf=nf+1
            if self.foutsave:
                self.fout.append(feat)
                self.name_out.append(name_feat)
                self.lastop_out.append(lastop)
                self.complexity_out.append(comp)
                self.dim_out.append(dimens)


    def combine(self,fin1,name_in1,lastop_in1,comp_in1,dim_in1,fin2,name_in2,lastop_in2,comp_in2,dim_in2,op,nf):
        nfin1=len(fin1)
        nfin2=len(fin2)
        counter=0
        skip=False
        first=True
        progress=0.2

        for i in range(nfin1):
            if op.strip()=='NO':
                lastop_tmp=''
                comp_tmp=comp_in1[i]
                name_tmp='('+name_in1[i].strip()+')'
                dimtmp=dim_in1[i][:]
                if self.isscalar(name_in1[i]):
                    tmp=fin1[i][:]
                    self.isgoodf(tmp,name_tmp,lastop_tmp,comp_tmp,dimtmp,nf)
                else:
                    print('breake in combinemethod')
                    return
                continue

            counter=counter+1
            comp_tmp=comp_in1[i]+1
            if not(comp_tmp>self.maxcomplexity):
                if op.find('(exp)')!=-1:
                    if lastop_in1[i].find('(exp') ==-1 and lastop_in1[i].find('(log)')==-1:
                        lastop_tmp='(exp)'
                        name_tmp='exp('+name_in1[i].strip()+')'
                        dimtmp=self.dimcomb(dim_in1[i][:],dim_in1[i][:],'(exp)')
                        if self.isscalar(name_in1[i]):
                            tmp=[ms.exp(tmp_i) for tmp_i in fin1[i][:]]
                            self.isgoodf(tmp,name_tmp,lastop_tmp,comp_tmp,dimtmp,nf)
                        else:
                            raise SystemExit

                if op.find('(exp-)')!=-1:
                    if lastop_in1[i].find('(exp') ==-1 and lastop_in1[i].find('(log)')==-1:
                        lastop_tmp='(exp-)'
                        name_tmp='exp(-'+name_in1[i].strip()+')'
                        dimtmp=self.dimcomb(dim_in1[i][:],dim_in1[i][:],'(exp-)')
                        if self.isscalar(name_in1[i]):
                            tmp=[ms.exp(-tmp_i) for tmp_i in fin1[i][:]]
                            self.isgoodf(tmp,name_tmp,lastop_tmp,comp_tmp,dimtmp,nf)
                        else:
                            raise SystemExit

                if op.find('(^-1)')!=-1:
                    if min(abs(fin1[i][:]))>1e-50:
                        lastop_tmp='(^-1)'
                        name_tmp='('+name_in1[i].strip()+')^-1'
                        dimtmp=self.dimcomb(dim_in1[i][:],dim_in1[i][:],'(^-1)')
                        if self.isscalar(name_in1[i]):
                            tmp=[(tmp_i)**(-1) for tmp_i in fin1[i][:]]
                            self.isgoodf(tmp,name_tmp,lastop_tmp,comp_tmp,dimtmp,nf)
                        else:
                            raise SystemExit

                if op.find('(scd)')!=-1:
                    lastop_tmp='(scd)'
                    name_tmp='scd('+name_in1[i].strip()+')'
                    dimtmp=self.dimcomb(dim_in1[i][:],dim_in1[i][:],'(scd)')
                    if self.isscalar(name_in1[i]):
                        tmp=[1.0/(np.pi*(1.0+(tmp_i)**2)) for tmp_i in fin1[i][:]]
                        self.isgoodf(tmp,name_tmp,lastop_tmp,comp_tmp,dimtmp,nf)
                    else:
                        raise SystemExit

                if op.find('(^2)')!=-1:
                    if lastop_in1[i].find('(sqrt')==-1:
                        lastop_tmp='(^2)'
                        name_tmp='('+name_in1[i].strip()+')^2'
                        dimtmp=self.dimcomb(dim_in1[i][:],dim_in1[i][:],'(^2)')
                        if self.isscalar(name_in1[i]):
                            tmp=[(tmp_i)**2 for tmp_i in fin1[i][:]]
                            self.isgoodf(tmp,name_tmp,lastop_tmp,comp_tmp,dimtmp,nf)
                        else:
                            raise SystemExit

                if op.find('(^3)')!=-1:
                    if lastop_in1[i].find('(cbrt)')==-1:
                        lastop_tmp='(^3)'
                        name_tmp='('+name_in1[i].strip()+')^3'
                        dimtmp=self.dimcomb(dim_in1[i][:],dim_in1[i][:],'(^3)')
                        if self.isscalar(name_in1[i]):
                            tmp=[(tmp_i)**3 for tmp_i in fin1[i][:]]
                            self.isgoodf(tmp,name_tmp,lastop_tmp,comp_tmp,dimtmp,nf)
                        else:
                            raise SystemExit

                if op.find('(^6)')!=-1:
                    lastop_tmp='(^6)'
                    name_tmp='('+name_in1[i].strip()+')^6'
                    dimtmp=self.dimcomb(dim_in1[i][:],dim_in1[i][:],'(^6)')
                    if self.isscalar(name_in1[i]):
                        tmp=[(tmp_i)**6 for tmp_i in fin1[i][:]]
                        self.isgoodf(tmp,name_tmp,lastop_tmp,comp_tmp,dimtmp,nf)
                    else:
                        raise SystemExit

                if op.find('(sqrt)')!=-1:
                    if lastop_in1[i].find('(^2)')==-1:
                        if min(fin1[i][:])>0 and self.isscalar(name_in1[i]):
                            lastop_tmp='(sqrt)'
                            name_tmp='sqrt('+name_in1[i].strip()+')'
                            dimtmp=self.dimcomb(dim_in1[i][:],dim_in1[i][:],'(sqrt)')
                            tmp=[np.sqrt(tmp_i) for tmp_i in fin1[i][:]]
                            self.isgoodf(tmp,name_tmp,lastop_tmp,comp_tmp,dimtmp,nf)
                        else:
                            raise SystemExit

                if op.find('(cbrt)')!=-1:
                    if lastop_in1[i].find('(^3)')==-1:
                        if self.isscalar(name_in1[i]):
                            lastop_tmp='(cbrt)'
                            name_tmp='cbrt('+name_in1[i].strip()+')'
                            dimtmp=self.dimcomb(dim_in1[i][:],dim_in1[i][:],'(cbrt)')
                            tmp=[tmp_i**(1/3) for tmp_i in fin1[i][:]]
                            self.isgoodf(tmp,name_tmp,lastop_tmp,comp_tmp,dimtmp,nf)
                        else:
                            raise SystemExit

                if op.find('(log)')!=-1:
                    if lastop_in1[i].find('(exp')==-1 and lastop_in1[i].find('(log)')==-1:
                        if min(fin1[i][:]>0 and self.isscalar(name_in1[i])):
                            lastop_tmp='(log)'
                            name_tmp='log('+name_in1[i].strip()+')'
                            dimtmp=self.dimcomb(dim_in1[i][:],dim_in1[i][:],'(log)')
                            tmp=[math.log(tmp_i) for tmp_i in fin1[i][:]]
                            self.isgoodf(tmp,name_tmp,lastop_tmp,comp_tmp,dimtmp,nf)
                        else:
                            raise SystemExit

                if op.find('(sin)')!=-1:
                    lastop_tmp='(sin)'
                    name_tmp='sin('+name_in1[i].strip()+')'
                    dimtmp=self.dimcomb(dim_in1[i][:],dim_in1[i][:],'(sin)')
                    if self.isscalar(name_in1[i]):
                        tmp=[math.sin(tmp_i) for tmp_i in fin1[i][:]]
                        self.isgoodf(tmp,name_tmp,lastop_tmp,comp_tmp,dimtmp,nf)
                    else:
                        raise SystemExit

                if op.find('(cos)')!=-1:
                    lastop_tmp='(cos)'
                    name_tmp='cos('+name_in1[i].strip()+')'
                    dimtmp=self.dimcomb(dim_in1[i][:],dim_in1[i][:],'(cos)')
                    if self.isscalar(name_in1[i]):
                        tmp=[math.cos(tmp_i) for tmp_i in fin1[i][:]]
                        self.isgoodf(tmp,name_tmp,lastop_tmp,comp_tmp,dimtmp,nf)
                    else:
                        raise SystemExit
            #599 line
            for j in range(nfin2):
                counter=counter+1
                comp_tmp=comp_in1[i]+comp_in2[j]+1
                if not(comp_tmp>self.maxcomplexity): continue
                if self.better_name(name_in2[j],name_in1[i])==1:
                    firts=True
                else:
                    first=False

                if op.find('(+)')!=-1 or op.find('(-)')!=-1 or op.find('(|-|)')!=-1:
                    if ((lastop_in1[i].find('(exp')==-1 and lastop_in2.find('(exp')!=-1) or \
                        (lastop_in1[i].find('(exp') != -1 and lastop_in2.find('(exp') == -1) or \
                        (lastop_in1[i].find('(log)') == -1 and lastop_in2.find('(log)') != -1) or \
                        (lastop_in1[i].find('(log)') != -1 and lastop_in2.find('(log)') == -1)) or not( max(abs(dim_in1[i][:]-dim_in2[j][:]))>1e-8):
                        [skip,first]=self.combine_600(fin1,name_in1,lastop_in1,dim_in1,fin2,name_in2,lastop_in2,dim_in2,op,nf,i,j,comp_tmp,skip,first)
                        continue

                    if max(abs([tmp_i-tmp_j for tmp_i,tmp_j in zip(dim_in1[i][:],dim_in2[j][:])]))>1e-8:
                        [skip,first]=self.combine_600(fin1,name_in1,lastop_in1,dim_in1,fin2,name_in2,lastop_in2,dim_in2,op,nf,i,j,comp_tmp,skip,first)
                        continue

                    if op.find('(+)')!=-1:
                        lastop_tmp='(+)'
                        if first:
                            name_tmp='('+name_in1[i]+'+'+name_in2[j]+')'
                        else:
                            name_tmp = '(' + name_in2[j] + '+' + name_in1[i] + ')'
                        dimtmp=dimcomb(dim_in1[i][:],dim_in2[j][:],'(+)')
                        if self.isscalar(name_in1[i]) and self.isscalar(name_in2[j]):
                            tmp=[tmp_i+tmp_j for tmp_i,tmp_j in zip(fin1[i][:],fin2[j][:])]
                            self.isgoodf(tmp,name_tmp,lastop_tmp,comp_tmp,dimtmp,nf)
                        else:
                            raise SystemExit

                if op.find('(-)')!=-1:
                    lastop_tmp='(-)'
                    name_tmp='('+name_in1[i]+'-'+name_in2[j]+')'
                    if op.find('(+)')==-1: dimtmp=dimcomb(dim_in1[i][:],dim_in2[j][:],'(-)')
                    if self.isscalar(name_in1[i]) and self.isscalar(name_in2[j]):
                        tmp=[tmp_i-tmp_j for tmp_i,tmp_j in zip(fin1[i][:],fin2[j][:])]
                        self.isgoodf(tmp,name_tmp,lastop_tmp,comp_tmp,dimtmp,nf)
                    else:
                        raise SystemExit
                    if self.icomb<self.rung:
                        name_tmp='('+name_in2[j]+'(-)'+name_in1[i]+')'
                        if self.isscalar(name_in2[j]) and self.isscalar(name_in1[i]):
                            tmp=-tmp
                            self.isgoodf(tmp,name_tmp,lastop_tmp,comp_tmp,dimtmp,nf)
                        else:
                            raise SystemExit

                if op.find('|-|')!=-1:
                    lastop_tmp='(|-|)'
                    if first:
                        name_tmp='abs('+name_in1[i]+'-'+name_in2[j]+')'
                    else:
                        name_tmp = 'abs(' + name_in2[j] + '-' + name_in1[i] + ')'

                    if op.find('(+)')==-1: dimtmp=dimcomb(dim_in1[i][:],dim_in2[j][:],'(|-|)')
                    if self.isscalar(name_in1[i]) and self.isscalar(name_in2[j]):
                        if (op.find('(-)'))!=-1:
                            tmp=abs(tmp)
                        else:
                            tmp=[abs(tmp_i-tmp_j for tmp_i,tmp_j in zip(fin1[i][:],fin2[j][:]))]
                        self.isgoodf(tmp,name_tmp,lastop_tmp,comp_tmp,dimtmp,nf)
                    else:
                        raise SystemExit
                [skip,first]=self.combine_600(fin1,name_in1,lastop_in1,dim_in1,fin2,name_in2,lastop_in2,dim_in2,op,nf,i,j,comp_tmp,skip,first)

    def combine_600(self,fin1,name_in1,lastop_in1,dim_in1,fin2,name_in2,lastop_in2,dim_in2,op,nf,i,j,comp_tmp,skip,first):
        if (lastop_in1[i].find('(exp)')!=-1 and lastop_in2[j].find('(exp)')!=-1) or \
            (lastop_in1[i].find('(exp-)')!=-1 and lastop_in2[j].find('(exp-)')!=-1) or \
            (lastop_in1[i].find('(log)') != -1 and lastop_in2[j].find('(log)') != -1):
            return
        if op.find('(*)')!=-1:
            lastop_tmp='(*)'
            if first:
                name_tmp = '(' + name_in1[i] + '*' + name_in2[j] + ')'
            else:
                name_tmp='('+name_in2[i]+'*'+name_in1[j]+')'
            dimtmp=dimcomb(dim_in1[i][:],dim_in2[j][:],'(*)')
            if self.isscalar(name_in1[i]) and self.isscalar(name_in2[j]):
                tmp=[tmp_i*tmp_j for tmp_i,tmp_j in zip(fin1[i][:],fin2[j][:])]
                self.isgoodf(tmp,name_tmp,lastop_tmp,comp_tmp,dimtmp,nf)
            else:
                raise SystemExit

        if op.find('(/)')!=-1:
            if op.find('(*)')!=-1 and op.find('(/)')!=-1 or lastop_in2.find('(/)')!=-1:return
            skip=False
            if lastop_in1.find('(*)')!=-1:
                l=name_in1[i].find(name_in2[j])
                if l>0:
                    k=name_in1[i].find(name_in2[j])
                    kk=len(name_in1[i][k:].strip())
                    kkk=len(name_in2[j])
                    if (l==2 and name_in1[i][k+kkk:k+kkk]=='*') or (kk==kkk+1 and name_in1[i][k-1:k-1]=='*'):
                        if op.find('(^-1)')!=-1:
                            return
                        else:
                            self.combine_601(fin1,name_in1,dim_in1,fin2,name_in2,dim_in2,nf,i,j,comp_tmp,skip)

            if lastop_in1.find('(*)')!=-1:
                l=name_in2[j].find(name_in1[i])
                if l>0:
                    k=name_in2[j].find(name_in1[i])
                    kk=len(name_in2[j][k:].strip())
                    kkk=len(name_in1[i])
                    if (l==2 and name_in2[j][k+kkk:k+kkk]=='*') or (kk==kkk+1 and name_in2[j][k-1:k-1]=='*'):
                        if op.find('(^-1)')!=-1:
                            return
                        else:
                            skip=True
            lastop_tmp='(/)'
            name_tmp='('+name_in1[i]+'/'+name_in2[j]+')'
            dimtmp=dimcomb(dim_in1[i][:],dim_in2[j][:],'(/)')
            if self.isscalar(name_in1[i]) and self.isscalar(name_in2[j]):
                if min(abs(fin2[j][:]))>1e-50:
                    tmp=[tmp_i/tmp_j for tmp_i,tmp_j in zip(fin1[i][:],fin2[j][:])]
                    self.isgoodf(tmp,name_tmp,lastop_tmp,comp_tmp,dimtmp,nf)
                else:
                    raise SystemExit
        self.combine_601(fin1,name_in1,dim_in1,fin2,name_in2,dim_in2,nf,i,j,comp_tmp,skip)
        return skip,first

    def combine_601(self,fin1,name_in1,dim_in1,fin2,name_in2,dim_in2,nf,i,j,comp_tmp,skip):
        if skip: retun
        lastop_tmp='(/)'
        name_tmp='('+name_in2[j]+'/'+name_in1[i]+')'
        dimtmp=self.dimcomb(dim_in2[j][:],dim_in1[i][:],'((/)')
        if self.isscalar(name_in1[i]) and self.isscalar(name_in2[j]):
            tmp=fin2[j][:]/fin1[i][:]
            isgoodf(tmp,name_tmp,lastop_tmp,comp_tmp,dimtmp,nf)
        else:
            raise SystemExit

    def dimcomb(self,dim1,dim2,op):
        if op.strip()=='(+)' or op.strip()=='(-)' or op.strip()=='(|-|)':
            for i in range(len(dim1[0])):
                if abs(dim1[i]-dim2[i])>1e-8:
                    print("error")
                    raise SystemExit
        elif op.strip()=='(*)':
            dimcomb=dim1+dim2
        elif op.strip()=='(/)':
            dimcomb=dim1-dim2
        elif op.strip()=='(exp)':
            dimcomb=0.0
        elif op.strip()=='(exp-)':
            dimcomb=0.0
        elif op.strip()=='(log)':
            dimcomb=0.0
        elif op.strip()=='(scd)':
            dimcomb=0.0
        elif op.strip()=='(sin)':
            dimcomb=0.0
        elif op.strip()=='(cos)':
            dimcomb=0.0
        elif op.strip()=='(^-1)':
            dimcomb=dim1*(-1)
        elif op.strip()=='(^2)':
            dimcomb=dim1*2
        elif op.strip()=='(^3)':
            dimcomb=dim1*3
        elif op.strip()=='(^6)':
            dimcomb=dim1*6
        elif op.strip()=='(sqrt)':
            dimcomb=dim1/2.0
        elif op.strip()=='(cbrt)':
            dimcomb=dim1/3.0
        return dimcomb
    def goodf(self,feat,name_feat,dimens,comp):
        bool_good=True
        nselect=self.nselect
        scoretmp=[0]*2
        name_feat=str(name_feat)
        mm1=0
        mm2=sum(self.nsample[:self.ntask+1])-1
        if max([abs(i-feat[mm1]) for i in feat[mm1:mm2]])<=1e-8:
            bool_good=False
            return
        maxabs=max([abs(i) for i in feat[mm1:mm2]])
        if maxabs>1e50 or maxabs<=1e-50:
            bool_good=False
            return
        if maxabs>self.maxfval_ub or maxabs<self.maxfval_lb:
            return

        # must be regression model
        scoretmp=self.sis_score(feat,self.trainy_c)
        if self.sis_on:
            if scoretmp(1)<score_select(subs_sis,1):
                return

        if self.npf_must>0:
            if not self.isABC(name_feat):
                return
        if self.nreject>0:
            name_feat=name_feat.strip()
            lsame=False
            i=0
            j=self.nreject
            k=i+ms.ceil(j-i/2.0)
            while True:
                if name_feat.split()==self.reject[k].split():
                    lsame=True
                    i=j
                elif name_feat<self.reject[k]:
                    i=k
                elif name_feat>self.reject[k]:
                    j=k;

                if(k==i+ms.ceil((j-i)/2.0)):
                    i=j
                else:
                    k=i+ms.ceil((j-i)/2.0)

                if i!=j:
                    continue
                else:
                    break
            if lsame:
                return
        nselect=self.nselect+1
        self.f_select.append(feat)
        fnorm=np.sqrt(sum([i**2 for i in feat]))
        self.name_select.append(name_feat)
        self.score_select.append(scoretmp)
        self.ftag_select.append(abs(sum(np.multiply(self.tag,feat))/fnorm/self.ntask))
        if (nselect==2*self.subs_sis):
            sis_s(self)
            self.sis_on=True
        return bool_good

            # do:
            #     if(name_feat==)
            #
            # while (i != j)
    def isABC(self,fname):
        fname=str(fname)
        result=False
        if self.npf_must==6:
            if((fname.find('A')!=-1) and (fname.find('B')!=-1) and (fname.find('C')!=-1) and (fname.find('D')!=-1) and (fname.find('E'))!=-1 and (fname.find('F'))!=-1):
                result=True
        elif self.npf_must==5:
            if((fname.find('A')!=-1) and (fname.find('B')!=-1) and (fname.find('C')!=-1) and (fname.find('D')!=-1) and (fname.find('E'))!=-1):
                result=True
        elif self.npf_must==4:
            if(((fname.find('A')!=-1) and (fname.find('B')!=-1) and (fname.find('C')!=-1) and (fname.find('D')!=-1))!=-1):
                result=True
        elif self.npf_must == 3:
            if (((fname.find('A') != -1) and (fname.find('B') != -1) and (fname.find('C') != -1)) != -1):
                result = True
        elif self.npf_must == 2:
            if (((fname.find('A') != -1) and (fname.find('B') != -1)) != -1):
                result = True
        elif self.npf_must == 1:
            if ((fname.find('A') != -1) != -1):
                result = True
        return result

    def reject_list(self):
        lreject=os.path.isfile('./feature_space/reject.name')
        reject=[]
        if lreject:
            rejectname='./feature_space/reject.name'
        else:
            lreject=os.path.isfile('./feature_space/Uspace.name')
            if lreject:
                rejectname='./feature_space/Uspace.name'
        nreject=0
        if lreject:
            funit=open(rejectname.strip(),'r')
            nreject=len(funit.readlines())
            funit.seek(0,0)
        for j in range(nreject):
            line=funit.readline()
            reject.append(line.split(' ')[0])
        # funit.close()

        for i in range(nreject-1):
            loc=i
            for j in range(i+1,nreject):
                if(reject[j]>reject[loc]):
                    loc=j
            if loc>i:
                line=str(reject[i]).strip()
                reject[i]=reject[loc]
                reject[loc]=str(line).strip()

        print(reject)
        # print(len(reject))
        return reject,nreject


    def sis_s(self):
        order=[0]*(self.nselect)
        tmpf=[]
        tmpname=[]
        tmpftag=[]
        tmpscore=[]
        n=1
        order[0]=0
        for i in range(1,self.nselect):
            l=0
            ll=n
            j=l+ms.ceil(float(ll-l)/2.0-0.5)
            while True:
                if(abs(self.score_select[0][i]-self.score_select[0][order[j]]))<=1e-8 and abs(ftag_select[j]-ftag_select[order[j]])<1e-8:
                    if(self.better_name(name_select(order[i]),name_select[j])):
                        order[j]=i
                    break
                else:
                    if (self.score_select[0][i]-self.score_select[0][order[j]])>1e-8 or (abs(self.score_select[0][i]-self.score_select[0][order[j]]))<=1e-8 \
                        and abs(self.score_select[1][i]-self.score_select[order[1]][j])>1e-8 or \
                        abs(self.score_select[0][i]-self.score_select[0][order[j]])<=1e-8 and abs(self.score_select[1][i]-self.score_select[1][order[j]])<=1e-8 \
                        and (ftag_select[i]-ftag_select[j])>1e-8:
                        ll=j
                        if j==l+ms.ceil(float(ll-l)/2.0-0.5):
                            order[j+1:n+1]=order[j:n]
                            order[j]=i
                            n=n+1
                            break

                    else:
                        l=j
                        if(j==l+ms.ceil(float(ll-l)/2.0-0.5)):
                            if n>j:
                                order[j+2:n+1]=order[j+1:n]
                            order[j+1]=i
                            n=n+1
                            break
                    j=l+ms.ceil(float(ll-l)/2.0-0.5)
        n=min(n,self.subs_sis)-1

        for i in range(n):
            tmpf.append(self.f_select[order[i]])
            tmpname.append(self.name_select[order[i]])
            tmpftag.append(self.ftag_select[order[i]])
            tmpscore.append(self.score_select[:][order[i]])

        nselect=n
        self.f_select[:n][:]=tmpf[:n][:]
        self.name_select[:self.subs_sis]=tmpname[:n]
        self.ftag_select[:n]=tmpftag[:n]
        self.score_select[:][:n]=tmpscore[:n][:]
        return f_select, name_select, ftag_select



    def better_name(self,name1,name2):
        name1=str(name1); name2=str(name2)
        if len(name1.strip())<len(name2.strip()) or (len(name1.strip())==len(name2.strip()) and name1.strip()<=name2.strip()):
            return 1
        else:
            return 2


def main():
    feat = [-1.37177875293756,-1.24575922581336,-0.541371170289951,-79.8059443404863,-0.921773219406105]
    yyy = [-4.430929943919181823730468750000000E-0002,
           5.101694539189338684082031250000000E-0002,
           -0.111230455338954925537109375000000,
           0.188298255205154418945312500000000,
           -8.377544581890106201171875000000000E-0002]
    name_feat='(feature3/log(feature3))'
    dimens=[0.000000000000000E+000,-1.00000000000000 ]
    f_select=[[-2.01846772093827,  -1.73469037700779, -0.861128891538760,-8.31884667390747, -0.521592728360521],[ -0.483370082660210,-0.546513382266653,  -1.03259561080944, -5.634056411215175E-002,-1.29956227047499], [6.31711939355654,  -17.9211469534050,2.04123290467442,  -34.1296928327645,  -4.48631673396142],[3.98736576121289,  -10.9659498207885,  0.902224943866095,-33.7098976109215,  -2.46926873037236],  [11.8753320554643,-33.0453681455719,   3.17578228298450,  -91.6399913284573,-7.77903300921164],   [2.51682526847757,  -6.71006469534049,0.398783425188814,  -33.2952658703072,  -1.35908550919695]]
    name_select=['(sqrt(feature1)/log(feature3))','(log(feature3)/cbrt(feature1))','((feature1-feature2))^-1','(feature3/(feature1-feature2))','(exp(feature3)/(feature1-feature2))','((feature3)^2/(feature1-feature2))']
    ftag_select=[1.53532773794274,   1.89072222412933,1.22829411849416,   1.18588660138395,   1.19662878834092,1.13234411312819]
    score_select=[0.906108704402843,  0.901128081550113,0.890064001214225,  0.914001838777997,  0.912264141449710,0.905684518988813,   1.00000000000000,   1.00000000000000,1.00000000000000,   1.00000000000000,   1.00000000000000,1.00000000000000]
    comp=2

    a = feature_generator()
    # print(a.sis_score(feat, yyy))
    # b=a.reject_list()
    # print(a.goodf(feat,name_feat,dimens,comp))
    # #print(a.goodf(feat,name_feat,dimens,comp,yyy))
    # print(a.isscalar('feature1'))
    #self,fin1,name_in1,lastop_in1,comp_in1,dim_in1,fin2,name_in2,lastop_in2,comp_in2,dim_in2,op,nf
    fin1=[[0.862600000000000,0.726000000000000,0.494300000000000,1.060000000000000E-002,9.700000000000000E-002],[0.704300000000000, 0.781800000000000 ,4.400000000000000E-003,  3.990000000000000E-002, 0.319900000000000],[0.631200000000000,0.611900000000000 ,0.442000000000000 ,0.987700000000000 ,0.550400000000000]]
    name_in1=['feature1' ,'feature2' ,'feature3']
    lastop_in1=[]
    comop_in1=[0, 0, 0]
    dim_in1=[[1.00000000000000,0.000000000000000E+000],[1.00000000000000,0.000000000000000E+000],[0.000000000000000E+000,1.00000000000000]]
    fin2=[[0.862600000000000,  0.726000000000000,0.494300000000000,  1.060000000000000E-002,9.700000000000000E-002],[0.704300000000000,  0.781800000000000,  4.400000000000000E-003,3.990000000000000E-002,0.319900000000000],[0.631200000000000,0.611900000000000,  0.442000000000000,  0.987700000000000,0.550400000000000]]
    name_in2=['feature1' ,'feature2' ,'feature3']
    lastop_in2=[]
    comop_in2=[0, 0, 0]
    dim_in2=[[1.00000000000000,0.000000000000000E+000],[1.00000000000000,0.000000000000000E+000],[0.000000000000000E+000,1.00000000000000]]
    op='NO'
    nf=0
    # a.combine(fin1,name_in1,lastop_in1,comop_in1,dim_in1,fin2,name_in2,lastop_in2,comop_in2,dim_in2,op,nf)
    # print(a.name_select)
    a.feature_contruction()
if __name__== "__main__":
    main()