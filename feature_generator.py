import numpy as np



class feature_generator():
    def __init__(self):
        self.data=[]
        self.ntask = 1
        self.nsample=[0]*(self.ntask+1)
        self.nsample[1]=5

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

def main():
    feat = [0.369091719388961791992187500000000,
            0.322154790163040161132812500000000,
            4.730276941700140014290809631347656E-0006,
            1.788782384437581640668213367462158E-0007,
            9.628795087337493896484375000000000E-0004
            ]
    yyy = [-4.430929943919181823730468750000000E-0002,
           5.101694539189338684082031250000000E-0002,
           -0.111230455338954925537109375000000,
           0.188298255205154418945312500000000,
           -8.377544581890106201171875000000000E-0002]
    a = feature_generator()
    print(a.sis_score(feat, yyy))

if __name__== "__main__":
    main()