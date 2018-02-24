import numpy as np
from sklearn.cluster import KMeans, MeanShift, AffinityPropagation, SpectralClustering, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GMM
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift



def find_pNt_kmeans(prices, num_peaks=11,pDropoff=0.07, dT=10):


    max_points=[]
    min_points=[]
    drop_off=[]

    for i in range(len(prices)-dT):
        s = prices[ i:i+dT ]
        #print(s)
        minp, maxp =  min( s ), max( s )


        idx_minp, idx_maxp = np.where(np.array(prices) == minp)[0], np.where(np.array(prices) == maxp)[0]
        idx_minp, idx_maxp = idx_minp[len(idx_minp)-1], idx_maxp[len(idx_maxp)-1]

        print(idx_minp,idx_maxp)
        print("Iter:",i)
        if idx_minp > idx_maxp and (maxp-minp)/minp >= pDropoff:

            max_points.append( (maxp,idx_maxp) )
            min_points.append( (minp,idx_minp) )

    maxPoints = list( set(max_points) )
    minPoints = list( set(min_points) )

    #Seperate Data using k-means
    map_idx = np.array(list(list(zip(*maxPoints))[1]))
    mip_idx = np.array(list(list(zip(*minPoints))[1]))


    map_idxf = map_idx.reshape(-1,1)
    mip_idxf = mip_idx.reshape(-1,1)


    #kmeans = KMeans(n_clusters=num_peaks)
    #kmeans = KMeans( max_iter=500, n_init=20,tol=0.1, algorithm='elkan')
    kmeans = KMeans()
    #kmeans = AffinityPropagation()
    #kmeans = MeanShift()
    #kmeans = AgglomerativeClustering()
    kmeans.fit(mip_idxf)
    ymip_kmeans = kmeans.predict(mip_idxf)

    print( kmeans.fit(map_idxf) )
    ymap_kmeans = kmeans.predict(map_idxf)

    print(ymap_kmeans)


    print('start')
    maxPointsN=[]
    for i in range(max(ymap_kmeans)):
        group=[]
        for j in range(len(ymap_kmeans)):
            if i==ymap_kmeans[j]:
                group.append(map_idx[j])
        max_idx = min(group)
        for k in range(len(maxPoints)):
            if maxPoints[k][1]==max_idx:
                maxPointsN.append(  maxPoints[k] )

    print('start')
    minPointsN=[]
    for i in range(max(ymip_kmeans)):
        group=[]
        for j in range(len(ymip_kmeans)):
            if i==ymip_kmeans[j]:
                group.append(mip_idx[j])
        min_idx = max(group)
        for k in range(len(minPoints)):
            if minPoints[k][1]==min_idx:
                minPointsN.append(  minPoints[k] )


    return maxPoints, minPoints, ymap_kmeans ,ymip_kmeans, maxPointsN, minPointsN


def find_pNt_gmm(prices, num_peaks=10,pDropoff=0.07, dT=10):


    max_points=[]
    min_points=[]
    drop_off=[]

    for i in range(len(prices)-dT):
        s = prices[ i:i+dT ]
        #print(s)
        minp, maxp =  min( s ), max( s )


        idx_minp, idx_maxp = np.where(np.array(prices) == minp)[0], np.where(np.array(prices) == maxp)[0]
        idx_minp, idx_maxp = idx_minp[len(idx_minp)-1], idx_maxp[len(idx_maxp)-1]

        print(idx_minp,idx_maxp)
        print("Iter:",i)
        if idx_minp > idx_maxp and (maxp-minp)/minp >= pDropoff:

            max_points.append( (maxp,idx_maxp) )
            min_points.append( (minp,idx_minp) )

    maxPoints = list( set(max_points) )
    minPoints = list( set(min_points) )

    #Seperate Data using k-means
    map_idx = np.array(list(list(zip(*maxPoints))[1]))
    mip_idx = np.array(list(list(zip(*minPoints))[1]))


    map_idxf = map_idx.reshape(-1,1)
    mip_idxf = mip_idx.reshape(-1,1)


    #kmeans = KMeans(n_clusters=num_peaks)
    #gmm = GMM(n_components=num_peaks)
    gmm = GMM(n_components=num_peaks,covariance_type='full',random_state=0)
    gmm.fit(mip_idxf)
    ymip_gmm = gmm.predict(mip_idxf)
    prob = gmm.predict_proba(mip_idxf)
    print(prob)

    print( gmm.fit(map_idxf) )
    ymap_gmm = gmm.predict(map_idxf)

    print('gmm predicts')
    print(ymap_gmm)
    print( max(list(ymap_gmm)) )
    print(ymip_gmm)


    print('start')
    maxPointsN=[]
    for i in range( max(list(ymap_gmm)) ):
        group=[]
        for j in range(len(ymap_gmm)):
            if i==ymap_gmm[j]:
                group.append( map_idx[j] )
        if len(group) != 0:
            max_idx = min(group)
            for k in range(len(maxPoints)):
                if maxPoints[k][1]==max_idx:
                    maxPointsN.append(  maxPoints[k] )

    print('start')
    minPointsN=[]
    for i in range(max(ymip_gmm)):
        group=[]
        for j in range(len(ymip_gmm)):
            if i==ymip_gmm[j]:
                group.append(mip_idx[j])
        if len(group) != 0:
            min_idx = max(group)
            for k in range(len(minPoints)):
                if minPoints[k][1]==min_idx:
                    minPointsN.append(  minPoints[k] )


    return maxPoints, minPoints, ymap_gmm ,ymip_gmm, maxPointsN, minPointsN



def find_pNt_gmm(prices, num_peaks=10,pDropoff=0.07, dT=10):


    max_points=[]
    min_points=[]
    drop_off=[]

    for i in range(len(prices)-dT):
        s = prices[ i:i+dT ]
        #print(s)
        minp, maxp =  min( s ), max( s )


        idx_minp, idx_maxp = np.where(np.array(prices) == minp)[0], np.where(np.array(prices) == maxp)[0]
        idx_minp, idx_maxp = idx_minp[len(idx_minp)-1], idx_maxp[len(idx_maxp)-1]

        print(idx_minp,idx_maxp)
        print("Iter:",i)
        if idx_minp > idx_maxp and (maxp-minp)/minp >= pDropoff:

            max_points.append( (maxp,idx_maxp) )
            min_points.append( (minp,idx_minp) )

    maxPoints = list( set(max_points) )
    minPoints = list( set(min_points) )

    #Seperate Data using k-means
    map_idx = np.array(list(list(zip(*maxPoints))[1]))
    mip_idx = np.array(list(list(zip(*minPoints))[1]))


    map_idxf = map_idx.reshape(-1,1)
    mip_idxf = mip_idx.reshape(-1,1)


    #kmeans = KMeans(n_clusters=num_peaks)
    #gmm = GMM(n_components=num_peaks)
    gmm = GMM(n_components=num_peaks,covariance_type='full',random_state=0)
    gmm.fit(mip_idxf)
    ymip_gmm = gmm.predict(mip_idxf)
    prob = gmm.predict_proba(mip_idxf)
    print(prob)

    print( gmm.fit(map_idxf) )
    ymap_gmm = gmm.predict(map_idxf)

    print('gmm predicts')
    print(ymap_gmm)
    print( max(list(ymap_gmm)) )
    print(ymip_gmm)


    print('start')
    maxPointsN=[]
    for i in range( max(list(ymap_gmm)) ):
        group=[]
        for j in range(len(ymap_gmm)):
            if i==ymap_gmm[j]:
                group.append( map_idx[j] )
        if len(group) != 0:
            max_idx = min(group)
            for k in range(len(maxPoints)):
                if maxPoints[k][1]==max_idx:
                    maxPointsN.append(  maxPoints[k] )

    print('start')
    minPointsN=[]
    for i in range(max(ymip_gmm)):
        group=[]
        for j in range(len(ymip_gmm)):
            if i==ymip_gmm[j]:
                group.append(mip_idx[j])
        if len(group) != 0:
            min_idx = max(group)
            for k in range(len(minPoints)):
                if minPoints[k][1]==min_idx:
                    minPointsN.append(  minPoints[k] )


    return maxPoints, minPoints, ymap_gmm ,ymip_gmm, maxPointsN, minPointsN