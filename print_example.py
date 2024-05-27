import umap

data = [[-5, -4],
        [-6, -5],
        [-6, -6],
        [5, 3],
        [5, 5],
        [6, 4]]

reducer = umap.UMAP(n_neighbors=4, n_components=1, random_state=69, n_jobs=1)

res = reducer.fit_transform(data)
