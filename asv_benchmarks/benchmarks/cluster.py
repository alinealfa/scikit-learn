from sklearn.cluster import KMeans, MiniBatchKMeans

from .common import Benchmark, Estimator, Predictor, Transformer
from .datasets import _blobs_dataset, _20newsgroups_highdim_dataset
from .utils import neg_mean_inertia


class KMeansBenchmark(Predictor, Transformer, Estimator, Benchmark):
    """
    Benchmarks for KMeans.
    """

    param_names = ["representation", "algorithm", "init"]
    params = (["dense", "sparse"], ["full", "elkan"], ["random", "k-means++"])

    def setup_cache(self):
        super().setup_cache()

    def make_data(self, params):
        representation, algorithm, init = params

        return (
            _20newsgroups_highdim_dataset(n_samples=8000)
            if representation == "sparse"
            else _blobs_dataset(n_clusters=20)
        )

    def make_estimator(self, params):
        representation, algorithm, init = params

        max_iter = 30 if representation == "sparse" else 100

        return KMeans(
            n_clusters=20,
            algorithm=algorithm,
            init=init,
            n_init=1,
            max_iter=max_iter,
            tol=-1,
            random_state=0,
        )

    def make_scorers(self):
        self.train_scorer = lambda _, __: neg_mean_inertia(
            self.X, self.estimator.predict(self.X), self.estimator.cluster_centers_
        )
        self.test_scorer = lambda _, __: neg_mean_inertia(
            self.X_val,
            self.estimator.predict(self.X_val),
            self.estimator.cluster_centers_,
        )


class MiniBatchKMeansBenchmark(Predictor, Transformer, Estimator, Benchmark):
    """
    Benchmarks for MiniBatchKMeans.
    """

    param_names = ["representation", "init"]
    params = (["dense", "sparse"], ["random", "k-means++"])

    def setup_cache(self):
        super().setup_cache()

    def make_data(self, params):
        representation, init = params

        return (
            _20newsgroups_highdim_dataset()
            if representation == "sparse"
            else _blobs_dataset(n_clusters=20)
        )

    def make_estimator(self, params):
        representation, init = params

        max_iter = 5 if representation == "sparse" else 2

        return MiniBatchKMeans(
            n_clusters=20,
            init=init,
            n_init=1,
            max_iter=max_iter,
            batch_size=1000,
            max_no_improvement=None,
            compute_labels=False,
            random_state=0,
        )

    def make_scorers(self):
        self.train_scorer = lambda _, __: neg_mean_inertia(
            self.X, self.estimator.predict(self.X), self.estimator.cluster_centers_
        )
        self.test_scorer = lambda _, __: neg_mean_inertia(
            self.X_val,
            self.estimator.predict(self.X_val),
            self.estimator.cluster_centers_,
        )
