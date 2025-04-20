//! Benchmark metrics.

use std::collections::BTreeMap;

#[derive(Clone, PartialEq, Debug, Copy)]
pub struct Metric {
    value: f64,
    noise: f64,
}

impl Metric {
    pub fn new(value: f64, noise: f64) -> Metric {
        Metric { value, noise }
    }
}

#[derive(Clone, PartialEq)]
pub struct MetricMap(BTreeMap<String, Metric>);

impl MetricMap {
    pub fn new() -> MetricMap {
        MetricMap(BTreeMap::new())
    }

    /// Insert a named `value` (+/- `noise`) metric into the map. The value
    /// must be non-negative. The `noise` indicates the uncertainty of the
    /// metric, which doubles as the "noise range" of acceptable
    /// pairwise-regressions on this named value, when comparing from one
    /// metric to the next using `compare_to_old`.
    ///
    /// If `noise` is positive, then it means this metric is of a value
    /// you want to see grow smaller, so a change larger than `noise` in the
    /// positive direction represents a regression.
    ///
    /// If `noise` is negative, then it means this metric is of a value
    /// you want to see grow larger, so a change larger than `noise` in the
    /// negative direction represents a regression.
    pub fn insert_metric(&mut self, name: &str, value: f64, noise: f64) {
        let m = Metric { value, noise };
        self.0.insert(name.to_owned(), m);
    }

    pub fn fmt_metrics(&self) -> String {
        let v = self
            .0
            .iter()
            .map(|(k, v)| format!("{}: {} (+/- {})", *k, v.value, v.noise))
            .collect::<Vec<_>>();
        v.join(", ")
    }
}
