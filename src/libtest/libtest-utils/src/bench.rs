//! Benchmark utilities

pub mod stats;

use std::{collections::BTreeMap, fmt};

pub type Desc = crate::test::Desc;

#[derive(Clone, Debug, PartialEq)]
pub struct BenchSamples {
    pub ns_iter_summ: stats::Summary,
    pub mb_s: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Result {
    Ok(BenchSamples),
    Failed,
}

// Format a number with thousands separators
fn fmt_thousands_sep(mut n: usize, sep: char) -> String {
    use std::fmt::Write;
    let mut output = String::new();
    let mut trailing = false;
    for &pow in &[9, 6, 3, 0] {
        let base = 10_usize.pow(pow);
        if pow == 0 || trailing || n / base != 0 {
            if !trailing {
                output.write_fmt(format_args!("{}", n / base)).unwrap();
            } else {
                output.write_fmt(format_args!("{:03}", n / base)).unwrap();
            }
            if pow != 0 {
                output.push(sep);
            }
            trailing = true;
        }
        n %= base;
    }

    output
}

impl fmt::Display for BenchSamples {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use std::fmt::Write;
        let mut output = String::new();

        let median = self.ns_iter_summ.median as usize;
        let deviation = (self.ns_iter_summ.max - self.ns_iter_summ.min) as usize;

        output
            .write_fmt(format_args!(
                "{:>11} ns/iter (+/- {})",
                fmt_thousands_sep(median, ','),
                fmt_thousands_sep(deviation, ',')
            ))
            .unwrap();
        if self.mb_s != 0 {
            output
                .write_fmt(format_args!(" = {} MB/s", self.mb_s))
                .unwrap();
        }
        write!(f, "{}", output)
    }
}

#[derive(Clone, PartialEq)]
pub struct MetricMap(pub BTreeMap<String, Metric>);

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

#[derive(Clone, PartialEq, Debug, Copy)]
pub struct Metric {
    pub value: f64,
    pub noise: f64,
}

impl Metric {
    pub fn new(value: f64, noise: f64) -> Metric {
        Metric { value, noise }
    }
}
