//! Benchmarking module.
pub use std::hint::black_box;

use super::{
    event::CompletedTest, helpers::sink::Sink, options::BenchMode, test_result::TestResult,
    types::TestDesc, Sender,
};

use crate::stats;
use std::cmp;
use std::io;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Manager of the benchmarking runs.
///
/// This is fed into functions marked with `#[bench]` to allow for
/// set-up & tear-down before running a piece of code repeatedly via a
/// call to `iter`.
#[derive(Clone)]
pub struct Bencher {
    mode: BenchMode,
    summary: Option<stats::Summary>,
    pub bytes: u64,
}

impl Bencher {
    /// Callback for benchmark functions to run in their body.
    pub fn iter<T, F>(&mut self, mut inner: F)
    where
        F: FnMut() -> T,
    {
        if self.mode == BenchMode::Single {
            ns_iter_inner(&mut inner, 1);
            return;
        }

        self.summary = Some(iter(&mut inner));
    }

    /// Like iter but limits the amount of execution cycles.
    /// Useful for benchmarking functions that need unique pre-generated
    /// data for each iteration.
    /// For better results set max_iter as high as possible.
    pub fn iter_max<T, F>(&mut self, mut inner: F, max_iter: u64)
    where
        F: FnMut() -> T,
    {
        if self.mode == BenchMode::Single {
            ns_iter_inner(&mut inner, 1);
            return;
        }

        self.summary = Some(iter_max(&mut inner, max_iter));
    }

    pub fn bench<F>(&mut self, mut f: F) -> Option<stats::Summary>
    where
        F: FnMut(&mut Bencher),
    {
        f(self);
        self.summary
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BenchSamples {
    pub ns_iter_summ: stats::Summary,
    pub mb_s: usize,
}

pub fn fmt_bench_samples(bs: &BenchSamples) -> String {
    use std::fmt::Write;
    let mut output = String::new();

    let median = bs.ns_iter_summ.median as usize;
    let deviation = (bs.ns_iter_summ.max - bs.ns_iter_summ.min) as usize;

    output
        .write_fmt(format_args!(
            "{:>11} ns/iter (+/- {})",
            fmt_thousands_sep(median, ','),
            fmt_thousands_sep(deviation, ',')
        ))
        .unwrap();
    if bs.mb_s != 0 {
        output.write_fmt(format_args!(" = {} MB/s", bs.mb_s)).unwrap();
    }
    output
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

fn ns_from_dur(dur: Duration) -> u64 {
    dur.as_secs() * 1_000_000_000 + (dur.subsec_nanos() as u64)
}

fn ns_iter_inner<T, F>(inner: &mut F, k: u64) -> u64
where
    F: FnMut() -> T,
{
    let start = Instant::now();
    for _ in 0..k {
        black_box(inner());
    }
    ns_from_dur(start.elapsed())
}

pub fn iter<T, F>(inner: &mut F) -> stats::Summary
where
    F: FnMut() -> T,
{
    // Initial bench run to get ballpark figure.
    let ns_single = ns_iter_inner(inner, 1);

    // Try to estimate iter count for 1ms falling back to 1m
    // iterations if first run took < 1ns.
    let ns_target_total = 1_000_000; // 1ms
    let mut n = ns_target_total / cmp::max(1, ns_single);

    // if the first run took more than 1ms we don't want to just
    // be left doing 0 iterations on every loop. The unfortunate
    // side effect of not being able to do as many runs is
    // automatically handled by the statistical analysis below
    // (i.e., larger error bars).
    n = cmp::max(1, n);

    let mut total_run = Duration::new(0, 0);
    let samples: &mut [f64] = &mut [0.0_f64; 50];
    loop {
        let loop_start = Instant::now();

        for p in &mut *samples {
            *p = ns_iter_inner(inner, n) as f64 / n as f64;
        }

        stats::winsorize(samples, 5.0);
        let summ = stats::Summary::new(samples);

        for p in &mut *samples {
            let ns = ns_iter_inner(inner, 5 * n);
            *p = ns as f64 / (5 * n) as f64;
        }

        stats::winsorize(samples, 5.0);
        let summ5 = stats::Summary::new(samples);

        let loop_run = loop_start.elapsed();

        // If we've run for 100ms and seem to have converged to a
        // stable median.
        if loop_run > Duration::from_millis(100)
            && summ.median_abs_dev_pct < 1.0
            && summ.median - summ5.median < summ5.median_abs_dev
        {
            return summ5;
        }

        total_run = total_run + loop_run;
        // Longest we ever run for is 3s.
        if total_run > Duration::from_secs(3) {
            return summ5;
        }

        // If we overflow here just return the results so far. We check a
        // multiplier of 10 because we're about to multiply by 2 and the
        // next iteration of the loop will also multiply by 5 (to calculate
        // the summ5 result)
        n = match n.checked_mul(10) {
            Some(_) => n * 2,
            None => {
                return summ5;
            }
        };
    }
}

pub fn iter_max<T, F>(inner: &mut F, max_iter: u64) -> stats::Summary
where
    F: FnMut() -> T,
{
    assert_ne!(max_iter, 0);

    // max_iter is so small that it's doesn't make sense to calculate a
    // ballpark figure.
    if max_iter < 10 {
        let mut samples = vec![0.0_f64; max_iter as usize];
        let samples: &mut [f64] = samples.as_mut_slice();

        for p in &mut *samples {
            *p = ns_iter_inner(inner, 1) as f64;
        }

        stats::winsorize(samples, 5.0);
        return stats::Summary::new(samples);
    }

    // Initial bench run to get ballpark figure.
    let ns_single = ns_iter_inner(inner, 1);
    let max_iter: u64 = max_iter - 1;

    // Try to estimate iter count for 1ms falling back to 1m
    // iterations if first run took < 1ns.
    let ns_target_total = 1_000_000; // 1ms
    let mut n = ns_target_total / cmp::max(1, ns_single);

    // If the first run took more than 1ms we don't want to just
    // be left doing 0 iterations on every loop. The unfortunate
    // side effect of not being able to do as many runs is
    // automatically handled by the statistical analysis below
    // (i.e., larger error bars).
    n = cmp::max(1, n);

    // max_iter is too small to run statistically accurate tests
    // with n iterations per sample. Instead we're measuring only
    // one iteration per sample.
    if max_iter < n * 5 {
        let mut samples = vec![0.0_f64; max_iter as usize];
        let samples: &mut [f64] = samples.as_mut_slice();

        for p in &mut *samples {
            *p = ns_iter_inner(inner, 1) as f64;
        }

        stats::winsorize(samples, 5.0);
        return stats::Summary::new(samples);
    }

    // max_iter is now large enough for at least five samples
    // with n iterations
    if max_iter < 300 * n {
        // If possible multiply the sample iterations so we can get more
        // stable results but still end up with about 40 samples on average.
        // Example: max_iter = 160 * n => k = n * 4; using 40 iterations
        let k: u64 = match n.checked_mul(max_iter / (50 * n) + 1) {
            Some(k) => k,
            None => n,
        };
        let iterations: u64 = max_iter / k;

        let mut samples = vec![0.0_f64; iterations as usize];
        let samples: &mut [f64] = samples.as_mut_slice();

        for p in &mut *samples {
            *p = ns_iter_inner(inner, k) as f64 / k as f64;
        }

        stats::winsorize(samples, 5.0);
        return stats::Summary::new(samples);
    }

    // max_iter is large enough to run the loop at least once.
    // We will keep track of the consumed iterations and only continue looping
    // if enough iterations are still available.
    let mut consumed_iterations: u64 = 0;

    let mut total_run = Duration::new(0, 0);
    let samples: &mut [f64] = &mut [0.0_f64; 50];
    loop {
        let loop_start = Instant::now();

        // measure the time of 50 samples running n times each
        for p in &mut *samples {
            *p = ns_iter_inner(inner, n) as f64 / n as f64;
        }

        stats::winsorize(samples, 5.0);
        let summ = stats::Summary::new(samples);

        // measure the time of 50 samples running n * 5 times each
        for p in &mut *samples {
            let ns = ns_iter_inner(inner, 5 * n);
            *p = ns as f64 / (5 * n) as f64;
        }

        stats::winsorize(samples, 5.0);
        let summ5 = stats::Summary::new(samples);

        let loop_run = loop_start.elapsed();

        // So far we've run (50 * n) + (50 * 5 * n) = 300 * n iterations
        consumed_iterations += 300 * n;

        // If we've run for 100ms and seem to have converged to a
        // stable median.
        if loop_run > Duration::from_millis(100)
            && summ.median_abs_dev_pct < 1.0
            && summ.median - summ5.median < summ5.median_abs_dev
        {
            return summ5;
        }

        total_run = total_run + loop_run;
        // Longest we ever run for is 3s.
        if total_run > Duration::from_secs(3) {
            return summ5;
        }

        // If we overflow here just return the results so far. We check a
        // multiplier of 10 because we're about to multiply by 2 and the
        // next iteration of the loop will also multiply by 5 (to calculate
        // the summ5 result)
        n = match n.checked_mul(10) {
            Some(_) => n * 2,
            None => {
                return summ5;
            }
        };

        // Check whether the next iteration of the loop would exceed the limit
        // specified by max_iter. If it does we just return the current results.
        if consumed_iterations + n * 300 > max_iter {
            return summ5;
        }
    }
}

pub fn benchmark<F>(desc: TestDesc, monitor_ch: Sender<CompletedTest>, nocapture: bool, f: F)
where
    F: FnMut(&mut Bencher),
{
    let mut bs = Bencher { mode: BenchMode::Auto, summary: None, bytes: 0 };

    let data = Arc::new(Mutex::new(Vec::new()));
    let oldio = if !nocapture {
        Some((
            io::set_print(Some(Sink::new_boxed(&data))),
            io::set_panic(Some(Sink::new_boxed(&data))),
        ))
    } else {
        None
    };

    let result = catch_unwind(AssertUnwindSafe(|| bs.bench(f)));

    if let Some((printio, panicio)) = oldio {
        io::set_print(printio);
        io::set_panic(panicio);
    }

    let test_result = match result {
        //bs.bench(f) {
        Ok(Some(ns_iter_summ)) => {
            let ns_iter = cmp::max(ns_iter_summ.median as u64, 1);
            let mb_s = bs.bytes * 1000 / ns_iter;

            let bs = BenchSamples { ns_iter_summ, mb_s: mb_s as usize };
            TestResult::TrBench(bs)
        }
        Ok(None) => {
            // iter not called, so no data.
            // FIXME: error in this case?
            let samples: &mut [f64] = &mut [0.0_f64; 1];
            let bs = BenchSamples { ns_iter_summ: stats::Summary::new(samples), mb_s: 0 };
            TestResult::TrBench(bs)
        }
        Err(_) => TestResult::TrFailed,
    };

    let stdout = data.lock().unwrap().to_vec();
    let message = CompletedTest::new(desc, test_result, None, stdout);
    monitor_ch.send(message).unwrap();
}

pub fn run_once<F>(f: F)
where
    F: FnMut(&mut Bencher),
{
    let mut bs = Bencher { mode: BenchMode::Single, summary: None, bytes: 0 };
    bs.bench(f);
}
