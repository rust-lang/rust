//! Benchmarking module.
pub use std::bench::{Bencher, TimeIterations};
pub use std::hint::black_box;

use super::{
    event::CompletedTest,
    helpers::sink::Sink,
    types::TestDesc,
    test_result::TestResult,
    Sender,
};

use crate::stats;
use std::time::{Duration, Instant};
use std::cmp;
use std::io;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

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
        output
            .write_fmt(format_args!(" = {} MB/s", bs.mb_s))
            .unwrap();
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

pub fn run_and_summarize(inner: TimeIterations<'_>) -> stats::Summary {
    // Initial bench run to get ballpark figure.
    let ns_single = inner(1);

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
            *p = inner(n) as f64 / n as f64;
        }

        stats::winsorize(samples, 5.0);
        let summ = stats::Summary::new(samples);

        for p in &mut *samples {
            let ns = inner(5 * n);
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

pub fn benchmark<F>(desc: TestDesc, monitor_ch: Sender<CompletedTest>, nocapture: bool, mut f: F)
where
    F: FnMut(&mut Bencher<'_>),
{
    let data = Arc::new(Mutex::new(Vec::new()));
    let oldio = if !nocapture {
        Some((
            io::set_print(Some(Sink::new_boxed(&data))),
            io::set_panic(Some(Sink::new_boxed(&data))),
        ))
    } else {
        None
    };

    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut summary = None;
        f(&mut Bencher::new(&mut |bytes, iter| {
            summary = Some((bytes, run_and_summarize(iter)));
        }));
        summary
    }));

    if let Some((printio, panicio)) = oldio {
        io::set_print(printio);
        io::set_panic(panicio);
    }

    let test_result = match result {
        //bs.bench(f) {
        Ok(Some((bytes, ns_iter_summ))) => {
            let ns_iter = cmp::max(ns_iter_summ.median as u64, 1);
            let mb_s = bytes * 1000 / ns_iter;

            let bs = BenchSamples {
                ns_iter_summ,
                mb_s: mb_s as usize,
            };
            TestResult::TrBench(bs)
        }
        Ok(None) => {
            // iter not called, so no data.
            // FIXME: error in this case?
            let samples: &mut [f64] = &mut [0.0_f64; 1];
            let bs = BenchSamples {
                ns_iter_summ: stats::Summary::new(samples),
                mb_s: 0,
            };
            TestResult::TrBench(bs)
        }
        Err(_) => TestResult::TrFailed,
    };

    let stdout = data.lock().unwrap().to_vec();
    let message = CompletedTest::new(desc, test_result, None, stdout);
    monitor_ch.send(message).unwrap();
}

pub fn run_once<F>(mut f: F)
where
    F: FnMut(&mut Bencher<'_>),
{
    f(&mut Bencher::new(&mut |_bytes, iter| {
        iter(1);
    }));
}
