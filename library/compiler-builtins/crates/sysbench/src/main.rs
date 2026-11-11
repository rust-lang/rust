//! Simple check of integer and float ops, for giving us an idea of how many test cases we can
//! expect to execute in a reasonable amount of time.

#![allow(clippy::type_complexity)]

use std::hint::black_box as bb;
use std::io::Write;
use std::time::{Duration, Instant};
use std::{io, thread};

const ESTIMATE_ITERS: u32 = 65535;
const ITER_MAX: u32 = u32::MAX;

fn main() {
    let int_mul = run_bench("integer multiplication", |i| {
        let i = i128::from(i);
        let i = i | (i << 32) | (i << 64) | (i << 96);
        bb(i).overflowing_mul(bb(i))
    });
    let float_mul = run_bench("float multiplication", |i| {
        let x = f32::from_bits(i);
        bb(x) * bb(x)
    });
    let float_fma = run_bench("float fma", |i| {
        let x = f32::from_bits(i);
        bb(x).mul_add(bb(x), bb(x))
    });

    let oneshot = bench_oneshot();

    let threads = thread::available_parallelism().unwrap().get();
    println!("Available parallelism: {threads}");

    // Super simple metric to give us a rough idea of how machines compare. Int math gets a larger
    // scale since u128 requires more instructions.
    let int_score = 500.0 / int_mul;
    let float_score = 200.0 / float_mul;
    let fma_score = 200.0 / float_fma;
    let oneshot_score = 50.0 / oneshot.as_secs_f64();
    let score = int_score + float_score + fma_score + oneshot_score;
    let thread_scale = 1.0 + 0.9 * (threads as f64 - 1.0);

    println!("Single-core performance score: {score:.0}");
    println!("Multi-core performance score: {:.0}", score * thread_scale);
}

/// Benchmark `f` by running it repeatedly after a warmup. Outputs are black boxed.
fn run_bench<T>(name: &str, mut f: impl FnMut(u32) -> T) -> f64 {
    print!("Starting {name} bench... ");
    io::stdout().flush().unwrap();

    // Warmup
    for i in 0..1000 {
        bb(f(bb(i)));
    }

    // Calculate how many to run
    let start = Instant::now();
    for i in 0..ESTIMATE_ITERS {
        bb(f(bb(i)));
    }
    let elapsed = start.elapsed();

    let ns_per_iter = elapsed.as_secs_f64() * 1e9 / ESTIMATE_ITERS as f64;
    let desired_iters = 1e9 / ns_per_iter;
    let step = ((ITER_MAX as u64 + 1) / desired_iters as u64) as usize;
    let iter = (0..=ITER_MAX).step_by(step);
    let count = iter_count(iter.clone());

    // Run the benchmark
    let start = Instant::now();
    for i in iter {
        bb(f(bb(i)));
    }
    let elapsed = start.elapsed();

    let ns_per_iter = elapsed.as_secs_f64() * 1e9 / count as f64;
    println!("{count} iterations completed in {elapsed:.3?} ({ns_per_iter:.3} ns/iter)");
    ns_per_iter
}

/// Qemu has a good JIT, so run something new once to try to catch it.
fn bench_oneshot() -> Duration {
    fn fib_int(n: u32) -> u32 {
        if n == 0 {
            0
        } else if n == 1 {
            1
        } else {
            fib_int(n - 1) + fib_int(n - 2)
        }
    }

    fn fib_float(n: f32) -> f32 {
        if n == 0.0 {
            0.0
        } else if n == 1.0 {
            1.0
        } else {
            fib_float(n - 1.0) + fib_float(n - 2.0)
        }
    }

    print!("Starting oneshot bench... ");
    io::stdout().flush().unwrap();
    let start = Instant::now();
    bb(fib_int(bb(38)));
    bb(fib_float(bb(38.0)));
    let elapsed = start.elapsed();
    println!("completed in {elapsed:.3?}");
    elapsed
}

/// `iter.count()` but allow results greater than `usize::MAX` on 32-bit.
fn iter_count(iter: impl Iterator) -> u64 {
    let mut ret = 0u64;
    for _ in iter {
        ret = ret.strict_add(1);
    }
    ret
}
