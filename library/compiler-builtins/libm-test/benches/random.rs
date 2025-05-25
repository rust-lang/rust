use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_main};
use libm_test::generate::random;
use libm_test::generate::random::RandomInput;
use libm_test::{CheckBasis, CheckCtx, GeneratorKind, MathOp, TupleCall};

/// Benchmark with this many items to get a variety
const BENCH_ITER_ITEMS: usize = if cfg!(feature = "short-benchmarks") {
    50
} else {
    500
};

/// Extra parameters we only care about if we are benchmarking against musl.
#[allow(dead_code)]
struct MuslExtra<F> {
    musl_fn: Option<F>,
    skip_on_i586: bool,
}

macro_rules! musl_rand_benches {
    (
        fn_name: $fn_name:ident,
        attrs: [$($attr:meta),*],
        fn_extra: ($skip_on_i586:expr, $musl_fn:expr),
    ) => {
        paste::paste! {
            $(#[$attr])*
            fn [< musl_bench_ $fn_name >](c: &mut Criterion) {
                type Op = libm_test::op::$fn_name::Routine;

                #[cfg(feature = "build-musl")]
                let musl_extra = MuslExtra::<libm_test::OpCFn<Op>> {
                    musl_fn: $musl_fn,
                    skip_on_i586: $skip_on_i586,
                };

                #[cfg(not(feature = "build-musl"))]
                let musl_extra = MuslExtra {
                    musl_fn: None,
                    skip_on_i586: $skip_on_i586,
                };

                bench_one::<Op>(c, musl_extra);
            }
        }
    };
}

fn bench_one<Op>(c: &mut Criterion, musl_extra: MuslExtra<Op::CFn>)
where
    Op: MathOp,
    Op::RustArgs: RandomInput,
{
    let name = Op::NAME;

    let ctx = CheckCtx::new(Op::IDENTIFIER, CheckBasis::Musl, GeneratorKind::Random);
    let benchvec: Vec<_> = random::get_test_cases::<Op::RustArgs>(&ctx)
        .0
        .take(BENCH_ITER_ITEMS)
        .collect();

    // Perform a sanity check that we are benchmarking the same thing
    // Don't test against musl if it is not available
    #[cfg(feature = "build-musl")]
    for input in benchvec.iter().copied() {
        use anyhow::Context;
        use libm_test::CheckOutput;

        if cfg!(x86_no_sse) && musl_extra.skip_on_i586 {
            break;
        }

        let Some(musl_fn) = musl_extra.musl_fn else {
            continue;
        };
        let musl_res = input.call(musl_fn);
        let crate_res = input.call(Op::ROUTINE);

        crate_res
            .validate(musl_res, input, &ctx)
            .context(name)
            .unwrap();
    }

    #[cfg(not(feature = "build-musl"))]
    let _ = musl_extra; // silence unused warnings

    /* Option pointers are black boxed to avoid inlining in the benchmark loop */

    let mut group = c.benchmark_group(name);
    group.bench_function("crate", |b| {
        b.iter(|| {
            let f = black_box(Op::ROUTINE);
            for input in benchvec.iter().copied() {
                input.call(f);
            }
        })
    });

    // Don't test against musl if it is not available
    #[cfg(feature = "build-musl")]
    {
        if let Some(musl_fn) = musl_extra.musl_fn {
            group.bench_function("musl", |b| {
                b.iter(|| {
                    let f = black_box(musl_fn);
                    for input in benchvec.iter().copied() {
                        input.call(f);
                    }
                })
            });
        }
    }
}

libm_macros::for_each_function! {
    callback: musl_rand_benches,
    skip: [],
    fn_extra: match MACRO_FN_NAME {
        // We pass a tuple of `(skip_on_i586, musl_fn)`

        // FIXME(correctness): exp functions have the wrong result on i586
        exp10 | exp10f | exp2 | exp2f => (true, Some(musl_math_sys::MACRO_FN_NAME)),

        // Musl does not provide `f16` and `f128` functions, as well as a handful of others
        fmaximum
        | fmaximum_num
        | fmaximum_numf
        | fmaximumf
        | fminimum
        | fminimum_num
        | fminimum_numf
        | fminimumf
        | roundeven
        | roundevenf
        | ALL_F16
        | ALL_F128 => (false, None),

        // By default we never skip (false) and always have a musl function available
        _ => (false, Some(musl_math_sys::MACRO_FN_NAME))
    }
}

macro_rules! run_callback {
    (
        fn_name: $fn_name:ident,
        attrs: [$($attr:meta),*],
        extra: [$criterion:ident],
    ) => {
        paste::paste! {
            $(#[$attr])*
            [< musl_bench_ $fn_name >](&mut $criterion)
        }
    };
}

pub fn musl_random() {
    let mut criterion = Criterion::default();

    // For CI, run a short 0.5s warmup and 1.0s tests. This makes benchmarks complete in
    // about the same time as other tests.
    if cfg!(feature = "short-benchmarks") {
        criterion = criterion
            .warm_up_time(Duration::from_millis(200))
            .measurement_time(Duration::from_millis(600));
    }

    criterion = criterion.configure_from_args();

    libm_macros::for_each_function! {
        callback: run_callback,
        extra: [criterion],
    };
}

criterion_main!(musl_random);
