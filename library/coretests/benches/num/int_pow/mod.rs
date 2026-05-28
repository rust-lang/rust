use rand::Rng;
use test::{Bencher, black_box};

const ITERATIONS: usize = 128; // Uses an ITERATIONS * 20 Byte stack allocation
type IntType = i128; // Hardest native type to multiply
const EXPONENT_MAX: u32 = 31;
const MAX_BASE: IntType = 17; // +-17 ** 31 <= IntType::MAX

macro_rules! pow_bench_template {
    ($name:ident, $inner_macro:ident, $base_macro:ident) => {
        #[bench]
        fn $name(bench: &mut Bencher) {
            // Frequent black_box calls can add latency and prevent optimizations, so for
            // variable parameters we premake an array and pass the
            // reference through black_box outside of the loop.
            let mut rng = crate::bench_rng();
            let base_array: [IntType; ITERATIONS] =
                core::array::from_fn(|_| rng.random_range((-MAX_BASE..=MAX_BASE)));
            let exp_array: [u32; ITERATIONS] =
                core::array::from_fn(|_| rng.random_range((0..=EXPONENT_MAX)));

            bench.iter(|| {
                #[allow(unused, unused_mut)]
                let mut base_iter = black_box(&base_array).into_iter();
                let mut exp_iter = black_box(&exp_array).into_iter();

                (0..ITERATIONS).fold((0 as IntType, false), |acc, _| {
                    // Sometimes constants don't propagate all the way to the
                    // inside of the loop, so we call a custom expression every cycle
                    // rather than iter::repeat(CONST)
                    let base: IntType = $base_macro!(base_iter);
                    let exp: u32 = *exp_iter.next().unwrap();

                    let r: (IntType, bool) = $inner_macro!(base, exp);
                    (acc.0 ^ r.0, acc.1 ^ r.1)
                })
            });
        }
    };
}

// This may panic if it overflows.
macro_rules! inner_pow {
    ($base:ident, $exp:ident) => {
        ($base.pow($exp), false)
    };
}

macro_rules! inner_wrapping {
    ($base:ident, $exp:ident) => {
        ($base.wrapping_pow($exp), false)
    };
}

macro_rules! inner_overflowing {
    ($base:ident, $exp:ident) => {
        $base.overflowing_pow($exp)
    };
}

// This will panic if it overflows.
macro_rules! inner_checked_unwrapped {
    ($base:ident, $exp:ident) => {
        ($base.checked_pow($exp).unwrap(), false)
    };
}

macro_rules! inner_saturating {
    ($base:ident, $exp:ident) => {
        ($base.saturating_pow($exp), false)
    };
}

macro_rules! make_const_base {
    ($name:ident, $x:literal) => {
        macro_rules! $name {
            ($iter:ident) => {
                $x
            };
        }
    };
}

make_const_base!(const_base_m7, -7);
make_const_base!(const_base_m8, -8);

macro_rules! variable_base {
    ($iter:ident) => {
        *$iter.next().unwrap()
    };
}

pow_bench_template!(pow_variable, inner_pow, variable_base);
pow_bench_template!(wrapping_pow_variable, inner_wrapping, variable_base);
pow_bench_template!(overflowing_pow_variable, inner_overflowing, variable_base);
pow_bench_template!(checked_pow_variable, inner_checked_unwrapped, variable_base);
pow_bench_template!(saturating_pow_variable, inner_saturating, variable_base);
pow_bench_template!(pow_m7, inner_pow, const_base_m7);
pow_bench_template!(pow_m8, inner_pow, const_base_m8);
