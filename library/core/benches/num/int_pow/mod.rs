#![allow(unused_macros)] // TODO
use rand::Rng;
use test::{black_box, Bencher};

const ITERATIONS: usize = 128;
const SMALL_EXPONENT_MAX: u32 = 6;

macro_rules! pow_bench_template {
    ($name:ident, $t:ty, $inner_macro:ident, $base_macro:ident, $exp_macro:ident, $attribute:meta) => {
        #[$attribute] // cfg(all()) makes a nice no-op.
        #[bench]
        fn $name(bench: &mut Bencher) {
            // Frequent black_box calls can add latency and prevent optimizations, so for
            // variable parameters we premake an array and pass the
            // reference through black_box outside of the loop.
            $base_macro!(1, $t, base_ident);
            $exp_macro!(1, $t, exp_ident);

            bench.iter(|| {
                (0..ITERATIONS).fold((0 as $t, false), |acc, _| {
                    // Sometimes constants don't propogate all the way to the
                    // inside of the loop, so we call a custom expression every cycle.
                    // This allows us to use literals and `const` variables.
                    let base: $t = $base_macro!(2, $t, base_ident, acc);
                    let exp: u32 = $exp_macro!(2, $t, exp_ident, acc);

                    let r: ($t, bool) = $inner_macro!($t, base, exp);
                    (acc.0 | r.0, acc.1 | r.1)
                })
            });
        }
    };
}

// This may panic if it overflows.
// Consider running this bench along with an equivalent pow_unwrapped.
macro_rules! inner_pow {
    ($t:ty, $base:ident, $exp:ident) => {
        ($base.pow($exp), false)
    };
}

macro_rules! inner_wrapping_pow {
    ($t:ty, $base:ident, $exp:ident) => {
        ($base.wrapping_pow($exp), false)
    };
}

macro_rules! inner_overflowing_pow {
    ($t:ty, $base:ident, $exp:ident) => {
        $base.overflowing_pow($exp)
    };
}

macro_rules! inner_overflowing_pow_value {
    ($t:ty, $base:ident, $exp:ident) => {
        ($base.overflowing_pow($exp).0, false)
    };
}

macro_rules! inner_overflowing_pow_overflow {
    ($t:ty, $base:ident, $exp:ident) => {
        (0 as $t, $base.overflowing_pow($exp).1)
    };
}

macro_rules! inner_checked_pow_option {
    ($t:ty, $base:ident, $exp:ident) => {
        match $base.checked_pow($exp) {
            Some(x) => (x, false),
            None => (0 as $t, true),
        }
    };
}

macro_rules! inner_checked_value {
    ($t:ty, $base:ident, $exp:ident) => {
        match $base.checked_pow($exp) {
            Some(x) => (x, false),
            None => (0 as $t, false),
        }
    };
}

macro_rules! inner_checked_pow_overflow {
    ($t:ty, $base:ident, $exp:ident) => {
        (0 as $t, $base.checked_pow($exp).ok())
    };
}

// This panics if it overflows.
macro_rules! inner_checked_pow_unwrapped {
    ($t:ty, $base:ident, $exp:ident) => {
        ($base.checked_pow($exp).unwrap(), false)
    };
}

// This has undefined behavior if it overflows.
// Consider running this bench along with an equivalent pow_unwrapped.
macro_rules! inner_checked_pow_unwrapped_unchecked {
    ($t:ty, $base:ident, $exp:ident) => {
        // SAFETY: Macro caller must ensure there is never an overflow.
        unsafe { ($base.checked_pow($exp).unwrap_unchecked(), false) }
    };
}

macro_rules! inner_saturating_pow {
    ($t:ty, $base:ident, $exp:ident) => {
        ($base.saturating_pow($exp), false)
    };
}

// ***

macro_rules! make_const_base {
    ($name:ident, $x:literal) => {
        macro_rules! $name {
            (1, $t:ty, $ident:ident) => {};
            (2, $t:ty, $ident:ident, $acc:ident) => {
                $x
            };
        }
    };
}

make_const_base!(base_const_2, 2);
make_const_base!(base_const_3, 3);

macro_rules! make_invariant_base {
    ($name:ident, $x:literal) => {
        macro_rules! $name {
            (1, $t:ty, $ident:ident) => {
                let $ident: $t = black_box($x);
            };
            (2, $t:ty, $ident:ident, $acc:ident) => {
                $ident
            };
        }
    };
}

macro_rules! make_repeated_base {
    ($name:ident, $x:literal) => {
        macro_rules! $name {
            (1, $t:ty, $ident:ident) => {
                let mut rng = crate::bench_rng();
                let mut exp_array = [$x as $t; ITERATIONS];
                let array_ref: &[$t; ITERATIONS] = black_box(&exp_array);
                let mut $ident = std::iter::repeat(array_ref.into_iter()).flatten();
            };
            (2, $t:ty, $ident:ident, $acc:ident) => {
                *$ident.next().unwrap()
            };
        }
    };
}

macro_rules! base_small_sequential {
    (1, $t:ty, $ident:ident) => {
        let mut rng = crate::bench_rng();
        let mut base_array = [0 as $t; ITERATIONS];
        #[allow(unused_comparisons)]
        const SIG_BITS: u32 = (<$t>::BITS - (<$t>::MIN < 0) as u32) / SMALL_EXPONENT_MAX;
        const BASE_MAX: $t = ((1 as $t) << SIG_BITS) - 1;
        const BASE_MIN: $t = (0 as $t).saturating_sub(BASE_MAX);
        for i in 0..base_array.len() {
            base_array[i] = rng.gen_range(BASE_MIN..=BASE_MAX);
        }
        let array_ref: &[$t; ITERATIONS] = black_box(&base_array);
        let mut $ident = std::iter::repeat(array_ref.into_iter()).flatten();
    };
    (2, $t:ty, $ident:ident, $acc:ident) => {
        *$ident.next().unwrap() ^ ($acc.0 & 1)
        // Worst case this changes -1 to -2 for i8
    };
}

macro_rules! base_exponential {
    (1, $t:ty, $ident:ident) => {
        let mut rng = crate::bench_rng();
        let mut base_array = [0 as $t; ITERATIONS];
        for i in 0..base_array.len() {
            base_array[i] = rng.gen::<$t>() >> rng.gen_range(0..<$t>::BITS);
        }
        let array_ref: &[$t; ITERATIONS] = black_box(&base_array);
        let mut $ident = std::iter::repeat(array_ref.into_iter()).flatten();
    };
    (2, $t:ty, $ident:ident, $acc:ident) => {
        *$ident.next().unwrap()
    };
}

macro_rules! make_const_exp {
    ($name:ident, $x:literal) => {
        macro_rules! $name {
            (1, $t:ty, $ident:ident) => {};
            (2, $t:ty, $ident:ident, $acc:ident) => {
                $x
            };
        }
    };
}

make_const_exp!(exp_const_6, 6);

macro_rules! make_invariant_exp {
    ($name:ident, $x:literal) => {
        macro_rules! $name {
            (1, $t:ty, $ident:ident) => {
                let $ident: u32 = black_box($x);
            };
            (2, $t:ty, $ident:ident, $acc:ident) => {
                $ident
            };
        }
    };
}

macro_rules! make_repeated_exp {
    ($name:ident, $x:literal) => {
        macro_rules! $name {
            (1, $t:ty, $ident:ident) => {
                let mut rng = crate::bench_rng();
                let mut exp_array = [$x as u32; ITERATIONS];
                let array_ref: &[u32; ITERATIONS] = black_box(&exp_array);
                let mut $ident = std::iter::repeat(array_ref.into_iter()).flatten();
            };
            (2, $t:ty, $ident:ident, $acc:ident) => {
                *$ident.next().unwrap()
            };
        }
    };
}

macro_rules! exp_small {
    (1, $t:ty, $ident:ident) => {
        let mut rng = crate::bench_rng();
        let mut exp_array = [0u32; ITERATIONS];
        for i in 0..exp_array.len() {
            exp_array[i] = rng.gen_range(0..=SMALL_EXPONENT_MAX);
        }
        let array_ref: &[u32; ITERATIONS] = black_box(&exp_array);
        let mut $ident = std::iter::repeat(array_ref.into_iter()).flatten();
    };
    (2, $t:ty, $ident:ident, $acc:ident) => {
        *$ident.next().unwrap()
    };
}

macro_rules! exp_exponential {
    (1, $t:ty, $ident:ident) => {
        let mut rng = crate::bench_rng();
        let mut exp_array = [0u32; ITERATIONS];
        for i in 0..exp_array.len() {
            exp_array[i] = rng.gen::<u32>() >> rng.gen_range(0..u32::BITS);
        }
        let array_ref: &[u32; ITERATIONS] = black_box(&exp_array);
        let mut $ident = std::iter::repeat(array_ref.into_iter()).flatten();
    };
    (2, $t:ty, $ident:ident, $acc:ident) => {
        *$ident.next().unwrap()
    };
}

macro_rules! exp_full {
    (1, $t:ty, $ident:ident) => {
        let mut rng = crate::bench_rng();
        let mut exp_array = [0u32; ITERATIONS];
        for i in 0..exp_array.len() {
            exp_array[i] = rng.gen() & (1 << 31);
        }
        let array_ref: &[u32; ITERATIONS] = black_box(&exp_array);
        let mut $ident = std::iter::repeat(array_ref.into_iter()).flatten();
    };
    (2, $t:ty, $ident:ident, $acc:ident) => {
        *$ident.next().unwrap()
    };
}

// ***

macro_rules! default_benches {
    (
        $t:ty,
        $ignored:meta,
        $pow_small_name:ident,
        $wrapping_small_name:ident,
        $overflowing_small_name:ident,
        $unwrapped_small_name:ident,
        $saturating_small_name:ident,
        $wrapping_exponential_name:ident,
        $overflowing_exponential_name:ident,
        $pow_const_2_name:ident,
        $overflowing_const_2_name:ident,
        $overflowing_const_3_name:ident,
        $pow_const_6_name:ident
    ) => {
        pow_bench_template!(
            $pow_small_name,
            $t,
            inner_pow,
            base_small_sequential,
            exp_small,
            $ignored
        );
        pow_bench_template!(
            $wrapping_small_name,
            $t,
            inner_wrapping_pow,
            base_small_sequential,
            exp_small,
            $ignored
        );
        pow_bench_template!(
            $overflowing_small_name,
            $t,
            inner_overflowing_pow,
            base_small_sequential,
            exp_small,
            $ignored
        );
        pow_bench_template!(
            $unwrapped_small_name,
            $t,
            inner_checked_pow_unwrapped,
            base_small_sequential,
            exp_small,
            $ignored
        );
        pow_bench_template!(
            $saturating_small_name,
            $t,
            inner_saturating_pow,
            base_small_sequential,
            exp_small,
            $ignored
        );

        pow_bench_template!(
            $wrapping_exponential_name,
            $t,
            inner_wrapping_pow,
            base_small_sequential,
            exp_exponential,
            $ignored
        );
        pow_bench_template!(
            $overflowing_exponential_name,
            $t,
            inner_overflowing_pow,
            base_small_sequential,
            exp_exponential,
            $ignored
        );

        pow_bench_template!(
            $pow_const_2_name,
            $t,
            inner_pow,
            base_const_2,
            exp_exponential,
            $ignored
        );
        pow_bench_template!(
            $overflowing_const_2_name,
            $t,
            inner_overflowing_pow,
            base_const_2,
            exp_exponential,
            $ignored
        );
        pow_bench_template!(
            $overflowing_const_3_name,
            $t,
            inner_overflowing_pow,
            base_const_2,
            exp_exponential,
            $ignored
        );

        pow_bench_template!(
            $pow_const_6_name,
            $t,
            inner_pow,
            base_small_sequential,
            exp_const_6,
            $ignored
        );
    };
}

default_benches!(
    u8,
    ignore,
    u8_pow_small,
    u8_wrapping_pow_small,
    u8_overflowing_pow_small,
    u8_checked_unwrapped_pow_small,
    u8_staurating_pow_small,
    u8_wrapping_pow_base_small_exp_exponential,
    u8_overflowing_pow_base_small_exp_exponential,
    u8_pow_base_const_2_exp_small,
    u8_overfowing_pow_base_const_2_exp_exponential,
    u8_overfowing_pow_base_const_3_exp_exponential,
    u8_pow_base_small_exp_const_6
);

default_benches!(
    i8,
    ignore,
    i8_pow_small,
    i8_wrapping_pow_small,
    i8_overflowing_pow_small,
    i8_checked_unwrapped_pow_small,
    i8_staurating_pow_small,
    i8_wrapping_pow_base_small_exp_exponential,
    i8_overflowing_pow_base_small_exp_exponential,
    i8_pow_base_const_2_exp_small,
    i8_overfowing_pow_base_const_2_exp_exponential,
    i8_overfowing_pow_base_const_3_exp_exponential,
    i8_pow_base_small_exp_const_6
);

default_benches!(
    u16,
    ignore,
    u16_pow_small,
    u16_wrapping_pow_small,
    u16_overflowing_pow_small,
    u16_checked_unwrapped_pow_small,
    u16_staurating_pow_small,
    u16_wrapping_pow_base_small_exp_exponential,
    u16_overflowing_pow_base_small_exp_exponential,
    u16_pow_base_const_2_exp_small,
    u16_overfowing_pow_base_const_2_exp_exponential,
    u16_overfowing_pow_base_const_3_exp_exponential,
    u16_pow_base_small_exp_const_6
);

default_benches!(
    i16,
    ignore,
    i16_pow_small,
    i16_wrapping_pow_small,
    i16_overflowing_pow_small,
    i16_checked_unwrapped_pow_small,
    i16_staurating_pow_small,
    i16_wrapping_pow_base_small_exp_exponential,
    i16_overflowing_pow_base_small_exp_exponential,
    i16_pow_base_const_2_exp_small,
    i16_overfowing_pow_base_const_2_exp_exponential,
    i16_overfowing_pow_base_const_3_exp_exponential,
    i16_pow_base_small_exp_const_6
);

default_benches!(
    u32,
    ignore,
    u32_pow_small,
    u32_wrapping_pow_small,
    u32_overflowing_pow_small,
    u32_checked_unwrapped_pow_small,
    u32_staurating_pow_small,
    u32_wrapping_pow_base_small_exp_exponential,
    u32_overflowing_pow_base_small_exp_exponential,
    u32_pow_base_const_2_exp_small,
    u32_overfowing_pow_base_const_2_exp_exponential,
    u32_overfowing_pow_base_const_3_exp_exponential,
    u32_pow_base_small_exp_const_6
);

default_benches!(
    i32,
    cfg(all()),
    i32_pow_small,
    i32_wrapping_pow_small,
    i32_overflowing_pow_small,
    i32_checked_unwrapped_pow_small,
    i32_staurating_pow_small,
    i32_wrapping_pow_base_small_exp_exponential,
    i32_overflowing_pow_base_small_exp_exponential,
    i32_pow_base_const_2_exp_small,
    i32_overfowing_pow_base_const_2_exp_exponential,
    i32_overfowing_pow_base_const_3_exp_exponential,
    i32_pow_base_small_exp_const_6
);

default_benches!(
    u64,
    cfg(all()),
    u64_pow_small,
    u64_wrapping_pow_small,
    u64_overflowing_pow_small,
    u64_checked_unwrapped_pow_small,
    u64_staurating_pow_small,
    u64_wrapping_pow_base_small_exp_exponential,
    u64_overflowing_pow_base_small_exp_exponential,
    u64_pow_base_const_2_exp_small,
    u64_overfowing_pow_base_const_2_exp_exponential,
    u64_overfowing_pow_base_const_3_exp_exponential,
    u64_pow_base_small_exp_const_6
);

default_benches!(
    i64,
    ignore,
    i64_pow_small,
    i64_wrapping_pow_small,
    i64_overflowing_pow_small,
    i64_checked_unwrapped_pow_small,
    i64_staurating_pow_small,
    i64_wrapping_pow_base_small_exp_exponential,
    i64_overflowing_pow_base_small_exp_exponential,
    i64_pow_base_const_2_exp_small,
    i64_overfowing_pow_base_const_2_exp_exponential,
    i64_overfowing_pow_base_const_3_exp_exponential,
    i64_pow_base_small_exp_const_6
);

default_benches!(
    u128,
    ignore,
    u128_pow_small,
    u128_wrapping_pow_small,
    u128_overflowing_pow_small,
    u128_checked_unwrapped_pow_small,
    u128_staurating_pow_small,
    u128_wrapping_pow_base_small_exp_exponential,
    u128_overflowing_pow_base_small_exp_exponential,
    u128_pow_base_const_2_exp_small,
    u128_overfowing_pow_base_const_2_exp_exponential,
    u128_overfowing_pow_base_const_3_exp_exponential,
    u128_pow_base_small_exp_const_6
);

default_benches!(
    i128,
    ignore,
    i128_pow_small,
    i128_wrapping_pow_small,
    i128_overflowing_pow_small,
    i128_checked_unwrapped_pow_small,
    i128_staurating_pow_small,
    i128_wrapping_pow_base_small_exp_exponential,
    i128_overflowing_pow_base_small_exp_exponential,
    i128_pow_base_const_2_exp_small,
    i128_overfowing_pow_base_const_2_exp_exponential,
    i128_overfowing_pow_base_const_3_exp_exponential,
    i128_pow_base_small_exp_const_6
);
