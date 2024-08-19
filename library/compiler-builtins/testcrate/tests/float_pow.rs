#![allow(unused_macros)]
#![cfg(not(all(target_arch = "x86", not(target_feature = "sse"))))]

use testcrate::*;

// This is approximate because of issues related to
// https://github.com/rust-lang/rust/issues/73920.
// TODO how do we resolve this indeterminacy?
macro_rules! pow {
    ($($f:ty, $tolerance:expr, $fn:ident);*;) => {
        $(
            #[test]
            fn $fn() {
                use compiler_builtins::float::pow::$fn;
                use compiler_builtins::float::Float;
                fuzz_float_2(N, |x: $f, y: $f| {
                    if !(Float::is_subnormal(x) || Float::is_subnormal(y) || x.is_nan()) {
                        let n = y.to_bits() & !<$f as Float>::SIGNIFICAND_MASK;
                        let n = (n as <$f as Float>::SignedInt) >> <$f as Float>::SIGNIFICAND_BITS;
                        let n = n as i32;
                        let tmp0: $f = x.powi(n);
                        let tmp1: $f = $fn(x, n);
                        let (a, b) = if tmp0 < tmp1 {
                            (tmp0, tmp1)
                        } else {
                            (tmp1, tmp0)
                        };

                        let good = if a == b {
                            // handles infinity equality
                            true
                        } else if a < $tolerance {
                            b < $tolerance
                        } else {
                            let quo = b / a;
                            (quo < (1. + $tolerance)) && (quo > (1. - $tolerance))
                        };

                        assert!(
                            good,
                            "{}({:?}, {:?}): std: {:?}, builtins: {:?}",
                            stringify!($fn), x, n, tmp0, tmp1
                        );
                    }
                });
            }
        )*
    };
}

pow! {
    f32, 1e-4, __powisf2;
    f64, 1e-12, __powidf2;
}
