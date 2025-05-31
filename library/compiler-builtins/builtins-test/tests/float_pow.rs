#![allow(unused_macros)]
#![cfg_attr(f128_enabled, feature(f128))]
#![cfg(not(all(target_arch = "x86", not(target_feature = "sse"))))]

use builtins_test::*;

// This is approximate because of issues related to
// https://github.com/rust-lang/rust/issues/73920.
// TODO how do we resolve this indeterminacy?
macro_rules! pow {
    ($($f:ty, $tolerance:expr, $fn:ident, $sys_available:meta);*;) => {
        $(
            #[test]
            // FIXME(apfloat): We skip tests if system symbols aren't available rather
            // than providing a fallback, since `rustc_apfloat` does not provide `pow`.
            #[cfg($sys_available)]
            fn $fn() {
                use compiler_builtins::float::pow::$fn;
                use compiler_builtins::float::Float;
                fuzz_float_2(N, |x: $f, y: $f| {
                    if !(Float::is_subnormal(x) || Float::is_subnormal(y) || x.is_nan()) {
                        let n = y.to_bits() & !<$f as Float>::SIG_MASK;
                        let n = (n as <$f as Float>::SignedInt) >> <$f as Float>::SIG_BITS;
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
    f32, 1e-4, __powisf2, all();
    f64, 1e-12, __powidf2, all();
}

#[cfg(f128_enabled)]
// FIXME(f16_f128): MSVC cannot build these until `__divtf3` is available in nightly.
#[cfg(not(target_env = "msvc"))]
#[cfg(not(any(target_arch = "powerpc", target_arch = "powerpc64")))]
pow! {
    f128, 1e-36, __powitf2, not(feature = "no-sys-f128");
}

#[cfg(f128_enabled)]
#[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
pow! {
    f128, 1e-36, __powikf2, not(feature = "no-sys-f128");
}
