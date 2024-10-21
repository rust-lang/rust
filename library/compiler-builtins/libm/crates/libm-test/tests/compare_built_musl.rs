//! Compare our implementations with the result of musl functions, as provided by `musl-math-sys`.
//!
//! Currently this only tests randomized inputs. In the future this may be improved to test edge
//! cases or run exhaustive tests.
//!
//! Note that musl functions do not always provide 0.5ULP rounding, so our functions can do better
//! than these results.

// There are some targets we can't build musl for
#![cfg(feature = "build-musl")]

use libm_test::gen::random;
use libm_test::{CheckBasis, CheckCtx, CheckOutput, TupleCall, musl_allowed_ulp};
use musl_math_sys as musl;

macro_rules! musl_rand_tests {
    (
        fn_name: $fn_name:ident,
        CFn: $CFn:ty,
        CArgs: $CArgs:ty,
        CRet: $CRet:ty,
        RustFn: $RustFn:ty,
        RustArgs: $RustArgs:ty,
        RustRet: $RustRet:ty,
        attrs: [$($meta:meta)*]
    ) => { paste::paste! {
        #[test]
        $(#[$meta])*
        fn [< musl_random_ $fn_name >]() {
            let fname = stringify!($fn_name);
            let ulp = musl_allowed_ulp(fname);
            let ctx = CheckCtx::new(ulp, fname, CheckBasis::Musl);
            let cases = random::get_test_cases::<$RustArgs>(&ctx);

            for input in cases {
                let musl_res = input.call(musl::$fn_name as $CFn);
                let crate_res = input.call(libm::$fn_name as $RustFn);

                crate_res.validate(musl_res, input, &ctx).unwrap();
            }
        }
    } };
}

libm_macros::for_each_function! {
    callback: musl_rand_tests,
    skip: [],
    attributes: [
        #[cfg_attr(x86_no_sse, ignore)] // FIXME(correctness): wrong result on i586
        [exp10, exp10f, exp2, exp2f, rint]
    ],
}
