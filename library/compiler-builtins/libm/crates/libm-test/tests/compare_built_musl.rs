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
use libm_test::gen::random::RandomInput;
use libm_test::{CheckBasis, CheckCtx, CheckOutput, MathOp, TupleCall};

macro_rules! musl_rand_tests {
    (
        fn_name: $fn_name:ident,
        attrs: [$($attr:meta),*],
    ) => {
        paste::paste! {
            #[test]
            $(#[$attr])*
            fn [< musl_random_ $fn_name >]() {
                test_one_random::<libm_test::op::$fn_name::Routine>(musl_math_sys::$fn_name);
            }
        }
    };
}

fn test_one_random<Op>(musl_fn: Op::CFn)
where
    Op: MathOp,
    Op::RustArgs: RandomInput,
{
    let ctx = CheckCtx::new(Op::IDENTIFIER, CheckBasis::Musl);
    let cases = random::get_test_cases::<Op::RustArgs>(&ctx);

    for input in cases {
        let musl_res = input.call(musl_fn);
        let crate_res = input.call(Op::ROUTINE);

        crate_res.validate(musl_res, input, &ctx).unwrap();
    }
}

libm_macros::for_each_function! {
    callback: musl_rand_tests,
    // Musl does not support `f16` and `f128` on all platforms.
    skip: [copysignf16, copysignf128, fabsf16, fabsf128],
    attributes: [
        #[cfg_attr(x86_no_sse, ignore)] // FIXME(correctness): wrong result on i586
        [exp10, exp10f, exp2, exp2f, rint]
    ],
}
