//! Test with "infinite precision"

#![cfg(feature = "test-multiprecision")]

use libm_test::gen::{CachedInput, random};
use libm_test::mpfloat::MpOp;
use libm_test::{CheckBasis, CheckCtx, CheckOutput, GenerateInput, MathOp, TupleCall};

/// Implement a test against MPFR with random inputs.
macro_rules! mp_rand_tests {
    (
        fn_name: $fn_name:ident,
        attrs: [$($meta:meta)*]
    ) => {
        paste::paste! {
            #[test]
            $(#[$meta])*
            fn [< mp_random_ $fn_name >]() {
                test_one::<libm_test::op::$fn_name::Routine>();
            }
        }
    };
}

fn test_one<Op>()
where
    Op: MathOp + MpOp,
    CachedInput: GenerateInput<Op::RustArgs>,
{
    let mut mp_vals = Op::new_mp();
    let ctx = CheckCtx::new(Op::IDENTIFIER, CheckBasis::Mpfr);
    let cases = random::get_test_cases::<Op::RustArgs>(&ctx);

    for input in cases {
        let mp_res = Op::run(&mut mp_vals, input);
        let crate_res = input.call(Op::ROUTINE);

        crate_res.validate(mp_res, input, &ctx).unwrap();
    }
}

libm_macros::for_each_function! {
    callback: mp_rand_tests,
    attributes: [
        // Also an assertion failure on i686: at `MPFR_ASSERTN (! mpfr_erangeflag_p ())`
        #[ignore = "large values are infeasible in MPFR"]
        [jn, jnf],
    ],
    skip: [
        // FIXME: MPFR tests needed
        frexp,
        frexpf,
        ilogb,
        ilogbf,
        ldexp,
        ldexpf,
        modf,
        modff,
        remquo,
        remquof,
        scalbn,
        scalbnf,

        // FIXME: test needed, see
        // https://github.com/rust-lang/libm/pull/311#discussion_r1818273392
        nextafter,
        nextafterf,
    ],
}
