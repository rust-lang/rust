//! Test with "infinite precision"

#![cfg(feature = "build-mpfr")]

use libm_test::gen::{edge_cases, random, spaced};
use libm_test::mpfloat::MpOp;
use libm_test::{CheckBasis, CheckCtx, CheckOutput, GeneratorKind, MathOp, TupleCall};

const BASIS: CheckBasis = CheckBasis::Mpfr;

fn mp_runner<Op: MathOp + MpOp>(ctx: &CheckCtx, cases: impl Iterator<Item = Op::RustArgs>) {
    let mut mp_vals = Op::new_mp();
    for input in cases {
        let mp_res = Op::run(&mut mp_vals, input);
        let crate_res = input.call(Op::ROUTINE);

        crate_res.validate(mp_res, input, ctx).unwrap();
    }
}

macro_rules! mp_tests {
    (
        fn_name: $fn_name:ident,
        attrs: [$($attr:meta),*],
    ) => {
        paste::paste! {
            #[test]
            $(#[$attr])*
            fn [< mp_random_ $fn_name >]() {
                type Op = libm_test::op::$fn_name::Routine;
                let ctx = CheckCtx::new(Op::IDENTIFIER, BASIS, GeneratorKind::Random);
                let cases = random::get_test_cases::<<Op as MathOp>::RustArgs>(&ctx);
                mp_runner::<Op>(&ctx, cases);
            }

            #[test]
            $(#[$attr])*
            fn [< mp_edge_case_ $fn_name >]() {
                type Op = libm_test::op::$fn_name::Routine;
                let ctx = CheckCtx::new(Op::IDENTIFIER, BASIS, GeneratorKind::EdgeCases);
                let cases = edge_cases::get_test_cases::<Op>(&ctx);
                mp_runner::<Op>(&ctx, cases);
            }

            #[test]
            $(#[$attr])*
            fn [< mp_quickspace_ $fn_name >]() {
                type Op = libm_test::op::$fn_name::Routine;
                let ctx = CheckCtx::new(Op::IDENTIFIER, BASIS, GeneratorKind::QuickSpaced);
                let cases = spaced::get_test_cases::<Op>(&ctx).0;
                mp_runner::<Op>(&ctx, cases);
            }
        }
    };
}

libm_macros::for_each_function! {
    callback: mp_tests,
    attributes: [
        // Also an assertion failure on i686: at `MPFR_ASSERTN (! mpfr_erangeflag_p ())`
        #[ignore = "large values are infeasible in MPFR"]
        [jn, jnf, yn, ynf],
    ],
    skip: [
        // FIXME: test needed, see
        // https://github.com/rust-lang/libm/pull/311#discussion_r1818273392
        nextafter,
        nextafterf,
    ],
}
