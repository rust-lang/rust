//! Test cases that have both an input and an output, so do not require a basis.

use libm_test::generate::case_list;
use libm_test::{CheckBasis, CheckCtx, CheckOutput, GeneratorKind, MathOp, TupleCall};

const BASIS: CheckBasis = CheckBasis::None;

fn standalone_runner<Op: MathOp>(
    ctx: &CheckCtx,
    cases: impl Iterator<Item = (Op::RustArgs, Op::RustRet)>,
) {
    for (input, expected) in cases {
        let crate_res = input.call_intercept_panics(Op::ROUTINE);
        crate_res.validate(expected, input, ctx).unwrap();
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
            fn [< standalone_ $fn_name >]() {
                type Op = libm_test::op::$fn_name::Routine;
                let ctx = CheckCtx::new(Op::IDENTIFIER, BASIS, GeneratorKind::List);
                let cases = case_list::get_test_cases_standalone::<Op>(&ctx);
                standalone_runner::<Op>(&ctx, cases);
            }
        }
    };
}

libm_macros::for_each_function! {
    callback: mp_tests,
}
