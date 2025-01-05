//! Test with "infinite precision"

#![cfg(feature = "test-multiprecision")]

use libm_test::domain::HasDomain;
use libm_test::gen::random::RandomInput;
use libm_test::gen::{domain_logspace, edge_cases, random};
use libm_test::mpfloat::MpOp;
use libm_test::{CheckBasis, CheckCtx, CheckOutput, MathOp, OpFTy, OpRustFn, OpRustRet, TupleCall};

/// Test against MPFR with random inputs.
macro_rules! mp_rand_tests {
    (
        fn_name: $fn_name:ident,
        attrs: [$($attr:meta),*],
    ) => {
        paste::paste! {
            #[test]
            $(#[$attr])*
            fn [< mp_random_ $fn_name >]() {
                test_one_random::<libm_test::op::$fn_name::Routine>();
            }
        }
    };
}

/// Test a single routine with random inputs
fn test_one_random<Op>()
where
    Op: MathOp + MpOp,
    Op::RustArgs: RandomInput,
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
        [jn, jnf, yn, ynf],
    ],
    skip: [
        // FIXME: MPFR tests needed
        remquo,
        remquof,

        // FIXME: test needed, see
        // https://github.com/rust-lang/libm/pull/311#discussion_r1818273392
        nextafter,
        nextafterf,
    ],
}

/// Test against MPFR with generators from a domain.
macro_rules! mp_domain_tests {
    (
        fn_name: $fn_name:ident,
        attrs: [$($attr:meta),*],
    ) => {
        paste::paste! {
            #[test]
            $(#[$attr])*
            fn [< mp_edge_case_ $fn_name >]() {
                type Op = libm_test::op::$fn_name::Routine;
                domain_test_runner::<Op, _>(edge_cases::get_test_cases::<Op, _>);
            }

            #[test]
            $(#[$attr])*
            fn [< mp_logspace_ $fn_name >]() {
                type Op = libm_test::op::$fn_name::Routine;
                domain_test_runner::<Op, _>(domain_logspace::get_test_cases::<Op>);
            }
        }
    };
}

/// Test a single routine against domaine-aware inputs.
fn domain_test_runner<Op, I>(gen: impl FnOnce(&CheckCtx) -> I)
where
    // Complicated generics...
    // The operation must take a single float argument (unary only)
    Op: MathOp<RustArgs = (<Op as MathOp>::FTy,)>,
    // It must also support multiprecision operations
    Op: MpOp,
    // And it must have a domain specified
    Op: HasDomain<Op::FTy>,
    // The single float argument tuple must be able to call the `RustFn` and return `RustRet`
    (OpFTy<Op>,): TupleCall<OpRustFn<Op>, Output = OpRustRet<Op>>,
    I: Iterator<Item = (Op::FTy,)>,
{
    let mut mp_vals = Op::new_mp();
    let ctx = CheckCtx::new(Op::IDENTIFIER, CheckBasis::Mpfr);
    let cases = gen(&ctx);

    for input in cases {
        let mp_res = Op::run(&mut mp_vals, input);
        let crate_res = input.call(Op::ROUTINE);

        crate_res.validate(mp_res, input, &ctx).unwrap();
    }
}

libm_macros::for_each_function! {
    callback: mp_domain_tests,
    attributes: [],
    skip: [
        // Functions with multiple inputs
        atan2,
        atan2f,
        copysign,
        copysignf,
        fdim,
        fdimf,
        fma,
        fmaf,
        fmax,
        fmaxf,
        fmin,
        fminf,
        fmod,
        fmodf,
        hypot,
        hypotf,
        jn,
        jnf,
        ldexp,
        ldexpf,
        nextafter,
        nextafterf,
        pow,
        powf,
        remainder,
        remainderf,
        remquo,
        remquof,
        scalbn,
        scalbnf,
        yn,
        ynf,
    ],
}
