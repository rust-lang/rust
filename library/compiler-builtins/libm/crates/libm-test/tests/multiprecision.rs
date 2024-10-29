//! Test with "infinite precision"

#![cfg(feature = "test-multiprecision")]

use libm_test::gen::random;
use libm_test::mpfloat::{self, MpOp};
use libm_test::{CheckBasis, CheckCtx, CheckOutput, TupleCall, multiprec_allowed_ulp};

/// Implement a test against MPFR with random inputs.
macro_rules! multiprec_rand_tests {
    (
        fn_name: $fn_name:ident,
        CFn: $CFn:ty,
        CArgs: $CArgs:ty,
        CRet: $CRet:ty,
        RustFn: $RustFn:ty,
        RustArgs: $RustArgs:ty,
        RustRet: $RustRet:ty,
        attrs: [$($meta:meta)*]
    ) => {
        paste::paste! {
            #[test]
            $(#[$meta])*
            fn [< multiprec_random_ $fn_name >]() {
                type MpOpTy = mpfloat::$fn_name::Operation;

                let fname = stringify!($fn_name);
                let ulp = multiprec_allowed_ulp(fname);
                let mut mp_vals = MpOpTy::new();
                let ctx = CheckCtx::new(ulp, fname, CheckBasis::Mpfr);
                let cases = random::get_test_cases::<$RustArgs>(&ctx);

                for input in cases {
                    let mp_res = mp_vals.run(input);
                    let crate_res = input.call(libm::$fn_name as $RustFn);

                    crate_res.validate(mp_res, input, &ctx).unwrap();
                }
            }
        }
    };
}

libm_macros::for_each_function! {
    callback: multiprec_rand_tests,
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
