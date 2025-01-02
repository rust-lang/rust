//! Interfaces needed to support testing with multi-precision floating point numbers.
//!
//! Within this module, the macros create a submodule for each `libm` function. These contain
//! a struct named `Operation` that implements [`MpOp`].

use std::cmp::Ordering;

use az::Az;
use rug::Assign;
pub use rug::Float as MpFloat;
use rug::float::Round::Nearest;
use rug::ops::{PowAssignRound, RemAssignRound};

use crate::{Float, MathOp};

/// Create a multiple-precision float with the correct number of bits for a concrete float type.
fn new_mpfloat<F: Float>() -> MpFloat {
    MpFloat::new(F::SIG_BITS + 1)
}

/// Set subnormal emulation and convert to a concrete float type.
fn prep_retval<F: Float>(mp: &mut MpFloat, ord: Ordering) -> F
where
    for<'a> &'a MpFloat: az::Cast<F>,
{
    mp.subnormalize_ieee_round(ord, Nearest);
    (&*mp).az::<F>()
}

/// Structures that represent a float operation.
///
pub trait MpOp: MathOp {
    /// The struct itself should hold any context that can be reused among calls to `run` (allocated
    /// `MpFloat`s).
    type MpTy;

    /// Create a new instance.
    fn new_mp() -> Self::MpTy;

    /// Perform the operation.
    ///
    /// Usually this means assigning inputs to cached floats, performing the operation, applying
    /// subnormal approximation, and converting the result back to concrete values.
    fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet;
}

/// Implement `MpOp` for functions with a single return value.
macro_rules! impl_mp_op {
    // Matcher for unary functions
    (
        fn_name: $fn_name:ident,
        RustFn: fn($_fty:ty,) -> $_ret:ty,
        attrs: [$($attr:meta),*],
        fn_extra: $fn_name_normalized:expr,
    ) => {
        paste::paste! {
            $(#[$attr])*
            impl MpOp for crate::op::$fn_name::Routine {
                type MpTy = MpFloat;

                fn new_mp() -> Self::MpTy {
                    new_mpfloat::<Self::FTy>()
                }

                fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
                    this.assign(input.0);
                    let ord = this.[< $fn_name_normalized _round >](Nearest);
                    prep_retval::<Self::RustRet>(this, ord)
                }
            }
        }
    };
    // Matcher for binary functions
    (
        fn_name: $fn_name:ident,
        RustFn: fn($_fty:ty, $_fty2:ty,) -> $_ret:ty,
        attrs: [$($attr:meta),*],
        fn_extra: $fn_name_normalized:expr,
    ) => {
        paste::paste! {
            $(#[$attr])*
            impl MpOp for crate::op::$fn_name::Routine {
                type MpTy = (MpFloat, MpFloat);

                fn new_mp() -> Self::MpTy {
                    (new_mpfloat::<Self::FTy>(), new_mpfloat::<Self::FTy>())
                }

                fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
                    this.0.assign(input.0);
                    this.1.assign(input.1);
                    let ord = this.0.[< $fn_name_normalized _round >](&this.1, Nearest);
                    prep_retval::<Self::RustRet>(&mut this.0, ord)
                }
            }
        }
    };
    // Matcher for ternary functions
    (
        fn_name: $fn_name:ident,
        RustFn: fn($_fty:ty, $_fty2:ty, $_fty3:ty,) -> $_ret:ty,
        attrs: [$($attr:meta),*],
        fn_extra: $fn_name_normalized:expr,
    ) => {
        paste::paste! {
            $(#[$attr])*
            impl MpOp for crate::op::$fn_name::Routine {
                type MpTy = (MpFloat, MpFloat, MpFloat);

                fn new_mp() -> Self::MpTy {
                    (
                        new_mpfloat::<Self::FTy>(),
                        new_mpfloat::<Self::FTy>(),
                        new_mpfloat::<Self::FTy>(),
                    )
                }

                fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
                    this.0.assign(input.0);
                    this.1.assign(input.1);
                    this.2.assign(input.2);
                    let ord = this.0.[< $fn_name_normalized _round >](&this.1, &this.2, Nearest);
                    prep_retval::<Self::RustRet>(&mut this.0, ord)
                }
            }
        }
    };
}

libm_macros::for_each_function! {
    callback: impl_mp_op,
    emit_types: [RustFn],
    skip: [
        // Most of these need a manual implementation
        fabs, ceil, copysign, floor, rint, round, trunc,
        fabsf, ceilf, copysignf, floorf, rintf, roundf, truncf,
        fmod, fmodf, frexp, frexpf, ilogb, ilogbf, jn, jnf, ldexp, ldexpf,
        lgamma_r, lgammaf_r, modf, modff, nextafter, nextafterf, pow,powf,
        remquo, remquof, scalbn, scalbnf, sincos, sincosf, yn, ynf,
    ],
    fn_extra: match MACRO_FN_NAME {
        // Remap function names that are different between mpfr and libm
        expm1 | expm1f => exp_m1,
        fabs | fabsf => abs,
        fdim | fdimf => positive_diff,
        fma | fmaf => mul_add,
        fmax | fmaxf => max,
        fmin | fminf => min,
        lgamma | lgammaf => ln_gamma,
        log | logf => ln,
        log1p | log1pf => ln_1p,
        tgamma | tgammaf => gamma,
        _ => MACRO_FN_NAME_NORMALIZED
    }
}

/// Implement unary functions that don't have a `_round` version
macro_rules! impl_no_round {
    // Unary matcher
    ($($fn_name:ident, $rug_name:ident;)*) => {
        paste::paste! {
            // Implement for both f32 and f64
            $( impl_no_round!{ @inner_unary [< $fn_name f >], $rug_name } )*
            $( impl_no_round!{ @inner_unary $fn_name, $rug_name } )*
        }
    };

    (@inner_unary $fn_name:ident, $rug_name:ident) => {
        impl MpOp for crate::op::$fn_name::Routine {
            type MpTy = MpFloat;

            fn new_mp() -> Self::MpTy {
                new_mpfloat::<Self::FTy>()
            }

            fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
                this.assign(input.0);
                this.$rug_name();
                prep_retval::<Self::RustRet>(this, Ordering::Equal)
            }
        }
    };
}

impl_no_round! {
    fabs, abs_mut;
    ceil, ceil_mut;
    floor, floor_mut;
    rint, round_even_mut; // FIXME: respect rounding mode
    round, round_mut;
    trunc, trunc_mut;
}

/// Some functions are difficult to do in a generic way. Implement them here.
macro_rules! impl_op_for_ty {
    ($fty:ty, $suffix:literal) => {
        paste::paste! {
            impl MpOp for crate::op::[<copysign $suffix>]::Routine {
                type MpTy = (MpFloat, MpFloat);

                fn new_mp() -> Self::MpTy {
                    (new_mpfloat::<Self::FTy>(), new_mpfloat::<Self::FTy>())
                }

                fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
                    this.0.assign(input.0);
                    this.1.assign(input.1);
                    this.0.copysign_mut(&this.1);
                    prep_retval::<Self::RustRet>(&mut this.0, Ordering::Equal)
                }
            }

            impl MpOp for crate::op::[<pow $suffix>]::Routine {
                type MpTy = (MpFloat, MpFloat);

                fn new_mp() -> Self::MpTy {
                    (new_mpfloat::<Self::FTy>(), new_mpfloat::<Self::FTy>())
                }

                fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
                    this.0.assign(input.0);
                    this.1.assign(input.1);
                    let ord = this.0.pow_assign_round(&this.1, Nearest);
                    prep_retval::<Self::RustRet>(&mut this.0, ord)
                }
            }

            impl MpOp for crate::op::[<fmod $suffix>]::Routine {
                type MpTy = (MpFloat, MpFloat);

                fn new_mp() -> Self::MpTy {
                    (new_mpfloat::<Self::FTy>(), new_mpfloat::<Self::FTy>())
                }

                fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
                    this.0.assign(input.0);
                    this.1.assign(input.1);
                    let ord = this.0.rem_assign_round(&this.1, Nearest);
                    prep_retval::<Self::RustRet>(&mut this.0, ord)
                }
            }

            impl MpOp for crate::op::[<jn $suffix>]::Routine {
                type MpTy = (i32, MpFloat);

                fn new_mp() -> Self::MpTy {
                    (0, new_mpfloat::<Self::FTy>())
                }

                fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
                    this.0 = input.0;
                    this.1.assign(input.1);
                    let ord = this.1.jn_round(this.0, Nearest);
                    prep_retval::<Self::FTy>(&mut this.1, ord)
                }
            }

            impl MpOp for crate::op::[<sincos $suffix>]::Routine {
                type MpTy = (MpFloat, MpFloat);

                fn new_mp() -> Self::MpTy {
                    (new_mpfloat::<Self::FTy>(), new_mpfloat::<Self::FTy>())
                }

                fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
                    this.0.assign(input.0);
                    this.1.assign(0.0);
                    let (sord, cord) = this.0.sin_cos_round(&mut this.1, Nearest);
                    (
                        prep_retval::<Self::FTy>(&mut this.0, sord),
                        prep_retval::<Self::FTy>(&mut this.1, cord)
                    )
                }
            }

            impl MpOp for crate::op::[<yn $suffix>]::Routine {
                type MpTy = (i32, MpFloat);

                fn new_mp() -> Self::MpTy {
                    (0, new_mpfloat::<Self::FTy>())
                }

                fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
                    this.0 = input.0;
                    this.1.assign(input.1);
                    let ord = this.1.yn_round(this.0, Nearest);
                    prep_retval::<Self::FTy>(&mut this.1, ord)
                }
            }
        }
    };
}

impl_op_for_ty!(f32, "f");
impl_op_for_ty!(f64, "");

// `lgamma_r` is not a simple suffix so we can't use the above macro.
impl MpOp for crate::op::lgamma_r::Routine {
    type MpTy = MpFloat;

    fn new_mp() -> Self::MpTy {
        new_mpfloat::<Self::FTy>()
    }

    fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
        this.assign(input.0);
        let (sign, ord) = this.ln_abs_gamma_round(Nearest);
        let ret = prep_retval::<Self::FTy>(this, ord);
        (ret, sign as i32)
    }
}

impl MpOp for crate::op::lgammaf_r::Routine {
    type MpTy = MpFloat;

    fn new_mp() -> Self::MpTy {
        new_mpfloat::<Self::FTy>()
    }

    fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
        this.assign(input.0);
        let (sign, ord) = this.ln_abs_gamma_round(Nearest);
        let ret = prep_retval::<Self::FTy>(this, ord);
        (ret, sign as i32)
    }
}
