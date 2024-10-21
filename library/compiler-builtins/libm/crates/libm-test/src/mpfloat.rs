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

use crate::Float;

/// Create a multiple-precision float with the correct number of bits for a concrete float type.
fn new_mpfloat<F: Float>() -> MpFloat {
    MpFloat::new(F::SIGNIFICAND_BITS + 1)
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
/// The struct itself should hold any context that can be reused among calls to `run` (allocated
/// `MpFloat`s).
pub trait MpOp {
    /// Inputs to the operation (concrete float types).
    type Input;

    /// Outputs from the operation (concrete float types).
    type Output;

    /// Create a new instance.
    fn new() -> Self;

    /// Perform the operation.
    ///
    /// Usually this means assigning inputs to cached floats, performing the operation, applying
    /// subnormal approximation, and converting the result back to concrete values.
    fn run(&mut self, input: Self::Input) -> Self::Output;
}

/// Implement `MpOp` for functions with a single return value.
macro_rules! impl_mp_op {
    // Matcher for unary functions
    (
        fn_name: $fn_name:ident,
        CFn: $CFn:ty,
        CArgs: $CArgs:ty,
        CRet: $CRet:ty,
        RustFn: fn($fty:ty,) -> $_ret:ty,
        RustArgs: $RustArgs:ty,
        RustRet: $RustRet:ty,
        fn_extra: $fn_name_normalized:expr,
    ) => {
        paste::paste! {
            pub mod $fn_name {
                use super::*;
                pub struct Operation(MpFloat);

                impl MpOp for Operation {
                    type Input = $RustArgs;
                    type Output = $RustRet;

                    fn new() -> Self {
                        Self(new_mpfloat::<$fty>())
                    }

                    fn run(&mut self, input: Self::Input) -> Self::Output {
                        self.0.assign(input.0);
                        let ord = self.0.[< $fn_name_normalized _round >](Nearest);
                        prep_retval::<Self::Output>(&mut self.0, ord)
                    }
                }
            }
        }
    };
    // Matcher for binary functions
    (
        fn_name: $fn_name:ident,
        CFn: $CFn:ty,
        CArgs: $CArgs:ty,
        CRet: $CRet:ty,
        RustFn: fn($fty:ty, $_fty2:ty,) -> $_ret:ty,
        RustArgs: $RustArgs:ty,
        RustRet: $RustRet:ty,
        fn_extra: $fn_name_normalized:expr,
    ) => {
        paste::paste! {
            pub mod $fn_name {
                use super::*;
                pub struct Operation(MpFloat, MpFloat);

                impl MpOp for Operation {
                    type Input = $RustArgs;
                    type Output = $RustRet;

                    fn new() -> Self {
                        Self(new_mpfloat::<$fty>(), new_mpfloat::<$fty>())
                    }

                    fn run(&mut self, input: Self::Input) -> Self::Output {
                        self.0.assign(input.0);
                        self.1.assign(input.1);
                        let ord = self.0.[< $fn_name_normalized _round >](&self.1, Nearest);
                        prep_retval::<Self::Output>(&mut self.0, ord)
                    }
                }
            }
        }
    };
    // Matcher for ternary functions
    (
        fn_name: $fn_name:ident,
        CFn: $CFn:ty,
        CArgs: $CArgs:ty,
        CRet: $CRet:ty,
        RustFn: fn($fty:ty, $_fty2:ty, $_fty3:ty,) -> $_ret:ty,
        RustArgs: $RustArgs:ty,
        RustRet: $RustRet:ty,
        fn_extra: $fn_name_normalized:expr,
    ) => {
        paste::paste! {
            pub mod $fn_name {
                use super::*;
                pub struct Operation(MpFloat, MpFloat, MpFloat);

                impl MpOp for Operation {
                    type Input = $RustArgs;
                    type Output = $RustRet;

                    fn new() -> Self {
                        Self(new_mpfloat::<$fty>(), new_mpfloat::<$fty>(), new_mpfloat::<$fty>())
                    }

                    fn run(&mut self, input: Self::Input) -> Self::Output {
                        self.0.assign(input.0);
                        self.1.assign(input.1);
                        self.2.assign(input.2);
                        let ord = self.0.[< $fn_name_normalized _round >](&self.1, &self.2, Nearest);
                        prep_retval::<Self::Output>(&mut self.0, ord)
                    }
                }
            }
        }
    };
}

libm_macros::for_each_function! {
    callback: impl_mp_op,
    skip: [
        // Most of these need a manual implementation
        fabs, ceil, copysign, floor, rint, round, trunc,
        fabsf, ceilf, copysignf, floorf, rintf, roundf, truncf,
        fmod, fmodf, frexp, frexpf, ilogb, ilogbf, jn, jnf, ldexp, ldexpf,
        lgamma_r, lgammaf_r, modf, modff, nextafter, nextafterf, pow,powf,
        remquo, remquof, scalbn, scalbnf, sincos, sincosf,
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
            $( impl_no_round!{ @inner_unary [< $fn_name f >], (f32,), $rug_name } )*
            $( impl_no_round!{ @inner_unary $fn_name, (f64,), $rug_name } )*
        }
    };

    (@inner_unary $fn_name:ident, ($fty:ty,), $rug_name:ident) => {
        pub mod $fn_name {
            use super::*;
            pub struct Operation(MpFloat);

            impl MpOp for Operation {
                type Input = ($fty,);
                type Output = $fty;

                fn new() -> Self {
                    Self(new_mpfloat::<$fty>())
                }

                fn run(&mut self, input: Self::Input) -> Self::Output {
                    self.0.assign(input.0);
                    self.0.$rug_name();
                    prep_retval::<Self::Output>(&mut self.0, Ordering::Equal)
                }
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
            pub mod [<copysign $suffix>] {
                use super::*;
                pub struct Operation(MpFloat, MpFloat);

                impl MpOp for Operation {
                    type Input = ($fty, $fty);
                    type Output = $fty;

                    fn new() -> Self {
                        Self(new_mpfloat::<$fty>(), new_mpfloat::<$fty>())
                    }

                    fn run(&mut self, input: Self::Input) -> Self::Output {
                        self.0.assign(input.0);
                        self.1.assign(input.1);
                        self.0.copysign_mut(&self.1);
                        prep_retval::<Self::Output>(&mut self.0, Ordering::Equal)
                    }
                }
            }

            pub mod [<pow $suffix>] {
                use super::*;
                pub struct Operation(MpFloat, MpFloat);

                impl MpOp for Operation {
                    type Input = ($fty, $fty);
                    type Output = $fty;

                    fn new() -> Self {
                        Self(new_mpfloat::<$fty>(), new_mpfloat::<$fty>())
                    }

                    fn run(&mut self, input: Self::Input) -> Self::Output {
                        self.0.assign(input.0);
                        self.1.assign(input.1);
                        let ord = self.0.pow_assign_round(&self.1, Nearest);
                        prep_retval::<Self::Output>(&mut self.0, ord)
                    }
                }
            }

            pub mod [<fmod $suffix>] {
                use super::*;
                pub struct Operation(MpFloat, MpFloat);

                impl MpOp for Operation {
                    type Input = ($fty, $fty);
                    type Output = $fty;

                    fn new() -> Self {
                        Self(new_mpfloat::<$fty>(), new_mpfloat::<$fty>())
                    }

                    fn run(&mut self, input: Self::Input) -> Self::Output {
                        self.0.assign(input.0);
                        self.1.assign(input.1);
                        let ord = self.0.rem_assign_round(&self.1, Nearest);
                        prep_retval::<Self::Output>(&mut self.0, ord)
                    }
                }
            }

            pub mod [<lgamma_r $suffix>] {
                use super::*;
                pub struct Operation(MpFloat);

                impl MpOp for Operation {
                    type Input = ($fty,);
                    type Output = ($fty, i32);

                    fn new() -> Self {
                        Self(new_mpfloat::<$fty>())
                    }

                    fn run(&mut self, input: Self::Input) -> Self::Output {
                        self.0.assign(input.0);
                        let (sign, ord) = self.0.ln_abs_gamma_round(Nearest);
                        let ret = prep_retval::<$fty>(&mut self.0, ord);
                        (ret, sign as i32)
                    }
                }
            }

            pub mod [<jn $suffix>] {
                use super::*;
                pub struct Operation(i32, MpFloat);

                impl MpOp for Operation {
                    type Input = (i32, $fty);
                    type Output = $fty;

                    fn new() -> Self {
                        Self(0, new_mpfloat::<$fty>())
                    }

                    fn run(&mut self, input: Self::Input) -> Self::Output {
                        self.0 = input.0;
                        self.1.assign(input.1);
                        let ord = self.1.jn_round(self.0, Nearest);
                        prep_retval::<$fty>(&mut self.1, ord)
                    }
                }
            }

            pub mod [<sincos $suffix>] {
                use super::*;
                pub struct Operation(MpFloat, MpFloat);

                impl MpOp for Operation {
                    type Input = ($fty,);
                    type Output = ($fty, $fty);

                    fn new() -> Self {
                        Self(new_mpfloat::<$fty>(), new_mpfloat::<$fty>())
                    }

                    fn run(&mut self, input: Self::Input) -> Self::Output {
                        self.0.assign(input.0);
                        self.1.assign(0.0);
                        let (sord, cord) = self.0.sin_cos_round(&mut self.1, Nearest);
                        (
                            prep_retval::<$fty>(&mut self.0, sord),
                            prep_retval::<$fty>(&mut self.1, cord)
                        )
                    }
                }
            }
        }
    };
}

impl_op_for_ty!(f32, "f");
impl_op_for_ty!(f64, "");

// Account for `lgamma_r` not having a simple `f` suffix
pub mod lgammaf_r {
    pub use super::lgamma_rf::*;
}
