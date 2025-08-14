//! Interfaces needed to support testing with multi-precision floating point numbers.
//!
//! Within this module, the macros create a submodule for each `libm` function. These contain
//! a struct named `Operation` that implements [`MpOp`].

use std::cmp::Ordering;

use rug::Assign;
pub use rug::Float as MpFloat;
use rug::az::{self, Az};
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
        // verify-sorted-start
        ceil,
        ceilf,
        ceilf128,
        ceilf16,
        copysign,
        copysignf,
        copysignf128,
        copysignf16,
        fabs,
        fabsf,
        fabsf128,
        fabsf16,floor,
        floorf,
        floorf128,
        floorf16,
        fmaximum,
        fmaximumf,
        fmaximumf128,
        fmaximumf16,
        fminimum,
        fminimumf,
        fminimumf128,
        fminimumf16,
        fmod,
        fmodf,
        fmodf128,
        fmodf16,
        frexp,
        frexpf,
        ilogb,
        ilogbf,
        jn,
        jnf,
        ldexp,
        ldexpf,
        ldexpf128,
        ldexpf16,
        lgamma_r,
        lgammaf_r,
        modf,
        modff,
        nextafter,
        nextafterf,
        pow,
        powf,remquo,
        remquof,
        rint,
        rintf,
        rintf128,
        rintf16,
        round,
        roundeven,
        roundevenf,
        roundevenf128,
        roundevenf16,
        roundf,
        roundf128,
        roundf16,
        scalbn,
        scalbnf,
        scalbnf128,
        scalbnf16,
        sincos,sincosf,
        trunc,
        truncf,
        truncf128,
        truncf16,yn,
        ynf,
        // verify-sorted-end
    ],
    fn_extra: match MACRO_FN_NAME {
        // Remap function names that are different between mpfr and libm
        expm1 | expm1f => exp_m1,
        fabs | fabsf => abs,
        fdim | fdimf | fdimf16 | fdimf128  => positive_diff,
        fma | fmaf | fmaf128 => mul_add,
        fmax | fmaxf | fmaxf16 | fmaxf128 |
        fmaximum_num | fmaximum_numf | fmaximum_numf16 | fmaximum_numf128 => max,
        fmin | fminf | fminf16 | fminf128 |
        fminimum_num | fminimum_numf | fminimum_numf16 | fminimum_numf128 => min,
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
    ($($fn_name:ident => $rug_name:ident;)*) => {
        paste::paste! {
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
    ceil => ceil_mut;
    ceilf => ceil_mut;
    fabs => abs_mut;
    fabsf => abs_mut;
    floor => floor_mut;
    floorf => floor_mut;
    rint => round_even_mut; // FIXME: respect rounding mode
    rintf => round_even_mut; // FIXME: respect rounding mode
    round => round_mut;
    roundeven => round_even_mut;
    roundevenf => round_even_mut;
    roundf => round_mut;
    trunc => trunc_mut;
    truncf => trunc_mut;
}

#[cfg(f16_enabled)]
impl_no_round! {
    ceilf16 => ceil_mut;
    fabsf16 => abs_mut;
    floorf16 => floor_mut;
    rintf16 => round_even_mut; // FIXME: respect rounding mode
    roundf16 => round_mut;
    roundevenf16 => round_even_mut;
    truncf16 => trunc_mut;
}

#[cfg(f128_enabled)]
impl_no_round! {
    ceilf128 => ceil_mut;
    fabsf128 => abs_mut;
    floorf128 => floor_mut;
    rintf128 => round_even_mut; // FIXME: respect rounding mode
    roundf128 => round_mut;
    roundevenf128 => round_even_mut;
    truncf128 => trunc_mut;
}

/// Some functions are difficult to do in a generic way. Implement them here.
macro_rules! impl_op_for_ty {
    ($fty:ty, $suffix:literal) => {
        paste::paste! {
            impl MpOp for crate::op::[<modf $suffix>]::Routine {
                type MpTy = (MpFloat, MpFloat);

                fn new_mp() -> Self::MpTy {
                    (new_mpfloat::<Self::FTy>(), new_mpfloat::<Self::FTy>())
                }

                fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
                    this.0.assign(input.0);
                    this.1.assign(&this.0);
                    let (ord0, ord1) = this.0.trunc_fract_round(&mut this.1, Nearest);
                    (
                        prep_retval::<Self::FTy>(&mut this.1, ord0),
                        prep_retval::<Self::FTy>(&mut this.0, ord1),
                    )
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

            impl MpOp for crate::op::[<frexp $suffix>]::Routine {
                type MpTy = MpFloat;

                fn new_mp() -> Self::MpTy {
                    new_mpfloat::<Self::FTy>()
                }

                fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
                    this.assign(input.0);
                    let exp = this.frexp_mut();
                    (prep_retval::<Self::FTy>(this, Ordering::Equal), exp)
                }
            }

            impl MpOp for crate::op::[<ilogb $suffix>]::Routine {
                type MpTy = MpFloat;

                fn new_mp() -> Self::MpTy {
                    new_mpfloat::<Self::FTy>()
                }

                fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
                    this.assign(input.0);

                    // `get_exp` follows `frexp` for `0.5 <= |m| < 1.0`. Adjust the exponent by
                    // one to scale the significand to `1.0 <= |m| < 2.0`.
                    this.get_exp().map(|v| v - 1).unwrap_or_else(|| {
                        if this.is_infinite() {
                            i32::MAX
                        } else {
                            // Zero or NaN
                            i32::MIN
                        }
                    })
                }
            }

            impl MpOp for crate::op::[<jn $suffix>]::Routine {
                type MpTy = MpFloat;

                fn new_mp() -> Self::MpTy {
                    new_mpfloat::<Self::FTy>()
                }

                fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
                    let (n, x) = input;
                    this.assign(x);
                    let ord = this.jn_round(n, Nearest);
                    prep_retval::<Self::FTy>(this, ord)
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

            impl MpOp for crate::op::[<remquo $suffix>]::Routine {
                type MpTy = (MpFloat, MpFloat);

                fn new_mp() -> Self::MpTy {
                    (
                        new_mpfloat::<Self::FTy>(),
                        new_mpfloat::<Self::FTy>(),
                    )
                }

                fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
                    this.0.assign(input.0);
                    this.1.assign(input.1);
                    let (ord, q) = this.0.remainder_quo31_round(&this.1, Nearest);
                    (prep_retval::<Self::FTy>(&mut this.0, ord), q)
                }
            }

            impl MpOp for crate::op::[<yn $suffix>]::Routine {
                type MpTy = MpFloat;

                fn new_mp() -> Self::MpTy {
                    new_mpfloat::<Self::FTy>()
                }

                fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
                    let (n, x) = input;
                    this.assign(x);
                    let ord = this.yn_round(n, Nearest);
                    prep_retval::<Self::FTy>(this, ord)
                }
            }
        }
    };
}

/// Version of `impl_op_for_ty` with only functions that have `f16` and `f128` implementations.
macro_rules! impl_op_for_ty_all {
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

            impl MpOp for crate::op::[< fmaximum $suffix >]::Routine {
                type MpTy = (MpFloat, MpFloat);

                fn new_mp() -> Self::MpTy {
                    (new_mpfloat::<Self::FTy>(), new_mpfloat::<Self::FTy>())
                }

                fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
                    this.0.assign(input.0);
                    this.1.assign(input.1);
                    let ord = if this.0.is_nan() || this.1.is_nan() {
                        this.0.assign($fty::NAN);
                        Ordering::Equal
                    } else {
                        this.0.max_round(&this.1, Nearest)
                    };
                    prep_retval::<Self::RustRet>(&mut this.0, ord)
                }
            }

            impl MpOp for crate::op::[< fminimum $suffix >]::Routine {
                type MpTy = (MpFloat, MpFloat);

                fn new_mp() -> Self::MpTy {
                    (new_mpfloat::<Self::FTy>(), new_mpfloat::<Self::FTy>())
                }

                fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
                    this.0.assign(input.0);
                    this.1.assign(input.1);
                    let ord = if this.0.is_nan() || this.1.is_nan() {
                        this.0.assign($fty::NAN);
                        Ordering::Equal
                    } else {
                        this.0.min_round(&this.1, Nearest)
                    };
                    prep_retval::<Self::RustRet>(&mut this.0, ord)
                }
            }

            // `ldexp` and `scalbn` are the same for binary floating point, so just forward all
            // methods.
            impl MpOp for crate::op::[<ldexp $suffix>]::Routine {
                type MpTy = <crate::op::[<scalbn $suffix>]::Routine as MpOp>::MpTy;

                fn new_mp() -> Self::MpTy {
                    <crate::op::[<scalbn $suffix>]::Routine as MpOp>::new_mp()
                }

                fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
                    <crate::op::[<scalbn $suffix>]::Routine as MpOp>::run(this, input)
                }
            }

            impl MpOp for crate::op::[<scalbn $suffix>]::Routine {
                type MpTy = MpFloat;

                fn new_mp() -> Self::MpTy {
                    new_mpfloat::<Self::FTy>()
                }

                fn run(this: &mut Self::MpTy, input: Self::RustArgs) -> Self::RustRet {
                    this.assign(input.0);
                    *this <<= input.1;
                    prep_retval::<Self::FTy>(this, Ordering::Equal)
                }
            }
        }
    };
}

impl_op_for_ty!(f32, "f");
impl_op_for_ty!(f64, "");

#[cfg(f16_enabled)]
impl_op_for_ty_all!(f16, "f16");
impl_op_for_ty_all!(f32, "f");
impl_op_for_ty_all!(f64, "");
#[cfg(f128_enabled)]
impl_op_for_ty_all!(f128, "f128");

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

/* stub implementations so we don't need to special case them */

impl MpOp for crate::op::nextafter::Routine {
    type MpTy = MpFloat;

    fn new_mp() -> Self::MpTy {
        unimplemented!("nextafter does not yet have a MPFR operation");
    }

    fn run(_this: &mut Self::MpTy, _input: Self::RustArgs) -> Self::RustRet {
        unimplemented!("nextafter does not yet have a MPFR operation");
    }
}

impl MpOp for crate::op::nextafterf::Routine {
    type MpTy = MpFloat;

    fn new_mp() -> Self::MpTy {
        unimplemented!("nextafter does not yet have a MPFR operation");
    }

    fn run(_this: &mut Self::MpTy, _input: Self::RustArgs) -> Self::RustRet {
        unimplemented!("nextafter does not yet have a MPFR operation");
    }
}
