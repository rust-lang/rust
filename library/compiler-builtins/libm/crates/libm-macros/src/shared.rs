/* List of all functions that is shared between `libm-macros` and `libm-test`. */

use std::fmt;
use std::sync::LazyLock;

const ALL_OPERATIONS_NESTED: &[(FloatTy, Signature, Option<Signature>, &[&str])] = &[
    (
        // `fn(f16) -> f16`
        FloatTy::F16,
        Signature { args: &[Ty::F16], returns: &[Ty::F16] },
        None,
        &["ceilf16", "fabsf16", "floorf16", "rintf16", "roundf16", "sqrtf16", "truncf16"],
    ),
    (
        // `fn(f32) -> f32`
        FloatTy::F32,
        Signature { args: &[Ty::F32], returns: &[Ty::F32] },
        None,
        &[
            "acosf", "acoshf", "asinf", "asinhf", "atanf", "atanhf", "cbrtf", "ceilf", "cosf",
            "coshf", "erff", "erfcf", "exp10f", "exp2f", "expf", "expm1f", "fabsf", "floorf",
            "j0f", "j1f", "lgammaf", "log10f", "log1pf", "log2f", "logf", "rintf", "roundf",
            "sinf", "sinhf", "sqrtf", "tanf", "tanhf", "tgammaf", "truncf", "y0f", "y1f",
        ],
    ),
    (
        // `(f64) -> f64`
        FloatTy::F64,
        Signature { args: &[Ty::F64], returns: &[Ty::F64] },
        None,
        &[
            "acos", "acosh", "asin", "asinh", "atan", "atanh", "cbrt", "ceil", "cos", "cosh",
            "erf", "erfc", "exp10", "exp2", "exp", "expm1", "fabs", "floor", "j0", "j1", "lgamma",
            "log10", "log1p", "log2", "log", "rint", "round", "sin", "sinh", "sqrt", "tan", "tanh",
            "tgamma", "trunc", "y0", "y1",
        ],
    ),
    (
        // `fn(f128) -> f128`
        FloatTy::F128,
        Signature { args: &[Ty::F128], returns: &[Ty::F128] },
        None,
        &["ceilf128", "fabsf128", "floorf128", "rintf128", "roundf128", "sqrtf128", "truncf128"],
    ),
    (
        // `(f16, f16) -> f16`
        FloatTy::F16,
        Signature { args: &[Ty::F16, Ty::F16], returns: &[Ty::F16] },
        None,
        &["copysignf16", "fdimf16", "fmaxf16", "fminf16", "fmodf16"],
    ),
    (
        // `(f32, f32) -> f32`
        FloatTy::F32,
        Signature { args: &[Ty::F32, Ty::F32], returns: &[Ty::F32] },
        None,
        &[
            "atan2f",
            "copysignf",
            "fdimf",
            "fmaxf",
            "fminf",
            "fmodf",
            "hypotf",
            "nextafterf",
            "powf",
            "remainderf",
        ],
    ),
    (
        // `(f64, f64) -> f64`
        FloatTy::F64,
        Signature { args: &[Ty::F64, Ty::F64], returns: &[Ty::F64] },
        None,
        &[
            "atan2",
            "copysign",
            "fdim",
            "fmax",
            "fmin",
            "fmod",
            "hypot",
            "nextafter",
            "pow",
            "remainder",
        ],
    ),
    (
        // `(f128, f128) -> f128`
        FloatTy::F128,
        Signature { args: &[Ty::F128, Ty::F128], returns: &[Ty::F128] },
        None,
        &["copysignf128", "fdimf128", "fmaxf128", "fminf128"],
    ),
    (
        // `(f32, f32, f32) -> f32`
        FloatTy::F32,
        Signature { args: &[Ty::F32, Ty::F32, Ty::F32], returns: &[Ty::F32] },
        None,
        &["fmaf"],
    ),
    (
        // `(f64, f64, f64) -> f64`
        FloatTy::F64,
        Signature { args: &[Ty::F64, Ty::F64, Ty::F64], returns: &[Ty::F64] },
        None,
        &["fma"],
    ),
    (
        // `(f32) -> i32`
        FloatTy::F32,
        Signature { args: &[Ty::F32], returns: &[Ty::I32] },
        None,
        &["ilogbf"],
    ),
    (
        // `(f64) -> i32`
        FloatTy::F64,
        Signature { args: &[Ty::F64], returns: &[Ty::I32] },
        None,
        &["ilogb"],
    ),
    (
        // `(i32, f32) -> f32`
        FloatTy::F32,
        Signature { args: &[Ty::I32, Ty::F32], returns: &[Ty::F32] },
        None,
        &["jnf", "ynf"],
    ),
    (
        // `(i32, f64) -> f64`
        FloatTy::F64,
        Signature { args: &[Ty::I32, Ty::F64], returns: &[Ty::F64] },
        None,
        &["jn", "yn"],
    ),
    (
        // `(f32, i32) -> f32`
        FloatTy::F32,
        Signature { args: &[Ty::F32, Ty::I32], returns: &[Ty::F32] },
        None,
        &["scalbnf", "ldexpf"],
    ),
    (
        // `(f64, i64) -> f64`
        FloatTy::F64,
        Signature { args: &[Ty::F64, Ty::I32], returns: &[Ty::F64] },
        None,
        &["scalbn", "ldexp"],
    ),
    (
        // `(f32, &mut f32) -> f32` as `(f32) -> (f32, f32)`
        FloatTy::F32,
        Signature { args: &[Ty::F32], returns: &[Ty::F32, Ty::F32] },
        Some(Signature { args: &[Ty::F32, Ty::MutF32], returns: &[Ty::F32] }),
        &["modff"],
    ),
    (
        // `(f64, &mut f64) -> f64` as  `(f64) -> (f64, f64)`
        FloatTy::F64,
        Signature { args: &[Ty::F64], returns: &[Ty::F64, Ty::F64] },
        Some(Signature { args: &[Ty::F64, Ty::MutF64], returns: &[Ty::F64] }),
        &["modf"],
    ),
    (
        // `(f32, &mut c_int) -> f32` as `(f32) -> (f32, i32)`
        FloatTy::F32,
        Signature { args: &[Ty::F32], returns: &[Ty::F32, Ty::I32] },
        Some(Signature { args: &[Ty::F32, Ty::MutCInt], returns: &[Ty::F32] }),
        &["frexpf", "lgammaf_r"],
    ),
    (
        // `(f64, &mut c_int) -> f64` as `(f64) -> (f64, i32)`
        FloatTy::F64,
        Signature { args: &[Ty::F64], returns: &[Ty::F64, Ty::I32] },
        Some(Signature { args: &[Ty::F64, Ty::MutCInt], returns: &[Ty::F64] }),
        &["frexp", "lgamma_r"],
    ),
    (
        // `(f32, f32, &mut c_int) -> f32` as `(f32, f32) -> (f32, i32)`
        FloatTy::F32,
        Signature { args: &[Ty::F32, Ty::F32], returns: &[Ty::F32, Ty::I32] },
        Some(Signature { args: &[Ty::F32, Ty::F32, Ty::MutCInt], returns: &[Ty::F32] }),
        &["remquof"],
    ),
    (
        // `(f64, f64, &mut c_int) -> f64` as `(f64, f64) -> (f64, i32)`
        FloatTy::F64,
        Signature { args: &[Ty::F64, Ty::F64], returns: &[Ty::F64, Ty::I32] },
        Some(Signature { args: &[Ty::F64, Ty::F64, Ty::MutCInt], returns: &[Ty::F64] }),
        &["remquo"],
    ),
    (
        // `(f32, &mut f32, &mut f32)` as `(f32) -> (f32, f32)`
        FloatTy::F32,
        Signature { args: &[Ty::F32], returns: &[Ty::F32, Ty::F32] },
        Some(Signature { args: &[Ty::F32, Ty::MutF32, Ty::MutF32], returns: &[] }),
        &["sincosf"],
    ),
    (
        // `(f64, &mut f64, &mut f64)` as `(f64) -> (f64, f64)`
        FloatTy::F64,
        Signature { args: &[Ty::F64], returns: &[Ty::F64, Ty::F64] },
        Some(Signature { args: &[Ty::F64, Ty::MutF64, Ty::MutF64], returns: &[] }),
        &["sincos"],
    ),
];

/// A type used in a function signature.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    F16,
    F32,
    F64,
    F128,
    I32,
    CInt,
    MutF16,
    MutF32,
    MutF64,
    MutF128,
    MutI32,
    MutCInt,
}

/// A subset of [`Ty`] representing only floats.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatTy {
    F16,
    F32,
    F64,
    F128,
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Ty::F16 => "f16",
            Ty::F32 => "f32",
            Ty::F64 => "f64",
            Ty::F128 => "f128",
            Ty::I32 => "i32",
            Ty::CInt => "::core::ffi::c_int",
            Ty::MutF16 => "&mut f16",
            Ty::MutF32 => "&mut f32",
            Ty::MutF64 => "&mut f64",
            Ty::MutF128 => "&mut f128",
            Ty::MutI32 => "&mut i32",
            Ty::MutCInt => "&mut ::core::ffi::c_int",
        };
        f.write_str(s)
    }
}

impl fmt::Display for FloatTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FloatTy::F16 => "f16",
            FloatTy::F32 => "f32",
            FloatTy::F64 => "f64",
            FloatTy::F128 => "f128",
        };
        f.write_str(s)
    }
}

/// Representation of e.g. `(f32, f32) -> f32`
#[derive(Debug, Clone)]
pub struct Signature {
    pub args: &'static [Ty],
    pub returns: &'static [Ty],
}

/// Combined information about a function implementation.
#[derive(Debug, Clone)]
pub struct MathOpInfo {
    pub name: &'static str,
    pub float_ty: FloatTy,
    /// Function signature for C implementations
    pub c_sig: Signature,
    /// Function signature for Rust implementations
    pub rust_sig: Signature,
}

/// A flat representation of `ALL_FUNCTIONS`.
pub static ALL_OPERATIONS: LazyLock<Vec<MathOpInfo>> = LazyLock::new(|| {
    let mut ret = Vec::new();

    for (base_fty, rust_sig, c_sig, names) in ALL_OPERATIONS_NESTED {
        for name in *names {
            let api = MathOpInfo {
                name,
                float_ty: *base_fty,
                rust_sig: rust_sig.clone(),
                c_sig: c_sig.clone().unwrap_or_else(|| rust_sig.clone()),
            };
            ret.push(api);
        }
    }

    ret.sort_by_key(|item| item.name);
    ret
});
