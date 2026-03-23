/* List of all functions that is shared between `libm-macros` and `libm-test`. */

use std::collections::HashSet;
use std::fmt;
use std::sync::LazyLock;

struct NestedOp {
    float_ty: FloatTy,
    rust_sig: Signature,
    c_sig: Option<Signature>,
    fn_list: &'static [&'static str],
    scope: OpScope,
}

/// Indicate where a function is defined and whether it is public or private.
#[derive(Clone, Copy, Debug)]
pub enum OpScope {
    /// Part of `libm`'s public API.
    LibmPublic,
    /// Functions internal to `libm`, e.g. `rem_pio2`.
    #[allow(dead_code)]
    LibmPrivate,
    /// Functions part of the public API for `compiler-builtins`.
    BuiltinsPublic,
}

impl OpScope {
    /// Where we should look for functions of this scope.
    pub const fn path_root(self) -> &'static str {
        match self {
            OpScope::LibmPublic => "libm",
            OpScope::LibmPrivate => todo!(),
            OpScope::BuiltinsPublic => "crate::builtins_wrapper",
        }
    }

    pub fn defined_in_compiler_builtins(self) -> bool {
        match self {
            OpScope::LibmPublic | OpScope::LibmPrivate => false,
            OpScope::BuiltinsPublic => true,
        }
    }
}

/// We need a flat list to work with most of the time, but define things as a more convenient
/// nested list.
const ALL_OPERATIONS_NESTED: &[NestedOp] = &[
    /* compiler-builtins operations */
    NestedOp {
        float_ty: FloatTy::F16,
        rust_sig: Signature {
            args: &[Ty::F16, Ty::F16],
            returns: &[Ty::F16],
        },
        c_sig: None,
        fn_list: &["addf16", "mulf16", "subf16"],
        scope: OpScope::BuiltinsPublic,
    },
    NestedOp {
        float_ty: FloatTy::F32,
        rust_sig: Signature {
            args: &[Ty::F32, Ty::F32],
            returns: &[Ty::F32],
        },
        c_sig: None,
        fn_list: &["addf32", "divf32", "mulf32", "subf32"],
        scope: OpScope::BuiltinsPublic,
    },
    NestedOp {
        float_ty: FloatTy::F64,
        rust_sig: Signature {
            args: &[Ty::F64, Ty::F64],
            returns: &[Ty::F64],
        },
        c_sig: None,
        fn_list: &["addf64", "divf64", "mulf64", "subf64"],
        scope: OpScope::BuiltinsPublic,
    },
    NestedOp {
        float_ty: FloatTy::F128,
        rust_sig: Signature {
            args: &[Ty::F128, Ty::F128],
            returns: &[Ty::F128],
        },
        c_sig: None,
        fn_list: &["addf128", "divf128", "mulf128", "subf128"],
        scope: OpScope::BuiltinsPublic,
    },
    NestedOp {
        float_ty: FloatTy::F32,
        rust_sig: Signature {
            args: &[Ty::F32, Ty::I32],
            returns: &[Ty::F32],
        },
        c_sig: None,
        fn_list: &["powif32"],
        scope: OpScope::BuiltinsPublic,
    },
    NestedOp {
        float_ty: FloatTy::F64,
        rust_sig: Signature {
            args: &[Ty::F64, Ty::I32],
            returns: &[Ty::F64],
        },
        c_sig: None,
        fn_list: &["powif64"],
        scope: OpScope::BuiltinsPublic,
    },
    NestedOp {
        float_ty: FloatTy::F128,
        rust_sig: Signature {
            args: &[Ty::F128, Ty::I32],
            returns: &[Ty::F128],
        },
        c_sig: None,
        fn_list: &["powif128"],
        scope: OpScope::BuiltinsPublic,
    },
    /* libm operations */
    NestedOp {
        // `fn(f16) -> f16`
        float_ty: FloatTy::F16,
        rust_sig: Signature {
            args: &[Ty::F16],
            returns: &[Ty::F16],
        },
        c_sig: None,
        fn_list: &[
            "ceilf16",
            "fabsf16",
            "floorf16",
            "rintf16",
            "roundevenf16",
            "roundf16",
            "sqrtf16",
            "truncf16",
        ],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `fn(f32) -> f32`
        float_ty: FloatTy::F32,
        rust_sig: Signature {
            args: &[Ty::F32],
            returns: &[Ty::F32],
        },
        c_sig: None,
        fn_list: &[
            "acosf",
            "acoshf",
            "asinf",
            "asinhf",
            "atanf",
            "atanhf",
            "cbrtf",
            "ceilf",
            "cosf",
            "coshf",
            "erfcf",
            "erff",
            "exp10f",
            "exp2f",
            "expf",
            "expm1f",
            "fabsf",
            "floorf",
            "j0f",
            "j1f",
            "lgammaf",
            "log10f",
            "log1pf",
            "log2f",
            "logf",
            "rintf",
            "roundevenf",
            "roundf",
            "sinf",
            "sinhf",
            "sqrtf",
            "tanf",
            "tanhf",
            "tgammaf",
            "truncf",
            "y0f",
            "y1f",
        ],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f64) -> f64`
        float_ty: FloatTy::F64,
        rust_sig: Signature {
            args: &[Ty::F64],
            returns: &[Ty::F64],
        },
        c_sig: None,
        fn_list: &[
            "acos",
            "acosh",
            "asin",
            "asinh",
            "atan",
            "atanh",
            "cbrt",
            "ceil",
            "cos",
            "cosh",
            "erf",
            "erfc",
            "exp",
            "exp10",
            "exp2",
            "expm1",
            "fabs",
            "floor",
            "j0",
            "j1",
            "lgamma",
            "log",
            "log10",
            "log1p",
            "log2",
            "rint",
            "round",
            "roundeven",
            "sin",
            "sinh",
            "sqrt",
            "tan",
            "tanh",
            "tgamma",
            "trunc",
            "y0",
            "y1",
        ],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `fn(f128) -> f128`
        float_ty: FloatTy::F128,
        rust_sig: Signature {
            args: &[Ty::F128],
            returns: &[Ty::F128],
        },
        c_sig: None,
        fn_list: &[
            "ceilf128",
            "fabsf128",
            "floorf128",
            "rintf128",
            "roundevenf128",
            "roundf128",
            "sqrtf128",
            "truncf128",
        ],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f16, f16) -> f16`
        float_ty: FloatTy::F16,
        rust_sig: Signature {
            args: &[Ty::F16, Ty::F16],
            returns: &[Ty::F16],
        },
        c_sig: None,
        fn_list: &[
            "copysignf16",
            "fdimf16",
            "fmaxf16",
            "fmaximum_numf16",
            "fmaximumf16",
            "fminf16",
            "fminimum_numf16",
            "fminimumf16",
            "fmodf16",
        ],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f32, f32) -> f32`
        float_ty: FloatTy::F32,
        rust_sig: Signature {
            args: &[Ty::F32, Ty::F32],
            returns: &[Ty::F32],
        },
        c_sig: None,
        fn_list: &[
            "atan2f",
            "copysignf",
            "fdimf",
            "fmaxf",
            "fmaximum_numf",
            "fmaximumf",
            "fminf",
            "fminimum_numf",
            "fminimumf",
            "fmodf",
            "hypotf",
            "nextafterf",
            "powf",
            "remainderf",
        ],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f64, f64) -> f64`
        float_ty: FloatTy::F64,
        rust_sig: Signature {
            args: &[Ty::F64, Ty::F64],
            returns: &[Ty::F64],
        },
        c_sig: None,
        fn_list: &[
            "atan2",
            "copysign",
            "fdim",
            "fmax",
            "fmaximum",
            "fmaximum_num",
            "fmin",
            "fminimum",
            "fminimum_num",
            "fmod",
            "hypot",
            "nextafter",
            "pow",
            "remainder",
        ],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f128, f128) -> f128`
        float_ty: FloatTy::F128,
        rust_sig: Signature {
            args: &[Ty::F128, Ty::F128],
            returns: &[Ty::F128],
        },
        c_sig: None,
        fn_list: &[
            "copysignf128",
            "fdimf128",
            "fmaxf128",
            "fmaximum_numf128",
            "fmaximumf128",
            "fminf128",
            "fminimum_numf128",
            "fminimumf128",
            "fmodf128",
        ],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f32, f32, f32) -> f32`
        float_ty: FloatTy::F32,
        rust_sig: Signature {
            args: &[Ty::F32, Ty::F32, Ty::F32],
            returns: &[Ty::F32],
        },
        c_sig: None,
        fn_list: &["fmaf"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f64, f64, f64) -> f64`
        float_ty: FloatTy::F64,
        rust_sig: Signature {
            args: &[Ty::F64, Ty::F64, Ty::F64],
            returns: &[Ty::F64],
        },
        c_sig: None,
        fn_list: &["fma"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f128, f128, f128) -> f128`
        float_ty: FloatTy::F128,
        rust_sig: Signature {
            args: &[Ty::F128, Ty::F128, Ty::F128],
            returns: &[Ty::F128],
        },
        c_sig: None,
        fn_list: &["fmaf128"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f16) -> i32`
        float_ty: FloatTy::F16,
        rust_sig: Signature {
            args: &[Ty::F16],
            returns: &[Ty::I32],
        },
        c_sig: None,
        fn_list: &["ilogbf16"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f32) -> i32`
        float_ty: FloatTy::F32,
        rust_sig: Signature {
            args: &[Ty::F32],
            returns: &[Ty::I32],
        },
        c_sig: None,
        fn_list: &["ilogbf"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f64) -> i32`
        float_ty: FloatTy::F64,
        rust_sig: Signature {
            args: &[Ty::F64],
            returns: &[Ty::I32],
        },
        c_sig: None,
        fn_list: &["ilogb"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f128) -> i32`
        float_ty: FloatTy::F128,
        rust_sig: Signature {
            args: &[Ty::F128],
            returns: &[Ty::I32],
        },
        c_sig: None,
        fn_list: &["ilogbf128"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(i32, f32) -> f32`
        float_ty: FloatTy::F32,
        rust_sig: Signature {
            args: &[Ty::I32, Ty::F32],
            returns: &[Ty::F32],
        },
        c_sig: None,
        fn_list: &["jnf", "ynf"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(i32, f64) -> f64`
        float_ty: FloatTy::F64,
        rust_sig: Signature {
            args: &[Ty::I32, Ty::F64],
            returns: &[Ty::F64],
        },
        c_sig: None,
        fn_list: &["jn", "yn"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f16, i32) -> f16`
        float_ty: FloatTy::F16,
        rust_sig: Signature {
            args: &[Ty::F16, Ty::I32],
            returns: &[Ty::F16],
        },
        c_sig: None,
        fn_list: &["ldexpf16", "scalbnf16"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f32, i32) -> f32`
        float_ty: FloatTy::F32,
        rust_sig: Signature {
            args: &[Ty::F32, Ty::I32],
            returns: &[Ty::F32],
        },
        c_sig: None,
        fn_list: &["ldexpf", "scalbnf"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f64, i64) -> f64`
        float_ty: FloatTy::F64,
        rust_sig: Signature {
            args: &[Ty::F64, Ty::I32],
            returns: &[Ty::F64],
        },
        c_sig: None,
        fn_list: &["ldexp", "scalbn"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f128, i32) -> f128`
        float_ty: FloatTy::F128,
        rust_sig: Signature {
            args: &[Ty::F128, Ty::I32],
            returns: &[Ty::F128],
        },
        c_sig: None,
        fn_list: &["ldexpf128", "scalbnf128"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f32, &mut f32) -> f32` as `(f32) -> (f32, f32)`
        float_ty: FloatTy::F32,
        rust_sig: Signature {
            args: &[Ty::F32],
            returns: &[Ty::F32, Ty::F32],
        },
        c_sig: Some(Signature {
            args: &[Ty::F32, Ty::MutF32],
            returns: &[Ty::F32],
        }),
        fn_list: &["modff"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f64, &mut f64) -> f64` as  `(f64) -> (f64, f64)`
        float_ty: FloatTy::F64,
        rust_sig: Signature {
            args: &[Ty::F64],
            returns: &[Ty::F64, Ty::F64],
        },
        c_sig: Some(Signature {
            args: &[Ty::F64, Ty::MutF64],
            returns: &[Ty::F64],
        }),
        fn_list: &["modf"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f16, &mut c_int) -> f16` as `(f16) -> (f16, i32)`
        float_ty: FloatTy::F16,
        rust_sig: Signature {
            args: &[Ty::F16],
            returns: &[Ty::F16, Ty::I32],
        },
        c_sig: Some(Signature {
            args: &[Ty::F16, Ty::MutCInt],
            returns: &[Ty::F16],
        }),
        fn_list: &["frexpf16"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f32, &mut c_int) -> f32` as `(f32) -> (f32, i32)`
        float_ty: FloatTy::F32,
        rust_sig: Signature {
            args: &[Ty::F32],
            returns: &[Ty::F32, Ty::I32],
        },
        c_sig: Some(Signature {
            args: &[Ty::F32, Ty::MutCInt],
            returns: &[Ty::F32],
        }),
        fn_list: &["frexpf", "lgammaf_r"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f64, &mut c_int) -> f64` as `(f64) -> (f64, i32)`
        float_ty: FloatTy::F64,
        rust_sig: Signature {
            args: &[Ty::F64],
            returns: &[Ty::F64, Ty::I32],
        },
        c_sig: Some(Signature {
            args: &[Ty::F64, Ty::MutCInt],
            returns: &[Ty::F64],
        }),
        fn_list: &["frexp", "lgamma_r"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f128, &mut c_int) -> f128` as `(f128) -> (f128, i32)`
        float_ty: FloatTy::F128,
        rust_sig: Signature {
            args: &[Ty::F128],
            returns: &[Ty::F128, Ty::I32],
        },
        c_sig: Some(Signature {
            args: &[Ty::F128, Ty::MutCInt],
            returns: &[Ty::F128],
        }),
        fn_list: &["frexpf128"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f32, f32, &mut c_int) -> f32` as `(f32, f32) -> (f32, i32)`
        float_ty: FloatTy::F32,
        rust_sig: Signature {
            args: &[Ty::F32, Ty::F32],
            returns: &[Ty::F32, Ty::I32],
        },
        c_sig: Some(Signature {
            args: &[Ty::F32, Ty::F32, Ty::MutCInt],
            returns: &[Ty::F32],
        }),
        fn_list: &["remquof"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f64, f64, &mut c_int) -> f64` as `(f64, f64) -> (f64, i32)`
        float_ty: FloatTy::F64,
        rust_sig: Signature {
            args: &[Ty::F64, Ty::F64],
            returns: &[Ty::F64, Ty::I32],
        },
        c_sig: Some(Signature {
            args: &[Ty::F64, Ty::F64, Ty::MutCInt],
            returns: &[Ty::F64],
        }),
        fn_list: &["remquo"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f32, &mut f32, &mut f32)` as `(f32) -> (f32, f32)`
        float_ty: FloatTy::F32,
        rust_sig: Signature {
            args: &[Ty::F32],
            returns: &[Ty::F32, Ty::F32],
        },
        c_sig: Some(Signature {
            args: &[Ty::F32, Ty::MutF32, Ty::MutF32],
            returns: &[],
        }),
        fn_list: &["sincosf"],
        scope: OpScope::LibmPublic,
    },
    NestedOp {
        // `(f64, &mut f64, &mut f64)` as `(f64) -> (f64, f64)`
        float_ty: FloatTy::F64,
        rust_sig: Signature {
            args: &[Ty::F64],
            returns: &[Ty::F64, Ty::F64],
        },
        c_sig: Some(Signature {
            args: &[Ty::F64, Ty::MutF64, Ty::MutF64],
            returns: &[],
        }),
        fn_list: &["sincos"],
        scope: OpScope::LibmPublic,
    },
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
    /// Indicate what crate this function is defined in and whether it is public or private.
    pub scope: OpScope,
    /// The path to this function, including crate but excluding the function itself.
    pub path: String,
}

/// A flat representation of `ALL_FUNCTIONS`.
pub static ALL_OPERATIONS: LazyLock<Vec<MathOpInfo>> = LazyLock::new(|| {
    let mut ret = Vec::new();

    for op in ALL_OPERATIONS_NESTED {
        let fn_names = op.fn_list;
        for name in fn_names {
            let api = MathOpInfo {
                name,
                float_ty: op.float_ty,
                rust_sig: op.rust_sig.clone(),
                c_sig: op.c_sig.clone().unwrap_or_else(|| op.rust_sig.clone()),
                scope: op.scope,
                path: format!("{}::{name}", op.scope.path_root()),
            };
            ret.push(api);
        }

        if !fn_names.is_sorted() {
            let mut sorted = (*fn_names).to_owned();
            sorted.sort_unstable();
            panic!("names list is not sorted: {fn_names:?}\nExpected: {sorted:?}");
        }
    }

    ret.sort_by_key(|item| item.name);

    let mut names = HashSet::new();
    let mut paths = HashSet::new();
    for item in &ret {
        let new_name = names.insert(item.name);
        assert!(new_name, "duplicate name `{item:?}`");
        let new_path = paths.insert(&item.path);
        assert!(new_path, "duplicate path`{item:?}`");
    }

    ret
});
