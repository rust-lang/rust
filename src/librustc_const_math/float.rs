// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp::Ordering;
use std::hash;
use std::mem::transmute;

use super::err::*;

#[derive(Copy, Clone, Debug, RustcEncodable, RustcDecodable)]
pub enum ConstFloat {
    F32(f32),
    F64(f64),

    // When the type isn't known, we have to operate on both possibilities.
    FInfer {
        f32: f32,
        f64: f64
    }
}
pub use self::ConstFloat::*;

impl ConstFloat {
    /// Description of the type, not the value
    pub fn description(&self) -> &'static str {
        match *self {
            FInfer {..} => "float",
            F32(_) => "f32",
            F64(_) => "f64",
        }
    }

    pub fn is_nan(&self) -> bool {
        match *self {
            F32(f) => f.is_nan(),
            F64(f) => f.is_nan(),
            FInfer { f32, f64 } => f32.is_nan() || f64.is_nan()
        }
    }

    /// Compares the values if they are of the same type
    pub fn try_cmp(self, rhs: Self) -> Result<Ordering, ConstMathErr> {
        match (self, rhs) {
            (F64(a), F64(b)) |
            (F64(a), FInfer { f64: b, .. }) |
            (FInfer { f64: a, .. }, F64(b)) |
            (FInfer { f64: a, .. }, FInfer { f64: b, .. })  => {
                // This is pretty bad but it is the existing behavior.
                Ok(if a == b {
                    Ordering::Equal
                } else if a < b {
                    Ordering::Less
                } else {
                    Ordering::Greater
                })
            }

            (F32(a), F32(b)) |
            (F32(a), FInfer { f32: b, .. }) |
            (FInfer { f32: a, .. }, F32(b)) => {
                Ok(if a == b {
                    Ordering::Equal
                } else if a < b {
                    Ordering::Less
                } else {
                    Ordering::Greater
                })
            }

            _ => Err(CmpBetweenUnequalTypes),
        }
    }
}

/// Note that equality for `ConstFloat` means that the it is the same
/// constant, not that the rust values are equal. In particular, `NaN
/// == NaN` (at least if it's the same NaN; distinct encodings for NaN
/// are considering unequal).
impl PartialEq for ConstFloat {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (F64(a), F64(b)) |
            (F64(a), FInfer { f64: b, .. }) |
            (FInfer { f64: a, .. }, F64(b)) |
            (FInfer { f64: a, .. }, FInfer { f64: b, .. }) => {
                unsafe{transmute::<_,u64>(a) == transmute::<_,u64>(b)}
            }
            (F32(a), F32(b)) => {
                unsafe{transmute::<_,u32>(a) == transmute::<_,u32>(b)}
            }
            _ => false
        }
    }
}

impl Eq for ConstFloat {}

impl hash::Hash for ConstFloat {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        match *self {
            F64(a) | FInfer { f64: a, .. } => {
                unsafe { transmute::<_,u64>(a) }.hash(state)
            }
            F32(a) => {
                unsafe { transmute::<_,u32>(a) }.hash(state)
            }
        }
    }
}

impl ::std::fmt::Display for ConstFloat {
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        match *self {
            FInfer { f64, .. } => write!(fmt, "{}", f64),
            F32(f) => write!(fmt, "{}f32", f),
            F64(f) => write!(fmt, "{}f64", f),
        }
    }
}

macro_rules! derive_binop {
    ($op:ident, $func:ident) => {
        impl ::std::ops::$op for ConstFloat {
            type Output = Result<Self, ConstMathErr>;
            fn $func(self, rhs: Self) -> Result<Self, ConstMathErr> {
                match (self, rhs) {
                    (F32(a), F32(b)) |
                    (F32(a), FInfer { f32: b, .. }) |
                    (FInfer { f32: a, .. }, F32(b)) => Ok(F32(a.$func(b))),

                    (F64(a), F64(b)) |
                    (FInfer { f64: a, .. }, F64(b)) |
                    (F64(a), FInfer { f64: b, .. }) => Ok(F64(a.$func(b))),

                    (FInfer { f32: a32, f64: a64 },
                     FInfer { f32: b32, f64: b64 }) => Ok(FInfer {
                        f32: a32.$func(b32),
                        f64: a64.$func(b64)
                    }),

                    _ => Err(UnequalTypes(Op::$op)),
                }
            }
        }
    }
}

derive_binop!(Add, add);
derive_binop!(Sub, sub);
derive_binop!(Mul, mul);
derive_binop!(Div, div);
derive_binop!(Rem, rem);

impl ::std::ops::Neg for ConstFloat {
    type Output = Self;
    fn neg(self) -> Self {
        match self {
            F32(f) => F32(-f),
            F64(f) => F64(-f),
            FInfer { f32, f64 } => FInfer {
                f32: -f32,
                f64: -f64
            }
        }
    }
}
