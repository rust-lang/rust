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
use std::num::ParseFloatError;

use syntax::ast;

use rustc_apfloat::{Float, FloatConvert, Status};
use rustc_apfloat::ieee::{Single, Double};

use super::err::*;

// Note that equality for `ConstFloat` means that the it is the same
// constant, not that the rust values are equal. In particular, `NaN
// == NaN` (at least if it's the same NaN; distinct encodings for NaN
// are considering unequal).
#[derive(Copy, Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct ConstFloat {
    pub ty: ast::FloatTy,

    // This is a bit inefficient but it makes conversions below more
    // ergonomic, and all of this will go away once `miri` is merged.
    pub bits: u128,
}

impl ConstFloat {
    /// Description of the type, not the value
    pub fn description(&self) -> &'static str {
        self.ty.ty_to_string()
    }

    /// Compares the values if they are of the same type
    pub fn try_cmp(self, rhs: Self) -> Result<Ordering, ConstMathErr> {
        match (self.ty, rhs.ty) {
            (ast::FloatTy::F64, ast::FloatTy::F64)  => {
                let a = Double::from_bits(self.bits);
                let b = Double::from_bits(rhs.bits);
                // This is pretty bad but it is the existing behavior.
                Ok(a.partial_cmp(&b).unwrap_or(Ordering::Greater))
            }

            (ast::FloatTy::F32, ast::FloatTy::F32) => {
                let a = Single::from_bits(self.bits);
                let b = Single::from_bits(rhs.bits);
                Ok(a.partial_cmp(&b).unwrap_or(Ordering::Greater))
            }

            _ => Err(CmpBetweenUnequalTypes),
        }
    }

    pub fn from_i128(input: i128, ty: ast::FloatTy) -> Self {
        let bits = match ty {
            ast::FloatTy::F32 => Single::from_i128(input).value.to_bits(),
            ast::FloatTy::F64 => Double::from_i128(input).value.to_bits()
        };
        ConstFloat { bits, ty }
    }

    pub fn from_u128(input: u128, ty: ast::FloatTy) -> Self {
        let bits = match ty {
            ast::FloatTy::F32 => Single::from_u128(input).value.to_bits(),
            ast::FloatTy::F64 => Double::from_u128(input).value.to_bits()
        };
        ConstFloat { bits, ty }
    }

    pub fn from_str(num: &str, ty: ast::FloatTy) -> Result<Self, ParseFloatError> {
        let bits = match ty {
            ast::FloatTy::F32 => {
                let rust_bits = num.parse::<f32>()?.to_bits() as u128;
                let apfloat = num.parse::<Single>().unwrap_or_else(|e| {
                    panic!("apfloat::ieee::Single failed to parse `{}`: {:?}", num, e);
                });
                let apfloat_bits = apfloat.to_bits();
                assert!(rust_bits == apfloat_bits,
                    "apfloat::ieee::Single gave different result for `{}`: \
                     {}({:#x}) vs Rust's {}({:#x})",
                    num, apfloat, apfloat_bits,
                    Single::from_bits(rust_bits), rust_bits);
                apfloat_bits
            }
            ast::FloatTy::F64 => {
                let rust_bits = num.parse::<f64>()?.to_bits() as u128;
                let apfloat = num.parse::<Double>().unwrap_or_else(|e| {
                    panic!("apfloat::ieee::Double failed to parse `{}`: {:?}", num, e);
                });
                let apfloat_bits = apfloat.to_bits();
                assert!(rust_bits == apfloat_bits,
                    "apfloat::ieee::Double gave different result for `{}`: \
                     {}({:#x}) vs Rust's {}({:#x})",
                    num, apfloat, apfloat_bits,
                    Double::from_bits(rust_bits), rust_bits);
                apfloat_bits
            }
        };
        Ok(ConstFloat { bits, ty })
    }

    pub fn to_i128(self, width: usize) -> Option<i128> {
        assert!(width <= 128);
        let r = match self.ty {
            ast::FloatTy::F32 => Single::from_bits(self.bits).to_i128(width),
            ast::FloatTy::F64 => Double::from_bits(self.bits).to_i128(width)
        };
        if r.status.intersects(Status::INVALID_OP) {
            None
        } else {
            Some(r.value)
        }
    }

    pub fn to_u128(self, width: usize) -> Option<u128> {
        assert!(width <= 128);
        let r = match self.ty {
            ast::FloatTy::F32 => Single::from_bits(self.bits).to_u128(width),
            ast::FloatTy::F64 => Double::from_bits(self.bits).to_u128(width)
        };
        if r.status.intersects(Status::INVALID_OP) {
            None
        } else {
            Some(r.value)
        }
    }

    pub fn convert(self, to: ast::FloatTy) -> Self {
        let bits = match (self.ty, to) {
            (ast::FloatTy::F32, ast::FloatTy::F32) |
            (ast::FloatTy::F64, ast::FloatTy::F64) => return self,

            (ast::FloatTy::F32, ast::FloatTy::F64) => {
                Double::to_bits(Single::from_bits(self.bits).convert(&mut false).value)
            }
            (ast::FloatTy::F64, ast::FloatTy::F32) => {
                Single::to_bits(Double::from_bits(self.bits).convert(&mut false).value)
            }
        };
        ConstFloat { bits, ty: to }
    }
}

impl ::std::fmt::Display for ConstFloat {
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        match self.ty {
            ast::FloatTy::F32 => write!(fmt, "{:#}", Single::from_bits(self.bits))?,
            ast::FloatTy::F64 => write!(fmt, "{:#}", Double::from_bits(self.bits))?,
        }
        write!(fmt, "{}", self.ty)
    }
}

impl ::std::fmt::Debug for ConstFloat {
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        ::std::fmt::Display::fmt(self, fmt)
    }
}

macro_rules! derive_binop {
    ($op:ident, $func:ident) => {
        impl ::std::ops::$op for ConstFloat {
            type Output = Result<Self, ConstMathErr>;
            fn $func(self, rhs: Self) -> Result<Self, ConstMathErr> {
                let bits = match (self.ty, rhs.ty) {
                    (ast::FloatTy::F32, ast::FloatTy::F32) =>{
                        let a = Single::from_bits(self.bits);
                        let b = Single::from_bits(rhs.bits);
                        a.$func(b).value.to_bits()
                    }
                    (ast::FloatTy::F64, ast::FloatTy::F64) => {
                        let a = Double::from_bits(self.bits);
                        let b = Double::from_bits(rhs.bits);
                        a.$func(b).value.to_bits()
                    }
                    _ => return Err(UnequalTypes(Op::$op)),
                };
                Ok(ConstFloat { bits, ty: self.ty })
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
        let bits = match self.ty {
            ast::FloatTy::F32 => (-Single::from_bits(self.bits)).to_bits(),
            ast::FloatTy::F64 => (-Double::from_bits(self.bits)).to_bits(),
        };
        ConstFloat { bits, ty: self.ty }
    }
}

/// This is `f32::MAX + (0.5 ULP)` as an integer. Numbers greater or equal to this
/// are rounded to infinity when converted to `f32`.
///
/// NB: Computed as maximum significand with an extra 1 bit added (for the half ULP)
/// shifted by the maximum exponent (accounting for normalization).
pub const MAX_F32_PLUS_HALF_ULP: u128 = ((1 << (Single::PRECISION + 1)) - 1)
                                        << (Single::MAX_EXP - Single::PRECISION as i16);
