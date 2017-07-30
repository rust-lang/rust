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
use syntax::attr::IntType;
use syntax::ast::{IntTy, UintTy};

use super::is::*;
use super::us::*;
use super::err::*;

#[derive(Copy, Clone, Debug, RustcEncodable, RustcDecodable, Hash, Eq, PartialEq)]
pub enum ConstInt {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
    Isize(ConstIsize),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
    Usize(ConstUsize),
}
pub use self::ConstInt::*;


macro_rules! bounds {
    ($ct: ty, $($t:ident $min:ident $max:ident)*) => {
        $(
            pub const $min: $ct = $t::min_value() as $ct;
            pub const $max: $ct = $t::max_value() as $ct;
        )*
    };
    ($ct: ty: $min_val: expr, $($t:ident $min:ident $max:ident)*) => {
        $(
            pub const $min: $ct = $min_val;
            pub const $max: $ct = $t::max_value() as $ct;
        )*
    }
}

mod ubounds {
    #![allow(dead_code)]
    bounds!{u128: 0,
        i8 I8MIN I8MAX i16 I16MIN I16MAX i32 I32MIN I32MAX i64 I64MIN I64MAX i128 I128MIN I128MAX
        u8 U8MIN U8MAX u16 U16MIN U16MAX u32 U32MIN U32MAX u64 U64MIN U64MAX u128 U128MIN U128MAX
        // do not add constants for isize/usize, because these are guaranteed to be wrong for
        // arbitrary host/target combinations
    }
}

mod ibounds {
    #![allow(dead_code)]
    bounds!(i128, u64 U64MIN U64MAX);

    pub const U128MIN: i128 = 0;
    pub const U128MAX: i128 = i128::max_value();

    bounds!{i128,
        i8 I8MIN I8MAX i16 I16MIN I16MAX i32 I32MIN I32MAX i64 I64MIN I64MAX i128 I128MIN I128MAX
        u8 U8MIN U8MAX u16 U16MIN U16MAX u32 U32MIN U32MAX
        // do not add constants for isize/usize, because these are guaranteed to be wrong for
        // arbitrary host/target combinations
    }
}

impl ConstInt {
    /// Creates a new unsigned ConstInt with matching type while also checking that overflow does
    /// not happen.
    pub fn new_unsigned(val: u128, ty: UintTy, usize_ty: UintTy) -> Option<ConstInt> {
        match ty {
            UintTy::U8 if val <= ubounds::U8MAX => Some(U8(val as u8)),
            UintTy::U16 if val <= ubounds::U16MAX => Some(U16(val as u16)),
            UintTy::U32 if val <= ubounds::U32MAX => Some(U32(val as u32)),
            UintTy::U64 if val <= ubounds::U64MAX => Some(U64(val as u64)),
            UintTy::Us if val <= ubounds::U64MAX => ConstUsize::new(val as u64, usize_ty).ok()
                .map(Usize),
            UintTy::U128 => Some(U128(val)),
            _ => None
        }
    }

    /// Creates a new signed ConstInt with matching type while also checking that overflow does
    /// not happen.
    pub fn new_signed(val: i128, ty: IntTy, isize_ty: IntTy) -> Option<ConstInt> {
        match ty {
            IntTy::I8 if val <= ibounds::I8MAX => Some(I8(val as i8)),
            IntTy::I16 if val <= ibounds::I16MAX => Some(I16(val as i16)),
            IntTy::I32 if val <= ibounds::I32MAX => Some(I32(val as i32)),
            IntTy::I64 if val <= ibounds::I64MAX => Some(I64(val as i64)),
            IntTy::Is if val <= ibounds::I64MAX => ConstIsize::new(val as i64, isize_ty).ok()
                .map(Isize),
            IntTy::I128 => Some(I128(val)),
            _ => None
        }
    }

    /// Creates a new unsigned ConstInt with matching type.
    pub fn new_unsigned_truncating(val: u128, ty: UintTy, usize_ty: UintTy) -> ConstInt {
        match ty {
            UintTy::U8 => U8(val as u8),
            UintTy::U16 => U16(val as u16),
            UintTy::U32 => U32(val as u32),
            UintTy::U64 => U64(val as u64),
            UintTy::Us => Usize(ConstUsize::new_truncating(val, usize_ty)),
            UintTy::U128 => U128(val)
        }
    }

    /// Creates a new signed ConstInt with matching type.
    pub fn new_signed_truncating(val: i128, ty: IntTy, isize_ty: IntTy) -> ConstInt {
        match ty {
            IntTy::I8 => I8(val as i8),
            IntTy::I16 => I16(val as i16),
            IntTy::I32 => I32(val as i32),
            IntTy::I64 => I64(val as i64),
            IntTy::Is => Isize(ConstIsize::new_truncating(val, isize_ty)),
            IntTy::I128 => I128(val)
        }
    }

    /// Description of the type, not the value
    pub fn description(&self) -> &'static str {
        match *self {
            I8(_) => "i8",
            I16(_) => "i16",
            I32(_) => "i32",
            I64(_) => "i64",
            I128(_) => "i128",
            Isize(_) => "isize",
            U8(_) => "u8",
            U16(_) => "u16",
            U32(_) => "u32",
            U64(_) => "u64",
            U128(_) => "u128",
            Usize(_) => "usize",
        }
    }

    /// Erases the type and returns a u128.
    /// This is not the same as `-5i8 as u128` but as `-5i8 as i128 as u128`
    pub fn to_u128_unchecked(self) -> u128 {
        match self {
            I8(i) => i as i128 as u128,
            I16(i) => i as i128 as u128,
            I32(i) => i as i128 as u128,
            I64(i) => i as i128 as u128,
            I128(i) => i as i128 as u128,
            Isize(Is16(i)) => i as i128 as u128,
            Isize(Is32(i)) => i as i128 as u128,
            Isize(Is64(i)) => i as i128 as u128,
            U8(i) => i as u128,
            U16(i) => i as u128,
            U32(i) => i as u128,
            U64(i) => i as u128,
            U128(i) => i as u128,
            Usize(Us16(i)) => i as u128,
            Usize(Us32(i)) => i as u128,
            Usize(Us64(i)) => i as u128,
        }
    }

    /// Converts the value to a `u32` if it's in the range 0...std::u32::MAX
    pub fn to_u32(&self) -> Option<u32> {
        self.to_u128().and_then(|v| if v <= u32::max_value() as u128 {
            Some(v as u32)
        } else {
            None
        })
    }

    /// Converts the value to a `u64` if it's in the range 0...std::u64::MAX
    pub fn to_u64(&self) -> Option<u64> {
        self.to_u128().and_then(|v| if v <= u64::max_value() as u128 {
            Some(v as u64)
        } else {
            None
        })
    }

    /// Converts the value to a `u128` if it's in the range 0...std::u128::MAX
    pub fn to_u128(&self) -> Option<u128> {
        match *self {
            I8(v) if v >= 0 => Some(v as u128),
            I16(v) if v >= 0 => Some(v as u128),
            I32(v) if v >= 0 => Some(v as u128),
            I64(v) if v >= 0 => Some(v as u128),
            I128(v) if v >= 0 => Some(v as u128),
            Isize(Is16(v)) if v >= 0 => Some(v as u128),
            Isize(Is32(v)) if v >= 0 => Some(v as u128),
            Isize(Is64(v)) if v >= 0 => Some(v as u128),
            U8(v) => Some(v as u128),
            U16(v) => Some(v as u128),
            U32(v) => Some(v as u128),
            U64(v) => Some(v as u128),
            U128(v) => Some(v as u128),
            Usize(Us16(v)) => Some(v as u128),
            Usize(Us32(v)) => Some(v as u128),
            Usize(Us64(v)) => Some(v as u128),
            _ => None,
        }
    }

    pub fn is_negative(&self) -> bool {
        match *self {
            I8(v) => v < 0,
            I16(v) => v < 0,
            I32(v) => v < 0,
            I64(v) => v < 0,
            I128(v) => v < 0,
            Isize(Is16(v)) => v < 0,
            Isize(Is32(v)) => v < 0,
            Isize(Is64(v)) => v < 0,
            _ => false,
        }
    }

    /// Compares the values if they are of the same type
    pub fn try_cmp(self, rhs: Self) -> Result<::std::cmp::Ordering, ConstMathErr> {
        match (self, rhs) {
            (I8(a), I8(b)) => Ok(a.cmp(&b)),
            (I16(a), I16(b)) => Ok(a.cmp(&b)),
            (I32(a), I32(b)) => Ok(a.cmp(&b)),
            (I64(a), I64(b)) => Ok(a.cmp(&b)),
            (I128(a), I128(b)) => Ok(a.cmp(&b)),
            (Isize(Is16(a)), Isize(Is16(b))) => Ok(a.cmp(&b)),
            (Isize(Is32(a)), Isize(Is32(b))) => Ok(a.cmp(&b)),
            (Isize(Is64(a)), Isize(Is64(b))) => Ok(a.cmp(&b)),
            (U8(a), U8(b)) => Ok(a.cmp(&b)),
            (U16(a), U16(b)) => Ok(a.cmp(&b)),
            (U32(a), U32(b)) => Ok(a.cmp(&b)),
            (U64(a), U64(b)) => Ok(a.cmp(&b)),
            (U128(a), U128(b)) => Ok(a.cmp(&b)),
            (Usize(Us16(a)), Usize(Us16(b))) => Ok(a.cmp(&b)),
            (Usize(Us32(a)), Usize(Us32(b))) => Ok(a.cmp(&b)),
            (Usize(Us64(a)), Usize(Us64(b))) => Ok(a.cmp(&b)),
            _ => Err(CmpBetweenUnequalTypes),
        }
    }

    /// Adds 1 to the value and wraps around if the maximum for the type is reached
    pub fn wrap_incr(self) -> Self {
        macro_rules! add1 {
            ($e:expr) => { ($e).wrapping_add(1) }
        }
        match self {
            ConstInt::I8(i) => ConstInt::I8(add1!(i)),
            ConstInt::I16(i) => ConstInt::I16(add1!(i)),
            ConstInt::I32(i) => ConstInt::I32(add1!(i)),
            ConstInt::I64(i) => ConstInt::I64(add1!(i)),
            ConstInt::I128(i) => ConstInt::I128(add1!(i)),
            ConstInt::Isize(ConstIsize::Is16(i)) => ConstInt::Isize(ConstIsize::Is16(add1!(i))),
            ConstInt::Isize(ConstIsize::Is32(i)) => ConstInt::Isize(ConstIsize::Is32(add1!(i))),
            ConstInt::Isize(ConstIsize::Is64(i)) => ConstInt::Isize(ConstIsize::Is64(add1!(i))),
            ConstInt::U8(i) => ConstInt::U8(add1!(i)),
            ConstInt::U16(i) => ConstInt::U16(add1!(i)),
            ConstInt::U32(i) => ConstInt::U32(add1!(i)),
            ConstInt::U64(i) => ConstInt::U64(add1!(i)),
            ConstInt::U128(i) => ConstInt::U128(add1!(i)),
            ConstInt::Usize(ConstUsize::Us16(i)) => ConstInt::Usize(ConstUsize::Us16(add1!(i))),
            ConstInt::Usize(ConstUsize::Us32(i)) => ConstInt::Usize(ConstUsize::Us32(add1!(i))),
            ConstInt::Usize(ConstUsize::Us64(i)) => ConstInt::Usize(ConstUsize::Us64(add1!(i))),
        }
    }

    pub fn int_type(self) -> IntType {
        match self {
            ConstInt::I8(_) => IntType::SignedInt(IntTy::I8),
            ConstInt::I16(_) => IntType::SignedInt(IntTy::I16),
            ConstInt::I32(_) => IntType::SignedInt(IntTy::I32),
            ConstInt::I64(_) => IntType::SignedInt(IntTy::I64),
            ConstInt::I128(_) => IntType::SignedInt(IntTy::I128),
            ConstInt::Isize(_) => IntType::SignedInt(IntTy::Is),
            ConstInt::U8(_) => IntType::UnsignedInt(UintTy::U8),
            ConstInt::U16(_) => IntType::UnsignedInt(UintTy::U16),
            ConstInt::U32(_) => IntType::UnsignedInt(UintTy::U32),
            ConstInt::U64(_) => IntType::UnsignedInt(UintTy::U64),
            ConstInt::U128(_) => IntType::UnsignedInt(UintTy::U128),
            ConstInt::Usize(_) => IntType::UnsignedInt(UintTy::Us),
        }
    }
}

impl ::std::cmp::PartialOrd for ConstInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.try_cmp(*other).ok()
    }
}

impl ::std::cmp::Ord for ConstInt {
    fn cmp(&self, other: &Self) -> Ordering {
        self.try_cmp(*other).unwrap()
    }
}

impl ::std::fmt::Display for ConstInt {
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        match *self {
            I8(i) => write!(fmt, "{}i8", i),
            I16(i) => write!(fmt, "{}i16", i),
            I32(i) => write!(fmt, "{}i32", i),
            I64(i) => write!(fmt, "{}i64", i),
            I128(i) => write!(fmt, "{}i128", i),
            Isize(ConstIsize::Is64(i)) => write!(fmt, "{}isize", i),
            Isize(ConstIsize::Is32(i)) => write!(fmt, "{}isize", i),
            Isize(ConstIsize::Is16(i)) => write!(fmt, "{}isize", i),
            U8(i) => write!(fmt, "{}u8", i),
            U16(i) => write!(fmt, "{}u16", i),
            U32(i) => write!(fmt, "{}u32", i),
            U64(i) => write!(fmt, "{}u64", i),
            U128(i) => write!(fmt, "{}u128", i),
            Usize(ConstUsize::Us64(i)) => write!(fmt, "{}usize", i),
            Usize(ConstUsize::Us32(i)) => write!(fmt, "{}usize", i),
            Usize(ConstUsize::Us16(i)) => write!(fmt, "{}usize", i),
        }
    }
}

macro_rules! overflowing {
    ($e:expr, $err:expr) => {{
        if $e.1 {
            return Err(Overflow($err));
        } else {
            $e.0
        }
    }}
}

macro_rules! impl_binop {
    ($op:ident, $func:ident, $checked_func:ident) => {
        impl ::std::ops::$op for ConstInt {
            type Output = Result<Self, ConstMathErr>;
            fn $func(self, rhs: Self) -> Result<Self, ConstMathErr> {
                match (self, rhs) {
                    (I8(a), I8(b)) => a.$checked_func(b).map(I8),
                    (I16(a), I16(b)) => a.$checked_func(b).map(I16),
                    (I32(a), I32(b)) => a.$checked_func(b).map(I32),
                    (I64(a), I64(b)) => a.$checked_func(b).map(I64),
                    (I128(a), I128(b)) => a.$checked_func(b).map(I128),
                    (Isize(Is16(a)), Isize(Is16(b))) => a.$checked_func(b).map(Is16).map(Isize),
                    (Isize(Is32(a)), Isize(Is32(b))) => a.$checked_func(b).map(Is32).map(Isize),
                    (Isize(Is64(a)), Isize(Is64(b))) => a.$checked_func(b).map(Is64).map(Isize),
                    (U8(a), U8(b)) => a.$checked_func(b).map(U8),
                    (U16(a), U16(b)) => a.$checked_func(b).map(U16),
                    (U32(a), U32(b)) => a.$checked_func(b).map(U32),
                    (U64(a), U64(b)) => a.$checked_func(b).map(U64),
                    (U128(a), U128(b)) => a.$checked_func(b).map(U128),
                    (Usize(Us16(a)), Usize(Us16(b))) => a.$checked_func(b).map(Us16).map(Usize),
                    (Usize(Us32(a)), Usize(Us32(b))) => a.$checked_func(b).map(Us32).map(Usize),
                    (Usize(Us64(a)), Usize(Us64(b))) => a.$checked_func(b).map(Us64).map(Usize),
                    _ => return Err(UnequalTypes(Op::$op)),
                }.ok_or(Overflow(Op::$op))
            }
        }
    }
}

macro_rules! derive_binop {
    ($op:ident, $func:ident) => {
        impl ::std::ops::$op for ConstInt {
            type Output = Result<Self, ConstMathErr>;
            fn $func(self, rhs: Self) -> Result<Self, ConstMathErr> {
                match (self, rhs) {
                    (I8(a), I8(b)) => Ok(I8(a.$func(b))),
                    (I16(a), I16(b)) => Ok(I16(a.$func(b))),
                    (I32(a), I32(b)) => Ok(I32(a.$func(b))),
                    (I64(a), I64(b)) => Ok(I64(a.$func(b))),
                    (I128(a), I128(b)) => Ok(I128(a.$func(b))),
                    (Isize(Is16(a)), Isize(Is16(b))) => Ok(Isize(Is16(a.$func(b)))),
                    (Isize(Is32(a)), Isize(Is32(b))) => Ok(Isize(Is32(a.$func(b)))),
                    (Isize(Is64(a)), Isize(Is64(b))) => Ok(Isize(Is64(a.$func(b)))),
                    (U8(a), U8(b)) => Ok(U8(a.$func(b))),
                    (U16(a), U16(b)) => Ok(U16(a.$func(b))),
                    (U32(a), U32(b)) => Ok(U32(a.$func(b))),
                    (U64(a), U64(b)) => Ok(U64(a.$func(b))),
                    (U128(a), U128(b)) => Ok(U128(a.$func(b))),
                    (Usize(Us16(a)), Usize(Us16(b))) => Ok(Usize(Us16(a.$func(b)))),
                    (Usize(Us32(a)), Usize(Us32(b))) => Ok(Usize(Us32(a.$func(b)))),
                    (Usize(Us64(a)), Usize(Us64(b))) => Ok(Usize(Us64(a.$func(b)))),
                    _ => Err(UnequalTypes(Op::$op)),
                }
            }
        }
    }
}

impl_binop!(Add, add, checked_add);
impl_binop!(Sub, sub, checked_sub);
impl_binop!(Mul, mul, checked_mul);
derive_binop!(BitAnd, bitand);
derive_binop!(BitOr, bitor);
derive_binop!(BitXor, bitxor);

const I128_MIN: i128 = ::std::i128::MIN;

fn check_division(
    lhs: ConstInt,
    rhs: ConstInt,
    op: Op,
    zerr: ConstMathErr,
) -> Result<(), ConstMathErr> {
    match (lhs, rhs) {
        (I8(_), I8(0)) => Err(zerr),
        (I16(_), I16(0)) => Err(zerr),
        (I32(_), I32(0)) => Err(zerr),
        (I64(_), I64(0)) => Err(zerr),
        (I128(_), I128(0)) => Err(zerr),
        (Isize(_), Isize(Is16(0))) => Err(zerr),
        (Isize(_), Isize(Is32(0))) => Err(zerr),
        (Isize(_), Isize(Is64(0))) => Err(zerr),

        (U8(_), U8(0)) => Err(zerr),
        (U16(_), U16(0)) => Err(zerr),
        (U32(_), U32(0)) => Err(zerr),
        (U64(_), U64(0)) => Err(zerr),
        (U128(_), U128(0)) => Err(zerr),
        (Usize(_), Usize(Us16(0))) => Err(zerr),
        (Usize(_), Usize(Us32(0))) => Err(zerr),
        (Usize(_), Usize(Us64(0))) => Err(zerr),

        (I8(::std::i8::MIN), I8(-1)) => Err(Overflow(op)),
        (I16(::std::i16::MIN), I16(-1)) => Err(Overflow(op)),
        (I32(::std::i32::MIN), I32(-1)) => Err(Overflow(op)),
        (I64(::std::i64::MIN), I64(-1)) => Err(Overflow(op)),
        (I128(I128_MIN), I128(-1)) => Err(Overflow(op)),
        (Isize(Is16(::std::i16::MIN)), Isize(Is16(-1))) => Err(Overflow(op)),
        (Isize(Is32(::std::i32::MIN)), Isize(Is32(-1))) => Err(Overflow(op)),
        (Isize(Is64(::std::i64::MIN)), Isize(Is64(-1))) => Err(Overflow(op)),

        _ => Ok(()),
    }
}

impl ::std::ops::Div for ConstInt {
    type Output = Result<Self, ConstMathErr>;
    fn div(self, rhs: Self) -> Result<Self, ConstMathErr> {
        let (lhs, rhs) = (self, rhs);
        check_division(lhs, rhs, Op::Div, DivisionByZero)?;
        match (lhs, rhs) {
            (I8(a), I8(b)) => Ok(I8(a/b)),
            (I16(a), I16(b)) => Ok(I16(a/b)),
            (I32(a), I32(b)) => Ok(I32(a/b)),
            (I64(a), I64(b)) => Ok(I64(a/b)),
            (I128(a), I128(b)) => Ok(I128(a/b)),
            (Isize(Is16(a)), Isize(Is16(b))) => Ok(Isize(Is16(a/b))),
            (Isize(Is32(a)), Isize(Is32(b))) => Ok(Isize(Is32(a/b))),
            (Isize(Is64(a)), Isize(Is64(b))) => Ok(Isize(Is64(a/b))),

            (U8(a), U8(b)) => Ok(U8(a/b)),
            (U16(a), U16(b)) => Ok(U16(a/b)),
            (U32(a), U32(b)) => Ok(U32(a/b)),
            (U64(a), U64(b)) => Ok(U64(a/b)),
            (U128(a), U128(b)) => Ok(U128(a/b)),
            (Usize(Us16(a)), Usize(Us16(b))) => Ok(Usize(Us16(a/b))),
            (Usize(Us32(a)), Usize(Us32(b))) => Ok(Usize(Us32(a/b))),
            (Usize(Us64(a)), Usize(Us64(b))) => Ok(Usize(Us64(a/b))),

            _ => Err(UnequalTypes(Op::Div)),
        }
    }
}

impl ::std::ops::Rem for ConstInt {
    type Output = Result<Self, ConstMathErr>;
    fn rem(self, rhs: Self) -> Result<Self, ConstMathErr> {
        let (lhs, rhs) = (self, rhs);
        // should INT_MIN%-1 be zero or an error?
        check_division(lhs, rhs, Op::Rem, RemainderByZero)?;
        match (lhs, rhs) {
            (I8(a), I8(b)) => Ok(I8(a%b)),
            (I16(a), I16(b)) => Ok(I16(a%b)),
            (I32(a), I32(b)) => Ok(I32(a%b)),
            (I64(a), I64(b)) => Ok(I64(a%b)),
            (I128(a), I128(b)) => Ok(I128(a%b)),
            (Isize(Is16(a)), Isize(Is16(b))) => Ok(Isize(Is16(a%b))),
            (Isize(Is32(a)), Isize(Is32(b))) => Ok(Isize(Is32(a%b))),
            (Isize(Is64(a)), Isize(Is64(b))) => Ok(Isize(Is64(a%b))),

            (U8(a), U8(b)) => Ok(U8(a%b)),
            (U16(a), U16(b)) => Ok(U16(a%b)),
            (U32(a), U32(b)) => Ok(U32(a%b)),
            (U64(a), U64(b)) => Ok(U64(a%b)),
            (U128(a), U128(b)) => Ok(U128(a%b)),
            (Usize(Us16(a)), Usize(Us16(b))) => Ok(Usize(Us16(a%b))),
            (Usize(Us32(a)), Usize(Us32(b))) => Ok(Usize(Us32(a%b))),
            (Usize(Us64(a)), Usize(Us64(b))) => Ok(Usize(Us64(a%b))),

            _ => Err(UnequalTypes(Op::Rem)),
        }
    }
}

impl ::std::ops::Shl<ConstInt> for ConstInt {
    type Output = Result<Self, ConstMathErr>;
    fn shl(self, rhs: Self) -> Result<Self, ConstMathErr> {
        let b = rhs.to_u32().ok_or(ShiftNegative)?;
        match self {
            I8(a) => Ok(I8(overflowing!(a.overflowing_shl(b), Op::Shl))),
            I16(a) => Ok(I16(overflowing!(a.overflowing_shl(b), Op::Shl))),
            I32(a) => Ok(I32(overflowing!(a.overflowing_shl(b), Op::Shl))),
            I64(a) => Ok(I64(overflowing!(a.overflowing_shl(b), Op::Shl))),
            I128(a) => Ok(I128(overflowing!(a.overflowing_shl(b), Op::Shl))),
            Isize(Is16(a)) => Ok(Isize(Is16(overflowing!(a.overflowing_shl(b), Op::Shl)))),
            Isize(Is32(a)) => Ok(Isize(Is32(overflowing!(a.overflowing_shl(b), Op::Shl)))),
            Isize(Is64(a)) => Ok(Isize(Is64(overflowing!(a.overflowing_shl(b), Op::Shl)))),
            U8(a) => Ok(U8(overflowing!(a.overflowing_shl(b), Op::Shl))),
            U16(a) => Ok(U16(overflowing!(a.overflowing_shl(b), Op::Shl))),
            U32(a) => Ok(U32(overflowing!(a.overflowing_shl(b), Op::Shl))),
            U64(a) => Ok(U64(overflowing!(a.overflowing_shl(b), Op::Shl))),
            U128(a) => Ok(U128(overflowing!(a.overflowing_shl(b), Op::Shl))),
            Usize(Us16(a)) => Ok(Usize(Us16(overflowing!(a.overflowing_shl(b), Op::Shl)))),
            Usize(Us32(a)) => Ok(Usize(Us32(overflowing!(a.overflowing_shl(b), Op::Shl)))),
            Usize(Us64(a)) => Ok(Usize(Us64(overflowing!(a.overflowing_shl(b), Op::Shl)))),
        }
    }
}

impl ::std::ops::Shr<ConstInt> for ConstInt {
    type Output = Result<Self, ConstMathErr>;
    fn shr(self, rhs: Self) -> Result<Self, ConstMathErr> {
        let b = rhs.to_u32().ok_or(ShiftNegative)?;
        match self {
            I8(a) => Ok(I8(overflowing!(a.overflowing_shr(b), Op::Shr))),
            I16(a) => Ok(I16(overflowing!(a.overflowing_shr(b), Op::Shr))),
            I32(a) => Ok(I32(overflowing!(a.overflowing_shr(b), Op::Shr))),
            I64(a) => Ok(I64(overflowing!(a.overflowing_shr(b), Op::Shr))),
            I128(a) => Ok(I128(overflowing!(a.overflowing_shr(b), Op::Shr))),
            Isize(Is16(a)) => Ok(Isize(Is16(overflowing!(a.overflowing_shr(b), Op::Shr)))),
            Isize(Is32(a)) => Ok(Isize(Is32(overflowing!(a.overflowing_shr(b), Op::Shr)))),
            Isize(Is64(a)) => Ok(Isize(Is64(overflowing!(a.overflowing_shr(b), Op::Shr)))),
            U8(a) => Ok(U8(overflowing!(a.overflowing_shr(b), Op::Shr))),
            U16(a) => Ok(U16(overflowing!(a.overflowing_shr(b), Op::Shr))),
            U32(a) => Ok(U32(overflowing!(a.overflowing_shr(b), Op::Shr))),
            U64(a) => Ok(U64(overflowing!(a.overflowing_shr(b), Op::Shr))),
            U128(a) => Ok(U128(overflowing!(a.overflowing_shr(b), Op::Shr))),
            Usize(Us16(a)) => Ok(Usize(Us16(overflowing!(a.overflowing_shr(b), Op::Shr)))),
            Usize(Us32(a)) => Ok(Usize(Us32(overflowing!(a.overflowing_shr(b), Op::Shr)))),
            Usize(Us64(a)) => Ok(Usize(Us64(overflowing!(a.overflowing_shr(b), Op::Shr)))),
        }
    }
}

impl ::std::ops::Neg for ConstInt {
    type Output = Result<Self, ConstMathErr>;
    fn neg(self) -> Result<Self, ConstMathErr> {
        match self {
            I8(a) => Ok(I8(overflowing!(a.overflowing_neg(), Op::Neg))),
            I16(a) => Ok(I16(overflowing!(a.overflowing_neg(), Op::Neg))),
            I32(a) => Ok(I32(overflowing!(a.overflowing_neg(), Op::Neg))),
            I64(a) => Ok(I64(overflowing!(a.overflowing_neg(), Op::Neg))),
            I128(a) => Ok(I128(overflowing!(a.overflowing_neg(), Op::Neg))),
            Isize(Is16(a)) => Ok(Isize(Is16(overflowing!(a.overflowing_neg(), Op::Neg)))),
            Isize(Is32(a)) => Ok(Isize(Is32(overflowing!(a.overflowing_neg(), Op::Neg)))),
            Isize(Is64(a)) => Ok(Isize(Is64(overflowing!(a.overflowing_neg(), Op::Neg)))),
            a@U8(0) | a@U16(0) | a@U32(0) | a@U64(0) | a@U128(0) |
            a@Usize(Us16(0)) | a@Usize(Us32(0)) | a@Usize(Us64(0)) => Ok(a),
            U8(_) | U16(_) | U32(_) | U64(_) | U128(_) | Usize(_) => Err(UnsignedNegation),
        }
    }
}

impl ::std::ops::Not for ConstInt {
    type Output = Result<Self, ConstMathErr>;
    fn not(self) -> Result<Self, ConstMathErr> {
        match self {
            I8(a) => Ok(I8(!a)),
            I16(a) => Ok(I16(!a)),
            I32(a) => Ok(I32(!a)),
            I64(a) => Ok(I64(!a)),
            I128(a) => Ok(I128(!a)),
            Isize(Is16(a)) => Ok(Isize(Is16(!a))),
            Isize(Is32(a)) => Ok(Isize(Is32(!a))),
            Isize(Is64(a)) => Ok(Isize(Is64(!a))),
            U8(a) => Ok(U8(!a)),
            U16(a) => Ok(U16(!a)),
            U32(a) => Ok(U32(!a)),
            U64(a) => Ok(U64(!a)),
            U128(a) => Ok(U128(!a)),
            Usize(Us16(a)) => Ok(Usize(Us16(!a))),
            Usize(Us32(a)) => Ok(Usize(Us32(!a))),
            Usize(Us64(a)) => Ok(Usize(Us64(!a))),
        }
    }
}
