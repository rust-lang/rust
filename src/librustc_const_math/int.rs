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
    Isize(ConstIsize),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    Usize(ConstUsize),
    Infer(u64),
    InferSigned(i64),
}
pub use self::ConstInt::*;


macro_rules! bounds {
    ($($t:ident $min:ident $max:ident)*) => {
        mod as_u64 {
            $(
                #[allow(dead_code)]
                pub const $min: u64 = ::std::$t::MIN as u64;
                #[allow(dead_code)]
                pub const $max: u64 = ::std::$t::MAX as u64;
            )*
        }
        mod as_i64 {
            $(
                #[allow(dead_code)]
                pub const $min: i64 = ::std::$t::MIN as i64;
                #[allow(dead_code)]
                pub const $max: i64 = ::std::$t::MAX as i64;
            )*
        }
    }
}

bounds!{
    i8 I8MIN I8MAX i16 I16MIN I16MAX i32 I32MIN I32MAX i64 I64MIN I64MAX isize IMIN IMAX
    u8 U8MIN U8MAX u16 U16MIN U16MAX u32 U32MIN U32MAX u64 U64MIN U64MAX usize UMIN UMAX
}

impl ConstInt {
    /// If either value is `Infer` or `InferSigned`, try to turn the value into the type of
    /// the other value. If both values have no type, don't do anything
    pub fn infer(self, other: Self) -> Result<(Self, Self), ConstMathErr> {
        let inferred = match (self, other) {
            (InferSigned(_), InferSigned(_))
            | (Infer(_), Infer(_)) => self, // no inference possible
            // kindof wrong, you could have had values > I64MAX during computation of a
            (Infer(a @ 0...as_u64::I64MAX), InferSigned(_)) => InferSigned(a as i64),
            (Infer(_), InferSigned(_)) => return Err(ConstMathErr::NotInRange),
            (_, InferSigned(_))
            | (_, Infer(_)) => return other.infer(self).map(|(b, a)| (a, b)),

            (Infer(a @ 0...as_u64::I8MAX), I8(_)) => I8(a as i64 as i8),
            (Infer(a @ 0...as_u64::I16MAX), I16(_)) => I16(a as i64 as i16),
            (Infer(a @ 0...as_u64::I32MAX), I32(_)) => I32(a as i64 as i32),
            (Infer(a @ 0...as_u64::I64MAX), I64(_)) => I64(a as i64),
            (Infer(a @ 0...as_u64::I16MAX), Isize(Is16(_))) => Isize(Is16(a as i64 as i16)),
            (Infer(a @ 0...as_u64::I32MAX), Isize(Is32(_))) => Isize(Is32(a as i64 as i32)),
            (Infer(a @ 0...as_u64::I64MAX), Isize(Is64(_))) => Isize(Is64(a as i64)),
            (Infer(a @ 0...as_u64::U8MAX), U8(_)) => U8(a as u8),
            (Infer(a @ 0...as_u64::U16MAX), U16(_)) => U16(a as u16),
            (Infer(a @ 0...as_u64::U32MAX), U32(_)) => U32(a as u32),
            (Infer(a), U64(_)) => U64(a),
            (Infer(a @ 0...as_u64::U16MAX), Usize(Us16(_))) => Usize(Us16(a as u16)),
            (Infer(a @ 0...as_u64::U32MAX), Usize(Us32(_))) => Usize(Us32(a as u32)),
            (Infer(a), Usize(Us64(_))) => Usize(Us64(a)),

            (Infer(_), _) => return Err(ConstMathErr::NotInRange),

            (InferSigned(a @ as_i64::I8MIN...as_i64::I8MAX), I8(_)) => I8(a as i8),
            (InferSigned(a @ as_i64::I16MIN...as_i64::I16MAX), I16(_)) => I16(a as i16),
            (InferSigned(a @ as_i64::I32MIN...as_i64::I32MAX), I32(_)) => I32(a as i32),
            (InferSigned(a), I64(_)) => I64(a),
            (InferSigned(a @ as_i64::I16MIN...as_i64::I16MAX), Isize(Is16(_))) => {
                Isize(Is16(a as i16))
            },
            (InferSigned(a @ as_i64::I32MIN...as_i64::I32MAX), Isize(Is32(_))) => {
                Isize(Is32(a as i32))
            },
            (InferSigned(a), Isize(Is64(_))) => Isize(Is64(a)),
            (InferSigned(a @ 0...as_i64::U8MAX), U8(_)) => U8(a as u8),
            (InferSigned(a @ 0...as_i64::U16MAX), U16(_)) => U16(a as u16),
            (InferSigned(a @ 0...as_i64::U32MAX), U32(_)) => U32(a as u32),
            (InferSigned(a @ 0...as_i64::I64MAX), U64(_)) => U64(a as u64),
            (InferSigned(a @ 0...as_i64::U16MAX), Usize(Us16(_))) => Usize(Us16(a as u16)),
            (InferSigned(a @ 0...as_i64::U32MAX), Usize(Us32(_))) => Usize(Us32(a as u32)),
            (InferSigned(a @ 0...as_i64::I64MAX), Usize(Us64(_))) => Usize(Us64(a as u64)),
            (InferSigned(_), _) => return Err(ConstMathErr::NotInRange),
            _ => self, // already known types
        };
        Ok((inferred, other))
    }

    /// Turn this value into an `Infer` or an `InferSigned`
    pub fn erase_type(self) -> Self {
        match self {
            Infer(i) => Infer(i),
            InferSigned(i) if i < 0 => InferSigned(i),
            I8(i) if i < 0 => InferSigned(i as i64),
            I16(i) if i < 0 => InferSigned(i as i64),
            I32(i) if i < 0 => InferSigned(i as i64),
            I64(i) if i < 0 => InferSigned(i as i64),
            Isize(Is16(i)) if i < 0 => InferSigned(i as i64),
            Isize(Is32(i)) if i < 0 => InferSigned(i as i64),
            Isize(Is64(i)) if i < 0 => InferSigned(i as i64),
            InferSigned(i) => Infer(i as u64),
            I8(i) => Infer(i as u64),
            I16(i) => Infer(i as u64),
            I32(i) => Infer(i as u64),
            I64(i) => Infer(i as u64),
            Isize(Is16(i)) => Infer(i as u64),
            Isize(Is32(i)) => Infer(i as u64),
            Isize(Is64(i)) => Infer(i as u64),
            U8(i) => Infer(i as u64),
            U16(i) => Infer(i as u64),
            U32(i) => Infer(i as u64),
            U64(i) => Infer(i as u64),
            Usize(Us16(i)) => Infer(i as u64),
            Usize(Us32(i)) => Infer(i as u64),
            Usize(Us64(i)) => Infer(i),
        }
    }

    /// Description of the type, not the value
    pub fn description(&self) -> &'static str {
        match *self {
            Infer(_) => "not yet inferred integral",
            InferSigned(_) => "not yet inferred signed integral",
            I8(_) => "i8",
            I16(_) => "i16",
            I32(_) => "i32",
            I64(_) => "i64",
            Isize(_) => "isize",
            U8(_) => "u8",
            U16(_) => "u16",
            U32(_) => "u32",
            U64(_) => "u64",
            Usize(_) => "usize",
        }
    }

    /// Erases the type and returns a u64.
    /// This is not the same as `-5i8 as u64` but as `-5i8 as i64 as u64`
    pub fn to_u64_unchecked(self) -> u64 {
        match self.erase_type() {
            ConstInt::Infer(i) => i,
            ConstInt::InferSigned(i) => i as u64,
            _ => unreachable!(),
        }
    }

    /// Converts the value to a `u32` if it's in the range 0...std::u32::MAX
    pub fn to_u32(&self) -> Option<u32> {
        match *self {
            I8(v) if v >= 0 => Some(v as u32),
            I16(v) if v >= 0 => Some(v as u32),
            I32(v) if v >= 0 => Some(v as u32),
            InferSigned(v)
            | Isize(Is64(v))
            | I64(v) if v >= 0 && v <= ::std::u32::MAX as i64 => Some(v as u32),
            Isize(Is32(v)) if v >= 0 => Some(v as u32),
            Isize(Is16(v)) if v >= 0 => Some(v as u32),
            U8(v) => Some(v as u32),
            U16(v) => Some(v as u32),
            U32(v) => Some(v),
            Infer(v)
            | Usize(Us64(v))
            | U64(v) if v <= ::std::u32::MAX as u64 => Some(v as u32),
            Usize(Us32(v)) => Some(v),
            Usize(Us16(v)) => Some(v as u32),
            _ => None,
        }
    }

    /// Converts the value to a `u64` if it's >= 0
    pub fn to_u64(&self) -> Option<u64> {
        match *self {
            Infer(v) => Some(v),
            InferSigned(v) if v >= 0 => Some(v as u64),
            I8(v) if v >= 0 => Some(v as u64),
            I16(v) if v >= 0 => Some(v as u64),
            I32(v) if v >= 0 => Some(v as u64),
            I64(v) if v >= 0 => Some(v as u64),
            Isize(Is16(v)) if v >= 0 => Some(v as u64),
            Isize(Is32(v)) if v >= 0 => Some(v as u64),
            Isize(Is64(v)) if v >= 0 => Some(v as u64),
            U8(v) => Some(v as u64),
            U16(v) => Some(v as u64),
            U32(v) => Some(v as u64),
            U64(v) => Some(v),
            Usize(Us16(v)) => Some(v as u64),
            Usize(Us32(v)) => Some(v as u64),
            Usize(Us64(v)) => Some(v),
            _ => None,
        }
    }

    pub fn is_negative(&self) -> bool {
        match *self {
            I8(v) => v < 0,
            I16(v) => v < 0,
            I32(v) => v < 0,
            I64(v) => v < 0,
            Isize(Is16(v)) => v < 0,
            Isize(Is32(v)) => v < 0,
            Isize(Is64(v)) => v < 0,
            InferSigned(v) => v < 0,
            _ => false,
        }
    }

    /// Compares the values if they are of the same type
    pub fn try_cmp(self, rhs: Self) -> Result<::std::cmp::Ordering, ConstMathErr> {
        match self.infer(rhs)? {
            (I8(a), I8(b)) => Ok(a.cmp(&b)),
            (I16(a), I16(b)) => Ok(a.cmp(&b)),
            (I32(a), I32(b)) => Ok(a.cmp(&b)),
            (I64(a), I64(b)) => Ok(a.cmp(&b)),
            (Isize(Is16(a)), Isize(Is16(b))) => Ok(a.cmp(&b)),
            (Isize(Is32(a)), Isize(Is32(b))) => Ok(a.cmp(&b)),
            (Isize(Is64(a)), Isize(Is64(b))) => Ok(a.cmp(&b)),
            (U8(a), U8(b)) => Ok(a.cmp(&b)),
            (U16(a), U16(b)) => Ok(a.cmp(&b)),
            (U32(a), U32(b)) => Ok(a.cmp(&b)),
            (U64(a), U64(b)) => Ok(a.cmp(&b)),
            (Usize(Us16(a)), Usize(Us16(b))) => Ok(a.cmp(&b)),
            (Usize(Us32(a)), Usize(Us32(b))) => Ok(a.cmp(&b)),
            (Usize(Us64(a)), Usize(Us64(b))) => Ok(a.cmp(&b)),
            (Infer(a), Infer(b)) => Ok(a.cmp(&b)),
            (InferSigned(a), InferSigned(b)) => Ok(a.cmp(&b)),
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
            ConstInt::Isize(ConstIsize::Is16(i)) => ConstInt::Isize(ConstIsize::Is16(add1!(i))),
            ConstInt::Isize(ConstIsize::Is32(i)) => ConstInt::Isize(ConstIsize::Is32(add1!(i))),
            ConstInt::Isize(ConstIsize::Is64(i)) => ConstInt::Isize(ConstIsize::Is64(add1!(i))),
            ConstInt::U8(i) => ConstInt::U8(add1!(i)),
            ConstInt::U16(i) => ConstInt::U16(add1!(i)),
            ConstInt::U32(i) => ConstInt::U32(add1!(i)),
            ConstInt::U64(i) => ConstInt::U64(add1!(i)),
            ConstInt::Usize(ConstUsize::Us16(i)) => ConstInt::Usize(ConstUsize::Us16(add1!(i))),
            ConstInt::Usize(ConstUsize::Us32(i)) => ConstInt::Usize(ConstUsize::Us32(add1!(i))),
            ConstInt::Usize(ConstUsize::Us64(i)) => ConstInt::Usize(ConstUsize::Us64(add1!(i))),
            ConstInt::Infer(_) | ConstInt::InferSigned(_) => panic!("no type info for const int"),
        }
    }

    pub fn int_type(self) -> Option<IntType> {
        match self {
            ConstInt::I8(_) => Some(IntType::SignedInt(IntTy::I8)),
            ConstInt::I16(_) => Some(IntType::SignedInt(IntTy::I16)),
            ConstInt::I32(_) => Some(IntType::SignedInt(IntTy::I32)),
            ConstInt::I64(_) => Some(IntType::SignedInt(IntTy::I64)),
            ConstInt::Isize(_) => Some(IntType::SignedInt(IntTy::Is)),
            ConstInt::U8(_) => Some(IntType::UnsignedInt(UintTy::U8)),
            ConstInt::U16(_) => Some(IntType::UnsignedInt(UintTy::U16)),
            ConstInt::U32(_) => Some(IntType::UnsignedInt(UintTy::U32)),
            ConstInt::U64(_) => Some(IntType::UnsignedInt(UintTy::U64)),
            ConstInt::Usize(_) => Some(IntType::UnsignedInt(UintTy::Us)),
            _ => None,
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
            Infer(i) => write!(fmt, "{}", i),
            InferSigned(i) => write!(fmt, "{}", i),
            I8(i) => write!(fmt, "{}i8", i),
            I16(i) => write!(fmt, "{}i16", i),
            I32(i) => write!(fmt, "{}i32", i),
            I64(i) => write!(fmt, "{}i64", i),
            Isize(ConstIsize::Is64(i)) => write!(fmt, "{}isize", i),
            Isize(ConstIsize::Is32(i)) => write!(fmt, "{}isize", i),
            Isize(ConstIsize::Is16(i)) => write!(fmt, "{}isize", i),
            U8(i) => write!(fmt, "{}u8", i),
            U16(i) => write!(fmt, "{}u16", i),
            U32(i) => write!(fmt, "{}u32", i),
            U64(i) => write!(fmt, "{}u64", i),
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
                match self.infer(rhs)? {
                    (I8(a), I8(b)) => a.$checked_func(b).map(I8),
                    (I16(a), I16(b)) => a.$checked_func(b).map(I16),
                    (I32(a), I32(b)) => a.$checked_func(b).map(I32),
                    (I64(a), I64(b)) => a.$checked_func(b).map(I64),
                    (Isize(Is16(a)), Isize(Is16(b))) => a.$checked_func(b).map(Is16).map(Isize),
                    (Isize(Is32(a)), Isize(Is32(b))) => a.$checked_func(b).map(Is32).map(Isize),
                    (Isize(Is64(a)), Isize(Is64(b))) => a.$checked_func(b).map(Is64).map(Isize),
                    (U8(a), U8(b)) => a.$checked_func(b).map(U8),
                    (U16(a), U16(b)) => a.$checked_func(b).map(U16),
                    (U32(a), U32(b)) => a.$checked_func(b).map(U32),
                    (U64(a), U64(b)) => a.$checked_func(b).map(U64),
                    (Usize(Us16(a)), Usize(Us16(b))) => a.$checked_func(b).map(Us16).map(Usize),
                    (Usize(Us32(a)), Usize(Us32(b))) => a.$checked_func(b).map(Us32).map(Usize),
                    (Usize(Us64(a)), Usize(Us64(b))) => a.$checked_func(b).map(Us64).map(Usize),
                    (Infer(a), Infer(b)) => a.$checked_func(b).map(Infer),
                    (InferSigned(a), InferSigned(b)) => a.$checked_func(b).map(InferSigned),
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
                match self.infer(rhs)? {
                    (I8(a), I8(b)) => Ok(I8(a.$func(b))),
                    (I16(a), I16(b)) => Ok(I16(a.$func(b))),
                    (I32(a), I32(b)) => Ok(I32(a.$func(b))),
                    (I64(a), I64(b)) => Ok(I64(a.$func(b))),
                    (Isize(Is16(a)), Isize(Is16(b))) => Ok(Isize(Is16(a.$func(b)))),
                    (Isize(Is32(a)), Isize(Is32(b))) => Ok(Isize(Is32(a.$func(b)))),
                    (Isize(Is64(a)), Isize(Is64(b))) => Ok(Isize(Is64(a.$func(b)))),
                    (U8(a), U8(b)) => Ok(U8(a.$func(b))),
                    (U16(a), U16(b)) => Ok(U16(a.$func(b))),
                    (U32(a), U32(b)) => Ok(U32(a.$func(b))),
                    (U64(a), U64(b)) => Ok(U64(a.$func(b))),
                    (Usize(Us16(a)), Usize(Us16(b))) => Ok(Usize(Us16(a.$func(b)))),
                    (Usize(Us32(a)), Usize(Us32(b))) => Ok(Usize(Us32(a.$func(b)))),
                    (Usize(Us64(a)), Usize(Us64(b))) => Ok(Usize(Us64(a.$func(b)))),
                    (Infer(a), Infer(b)) => Ok(Infer(a.$func(b))),
                    (InferSigned(a), InferSigned(b)) => Ok(InferSigned(a.$func(b))),
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
        (Isize(_), Isize(Is16(0))) => Err(zerr),
        (Isize(_), Isize(Is32(0))) => Err(zerr),
        (Isize(_), Isize(Is64(0))) => Err(zerr),
        (InferSigned(_), InferSigned(0)) => Err(zerr),

        (U8(_), U8(0)) => Err(zerr),
        (U16(_), U16(0)) => Err(zerr),
        (U32(_), U32(0)) => Err(zerr),
        (U64(_), U64(0)) => Err(zerr),
        (Usize(_), Usize(Us16(0))) => Err(zerr),
        (Usize(_), Usize(Us32(0))) => Err(zerr),
        (Usize(_), Usize(Us64(0))) => Err(zerr),
        (Infer(_), Infer(0)) => Err(zerr),

        (I8(::std::i8::MIN), I8(-1)) => Err(Overflow(op)),
        (I16(::std::i16::MIN), I16(-1)) => Err(Overflow(op)),
        (I32(::std::i32::MIN), I32(-1)) => Err(Overflow(op)),
        (I64(::std::i64::MIN), I64(-1)) => Err(Overflow(op)),
        (Isize(Is16(::std::i16::MIN)), Isize(Is16(-1))) => Err(Overflow(op)),
        (Isize(Is32(::std::i32::MIN)), Isize(Is32(-1))) => Err(Overflow(op)),
        (Isize(Is64(::std::i64::MIN)), Isize(Is64(-1))) => Err(Overflow(op)),
        (InferSigned(::std::i64::MIN), InferSigned(-1)) => Err(Overflow(op)),

        _ => Ok(()),
    }
}

impl ::std::ops::Div for ConstInt {
    type Output = Result<Self, ConstMathErr>;
    fn div(self, rhs: Self) -> Result<Self, ConstMathErr> {
        let (lhs, rhs) = self.infer(rhs)?;
        check_division(lhs, rhs, Op::Div, DivisionByZero)?;
        match (lhs, rhs) {
            (I8(a), I8(b)) => Ok(I8(a/b)),
            (I16(a), I16(b)) => Ok(I16(a/b)),
            (I32(a), I32(b)) => Ok(I32(a/b)),
            (I64(a), I64(b)) => Ok(I64(a/b)),
            (Isize(Is16(a)), Isize(Is16(b))) => Ok(Isize(Is16(a/b))),
            (Isize(Is32(a)), Isize(Is32(b))) => Ok(Isize(Is32(a/b))),
            (Isize(Is64(a)), Isize(Is64(b))) => Ok(Isize(Is64(a/b))),
            (InferSigned(a), InferSigned(b)) => Ok(InferSigned(a/b)),

            (U8(a), U8(b)) => Ok(U8(a/b)),
            (U16(a), U16(b)) => Ok(U16(a/b)),
            (U32(a), U32(b)) => Ok(U32(a/b)),
            (U64(a), U64(b)) => Ok(U64(a/b)),
            (Usize(Us16(a)), Usize(Us16(b))) => Ok(Usize(Us16(a/b))),
            (Usize(Us32(a)), Usize(Us32(b))) => Ok(Usize(Us32(a/b))),
            (Usize(Us64(a)), Usize(Us64(b))) => Ok(Usize(Us64(a/b))),
            (Infer(a), Infer(b)) => Ok(Infer(a/b)),

            _ => Err(UnequalTypes(Op::Div)),
        }
    }
}

impl ::std::ops::Rem for ConstInt {
    type Output = Result<Self, ConstMathErr>;
    fn rem(self, rhs: Self) -> Result<Self, ConstMathErr> {
        let (lhs, rhs) = self.infer(rhs)?;
        // should INT_MIN%-1 be zero or an error?
        check_division(lhs, rhs, Op::Rem, RemainderByZero)?;
        match (lhs, rhs) {
            (I8(a), I8(b)) => Ok(I8(a%b)),
            (I16(a), I16(b)) => Ok(I16(a%b)),
            (I32(a), I32(b)) => Ok(I32(a%b)),
            (I64(a), I64(b)) => Ok(I64(a%b)),
            (Isize(Is16(a)), Isize(Is16(b))) => Ok(Isize(Is16(a%b))),
            (Isize(Is32(a)), Isize(Is32(b))) => Ok(Isize(Is32(a%b))),
            (Isize(Is64(a)), Isize(Is64(b))) => Ok(Isize(Is64(a%b))),
            (InferSigned(a), InferSigned(b)) => Ok(InferSigned(a%b)),

            (U8(a), U8(b)) => Ok(U8(a%b)),
            (U16(a), U16(b)) => Ok(U16(a%b)),
            (U32(a), U32(b)) => Ok(U32(a%b)),
            (U64(a), U64(b)) => Ok(U64(a%b)),
            (Usize(Us16(a)), Usize(Us16(b))) => Ok(Usize(Us16(a%b))),
            (Usize(Us32(a)), Usize(Us32(b))) => Ok(Usize(Us32(a%b))),
            (Usize(Us64(a)), Usize(Us64(b))) => Ok(Usize(Us64(a%b))),
            (Infer(a), Infer(b)) => Ok(Infer(a%b)),

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
            Isize(Is16(a)) => Ok(Isize(Is16(overflowing!(a.overflowing_shl(b), Op::Shl)))),
            Isize(Is32(a)) => Ok(Isize(Is32(overflowing!(a.overflowing_shl(b), Op::Shl)))),
            Isize(Is64(a)) => Ok(Isize(Is64(overflowing!(a.overflowing_shl(b), Op::Shl)))),
            U8(a) => Ok(U8(overflowing!(a.overflowing_shl(b), Op::Shl))),
            U16(a) => Ok(U16(overflowing!(a.overflowing_shl(b), Op::Shl))),
            U32(a) => Ok(U32(overflowing!(a.overflowing_shl(b), Op::Shl))),
            U64(a) => Ok(U64(overflowing!(a.overflowing_shl(b), Op::Shl))),
            Usize(Us16(a)) => Ok(Usize(Us16(overflowing!(a.overflowing_shl(b), Op::Shl)))),
            Usize(Us32(a)) => Ok(Usize(Us32(overflowing!(a.overflowing_shl(b), Op::Shl)))),
            Usize(Us64(a)) => Ok(Usize(Us64(overflowing!(a.overflowing_shl(b), Op::Shl)))),
            Infer(a) => Ok(Infer(overflowing!(a.overflowing_shl(b), Op::Shl))),
            InferSigned(a) => Ok(InferSigned(overflowing!(a.overflowing_shl(b), Op::Shl))),
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
            Isize(Is16(a)) => Ok(Isize(Is16(overflowing!(a.overflowing_shr(b), Op::Shr)))),
            Isize(Is32(a)) => Ok(Isize(Is32(overflowing!(a.overflowing_shr(b), Op::Shr)))),
            Isize(Is64(a)) => Ok(Isize(Is64(overflowing!(a.overflowing_shr(b), Op::Shr)))),
            U8(a) => Ok(U8(overflowing!(a.overflowing_shr(b), Op::Shr))),
            U16(a) => Ok(U16(overflowing!(a.overflowing_shr(b), Op::Shr))),
            U32(a) => Ok(U32(overflowing!(a.overflowing_shr(b), Op::Shr))),
            U64(a) => Ok(U64(overflowing!(a.overflowing_shr(b), Op::Shr))),
            Usize(Us16(a)) => Ok(Usize(Us16(overflowing!(a.overflowing_shr(b), Op::Shr)))),
            Usize(Us32(a)) => Ok(Usize(Us32(overflowing!(a.overflowing_shr(b), Op::Shr)))),
            Usize(Us64(a)) => Ok(Usize(Us64(overflowing!(a.overflowing_shr(b), Op::Shr)))),
            Infer(a) => Ok(Infer(overflowing!(a.overflowing_shr(b), Op::Shr))),
            InferSigned(a) => Ok(InferSigned(overflowing!(a.overflowing_shr(b), Op::Shr))),
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
            Isize(Is16(a)) => Ok(Isize(Is16(overflowing!(a.overflowing_neg(), Op::Neg)))),
            Isize(Is32(a)) => Ok(Isize(Is32(overflowing!(a.overflowing_neg(), Op::Neg)))),
            Isize(Is64(a)) => Ok(Isize(Is64(overflowing!(a.overflowing_neg(), Op::Neg)))),
            U8(0) => Ok(U8(0)),
            U16(0) => Ok(U16(0)),
            U32(0) => Ok(U32(0)),
            U64(0) => Ok(U64(0)),
            Usize(Us16(0)) => Ok(Usize(Us16(0))),
            Usize(Us32(0)) => Ok(Usize(Us32(0))),
            Usize(Us64(0)) => Ok(Usize(Us64(0))),
            U8(_) => Err(UnsignedNegation),
            U16(_) => Err(UnsignedNegation),
            U32(_) => Err(UnsignedNegation),
            U64(_) => Err(UnsignedNegation),
            Usize(_) => Err(UnsignedNegation),
            Infer(a @ 0...as_u64::I64MAX) => Ok(InferSigned(-(a as i64))),
            Infer(_) => Err(Overflow(Op::Neg)),
            InferSigned(a) => Ok(InferSigned(overflowing!(a.overflowing_neg(), Op::Neg))),
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
            Isize(Is16(a)) => Ok(Isize(Is16(!a))),
            Isize(Is32(a)) => Ok(Isize(Is32(!a))),
            Isize(Is64(a)) => Ok(Isize(Is64(!a))),
            U8(a) => Ok(U8(!a)),
            U16(a) => Ok(U16(!a)),
            U32(a) => Ok(U32(!a)),
            U64(a) => Ok(U64(!a)),
            Usize(Us16(a)) => Ok(Usize(Us16(!a))),
            Usize(Us32(a)) => Ok(Usize(Us32(!a))),
            Usize(Us64(a)) => Ok(Usize(Us64(!a))),
            Infer(a) => Ok(Infer(!a)),
            InferSigned(a) => Ok(InferSigned(!a)),
        }
    }
}
