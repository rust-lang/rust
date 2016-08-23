// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast;

#[derive(Debug, PartialEq, Eq, Clone, RustcEncodable, RustcDecodable)]
pub enum ConstMathErr {
    NotInRange,
    CmpBetweenUnequalTypes,
    UnequalTypes(Op),
    Overflow(Op),
    ShiftNegative,
    DivisionByZero,
    RemainderByZero,
    UnsignedNegation,
    ULitOutOfRange(ast::UintTy),
    LitOutOfRange(ast::IntTy),
}
pub use self::ConstMathErr::*;

#[derive(Debug, PartialEq, Eq, Clone, RustcEncodable, RustcDecodable)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Shr,
    Shl,
    Neg,
    BitAnd,
    BitOr,
    BitXor,
}

impl ConstMathErr {
    pub fn description(&self) -> &'static str {
        use self::Op::*;
        match *self {
            NotInRange => "inferred value out of range",
            CmpBetweenUnequalTypes => "compared two values of different types",
            UnequalTypes(Add) => "tried to add two values of different types",
            UnequalTypes(Sub) => "tried to subtract two values of different types",
            UnequalTypes(Mul) => "tried to multiply two values of different types",
            UnequalTypes(Div) => "tried to divide two values of different types",
            UnequalTypes(Rem) => {
                "tried to calculate the remainder of two values of different types"
            },
            UnequalTypes(BitAnd) => "tried to bitand two values of different types",
            UnequalTypes(BitOr) => "tried to bitor two values of different types",
            UnequalTypes(BitXor) => "tried to xor two values of different types",
            UnequalTypes(_) => unreachable!(),
            Overflow(Add) => "attempt to add with overflow",
            Overflow(Sub) => "attempt to subtract with overflow",
            Overflow(Mul) => "attempt to multiply with overflow",
            Overflow(Div) => "attempt to divide with overflow",
            Overflow(Rem) => "attempt to calculate the remainder with overflow",
            Overflow(Neg) => "attempt to negate with overflow",
            Overflow(Shr) => "attempt to shift right with overflow",
            Overflow(Shl) => "attempt to shift left with overflow",
            Overflow(_) => unreachable!(),
            ShiftNegative => "attempt to shift by a negative amount",
            DivisionByZero => "attempt to divide by zero",
            RemainderByZero => "attempt to calculate the remainder with a divisor of zero",
            UnsignedNegation => "unary negation of unsigned integer",
            ULitOutOfRange(ast::UintTy::U8) => "literal out of range for u8",
            ULitOutOfRange(ast::UintTy::U16) => "literal out of range for u16",
            ULitOutOfRange(ast::UintTy::U32) => "literal out of range for u32",
            ULitOutOfRange(ast::UintTy::U64) => "literal out of range for u64",
            ULitOutOfRange(ast::UintTy::U128) => "literal out of range for u128",
            ULitOutOfRange(ast::UintTy::Us) => "literal out of range for usize",
            LitOutOfRange(ast::IntTy::I8) => "literal out of range for i8",
            LitOutOfRange(ast::IntTy::I16) => "literal out of range for i16",
            LitOutOfRange(ast::IntTy::I32) => "literal out of range for i32",
            LitOutOfRange(ast::IntTy::I64) => "literal out of range for i64",
            LitOutOfRange(ast::IntTy::I128) => "literal out of range for i128",
            LitOutOfRange(ast::IntTy::Is) => "literal out of range for isize",
        }
    }
}
