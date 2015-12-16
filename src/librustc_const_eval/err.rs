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

#[derive(Debug, PartialEq, Eq, Clone)]
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

#[derive(Debug, PartialEq, Eq, Clone)]
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
            CmpBetweenUnequalTypes => "compared two integrals of different types",
            UnequalTypes(Add) => "tried to add two integrals of different types",
            UnequalTypes(Sub) => "tried to subtract two integrals of different types",
            UnequalTypes(Mul) => "tried to multiply two integrals of different types",
            UnequalTypes(Div) => "tried to divide two integrals of different types",
            UnequalTypes(Rem) => {
                "tried to calculate the remainder of two integrals of different types"
            },
            UnequalTypes(BitAnd) => "tried to bitand two integrals of different types",
            UnequalTypes(BitOr) => "tried to bitor two integrals of different types",
            UnequalTypes(BitXor) => "tried to xor two integrals of different types",
            UnequalTypes(_) => unreachable!(),
            Overflow(Add) => "attempted to add with overflow",
            Overflow(Sub) => "attempted to subtract with overflow",
            Overflow(Mul) => "attempted to multiply with overflow",
            Overflow(Div) => "attempted to divide with overflow",
            Overflow(Rem) => "attempted to calculate the remainder with overflow",
            Overflow(Neg) => "attempted to negate with overflow",
            Overflow(Shr) => "attempted to shift right with overflow",
            Overflow(Shl) => "attempted to shift left with overflow",
            Overflow(_) => unreachable!(),
            ShiftNegative => "attempted to shift by a negative amount",
            DivisionByZero => "attempted to divide by zero",
            RemainderByZero => "attempted to calculate the remainder with a divisor of zero",
            UnsignedNegation => "unary negation of unsigned integer",
            ULitOutOfRange(ast::UintTy::U8) => "literal out of range for u8",
            ULitOutOfRange(ast::UintTy::U16) => "literal out of range for u16",
            ULitOutOfRange(ast::UintTy::U32) => "literal out of range for u32",
            ULitOutOfRange(ast::UintTy::U64) => "literal out of range for u64",
            ULitOutOfRange(ast::UintTy::Us) => "literal out of range for usize",
            LitOutOfRange(ast::IntTy::I8) => "literal out of range for i8",
            LitOutOfRange(ast::IntTy::I16) => "literal out of range for i16",
            LitOutOfRange(ast::IntTy::I32) => "literal out of range for i32",
            LitOutOfRange(ast::IntTy::I64) => "literal out of range for i64",
            LitOutOfRange(ast::IntTy::Is) => "literal out of range for isize",
        }
    }
}
