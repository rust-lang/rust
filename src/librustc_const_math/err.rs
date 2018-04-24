// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(Debug, PartialEq, Eq, Clone, RustcEncodable, RustcDecodable)]
pub enum ConstMathErr {
    Overflow(Op),
    DivisionByZero,
    RemainderByZero,
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
            Overflow(Add) => "attempt to add with overflow",
            Overflow(Sub) => "attempt to subtract with overflow",
            Overflow(Mul) => "attempt to multiply with overflow",
            Overflow(Div) => "attempt to divide with overflow",
            Overflow(Rem) => "attempt to calculate the remainder with overflow",
            Overflow(Neg) => "attempt to negate with overflow",
            Overflow(Shr) => "attempt to shift right with overflow",
            Overflow(Shl) => "attempt to shift left with overflow",
            Overflow(_) => unreachable!(),
            DivisionByZero => "attempt to divide by zero",
            RemainderByZero => "attempt to calculate the remainder with a divisor of zero",
        }
    }
}
