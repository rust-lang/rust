// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module contains `HashStable` implementations for various data types
//! from `rustc_const_math` in no particular order.

impl_stable_hash_for!(struct ::rustc_const_math::ConstFloat {
    ty,
    bits
});

impl_stable_hash_for!(enum ::rustc_const_math::ConstMathErr {
    NotInRange,
    CmpBetweenUnequalTypes,
    UnequalTypes(op),
    Overflow(op),
    ShiftNegative,
    DivisionByZero,
    RemainderByZero,
    UnsignedNegation,
    ULitOutOfRange(int_ty),
    LitOutOfRange(int_ty)
});

impl_stable_hash_for!(enum ::rustc_const_math::Op {
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
    BitXor
});
