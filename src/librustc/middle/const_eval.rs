// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::ConstVal::*;

use middle::def_id::DefId;

use syntax::ast;
use syntax::parse::token::InternedString;

use std::hash;
use std::mem::transmute;
use std::rc::Rc;

#[derive(Clone, Debug, RustcEncodable, RustcDecodable)]
pub enum ConstVal {
    Float(f64),
    Int(i64),
    Uint(u64),
    Str(InternedString),
    ByteStr(Rc<Vec<u8>>),
    Bool(bool),
    Struct(ast::NodeId),
    Tuple(ast::NodeId),
    Function(DefId),
    Array(ast::NodeId, u64),
    Repeat(ast::NodeId, u64),
}

impl hash::Hash for ConstVal {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        match *self {
            Float(a) => unsafe { transmute::<_,u64>(a) }.hash(state),
            Int(a) => a.hash(state),
            Uint(a) => a.hash(state),
            Str(ref a) => a.hash(state),
            ByteStr(ref a) => a.hash(state),
            Bool(a) => a.hash(state),
            Struct(a) => a.hash(state),
            Tuple(a) => a.hash(state),
            Function(a) => a.hash(state),
            Array(a, n) => { a.hash(state); n.hash(state) },
            Repeat(a, n) => { a.hash(state); n.hash(state) },
        }
    }
}

/// Note that equality for `ConstVal` means that the it is the same
/// constant, not that the rust values are equal. In particular, `NaN
/// == NaN` (at least if it's the same NaN; distinct encodings for NaN
/// are considering unequal).
impl PartialEq for ConstVal {
    fn eq(&self, other: &ConstVal) -> bool {
        match (self, other) {
            (&Float(a), &Float(b)) => unsafe{transmute::<_,u64>(a) == transmute::<_,u64>(b)},
            (&Int(a), &Int(b)) => a == b,
            (&Uint(a), &Uint(b)) => a == b,
            (&Str(ref a), &Str(ref b)) => a == b,
            (&ByteStr(ref a), &ByteStr(ref b)) => a == b,
            (&Bool(a), &Bool(b)) => a == b,
            (&Struct(a), &Struct(b)) => a == b,
            (&Tuple(a), &Tuple(b)) => a == b,
            (&Function(a), &Function(b)) => a == b,
            (&Array(a, an), &Array(b, bn)) => (a == b) && (an == bn),
            (&Repeat(a, an), &Repeat(b, bn)) => (a == b) && (an == bn),
            _ => false,
        }
    }
}

impl Eq for ConstVal { }

impl ConstVal {
    pub fn description(&self) -> &'static str {
        match *self {
            Float(_) => "float",
            Int(i) if i < 0 => "negative integer",
            Int(_) => "positive integer",
            Uint(_) => "unsigned integer",
            Str(_) => "string literal",
            ByteStr(_) => "byte string literal",
            Bool(_) => "boolean",
            Struct(_) => "struct",
            Tuple(_) => "tuple",
            Function(_) => "function definition",
            Array(..) => "array",
            Repeat(..) => "repeat",
        }
    }
}
