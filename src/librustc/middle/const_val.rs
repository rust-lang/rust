// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::parse::token::InternedString;
use syntax::ast;
use std::rc::Rc;
use hir::def_id::DefId;
use std::hash;
use std::mem::transmute;
use rustc_const_math::*;
use self::ConstVal::*;

#[derive(Clone, Debug, RustcEncodable, RustcDecodable)]
pub enum ConstVal {
    Float(f64),
    Integral(ConstInt),
    Str(InternedString),
    ByteStr(Rc<Vec<u8>>),
    Bool(bool),
    Struct(ast::NodeId),
    Tuple(ast::NodeId),
    Function(DefId),
    Array(ast::NodeId, u64),
    Repeat(ast::NodeId, u64),
    Char(char),
    /// A value that only occurs in case `eval_const_expr` reported an error. You should never
    /// handle this case. Its sole purpose is to allow more errors to be reported instead of
    /// causing a fatal error.
    Dummy,
}

impl hash::Hash for ConstVal {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        match *self {
            Float(a) => unsafe { transmute::<_,u64>(a) }.hash(state),
            Integral(a) => a.hash(state),
            Str(ref a) => a.hash(state),
            ByteStr(ref a) => a.hash(state),
            Bool(a) => a.hash(state),
            Struct(a) => a.hash(state),
            Tuple(a) => a.hash(state),
            Function(a) => a.hash(state),
            Array(a, n) => { a.hash(state); n.hash(state) },
            Repeat(a, n) => { a.hash(state); n.hash(state) },
            Char(c) => c.hash(state),
            Dummy => ().hash(state),
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
            (&Integral(a), &Integral(b)) => a == b,
            (&Str(ref a), &Str(ref b)) => a == b,
            (&ByteStr(ref a), &ByteStr(ref b)) => a == b,
            (&Bool(a), &Bool(b)) => a == b,
            (&Struct(a), &Struct(b)) => a == b,
            (&Tuple(a), &Tuple(b)) => a == b,
            (&Function(a), &Function(b)) => a == b,
            (&Array(a, an), &Array(b, bn)) => (a == b) && (an == bn),
            (&Repeat(a, an), &Repeat(b, bn)) => (a == b) && (an == bn),
            (&Char(a), &Char(b)) => a == b,
            (&Dummy, &Dummy) => true, // FIXME: should this be false?
            _ => false,
        }
    }
}

impl Eq for ConstVal { }

impl ConstVal {
    pub fn description(&self) -> &'static str {
        match *self {
            Float(_) => "float",
            Integral(i) => i.description(),
            Str(_) => "string literal",
            ByteStr(_) => "byte string literal",
            Bool(_) => "boolean",
            Struct(_) => "struct",
            Tuple(_) => "tuple",
            Function(_) => "function definition",
            Array(..) => "array",
            Repeat(..) => "repeat",
            Char(..) => "char",
            Dummy => "dummy value",
        }
    }
}
