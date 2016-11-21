// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::symbol::InternedString;
use syntax::ast;
use std::rc::Rc;
use hir::def_id::DefId;
use rustc_const_math::*;
use self::ConstVal::*;

#[derive(Clone, Debug, Hash, RustcEncodable, RustcDecodable, Eq, PartialEq)]
pub enum ConstVal {
    Float(ConstFloat),
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

impl ConstVal {
    pub fn description(&self) -> &'static str {
        match *self {
            Float(f) => f.description(),
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
