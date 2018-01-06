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
use super::err::*;

/// Depending on the target only one variant is ever used in a compilation.
/// Anything else is an error. This invariant is checked at several locations
#[derive(Copy, Clone, Debug, RustcEncodable, RustcDecodable, Hash, Eq, PartialEq)]
pub enum ConstIsize {
    Is16(i16),
    Is32(i32),
    Is64(i64),
}
pub use self::ConstIsize::*;

impl ::std::fmt::Display for ConstIsize {
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        write!(fmt, "{}", self.as_i64())
    }
}

impl ConstIsize {
    pub fn as_i64(self) -> i64 {
        match self {
            Is16(i) => i as i64,
            Is32(i) => i as i64,
            Is64(i) => i,
        }
    }
    pub fn new(i: i64, isize_ty: ast::IntTy) -> Result<Self, ConstMathErr> {
        match isize_ty {
            ast::IntTy::I16 if i as i16 as i64 == i => Ok(Is16(i as i16)),
            ast::IntTy::I16 => Err(LitOutOfRange(ast::IntTy::Isize)),
            ast::IntTy::I32 if i as i32 as i64 == i => Ok(Is32(i as i32)),
            ast::IntTy::I32 => Err(LitOutOfRange(ast::IntTy::Isize)),
            ast::IntTy::I64 => Ok(Is64(i)),
            _ => unreachable!(),
        }
    }
    pub fn new_truncating(i: i128, isize_ty: ast::IntTy) -> Self {
        match isize_ty {
            ast::IntTy::I16 => Is16(i as i16),
            ast::IntTy::I32 => Is32(i as i32),
            ast::IntTy::I64 => Is64(i as i64),
            _ => unreachable!(),
        }
    }
}
