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
    Is32(i32),
    Is64(i64),
}
pub use self::ConstIsize::*;

impl ConstIsize {
    pub fn as_i64(self, target_int_ty: ast::IntTy) -> i64 {
        match (self, target_int_ty) {
            (Is32(i), ast::IntTy::I32) => i as i64,
            (Is64(i), ast::IntTy::I64) => i,
            _ => panic!("got invalid isize size for target"),
        }
    }
    pub fn new(i: i64, target_int_ty: ast::IntTy) -> Result<Self, ConstMathErr> {
        match target_int_ty {
            ast::IntTy::I32 if i as i32 as i64 == i => Ok(Is32(i as i32)),
            ast::IntTy::I32 => Err(LitOutOfRange(ast::IntTy::Is)),
            ast::IntTy::I64 => Ok(Is64(i)),
            _ => unreachable!(),
        }
    }
}
