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
use rustc_i128::i128;

/// Depending on the target only one variant is ever used in a compilation.
/// Anything else is an error. This invariant is checked at several locations
#[derive(Copy, Clone, Debug, RustcEncodable, RustcDecodable, Hash, Eq, PartialEq)]
pub enum ConstIsize {
    Is16(i16),
    Is32(i32),
    Is64(i64),
}
pub use self::ConstIsize::*;

impl ConstIsize {
    pub fn as_i64(self, target_int_ty: ast::IntTy) -> i64 {
        match (self, target_int_ty) {
            (Is16(i), ast::IntTy::I16) => i as i64,
            (Is32(i), ast::IntTy::I32) => i as i64,
            (Is64(i), ast::IntTy::I64) => i,
            _ => panic!("unable to convert self ({:?}) to target isize ({:?})",
                        self, target_int_ty),
        }
    }
    pub fn new(i: i64, target_int_ty: ast::IntTy) -> Result<Self, ConstMathErr> {
        match target_int_ty {
            ast::IntTy::I16 if i as i16 as i64 == i => Ok(Is16(i as i16)),
            ast::IntTy::I16 => Err(LitOutOfRange(ast::IntTy::Is)),
            ast::IntTy::I32 if i as i32 as i64 == i => Ok(Is32(i as i32)),
            ast::IntTy::I32 => Err(LitOutOfRange(ast::IntTy::Is)),
            ast::IntTy::I64 => Ok(Is64(i)),
            _ => unreachable!(),
        }
    }
    pub fn new_truncating(i: i128, target_int_ty: ast::IntTy) -> Self {
        match target_int_ty {
            ast::IntTy::I16 => Is16(i as i16),
            ast::IntTy::I32 => Is32(i as i32),
            ast::IntTy::I64 => Is64(i as i64),
            _ => unreachable!(),
        }
    }
}
