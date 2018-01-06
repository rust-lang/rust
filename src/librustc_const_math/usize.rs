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
pub enum ConstUsize {
    Us16(u16),
    Us32(u32),
    Us64(u64),
}
pub use self::ConstUsize::*;

impl ::std::fmt::Display for ConstUsize {
    fn fmt(&self, fmt: &mut ::std::fmt::Formatter) -> Result<(), ::std::fmt::Error> {
        write!(fmt, "{}", self.as_u64())
    }
}

impl ConstUsize {
    pub fn as_u64(self) -> u64 {
        match self {
            Us16(i) => i as u64,
            Us32(i) => i as u64,
            Us64(i) => i,
        }
    }
    pub fn new(i: u64, usize_ty: ast::UintTy) -> Result<Self, ConstMathErr> {
        match usize_ty {
            ast::UintTy::U16 if i as u16 as u64 == i => Ok(Us16(i as u16)),
            ast::UintTy::U16 => Err(ULitOutOfRange(ast::UintTy::Usize)),
            ast::UintTy::U32 if i as u32 as u64 == i => Ok(Us32(i as u32)),
            ast::UintTy::U32 => Err(ULitOutOfRange(ast::UintTy::Usize)),
            ast::UintTy::U64 => Ok(Us64(i)),
            _ => unreachable!(),
        }
    }
    pub fn new_truncating(i: u128, usize_ty: ast::UintTy) -> Self {
        match usize_ty {
            ast::UintTy::U16 => Us16(i as u16),
            ast::UintTy::U32 => Us32(i as u32),
            ast::UintTy::U64 => Us64(i as u64),
            _ => unreachable!(),
        }
    }
}
