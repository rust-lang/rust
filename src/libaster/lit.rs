// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast;
use syntax::codemap::{DUMMY_SP, Span};
use syntax::ptr::P;

use invoke::{Invoke, Identity};

use str::ToInternedString;

//////////////////////////////////////////////////////////////////////////////

pub struct LitBuilder<F=Identity> {
    callback: F,
    span: Span,
}

impl LitBuilder {
    pub fn new() -> LitBuilder {
        LitBuilder::new_with_callback(Identity)
    }
}

impl<F> LitBuilder<F>
    where F: Invoke<P<ast::Lit>>,
{
    pub fn new_with_callback(callback: F) -> Self {
        LitBuilder {
            callback: callback,
            span: DUMMY_SP,
        }
    }

    pub fn span(mut self, span: Span) -> LitBuilder<F> {
        self.span = span;
        self
    }

    pub fn build_lit(self, lit: ast::Lit_) -> F::Result {
        self.callback.invoke(P(ast::Lit {
            span: self.span,
            node: lit,
        }))
    }

    pub fn bool(self, value: bool) -> F::Result {
        self.build_lit(ast::LitBool(value))
    }

    pub fn int(self, value: i64) -> F::Result {
        let sign = ast::Sign::new(value);
        self.build_lit(ast::LitInt(value as u64, ast::UnsuffixedIntLit(sign)))
    }

    fn build_int(self, value: i64, ty: ast::IntTy) -> F::Result {
        let sign = ast::Sign::new(value);
        self.build_lit(ast::LitInt(value as u64, ast::LitIntType::SignedIntLit(ty, sign)))
    }

    pub fn isize(self, value: isize) -> F::Result {
        self.build_int(value as i64, ast::IntTy::TyIs)
    }

    pub fn i8(self, value: i8) -> F::Result {
        self.build_int(value as i64, ast::IntTy::TyI8)
    }

    pub fn i16(self, value: i16) -> F::Result {
        self.build_int(value as i64, ast::IntTy::TyI16)
    }

    pub fn i32(self, value: i32) -> F::Result {
        self.build_int(value as i64, ast::IntTy::TyI32)
    }

    pub fn i64(self, value: i64) -> F::Result {
        self.build_int(value, ast::IntTy::TyI64)
    }

    fn build_uint(self, value: u64, ty: ast::UintTy) -> F::Result {
        self.build_lit(ast::LitInt(value, ast::LitIntType::UnsignedIntLit(ty)))
    }

    pub fn usize(self, value: usize) -> F::Result {
        self.build_uint(value as u64, ast::UintTy::TyUs)
    }

    pub fn u8(self, value: u8) -> F::Result {
        self.build_uint(value as u64, ast::UintTy::TyU8)
    }

    pub fn u16(self, value: u16) -> F::Result {
        self.build_uint(value as u64, ast::UintTy::TyU16)
    }

    pub fn u32(self, value: u32) -> F::Result {
        self.build_uint(value as u64, ast::UintTy::TyU32)
    }

    pub fn u64(self, value: u64) -> F::Result {
        self.build_uint(value, ast::UintTy::TyU64)
    }

    fn build_float<S>(self, value: S, ty: ast::FloatTy) -> F::Result
        where S: ToInternedString,
    {
        self.build_lit(ast::LitFloat(value.to_interned_string(), ty))
    }

    pub fn f32<S>(self, value: S) -> F::Result
        where S: ToInternedString,
    {
        self.build_float(value, ast::FloatTy::TyF32)
    }

    pub fn f64<S>(self, value: S) -> F::Result
        where S: ToInternedString,
    {
        self.build_float(value, ast::FloatTy::TyF64)
    }

    pub fn str<S>(self, value: S) -> F::Result
        where S: ToInternedString,
    {
        let value = value.to_interned_string();
        self.build_lit(ast::LitStr(value, ast::CookedStr))
    }
}
