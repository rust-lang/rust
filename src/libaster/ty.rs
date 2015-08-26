// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::iter::IntoIterator;

use syntax::ast;
use syntax::codemap::{DUMMY_SP, Span};
use syntax::ptr::P;

use ident::ToIdent;
use invoke::{Invoke, Identity};
use name::ToName;
use path::PathBuilder;
use qpath::QPathBuilder;

//////////////////////////////////////////////////////////////////////////////

pub struct TyBuilder<F=Identity> {
    callback: F,
    span: Span,
}

impl TyBuilder {
    pub fn new() -> Self {
        TyBuilder::new_with_callback(Identity)
    }
}

impl<F> TyBuilder<F>
    where F: Invoke<P<ast::Ty>>,
{
    pub fn new_with_callback(callback: F) -> Self {
        TyBuilder {
            callback: callback,
            span: DUMMY_SP,
        }
    }

    pub fn build(self, ty: P<ast::Ty>) -> F::Result {
        self.callback.invoke(ty)
    }

    pub fn span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    pub fn build_ty_(self, ty_: ast::Ty_) -> F::Result {
        let span = self.span;
        self.build(P(ast::Ty {
            id: ast::DUMMY_NODE_ID,
            node: ty_,
            span: span,
        }))
    }

    pub fn id<I>(self, id: I) -> F::Result
        where I: ToIdent,
    {
        self.path().id(id).build()
    }

    pub fn build_path(self, path: ast::Path) -> F::Result {
        self.build_ty_(ast::Ty_::TyPath(None, path))
    }

    pub fn build_qpath(self, qself: ast::QSelf, path: ast::Path) -> F::Result {
        self.build_ty_(ast::Ty_::TyPath(Some(qself), path))
    }

    pub fn path(self) -> PathBuilder<TyPathBuilder<F>> {
        PathBuilder::new_with_callback(TyPathBuilder(self))
    }

    pub fn qpath(self) -> QPathBuilder<TyQPathBuilder<F>> {
        QPathBuilder::new_with_callback(TyQPathBuilder(self))
    }

    pub fn isize(self) -> F::Result {
        self.id("isize")
    }

    pub fn i8(self) -> F::Result {
        self.id("i8")
    }

    pub fn i16(self) -> F::Result {
        self.id("i16")
    }

    pub fn i32(self) -> F::Result {
        self.id("i32")
    }

    pub fn i64(self) -> F::Result {
        self.id("i64")
    }

    pub fn usize(self) -> F::Result {
        self.id("usize")
    }

    pub fn u8(self) -> F::Result {
        self.id("u8")
    }

    pub fn u16(self) -> F::Result {
        self.id("u16")
    }

    pub fn u32(self) -> F::Result {
        self.id("u32")
    }

    pub fn u64(self) -> F::Result {
        self.id("u64")
    }

    pub fn f32(self) -> F::Result {
        self.id("f32")
    }

    pub fn f64(self) -> F::Result {
        self.id("f64")
    }

    pub fn unit(self) -> F::Result {
        self.tuple().build()
    }

    pub fn tuple(self) -> TyTupleBuilder<F> {
        TyTupleBuilder {
            builder: self,
            tys: vec![],
        }
    }

    pub fn build_slice(self, ty: P<ast::Ty>) -> F::Result {
        self.build_ty_(ast::Ty_::TyVec(ty))
    }

    pub fn slice(self) -> TyBuilder<TySliceBuilder<F>> {
        TyBuilder::new_with_callback(TySliceBuilder(self))
    }

    pub fn ref_(self) -> TyRefBuilder<F> {
        TyRefBuilder {
            builder: self,
            lifetime: None,
            mutability: ast::MutImmutable,
        }
    }

    pub fn infer(self) -> F::Result {
        self.build_ty_(ast::TyInfer)
    }

    pub fn option(self) -> TyBuilder<TyOptionBuilder<F>> {
        TyBuilder::new_with_callback(TyOptionBuilder(self))
    }

    pub fn result(self) -> TyBuilder<TyResultOkBuilder<F>> {
        TyBuilder::new_with_callback(TyResultOkBuilder(self))
    }

    pub fn phantom_data(self) -> TyBuilder<TyPhantomDataBuilder<F>> {
        TyBuilder::new_with_callback(TyPhantomDataBuilder(self))
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct TyPathBuilder<F>(TyBuilder<F>);

impl<F> Invoke<ast::Path> for TyPathBuilder<F>
    where F: Invoke<P<ast::Ty>>,
{
    type Result = F::Result;

    fn invoke(self, path: ast::Path) -> F::Result {
        self.0.build_path(path)
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct TyQPathBuilder<F>(TyBuilder<F>);

impl<F> Invoke<(ast::QSelf, ast::Path)> for TyQPathBuilder<F>
    where F: Invoke<P<ast::Ty>>,
{
    type Result = F::Result;

    fn invoke(self, (qself, path): (ast::QSelf, ast::Path)) -> F::Result {
        self.0.build_qpath(qself, path)
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct TySliceBuilder<F>(TyBuilder<F>);

impl<F> Invoke<P<ast::Ty>> for TySliceBuilder<F>
    where F: Invoke<P<ast::Ty>>,
{
    type Result = F::Result;

    fn invoke(self, ty: P<ast::Ty>) -> F::Result {
        self.0.build_slice(ty)
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct TyRefBuilder<F> {
    builder: TyBuilder<F>,
    lifetime: Option<ast::Lifetime>,
    mutability: ast::Mutability,
}

impl<F> TyRefBuilder<F>
    where F: Invoke<P<ast::Ty>>,
{
    pub fn mut_(mut self) -> Self {
        self.mutability = ast::MutMutable;
        self
    }

    pub fn lifetime<N>(mut self, name: N) -> Self
        where N: ToName,
    {
        self.lifetime = Some(ast::Lifetime {
            id: ast::DUMMY_NODE_ID,
            span: self.builder.span,
            name: name.to_name(),
        });
        self
    }

    pub fn build_ty(self, ty: P<ast::Ty>) -> F::Result {
        let ty = ast::MutTy {
            ty: ty,
            mutbl: self.mutability,
        };
        self.builder.build_ty_(ast::TyRptr(self.lifetime, ty))
    }

    pub fn ty(self) -> TyBuilder<Self> {
        TyBuilder::new_with_callback(self)
    }
}

impl<F> Invoke<P<ast::Ty>> for TyRefBuilder<F>
    where F: Invoke<P<ast::Ty>>,
{
    type Result = F::Result;

    fn invoke(self, ty: P<ast::Ty>) -> F::Result {
        self.build_ty(ty)
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct TyOptionBuilder<F>(TyBuilder<F>);

impl<F> Invoke<P<ast::Ty>> for TyOptionBuilder<F>
    where F: Invoke<P<ast::Ty>>,
{
    type Result = F::Result;

    fn invoke(self, ty: P<ast::Ty>) -> F::Result {
        let path = PathBuilder::new()
            .global()
            .id("std")
            .id("option")
            .segment("Option")
                .with_ty(ty)
                .build()
            .build();

        self.0.build_path(path)
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct TyResultOkBuilder<F>(TyBuilder<F>);

impl<F> Invoke<P<ast::Ty>> for TyResultOkBuilder<F>
    where F: Invoke<P<ast::Ty>>,
{
    type Result = TyBuilder<TyResultErrBuilder<F>>;

    fn invoke(self, ty: P<ast::Ty>) -> TyBuilder<TyResultErrBuilder<F>> {
        TyBuilder::new_with_callback(TyResultErrBuilder(self.0, ty))
    }
}

pub struct TyResultErrBuilder<F>(TyBuilder<F>, P<ast::Ty>);

impl<F> Invoke<P<ast::Ty>> for TyResultErrBuilder<F>
    where F: Invoke<P<ast::Ty>>,
{
    type Result = F::Result;

    fn invoke(self, ty: P<ast::Ty>) -> F::Result {
        let path = PathBuilder::new()
            .global()
            .id("std")
            .id("result")
            .segment("Result")
                .with_ty(self.1)
                .with_ty(ty)
                .build()
            .build();

        self.0.build_path(path)
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct TyPhantomDataBuilder<F>(TyBuilder<F>);

impl<F> Invoke<P<ast::Ty>> for TyPhantomDataBuilder<F>
    where F: Invoke<P<ast::Ty>>,
{
    type Result = F::Result;

    fn invoke(self, ty: P<ast::Ty>) -> F::Result {
        let path = PathBuilder::new()
            .global()
            .id("std")
            .id("marker")
            .segment("PhantomData")
                .with_ty(ty)
                .build()
            .build();

        self.0.build_path(path)
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct TyTupleBuilder<F> {
    builder: TyBuilder<F>,
    tys: Vec<P<ast::Ty>>,
}

impl<F> TyTupleBuilder<F>
    where F: Invoke<P<ast::Ty>>,
{
    pub fn with_tys<I>(mut self, iter: I) -> Self
        where I: IntoIterator<Item=P<ast::Ty>>,
    {
        self.tys.extend(iter);
        self
    }

    pub fn with_ty(mut self, ty: P<ast::Ty>) -> Self {
        self.tys.push(ty);
        self
    }

    pub fn ty(self) -> TyBuilder<Self> {
        TyBuilder::new_with_callback(self)
    }

    pub fn build(self) -> F::Result {
        self.builder.build_ty_(ast::TyTup(self.tys))
    }
}

impl<F> Invoke<P<ast::Ty>> for TyTupleBuilder<F>
    where F: Invoke<P<ast::Ty>>,
{
    type Result = Self;

    fn invoke(self, ty: P<ast::Ty>) -> Self {
        self.with_ty(ty)
    }
}
