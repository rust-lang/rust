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
use syntax::codemap::{DUMMY_SP, Span, respan};
use syntax::ptr::P;

use attr::AttrBuilder;
use ident::ToIdent;
use invoke::{Invoke, Identity};
use ty::TyBuilder;
use struct_def::{StructDefBuilder, StructFieldBuilder};

//////////////////////////////////////////////////////////////////////////////

pub struct VariantBuilder<F=Identity> {
    callback: F,
    span: Span,
    attrs: Vec<ast::Attribute>,
    id: ast::Ident,
}

impl VariantBuilder {
    pub fn new<T>(id: T) -> Self
        where T: ToIdent,
    {
        VariantBuilder::new_with_callback(id, Identity)
    }
}

impl<F> VariantBuilder<F>
    where F: Invoke<P<ast::Variant>>,
{
    pub fn new_with_callback<T>(id: T, callback: F) -> Self
        where T: ToIdent,
    {
        VariantBuilder {
            callback: callback,
            span: DUMMY_SP,
            attrs: vec![],
            id: id.to_ident(),
        }
    }

    pub fn span(mut self, span: Span) -> Self {
        self.span = span;
        self
    }

    pub fn attr(self) -> AttrBuilder<Self> {
        let span = self.span;
        AttrBuilder::new_with_callback(self).span(span)
    }

    pub fn tuple(self) -> VariantTupleBuilder<F> {
        VariantTupleBuilder {
            builder: self,
            args: vec![],
        }
    }

    pub fn struct_(self) -> StructDefBuilder<VariantStructBuilder<F>> {
        StructDefBuilder::new_with_callback(VariantStructBuilder {
            builder: self,
        })
    }

    pub fn build_variant_kind(self, kind: ast::VariantKind) -> F::Result {
        let variant_ = ast::Variant_ {
            name: self.id,
            attrs: self.attrs,
            kind: kind,
            id: ast::DUMMY_NODE_ID,
            disr_expr: None,
            vis: ast::Visibility::Inherited,
        };
        let variant = P(respan(self.span, variant_));
        self.callback.invoke(variant)
    }

    pub fn build_variant_(self, variant: ast::Variant_) -> F::Result {
        let variant = P(respan(self.span, variant));
        self.build(variant)
    }

    pub fn build(self, variant: P<ast::Variant>) -> F::Result {
        self.callback.invoke(variant)
    }
}

impl<F> Invoke<ast::Attribute> for VariantBuilder<F>
    where F: Invoke<P<ast::Variant>>,
{
    type Result = Self;

    fn invoke(mut self, attr: ast::Attribute) -> Self {
        self.attrs.push(attr);
        self
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct VariantTupleBuilder<F> {
    builder: VariantBuilder<F>,
    args: Vec<ast::VariantArg>,
}

impl<F> VariantTupleBuilder<F>
    where F: Invoke<P<ast::Variant>>,
{
    pub fn with_tys<I>(mut self, iter: I) -> Self
        where I: IntoIterator<Item=P<ast::Ty>>,
    {
        for ty in iter {
            self = self.with_ty(ty);
        }
        self
    }

    pub fn with_ty(mut self, ty: P<ast::Ty>) -> Self {
        self.args.push(ast::VariantArg {
            ty: ty,
            id: ast::DUMMY_NODE_ID,
        });
        self
    }

    pub fn ty(self) -> TyBuilder<Self> {
        TyBuilder::new_with_callback(self)
    }

    pub fn build(self) -> F::Result {
        let kind = ast::TupleVariantKind(self.args);
        self.builder.build_variant_kind(kind)
    }
}

impl<F> Invoke<P<ast::Ty>> for VariantTupleBuilder<F>
    where F: Invoke<P<ast::Variant>>,
{
    type Result = Self;

    fn invoke(self, ty: P<ast::Ty>) -> Self {
        self.with_ty(ty)
    }
}

//////////////////////////////////////////////////////////////////////////////

pub struct VariantStructBuilder<F> {
    builder: VariantBuilder<F>,
}

impl<F> VariantStructBuilder<F>
    where F: Invoke<P<ast::Variant>>,
{
    pub fn with_field(self, field: ast::StructField) -> StructDefBuilder<Self> {
        let span = self.builder.span;
        StructDefBuilder::new_with_callback(self).span(span).with_field(field)
    }

    pub fn field<T>(self, id: T) -> StructFieldBuilder<StructDefBuilder<Self>>
        where T: ToIdent,
    {
        let span = self.builder.span;
        StructDefBuilder::new_with_callback(self).span(span).field(id)
    }

    pub fn build(self) -> F::Result {
        StructDefBuilder::new_with_callback(self).build()
    }
}

impl<F> Invoke<P<ast::StructDef>> for VariantStructBuilder<F>
    where F: Invoke<P<ast::Variant>>,
{
    type Result = F::Result;

    fn invoke(self, struct_def: P<ast::StructDef>) -> F::Result {
        let kind = ast::StructVariantKind(struct_def);
        self.builder.build_variant_kind(kind)
    }
}
