// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module is used to store stuff from Rust's AST in a more convenient
//! manner (and with prettier names) before cleaning.

use syntax;
use syntax::codemap::Span;
use syntax::ast;
use syntax::ast::{Ident, NodeId};

pub struct Module {
    name: Option<Ident>,
    attrs: ~[ast::Attribute],
    where: Span,
    structs: ~[Struct],
    enums: ~[Enum],
    fns: ~[Function],
    mods: ~[Module],
    id: NodeId,
    typedefs: ~[Typedef],
    statics: ~[Static],
    traits: ~[Trait],
    vis: ast::Visibility,
    impls: ~[Impl],
    foreigns: ~[ast::ForeignMod],
    view_items: ~[ast::ViewItem],
}

impl Module {
    pub fn new(name: Option<Ident>) -> Module {
        Module {
            name       : name,
            id: 0,
            vis: ast::Private,
            where: syntax::codemap::DUMMY_SP,
            attrs      : ~[],
            structs    : ~[],
            enums      : ~[],
            fns        : ~[],
            mods       : ~[],
            typedefs   : ~[],
            statics    : ~[],
            traits     : ~[],
            impls      : ~[],
            view_items : ~[],
            foreigns   : ~[],
        }
    }
}

#[deriving(ToStr, Clone, Encodable, Decodable)]
pub enum StructType {
    /// A normal struct
    Plain,
    /// A tuple struct
    Tuple,
    /// A newtype struct (tuple struct with one element)
    Newtype,
    /// A unit struct
    Unit
}

pub enum TypeBound {
    RegionBound,
    TraitBound(ast::TraitRef)
}

pub struct Struct {
    vis: ast::Visibility,
    id: NodeId,
    struct_type: StructType,
    name: Ident,
    generics: ast::Generics,
    attrs: ~[ast::Attribute],
    fields: ~[ast::StructField],
    where: Span,
}

pub struct Enum {
    vis: ast::Visibility,
    variants: ~[Variant],
    generics: ast::Generics,
    attrs: ~[ast::Attribute],
    id: NodeId,
    where: Span,
    name: Ident,
}

pub struct Variant {
    name: Ident,
    attrs: ~[ast::Attribute],
    kind: ast::VariantKind,
    id: ast::NodeId,
    vis: ast::Visibility,
    where: Span,
}

pub struct Function {
    decl: ast::FnDecl,
    attrs: ~[ast::Attribute],
    id: NodeId,
    name: Ident,
    vis: ast::Visibility,
    purity: ast::Purity,
    where: Span,
    generics: ast::Generics,
}

pub struct Typedef {
    ty: ast::P<ast::Ty>,
    gen: ast::Generics,
    name: Ident,
    id: ast::NodeId,
    attrs: ~[ast::Attribute],
    where: Span,
    vis: ast::Visibility,
}

pub struct Static {
    type_: ast::P<ast::Ty>,
    mutability: ast::Mutability,
    expr: @ast::Expr,
    name: Ident,
    attrs: ~[ast::Attribute],
    vis: ast::Visibility,
    id: ast::NodeId,
    where: Span,
}

pub struct Trait {
    name: Ident,
    methods: ~[ast::TraitMethod], //should be TraitMethod
    generics: ast::Generics,
    parents: ~[ast::TraitRef],
    attrs: ~[ast::Attribute],
    id: ast::NodeId,
    where: Span,
    vis: ast::Visibility,
}

pub struct Impl {
    generics: ast::Generics,
    trait_: Option<ast::TraitRef>,
    for_: ast::P<ast::Ty>,
    methods: ~[@ast::Method],
    attrs: ~[ast::Attribute],
    where: Span,
    vis: ast::Visibility,
    id: ast::NodeId,
}

pub fn struct_type_from_def(sd: &ast::StructDef) -> StructType {
    if sd.ctor_id.is_some() {
        // We are in a tuple-struct
        match sd.fields.len() {
            0 => Unit,
            1 => Newtype,
            _ => Tuple
        }
    } else {
        Plain
    }
}
