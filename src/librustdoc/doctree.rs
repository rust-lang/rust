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

use std::gc::Gc;

pub struct Module {
    pub name: Option<Ident>,
    pub attrs: Vec<ast::Attribute>,
    pub where_outer: Span,
    pub where_inner: Span,
    pub structs: Vec<Struct>,
    pub enums: Vec<Enum>,
    pub fns: Vec<Function>,
    pub mods: Vec<Module>,
    pub id: NodeId,
    pub typedefs: Vec<Typedef>,
    pub statics: Vec<Static>,
    pub traits: Vec<Trait>,
    pub vis: ast::Visibility,
    pub impls: Vec<Impl>,
    pub foreigns: Vec<ast::ForeignMod>,
    pub view_items: Vec<ast::ViewItem>,
    pub macros: Vec<Macro>,
    pub is_crate: bool,
}

impl Module {
    pub fn new(name: Option<Ident>) -> Module {
        Module {
            name       : name,
            id: 0,
            vis: ast::Inherited,
            where_outer: syntax::codemap::DUMMY_SP,
            where_inner: syntax::codemap::DUMMY_SP,
            attrs      : Vec::new(),
            structs    : Vec::new(),
            enums      : Vec::new(),
            fns        : Vec::new(),
            mods       : Vec::new(),
            typedefs   : Vec::new(),
            statics    : Vec::new(),
            traits     : Vec::new(),
            impls      : Vec::new(),
            view_items : Vec::new(),
            foreigns   : Vec::new(),
            macros     : Vec::new(),
            is_crate   : false,
        }
    }
}

#[deriving(Show, Clone, Encodable, Decodable)]
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
    pub vis: ast::Visibility,
    pub id: NodeId,
    pub struct_type: StructType,
    pub name: Ident,
    pub generics: ast::Generics,
    pub attrs: Vec<ast::Attribute>,
    pub fields: Vec<ast::StructField>,
    pub where: Span,
}

pub struct Enum {
    pub vis: ast::Visibility,
    pub variants: Vec<Variant>,
    pub generics: ast::Generics,
    pub attrs: Vec<ast::Attribute>,
    pub id: NodeId,
    pub where: Span,
    pub name: Ident,
}

pub struct Variant {
    pub name: Ident,
    pub attrs: Vec<ast::Attribute>,
    pub kind: ast::VariantKind,
    pub id: ast::NodeId,
    pub vis: ast::Visibility,
    pub where: Span,
}

pub struct Function {
    pub decl: ast::FnDecl,
    pub attrs: Vec<ast::Attribute>,
    pub id: NodeId,
    pub name: Ident,
    pub vis: ast::Visibility,
    pub fn_style: ast::FnStyle,
    pub where: Span,
    pub generics: ast::Generics,
}

pub struct Typedef {
    pub ty: ast::P<ast::Ty>,
    pub gen: ast::Generics,
    pub name: Ident,
    pub id: ast::NodeId,
    pub attrs: Vec<ast::Attribute>,
    pub where: Span,
    pub vis: ast::Visibility,
}

pub struct Static {
    pub type_: ast::P<ast::Ty>,
    pub mutability: ast::Mutability,
    pub expr: Gc<ast::Expr>,
    pub name: Ident,
    pub attrs: Vec<ast::Attribute>,
    pub vis: ast::Visibility,
    pub id: ast::NodeId,
    pub where: Span,
}

pub struct Trait {
    pub name: Ident,
    pub methods: Vec<ast::TraitMethod>, //should be TraitMethod
    pub generics: ast::Generics,
    pub parents: Vec<ast::TraitRef>,
    pub attrs: Vec<ast::Attribute>,
    pub id: ast::NodeId,
    pub where: Span,
    pub vis: ast::Visibility,
}

pub struct Impl {
    pub generics: ast::Generics,
    pub trait_: Option<ast::TraitRef>,
    pub for_: ast::P<ast::Ty>,
    pub methods: Vec<Gc<ast::Method>>,
    pub attrs: Vec<ast::Attribute>,
    pub where: Span,
    pub vis: ast::Visibility,
    pub id: ast::NodeId,
}

pub struct Macro {
    pub name: Ident,
    pub id: ast::NodeId,
    pub attrs: Vec<ast::Attribute>,
    pub where: Span,
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
