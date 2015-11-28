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
pub use self::StructType::*;
pub use self::TypeBound::*;

use syntax;
use syntax::codemap::Span;
use syntax::abi;
use syntax::ast;
use syntax::ast::{Name, NodeId};
use syntax::attr;
use syntax::ptr::P;
use rustc_front::hir;
use rustc_front::hir::Vec;

use std::vec;

pub struct Module {
    pub name: Option<Name>,
    pub attrs: Vec<ast::Attribute>,
    pub where_outer: Span,
    pub where_inner: Span,
    pub extern_crates: vec::Vec<ExternCrate>,
    pub imports: vec::Vec<Import>,
    pub structs: vec::Vec<Struct>,
    pub enums: vec::Vec<Enum>,
    pub fns: vec::Vec<Function>,
    pub mods: vec::Vec<Module>,
    pub id: NodeId,
    pub typedefs: vec::Vec<Typedef>,
    pub statics: vec::Vec<Static>,
    pub constants: vec::Vec<Constant>,
    pub traits: vec::Vec<Trait>,
    pub vis: hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub impls: vec::Vec<Impl>,
    pub def_traits: vec::Vec<DefaultImpl>,
    pub foreigns: vec::Vec<hir::ForeignMod>,
    pub macros: vec::Vec<Macro>,
    pub is_crate: bool,
}

impl Module {
    pub fn new(name: Option<Name>) -> Module {
        Module {
            name       : name,
            id: 0,
            vis: hir::Inherited,
            stab: None,
            where_outer: syntax::codemap::DUMMY_SP,
            where_inner: syntax::codemap::DUMMY_SP,
            attrs      : Vec::new(),
            extern_crates: vec::Vec::new(),
            imports    : vec::Vec::new(),
            structs    : vec::Vec::new(),
            enums      : vec::Vec::new(),
            fns        : vec::Vec::new(),
            mods       : vec::Vec::new(),
            typedefs   : vec::Vec::new(),
            statics    : vec::Vec::new(),
            constants  : vec::Vec::new(),
            traits     : vec::Vec::new(),
            impls      : vec::Vec::new(),
            def_traits : vec::Vec::new(),
            foreigns   : vec::Vec::new(),
            macros     : vec::Vec::new(),
            is_crate   : false,
        }
    }
}

#[derive(Debug, Clone, RustcEncodable, RustcDecodable, Copy)]
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
    TraitBound(hir::TraitRef)
}

pub struct Struct {
    pub vis: hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub id: NodeId,
    pub struct_type: StructType,
    pub name: Name,
    pub generics: hir::Generics,
    pub attrs: Vec<ast::Attribute>,
    pub fields: Vec<hir::StructField>,
    pub whence: Span,
}

pub struct Enum {
    pub vis: hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub variants: Vec<Variant>,
    pub generics: hir::Generics,
    pub attrs: Vec<ast::Attribute>,
    pub id: NodeId,
    pub whence: Span,
    pub name: Name,
}

pub struct Variant {
    pub name: Name,
    pub attrs: Vec<ast::Attribute>,
    pub def: hir::VariantData,
    pub stab: Option<attr::Stability>,
    pub whence: Span,
}

pub struct Function {
    pub decl: hir::FnDecl,
    pub attrs: Vec<ast::Attribute>,
    pub id: NodeId,
    pub name: Name,
    pub vis: hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub unsafety: hir::Unsafety,
    pub constness: hir::Constness,
    pub whence: Span,
    pub generics: hir::Generics,
    pub abi: abi::Abi,
}

pub struct Typedef {
    pub ty: P<hir::Ty>,
    pub gen: hir::Generics,
    pub name: Name,
    pub id: ast::NodeId,
    pub attrs: Vec<ast::Attribute>,
    pub whence: Span,
    pub vis: hir::Visibility,
    pub stab: Option<attr::Stability>,
}

#[derive(Debug)]
pub struct Static {
    pub type_: P<hir::Ty>,
    pub mutability: hir::Mutability,
    pub expr: P<hir::Expr>,
    pub name: Name,
    pub attrs: Vec<ast::Attribute>,
    pub vis: hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub id: ast::NodeId,
    pub whence: Span,
}

pub struct Constant {
    pub type_: P<hir::Ty>,
    pub expr: P<hir::Expr>,
    pub name: Name,
    pub attrs: Vec<ast::Attribute>,
    pub vis: hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub id: ast::NodeId,
    pub whence: Span,
}

pub struct Trait {
    pub unsafety: hir::Unsafety,
    pub name: Name,
    pub items: Vec<hir::TraitItem>,
    pub generics: hir::Generics,
    pub bounds: Vec<hir::TyParamBound>,
    pub attrs: Vec<ast::Attribute>,
    pub id: ast::NodeId,
    pub whence: Span,
    pub vis: hir::Visibility,
    pub stab: Option<attr::Stability>,
}

pub struct Impl {
    pub unsafety: hir::Unsafety,
    pub polarity: hir::ImplPolarity,
    pub generics: hir::Generics,
    pub trait_: Option<hir::TraitRef>,
    pub for_: P<hir::Ty>,
    pub items: Vec<hir::ImplItem>,
    pub attrs: Vec<ast::Attribute>,
    pub whence: Span,
    pub vis: hir::Visibility,
    pub stab: Option<attr::Stability>,
    pub id: ast::NodeId,
}

pub struct DefaultImpl {
    pub unsafety: hir::Unsafety,
    pub trait_: hir::TraitRef,
    pub id: ast::NodeId,
    pub attrs: Vec<ast::Attribute>,
    pub whence: Span,
}

pub struct Macro {
    pub name: Name,
    pub id: ast::NodeId,
    pub attrs: Vec<ast::Attribute>,
    pub whence: Span,
    pub matchers: Vec<Span>,
    pub stab: Option<attr::Stability>,
    pub imported_from: Option<Name>,
}

pub struct ExternCrate {
    pub name: Name,
    pub path: Option<String>,
    pub vis: hir::Visibility,
    pub attrs: Vec<ast::Attribute>,
    pub whence: Span,
}

pub struct Import {
    pub id: NodeId,
    pub vis: hir::Visibility,
    pub attrs: Vec<ast::Attribute>,
    pub node: hir::ViewPath_,
    pub whence: Span,
}

pub fn struct_type_from_def(sd: &hir::VariantData) -> StructType {
    if !sd.is_struct() {
        // We are in a tuple-struct
        match sd.fields().len() {
            0 => Unit,
            1 => Newtype,
            _ => Tuple
        }
    } else {
        Plain
    }
}
