// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Structs representing the analysis data from a crate.
//!
//! The `Dump` trait can be used together with `DumpVisitor` in order to
//! retrieve the data from a crate.

use rustc::hir;
use rustc::hir::def_id::{CrateNum, DefId};
use syntax::ast::{self, NodeId};
use syntax_pos::Span;

pub struct CrateData {
    pub name: String,
    pub number: u32,
    pub span: Span,
}

/// Data for any entity in the Rust language. The actual data contained varies
/// with the kind of entity being queried. See the nested structs for details.
#[derive(Debug, RustcEncodable)]
pub enum Data {
    /// Data for Enums.
    EnumData(EnumData),
    /// Data for extern crates.
    ExternCrateData(ExternCrateData),
    /// Data about a function call.
    FunctionCallData(FunctionCallData),
    /// Data for all kinds of functions and methods.
    FunctionData(FunctionData),
    /// Data about a function ref.
    FunctionRefData(FunctionRefData),
    /// Data for impls.
    ImplData(ImplData2),
    /// Data for trait inheritance.
    InheritanceData(InheritanceData),
    /// Data about a macro declaration.
    MacroData(MacroData),
    /// Data about a macro use.
    MacroUseData(MacroUseData),
    /// Data about a method call.
    MethodCallData(MethodCallData),
    /// Data for method declarations (methods with a body are treated as functions).
    MethodData(MethodData),
    /// Data for modules.
    ModData(ModData),
    /// Data for a reference to a module.
    ModRefData(ModRefData),
    /// Data for a struct declaration.
    StructData(StructData),
    /// Data for a struct variant.
    StructVariantDat(StructVariantData),
    /// Data for a trait declaration.
    TraitData(TraitData),
    /// Data for a tuple variant.
    TupleVariantData(TupleVariantData),
    /// Data for a typedef.
    TypeDefData(TypeDefData),
    /// Data for a reference to a type or trait.
    TypeRefData(TypeRefData),
    /// Data for a use statement.
    UseData(UseData),
    /// Data for a global use statement.
    UseGlobData(UseGlobData),
    /// Data for local and global variables (consts and statics), and fields.
    VariableData(VariableData),
    /// Data for the use of some variable (e.g., the use of a local variable, which
    /// will refere to that variables declaration).
    VariableRefData(VariableRefData),
}

#[derive(Eq, PartialEq, Clone, Copy, Debug, RustcEncodable)]
pub enum Visibility {
    Public,
    Restricted,
    Inherited,
}

impl<'a> From<&'a ast::Visibility> for Visibility {
    fn from(v: &'a ast::Visibility) -> Visibility {
        match *v {
            ast::Visibility::Public => Visibility::Public,
            ast::Visibility::Crate(_) => Visibility::Restricted,
            ast::Visibility::Restricted { .. } => Visibility::Restricted,
            ast::Visibility::Inherited => Visibility::Inherited,
        }
    }
}

impl<'a> From<&'a hir::Visibility> for Visibility {
    fn from(v: &'a hir::Visibility) -> Visibility {
        match *v {
            hir::Visibility::Public => Visibility::Public,
            hir::Visibility::Crate => Visibility::Restricted,
            hir::Visibility::Restricted { .. } => Visibility::Restricted,
            hir::Visibility::Inherited => Visibility::Inherited,
        }
    }
}

/// Data for the prelude of a crate.
#[derive(Debug, RustcEncodable)]
pub struct CratePreludeData {
    pub crate_name: String,
    pub crate_root: String,
    pub external_crates: Vec<ExternalCrateData>,
    pub span: Span,
}

/// Data for external crates in the prelude of a crate.
#[derive(Debug, RustcEncodable)]
pub struct ExternalCrateData {
    pub name: String,
    pub num: CrateNum,
    pub file_name: String,
}

/// Data for enum declarations.
#[derive(Clone, Debug, RustcEncodable)]
pub struct EnumData {
    pub id: NodeId,
    pub name: String,
    pub value: String,
    pub qualname: String,
    pub span: Span,
    pub scope: NodeId,
    pub variants: Vec<NodeId>,
    pub visibility: Visibility,
    pub docs: String,
    pub sig: Signature,
}

/// Data for extern crates.
#[derive(Debug, RustcEncodable)]
pub struct ExternCrateData {
    pub id: NodeId,
    pub name: String,
    pub crate_num: CrateNum,
    pub location: String,
    pub span: Span,
    pub scope: NodeId,
}

/// Data about a function call.
#[derive(Debug, RustcEncodable)]
pub struct FunctionCallData {
    pub span: Span,
    pub scope: NodeId,
    pub ref_id: DefId,
}

/// Data for all kinds of functions and methods.
#[derive(Clone, Debug, RustcEncodable)]
pub struct FunctionData {
    pub id: NodeId,
    pub name: String,
    pub qualname: String,
    pub declaration: Option<DefId>,
    pub span: Span,
    pub scope: NodeId,
    pub value: String,
    pub visibility: Visibility,
    pub parent: Option<DefId>,
    pub docs: String,
    pub sig: Signature,
}

/// Data about a function call.
#[derive(Debug, RustcEncodable)]
pub struct FunctionRefData {
    pub span: Span,
    pub scope: NodeId,
    pub ref_id: DefId,
}

#[derive(Debug, RustcEncodable)]
pub struct ImplData {
    pub id: NodeId,
    pub span: Span,
    pub scope: NodeId,
    pub trait_ref: Option<DefId>,
    pub self_ref: Option<DefId>,
}

#[derive(Debug, RustcEncodable)]
// FIXME: this struct should not exist. However, removing it requires heavy
// refactoring of dump_visitor.rs. See PR 31838 for more info.
pub struct ImplData2 {
    pub id: NodeId,
    pub span: Span,
    pub scope: NodeId,
    // FIXME: I'm not really sure inline data is the best way to do this. Seems
    // OK in this case, but generalising leads to returning chunks of AST, which
    // feels wrong.
    pub trait_ref: Option<TypeRefData>,
    pub self_ref: Option<TypeRefData>,
}

#[derive(Debug, RustcEncodable)]
pub struct InheritanceData {
    pub span: Span,
    pub base_id: DefId,
    pub deriv_id: NodeId
}

/// Data about a macro declaration.
#[derive(Debug, RustcEncodable)]
pub struct MacroData {
    pub span: Span,
    pub name: String,
    pub qualname: String,
    pub docs: String,
}

/// Data about a macro use.
#[derive(Debug, RustcEncodable)]
pub struct MacroUseData {
    pub span: Span,
    pub name: String,
    pub qualname: String,
    // Because macro expansion happens before ref-ids are determined,
    // we use the callee span to reference the associated macro definition.
    pub callee_span: Span,
    pub scope: NodeId,
    pub imported: bool,
}

/// Data about a method call.
#[derive(Debug, RustcEncodable)]
pub struct MethodCallData {
    pub span: Span,
    pub scope: NodeId,
    pub ref_id: Option<DefId>,
    pub decl_id: Option<DefId>,
}

/// Data for method declarations (methods with a body are treated as functions).
#[derive(Clone, Debug, RustcEncodable)]
pub struct MethodData {
    pub id: NodeId,
    pub name: String,
    pub qualname: String,
    pub span: Span,
    pub scope: NodeId,
    pub value: String,
    pub decl_id: Option<DefId>,
    pub parent: Option<DefId>,
    pub visibility: Visibility,
    pub docs: String,
    pub sig: Signature,
}

/// Data for modules.
#[derive(Debug, RustcEncodable)]
pub struct ModData {
    pub id: NodeId,
    pub name: String,
    pub qualname: String,
    pub span: Span,
    pub scope: NodeId,
    pub filename: String,
    pub items: Vec<NodeId>,
    pub visibility: Visibility,
    pub docs: String,
    pub sig: Signature,
}

/// Data for a reference to a module.
#[derive(Debug, RustcEncodable)]
pub struct ModRefData {
    pub span: Span,
    pub scope: NodeId,
    pub ref_id: Option<DefId>,
    pub qualname: String
}

#[derive(Debug, RustcEncodable)]
pub struct StructData {
    pub span: Span,
    pub name: String,
    pub id: NodeId,
    pub ctor_id: NodeId,
    pub qualname: String,
    pub scope: NodeId,
    pub value: String,
    pub fields: Vec<NodeId>,
    pub visibility: Visibility,
    pub docs: String,
    pub sig: Signature,
}

#[derive(Debug, RustcEncodable)]
pub struct StructVariantData {
    pub span: Span,
    pub name: String,
    pub id: NodeId,
    pub qualname: String,
    pub type_value: String,
    pub value: String,
    pub scope: NodeId,
    pub parent: Option<DefId>,
    pub docs: String,
    pub sig: Signature,
}

#[derive(Debug, RustcEncodable)]
pub struct TraitData {
    pub span: Span,
    pub id: NodeId,
    pub name: String,
    pub qualname: String,
    pub scope: NodeId,
    pub value: String,
    pub items: Vec<NodeId>,
    pub visibility: Visibility,
    pub docs: String,
    pub sig: Signature,
}

#[derive(Debug, RustcEncodable)]
pub struct TupleVariantData {
    pub span: Span,
    pub id: NodeId,
    pub name: String,
    pub qualname: String,
    pub type_value: String,
    pub value: String,
    pub scope: NodeId,
    pub parent: Option<DefId>,
    pub docs: String,
    pub sig: Signature,
}

/// Data for a typedef.
#[derive(Debug, RustcEncodable)]
pub struct TypeDefData {
    pub id: NodeId,
    pub name: String,
    pub span: Span,
    pub qualname: String,
    pub value: String,
    pub visibility: Visibility,
    pub parent: Option<DefId>,
    pub docs: String,
    pub sig: Option<Signature>,
}

/// Data for a reference to a type or trait.
#[derive(Clone, Debug, RustcEncodable)]
pub struct TypeRefData {
    pub span: Span,
    pub scope: NodeId,
    pub ref_id: Option<DefId>,
    pub qualname: String,
}

#[derive(Debug, RustcEncodable)]
pub struct UseData {
    pub id: NodeId,
    pub span: Span,
    pub name: String,
    pub mod_id: Option<DefId>,
    pub scope: NodeId,
    pub visibility: Visibility,
}

#[derive(Debug, RustcEncodable)]
pub struct UseGlobData {
    pub id: NodeId,
    pub span: Span,
    pub names: Vec<String>,
    pub scope: NodeId,
    pub visibility: Visibility,
}

/// Data for local and global variables (consts and statics).
#[derive(Debug, RustcEncodable)]
pub struct VariableData {
    pub id: NodeId,
    pub kind: VariableKind,
    pub name: String,
    pub qualname: String,
    pub span: Span,
    pub scope: NodeId,
    pub parent: Option<DefId>,
    pub value: String,
    pub type_value: String,
    pub visibility: Visibility,
    pub docs: String,
    pub sig: Option<Signature>,
}

#[derive(Debug, RustcEncodable)]
pub enum VariableKind {
    Static,
    Const,
    Local,
    Field,
}

/// Data for the use of some item (e.g., the use of a local variable, which
/// will refer to that variables declaration (by ref_id)).
#[derive(Debug, RustcEncodable)]
pub struct VariableRefData {
    pub name: String,
    pub span: Span,
    pub scope: NodeId,
    pub ref_id: DefId,
}


/// Encodes information about the signature of a definition. This should have
/// enough information to create a nice display about a definition without
/// access to the source code.
#[derive(Clone, Debug, RustcEncodable)]
pub struct Signature {
    pub span: Span,
    pub text: String,
    // These identify the main identifier for the defintion as byte offsets into
    // `text`. E.g., of `foo` in `pub fn foo(...)`
    pub ident_start: usize,
    pub ident_end: usize,
    pub defs: Vec<SigElement>,
    pub refs: Vec<SigElement>,
}

/// An element of a signature. `start` and `end` are byte offsets into the `text`
/// of the parent `Signature`.
#[derive(Clone, Debug, RustcEncodable)]
pub struct SigElement {
    pub id: DefId,
    pub start: usize,
    pub end: usize,
}
