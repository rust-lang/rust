//! This module is used to store stuff from Rust's AST in a more convenient
//! manner (and with prettier names) before cleaning.
crate use self::StructType::*;

use rustc_ast as ast;
use rustc_span::hygiene::MacroKind;
use rustc_span::{self, Span, Symbol};

use rustc_hir as hir;

#[derive(Debug, Clone, Copy)]
crate enum StructType {
    /// A braced struct
    Plain,
    /// A tuple struct
    Tuple,
    /// A unit struct
    Unit,
}

#[derive(Debug)]
crate struct Static<'hir> {
    crate type_: &'hir hir::Ty<'hir>,
    crate mutability: hir::Mutability,
    crate expr: hir::BodyId,
    crate name: Symbol,
    crate attrs: &'hir [ast::Attribute],
    crate vis: &'hir hir::Visibility<'hir>,
    crate id: hir::HirId,
    crate span: Span,
}

#[derive(Debug)]
crate struct Import<'hir> {
    crate name: Symbol,
    crate id: hir::HirId,
    crate vis: &'hir hir::Visibility<'hir>,
    crate attrs: &'hir [ast::Attribute],
    crate path: &'hir hir::Path<'hir>,
    crate glob: bool,
    crate span: Span,
}

crate struct ProcMacro {
    crate name: Symbol,
    crate id: hir::HirId,
    crate kind: MacroKind,
    crate helpers: Vec<Symbol>,
}

crate fn struct_type_from_def(vdata: &hir::VariantData<'_>) -> StructType {
    match *vdata {
        hir::VariantData::Struct(..) => Plain,
        hir::VariantData::Tuple(..) => Tuple,
        hir::VariantData::Unit(..) => Unit,
    }
}
