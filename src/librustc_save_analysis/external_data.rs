// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt::{self, Display, Formatter};

use rustc::hir::def_id::DefId;
use rustc::hir::map::Map;
use rustc::ty::TyCtxt;
use syntax::ast::{CrateNum, NodeId};
use syntax::codemap::{Span, CodeMap};

use super::data;

// FIXME: this should be pub(crate), but the current snapshot doesn't allow it yet
pub trait Lower {
    type Target;
    fn lower(self, tcx: &TyCtxt) -> Self::Target;
}

// We use a newtype to enforce conversion of all NodeIds (which are u32s as well)
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, RustcEncodable)]
pub struct Id(u32);

impl Id {
    pub fn from_def_id(id: DefId) -> Id {
        Id(id.index.as_u32())
    }

    // FIXME: this function is called with non-local NodeIds. This means that they
    // cannot be mapped to a DefId. We should remove those calls. In the meantime,
    // we return a "null Id" when the NodeId is invalid.
    pub fn from_node_id(id: NodeId, map: &Map) -> Id {
        map.opt_local_def_id(id).map(|id| Id(id.index.as_u32()))
                                .unwrap_or(Id::null())
    }

    pub fn null() -> Id {
        Id(u32::max_value())
    }
}

impl Display for Id {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug, RustcEncodable)]
pub struct SpanData {
    pub file_name: String,
    pub byte_start: u32,
    pub byte_end: u32,
    /// 1-based.
    pub line_start: usize,
    pub line_end: usize,
    /// 1-based, character offset.
    pub column_start: usize,
    pub column_end: usize,
}

impl SpanData {
    pub fn from_span(span: Span, cm: &CodeMap) -> SpanData {
        let start = cm.lookup_char_pos(span.lo);
        let end = cm.lookup_char_pos(span.hi);

        SpanData {
            file_name: start.file.name.clone(),
            byte_start: span.lo.0,
            byte_end: span.hi.0,
            line_start: start.line,
            line_end: end.line,
            column_start: start.col.0 + 1,
            column_end: end.col.0 + 1,
        }
    }
}

#[derive(Debug, RustcEncodable)]
pub struct CratePreludeData {
    pub crate_name: String,
    pub crate_root: String,
    pub external_crates: Vec<data::ExternalCrateData>,
    pub span: SpanData,
}

impl Lower for data::CratePreludeData {
    type Target = CratePreludeData;

    fn lower(self, tcx: &TyCtxt) -> CratePreludeData {
        CratePreludeData {
            crate_name: self.crate_name,
            crate_root: self.crate_root,
            external_crates: self.external_crates,
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
        }
    }
}

/// Data for enum declarations.
#[derive(Clone, Debug, RustcEncodable)]
pub struct EnumData {
    pub id: Id,
    pub value: String,
    pub qualname: String,
    pub span: SpanData,
    pub scope: Id,
}

impl Lower for data::EnumData {
    type Target = EnumData;

    fn lower(self, tcx: &TyCtxt) -> EnumData {
        EnumData {
            id: Id::from_node_id(self.id, &tcx.map),
            value: self.value,
            qualname: self.qualname,
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: Id::from_node_id(self.scope, &tcx.map),
        }
    }
}

/// Data for extern crates.
#[derive(Debug, RustcEncodable)]
pub struct ExternCrateData {
    pub id: Id,
    pub name: String,
    pub crate_num: CrateNum,
    pub location: String,
    pub span: SpanData,
    pub scope: Id,
}

impl Lower for data::ExternCrateData {
    type Target = ExternCrateData;

    fn lower(self, tcx: &TyCtxt) -> ExternCrateData {
        ExternCrateData {
            id: Id::from_node_id(self.id, &tcx.map),
            name: self.name,
            crate_num: self.crate_num,
            location: self.location,
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: Id::from_node_id(self.scope, &tcx.map),
        }
    }
}

/// Data about a function call.
#[derive(Debug, RustcEncodable)]
pub struct FunctionCallData {
    pub span: SpanData,
    pub scope: Id,
    pub ref_id: Id,
}

impl Lower for data::FunctionCallData {
    type Target = FunctionCallData;

    fn lower(self, tcx: &TyCtxt) -> FunctionCallData {
        FunctionCallData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: Id::from_node_id(self.scope, &tcx.map),
            ref_id: Id::from_def_id(self.ref_id),
        }
    }
}

/// Data for all kinds of functions and methods.
#[derive(Clone, Debug, RustcEncodable)]
pub struct FunctionData {
    pub id: Id,
    pub name: String,
    pub qualname: String,
    pub declaration: Option<Id>,
    pub span: SpanData,
    pub scope: Id,
}

impl Lower for data::FunctionData {
    type Target = FunctionData;

    fn lower(self, tcx: &TyCtxt) -> FunctionData {
        FunctionData {
            id: Id::from_node_id(self.id, &tcx.map),
            name: self.name,
            qualname: self.qualname,
            declaration: self.declaration.map(Id::from_def_id),
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: Id::from_node_id(self.scope, &tcx.map),
        }
    }
}

/// Data about a function call.
#[derive(Debug, RustcEncodable)]
pub struct FunctionRefData {
    pub span: SpanData,
    pub scope: Id,
    pub ref_id: Id,
}

impl Lower for data::FunctionRefData {
    type Target = FunctionRefData;

    fn lower(self, tcx: &TyCtxt) -> FunctionRefData {
        FunctionRefData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: Id::from_node_id(self.scope, &tcx.map),
            ref_id: Id::from_def_id(self.ref_id),
        }
    }
}
#[derive(Debug, RustcEncodable)]
pub struct ImplData {
    pub id: Id,
    pub span: SpanData,
    pub scope: Id,
    pub trait_ref: Option<Id>,
    pub self_ref: Option<Id>,
}

impl Lower for data::ImplData {
    type Target = ImplData;

    fn lower(self, tcx: &TyCtxt) -> ImplData {
        ImplData {
            id: Id::from_node_id(self.id, &tcx.map),
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: Id::from_node_id(self.scope, &tcx.map),
            trait_ref: self.trait_ref.map(Id::from_def_id),
            self_ref: self.self_ref.map(Id::from_def_id),
        }
    }
}

#[derive(Debug, RustcEncodable)]
pub struct InheritanceData {
    pub span: SpanData,
    pub base_id: Id,
    pub deriv_id: Id
}

impl Lower for data::InheritanceData {
    type Target = InheritanceData;

    fn lower(self, tcx: &TyCtxt) -> InheritanceData {
        InheritanceData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            base_id: Id::from_def_id(self.base_id),
            deriv_id: Id::from_node_id(self.deriv_id, &tcx.map)
        }
    }
}

/// Data about a macro declaration.
#[derive(Debug, RustcEncodable)]
pub struct MacroData {
    pub span: SpanData,
    pub name: String,
    pub qualname: String,
}

impl Lower for data::MacroData {
    type Target = MacroData;

    fn lower(self, tcx: &TyCtxt) -> MacroData {
        MacroData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            name: self.name,
            qualname: self.qualname,
        }
    }
}

/// Data about a macro use.
#[derive(Debug, RustcEncodable)]
pub struct MacroUseData {
    pub span: SpanData,
    pub name: String,
    pub qualname: String,
    // Because macro expansion happens before ref-ids are determined,
    // we use the callee span to reference the associated macro definition.
    pub callee_span: SpanData,
    pub scope: Id,
    pub imported: bool,
}

impl Lower for data::MacroUseData {
    type Target = MacroUseData;

    fn lower(self, tcx: &TyCtxt) -> MacroUseData {
        MacroUseData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            name: self.name,
            qualname: self.qualname,
            callee_span: SpanData::from_span(self.callee_span, tcx.sess.codemap()),
            scope: Id::from_node_id(self.scope, &tcx.map),
            imported: self.imported,
        }
    }
}

/// Data about a method call.
#[derive(Debug, RustcEncodable)]
pub struct MethodCallData {
    pub span: SpanData,
    pub scope: Id,
    pub ref_id: Option<Id>,
    pub decl_id: Option<Id>,
}

impl Lower for data::MethodCallData {
    type Target = MethodCallData;

    fn lower(self, tcx: &TyCtxt) -> MethodCallData {
        MethodCallData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: Id::from_node_id(self.scope, &tcx.map),
            ref_id: self.ref_id.map(Id::from_def_id),
            decl_id: self.decl_id.map(Id::from_def_id),
        }
    }
}

/// Data for method declarations (methods with a body are treated as functions).
#[derive(Clone, Debug, RustcEncodable)]
pub struct MethodData {
    pub id: Id,
    pub qualname: String,
    pub span: SpanData,
    pub scope: Id,
}

impl Lower for data::MethodData {
    type Target = MethodData;

    fn lower(self, tcx: &TyCtxt) -> MethodData {
        MethodData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: Id::from_node_id(self.scope, &tcx.map),
            id: Id::from_node_id(self.id, &tcx.map),
            qualname: self.qualname,
        }
    }
}

/// Data for modules.
#[derive(Debug, RustcEncodable)]
pub struct ModData {
    pub id: Id,
    pub name: String,
    pub qualname: String,
    pub span: SpanData,
    pub scope: Id,
    pub filename: String,
}

impl Lower for data::ModData {
    type Target = ModData;

    fn lower(self, tcx: &TyCtxt) -> ModData {
        ModData {
            id: Id::from_node_id(self.id, &tcx.map),
            name: self.name,
            qualname: self.qualname,
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: Id::from_node_id(self.scope, &tcx.map),
            filename: self.filename,
        }
    }
}

/// Data for a reference to a module.
#[derive(Debug, RustcEncodable)]
pub struct ModRefData {
    pub span: SpanData,
    pub scope: Id,
    pub ref_id: Option<Id>,
    pub qualname: String
}

impl Lower for data::ModRefData {
    type Target = ModRefData;

    fn lower(self, tcx: &TyCtxt) -> ModRefData {
        ModRefData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: Id::from_node_id(self.scope, &tcx.map),
            ref_id: self.ref_id.map(Id::from_def_id),
            qualname: self.qualname,
        }
    }
}

#[derive(Debug, RustcEncodable)]
pub struct StructData {
    pub span: SpanData,
    pub id: Id,
    pub ctor_id: Id,
    pub qualname: String,
    pub scope: Id,
    pub value: String
}

impl Lower for data::StructData {
    type Target = StructData;

    fn lower(self, tcx: &TyCtxt) -> StructData {
        StructData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            id: Id::from_node_id(self.id, &tcx.map),
            ctor_id: Id::from_node_id(self.ctor_id, &tcx.map),
            qualname: self.qualname,
            scope: Id::from_node_id(self.scope, &tcx.map),
            value: self.value
        }
    }
}

#[derive(Debug, RustcEncodable)]
pub struct StructVariantData {
    pub span: SpanData,
    pub id: Id,
    pub qualname: String,
    pub type_value: String,
    pub value: String,
    pub scope: Id
}

impl Lower for data::StructVariantData {
    type Target = StructVariantData;

    fn lower(self, tcx: &TyCtxt) -> StructVariantData {
        StructVariantData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            id: Id::from_node_id(self.id, &tcx.map),
            qualname: self.qualname,
            type_value: self.type_value,
            value: self.value,
            scope: Id::from_node_id(self.scope, &tcx.map),
        }
    }
}

#[derive(Debug, RustcEncodable)]
pub struct TraitData {
    pub span: SpanData,
    pub id: Id,
    pub qualname: String,
    pub scope: Id,
    pub value: String
}

impl Lower for data::TraitData {
    type Target = TraitData;

    fn lower(self, tcx: &TyCtxt) -> TraitData {
        TraitData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            id: Id::from_node_id(self.id, &tcx.map),
            qualname: self.qualname,
            scope: Id::from_node_id(self.scope, &tcx.map),
            value: self.value,
        }
    }
}

#[derive(Debug, RustcEncodable)]
pub struct TupleVariantData {
    pub span: SpanData,
    pub id: Id,
    pub name: String,
    pub qualname: String,
    pub type_value: String,
    pub value: String,
    pub scope: Id,
}

impl Lower for data::TupleVariantData {
    type Target = TupleVariantData;

    fn lower(self, tcx: &TyCtxt) -> TupleVariantData {
        TupleVariantData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            id: Id::from_node_id(self.id, &tcx.map),
            name: self.name,
            qualname: self.qualname,
            type_value: self.type_value,
            value: self.value,
            scope: Id::from_node_id(self.scope, &tcx.map),
        }
    }
}

/// Data for a typedef.
#[derive(Debug, RustcEncodable)]
pub struct TypedefData {
    pub id: Id,
    pub span: SpanData,
    pub qualname: String,
    pub value: String,
}

impl Lower for data::TypedefData {
    type Target = TypedefData;

    fn lower(self, tcx: &TyCtxt) -> TypedefData {
        TypedefData {
            id: Id::from_node_id(self.id, &tcx.map),
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            qualname: self.qualname,
            value: self.value,
        }
    }
}

/// Data for a reference to a type or trait.
#[derive(Clone, Debug, RustcEncodable)]
pub struct TypeRefData {
    pub span: SpanData,
    pub scope: Id,
    pub ref_id: Option<Id>,
    pub qualname: String,
}

impl Lower for data::TypeRefData {
    type Target = TypeRefData;

    fn lower(self, tcx: &TyCtxt) -> TypeRefData {
        TypeRefData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: Id::from_node_id(self.scope, &tcx.map),
            ref_id: self.ref_id.map(Id::from_def_id),
            qualname: self.qualname,
        }
    }
}

#[derive(Debug, RustcEncodable)]
pub struct UseData {
    pub id: Id,
    pub span: SpanData,
    pub name: String,
    pub mod_id: Option<Id>,
    pub scope: Id
}

impl Lower for data::UseData {
    type Target = UseData;

    fn lower(self, tcx: &TyCtxt) -> UseData {
        UseData {
            id: Id::from_node_id(self.id, &tcx.map),
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            name: self.name,
            mod_id: self.mod_id.map(Id::from_def_id),
            scope: Id::from_node_id(self.scope, &tcx.map),
        }
    }
}

#[derive(Debug, RustcEncodable)]
pub struct UseGlobData {
    pub id: Id,
    pub span: SpanData,
    pub names: Vec<String>,
    pub scope: Id
}

impl Lower for data::UseGlobData {
    type Target = UseGlobData;

    fn lower(self, tcx: &TyCtxt) -> UseGlobData {
        UseGlobData {
            id: Id::from_node_id(self.id, &tcx.map),
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            names: self.names,
            scope: Id::from_node_id(self.scope, &tcx.map),
        }
    }
}

/// Data for local and global variables (consts and statics).
#[derive(Debug, RustcEncodable)]
pub struct VariableData {
    pub id: Id,
    pub name: String,
    pub qualname: String,
    pub span: SpanData,
    pub scope: Id,
    pub value: String,
    pub type_value: String,
}

impl Lower for data::VariableData {
    type Target = VariableData;

    fn lower(self, tcx: &TyCtxt) -> VariableData {
        VariableData {
            id: Id::from_node_id(self.id, &tcx.map),
            name: self.name,
            qualname: self.qualname,
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: Id::from_node_id(self.scope, &tcx.map),
            value: self.value,
            type_value: self.type_value,
        }
    }
}

/// Data for the use of some item (e.g., the use of a local variable, which
/// will refer to that variables declaration (by ref_id)).
#[derive(Debug, RustcEncodable)]
pub struct VariableRefData {
    pub name: String,
    pub span: SpanData,
    pub scope: Id,
    pub ref_id: Id,
}

impl Lower for data::VariableRefData {
    type Target = VariableRefData;

    fn lower(self, tcx: &TyCtxt) -> VariableRefData {
        VariableRefData {
            name: self.name,
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: Id::from_node_id(self.scope, &tcx.map),
            ref_id: Id::from_def_id(self.ref_id),
        }
    }
}
