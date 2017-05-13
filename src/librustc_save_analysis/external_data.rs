// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::def_id::{CrateNum, DefId, DefIndex};
use rustc::hir::map::Map;
use rustc::ty::TyCtxt;
use syntax::ast::NodeId;
use syntax::codemap::CodeMap;
use syntax_pos::Span;

use data::{self, Visibility, SigElement};

// FIXME: this should be pub(crate), but the current snapshot doesn't allow it yet
pub trait Lower {
    type Target;
    fn lower(self, tcx: TyCtxt) -> Self::Target;
}

pub fn make_def_id(id: NodeId, map: &Map) -> DefId {
    map.opt_local_def_id(id).unwrap_or(null_def_id())
}

pub fn null_def_id() -> DefId {
    DefId {
        krate: CrateNum::from_u32(u32::max_value()),
        index: DefIndex::from_u32(u32::max_value())
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

    fn lower(self, tcx: TyCtxt) -> CratePreludeData {
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
    pub id: DefId,
    pub value: String,
    pub name: String,
    pub qualname: String,
    pub span: SpanData,
    pub scope: DefId,
    pub variants: Vec<DefId>,
    pub visibility: Visibility,
    pub docs: String,
    pub sig: Signature,
}

impl Lower for data::EnumData {
    type Target = EnumData;

    fn lower(self, tcx: TyCtxt) -> EnumData {
        EnumData {
            id: make_def_id(self.id, &tcx.map),
            name: self.name,
            value: self.value,
            qualname: self.qualname,
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.map),
            variants: self.variants.into_iter().map(|id| make_def_id(id, &tcx.map)).collect(),
            visibility: self.visibility,
            docs: self.docs,
            sig: self.sig.lower(tcx),
        }
    }
}

/// Data for extern crates.
#[derive(Debug, RustcEncodable)]
pub struct ExternCrateData {
    pub id: DefId,
    pub name: String,
    pub crate_num: CrateNum,
    pub location: String,
    pub span: SpanData,
    pub scope: DefId,
}

impl Lower for data::ExternCrateData {
    type Target = ExternCrateData;

    fn lower(self, tcx: TyCtxt) -> ExternCrateData {
        ExternCrateData {
            id: make_def_id(self.id, &tcx.map),
            name: self.name,
            crate_num: self.crate_num,
            location: self.location,
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.map),
        }
    }
}

/// Data about a function call.
#[derive(Debug, RustcEncodable)]
pub struct FunctionCallData {
    pub span: SpanData,
    pub scope: DefId,
    pub ref_id: DefId,
}

impl Lower for data::FunctionCallData {
    type Target = FunctionCallData;

    fn lower(self, tcx: TyCtxt) -> FunctionCallData {
        FunctionCallData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.map),
            ref_id: self.ref_id,
        }
    }
}

/// Data for all kinds of functions and methods.
#[derive(Clone, Debug, RustcEncodable)]
pub struct FunctionData {
    pub id: DefId,
    pub name: String,
    pub qualname: String,
    pub declaration: Option<DefId>,
    pub span: SpanData,
    pub scope: DefId,
    pub value: String,
    pub visibility: Visibility,
    pub parent: Option<DefId>,
    pub docs: String,
    pub sig: Signature,
}

impl Lower for data::FunctionData {
    type Target = FunctionData;

    fn lower(self, tcx: TyCtxt) -> FunctionData {
        FunctionData {
            id: make_def_id(self.id, &tcx.map),
            name: self.name,
            qualname: self.qualname,
            declaration: self.declaration,
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.map),
            value: self.value,
            visibility: self.visibility,
            parent: self.parent,
            docs: self.docs,
            sig: self.sig.lower(tcx),
        }
    }
}

/// Data about a function call.
#[derive(Debug, RustcEncodable)]
pub struct FunctionRefData {
    pub span: SpanData,
    pub scope: DefId,
    pub ref_id: DefId,
}

impl Lower for data::FunctionRefData {
    type Target = FunctionRefData;

    fn lower(self, tcx: TyCtxt) -> FunctionRefData {
        FunctionRefData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.map),
            ref_id: self.ref_id,
        }
    }
}
#[derive(Debug, RustcEncodable)]
pub struct ImplData {
    pub id: DefId,
    pub span: SpanData,
    pub scope: DefId,
    pub trait_ref: Option<DefId>,
    pub self_ref: Option<DefId>,
}

impl Lower for data::ImplData {
    type Target = ImplData;

    fn lower(self, tcx: TyCtxt) -> ImplData {
        ImplData {
            id: make_def_id(self.id, &tcx.map),
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.map),
            trait_ref: self.trait_ref,
            self_ref: self.self_ref,
        }
    }
}

#[derive(Debug, RustcEncodable)]
pub struct InheritanceData {
    pub span: SpanData,
    pub base_id: DefId,
    pub deriv_id: DefId
}

impl Lower for data::InheritanceData {
    type Target = InheritanceData;

    fn lower(self, tcx: TyCtxt) -> InheritanceData {
        InheritanceData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            base_id: self.base_id,
            deriv_id: make_def_id(self.deriv_id, &tcx.map)
        }
    }
}

/// Data about a macro declaration.
#[derive(Debug, RustcEncodable)]
pub struct MacroData {
    pub span: SpanData,
    pub name: String,
    pub qualname: String,
    pub docs: String,
}

impl Lower for data::MacroData {
    type Target = MacroData;

    fn lower(self, tcx: TyCtxt) -> MacroData {
        MacroData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            name: self.name,
            qualname: self.qualname,
            docs: self.docs,
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
    pub scope: DefId,
}

impl Lower for data::MacroUseData {
    type Target = MacroUseData;

    fn lower(self, tcx: TyCtxt) -> MacroUseData {
        MacroUseData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            name: self.name,
            qualname: self.qualname,
            callee_span: SpanData::from_span(self.callee_span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.map),
        }
    }
}

/// Data about a method call.
#[derive(Debug, RustcEncodable)]
pub struct MethodCallData {
    pub span: SpanData,
    pub scope: DefId,
    pub ref_id: Option<DefId>,
    pub decl_id: Option<DefId>,
}

impl Lower for data::MethodCallData {
    type Target = MethodCallData;

    fn lower(self, tcx: TyCtxt) -> MethodCallData {
        MethodCallData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.map),
            ref_id: self.ref_id,
            decl_id: self.decl_id,
        }
    }
}

/// Data for method declarations (methods with a body are treated as functions).
#[derive(Clone, Debug, RustcEncodable)]
pub struct MethodData {
    pub id: DefId,
    pub name: String,
    pub qualname: String,
    pub span: SpanData,
    pub scope: DefId,
    pub value: String,
    pub decl_id: Option<DefId>,
    pub visibility: Visibility,
    pub parent: Option<DefId>,
    pub docs: String,
    pub sig: Signature,
}

impl Lower for data::MethodData {
    type Target = MethodData;

    fn lower(self, tcx: TyCtxt) -> MethodData {
        MethodData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            name: self.name,
            scope: make_def_id(self.scope, &tcx.map),
            id: make_def_id(self.id, &tcx.map),
            qualname: self.qualname,
            value: self.value,
            decl_id: self.decl_id,
            visibility: self.visibility,
            parent: self.parent,
            docs: self.docs,
            sig: self.sig.lower(tcx),
        }
    }
}

/// Data for modules.
#[derive(Debug, RustcEncodable)]
pub struct ModData {
    pub id: DefId,
    pub name: String,
    pub qualname: String,
    pub span: SpanData,
    pub scope: DefId,
    pub filename: String,
    pub items: Vec<DefId>,
    pub visibility: Visibility,
    pub docs: String,
    pub sig: Signature,
}

impl Lower for data::ModData {
    type Target = ModData;

    fn lower(self, tcx: TyCtxt) -> ModData {
        ModData {
            id: make_def_id(self.id, &tcx.map),
            name: self.name,
            qualname: self.qualname,
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.map),
            filename: self.filename,
            items: self.items.into_iter().map(|id| make_def_id(id, &tcx.map)).collect(),
            visibility: self.visibility,
            docs: self.docs,
            sig: self.sig.lower(tcx),
        }
    }
}

/// Data for a reference to a module.
#[derive(Debug, RustcEncodable)]
pub struct ModRefData {
    pub span: SpanData,
    pub scope: DefId,
    pub ref_id: Option<DefId>,
    pub qualname: String
}

impl Lower for data::ModRefData {
    type Target = ModRefData;

    fn lower(self, tcx: TyCtxt) -> ModRefData {
        ModRefData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.map),
            ref_id: self.ref_id,
            qualname: self.qualname,
        }
    }
}

#[derive(Debug, RustcEncodable)]
pub struct StructData {
    pub span: SpanData,
    pub name: String,
    pub id: DefId,
    pub ctor_id: DefId,
    pub qualname: String,
    pub scope: DefId,
    pub value: String,
    pub fields: Vec<DefId>,
    pub visibility: Visibility,
    pub docs: String,
    pub sig: Signature,
}

impl Lower for data::StructData {
    type Target = StructData;

    fn lower(self, tcx: TyCtxt) -> StructData {
        StructData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            name: self.name,
            id: make_def_id(self.id, &tcx.map),
            ctor_id: make_def_id(self.ctor_id, &tcx.map),
            qualname: self.qualname,
            scope: make_def_id(self.scope, &tcx.map),
            value: self.value,
            fields: self.fields.into_iter().map(|id| make_def_id(id, &tcx.map)).collect(),
            visibility: self.visibility,
            docs: self.docs,
            sig: self.sig.lower(tcx),
        }
    }
}

#[derive(Debug, RustcEncodable)]
pub struct StructVariantData {
    pub span: SpanData,
    pub name: String,
    pub id: DefId,
    pub qualname: String,
    pub type_value: String,
    pub value: String,
    pub scope: DefId,
    pub parent: Option<DefId>,
    pub docs: String,
    pub sig: Signature,
}

impl Lower for data::StructVariantData {
    type Target = StructVariantData;

    fn lower(self, tcx: TyCtxt) -> StructVariantData {
        StructVariantData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            name: self.name,
            id: make_def_id(self.id, &tcx.map),
            qualname: self.qualname,
            type_value: self.type_value,
            value: self.value,
            scope: make_def_id(self.scope, &tcx.map),
            parent: self.parent,
            docs: self.docs,
            sig: self.sig.lower(tcx),
        }
    }
}

#[derive(Debug, RustcEncodable)]
pub struct TraitData {
    pub span: SpanData,
    pub name: String,
    pub id: DefId,
    pub qualname: String,
    pub scope: DefId,
    pub value: String,
    pub items: Vec<DefId>,
    pub visibility: Visibility,
    pub docs: String,
    pub sig: Signature,
}

impl Lower for data::TraitData {
    type Target = TraitData;

    fn lower(self, tcx: TyCtxt) -> TraitData {
        TraitData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            name: self.name,
            id: make_def_id(self.id, &tcx.map),
            qualname: self.qualname,
            scope: make_def_id(self.scope, &tcx.map),
            value: self.value,
            items: self.items.into_iter().map(|id| make_def_id(id, &tcx.map)).collect(),
            visibility: self.visibility,
            docs: self.docs,
            sig: self.sig.lower(tcx),
        }
    }
}

#[derive(Debug, RustcEncodable)]
pub struct TupleVariantData {
    pub span: SpanData,
    pub id: DefId,
    pub name: String,
    pub qualname: String,
    pub type_value: String,
    pub value: String,
    pub scope: DefId,
    pub parent: Option<DefId>,
    pub docs: String,
    pub sig: Signature,
}

impl Lower for data::TupleVariantData {
    type Target = TupleVariantData;

    fn lower(self, tcx: TyCtxt) -> TupleVariantData {
        TupleVariantData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            id: make_def_id(self.id, &tcx.map),
            name: self.name,
            qualname: self.qualname,
            type_value: self.type_value,
            value: self.value,
            scope: make_def_id(self.scope, &tcx.map),
            parent: self.parent,
            docs: self.docs,
            sig: self.sig.lower(tcx),
        }
    }
}

/// Data for a typedef.
#[derive(Debug, RustcEncodable)]
pub struct TypeDefData {
    pub id: DefId,
    pub name: String,
    pub span: SpanData,
    pub qualname: String,
    pub value: String,
    pub visibility: Visibility,
    pub parent: Option<DefId>,
    pub docs: String,
    pub sig: Option<Signature>,
}

impl Lower for data::TypeDefData {
    type Target = TypeDefData;

    fn lower(self, tcx: TyCtxt) -> TypeDefData {
        TypeDefData {
            id: make_def_id(self.id, &tcx.map),
            name: self.name,
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            qualname: self.qualname,
            value: self.value,
            visibility: self.visibility,
            parent: self.parent,
            docs: self.docs,
            sig: self.sig.map(|s| s.lower(tcx)),
        }
    }
}

/// Data for a reference to a type or trait.
#[derive(Clone, Debug, RustcEncodable)]
pub struct TypeRefData {
    pub span: SpanData,
    pub scope: DefId,
    pub ref_id: Option<DefId>,
    pub qualname: String,
}

impl Lower for data::TypeRefData {
    type Target = TypeRefData;

    fn lower(self, tcx: TyCtxt) -> TypeRefData {
        TypeRefData {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.map),
            ref_id: self.ref_id,
            qualname: self.qualname,
        }
    }
}

#[derive(Debug, RustcEncodable)]
pub struct UseData {
    pub id: DefId,
    pub span: SpanData,
    pub name: String,
    pub mod_id: Option<DefId>,
    pub scope: DefId,
    pub visibility: Visibility,
}

impl Lower for data::UseData {
    type Target = UseData;

    fn lower(self, tcx: TyCtxt) -> UseData {
        UseData {
            id: make_def_id(self.id, &tcx.map),
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            name: self.name,
            mod_id: self.mod_id,
            scope: make_def_id(self.scope, &tcx.map),
            visibility: self.visibility,
        }
    }
}

#[derive(Debug, RustcEncodable)]
pub struct UseGlobData {
    pub id: DefId,
    pub span: SpanData,
    pub names: Vec<String>,
    pub scope: DefId,
    pub visibility: Visibility,
}

impl Lower for data::UseGlobData {
    type Target = UseGlobData;

    fn lower(self, tcx: TyCtxt) -> UseGlobData {
        UseGlobData {
            id: make_def_id(self.id, &tcx.map),
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            names: self.names,
            scope: make_def_id(self.scope, &tcx.map),
            visibility: self.visibility,
        }
    }
}

/// Data for local and global variables (consts and statics).
#[derive(Debug, RustcEncodable)]
pub struct VariableData {
    pub id: DefId,
    pub name: String,
    pub kind: data::VariableKind,
    pub qualname: String,
    pub span: SpanData,
    pub scope: DefId,
    pub value: String,
    pub type_value: String,
    pub parent: Option<DefId>,
    pub visibility: Visibility,
    pub docs: String,
    pub sig: Option<Signature>,
}

impl Lower for data::VariableData {
    type Target = VariableData;

    fn lower(self, tcx: TyCtxt) -> VariableData {
        VariableData {
            id: make_def_id(self.id, &tcx.map),
            kind: self.kind,
            name: self.name,
            qualname: self.qualname,
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.map),
            value: self.value,
            type_value: self.type_value,
            parent: self.parent,
            visibility: self.visibility,
            docs: self.docs,
            sig: self.sig.map(|s| s.lower(tcx)),
        }
    }
}

/// Data for the use of some item (e.g., the use of a local variable, which
/// will refer to that variables declaration (by ref_id)).
#[derive(Debug, RustcEncodable)]
pub struct VariableRefData {
    pub name: String,
    pub span: SpanData,
    pub scope: DefId,
    pub ref_id: DefId,
}

impl Lower for data::VariableRefData {
    type Target = VariableRefData;

    fn lower(self, tcx: TyCtxt) -> VariableRefData {
        VariableRefData {
            name: self.name,
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.map),
            ref_id: self.ref_id,
        }
    }
}

#[derive(Clone, Debug, RustcEncodable)]
pub struct Signature {
    pub span: SpanData,
    pub text: String,
    // These identify the main identifier for the defintion as byte offsets into
    // `text`. E.g., of `foo` in `pub fn foo(...)`
    pub ident_start: usize,
    pub ident_end: usize,
    pub defs: Vec<SigElement>,
    pub refs: Vec<SigElement>,
}

impl Lower for data::Signature {
    type Target = Signature;

    fn lower(self, tcx: TyCtxt) -> Signature {
        Signature {
            span: SpanData::from_span(self.span, tcx.sess.codemap()),
            text: self.text,
            ident_start: self.ident_start,
            ident_end: self.ident_end,
            defs: self.defs,
            refs: self.refs,
        }
    }
}
