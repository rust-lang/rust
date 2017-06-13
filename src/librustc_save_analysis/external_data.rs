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
use syntax::ast::{self, NodeId};
use syntax::codemap::CodeMap;
use syntax::print::pprust;
use syntax_pos::Span;

use data::{self, Visibility};

use rls_data::{SpanData, CratePreludeData, Attribute, Signature};
use rls_span::{Column, Row};

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

pub fn span_from_span(span: Span, cm: &CodeMap) -> SpanData {
    let start = cm.lookup_char_pos(span.lo);
    let end = cm.lookup_char_pos(span.hi);

    SpanData {
        file_name: start.file.name.clone().into(),
        byte_start: span.lo.0,
        byte_end: span.hi.0,
        line_start: Row::new_one_indexed(start.line as u32),
        line_end: Row::new_one_indexed(end.line as u32),
        column_start: Column::new_one_indexed(start.col.0 as u32 + 1),
        column_end: Column::new_one_indexed(end.col.0 as u32 + 1),
    }
}

impl Lower for Vec<ast::Attribute> {
    type Target = Vec<Attribute>;

    fn lower(self, tcx: TyCtxt) -> Vec<Attribute> {
        self.into_iter()
        // Only retain real attributes. Doc comments are lowered separately.
        .filter(|attr| attr.path != "doc")
        .map(|mut attr| {
            // Remove the surrounding '#[..]' or '#![..]' of the pretty printed
            // attribute. First normalize all inner attribute (#![..]) to outer
            // ones (#[..]), then remove the two leading and the one trailing character.
            attr.style = ast::AttrStyle::Outer;
            let value = pprust::attribute_to_string(&attr);
            // This str slicing works correctly, because the leading and trailing characters
            // are in the ASCII range and thus exactly one byte each.
            let value = value[2..value.len()-1].to_string();

            Attribute {
                value: value,
                span: span_from_span(attr.span, tcx.sess.codemap()),
            }
        }).collect()
    }
}

impl Lower for data::CratePreludeData {
    type Target = CratePreludeData;

    fn lower(self, tcx: TyCtxt) -> CratePreludeData {
        CratePreludeData {
            crate_name: self.crate_name,
            crate_root: self.crate_root,
            external_crates: self.external_crates,
            span: span_from_span(self.span, tcx.sess.codemap()),
        }
    }
}

/// Data for enum declarations.
#[derive(Clone, Debug)]
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
    pub sig: Option<Signature>,
    pub attributes: Vec<Attribute>,
}

impl Lower for data::EnumData {
    type Target = EnumData;

    fn lower(self, tcx: TyCtxt) -> EnumData {
        EnumData {
            id: make_def_id(self.id, &tcx.hir),
            name: self.name,
            value: self.value,
            qualname: self.qualname,
            span: span_from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.hir),
            variants: self.variants.into_iter().map(|id| make_def_id(id, &tcx.hir)).collect(),
            visibility: self.visibility,
            docs: self.docs,
            sig: self.sig,
            attributes: self.attributes.lower(tcx),
        }
    }
}

/// Data for extern crates.
#[derive(Debug)]
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
            id: make_def_id(self.id, &tcx.hir),
            name: self.name,
            crate_num: self.crate_num,
            location: self.location,
            span: span_from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.hir),
        }
    }
}

/// Data about a function call.
#[derive(Debug)]
pub struct FunctionCallData {
    pub span: SpanData,
    pub scope: DefId,
    pub ref_id: DefId,
}

impl Lower for data::FunctionCallData {
    type Target = FunctionCallData;

    fn lower(self, tcx: TyCtxt) -> FunctionCallData {
        FunctionCallData {
            span: span_from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.hir),
            ref_id: self.ref_id,
        }
    }
}

/// Data for all kinds of functions and methods.
#[derive(Clone, Debug)]
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
    pub sig: Option<Signature>,
    pub attributes: Vec<Attribute>,
}

impl Lower for data::FunctionData {
    type Target = FunctionData;

    fn lower(self, tcx: TyCtxt) -> FunctionData {
        FunctionData {
            id: make_def_id(self.id, &tcx.hir),
            name: self.name,
            qualname: self.qualname,
            declaration: self.declaration,
            span: span_from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.hir),
            value: self.value,
            visibility: self.visibility,
            parent: self.parent,
            docs: self.docs,
            sig: self.sig,
            attributes: self.attributes.lower(tcx),
        }
    }
}

/// Data about a function call.
#[derive(Debug)]
pub struct FunctionRefData {
    pub span: SpanData,
    pub scope: DefId,
    pub ref_id: DefId,
}

impl Lower for data::FunctionRefData {
    type Target = FunctionRefData;

    fn lower(self, tcx: TyCtxt) -> FunctionRefData {
        FunctionRefData {
            span: span_from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.hir),
            ref_id: self.ref_id,
        }
    }
}
#[derive(Debug)]
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
            id: make_def_id(self.id, &tcx.hir),
            span: span_from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.hir),
            trait_ref: self.trait_ref,
            self_ref: self.self_ref,
        }
    }
}

#[derive(Debug)]
pub struct InheritanceData {
    pub span: SpanData,
    pub base_id: DefId,
    pub deriv_id: DefId
}

impl Lower for data::InheritanceData {
    type Target = InheritanceData;

    fn lower(self, tcx: TyCtxt) -> InheritanceData {
        InheritanceData {
            span: span_from_span(self.span, tcx.sess.codemap()),
            base_id: self.base_id,
            deriv_id: make_def_id(self.deriv_id, &tcx.hir)
        }
    }
}

/// Data about a macro declaration.
#[derive(Debug)]
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
            span: span_from_span(self.span, tcx.sess.codemap()),
            name: self.name,
            qualname: self.qualname,
            docs: self.docs,
        }
    }
}

/// Data about a macro use.
#[derive(Debug)]
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
            span: span_from_span(self.span, tcx.sess.codemap()),
            name: self.name,
            qualname: self.qualname,
            callee_span: span_from_span(self.callee_span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.hir),
        }
    }
}

/// Data about a method call.
#[derive(Debug)]
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
            span: span_from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.hir),
            ref_id: self.ref_id,
            decl_id: self.decl_id,
        }
    }
}

/// Data for method declarations (methods with a body are treated as functions).
#[derive(Clone, Debug)]
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
    pub sig: Option<Signature>,
    pub attributes: Vec<Attribute>,
}

impl Lower for data::MethodData {
    type Target = MethodData;

    fn lower(self, tcx: TyCtxt) -> MethodData {
        MethodData {
            span: span_from_span(self.span, tcx.sess.codemap()),
            name: self.name,
            scope: make_def_id(self.scope, &tcx.hir),
            id: make_def_id(self.id, &tcx.hir),
            qualname: self.qualname,
            value: self.value,
            decl_id: self.decl_id,
            visibility: self.visibility,
            parent: self.parent,
            docs: self.docs,
            sig: self.sig,
            attributes: self.attributes.lower(tcx),
        }
    }
}

/// Data for modules.
#[derive(Debug)]
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
    pub sig: Option<Signature>,
    pub attributes: Vec<Attribute>,
}

impl Lower for data::ModData {
    type Target = ModData;

    fn lower(self, tcx: TyCtxt) -> ModData {
        ModData {
            id: make_def_id(self.id, &tcx.hir),
            name: self.name,
            qualname: self.qualname,
            span: span_from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.hir),
            filename: self.filename,
            items: self.items.into_iter().map(|id| make_def_id(id, &tcx.hir)).collect(),
            visibility: self.visibility,
            docs: self.docs,
            sig: self.sig,
            attributes: self.attributes.lower(tcx),
        }
    }
}

/// Data for a reference to a module.
#[derive(Debug)]
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
            span: span_from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.hir),
            ref_id: self.ref_id,
            qualname: self.qualname,
        }
    }
}

#[derive(Debug)]
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
    pub sig: Option<Signature>,
    pub attributes: Vec<Attribute>,
}

impl Lower for data::StructData {
    type Target = StructData;

    fn lower(self, tcx: TyCtxt) -> StructData {
        StructData {
            span: span_from_span(self.span, tcx.sess.codemap()),
            name: self.name,
            id: make_def_id(self.id, &tcx.hir),
            ctor_id: make_def_id(self.ctor_id, &tcx.hir),
            qualname: self.qualname,
            scope: make_def_id(self.scope, &tcx.hir),
            value: self.value,
            fields: self.fields.into_iter().map(|id| make_def_id(id, &tcx.hir)).collect(),
            visibility: self.visibility,
            docs: self.docs,
            sig: self.sig,
            attributes: self.attributes.lower(tcx),
        }
    }
}

#[derive(Debug)]
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
    pub sig: Option<Signature>,
    pub attributes: Vec<Attribute>,
}

impl Lower for data::StructVariantData {
    type Target = StructVariantData;

    fn lower(self, tcx: TyCtxt) -> StructVariantData {
        StructVariantData {
            span: span_from_span(self.span, tcx.sess.codemap()),
            name: self.name,
            id: make_def_id(self.id, &tcx.hir),
            qualname: self.qualname,
            type_value: self.type_value,
            value: self.value,
            scope: make_def_id(self.scope, &tcx.hir),
            parent: self.parent,
            docs: self.docs,
            sig: self.sig,
            attributes: self.attributes.lower(tcx),
        }
    }
}

#[derive(Debug)]
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
    pub sig: Option<Signature>,
    pub attributes: Vec<Attribute>,
}

impl Lower for data::TraitData {
    type Target = TraitData;

    fn lower(self, tcx: TyCtxt) -> TraitData {
        TraitData {
            span: span_from_span(self.span, tcx.sess.codemap()),
            name: self.name,
            id: make_def_id(self.id, &tcx.hir),
            qualname: self.qualname,
            scope: make_def_id(self.scope, &tcx.hir),
            value: self.value,
            items: self.items.into_iter().map(|id| make_def_id(id, &tcx.hir)).collect(),
            visibility: self.visibility,
            docs: self.docs,
            sig: self.sig,
            attributes: self.attributes.lower(tcx),
        }
    }
}

#[derive(Debug)]
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
    pub sig: Option<Signature>,
    pub attributes: Vec<Attribute>,
}

impl Lower for data::TupleVariantData {
    type Target = TupleVariantData;

    fn lower(self, tcx: TyCtxt) -> TupleVariantData {
        TupleVariantData {
            span: span_from_span(self.span, tcx.sess.codemap()),
            id: make_def_id(self.id, &tcx.hir),
            name: self.name,
            qualname: self.qualname,
            type_value: self.type_value,
            value: self.value,
            scope: make_def_id(self.scope, &tcx.hir),
            parent: self.parent,
            docs: self.docs,
            sig: self.sig,
            attributes: self.attributes.lower(tcx),
        }
    }
}

/// Data for a typedef.
#[derive(Debug)]
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
    pub attributes: Vec<Attribute>,
}

impl Lower for data::TypeDefData {
    type Target = TypeDefData;

    fn lower(self, tcx: TyCtxt) -> TypeDefData {
        TypeDefData {
            id: make_def_id(self.id, &tcx.hir),
            name: self.name,
            span: span_from_span(self.span, tcx.sess.codemap()),
            qualname: self.qualname,
            value: self.value,
            visibility: self.visibility,
            parent: self.parent,
            docs: self.docs,
            sig: self.sig,
            attributes: self.attributes.lower(tcx),
        }
    }
}

/// Data for a reference to a type or trait.
#[derive(Clone, Debug)]
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
            span: span_from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.hir),
            ref_id: self.ref_id,
            qualname: self.qualname,
        }
    }
}

#[derive(Debug)]
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
            id: make_def_id(self.id, &tcx.hir),
            span: span_from_span(self.span, tcx.sess.codemap()),
            name: self.name,
            mod_id: self.mod_id,
            scope: make_def_id(self.scope, &tcx.hir),
            visibility: self.visibility,
        }
    }
}

#[derive(Debug)]
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
            id: make_def_id(self.id, &tcx.hir),
            span: span_from_span(self.span, tcx.sess.codemap()),
            names: self.names,
            scope: make_def_id(self.scope, &tcx.hir),
            visibility: self.visibility,
        }
    }
}

/// Data for local and global variables (consts and statics).
#[derive(Debug)]
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
    pub attributes: Vec<Attribute>,
}

impl Lower for data::VariableData {
    type Target = VariableData;

    fn lower(self, tcx: TyCtxt) -> VariableData {
        VariableData {
            id: make_def_id(self.id, &tcx.hir),
            kind: self.kind,
            name: self.name,
            qualname: self.qualname,
            span: span_from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.hir),
            value: self.value,
            type_value: self.type_value,
            parent: self.parent,
            visibility: self.visibility,
            docs: self.docs,
            sig: self.sig,
            attributes: self.attributes.lower(tcx),
        }
    }
}

/// Data for the use of some item (e.g., the use of a local variable, which
/// will refer to that variables declaration (by ref_id)).
#[derive(Debug)]
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
            span: span_from_span(self.span, tcx.sess.codemap()),
            scope: make_def_id(self.scope, &tcx.hir),
            ref_id: self.ref_id,
        }
    }
}
