// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io::Write;

use rustc::hir::def_id::DefId;
use rustc_serialize::json::as_json;

use rls_data::{self, Id, Analysis, Import, ImportKind, Def, DefKind, Ref, RefKind, MacroRef,
               Relation, RelationKind, CratePreludeData};
use rls_span::{Column, Row};

use external_data::*;
use data::VariableKind;
use dump::Dump;

pub struct JsonDumper<O: DumpOutput> {
    result: Analysis,
    output: O,
}

pub trait DumpOutput {
    fn dump(&mut self, result: &Analysis);
}

pub struct WriteOutput<'b, W: Write + 'b> {
    output: &'b mut W,
}

impl<'b, W: Write> DumpOutput for WriteOutput<'b, W> {
    fn dump(&mut self, result: &Analysis) {
        if let Err(_) = write!(self.output, "{}", as_json(&result)) {
            error!("Error writing output");
        }
    }
}

pub struct CallbackOutput<'b> {
    callback: &'b mut FnMut(&Analysis),
}

impl<'b> DumpOutput for CallbackOutput<'b> {
    fn dump(&mut self, result: &Analysis) {
        (self.callback)(result)
    }
}

impl<'b, W: Write> JsonDumper<WriteOutput<'b, W>> {
    pub fn new(writer: &'b mut W) -> JsonDumper<WriteOutput<'b, W>> {
        JsonDumper { output: WriteOutput { output: writer }, result: Analysis::new() }
    }
}

impl<'b> JsonDumper<CallbackOutput<'b>> {
    pub fn with_callback(callback: &'b mut FnMut(&Analysis)) -> JsonDumper<CallbackOutput<'b>> {
        JsonDumper { output: CallbackOutput { callback: callback }, result: Analysis::new() }
    }
}

impl<O: DumpOutput> Drop for JsonDumper<O> {
    fn drop(&mut self) {
        self.output.dump(&self.result);
    }
}

macro_rules! impl_fn {
    ($fn_name: ident, $data_type: ident, $bucket: ident) => {
        fn $fn_name(&mut self, data: $data_type) {
            self.result.$bucket.push(data.into());
        }
    }
}

impl<'b, O: DumpOutput + 'b> Dump for JsonDumper<O> {
    fn crate_prelude(&mut self, data: CratePreludeData) {
        self.result.prelude = Some(data)
    }

    impl_fn!(extern_crate, ExternCrateData, imports);
    impl_fn!(use_data, UseData, imports);
    impl_fn!(use_glob, UseGlobData, imports);

    impl_fn!(enum_data, EnumData, defs);
    impl_fn!(tuple_variant, TupleVariantData, defs);
    impl_fn!(struct_variant, StructVariantData, defs);
    impl_fn!(struct_data, StructData, defs);
    impl_fn!(trait_data, TraitData, defs);
    impl_fn!(function, FunctionData, defs);
    impl_fn!(method, MethodData, defs);
    impl_fn!(macro_data, MacroData, defs);
    impl_fn!(typedef, TypeDefData, defs);
    impl_fn!(variable, VariableData, defs);

    impl_fn!(function_ref, FunctionRefData, refs);
    impl_fn!(function_call, FunctionCallData, refs);
    impl_fn!(method_call, MethodCallData, refs);
    impl_fn!(mod_ref, ModRefData, refs);
    impl_fn!(type_ref, TypeRefData, refs);
    impl_fn!(variable_ref, VariableRefData, refs);

    impl_fn!(macro_use, MacroUseData, macro_refs);

    fn mod_data(&mut self, data: ModData) {
        let id: Id = id_from_def_id(data.id);
        let mut def = Def {
            kind: DefKind::Mod,
            id: id,
            span: data.span.into(),
            name: data.name,
            qualname: data.qualname,
            value: data.filename,
            parent: None,
            children: data.items.into_iter().map(|id| id_from_def_id(id)).collect(),
            decl_id: None,
            docs: data.docs,
            sig: data.sig,
            attributes: data.attributes.into_iter().map(|a| a.into()).collect(),
        };
        if def.span.file_name.to_str().unwrap() != def.value {
            // If the module is an out-of-line defintion, then we'll make the
            // defintion the first character in the module's file and turn the
            // the declaration into a reference to it.
            let rf = Ref {
                kind: RefKind::Mod,
                span: def.span,
                ref_id: id,
            };
            self.result.refs.push(rf);
            def.span = rls_data::SpanData {
                file_name: def.value.clone().into(),
                byte_start: 0,
                byte_end: 0,
                line_start: Row::new_one_indexed(1),
                line_end: Row::new_one_indexed(1),
                column_start: Column::new_one_indexed(1),
                column_end: Column::new_one_indexed(1),
            }
        }

        self.result.defs.push(def);
    }

    fn impl_data(&mut self, data: ImplData) {
        if data.self_ref.is_some() {
            self.result.relations.push(data.into());
        }
    }
    fn inheritance(&mut self, data: InheritanceData) {
        self.result.relations.push(data.into());
    }
}

// FIXME do we want to change ExternalData to this mode? It will break DXR.
// FIXME methods. The defs have information about possible overriding and the
// refs have decl information (e.g., a trait method where we know the required
// method, but not the supplied method). In both cases, we are currently
// ignoring it.

// DefId::index is a newtype and so the JSON serialisation is ugly. Therefore
// we use our own Id which is the same, but without the newtype.
pub fn id_from_def_id(id: DefId) -> Id {
    Id {
        krate: id.krate.as_u32(),
        index: id.index.as_u32(),
    }
}

impl Into<Import> for ExternCrateData {
    fn into(self) -> Import {
        Import {
            kind: ImportKind::ExternCrate,
            ref_id: None,
            span: self.span,
            name: self.name,
            value: String::new(),
        }
    }
}
impl Into<Import> for UseData {
    fn into(self) -> Import {
        Import {
            kind: ImportKind::Use,
            ref_id: self.mod_id.map(|id| id_from_def_id(id)),
            span: self.span,
            name: self.name,
            value: String::new(),
        }
    }
}
impl Into<Import> for UseGlobData {
    fn into(self) -> Import {
        Import {
            kind: ImportKind::GlobUse,
            ref_id: None,
            span: self.span,
            name: "*".to_owned(),
            value: self.names.join(", "),
        }
    }
}

impl Into<Def> for EnumData {
    fn into(self) -> Def {
        Def {
            kind: DefKind::Enum,
            id: id_from_def_id(self.id),
            span: self.span,
            name: self.name,
            qualname: self.qualname,
            value: self.value,
            parent: None,
            children: self.variants.into_iter().map(|id| id_from_def_id(id)).collect(),
            decl_id: None,
            docs: self.docs,
            sig: self.sig,
            attributes: self.attributes,
        }
    }
}

impl Into<Def> for TupleVariantData {
    fn into(self) -> Def {
        Def {
            kind: DefKind::Tuple,
            id: id_from_def_id(self.id),
            span: self.span,
            name: self.name,
            qualname: self.qualname,
            value: self.value,
            parent: None,
            children: vec![],
            decl_id: None,
            docs: self.docs,
            sig: self.sig,
            attributes: self.attributes,
        }
    }
}
impl Into<Def> for StructVariantData {
    fn into(self) -> Def {
        Def {
            kind: DefKind::Struct,
            id: id_from_def_id(self.id),
            span: self.span,
            name: self.name,
            qualname: self.qualname,
            value: self.value,
            parent: None,
            children: vec![],
            decl_id: None,
            docs: self.docs,
            sig: self.sig,
            attributes: self.attributes,
        }
    }
}
impl Into<Def> for StructData {
    fn into(self) -> Def {
        Def {
            kind: DefKind::Struct,
            id: id_from_def_id(self.id),
            span: self.span,
            name: self.name,
            qualname: self.qualname,
            value: self.value,
            parent: None,
            children: self.fields.into_iter().map(|id| id_from_def_id(id)).collect(),
            decl_id: None,
            docs: self.docs,
            sig: self.sig,
            attributes: self.attributes,
        }
    }
}
impl Into<Def> for TraitData {
    fn into(self) -> Def {
        Def {
            kind: DefKind::Trait,
            id: id_from_def_id(self.id),
            span: self.span,
            name: self.name,
            qualname: self.qualname,
            value: self.value,
            parent: None,
            children: self.items.into_iter().map(|id| id_from_def_id(id)).collect(),
            decl_id: None,
            docs: self.docs,
            sig: self.sig,
            attributes: self.attributes,
        }
    }
}
impl Into<Def> for FunctionData {
    fn into(self) -> Def {
        Def {
            kind: DefKind::Function,
            id: id_from_def_id(self.id),
            span: self.span,
            name: self.name,
            qualname: self.qualname,
            value: self.value,
            parent: None,
            children: vec![],
            decl_id: None,
            docs: self.docs,
            sig: self.sig,
            attributes: self.attributes,
        }
    }
}
impl Into<Def> for MethodData {
    fn into(self) -> Def {
        Def {
            kind: DefKind::Method,
            id: id_from_def_id(self.id),
            span: self.span,
            name: self.name,
            qualname: self.qualname,
            value: self.value,
            parent: None,
            children: vec![],
            decl_id: self.decl_id.map(|id| id_from_def_id(id)),
            docs: self.docs,
            sig: self.sig,
            attributes: self.attributes,
        }
    }
}
impl Into<Def> for MacroData {
    fn into(self) -> Def {
        Def {
            kind: DefKind::Macro,
            id: id_from_def_id(null_def_id()),
            span: self.span,
            name: self.name,
            qualname: self.qualname,
            value: String::new(),
            parent: None,
            children: vec![],
            decl_id: None,
            docs: self.docs,
            sig: None,
            attributes: vec![],
        }
    }
}
impl Into<Def> for TypeDefData {
    fn into(self) -> Def {
        Def {
            kind: DefKind::Type,
            id: id_from_def_id(self.id),
            span: self.span,
            name: self.name,
            qualname: self.qualname,
            value: self.value,
            parent: None,
            children: vec![],
            decl_id: None,
            docs: String::new(),
            sig: self.sig,
            attributes: self.attributes,
        }
    }
}
impl Into<Def> for VariableData {
    fn into(self) -> Def {
        Def {
            kind: match self.kind {
                VariableKind::Static => DefKind::Static,
                VariableKind::Const => DefKind::Const,
                VariableKind::Local => DefKind::Local,
                VariableKind::Field => DefKind::Field,
            },
            id: id_from_def_id(self.id),
            span: self.span,
            name: self.name,
            qualname: self.qualname,
            value: self.type_value,
            parent: None,
            children: vec![],
            decl_id: None,
            docs: self.docs,
            sig: None,
            attributes: self.attributes,
        }
    }
}

impl Into<Ref> for FunctionRefData {
    fn into(self) -> Ref {
        Ref {
            kind: RefKind::Function,
            span: self.span,
            ref_id: id_from_def_id(self.ref_id),
        }
    }
}
impl Into<Ref> for FunctionCallData {
    fn into(self) -> Ref {
        Ref {
            kind: RefKind::Function,
            span: self.span,
            ref_id: id_from_def_id(self.ref_id),
        }
    }
}
impl Into<Ref> for MethodCallData {
    fn into(self) -> Ref {
        Ref {
            kind: RefKind::Function,
            span: self.span,
            ref_id: id_from_def_id(self.ref_id.or(self.decl_id).unwrap_or(null_def_id())),
        }
    }
}
impl Into<Ref> for ModRefData {
    fn into(self) -> Ref {
        Ref {
            kind: RefKind::Mod,
            span: self.span,
            ref_id: id_from_def_id(self.ref_id.unwrap_or(null_def_id())),
        }
    }
}
impl Into<Ref> for TypeRefData {
    fn into(self) -> Ref {
        Ref {
            kind: RefKind::Type,
            span: self.span,
            ref_id: id_from_def_id(self.ref_id.unwrap_or(null_def_id())),
        }
    }
}
impl Into<Ref> for VariableRefData {
    fn into(self) -> Ref {
        Ref {
            kind: RefKind::Variable,
            span: self.span,
            ref_id: id_from_def_id(self.ref_id),
        }
    }
}

impl Into<MacroRef> for MacroUseData {
    fn into(self) -> MacroRef {
        MacroRef {
            span: self.span,
            qualname: self.qualname,
            callee_span: self.callee_span.into(),
        }
    }
}

impl Into<Relation> for ImplData {
    fn into(self) -> Relation {
        Relation {
            span: self.span,
            kind: RelationKind::Impl,
            from: id_from_def_id(self.self_ref.unwrap_or(null_def_id())),
            to: id_from_def_id(self.trait_ref.unwrap_or(null_def_id())),
        }
    }
}

impl Into<Relation> for InheritanceData {
    fn into(self) -> Relation {
        Relation {
            span: self.span,
            kind: RelationKind::SuperTrait,
            from: id_from_def_id(self.base_id),
            to: id_from_def_id(self.deriv_id),
        }
    }
}
