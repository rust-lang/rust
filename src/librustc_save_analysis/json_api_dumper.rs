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

use rustc_serialize::json::as_json;

use external_data::*;
use data::{VariableKind, Visibility};
use dump::Dump;
use id_from_def_id;

use rls_data::{Analysis, Import, ImportKind, Def, DefKind, CratePreludeData};


// A dumper to dump a restricted set of JSON information, designed for use with
// libraries distributed without their source. Clients are likely to use type
// information here, and (for example) generate Rustdoc URLs, but don't need
// information for navigating the source of the crate.
// Relative to the regular JSON save-analysis info, this form is filtered to
// remove non-visible items.
pub struct JsonApiDumper<'b, W: Write + 'b> {
    output: &'b mut W,
    result: Analysis,
}

impl<'b, W: Write> JsonApiDumper<'b, W> {
    pub fn new(writer: &'b mut W) -> JsonApiDumper<'b, W> {
        JsonApiDumper { output: writer, result: Analysis::new() }
    }
}

impl<'b, W: Write> Drop for JsonApiDumper<'b, W> {
    fn drop(&mut self) {
        if let Err(_) = write!(self.output, "{}", as_json(&self.result)) {
            error!("Error writing output");
        }
    }
}

macro_rules! impl_fn {
    ($fn_name: ident, $data_type: ident, $bucket: ident) => {
        fn $fn_name(&mut self, data: $data_type) {
            if let Some(datum) = data.into() {
                self.result.$bucket.push(datum);
            }
        }
    }
}

impl<'b, W: Write + 'b> Dump for JsonApiDumper<'b, W> {
    fn crate_prelude(&mut self, data: CratePreludeData) {
        self.result.prelude = Some(data)
    }

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
    impl_fn!(mod_data, ModData, defs);
    impl_fn!(typedef, TypeDefData, defs);
    impl_fn!(variable, VariableData, defs);

    fn impl_data(&mut self, data: ImplData) {
        if data.self_ref.is_some() {
            self.result.relations.push(data.into());
        }
    }
    fn inheritance(&mut self, data: InheritanceData) {
        self.result.relations.push(data.into());
    }
}

// FIXME methods. The defs have information about possible overriding and the
// refs have decl information (e.g., a trait method where we know the required
// method, but not the supplied method). In both cases, we are currently
// ignoring it.

impl Into<Option<Import>> for UseData {
    fn into(self) -> Option<Import> {
        match self.visibility {
            Visibility::Public => Some(Import {
                kind: ImportKind::Use,
                ref_id: self.mod_id.map(|id| id_from_def_id(id)),
                span: self.span,
                name: self.name,
                value: String::new(),
            }),
            _ => None,
        }
    }
}
impl Into<Option<Import>> for UseGlobData {
    fn into(self) -> Option<Import> {
        match self.visibility {
            Visibility::Public => Some(Import {
                kind: ImportKind::GlobUse,
                ref_id: None,
                span: self.span,
                name: "*".to_owned(),
                value: self.names.join(", "),
            }),
            _ => None,
        }
    }
}

impl Into<Option<Def>> for EnumData {
    fn into(self) -> Option<Def> {
        match self.visibility {
            Visibility::Public => Some(Def {
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
                attributes: vec![],
            }),
            _ => None,
        }
    }
}

impl Into<Option<Def>> for TupleVariantData {
    fn into(self) -> Option<Def> {
        Some(Def {
            kind: DefKind::Tuple,
            id: id_from_def_id(self.id),
            span: self.span,
            name: self.name,
            qualname: self.qualname,
            value: self.value,
            parent: self.parent.map(|id| id_from_def_id(id)),
            children: vec![],
            decl_id: None,
            docs: self.docs,
            sig: self.sig,
            attributes: vec![],
        })
    }
}
impl Into<Option<Def>> for StructVariantData {
    fn into(self) -> Option<Def> {
        Some(Def {
            kind: DefKind::Struct,
            id: id_from_def_id(self.id),
            span: self.span,
            name: self.name,
            qualname: self.qualname,
            value: self.value,
            parent: self.parent.map(|id| id_from_def_id(id)),
            children: vec![],
            decl_id: None,
            docs: self.docs,
            sig: self.sig,
            attributes: vec![],
        })
    }
}
impl Into<Option<Def>> for StructData {
    fn into(self) -> Option<Def> {
        match self.visibility {
            Visibility::Public => Some(Def {
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
            attributes: vec![],
        }),
            _ => None,
        }
    }
}
impl Into<Option<Def>> for TraitData {
    fn into(self) -> Option<Def> {
        match self.visibility {
            Visibility::Public => Some(Def {
                kind: DefKind::Trait,
                id: id_from_def_id(self.id),
                span: self.span,
                name: self.name,
                qualname: self.qualname,
                value: self.value,
                children: self.items.into_iter().map(|id| id_from_def_id(id)).collect(),
                parent: None,
                decl_id: None,
                docs: self.docs,
                sig: self.sig,
                attributes: vec![],
            }),
            _ => None,
        }
    }
}
impl Into<Option<Def>> for FunctionData {
    fn into(self) -> Option<Def> {
        match self.visibility {
            Visibility::Public => Some(Def {
                kind: DefKind::Function,
                id: id_from_def_id(self.id),
                span: self.span,
                name: self.name,
                qualname: self.qualname,
                value: self.value,
                children: vec![],
                parent: self.parent.map(|id| id_from_def_id(id)),
                decl_id: None,
                docs: self.docs,
                sig: self.sig,
                attributes: vec![],
            }),
            _ => None,
        }
    }
}
impl Into<Option<Def>> for MethodData {
    fn into(self) -> Option<Def> {
        match self.visibility {
            Visibility::Public => Some(Def {
                kind: DefKind::Method,
                id: id_from_def_id(self.id),
                span: self.span,
                name: self.name,
                qualname: self.qualname,
                value: self.value,
                children: vec![],
                parent: self.parent.map(|id| id_from_def_id(id)),
                decl_id: self.decl_id.map(|id| id_from_def_id(id)),
                docs: self.docs,
                sig: self.sig,
                attributes: vec![],
            }),
            _ => None,
        }
    }
}
impl Into<Option<Def>> for MacroData {
    fn into(self) -> Option<Def> {
        Some(Def {
            kind: DefKind::Macro,
            id: id_from_def_id(null_def_id()),
            span: self.span,
            name: self.name,
            qualname: self.qualname,
            value: String::new(),
            children: vec![],
            parent: None,
            decl_id: None,
            docs: self.docs,
            sig: None,
            attributes: vec![],
        })
    }
}
impl Into<Option<Def>> for ModData {
    fn into(self) -> Option<Def> {
        match self.visibility {
            Visibility::Public => Some(Def {
                kind: DefKind::Mod,
                id: id_from_def_id(self.id),
                span: self.span,
                name: self.name,
                qualname: self.qualname,
                value: self.filename,
                children: self.items.into_iter().map(|id| id_from_def_id(id)).collect(),
                parent: None,
                decl_id: None,
                docs: self.docs,
                sig: self.sig.map(|s| s.into()),
                attributes: vec![],
            }),
            _ => None,
        }
    }
}
impl Into<Option<Def>> for TypeDefData {
    fn into(self) -> Option<Def> {
        match self.visibility {
            Visibility::Public => Some(Def {
                kind: DefKind::Type,
                id: id_from_def_id(self.id),
                span: self.span,
                name: self.name,
                qualname: self.qualname,
                value: self.value,
                children: vec![],
                parent: self.parent.map(|id| id_from_def_id(id)),
                decl_id: None,
                docs: String::new(),
                sig: self.sig.map(|s| s.into()),
                attributes: vec![],
            }),
            _ => None,
        }
    }
}

impl Into<Option<Def>> for VariableData {
    fn into(self) -> Option<Def> {
        match self.visibility {
            Visibility::Public => Some(Def {
                kind: match self.kind {
                    VariableKind::Static => DefKind::Static,
                    VariableKind::Const => DefKind::Const,
                    VariableKind::Local => { return None }
                    VariableKind::Field => DefKind::Field,
                },
                id: id_from_def_id(self.id),
                span: self.span,
                name: self.name,
                qualname: self.qualname,
                value: self.value,
                children: vec![],
                parent: self.parent.map(|id| id_from_def_id(id)),
                decl_id: None,
                docs: self.docs,
                sig: self.sig.map(|s| s.into()),
                attributes: vec![],
            }),
            _ => None,
        }
    }
}
