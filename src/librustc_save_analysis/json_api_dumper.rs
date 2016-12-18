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

use external_data::*;
use data::{VariableKind, Visibility, SigElement};
use dump::Dump;
use super::Format;


// A dumper to dump a restricted set of JSON information, designed for use with
// libraries distributed without their source. Clients are likely to use type
// information here, and (for example) generate Rustdoc URLs, but don't need
// information for navigating the source of the crate.
// Relative to the regular JSON save-analysis info, this form is filtered to
// remove non-visible items, but includes some extra info for items (e.g., the
// parent field for finding the struct to which a field belongs).
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
            if let Some(datum) = From::from(data) {
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
}

// FIXME methods. The defs have information about possible overriding and the
// refs have decl information (e.g., a trait method where we know the required
// method, but not the supplied method). In both cases, we are currently
// ignoring it.

#[derive(Debug, RustcEncodable)]
struct Analysis {
    kind: Format,
    prelude: Option<CratePreludeData>,
    imports: Vec<Import>,
    defs: Vec<Def>,
    // These two fields are dummies so that clients can parse the two kinds of
    // JSON data in the same way.
    refs: Vec<()>,
    macro_refs: Vec<()>,
}

impl Analysis {
    fn new() -> Analysis {
        Analysis {
            kind: Format::JsonApi,
            prelude: None,
            imports: vec![],
            defs: vec![],
            refs: vec![],
            macro_refs: vec![],
        }
    }
}

// DefId::index is a newtype and so the JSON serialisation is ugly. Therefore
// we use our own Id which is the same, but without the newtype.
#[derive(Debug, RustcEncodable)]
struct Id {
    krate: u32,
    index: u32,
}

impl From<DefId> for Id {
    fn from(id: DefId) -> Id {
        Id {
            krate: id.krate.as_u32(),
            index: id.index.as_u32(),
        }
    }
}

#[derive(Debug, RustcEncodable)]
struct Import {
    kind: ImportKind,
    id: Id,
    span: SpanData,
    name: String,
    value: String,
}

#[derive(Debug, RustcEncodable)]
enum ImportKind {
    Use,
    GlobUse,
}

impl From<UseData> for Option<Import> {
    fn from(data: UseData) -> Option<Import> {
        match data.visibility {
            Visibility::Public => Some(Import {
                kind: ImportKind::Use,
                id: From::from(data.id),
                span: data.span,
                name: data.name,
                value: String::new(),
            }),
            _ => None,
        }
    }
}
impl From<UseGlobData> for Option<Import> {
    fn from(data: UseGlobData) -> Option<Import> {
        match data.visibility {
            Visibility::Public => Some(Import {
                kind: ImportKind::GlobUse,
                id: From::from(data.id),
                span: data.span,
                name: "*".to_owned(),
                value: data.names.join(", "),
            }),
            _ => None,
        }
    }
}

#[derive(Debug, RustcEncodable)]
struct Def {
    kind: DefKind,
    id: Id,
    span: SpanData,
    name: String,
    qualname: String,
    value: String,
    parent: Option<Id>,
    children: Vec<Id>,
    decl_id: Option<Id>,
    docs: String,
    sig: Option<JsonSignature>,
}

#[derive(Debug, RustcEncodable)]
enum DefKind {
    // value = variant names
    Enum,
    // value = enum name + variant name + types
    Tuple,
    // value = [enum name +] name + fields
    Struct,
    // value = signature
    Trait,
    // value = type + generics
    Function,
    // value = type + generics
    Method,
    // No id, no value.
    Macro,
    // value = file_name
    Mod,
    // value = aliased type
    Type,
    // value = type and init expression (for all variable kinds).
    Static,
    Const,
    Field,
}

impl From<EnumData> for Option<Def> {
    fn from(data: EnumData) -> Option<Def> {
        match data.visibility {
            Visibility::Public => Some(Def {
                kind: DefKind::Enum,
                id: From::from(data.id),
                span: data.span,
                name: data.name,
                qualname: data.qualname,
                value: data.value,
                parent: None,
                children: data.variants.into_iter().map(|id| From::from(id)).collect(),
                decl_id: None,
                docs: data.docs,
                sig: Some(From::from(data.sig)),
            }),
            _ => None,
        }
    }
}

impl From<TupleVariantData> for Option<Def> {
    fn from(data: TupleVariantData) -> Option<Def> {
        Some(Def {
            kind: DefKind::Tuple,
            id: From::from(data.id),
            span: data.span,
            name: data.name,
            qualname: data.qualname,
            value: data.value,
            parent: data.parent.map(|id| From::from(id)),
            children: vec![],
            decl_id: None,
            docs: data.docs,
            sig: Some(From::from(data.sig)),
        })
    }
}
impl From<StructVariantData> for Option<Def> {
    fn from(data: StructVariantData) -> Option<Def> {
        Some(Def {
            kind: DefKind::Struct,
            id: From::from(data.id),
            span: data.span,
            name: data.name,
            qualname: data.qualname,
            value: data.value,
            parent: data.parent.map(|id| From::from(id)),
            children: vec![],
            decl_id: None,
            docs: data.docs,
            sig: Some(From::from(data.sig)),
        })
    }
}
impl From<StructData> for Option<Def> {
    fn from(data: StructData) -> Option<Def> {
        match data.visibility {
            Visibility::Public => Some(Def {
            kind: DefKind::Struct,
            id: From::from(data.id),
            span: data.span,
            name: data.name,
            qualname: data.qualname,
            value: data.value,
            parent: None,
            children: data.fields.into_iter().map(|id| From::from(id)).collect(),
            decl_id: None,
            docs: data.docs,
            sig: Some(From::from(data.sig)),
        }),
            _ => None,
        }
    }
}
impl From<TraitData> for Option<Def> {
    fn from(data: TraitData) -> Option<Def> {
        match data.visibility {
            Visibility::Public => Some(Def {
                kind: DefKind::Trait,
                id: From::from(data.id),
                span: data.span,
                name: data.name,
                qualname: data.qualname,
                value: data.value,
                children: data.items.into_iter().map(|id| From::from(id)).collect(),
                parent: None,
                decl_id: None,
                docs: data.docs,
                sig: Some(From::from(data.sig)),
            }),
            _ => None,
        }
    }
}
impl From<FunctionData> for Option<Def> {
    fn from(data: FunctionData) -> Option<Def> {
        match data.visibility {
            Visibility::Public => Some(Def {
                kind: DefKind::Function,
                id: From::from(data.id),
                span: data.span,
                name: data.name,
                qualname: data.qualname,
                value: data.value,
                children: vec![],
                parent: data.parent.map(|id| From::from(id)),
                decl_id: None,
                docs: data.docs,
                sig: Some(From::from(data.sig)),
            }),
            _ => None,
        }
    }
}
impl From<MethodData> for Option<Def> {
    fn from(data: MethodData) -> Option<Def> {
        match data.visibility {
            Visibility::Public => Some(Def {
                kind: DefKind::Method,
                id: From::from(data.id),
                span: data.span,
                name: data.name,
                qualname: data.qualname,
                value: data.value,
                children: vec![],
                parent: data.parent.map(|id| From::from(id)),
                decl_id: data.decl_id.map(|id| From::from(id)),
                docs: data.docs,
                sig: Some(From::from(data.sig)),
            }),
            _ => None,
        }
    }
}
impl From<MacroData> for Option<Def> {
    fn from(data: MacroData) -> Option<Def> {
        Some(Def {
            kind: DefKind::Macro,
            id: From::from(null_def_id()),
            span: data.span,
            name: data.name,
            qualname: data.qualname,
            value: String::new(),
            children: vec![],
            parent: None,
            decl_id: None,
            docs: data.docs,
            sig: None,
        })
    }
}
impl From<ModData> for Option<Def> {
    fn from(data:ModData) -> Option<Def> {
        match data.visibility {
            Visibility::Public => Some(Def {
                kind: DefKind::Mod,
                id: From::from(data.id),
                span: data.span,
                name: data.name,
                qualname: data.qualname,
                value: data.filename,
                children: data.items.into_iter().map(|id| From::from(id)).collect(),
                parent: None,
                decl_id: None,
                docs: data.docs,
                sig: Some(From::from(data.sig)),
            }),
            _ => None,
        }
    }
}
impl From<TypeDefData> for Option<Def> {
    fn from(data: TypeDefData) -> Option<Def> {
        match data.visibility {
            Visibility::Public => Some(Def {
                kind: DefKind::Type,
                id: From::from(data.id),
                span: data.span,
                name: data.name,
                qualname: data.qualname,
                value: data.value,
                children: vec![],
                parent: data.parent.map(|id| From::from(id)),
                decl_id: None,
                docs: String::new(),
                sig: data.sig.map(|s| From::from(s)),
            }),
            _ => None,
        }
    }
}

impl From<VariableData> for Option<Def> {
    fn from(data: VariableData) -> Option<Def> {
        match data.visibility {
            Visibility::Public => Some(Def {
                kind: match data.kind {
                    VariableKind::Static => DefKind::Static,
                    VariableKind::Const => DefKind::Const,
                    VariableKind::Local => { return None }
                    VariableKind::Field => DefKind::Field,
                },
                id: From::from(data.id),
                span: data.span,
                name: data.name,
                qualname: data.qualname,
                value: data.value,
                children: vec![],
                parent: data.parent.map(|id| From::from(id)),
                decl_id: None,
                docs: data.docs,
                sig: data.sig.map(|s| From::from(s)),
            }),
            _ => None,
        }
    }
}

#[derive(Debug, RustcEncodable)]
pub struct JsonSignature {
    span: SpanData,
    text: String,
    ident_start: usize,
    ident_end: usize,
    defs: Vec<JsonSigElement>,
    refs: Vec<JsonSigElement>,
}

impl From<Signature> for JsonSignature {
    fn from(data: Signature) -> JsonSignature {
        JsonSignature {
            span: data.span,
            text: data.text,
            ident_start: data.ident_start,
            ident_end: data.ident_end,
            defs: data.defs.into_iter().map(|s| From::from(s)).collect(),
            refs: data.refs.into_iter().map(|s| From::from(s)).collect(),
        }
    }
}

#[derive(Debug, RustcEncodable)]
pub struct JsonSigElement {
    id: Id,
    start: usize,
    end: usize,
}

impl From<SigElement> for JsonSigElement {
    fn from(data: SigElement) -> JsonSigElement {
        JsonSigElement {
            id: From::from(data.id),
            start: data.start,
            end: data.end,
        }
    }
}
