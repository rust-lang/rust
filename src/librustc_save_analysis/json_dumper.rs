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

use super::external_data::*;
use super::dump::Dump;

pub struct JsonDumper<'b, W: Write + 'b> {
    output: &'b mut W,
    result: Analysis,
}

impl<'b, W: Write> JsonDumper<'b, W> {
    pub fn new(writer: &'b mut W) -> JsonDumper<'b, W> {
        JsonDumper { output: writer, result: Analysis::new() }
    }
}

impl<'b, W: Write> Drop for JsonDumper<'b, W> {
    fn drop(&mut self) {
        if let Err(_) = write!(self.output, "{}", as_json(&self.result)) {
            error!("Error writing output");
        }
    }
}

macro_rules! impl_fn {
    ($fn_name: ident, $data_type: ident, $bucket: ident) => {
        fn $fn_name(&mut self, data: $data_type) {
            self.result.$bucket.push(From::from(data));
        }
    }
}

impl<'b, W: Write + 'b> Dump for JsonDumper<'b, W> {
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
    impl_fn!(mod_data, ModData, defs);
    impl_fn!(typedef, TypeDefData, defs);
    impl_fn!(variable, VariableData, defs);

    impl_fn!(function_ref, FunctionRefData, refs);
    impl_fn!(function_call, FunctionCallData, refs);
    impl_fn!(method_call, MethodCallData, refs);
    impl_fn!(mod_ref, ModRefData, refs);
    impl_fn!(type_ref, TypeRefData, refs);
    impl_fn!(variable_ref, VariableRefData, refs);

    impl_fn!(macro_use, MacroUseData, macro_refs);

    // FIXME store this instead of throwing it away.
    fn impl_data(&mut self, _data: ImplData) {}
    fn inheritance(&mut self, _data: InheritanceData) {}
}

// FIXME do we want to change ExternalData to this mode? It will break DXR.
// FIXME methods. The defs have information about possible overriding and the
// refs have decl information (e.g., a trait method where we know the required
// method, but not the supplied method). In both cases, we are currently
// ignoring it.

#[derive(Debug, RustcEncodable)]
struct Analysis {
    prelude: Option<CratePreludeData>,
    imports: Vec<Import>,
    defs: Vec<Def>,
    refs: Vec<Ref>,
    macro_refs: Vec<MacroRef>,
}

impl Analysis {
    fn new() -> Analysis {
        Analysis {
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
            krate: id.krate,
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
    ExternCrate,
    Use,
    GlobUse,
}

impl From<ExternCrateData> for Import {
    fn from(data: ExternCrateData) -> Import {
        Import {
            kind: ImportKind::ExternCrate,
            id: From::from(data.id),
            span: data.span,
            name: data.name,
            value: String::new(),
        }
    }
}
impl From<UseData> for Import {
    fn from(data: UseData) -> Import {
        Import {
            kind: ImportKind::Use,
            id: From::from(data.id),
            span: data.span,
            name: data.name,
            value: String::new(),
        }
    }
}
impl From<UseGlobData> for Import {
    fn from(data: UseGlobData) -> Import {
        Import {
            kind: ImportKind::GlobUse,
            id: From::from(data.id),
            span: data.span,
            name: "*".to_owned(),
            value: data.names.join(", "),
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
    // No id, no value.
    Macro,
    // value = file_name
    Mod,
    // value = aliased type
    Type,
    // value = type and init expression
    Variable,
}

impl From<EnumData> for Def {
    fn from(data: EnumData) -> Def {
        Def {
            kind: DefKind::Enum,
            id: From::from(data.id),
            span: data.span,
            name: data.name,
            qualname: data.qualname,
            value: data.value,
        }
    }
}

impl From<TupleVariantData> for Def {
    fn from(data: TupleVariantData) -> Def {
        Def {
            kind: DefKind::Tuple,
            id: From::from(data.id),
            span: data.span,
            name: data.name,
            qualname: data.qualname,
            value: data.value,
        }
    }
}
impl From<StructVariantData> for Def {
    fn from(data: StructVariantData) -> Def {
        Def {
            kind: DefKind::Struct,
            id: From::from(data.id),
            span: data.span,
            name: data.name,
            qualname: data.qualname,
            value: data.value,
        }
    }
}
impl From<StructData> for Def {
    fn from(data: StructData) -> Def {
        Def {
            kind: DefKind::Struct,
            id: From::from(data.id),
            span: data.span,
            name: data.name,
            qualname: data.qualname,
            value: data.value,
        }
    }
}
impl From<TraitData> for Def {
    fn from(data: TraitData) -> Def {
        Def {
            kind: DefKind::Trait,
            id: From::from(data.id),
            span: data.span,
            name: data.name,
            qualname: data.qualname,
            value: data.value,
        }
    }
}
impl From<FunctionData> for Def {
    fn from(data: FunctionData) -> Def {
        Def {
            kind: DefKind::Function,
            id: From::from(data.id),
            span: data.span,
            name: data.name,
            qualname: data.qualname,
            value: data.value,
        }
    }
}
impl From<MethodData> for Def {
    fn from(data: MethodData) -> Def {
        Def {
            kind: DefKind::Function,
            id: From::from(data.id),
            span: data.span,
            name: data.name,
            qualname: data.qualname,
            value: data.value,
        }
    }
}
impl From<MacroData> for Def {
    fn from(data: MacroData) -> Def {
        Def {
            kind: DefKind::Macro,
            id: From::from(null_def_id()),
            span: data.span,
            name: data.name,
            qualname: data.qualname,
            value: String::new(),
        }
    }
}
impl From<ModData> for Def {
    fn from(data:ModData) -> Def {
        Def {
            kind: DefKind::Mod,
            id: From::from(data.id),
            span: data.span,
            name: data.name,
            qualname: data.qualname,
            value: data.filename,
        }
    }
}
impl From<TypeDefData> for Def {
    fn from(data: TypeDefData) -> Def {
        Def {
            kind: DefKind::Type,
            id: From::from(data.id),
            span: data.span,
            name: data.name,
            qualname: data.qualname,
            value: data.value,
        }
    }
}
impl From<VariableData> for Def {
    fn from(data: VariableData) -> Def {
        Def {
            kind: DefKind::Variable,
            id: From::from(data.id),
            span: data.span,
            name: data.name,
            qualname: data.qualname,
            value: data.value,
        }
    }
}

#[derive(Debug, RustcEncodable)]
enum RefKind {
    Function,
    Mod,
    Type,
    Variable,
}

#[derive(Debug, RustcEncodable)]
struct Ref {
    kind: RefKind,
    span: SpanData,
    ref_id: Id,
}

impl From<FunctionRefData> for Ref {
    fn from(data: FunctionRefData) -> Ref {
        Ref {
            kind: RefKind::Function,
            span: data.span,
            ref_id: From::from(data.ref_id),
        }
    }
}
impl From<FunctionCallData> for Ref {
    fn from(data: FunctionCallData) -> Ref {
        Ref {
            kind: RefKind::Function,
            span: data.span,
            ref_id: From::from(data.ref_id),
        }
    }
}
impl From<MethodCallData> for Ref {
    fn from(data: MethodCallData) -> Ref {
        Ref {
            kind: RefKind::Function,
            span: data.span,
            ref_id: From::from(data.ref_id.or(data.decl_id).unwrap_or(null_def_id())),
        }
    }
}
impl From<ModRefData> for Ref {
    fn from(data: ModRefData) -> Ref {
        Ref {
            kind: RefKind::Mod,
            span: data.span,
            ref_id: From::from(data.ref_id.unwrap_or(null_def_id())),
        }
    }
}
impl From<TypeRefData> for Ref {
    fn from(data: TypeRefData) -> Ref {
        Ref {
            kind: RefKind::Type,
            span: data.span,
            ref_id: From::from(data.ref_id.unwrap_or(null_def_id())),
        }
    }
}
impl From<VariableRefData> for Ref {
    fn from(data: VariableRefData) -> Ref {
        Ref {
            kind: RefKind::Variable,
            span: data.span,
            ref_id: From::from(data.ref_id),
        }
    }
}

#[derive(Debug, RustcEncodable)]
struct MacroRef {
    span: SpanData,
    qualname: String,
    callee_span: SpanData,
}

impl From<MacroUseData> for MacroRef {
    fn from(data: MacroUseData) -> MacroRef {
        MacroRef {
            span: data.span,
            qualname: data.qualname,
            callee_span: data.callee_span,
        }
    }
}
