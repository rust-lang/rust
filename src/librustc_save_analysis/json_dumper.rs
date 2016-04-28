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
use syntax::codemap::CodeMap;

use syntax::ast::CrateNum;

use super::data::{self, SpanData};
use super::dump::Dump;

pub struct JsonDumper<'a, 'b, W: Write + 'b> {
    output: &'b mut W,
    codemap: &'a CodeMap,
    first: bool,
}

impl<'a, 'b, W: Write> JsonDumper<'a, 'b, W> {
    pub fn new(writer: &'b mut W, codemap: &'a CodeMap) -> JsonDumper<'a, 'b, W> {
        if let Err(_) = write!(writer, "[") {
            error!("Error writing output");
        }
        JsonDumper { output: writer, codemap:codemap, first: true }
    }
}

impl<'a, 'b, W: Write> Drop for JsonDumper<'a, 'b, W> {
    fn drop(&mut self) {
        if let Err(_) = write!(self.output, "]") {
            error!("Error writing output");
        }
    }
}

macro_rules! impl_fn {
    ($fn_name: ident, $data_type: ident) => {
        fn $fn_name(&mut self, data: data::$data_type) {
            if self.first {
                self.first = false;
            } else {
                if let Err(_) = write!(self.output, ",") {
                    error!("Error writing output");
                }
            }
            let data = data.lower(self.codemap);
            if let Err(_) = write!(self.output, "{}", as_json(&data)) {
                error!("Error writing output '{}'", as_json(&data));
            }
        }
    }
}

impl<'a, 'b, W: Write + 'b> Dump for JsonDumper<'a, 'b, W> {
    impl_fn!(crate_prelude, CratePreludeData);
    impl_fn!(enum_data, EnumData);
    impl_fn!(extern_crate, ExternCrateData);
    impl_fn!(impl_data, ImplData);
    impl_fn!(inheritance, InheritanceData);
    impl_fn!(function, FunctionData);
    impl_fn!(function_ref, FunctionRefData);
    impl_fn!(function_call, FunctionCallData);
    impl_fn!(method, MethodData);
    impl_fn!(method_call, MethodCallData);
    impl_fn!(macro_data, MacroData);
    impl_fn!(macro_use, MacroUseData);
    impl_fn!(mod_data, ModData);
    impl_fn!(mod_ref, ModRefData);
    impl_fn!(struct_data, StructData);
    impl_fn!(struct_variant, StructVariantData);
    impl_fn!(trait_data, TraitData);
    impl_fn!(tuple_variant, TupleVariantData);
    impl_fn!(type_ref, TypeRefData);
    impl_fn!(typedef, TypedefData);
    impl_fn!(use_data, UseData);
    impl_fn!(use_glob, UseGlobData);
    impl_fn!(variable, VariableData);
    impl_fn!(variable_ref, VariableRefData);
}

trait Lower {
    type Target;
    fn lower(self, cm: &CodeMap) -> Self::Target;
}

pub type Id = u32;

#[derive(Debug, RustcEncodable)]
pub struct CratePreludeData {
    pub crate_name: String,
    pub crate_root: String,
    pub external_crates: Vec<data::ExternalCrateData>,
    pub span: SpanData,
}

impl Lower for data::CratePreludeData {
    type Target = CratePreludeData;

    fn lower(self, cm: &CodeMap) -> CratePreludeData {
        CratePreludeData {
            crate_name: self.crate_name,
            crate_root: self.crate_root,
            external_crates: self.external_crates,
            span: SpanData::from_span(self.span, cm),
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

    fn lower(self, cm: &CodeMap) -> EnumData {
        EnumData {
            id: self.id,
            value: self.value,
            qualname: self.qualname,
            span: SpanData::from_span(self.span, cm),
            scope: self.scope,
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

    fn lower(self, cm: &CodeMap) -> ExternCrateData {
        ExternCrateData {
            id: self.id,
            name: self.name,
            crate_num: self.crate_num,
            location: self.location,
            span: SpanData::from_span(self.span, cm),
            scope: self.scope,
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

    fn lower(self, cm: &CodeMap) -> FunctionCallData {
        FunctionCallData {
            span: SpanData::from_span(self.span, cm),
            scope: self.scope,
            ref_id: self.ref_id.index.as_u32(),
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

    fn lower(self, cm: &CodeMap) -> FunctionData {
        FunctionData {
            id: self.id,
            name: self.name,
            qualname: self.qualname,
            declaration: self.declaration.map(|id| id.index.as_u32()),
            span: SpanData::from_span(self.span, cm),
            scope: self.scope,
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

    fn lower(self, cm: &CodeMap) -> FunctionRefData {
        FunctionRefData {
            span: SpanData::from_span(self.span, cm),
            scope: self.scope,
            ref_id: self.ref_id.index.as_u32(),
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

    fn lower(self, cm: &CodeMap) -> ImplData {
        ImplData {
            id: self.id,
            span: SpanData::from_span(self.span, cm),
            scope: self.scope,
            trait_ref: self.trait_ref.map(|id| id.index.as_u32()),
            self_ref: self.self_ref.map(|id| id.index.as_u32()),
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

    fn lower(self, cm: &CodeMap) -> InheritanceData {
        InheritanceData {
            span: SpanData::from_span(self.span, cm),
            base_id: self.base_id.index.as_u32(),
            deriv_id: self.deriv_id
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

    fn lower(self, cm: &CodeMap) -> MacroData {
        MacroData {
            span: SpanData::from_span(self.span, cm),
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

    fn lower(self, cm: &CodeMap) -> MacroUseData {
        MacroUseData {
            span: SpanData::from_span(self.span, cm),
            name: self.name,
            qualname: self.qualname,
            callee_span: SpanData::from_span(self.callee_span, cm),
            scope: self.scope,
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

    fn lower(self, cm: &CodeMap) -> MethodCallData {
        MethodCallData {
            span: SpanData::from_span(self.span, cm),
            scope: self.scope,
            ref_id: self.ref_id.map(|id| id.index.as_u32()),
            decl_id: self.decl_id.map(|id| id.index.as_u32()),
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

    fn lower(self, cm: &CodeMap) -> MethodData {
        MethodData {
            span: SpanData::from_span(self.span, cm),
            scope: self.scope,
            id: self.id,
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

    fn lower(self, cm: &CodeMap) -> ModData {
        ModData {
            id: self.id,
            name: self.name,
            qualname: self.qualname,
            span: SpanData::from_span(self.span, cm),
            scope: self.scope,
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

    fn lower(self, cm: &CodeMap) -> ModRefData {
        ModRefData {
            span: SpanData::from_span(self.span, cm),
            scope: self.scope,
            ref_id: self.ref_id.map(|id| id.index.as_u32()),
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

    fn lower(self, cm: &CodeMap) -> StructData {
        StructData {
            span: SpanData::from_span(self.span, cm),
            id: self.id,
            ctor_id: self.ctor_id,
            qualname: self.qualname,
            scope: self.scope,
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

    fn lower(self, cm: &CodeMap) -> StructVariantData {
        StructVariantData {
            span: SpanData::from_span(self.span, cm),
            id: self.id,
            qualname: self.qualname,
            type_value: self.type_value,
            value: self.value,
            scope: self.scope,
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

    fn lower(self, cm: &CodeMap) -> TraitData {
        TraitData {
            span: SpanData::from_span(self.span, cm),
            id: self.id,
            qualname: self.qualname,
            scope: self.scope,
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

    fn lower(self, cm: &CodeMap) -> TupleVariantData {
        TupleVariantData {
            span: SpanData::from_span(self.span, cm),
            id: self.id,
            name: self.name,
            qualname: self.qualname,
            type_value: self.type_value,
            value: self.value,
            scope: self.scope,
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

    fn lower(self, cm: &CodeMap) -> TypedefData {
        TypedefData {
            id: self.id,
            span: SpanData::from_span(self.span, cm),
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

    fn lower(self, cm: &CodeMap) -> TypeRefData {
        TypeRefData {
            span: SpanData::from_span(self.span, cm),
            scope: self.scope,
            ref_id: self.ref_id.map(|id| id.index.as_u32()),
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

    fn lower(self, cm: &CodeMap) -> UseData {
        UseData {
            id: self.id,
            span: SpanData::from_span(self.span, cm),
            name: self.name,
            mod_id: self.mod_id.map(|id| id.index.as_u32()),
            scope: self.scope,
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

    fn lower(self, cm: &CodeMap) -> UseGlobData {
        UseGlobData {
            id: self.id,
            span: SpanData::from_span(self.span, cm),
            names: self.names,
            scope: self.scope,
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

    fn lower(self, cm: &CodeMap) -> VariableData {
        VariableData {
            id: self.id,
            name: self.name,
            qualname: self.qualname,
            span: SpanData::from_span(self.span, cm),
            scope: self.scope,
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

    fn lower(self, cm: &CodeMap) -> VariableRefData {
        VariableRefData {
            name: self.name,
            span: SpanData::from_span(self.span, cm),
            scope: self.scope,
            ref_id: self.ref_id.index.as_u32(),
        }
    }
}
