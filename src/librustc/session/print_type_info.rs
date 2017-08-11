// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use rustc_data_structures::fx::FxHashSet;
use std::mem;
use std::hash::{Hash, Hasher};
use std::collections::BTreeMap;
use rustc_serialize::json::Json;

pub struct CodeStats {
    types: FxHashSet<Type>,
}

/// A Type name, with certain modifiers applied.
///
/// While a Type can have a name like rust::u32, a ComplexTypeName
/// can include nested pointer/array modifiers:
///
/// * `*const ComplexTypeName`
/// * `[ComplexTypeName; N]`
///
/// This avoids metadata blowup.
///
/// For example: `*const [*mut [rust::u32, 12], 32]`
pub type ComplexTypeName = String;

pub struct Type {
    pub name: String,
    pub size: u64,
    pub align: u64,
    pub public: bool,
    pub kind: TypeKind,
}

pub enum TypeKind {
    PrimitiveInt,
    PrimitiveFloat,
    Opaque,
    Struct { fields: Vec<Field> },
    Union { fields: Vec<Field> },
    Enum { base_type: ComplexTypeName, cases: Vec<Case> },
}

pub struct Field {
    pub name: String,
    pub type_name: ComplexTypeName,
    pub offset: u64,
    pub public: bool,
}

pub struct Case {
    pub name: String,
    pub value: i64, // TODO: u64/u128/i128? (serialize doesn't support)
}

impl PartialEq for Type {
    fn eq(&self, other: &Self) -> bool { self.name == other.name }
}
impl Eq for Type {}
impl Hash for Type {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}


impl TypeKind {
    fn as_str(&self) -> &'static str {
        match *self {
            TypeKind::PrimitiveInt => "primitive_int",
            TypeKind::PrimitiveFloat => "primitive_float",
            TypeKind::Opaque => "opaque",
            TypeKind::Struct { .. } => "struct",
            TypeKind::Union { .. } => "union",
            TypeKind::Enum { .. } => "enum",
        }
    }
}

impl CodeStats {
    pub fn new() -> Self {
        CodeStats { types: FxHashSet::default() }
    }

    pub fn insert(&mut self, ty: Type) {
        self.types.insert(ty);
    }

    pub fn print_type_sizes(&mut self) {
        let types = mem::replace(&mut self.types, FxHashSet::default());

        let mut output = Vec::with_capacity(types.len());

        for ty in types {
            let mut json = BTreeMap::new();

            json.insert("name".to_string(), Json::String(ty.name));
            json.insert("size".to_string(), Json::U64(ty.size));
            json.insert("align".to_string(), Json::U64(ty.align));
            json.insert("public".to_string(), Json::Boolean(ty.public));
            json.insert("kind".to_string(), Json::String(ty.kind.as_str().to_string()));

            match ty.kind {
                TypeKind::Struct { fields } | TypeKind::Union { fields } => {
                    let fields_json = fields.into_iter().map(field_to_json).collect();
                    json.insert("fields".to_string(), Json::Array(fields_json));
                }
                TypeKind::Enum { base_type, cases } => {
                    json.insert("base_type".to_string(), Json::String(base_type));
                    let cases_json = cases.into_iter().map(case_to_json).collect();
                    json.insert("cases".to_string(), Json::Array(cases_json));
                }
                _ => { /* nothing */ }
            }

            output.push(Json::Object(json));
        }

        println!("WARNING: these values are platform-specific, implementation-specific, \
            and compilation-specific. They can and will change for absolutely no reason. \
            To use this properly, you must recompute and evaluate them on each compilation \
            of your crate. Yes we broke your JSON parsing just to say this. We're not \
            kidding here.");
        println!("{}", Json::Array(output));
    }
}

fn case_to_json(case: Case) -> Json {
    let mut json = BTreeMap::new();

    json.insert("name".to_string(), Json::String(case.name));
    json.insert("value".to_string(), Json::I64(case.value));

    Json::Object(json)
}

fn field_to_json(field: Field) -> Json {
    let mut json = BTreeMap::new();

    json.insert("name".to_string(), Json::String(field.name));
    json.insert("type".to_string(), Json::String(field.type_name));
    json.insert("offset".to_string(), Json::U64(field.offset));
    json.insert("public".to_string(), Json::Boolean(field.public));

    Json::Object(json)
}
