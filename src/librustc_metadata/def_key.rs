// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::def_id::DefIndex;
use rustc::hir::map as hir_map;
use syntax::parse::token::InternedString;

#[derive(RustcEncodable, RustcDecodable)]
pub struct DefKey {
    pub parent: Option<DefIndex>,
    pub disambiguated_data: DisambiguatedDefPathData,
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct DisambiguatedDefPathData {
    pub data: DefPathData,
    pub disambiguator: u32,
}

#[derive(RustcEncodable, RustcDecodable)]
pub enum DefPathData {
    CrateRoot,
    Misc,
    Impl,
    TypeNs,
    ValueNs,
    Module,
    MacroDef,
    ClosureExpr,
    TypeParam,
    LifetimeDef,
    EnumVariant,
    Field,
    StructCtor,
    Initializer,
    Binding,
    ImplTrait,
}

pub fn simplify_def_key(key: hir_map::DefKey) -> DefKey {
    let data = DisambiguatedDefPathData {
        data: simplify_def_path_data(key.disambiguated_data.data),
        disambiguator: key.disambiguated_data.disambiguator,
    };
    DefKey {
        parent: key.parent,
        disambiguated_data: data,
    }
}

fn simplify_def_path_data(data: hir_map::DefPathData) -> DefPathData {
    match data {
        hir_map::DefPathData::CrateRoot => DefPathData::CrateRoot,
        hir_map::DefPathData::InlinedRoot(_) => bug!("unexpected DefPathData"),
        hir_map::DefPathData::Misc => DefPathData::Misc,
        hir_map::DefPathData::Impl => DefPathData::Impl,
        hir_map::DefPathData::TypeNs(_) => DefPathData::TypeNs,
        hir_map::DefPathData::ValueNs(_) => DefPathData::ValueNs,
        hir_map::DefPathData::Module(_) => DefPathData::Module,
        hir_map::DefPathData::MacroDef(_) => DefPathData::MacroDef,
        hir_map::DefPathData::ClosureExpr => DefPathData::ClosureExpr,
        hir_map::DefPathData::TypeParam(_) => DefPathData::TypeParam,
        hir_map::DefPathData::LifetimeDef(_) => DefPathData::LifetimeDef,
        hir_map::DefPathData::EnumVariant(_) => DefPathData::EnumVariant,
        hir_map::DefPathData::Field(_) => DefPathData::Field,
        hir_map::DefPathData::StructCtor => DefPathData::StructCtor,
        hir_map::DefPathData::Initializer => DefPathData::Initializer,
        hir_map::DefPathData::Binding(_) => DefPathData::Binding,
        hir_map::DefPathData::ImplTrait => DefPathData::ImplTrait,
    }
}

pub fn recover_def_key(key: DefKey, name: Option<InternedString>) -> hir_map::DefKey {
    let data = hir_map::DisambiguatedDefPathData {
        data: recover_def_path_data(key.disambiguated_data.data, name),
        disambiguator: key.disambiguated_data.disambiguator,
    };
    hir_map::DefKey {
        parent: key.parent,
        disambiguated_data: data,
    }
}

fn recover_def_path_data(data: DefPathData, name: Option<InternedString>) -> hir_map::DefPathData {
    match data {
        DefPathData::CrateRoot => hir_map::DefPathData::CrateRoot,
        DefPathData::Misc => hir_map::DefPathData::Misc,
        DefPathData::Impl => hir_map::DefPathData::Impl,
        DefPathData::TypeNs => hir_map::DefPathData::TypeNs(name.unwrap()),
        DefPathData::ValueNs => hir_map::DefPathData::ValueNs(name.unwrap()),
        DefPathData::Module => hir_map::DefPathData::Module(name.unwrap()),
        DefPathData::MacroDef => hir_map::DefPathData::MacroDef(name.unwrap()),
        DefPathData::ClosureExpr => hir_map::DefPathData::ClosureExpr,
        DefPathData::TypeParam => hir_map::DefPathData::TypeParam(name.unwrap()),
        DefPathData::LifetimeDef => hir_map::DefPathData::LifetimeDef(name.unwrap()),
        DefPathData::EnumVariant => hir_map::DefPathData::EnumVariant(name.unwrap()),
        DefPathData::Field => hir_map::DefPathData::Field(name.unwrap()),
        DefPathData::StructCtor => hir_map::DefPathData::StructCtor,
        DefPathData::Initializer => hir_map::DefPathData::Initializer,
        DefPathData::Binding => hir_map::DefPathData::Binding(name.unwrap()),
        DefPathData::ImplTrait => hir_map::DefPathData::ImplTrait,
    }
}
