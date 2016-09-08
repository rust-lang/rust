// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types, non_upper_case_globals)]

use rustc::hir;
use rustc::hir::def;
use rustc::hir::def_id::{DefIndex, DefId};
use rustc::ty;
use rustc::session::config::PanicStrategy;

#[derive(Clone, Copy, Debug, PartialEq, RustcEncodable, RustcDecodable)]
pub enum Family {
    ImmStatic,
    MutStatic,
    ForeignImmStatic,
    ForeignMutStatic,
    Fn,
    ForeignFn,
    Method,
    AssociatedType,
    Type,
    Mod,
    ForeignMod,
    Enum,
    Variant,
    Impl,
    DefaultImpl,
    Trait,
    Struct,
    Union,
    Field,
    Const,
    AssociatedConst,
    Closure
}

// NB: increment this if you change the format of metadata such that
// rustc_version can't be found.
pub const metadata_encoding_version : &'static [u8] = &[b'r', b'u', b's', b't', 0, 0, 0, 2];

// GAP 0x7c
// GAP 0x108
pub fn rustc_version() -> String {
    format!(
        "rustc {}",
        option_env!("CFG_VERSION").unwrap_or("unknown version")
    )
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct CrateInfo {
    pub name: String,
    pub triple: String,
    pub hash: hir::svh::Svh,
    pub disambiguator: String,
    pub panic_strategy: PanicStrategy,
    pub plugin_registrar_fn: Option<DefIndex>,
    pub macro_derive_registrar: Option<DefIndex>
}

pub mod root_tag {
    pub const rustc_version: usize = 0x10f;

    pub const crate_info: usize = 0x104;

    pub const index: usize = 0x110;
    pub const crate_deps: usize = 0x102;
    pub const dylib_dependency_formats: usize = 0x106;
    pub const native_libraries: usize = 0x10a;
    pub const lang_items: usize = 0x107;
    pub const lang_items_missing: usize = 0x76;
    pub const impls: usize = 0x109;
    pub const reachable_ids: usize = 0x10c;
    pub const macro_defs: usize = 0x10e;
    pub const codemap: usize = 0xa1;
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct ModData {
    pub reexports: Vec<def::Export>
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct VariantData {
    pub kind: ty::VariantKind,
    pub disr: u64,

    /// If this is a struct's only variant, this
    /// is the index of the "struct ctor" item.
    pub struct_ctor: Option<DefIndex>
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct TraitData {
    pub unsafety: hir::Unsafety,
    pub paren_sugar: bool,
    pub has_default_impl: bool
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct ImplData {
    pub polarity: hir::ImplPolarity,
    pub parent_impl: Option<DefId>,
    pub coerce_unsized_kind: Option<ty::adjustment::CustomCoerceUnsized>,
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct TraitAssociatedData {
    pub has_default: bool
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct ImplAssociatedData {
    pub defaultness: hir::Defaultness,
    pub constness: hir::Constness
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct FnData {
    pub constness: hir::Constness
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct ClosureData {
    pub kind: ty::ClosureKind
}

#[derive(RustcEncodable, RustcDecodable)]
pub enum EntryData {
    Other,
    Mod(ModData),
    Variant(VariantData),
    Trait(TraitData),
    Impl(ImplData),
    TraitAssociated(TraitAssociatedData),
    ImplAssociated(ImplAssociatedData),
    Fn(FnData),
    Closure(ClosureData)
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct TraitTypedData<'tcx> {
    pub trait_ref: ty::TraitRef<'tcx>
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct ImplTypedData<'tcx> {
    pub trait_ref: Option<ty::TraitRef<'tcx>>
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct MethodTypedData<'tcx> {
    pub explicit_self: ty::ExplicitSelfCategory<'tcx>
}

#[derive(RustcEncodable, RustcDecodable)]
pub struct ClosureTypedData<'tcx> {
    pub ty: ty::ClosureTy<'tcx>
}

#[derive(RustcEncodable, RustcDecodable)]
pub enum EntryTypedData<'tcx> {
    Other,
    Trait(TraitTypedData<'tcx>),
    Impl(ImplTypedData<'tcx>),
    Method(MethodTypedData<'tcx>),
    Closure(ClosureTypedData<'tcx>)
}

pub mod item_tag {
    pub const def_key: usize = 0x2c;
    pub const family: usize = 0x24;
    pub const attributes: usize = 0x101;
    pub const visibility: usize = 0x78;
    pub const children: usize = 0x7b;
    pub const stability: usize = 0x88;
    pub const deprecation: usize = 0xa7;

    pub const ty: usize = 0x25;
    pub const inherent_impls: usize = 0x79;
    pub const variances: usize = 0x43;
    pub const generics: usize = 0x8f;
    pub const predicates: usize = 0x95;
    pub const super_predicates: usize = 0xa3;

    pub const ast: usize = 0x50;
    pub const mir: usize = 0x52;

    pub const data: usize = 0x3c;
    pub const typed_data: usize = 0x3d;

    pub const fn_arg_names: usize = 0x85;
}

/// The shorthand encoding uses an enum's variant index `usize`
/// and is offset by this value so it never matches a real variant.
/// This offset is also chosen so that the first byte is never < 0x80.
pub const SHORTHAND_OFFSET: usize = 0x80;
