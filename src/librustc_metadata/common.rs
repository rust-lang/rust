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

use rustc::ty;

#[derive(Clone, Copy, Debug, PartialEq, RustcEncodable, RustcDecodable)]
pub enum Family {
    ImmStatic,
    MutStatic,
    Fn,
    Method,
    AssociatedType,
    Type,
    Mod,
    ForeignMod,
    Enum,
    Variant(ty::VariantKind),
    Impl,
    DefaultImpl,
    Trait,
    Struct(ty::VariantKind),
    Union,
    PublicField,
    InheritedField,
    Const,
    AssociatedConst,
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

pub mod root_tag {
    pub const rustc_version: usize = 0x10f;
    pub const crate_deps: usize = 0x102;
    pub const crate_hash: usize = 0x103;
    pub const crate_crate_name: usize = 0x104;
    pub const crate_disambiguator: usize = 0x113;
    pub const items: usize = 0x100;
    pub const index: usize = 0x110;
    pub const xref_index: usize = 0x111;
    pub const xref_data: usize = 0x112;
    pub const crate_triple: usize = 0x105;
    pub const dylib_dependency_formats: usize = 0x106;
    pub const lang_items: usize = 0x107;
    pub const lang_items_missing: usize = 0x76;
    pub const impls: usize = 0x109;
    pub const native_libraries: usize = 0x10a;
    pub const plugin_registrar_fn: usize = 0x10b;
    pub const panic_strategy: usize = 0x114;
    pub const macro_derive_registrar: usize = 0x115;
    pub const reachable_ids: usize = 0x10c;
    pub const macro_defs: usize = 0x10e;
    pub const codemap: usize = 0xa1;
}

pub mod item_tag {
    pub const name: usize = 0x20;
    pub const def_index: usize = 0x21;
    pub const family: usize = 0x24;
    pub const ty: usize = 0x25;
    pub const parent_item: usize = 0x28;
    pub const is_tuple_struct_ctor: usize = 0x29;
    pub const closure_kind: usize = 0x2a;
    pub const closure_ty: usize = 0x2b;
    pub const def_key: usize = 0x2c;
    pub const attributes: usize = 0x101;
    pub const trait_ref: usize = 0x3b;
    pub const disr_val: usize = 0x3c;
    pub const fields: usize = 0x41;
    pub const variances: usize = 0x43;
    pub const trait_method_explicit_self: usize = 0x45;
    pub const ast: usize = 0x50;
    pub const mir: usize = 0x52;
    pub const trait_item_has_body: usize = 0x70;
    pub const visibility: usize = 0x78;
    pub const inherent_impls: usize = 0x79;
    pub const children: usize = 0x7b;
    pub const method_argument_names: usize = 0x85;
    pub const stability: usize = 0x88;
    pub const repr: usize = 0x89;
    pub const struct_ctor: usize = 0x8b;
    pub const generics: usize = 0x8f;
    pub const predicates: usize = 0x95;
    pub const unsafety: usize = 0x9a;
    pub const polarity: usize = 0x9d;
    pub const paren_sugar: usize = 0xa0;
    pub const super_predicates: usize = 0xa3;
    pub const defaulted_trait: usize = 0xa4;
    pub const impl_coerce_unsized_kind: usize = 0xa5;
    pub const constness: usize = 0xa6;
    pub const deprecation: usize = 0xa7;
    pub const defaultness: usize = 0xa8;
    pub const parent_impl: usize = 0xa9;
}

/// The shorthand encoding of `Ty` uses `TypeVariants`' variant `usize`
/// and is offset by this value so it never matches a real variant.
/// This offset is also chosen so that the first byte is never < 0x80.
pub const TYPE_SHORTHAND_OFFSET: usize = 0x80;
