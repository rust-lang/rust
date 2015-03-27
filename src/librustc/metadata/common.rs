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

pub use self::astencode_tag::*;

use back::svh::Svh;

// RBML enum definitions and utils shared by the encoder and decoder
//
// 0x00..0x1f: reserved for RBML generic type tags
// 0x20..0xef: free for use, preferred for frequent tags
// 0xf0..0xff: internally used by RBML to encode 0x100..0xfff in two bytes
// 0x100..0xfff: free for use, preferred for infrequent tags

pub const tag_items: uint = 0x100; // top-level only

pub const tag_paths_data_name: uint = 0x20;

pub const tag_def_id: uint = 0x21;

pub const tag_items_data: uint = 0x22;

pub const tag_items_data_item: uint = 0x23;

pub const tag_items_data_item_family: uint = 0x24;

pub const tag_items_data_item_type: uint = 0x25;

pub const tag_items_data_item_symbol: uint = 0x26;

pub const tag_items_data_item_variant: uint = 0x27;

pub const tag_items_data_parent_item: uint = 0x28;

pub const tag_items_data_item_is_tuple_struct_ctor: uint = 0x29;

pub const tag_index: uint = 0x2a;

pub const tag_index_buckets: uint = 0x2b;

pub const tag_index_buckets_bucket: uint = 0x2c;

pub const tag_index_buckets_bucket_elt: uint = 0x2d;

pub const tag_index_table: uint = 0x2e;

pub const tag_meta_item_name_value: uint = 0x2f;

pub const tag_meta_item_name: uint = 0x30;

pub const tag_meta_item_value: uint = 0x31;

pub const tag_attributes: uint = 0x101; // top-level only

pub const tag_attribute: uint = 0x32;

pub const tag_meta_item_word: uint = 0x33;

pub const tag_meta_item_list: uint = 0x34;

// The list of crates that this crate depends on
pub const tag_crate_deps: uint = 0x102; // top-level only

// A single crate dependency
pub const tag_crate_dep: uint = 0x35;

pub const tag_crate_hash: uint = 0x103; // top-level only
pub const tag_crate_crate_name: uint = 0x104; // top-level only

pub const tag_crate_dep_crate_name: uint = 0x36;
pub const tag_crate_dep_hash: uint = 0x37;

pub const tag_mod_impl: uint = 0x38;

pub const tag_item_trait_item: uint = 0x39;

pub const tag_item_trait_ref: uint = 0x3a;

// discriminator value for variants
pub const tag_disr_val: uint = 0x3c;

// used to encode ast_map::PathElem
pub const tag_path: uint = 0x3d;
pub const tag_path_len: uint = 0x3e;
pub const tag_path_elem_mod: uint = 0x3f;
pub const tag_path_elem_name: uint = 0x40;
pub const tag_item_field: uint = 0x41;
pub const tag_item_field_origin: uint = 0x42;

pub const tag_item_variances: uint = 0x43;
/*
  trait items contain tag_item_trait_item elements,
  impl items contain tag_item_impl_item elements, and classes
  have both. That's because some code treats classes like traits,
  and other code treats them like impls. Because classes can contain
  both, tag_item_trait_item and tag_item_impl_item have to be two
  different tags.
 */
pub const tag_item_impl_item: uint = 0x44;
pub const tag_item_trait_method_explicit_self: uint = 0x45;


// Reexports are found within module tags. Each reexport contains def_ids
// and names.
pub const tag_items_data_item_reexport: uint = 0x46;
pub const tag_items_data_item_reexport_def_id: uint = 0x47;
pub const tag_items_data_item_reexport_name: uint = 0x48;

// used to encode crate_ctxt side tables
#[derive(Copy, PartialEq, FromPrimitive)]
#[repr(uint)]
pub enum astencode_tag { // Reserves 0x50 -- 0x6f
    tag_ast = 0x50,

    tag_tree = 0x51,

    tag_id_range = 0x52,

    tag_table = 0x53,
    // GAP 0x54, 0x55
    tag_table_def = 0x56,
    tag_table_node_type = 0x57,
    tag_table_item_subst = 0x58,
    tag_table_freevars = 0x59,
    tag_table_tcache = 0x5a,
    tag_table_param_defs = 0x5b,
    tag_table_mutbl = 0x5c,
    tag_table_last_use = 0x5d,
    tag_table_spill = 0x5e,
    tag_table_method_map = 0x5f,
    tag_table_vtable_map = 0x60,
    tag_table_adjustments = 0x61,
    tag_table_moves_map = 0x62,
    tag_table_capture_map = 0x63,
    tag_table_closure_tys = 0x64,
    tag_table_closure_kinds = 0x65,
    tag_table_upvar_capture_map = 0x66,
    tag_table_capture_modes = 0x67,
    tag_table_object_cast_map = 0x68,
    tag_table_const_qualif = 0x69,
}

pub const tag_item_trait_item_sort: uint = 0x70;

pub const tag_item_trait_parent_sort: uint = 0x71;

pub const tag_item_impl_type_basename: uint = 0x72;

pub const tag_crate_triple: uint = 0x105; // top-level only

pub const tag_dylib_dependency_formats: uint = 0x106; // top-level only

// Language items are a top-level directory (for speed). Hierarchy:
//
// tag_lang_items
// - tag_lang_items_item
//   - tag_lang_items_item_id: u32
//   - tag_lang_items_item_node_id: u32

pub const tag_lang_items: uint = 0x107; // top-level only
pub const tag_lang_items_item: uint = 0x73;
pub const tag_lang_items_item_id: uint = 0x74;
pub const tag_lang_items_item_node_id: uint = 0x75;
pub const tag_lang_items_missing: uint = 0x76;

pub const tag_item_unnamed_field: uint = 0x77;
pub const tag_items_data_item_visibility: uint = 0x78;

pub const tag_item_method_tps: uint = 0x79;
pub const tag_item_method_fty: uint = 0x7a;

pub const tag_mod_child: uint = 0x7b;
pub const tag_misc_info: uint = 0x108; // top-level only
pub const tag_misc_info_crate_items: uint = 0x7c;

pub const tag_item_method_provided_source: uint = 0x7d;
pub const tag_item_impl_vtables: uint = 0x7e;

pub const tag_impls: uint = 0x109; // top-level only
pub const tag_impls_impl: uint = 0x7f;

pub const tag_items_data_item_inherent_impl: uint = 0x80;
pub const tag_items_data_item_extension_impl: uint = 0x81;

pub const tag_native_libraries: uint = 0x10a; // top-level only
pub const tag_native_libraries_lib: uint = 0x82;
pub const tag_native_libraries_name: uint = 0x83;
pub const tag_native_libraries_kind: uint = 0x84;

pub const tag_plugin_registrar_fn: uint = 0x10b; // top-level only

pub const tag_method_argument_names: uint = 0x85;
pub const tag_method_argument_name: uint = 0x86;

pub const tag_reachable_extern_fns: uint = 0x10c; // top-level only
pub const tag_reachable_extern_fn_id: uint = 0x87;

pub const tag_items_data_item_stability: uint = 0x88;

pub const tag_items_data_item_repr: uint = 0x89;

#[derive(Clone, Debug)]
pub struct LinkMeta {
    pub crate_name: String,
    pub crate_hash: Svh,
}

pub const tag_struct_fields: uint = 0x10d; // top-level only
pub const tag_struct_field: uint = 0x8a;
pub const tag_struct_field_id: uint = 0x8b;

pub const tag_attribute_is_sugared_doc: uint = 0x8c;

pub const tag_items_data_region: uint = 0x8e;

pub const tag_region_param_def: uint = 0x8f;
pub const tag_region_param_def_ident: uint = 0x90;
pub const tag_region_param_def_def_id: uint = 0x91;
pub const tag_region_param_def_space: uint = 0x92;
pub const tag_region_param_def_index: uint = 0x93;

pub const tag_type_param_def: uint = 0x94;

pub const tag_item_generics: uint = 0x95;
pub const tag_method_ty_generics: uint = 0x96;

pub const tag_predicate: uint = 0x97;
pub const tag_predicate_space: uint = 0x98;
pub const tag_predicate_data: uint = 0x99;

pub const tag_unsafety: uint = 0x9a;

pub const tag_associated_type_names: uint = 0x9b;
pub const tag_associated_type_name: uint = 0x9c;

pub const tag_polarity: uint = 0x9d;

pub const tag_macro_defs: uint = 0x10e; // top-level only
pub const tag_macro_def: uint = 0x9e;
pub const tag_macro_def_body: uint = 0x9f;

pub const tag_paren_sugar: uint = 0xa0;

pub const tag_codemap: uint = 0xa1;
pub const tag_codemap_filemap: uint = 0xa2;

pub const tag_item_super_predicates: uint = 0xa3;

pub const tag_defaulted_trait: uint = 0xa4;

pub const tag_impl_coerce_unsized_kind: uint = 0xa5;
