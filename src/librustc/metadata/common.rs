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

pub const tag_items: usize = 0x100; // top-level only

pub const tag_paths_data_name: usize = 0x20;

pub const tag_def_id: usize = 0x21;

pub const tag_items_data: usize = 0x22;

pub const tag_items_data_item: usize = 0x23;

pub const tag_items_data_item_family: usize = 0x24;

pub const tag_items_data_item_type: usize = 0x25;

pub const tag_items_data_item_symbol: usize = 0x26;

pub const tag_items_data_item_variant: usize = 0x27;

pub const tag_items_data_parent_item: usize = 0x28;

pub const tag_items_data_item_is_tuple_struct_ctor: usize = 0x29;

pub const tag_index: usize = 0x2a;

pub const tag_index_buckets: usize = 0x2b;

pub const tag_index_buckets_bucket: usize = 0x2c;

pub const tag_index_buckets_bucket_elt: usize = 0x2d;

pub const tag_index_table: usize = 0x2e;

pub const tag_meta_item_name_value: usize = 0x2f;

pub const tag_meta_item_name: usize = 0x30;

pub const tag_meta_item_value: usize = 0x31;

pub const tag_attributes: usize = 0x101; // top-level only

pub const tag_attribute: usize = 0x32;

pub const tag_meta_item_word: usize = 0x33;

pub const tag_meta_item_list: usize = 0x34;

// The list of crates that this crate depends on
pub const tag_crate_deps: usize = 0x102; // top-level only

// A single crate dependency
pub const tag_crate_dep: usize = 0x35;

pub const tag_crate_hash: usize = 0x103; // top-level only
pub const tag_crate_crate_name: usize = 0x104; // top-level only

pub const tag_crate_dep_crate_name: usize = 0x36;
pub const tag_crate_dep_hash: usize = 0x37;

pub const tag_mod_impl: usize = 0x38;

pub const tag_item_trait_item: usize = 0x39;

pub const tag_item_trait_ref: usize = 0x3a;

// discriminator value for variants
pub const tag_disr_val: usize = 0x3c;

// used to encode ast_map::PathElem
pub const tag_path: usize = 0x3d;
pub const tag_path_len: usize = 0x3e;
pub const tag_path_elem_mod: usize = 0x3f;
pub const tag_path_elem_name: usize = 0x40;
pub const tag_item_field: usize = 0x41;
pub const tag_item_field_origin: usize = 0x42;

pub const tag_item_variances: usize = 0x43;
/*
  trait items contain tag_item_trait_item elements,
  impl items contain tag_item_impl_item elements, and classes
  have both. That's because some code treats classes like traits,
  and other code treats them like impls. Because classes can contain
  both, tag_item_trait_item and tag_item_impl_item have to be two
  different tags.
 */
pub const tag_item_impl_item: usize = 0x44;
pub const tag_item_trait_method_explicit_self: usize = 0x45;


// Reexports are found within module tags. Each reexport contains def_ids
// and names.
pub const tag_items_data_item_reexport: usize = 0x46;
pub const tag_items_data_item_reexport_def_id: usize = 0x47;
pub const tag_items_data_item_reexport_name: usize = 0x48;

// used to encode crate_ctxt side tables
#[derive(Copy, PartialEq, FromPrimitive)]
#[repr(usize)]
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

pub const tag_item_trait_item_sort: usize = 0x70;

pub const tag_item_trait_parent_sort: usize = 0x71;

pub const tag_item_impl_type_basename: usize = 0x72;

pub const tag_crate_triple: usize = 0x105; // top-level only

pub const tag_dylib_dependency_formats: usize = 0x106; // top-level only

// Language items are a top-level directory (for speed). Hierarchy:
//
// tag_lang_items
// - tag_lang_items_item
//   - tag_lang_items_item_id: u32
//   - tag_lang_items_item_node_id: u32

pub const tag_lang_items: usize = 0x107; // top-level only
pub const tag_lang_items_item: usize = 0x73;
pub const tag_lang_items_item_id: usize = 0x74;
pub const tag_lang_items_item_node_id: usize = 0x75;
pub const tag_lang_items_missing: usize = 0x76;

pub const tag_item_unnamed_field: usize = 0x77;
pub const tag_items_data_item_visibility: usize = 0x78;

pub const tag_item_method_tps: usize = 0x79;
pub const tag_item_method_fty: usize = 0x7a;

pub const tag_mod_child: usize = 0x7b;
pub const tag_misc_info: usize = 0x108; // top-level only
pub const tag_misc_info_crate_items: usize = 0x7c;

pub const tag_item_method_provided_source: usize = 0x7d;
pub const tag_item_impl_vtables: usize = 0x7e;

pub const tag_impls: usize = 0x109; // top-level only
pub const tag_impls_impl: usize = 0x7f;

pub const tag_items_data_item_inherent_impl: usize = 0x80;
pub const tag_items_data_item_extension_impl: usize = 0x81;

pub const tag_native_libraries: usize = 0x10a; // top-level only
pub const tag_native_libraries_lib: usize = 0x82;
pub const tag_native_libraries_name: usize = 0x83;
pub const tag_native_libraries_kind: usize = 0x84;

pub const tag_plugin_registrar_fn: usize = 0x10b; // top-level only

pub const tag_method_argument_names: usize = 0x85;
pub const tag_method_argument_name: usize = 0x86;

pub const tag_reachable_extern_fns: usize = 0x10c; // top-level only
pub const tag_reachable_extern_fn_id: usize = 0x87;

pub const tag_items_data_item_stability: usize = 0x88;

pub const tag_items_data_item_repr: usize = 0x89;

#[derive(Clone, Debug)]
pub struct LinkMeta {
    pub crate_name: String,
    pub crate_hash: Svh,
}

pub const tag_struct_fields: usize = 0x10d; // top-level only
pub const tag_struct_field: usize = 0x8a;
pub const tag_struct_field_id: usize = 0x8b;

pub const tag_attribute_is_sugared_doc: usize = 0x8c;

pub const tag_items_data_region: usize = 0x8e;

pub const tag_region_param_def: usize = 0x8f;
pub const tag_region_param_def_ident: usize = 0x90;
pub const tag_region_param_def_def_id: usize = 0x91;
pub const tag_region_param_def_space: usize = 0x92;
pub const tag_region_param_def_index: usize = 0x93;

pub const tag_type_param_def: usize = 0x94;

pub const tag_item_generics: usize = 0x95;
pub const tag_method_ty_generics: usize = 0x96;

pub const tag_predicate: usize = 0x97;
pub const tag_predicate_space: usize = 0x98;
pub const tag_predicate_data: usize = 0x99;

pub const tag_unsafety: usize = 0x9a;

pub const tag_associated_type_names: usize = 0x9b;
pub const tag_associated_type_name: usize = 0x9c;

pub const tag_polarity: usize = 0x9d;

pub const tag_macro_defs: usize = 0x10e; // top-level only
pub const tag_macro_def: usize = 0x9e;
pub const tag_macro_def_body: usize = 0x9f;

pub const tag_paren_sugar: usize = 0xa0;

pub const tag_codemap: usize = 0xa1;
pub const tag_codemap_filemap: usize = 0xa2;

pub const tag_item_super_predicates: usize = 0xa3;

pub const tag_defaulted_trait: usize = 0xa4;
