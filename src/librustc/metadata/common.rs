// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cast;
use syntax::crateid::CrateId;

// EBML enum definitions and utils shared by the encoder and decoder

pub static tag_items: uint = 0x02u;

pub static tag_paths_data_name: uint = 0x04u;

pub static tag_def_id: uint = 0x07u;

pub static tag_items_data: uint = 0x08u;

pub static tag_items_data_item: uint = 0x09u;

pub static tag_items_data_item_family: uint = 0x0au;

pub static tag_items_data_item_ty_param_bounds: uint = 0x0bu;

pub static tag_items_data_item_type: uint = 0x0cu;

pub static tag_items_data_item_symbol: uint = 0x0du;

pub static tag_items_data_item_variant: uint = 0x0eu;

pub static tag_items_data_parent_item: uint = 0x0fu;

pub static tag_index: uint = 0x11u;

pub static tag_index_buckets: uint = 0x12u;

pub static tag_index_buckets_bucket: uint = 0x13u;

pub static tag_index_buckets_bucket_elt: uint = 0x14u;

pub static tag_index_table: uint = 0x15u;

pub static tag_meta_item_name_value: uint = 0x18u;

pub static tag_meta_item_name: uint = 0x19u;

pub static tag_meta_item_value: uint = 0x20u;

pub static tag_attributes: uint = 0x21u;

pub static tag_attribute: uint = 0x22u;

pub static tag_meta_item_word: uint = 0x23u;

pub static tag_meta_item_list: uint = 0x24u;

// The list of crates that this crate depends on
pub static tag_crate_deps: uint = 0x25u;

// A single crate dependency
pub static tag_crate_dep: uint = 0x26u;

pub static tag_crate_hash: uint = 0x28u;

pub static tag_parent_item: uint = 0x29u;

pub static tag_crate_dep_name: uint = 0x2au;
pub static tag_crate_dep_hash: uint = 0x2bu;
pub static tag_crate_dep_vers: uint = 0x2cu;

pub static tag_mod_impl: uint = 0x30u;

pub static tag_item_trait_method: uint = 0x31u;

pub static tag_item_trait_ref: uint = 0x32u;
pub static tag_item_super_trait_ref: uint = 0x33u;

// discriminator value for variants
pub static tag_disr_val: uint = 0x34u;

// used to encode ast_map::path and ast_map::path_elt
pub static tag_path: uint = 0x40u;
pub static tag_path_len: uint = 0x41u;
pub static tag_path_elt_mod: uint = 0x42u;
pub static tag_path_elt_name: uint = 0x43u;
pub static tag_item_field: uint = 0x44u;
pub static tag_struct_mut: uint = 0x45u;

pub static tag_item_variances: uint = 0x46;
pub static tag_mod_impl_trait: uint = 0x47u;
/*
  trait items contain tag_item_trait_method elements,
  impl items contain tag_item_impl_method elements, and classes
  have both. That's because some code treats classes like traits,
  and other code treats them like impls. Because classes can contain
  both, tag_item_trait_method and tag_item_impl_method have to be two
  different tags.
 */
pub static tag_item_impl_method: uint = 0x48u;
pub static tag_item_trait_method_explicit_self: uint = 0x4b;
pub static tag_item_trait_method_self_ty_region: uint = 0x4c;


// Reexports are found within module tags. Each reexport contains def_ids
// and names.
pub static tag_items_data_item_reexport: uint = 0x4d;
pub static tag_items_data_item_reexport_def_id: uint = 0x4e;
pub static tag_items_data_item_reexport_name: uint = 0x4f;

// used to encode crate_ctxt side tables
#[deriving(Eq)]
#[repr(uint)]
pub enum astencode_tag { // Reserves 0x50 -- 0x6f
    tag_ast = 0x50,

    tag_tree = 0x51,

    tag_id_range = 0x52,

    tag_table = 0x53,
    tag_table_id = 0x54,
    tag_table_val = 0x55,
    tag_table_def = 0x56,
    tag_table_node_type = 0x57,
    tag_table_node_type_subst = 0x58,
    tag_table_freevars = 0x59,
    tag_table_tcache = 0x5a,
    tag_table_param_defs = 0x5b,
    tag_table_mutbl = 0x5d,
    tag_table_last_use = 0x5e,
    tag_table_spill = 0x5f,
    tag_table_method_map = 0x60,
    tag_table_vtable_map = 0x61,
    tag_table_adjustments = 0x62,
    tag_table_moves_map = 0x63,
    tag_table_capture_map = 0x64
}
static first_astencode_tag : uint = tag_ast as uint;
static last_astencode_tag : uint = tag_table_capture_map as uint;
impl astencode_tag {
    pub fn from_uint(value : uint) -> Option<astencode_tag> {
        let is_a_tag = first_astencode_tag <= value && value <= last_astencode_tag;
        if !is_a_tag { None } else {
            Some(unsafe { cast::transmute(value) })
        }
    }
}

pub static tag_item_trait_method_sort: uint = 0x70;

pub static tag_item_impl_type_basename: uint = 0x71;

// Language items are a top-level directory (for speed). Hierarchy:
//
// tag_lang_items
// - tag_lang_items_item
//   - tag_lang_items_item_id: u32
//   - tag_lang_items_item_node_id: u32

pub static tag_lang_items: uint = 0x72;
pub static tag_lang_items_item: uint = 0x73;
pub static tag_lang_items_item_id: uint = 0x74;
pub static tag_lang_items_item_node_id: uint = 0x75;

pub static tag_item_unnamed_field: uint = 0x76;
pub static tag_items_data_item_struct_ctor: uint = 0x77;
pub static tag_items_data_item_visibility: uint = 0x78;

pub static tag_link_args: uint = 0x79;
pub static tag_link_args_arg: uint = 0x7a;

pub static tag_item_method_tps: uint = 0x7b;
pub static tag_item_method_fty: uint = 0x7c;
pub static tag_item_method_transformed_self_ty: uint = 0x7d;

pub static tag_mod_child: uint = 0x7e;
pub static tag_misc_info: uint = 0x7f;
pub static tag_misc_info_crate_items: uint = 0x80;

pub static tag_item_method_provided_source: uint = 0x81;
pub static tag_item_impl_vtables: uint = 0x82;

pub static tag_impls: uint = 0x83;
pub static tag_impls_impl: uint = 0x84;

pub static tag_items_data_item_inherent_impl: uint = 0x85;
pub static tag_items_data_item_extension_impl: uint = 0x86;

pub static tag_path_elt_pretty_name: uint = 0x87;
pub static tag_path_elt_pretty_name_ident: uint = 0x88;
pub static tag_path_elt_pretty_name_extra: uint = 0x89;

pub static tag_region_param_def: uint = 0x100;
pub static tag_region_param_def_ident: uint = 0x101;
pub static tag_region_param_def_def_id: uint = 0x102;

pub static tag_native_libraries: uint = 0x103;
pub static tag_native_libraries_lib: uint = 0x104;
pub static tag_native_libraries_name: uint = 0x105;
pub static tag_native_libraries_kind: uint = 0x106;

#[deriving(Clone)]
pub struct LinkMeta {
    crateid: CrateId,
    crate_hash: @str,
}
