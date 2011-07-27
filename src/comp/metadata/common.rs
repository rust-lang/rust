// EBML tag definitions and utils shared by the encoder and decoder

import std::str;

const tag_paths: uint = 0x01u;

const tag_items: uint = 0x02u;

const tag_paths_data: uint = 0x03u;

const tag_paths_data_name: uint = 0x04u;

const tag_paths_data_item: uint = 0x05u;

const tag_paths_data_mod: uint = 0x06u;

const tag_def_id: uint = 0x07u;

const tag_items_data: uint = 0x08u;

const tag_items_data_item: uint = 0x09u;

const tag_items_data_item_kind: uint = 0x0au;

const tag_items_data_item_ty_param_count: uint = 0x0bu;

const tag_items_data_item_type: uint = 0x0cu;

const tag_items_data_item_symbol: uint = 0x0du;

const tag_items_data_item_variant: uint = 0x0eu;

const tag_items_data_item_tag_id: uint = 0x0fu;

const tag_index: uint = 0x11u;

const tag_index_buckets: uint = 0x12u;

const tag_index_buckets_bucket: uint = 0x13u;

const tag_index_buckets_bucket_elt: uint = 0x14u;

const tag_index_table: uint = 0x15u;

const tag_meta_item_name_value: uint = 0x18u;

const tag_meta_item_name: uint = 0x19u;

const tag_meta_item_value: uint = 0x20u;

const tag_attributes: uint = 0x21u;

const tag_attribute: uint = 0x22u;

const tag_meta_item_word: uint = 0x23u;

const tag_meta_item_list: uint = 0x24u;

// The list of crates that this crate depends on
const tag_crate_deps: uint = 0x25u;

// A single crate dependency
const tag_crate_dep: uint = 0x26u;

// djb's cdb hashes.
fn hash_node_id(node_id: &int) -> uint { ret 177573u ^ (node_id as uint); }

fn hash_path(s: &str) -> uint {
    let h = 5381u;
    for ch: u8  in str::bytes(s) { h = (h << 5u) + h ^ (ch as uint); }
    ret h;
}

