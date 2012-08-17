// EBML enum definitions and utils shared by the encoder and decoder

const tag_paths: uint = 0x01u;

const tag_items: uint = 0x02u;

const tag_paths_data: uint = 0x03u;

const tag_paths_data_name: uint = 0x04u;

const tag_paths_data_item: uint = 0x05u;

const tag_paths_data_mod: uint = 0x06u;

const tag_def_id: uint = 0x07u;

const tag_items_data: uint = 0x08u;

const tag_items_data_item: uint = 0x09u;

const tag_items_data_item_family: uint = 0x0au;

const tag_items_data_item_ty_param_bounds: uint = 0x0bu;

const tag_items_data_item_type: uint = 0x0cu;

const tag_items_data_item_symbol: uint = 0x0du;

const tag_items_data_item_variant: uint = 0x0eu;

const tag_items_data_parent_item: uint = 0x0fu;

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

const tag_crate_hash: uint = 0x28u;

const tag_parent_item: uint = 0x29u;

const tag_crate_dep_name: uint = 0x2au;
const tag_crate_dep_hash: uint = 0x2bu;
const tag_crate_dep_vers: uint = 0x2cu;

const tag_mod_impl: uint = 0x30u;

const tag_item_trait_method: uint = 0x31u;
const tag_impl_trait: uint = 0x32u;

// discriminator value for variants
const tag_disr_val: uint = 0x34u;

// used to encode ast_map::path and ast_map::path_elt
const tag_path: uint = 0x40u;
const tag_path_len: uint = 0x41u;
const tag_path_elt_mod: uint = 0x42u;
const tag_path_elt_name: uint = 0x43u;
const tag_item_field: uint = 0x44u;
const tag_class_mut: uint = 0x45u;

const tag_region_param: uint = 0x46u;
const tag_mod_impl_trait: uint = 0x47u;
/*
  trait items contain tag_item_trait_method elements,
  impl items contain tag_item_impl_method elements, and classes
  have both. That's because some code treats classes like traits,
  and other code treats them like impls. Because classes can contain
  both, tag_item_trait_method and tag_item_impl_method have to be two
  different tags.
 */
const tag_item_impl_method: uint = 0x48u;
const tag_item_dtor: uint = 0x49u;
const tag_paths_foreign_path: uint = 0x4a;
const tag_item_trait_method_self_ty: uint = 0x4b;
const tag_item_trait_method_self_ty_region: uint = 0x4c;

// Reexports are found within module tags. Each reexport contains def_ids
// and names.
const tag_items_data_item_reexport: uint = 0x4d;
const tag_items_data_item_reexport_def_id: uint = 0x4e;
const tag_items_data_item_reexport_name: uint = 0x4f;

// used to encode crate_ctxt side tables
enum astencode_tag { // Reserves 0x50 -- 0x6f
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
    tag_table_param_bounds = 0x5b,
    tag_table_inferred_modes = 0x5c,
    tag_table_mutbl = 0x5d,
    tag_table_last_use = 0x5e,
    tag_table_spill = 0x5f,
    tag_table_method_map = 0x60,
    tag_table_vtable_map = 0x61,
    tag_table_borrowings = 0x62
}

// djb's cdb hashes.
fn hash_node_id(&&node_id: int) -> uint {
    return 177573u ^ (node_id as uint);
}

fn hash_path(&&s: ~str) -> uint {
    let mut h = 5381u;
    for str::each(s) |ch| { h = (h << 5u) + h ^ (ch as uint); }
    return h;
}

type link_meta = {name: @~str, vers: @~str, extras_hash: ~str};

