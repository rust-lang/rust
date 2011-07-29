// Metadata encoding

import std::ivec;
import std::str;
import std::uint;
import std::ioivec;
import std::option;
import std::option::some;
import std::option::none;
import std::ebmlivec;
import std::map;
import syntax::ast::*;
import common::*;
import middle::trans_common::crate_ctxt;
import middle::ty;
import middle::ty::node_id_to_monotype;
import front::attr;

export encode_metadata;
export encoded_ty;

type abbrev_map = map::hashmap[ty::t, tyencode::ty_abbrev];

type encode_ctxt = {ccx: @crate_ctxt, type_abbrevs: abbrev_map};

// Path table encoding
fn encode_name(ebml_w: &ebmlivec::writer, name: &str) {
    ebmlivec::start_tag(ebml_w, tag_paths_data_name);
    ebml_w.writer.write(str::bytes_ivec(name));
    ebmlivec::end_tag(ebml_w);
}

fn encode_def_id(ebml_w: &ebmlivec::writer, id: &def_id) {
    ebmlivec::start_tag(ebml_w, tag_def_id);
    ebml_w.writer.write(str::bytes_ivec(def_to_str(id)));
    ebmlivec::end_tag(ebml_w);
}

type entry[T] = {val: T, pos: uint};

fn encode_tag_variant_paths(ebml_w: &ebmlivec::writer, variants: &variant[],
                            path: &str[], index: &mutable (entry[str])[]) {
    for variant: variant  in variants {
        add_to_index(ebml_w, path, index, variant.node.name);
        ebmlivec::start_tag(ebml_w, tag_paths_data_item);
        encode_name(ebml_w, variant.node.name);
        encode_def_id(ebml_w, local_def(variant.node.id));
        ebmlivec::end_tag(ebml_w);
    }
}

fn add_to_index(ebml_w: &ebmlivec::writer, path: &str[],
                index: &mutable (entry[str])[], name: &str) {
    let full_path = path + ~[name];
    index +=
        ~[{val: str::connect_ivec(full_path, "::"),
           pos: ebml_w.writer.tell()}];
}

fn encode_native_module_item_paths(ebml_w: &ebmlivec::writer,
                                   nmod: &native_mod, path: &str[],
                                   index: &mutable (entry[str])[]) {
    for nitem: @native_item  in nmod.items {
        add_to_index(ebml_w, path, index, nitem.ident);
        ebmlivec::start_tag(ebml_w, tag_paths_data_item);
        encode_name(ebml_w, nitem.ident);
        encode_def_id(ebml_w, local_def(nitem.id));
        ebmlivec::end_tag(ebml_w);
    }
}

fn encode_module_item_paths(ebml_w: &ebmlivec::writer, module: &_mod,
                            path: &str[], index: &mutable (entry[str])[]) {
    for it: @item  in module.items {
        if !is_exported(it.ident, module) { cont; }
        alt it.node {
          item_const(_, _) {
            add_to_index(ebml_w, path, index, it.ident);
            ebmlivec::start_tag(ebml_w, tag_paths_data_item);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            ebmlivec::end_tag(ebml_w);
          }
          item_fn(_, tps) {
            add_to_index(ebml_w, path, index, it.ident);
            ebmlivec::start_tag(ebml_w, tag_paths_data_item);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            ebmlivec::end_tag(ebml_w);
          }
          item_mod(_mod) {
            add_to_index(ebml_w, path, index, it.ident);
            ebmlivec::start_tag(ebml_w, tag_paths_data_mod);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            encode_module_item_paths(ebml_w, _mod, path + ~[it.ident], index);
            ebmlivec::end_tag(ebml_w);
          }
          item_native_mod(nmod) {
            add_to_index(ebml_w, path, index, it.ident);
            ebmlivec::start_tag(ebml_w, tag_paths_data_mod);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            encode_native_module_item_paths(ebml_w, nmod, path + ~[it.ident],
                                            index);
            ebmlivec::end_tag(ebml_w);
          }
          item_ty(_, tps) {
            add_to_index(ebml_w, path, index, it.ident);
            ebmlivec::start_tag(ebml_w, tag_paths_data_item);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            ebmlivec::end_tag(ebml_w);
          }
          item_res(_, _, tps, ctor_id) {
            add_to_index(ebml_w, path, index, it.ident);
            ebmlivec::start_tag(ebml_w, tag_paths_data_item);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(ctor_id));
            ebmlivec::end_tag(ebml_w);
            add_to_index(ebml_w, path, index, it.ident);
            ebmlivec::start_tag(ebml_w, tag_paths_data_item);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            ebmlivec::end_tag(ebml_w);
          }
          item_tag(variants, tps) {
            add_to_index(ebml_w, path, index, it.ident);
            ebmlivec::start_tag(ebml_w, tag_paths_data_item);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            ebmlivec::end_tag(ebml_w);
            encode_tag_variant_paths(ebml_w, variants, path, index);
          }
          item_obj(_, tps, ctor_id) {
            add_to_index(ebml_w, path, index, it.ident);
            ebmlivec::start_tag(ebml_w, tag_paths_data_item);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(ctor_id));
            ebmlivec::end_tag(ebml_w);
            add_to_index(ebml_w, path, index, it.ident);
            ebmlivec::start_tag(ebml_w, tag_paths_data_item);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            ebmlivec::end_tag(ebml_w);
          }
        }
    }
}

fn encode_item_paths(ebml_w: &ebmlivec::writer, crate: &@crate) ->
   (entry[str])[] {
    let index: (entry[str])[] = ~[];
    let path: str[] = ~[];
    ebmlivec::start_tag(ebml_w, tag_paths);
    encode_module_item_paths(ebml_w, crate.node.module, path, index);
    ebmlivec::end_tag(ebml_w);
    ret index;
}


// Item info table encoding
fn encode_family(ebml_w: &ebmlivec::writer, c: u8) {
    ebmlivec::start_tag(ebml_w, tag_items_data_item_family);
    ebml_w.writer.write(~[c]);
    ebmlivec::end_tag(ebml_w);
}

fn def_to_str(did: &def_id) -> str { ret #fmt("%d:%d", did.crate, did.node); }

fn encode_type_param_kinds(ebml_w: &ebmlivec::writer, tps: &ty_param[]) {
    ebmlivec::start_tag(ebml_w, tag_items_data_item_ty_param_kinds);
    ebmlivec::write_vint(ebml_w.writer, ivec::len[ty_param](tps));
    for tp: ty_param in tps {
        let c = alt tp.kind {
          kind_unique. { 'u' }
          kind_shared. { 's' }
          kind_pinned. { 'p' }
        };
        ebml_w.writer.write(~[c as u8]);
    }
    ebmlivec::end_tag(ebml_w);
}

fn encode_variant_id(ebml_w: &ebmlivec::writer, vid: &def_id) {
    ebmlivec::start_tag(ebml_w, tag_items_data_item_variant);
    ebml_w.writer.write(str::bytes_ivec(def_to_str(vid)));
    ebmlivec::end_tag(ebml_w);
}

fn encode_type(ecx: &@encode_ctxt, ebml_w: &ebmlivec::writer, typ: &ty::t) {
    ebmlivec::start_tag(ebml_w, tag_items_data_item_type);
    let f = def_to_str;
    let ty_str_ctxt =
        @{ds: f,
          tcx: ecx.ccx.tcx,
          abbrevs: tyencode::ac_use_abbrevs(ecx.type_abbrevs)};
    tyencode::enc_ty(ioivec::new_writer_(ebml_w.writer), ty_str_ctxt, typ);
    ebmlivec::end_tag(ebml_w);
}

fn encode_symbol(ecx: &@encode_ctxt, ebml_w: &ebmlivec::writer, id: node_id) {
    ebmlivec::start_tag(ebml_w, tag_items_data_item_symbol);
    ebml_w.writer.write(str::bytes_ivec(ecx.ccx.item_symbols.get(id)));
    ebmlivec::end_tag(ebml_w);
}

fn encode_discriminant(ecx: &@encode_ctxt, ebml_w: &ebmlivec::writer,
                       id: node_id) {
    ebmlivec::start_tag(ebml_w, tag_items_data_item_symbol);
    ebml_w.writer.write(str::bytes_ivec(ecx.ccx.discrim_symbols.get(id)));
    ebmlivec::end_tag(ebml_w);
}

fn encode_tag_id(ebml_w: &ebmlivec::writer, id: &def_id) {
    ebmlivec::start_tag(ebml_w, tag_items_data_item_tag_id);
    ebml_w.writer.write(str::bytes_ivec(def_to_str(id)));
    ebmlivec::end_tag(ebml_w);
}

fn encode_tag_variant_info(ecx: &@encode_ctxt, ebml_w: &ebmlivec::writer,
                           id: node_id, variants: &variant[],
                           index: &mutable (entry[int])[],
                           ty_params: &ty_param[]) {
    for variant: variant  in variants {
        index += ~[{val: variant.node.id, pos: ebml_w.writer.tell()}];
        ebmlivec::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(variant.node.id));
        encode_family(ebml_w, 'v' as u8);
        encode_tag_id(ebml_w, local_def(id));
        encode_type(ecx, ebml_w,
                    node_id_to_monotype(ecx.ccx.tcx, variant.node.id));
        if ivec::len[variant_arg](variant.node.args) > 0u {
            encode_symbol(ecx, ebml_w, variant.node.id);
        }
        encode_discriminant(ecx, ebml_w, variant.node.id);
        encode_type_param_kinds(ebml_w, ty_params);
        ebmlivec::end_tag(ebml_w);
    }
}

fn encode_info_for_item(ecx: @encode_ctxt, ebml_w: &ebmlivec::writer,
                        item: @item, index: &mutable (entry[int])[]) {
    alt item.node {
      item_const(_, _) {
        ebmlivec::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'c' as u8);
        encode_type(ecx, ebml_w, node_id_to_monotype(ecx.ccx.tcx, item.id));
        encode_symbol(ecx, ebml_w, item.id);
        ebmlivec::end_tag(ebml_w);
      }
      item_fn(fd, tps) {
        ebmlivec::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w,
                    alt fd.decl.purity { pure_fn. { 'p' } impure_fn. { 'f' } }
                        as u8);
        encode_type_param_kinds(ebml_w, tps);
        encode_type(ecx, ebml_w, node_id_to_monotype(ecx.ccx.tcx, item.id));
        encode_symbol(ecx, ebml_w, item.id);
        ebmlivec::end_tag(ebml_w);
      }
      item_mod(_) {
        ebmlivec::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'm' as u8);
        ebmlivec::end_tag(ebml_w);
      }
      item_native_mod(_) {
        ebmlivec::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'n' as u8);
        ebmlivec::end_tag(ebml_w);
      }
      item_ty(_, tps) {
        ebmlivec::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'y' as u8);
        encode_type_param_kinds(ebml_w, tps);
        encode_type(ecx, ebml_w, node_id_to_monotype(ecx.ccx.tcx, item.id));
        ebmlivec::end_tag(ebml_w);
      }
      item_tag(variants, tps) {
        ebmlivec::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 't' as u8);
        encode_type_param_kinds(ebml_w, tps);
        encode_type(ecx, ebml_w, node_id_to_monotype(ecx.ccx.tcx, item.id));
        for v: variant  in variants {
            encode_variant_id(ebml_w, local_def(v.node.id));
        }
        ebmlivec::end_tag(ebml_w);
        encode_tag_variant_info(ecx, ebml_w, item.id, variants, index, tps);
      }
      item_res(_, _, tps, ctor_id) {
        let fn_ty = node_id_to_monotype(ecx.ccx.tcx, ctor_id);

        ebmlivec::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(ctor_id));
        encode_family(ebml_w, 'y' as u8);
        encode_type_param_kinds(ebml_w, tps);
        encode_type(ecx, ebml_w, ty::ty_fn_ret(ecx.ccx.tcx, fn_ty));
        encode_symbol(ecx, ebml_w, item.id);
        ebmlivec::end_tag(ebml_w);

        index += ~[{val: ctor_id, pos: ebml_w.writer.tell()}];
        ebmlivec::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(ctor_id));
        encode_family(ebml_w, 'f' as u8);
        encode_type_param_kinds(ebml_w, tps);
        encode_type(ecx, ebml_w, fn_ty);
        encode_symbol(ecx, ebml_w, ctor_id);
        ebmlivec::end_tag(ebml_w);
      }
      item_obj(_, tps, ctor_id) {
        let fn_ty = node_id_to_monotype(ecx.ccx.tcx, ctor_id);

        ebmlivec::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'y' as u8);
        encode_type_param_kinds(ebml_w, tps);
        encode_type(ecx, ebml_w, ty::ty_fn_ret(ecx.ccx.tcx, fn_ty));
        ebmlivec::end_tag(ebml_w);

        index += ~[{val: ctor_id, pos: ebml_w.writer.tell()}];
        ebmlivec::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(ctor_id));
        encode_family(ebml_w, 'f' as u8);
        encode_type_param_kinds(ebml_w, tps);
        encode_type(ecx, ebml_w, fn_ty);
        encode_symbol(ecx, ebml_w, ctor_id);
        ebmlivec::end_tag(ebml_w);
      }
    }
}

fn encode_info_for_native_item(ecx: &@encode_ctxt, ebml_w: &ebmlivec::writer,
                               nitem: &@native_item) {
    ebmlivec::start_tag(ebml_w, tag_items_data_item);
    alt nitem.node {
      native_item_ty. {
        encode_def_id(ebml_w, local_def(nitem.id));
        encode_family(ebml_w, 'T' as u8);
        encode_type(ecx, ebml_w,
                    ty::mk_native(ecx.ccx.tcx, local_def(nitem.id)));
      }
      native_item_fn(_, _, tps) {
        encode_def_id(ebml_w, local_def(nitem.id));
        encode_family(ebml_w, 'F' as u8);
        encode_type_param_kinds(ebml_w, tps);
        encode_type(ecx, ebml_w, node_id_to_monotype(ecx.ccx.tcx, nitem.id));
        encode_symbol(ecx, ebml_w, nitem.id);
      }
    }
    ebmlivec::end_tag(ebml_w);
}

fn encode_info_for_items(ecx: &@encode_ctxt, ebml_w: &ebmlivec::writer) ->
   (entry[int])[] {
    let index: (entry[int])[] = ~[];
    ebmlivec::start_tag(ebml_w, tag_items_data);
    for each kvp: @{key: node_id, val: middle::ast_map::ast_node}  in
             ecx.ccx.ast_map.items() {
        alt kvp.val {
          middle::ast_map::node_item(i) {
            index += ~[{val: kvp.key, pos: ebml_w.writer.tell()}];
            encode_info_for_item(ecx, ebml_w, i, index);
          }
          middle::ast_map::node_native_item(i) {
            index += ~[{val: kvp.key, pos: ebml_w.writer.tell()}];
            encode_info_for_native_item(ecx, ebml_w, i);
          }
          _ { }
        }
    }
    ebmlivec::end_tag(ebml_w);
    ret index;
}


// Path and definition ID indexing

fn create_index[T](index: &(entry[T])[], hash_fn: fn(&T) -> uint ) ->
   (@(entry[T])[])[] {
    let buckets: (@mutable (entry[T])[])[] = ~[];
    for each i: uint  in uint::range(0u, 256u) { buckets += ~[@mutable ~[]]; }
    for elt: entry[T]  in index {
        let h = hash_fn(elt.val);
        *buckets.(h % 256u) += ~[elt];
    }

    let buckets_frozen = ~[];
    for bucket: @mutable (entry[T])[]  in buckets {
        buckets_frozen += ~[@*bucket];
    }
    ret buckets_frozen;
}

fn encode_index[T](ebml_w: &ebmlivec::writer, buckets: &(@(entry[T])[])[],
                   write_fn: fn(&ioivec::writer, &T) ) {
    let writer = ioivec::new_writer_(ebml_w.writer);
    ebmlivec::start_tag(ebml_w, tag_index);
    let bucket_locs: uint[] = ~[];
    ebmlivec::start_tag(ebml_w, tag_index_buckets);
    for bucket: @(entry[T])[]  in buckets {
        bucket_locs += ~[ebml_w.writer.tell()];
        ebmlivec::start_tag(ebml_w, tag_index_buckets_bucket);
        for elt: entry[T]  in *bucket {
            ebmlivec::start_tag(ebml_w, tag_index_buckets_bucket_elt);
            writer.write_be_uint(elt.pos, 4u);
            write_fn(writer, elt.val);
            ebmlivec::end_tag(ebml_w);
        }
        ebmlivec::end_tag(ebml_w);
    }
    ebmlivec::end_tag(ebml_w);
    ebmlivec::start_tag(ebml_w, tag_index_table);
    for pos: uint  in bucket_locs { writer.write_be_uint(pos, 4u); }
    ebmlivec::end_tag(ebml_w);
    ebmlivec::end_tag(ebml_w);
}

fn write_str(writer: &ioivec::writer, s: &str) { writer.write_str(s); }

fn write_int(writer: &ioivec::writer, n: &int) {
    writer.write_be_uint(n as uint, 4u);
}

fn encode_meta_item(ebml_w: &ebmlivec::writer, mi: &meta_item) {
    alt mi.node {
      meta_word(name) {
        ebmlivec::start_tag(ebml_w, tag_meta_item_word);
        ebmlivec::start_tag(ebml_w, tag_meta_item_name);
        ebml_w.writer.write(str::bytes_ivec(name));
        ebmlivec::end_tag(ebml_w);
        ebmlivec::end_tag(ebml_w);
      }
      meta_name_value(name, value) {
        alt value.node {
          lit_str(value, _) {
            ebmlivec::start_tag(ebml_w, tag_meta_item_name_value);
            ebmlivec::start_tag(ebml_w, tag_meta_item_name);
            ebml_w.writer.write(str::bytes_ivec(name));
            ebmlivec::end_tag(ebml_w);
            ebmlivec::start_tag(ebml_w, tag_meta_item_value);
            ebml_w.writer.write(str::bytes_ivec(value));
            ebmlivec::end_tag(ebml_w);
            ebmlivec::end_tag(ebml_w);
          }
          _ {/* FIXME (#611) */ }
        }
      }
      meta_list(name, items) {
        ebmlivec::start_tag(ebml_w, tag_meta_item_list);
        ebmlivec::start_tag(ebml_w, tag_meta_item_name);
        ebml_w.writer.write(str::bytes_ivec(name));
        ebmlivec::end_tag(ebml_w);
        for inner_item: @meta_item  in items {
            encode_meta_item(ebml_w, *inner_item);
        }
        ebmlivec::end_tag(ebml_w);
      }
    }
}

fn encode_attributes(ebml_w: &ebmlivec::writer, attrs: &attribute[]) {
    ebmlivec::start_tag(ebml_w, tag_attributes);
    for attr: attribute  in attrs {
        ebmlivec::start_tag(ebml_w, tag_attribute);
        encode_meta_item(ebml_w, attr.node.value);
        ebmlivec::end_tag(ebml_w);
    }
    ebmlivec::end_tag(ebml_w);
}

// So there's a special crate attribute called 'link' which defines the
// metadata that Rust cares about for linking crates. This attribute requires
// 'name' and 'vers' items, so if the user didn't provide them we will throw
// them in anyway with default values.
fn synthesize_crate_attrs(ecx: &@encode_ctxt, crate: &@crate) -> attribute[] {

    fn synthesize_link_attr(ecx: &@encode_ctxt, items: &(@meta_item)[]) ->
       attribute {

        assert (ecx.ccx.link_meta.name != "");
        assert (ecx.ccx.link_meta.vers != "");

        let name_item =
            attr::mk_name_value_item_str("name", ecx.ccx.link_meta.name);
        let vers_item =
            attr::mk_name_value_item_str("vers", ecx.ccx.link_meta.vers);

        let other_items =
            {
                let tmp = attr::remove_meta_items_by_name(items, "name");
                attr::remove_meta_items_by_name(tmp, "vers")
            };

        let meta_items = ~[name_item, vers_item] + other_items;
        let link_item = attr::mk_list_item("link", meta_items);

        ret attr::mk_attr(link_item);
    }

    let attrs: attribute[] = ~[];
    let found_link_attr = false;
    for attr: attribute  in crate.node.attrs {
        attrs +=
            if attr::get_attr_name(attr) != "link" {
                ~[attr]
            } else {
                alt attr.node.value.node {
                  meta_list(n, l) {
                    found_link_attr = true;
                    ~[synthesize_link_attr(ecx, l)]
                  }
                  _ { ~[attr] }
                }
            }
    }

    if !found_link_attr { attrs += ~[synthesize_link_attr(ecx, ~[])]; }

    ret attrs;
}

fn encode_crate_deps(ebml_w: &ebmlivec::writer, cstore: &cstore::cstore) {

    fn get_ordered_names(cstore: &cstore::cstore) -> str[] {
        type hashkv = @{key: crate_num, val: cstore::crate_metadata};
        type numname = {crate: crate_num, ident: str};

        // Pull the cnums and names out of cstore
        let pairs: numname[mutable ] = ~[mutable ];
        for each hashkv: hashkv  in cstore::iter_crate_data(cstore) {
            pairs += ~[mutable {crate: hashkv.key, ident: hashkv.val.name}];
        }

        // Sort by cnum
        fn lteq(kv1: &numname, kv2: &numname) -> bool {
            kv1.crate <= kv2.crate
        }
        std::sort::ivector::quick_sort(lteq, pairs);

        // Sanity-check the crate numbers
        let expected_cnum = 1;
        for n: numname  in pairs {
            assert (n.crate == expected_cnum);
            expected_cnum += 1;
        }

        // Return just the names
        fn name(kv: &numname) -> str { kv.ident }
        // mutable -> immutable hack for ivec::map
        let immpairs = ivec::slice(pairs, 0u, ivec::len(pairs));
        ret ivec::map(name, immpairs);
    }

    // We're just going to write a list of crate names, with the assumption
    // that they are numbered 1 to n.
    // FIXME: This is not nearly enough to support correct versioning
    // but is enough to get transitive crate dependencies working.
    ebmlivec::start_tag(ebml_w, tag_crate_deps);
    for cname: str  in get_ordered_names(cstore) {
        ebmlivec::start_tag(ebml_w, tag_crate_dep);
        ebml_w.writer.write(str::bytes_ivec(cname));
        ebmlivec::end_tag(ebml_w);
    }
    ebmlivec::end_tag(ebml_w);
}

fn encode_metadata(cx: &@crate_ctxt, crate: &@crate) -> str {

    let abbrevs = map::mk_hashmap(ty::hash_ty, ty::eq_ty);
    let ecx = @{ccx: cx, type_abbrevs: abbrevs};

    let string_w = ioivec::string_writer();
    let buf_w = string_w.get_writer().get_buf_writer();
    let ebml_w = ebmlivec::create_writer(buf_w);

    let crate_attrs = synthesize_crate_attrs(ecx, crate);
    encode_attributes(ebml_w, crate_attrs);

    encode_crate_deps(ebml_w, cx.sess.get_cstore());

    // Encode and index the paths.

    ebmlivec::start_tag(ebml_w, tag_paths);
    let paths_index = encode_item_paths(ebml_w, crate);
    let paths_buckets = create_index(paths_index, hash_path);
    encode_index(ebml_w, paths_buckets, write_str);
    ebmlivec::end_tag(ebml_w);
    // Encode and index the items.

    ebmlivec::start_tag(ebml_w, tag_items);
    let items_index = encode_info_for_items(ecx, ebml_w);
    let items_buckets = create_index(items_index, hash_node_id);
    encode_index(ebml_w, items_buckets, write_int);
    ebmlivec::end_tag(ebml_w);
    // Pad this, since something (LLVM, presumably) is cutting off the
    // remaining % 4 bytes_ivec.

    buf_w.write(~[0u8, 0u8, 0u8, 0u8]);
    ret string_w.get_str();
}

// Get the encoded string for a type
fn encoded_ty(tcx: &ty::ctxt, t: &ty::t) -> str {
    let cx = @{ds: def_to_str, tcx: tcx, abbrevs: tyencode::ac_no_abbrevs};
    let sw = ioivec::string_writer();
    tyencode::enc_ty(sw.get_writer(), cx, t);
    ret sw.get_str();
}


// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
