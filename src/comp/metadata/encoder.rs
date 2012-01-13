// Metadata encoding

import std::{io, ebml, map};
import io::writer_util;
import syntax::ast::*;
import syntax::ast_util;
import syntax::ast_util::local_def;
import common::*;
import middle::trans_common::crate_ctxt;
import middle::ty;
import middle::ty::node_id_to_monotype;
import front::attr;

export encode_metadata;
export encoded_ty;

type abbrev_map = map::hashmap<ty::t, tyencode::ty_abbrev>;

type encode_ctxt = {ccx: @crate_ctxt, type_abbrevs: abbrev_map};

// Path table encoding
fn encode_name(ebml_w: ebml::writer, name: str) {
    ebml::start_tag(ebml_w, tag_paths_data_name);
    ebml_w.writer.write(str::bytes(name));
    ebml::end_tag(ebml_w);
}

fn encode_def_id(ebml_w: ebml::writer, id: def_id) {
    ebml::start_tag(ebml_w, tag_def_id);
    ebml_w.writer.write(str::bytes(def_to_str(id)));
    ebml::end_tag(ebml_w);
}

type entry<T> = {val: T, pos: uint};

fn encode_tag_variant_paths(ebml_w: ebml::writer, variants: [variant],
                            path: [str], &index: [entry<str>]) {
    for variant: variant in variants {
        add_to_index(ebml_w, path, index, variant.node.name);
        ebml::start_tag(ebml_w, tag_paths_data_item);
        encode_name(ebml_w, variant.node.name);
        encode_def_id(ebml_w, local_def(variant.node.id));
        ebml::end_tag(ebml_w);
    }
}

fn add_to_index(ebml_w: ebml::writer, path: [str], &index: [entry<str>],
                name: str) {
    let full_path = path + [name];
    index +=
        [{val: str::connect(full_path, "::"), pos: ebml_w.writer.tell()}];
}

fn encode_native_module_item_paths(ebml_w: ebml::writer, nmod: native_mod,
                                   path: [str], &index: [entry<str>]) {
    for nitem: @native_item in nmod.items {
        add_to_index(ebml_w, path, index, nitem.ident);
        ebml::start_tag(ebml_w, tag_paths_data_item);
        encode_name(ebml_w, nitem.ident);
        encode_def_id(ebml_w, local_def(nitem.id));
        ebml::end_tag(ebml_w);
    }
}

fn encode_module_item_paths(ebml_w: ebml::writer, module: _mod, path: [str],
                            &index: [entry<str>]) {
    // FIXME factor out add_to_index/start/encode_name/encode_def_id/end ops
    for it: @item in module.items {
        if !ast_util::is_exported(it.ident, module) { cont; }
        alt it.node {
          item_const(_, _) {
            add_to_index(ebml_w, path, index, it.ident);
            ebml::start_tag(ebml_w, tag_paths_data_item);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            ebml::end_tag(ebml_w);
          }
          item_fn(_, tps, _) {
            add_to_index(ebml_w, path, index, it.ident);
            ebml::start_tag(ebml_w, tag_paths_data_item);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            ebml::end_tag(ebml_w);
          }
          item_mod(_mod) {
            add_to_index(ebml_w, path, index, it.ident);
            ebml::start_tag(ebml_w, tag_paths_data_mod);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            encode_module_item_paths(ebml_w, _mod, path + [it.ident], index);
            ebml::end_tag(ebml_w);
          }
          item_native_mod(nmod) {
            add_to_index(ebml_w, path, index, it.ident);
            ebml::start_tag(ebml_w, tag_paths_data_mod);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            encode_native_module_item_paths(ebml_w, nmod, path + [it.ident],
                                            index);
            ebml::end_tag(ebml_w);
          }
          item_ty(_, tps) {
            add_to_index(ebml_w, path, index, it.ident);
            ebml::start_tag(ebml_w, tag_paths_data_item);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            ebml::end_tag(ebml_w);
          }
          item_res(_, tps, _, _, ctor_id) {
            add_to_index(ebml_w, path, index, it.ident);
            ebml::start_tag(ebml_w, tag_paths_data_item);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(ctor_id));
            ebml::end_tag(ebml_w);
            add_to_index(ebml_w, path, index, it.ident);
            ebml::start_tag(ebml_w, tag_paths_data_item);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            ebml::end_tag(ebml_w);
          }
          item_tag(variants, tps) {
            add_to_index(ebml_w, path, index, it.ident);
            ebml::start_tag(ebml_w, tag_paths_data_item);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            ebml::end_tag(ebml_w);
            encode_tag_variant_paths(ebml_w, variants, path, index);
          }
          item_iface(_, _) {
            add_to_index(ebml_w, path, index, it.ident);
            ebml::start_tag(ebml_w, tag_paths_data_item);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            ebml::end_tag(ebml_w);
          }
          item_impl(_, _, _, _) {}
        }
    }
}

fn encode_item_paths(ebml_w: ebml::writer, ecx: @encode_ctxt, crate: @crate)
    -> [entry<str>] {
    let index: [entry<str>] = [];
    let path: [str] = [];
    ebml::start_tag(ebml_w, tag_paths);
    encode_module_item_paths(ebml_w, crate.node.module, path, index);
    encode_reexport_paths(ebml_w, ecx, index);
    ebml::end_tag(ebml_w);
    ret index;
}

fn encode_reexport_paths(ebml_w: ebml::writer,
                         ecx: @encode_ctxt, &index: [entry<str>]) {
    ecx.ccx.exp_map.items {|key, def|
        let path = key.path;
        index += [{val: path, pos: ebml_w.writer.tell()}];
        ebml::start_tag(ebml_w, tag_paths_data_item);
        encode_name(ebml_w, path);
        encode_def_id(ebml_w, ast_util::def_id_of_def(def));
        ebml::end_tag(ebml_w);
    }
}


// Item info table encoding
fn encode_family(ebml_w: ebml::writer, c: u8) {
    ebml::start_tag(ebml_w, tag_items_data_item_family);
    ebml_w.writer.write([c]);
    ebml::end_tag(ebml_w);
}

fn def_to_str(did: def_id) -> str { ret #fmt["%d:%d", did.crate, did.node]; }

fn encode_type_param_bounds(ebml_w: ebml::writer, ecx: @encode_ctxt,
                            params: [ty_param]) {
    let ty_str_ctxt = @{ds: def_to_str,
                        tcx: ecx.ccx.tcx,
                        abbrevs: tyencode::ac_use_abbrevs(ecx.type_abbrevs)};
    for param in params {
        ebml::start_tag(ebml_w, tag_items_data_item_ty_param_bounds);
        let bs = ecx.ccx.tcx.ty_param_bounds.get(param.id);
        tyencode::enc_bounds(ebml_w.writer, ty_str_ctxt, bs);
        ebml::end_tag(ebml_w);
    }
}

fn encode_variant_id(ebml_w: ebml::writer, vid: def_id) {
    ebml::start_tag(ebml_w, tag_items_data_item_variant);
    ebml_w.writer.write(str::bytes(def_to_str(vid)));
    ebml::end_tag(ebml_w);
}

fn write_type(ecx: @encode_ctxt, ebml_w: ebml::writer, typ: ty::t) {
    let ty_str_ctxt =
        @{ds: def_to_str,
          tcx: ecx.ccx.tcx,
          abbrevs: tyencode::ac_use_abbrevs(ecx.type_abbrevs)};
    tyencode::enc_ty(ebml_w.writer, ty_str_ctxt, typ);
}

fn encode_type(ecx: @encode_ctxt, ebml_w: ebml::writer, typ: ty::t) {
    ebml::start_tag(ebml_w, tag_items_data_item_type);
    write_type(ecx, ebml_w, typ);
    ebml::end_tag(ebml_w);
}

fn encode_symbol(ecx: @encode_ctxt, ebml_w: ebml::writer, id: node_id) {
    ebml::start_tag(ebml_w, tag_items_data_item_symbol);
    ebml_w.writer.write(str::bytes(ecx.ccx.item_symbols.get(id)));
    ebml::end_tag(ebml_w);
}

fn encode_discriminant(ecx: @encode_ctxt, ebml_w: ebml::writer, id: node_id) {
    ebml::start_tag(ebml_w, tag_items_data_item_symbol);
    ebml_w.writer.write(str::bytes(ecx.ccx.discrim_symbols.get(id)));
    ebml::end_tag(ebml_w);
}

fn encode_disr_val(_ecx: @encode_ctxt, ebml_w: ebml::writer, disr_val: int) {
    ebml::start_tag(ebml_w, tag_disr_val);
    ebml_w.writer.write(str::bytes(int::to_str(disr_val,10u)));
    ebml::end_tag(ebml_w);
}

fn encode_tag_id(ebml_w: ebml::writer, id: def_id) {
    ebml::start_tag(ebml_w, tag_items_data_item_tag_id);
    ebml_w.writer.write(str::bytes(def_to_str(id)));
    ebml::end_tag(ebml_w);
}

fn encode_tag_variant_info(ecx: @encode_ctxt, ebml_w: ebml::writer,
                           id: node_id, variants: [variant],
                           &index: [entry<int>], ty_params: [ty_param]) {
    let disr_val = 0;
    for variant: variant in variants {
        index += [{val: variant.node.id, pos: ebml_w.writer.tell()}];
        ebml::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(variant.node.id));
        encode_family(ebml_w, 'v' as u8);
        encode_tag_id(ebml_w, local_def(id));
        encode_type(ecx, ebml_w,
                    node_id_to_monotype(ecx.ccx.tcx, variant.node.id));
        if vec::len::<variant_arg>(variant.node.args) > 0u {
            encode_symbol(ecx, ebml_w, variant.node.id);
        }
        encode_discriminant(ecx, ebml_w, variant.node.id);
        if variant.node.disr_val != disr_val {
            encode_disr_val(ecx, ebml_w, variant.node.disr_val);
            disr_val = variant.node.disr_val;
        }
        encode_type_param_bounds(ebml_w, ecx, ty_params);
        ebml::end_tag(ebml_w);
        disr_val += 1;
    }
}

fn encode_info_for_mod(ebml_w: ebml::writer, md: _mod,
                       id: node_id, name: ident) {
    ebml::start_tag(ebml_w, tag_items_data_item);
    encode_def_id(ebml_w, local_def(id));
    encode_family(ebml_w, 'm' as u8);
    encode_name(ebml_w, name);
    for i in md.items {
        alt i.node {
          item_impl(_, _, _, _) {
            if ast_util::is_exported(i.ident, md) {
                ebml::start_tag(ebml_w, tag_mod_impl);
                ebml_w.writer.write(str::bytes(def_to_str(local_def(i.id))));
                ebml::end_tag(ebml_w);
            }
          }
          _ {}
        }
    }
    ebml::end_tag(ebml_w);
}

fn encode_info_for_item(ecx: @encode_ctxt, ebml_w: ebml::writer, item: @item,
                        &index: [entry<int>]) {
    let tcx = ecx.ccx.tcx;
    alt item.node {
      item_const(_, _) {
        ebml::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'c' as u8);
        encode_type(ecx, ebml_w, node_id_to_monotype(tcx, item.id));
        encode_symbol(ecx, ebml_w, item.id);
        ebml::end_tag(ebml_w);
      }
      item_fn(decl, tps, _) {
        ebml::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w,
                      alt decl.purity {
                        unsafe_fn. { 'u' }
                        pure_fn. { 'p' }
                        impure_fn. { 'f' }
                      } as u8);
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_monotype(tcx, item.id));
        encode_symbol(ecx, ebml_w, item.id);
        ebml::end_tag(ebml_w);
      }
      item_mod(m) {
        encode_info_for_mod(ebml_w, m, item.id, item.ident);
      }
      item_native_mod(_) {
        ebml::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'n' as u8);
        encode_name(ebml_w, item.ident);
        ebml::end_tag(ebml_w);
      }
      item_ty(_, tps) {
        ebml::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'y' as u8);
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_monotype(tcx, item.id));
        encode_name(ebml_w, item.ident);
        ebml::end_tag(ebml_w);
      }
      item_tag(variants, tps) {
        ebml::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 't' as u8);
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_monotype(tcx, item.id));
        encode_name(ebml_w, item.ident);
        for v: variant in variants {
            encode_variant_id(ebml_w, local_def(v.node.id));
        }
        ebml::end_tag(ebml_w);
        encode_tag_variant_info(ecx, ebml_w, item.id, variants, index, tps);
      }
      item_res(_, tps, _, _, ctor_id) {
        let fn_ty = node_id_to_monotype(tcx, ctor_id);

        ebml::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(ctor_id));
        encode_family(ebml_w, 'y' as u8);
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, ty::ty_fn_ret(tcx, fn_ty));
        encode_name(ebml_w, item.ident);
        encode_symbol(ecx, ebml_w, item.id);
        ebml::end_tag(ebml_w);

        index += [{val: ctor_id, pos: ebml_w.writer.tell()}];
        ebml::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(ctor_id));
        encode_family(ebml_w, 'f' as u8);
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, fn_ty);
        encode_symbol(ecx, ebml_w, ctor_id);
        ebml::end_tag(ebml_w);
      }
      item_impl(tps, ifce, _, methods) {
        ebml::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'i' as u8);
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_monotype(tcx, item.id));
        encode_name(ebml_w, item.ident);
        for m in methods {
            ebml::start_tag(ebml_w, tag_item_method);
            ebml_w.writer.write(str::bytes(def_to_str(local_def(m.id))));
            ebml::end_tag(ebml_w);
        }
        alt ifce {
          some(_) {
            encode_symbol(ecx, ebml_w, item.id);
            let i_ty = ty::lookup_item_type(tcx, local_def(item.id)).ty;
            ebml::start_tag(ebml_w, tag_impl_iface);
            write_type(ecx, ebml_w, i_ty);
            ebml::end_tag(ebml_w);
          }
          _ {}
        }
        ebml::end_tag(ebml_w);

        for m in methods {
            index += [{val: m.id, pos: ebml_w.writer.tell()}];
            ebml::start_tag(ebml_w, tag_items_data_item);
            encode_def_id(ebml_w, local_def(m.id));
            encode_family(ebml_w, 'f' as u8);
            encode_type_param_bounds(ebml_w, ecx, tps + m.tps);
            encode_type(ecx, ebml_w,
                        node_id_to_monotype(tcx, m.id));
            encode_name(ebml_w, m.ident);
            encode_symbol(ecx, ebml_w, m.id);
            ebml::end_tag(ebml_w);
        }
      }
      item_iface(tps, ms) {
        ebml::start_tag(ebml_w, tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'I' as u8);
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_monotype(tcx, item.id));
        encode_name(ebml_w, item.ident);
        let i = 0u;
        for mty in *ty::iface_methods(tcx, local_def(item.id)) {
            ebml::start_tag(ebml_w, tag_item_method);
            encode_name(ebml_w, mty.ident);
            encode_type_param_bounds(ebml_w, ecx, ms[i].tps);
            encode_type(ecx, ebml_w, ty::mk_fn(tcx, mty.fty));
            ebml::end_tag(ebml_w);
            i += 1u;
        }
        ebml::end_tag(ebml_w);
      }
    }
}

fn encode_info_for_native_item(ecx: @encode_ctxt, ebml_w: ebml::writer,
                               nitem: @native_item) {
    ebml::start_tag(ebml_w, tag_items_data_item);
    alt nitem.node {
      native_item_ty. {
        encode_def_id(ebml_w, local_def(nitem.id));
        encode_family(ebml_w, 'T' as u8);
        encode_type(ecx, ebml_w,
                    ty::mk_native(ecx.ccx.tcx, local_def(nitem.id)));
      }
      native_item_fn(fn_decl, tps) {
        let letter =
            alt fn_decl.purity {
              unsafe_fn. { 'U' }
              pure_fn. { 'P' }   // this is currently impossible, but hey.
              impure_fn. { 'F' }
            } as u8;
        encode_def_id(ebml_w, local_def(nitem.id));
        encode_family(ebml_w, letter);
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_monotype(ecx.ccx.tcx, nitem.id));
        encode_symbol(ecx, ebml_w, nitem.id);
      }
    }
    ebml::end_tag(ebml_w);
}

fn encode_info_for_items(ecx: @encode_ctxt, ebml_w: ebml::writer,
                         crate_mod: _mod) -> [entry<int>] {
    let index: [entry<int>] = [];
    ebml::start_tag(ebml_w, tag_items_data);
    index += [{val: crate_node_id, pos: ebml_w.writer.tell()}];
    encode_info_for_mod(ebml_w, crate_mod, crate_node_id, "");
    ecx.ccx.ast_map.items {|key, val|
        alt val {
          middle::ast_map::node_item(i) {
            index += [{val: key, pos: ebml_w.writer.tell()}];
            encode_info_for_item(ecx, ebml_w, i, index);
          }
          middle::ast_map::node_native_item(i) {
            index += [{val: key, pos: ebml_w.writer.tell()}];
            encode_info_for_native_item(ecx, ebml_w, i);
          }
          _ { }
        }
    };
    ebml::end_tag(ebml_w);
    ret index;
}


// Path and definition ID indexing

fn create_index<T: copy>(index: [entry<T>], hash_fn: fn(T) -> uint) ->
   [@[entry<T>]] {
    let buckets: [@mutable [entry<T>]] = [];
    uint::range(0u, 256u) {|_i| buckets += [@mutable []]; };
    for elt: entry<T> in index {
        let h = hash_fn(elt.val);
        *buckets[h % 256u] += [elt];
    }

    let buckets_frozen = [];
    for bucket: @mutable [entry<T>] in buckets {
        buckets_frozen += [@*bucket];
    }
    ret buckets_frozen;
}

fn encode_index<T>(ebml_w: ebml::writer, buckets: [@[entry<T>]],
                   write_fn: fn(io::writer, T)) {
    let writer = ebml_w.writer;
    ebml::start_tag(ebml_w, tag_index);
    let bucket_locs: [uint] = [];
    ebml::start_tag(ebml_w, tag_index_buckets);
    for bucket: @[entry<T>] in buckets {
        bucket_locs += [ebml_w.writer.tell()];
        ebml::start_tag(ebml_w, tag_index_buckets_bucket);
        for elt: entry<T> in *bucket {
            ebml::start_tag(ebml_w, tag_index_buckets_bucket_elt);
            writer.write_be_uint(elt.pos, 4u);
            write_fn(writer, elt.val);
            ebml::end_tag(ebml_w);
        }
        ebml::end_tag(ebml_w);
    }
    ebml::end_tag(ebml_w);
    ebml::start_tag(ebml_w, tag_index_table);
    for pos: uint in bucket_locs { writer.write_be_uint(pos, 4u); }
    ebml::end_tag(ebml_w);
    ebml::end_tag(ebml_w);
}

fn write_str(writer: io::writer, &&s: str) { writer.write_str(s); }

fn write_int(writer: io::writer, &&n: int) {
    writer.write_be_uint(n as uint, 4u);
}

fn encode_meta_item(ebml_w: ebml::writer, mi: meta_item) {
    alt mi.node {
      meta_word(name) {
        ebml::start_tag(ebml_w, tag_meta_item_word);
        ebml::start_tag(ebml_w, tag_meta_item_name);
        ebml_w.writer.write(str::bytes(name));
        ebml::end_tag(ebml_w);
        ebml::end_tag(ebml_w);
      }
      meta_name_value(name, value) {
        alt value.node {
          lit_str(value) {
            ebml::start_tag(ebml_w, tag_meta_item_name_value);
            ebml::start_tag(ebml_w, tag_meta_item_name);
            ebml_w.writer.write(str::bytes(name));
            ebml::end_tag(ebml_w);
            ebml::start_tag(ebml_w, tag_meta_item_value);
            ebml_w.writer.write(str::bytes(value));
            ebml::end_tag(ebml_w);
            ebml::end_tag(ebml_w);
          }
          _ {/* FIXME (#611) */ }
        }
      }
      meta_list(name, items) {
        ebml::start_tag(ebml_w, tag_meta_item_list);
        ebml::start_tag(ebml_w, tag_meta_item_name);
        ebml_w.writer.write(str::bytes(name));
        ebml::end_tag(ebml_w);
        for inner_item: @meta_item in items {
            encode_meta_item(ebml_w, *inner_item);
        }
        ebml::end_tag(ebml_w);
      }
    }
}

fn encode_attributes(ebml_w: ebml::writer, attrs: [attribute]) {
    ebml::start_tag(ebml_w, tag_attributes);
    for attr: attribute in attrs {
        ebml::start_tag(ebml_w, tag_attribute);
        encode_meta_item(ebml_w, attr.node.value);
        ebml::end_tag(ebml_w);
    }
    ebml::end_tag(ebml_w);
}

// So there's a special crate attribute called 'link' which defines the
// metadata that Rust cares about for linking crates. This attribute requires
// 'name' and 'vers' items, so if the user didn't provide them we will throw
// them in anyway with default values.
fn synthesize_crate_attrs(ecx: @encode_ctxt, crate: @crate) -> [attribute] {

    fn synthesize_link_attr(ecx: @encode_ctxt, items: [@meta_item]) ->
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

        let meta_items = [name_item, vers_item] + other_items;
        let link_item = attr::mk_list_item("link", meta_items);

        ret attr::mk_attr(link_item);
    }

    let attrs: [attribute] = [];
    let found_link_attr = false;
    for attr: attribute in crate.node.attrs {
        attrs +=
            if attr::get_attr_name(attr) != "link" {
                [attr]
            } else {
                alt attr.node.value.node {
                  meta_list(n, l) {
                    found_link_attr = true;;
                    [synthesize_link_attr(ecx, l)]
                  }
                  _ { [attr] }
                }
            };
    }

    if !found_link_attr { attrs += [synthesize_link_attr(ecx, [])]; }

    ret attrs;
}

fn encode_crate_deps(ebml_w: ebml::writer, cstore: cstore::cstore) {

    fn get_ordered_names(cstore: cstore::cstore) -> [str] {
        type hashkv = @{key: crate_num, val: cstore::crate_metadata};
        type numname = {crate: crate_num, ident: str};

        // Pull the cnums and names out of cstore
        let pairs: [mutable numname] = [mutable];
        cstore::iter_crate_data(cstore) {|key, val|
            pairs += [mutable {crate: key, ident: val.name}];
        };

        // Sort by cnum
        fn lteq(kv1: numname, kv2: numname) -> bool { kv1.crate <= kv2.crate }
        std::sort::quick_sort(lteq, pairs);

        // Sanity-check the crate numbers
        let expected_cnum = 1;
        for n: numname in pairs {
            assert (n.crate == expected_cnum);
            expected_cnum += 1;
        }

        // Return just the names
        fn name(kv: numname) -> str { kv.ident }
        // mutable -> immutable hack for vec::map
        let immpairs = vec::slice(pairs, 0u, vec::len(pairs));
        ret vec::map(immpairs, name);
    }

    // We're just going to write a list of crate names, with the assumption
    // that they are numbered 1 to n.
    // FIXME: This is not nearly enough to support correct versioning
    // but is enough to get transitive crate dependencies working.
    ebml::start_tag(ebml_w, tag_crate_deps);
    for cname: str in get_ordered_names(cstore) {
        ebml::start_tag(ebml_w, tag_crate_dep);
        ebml_w.writer.write(str::bytes(cname));
        ebml::end_tag(ebml_w);
    }
    ebml::end_tag(ebml_w);
}

fn encode_hash(ebml_w: ebml::writer, hash: str) {
    ebml::start_tag(ebml_w, tag_crate_hash);
    ebml_w.writer.write(str::bytes(hash));
    ebml::end_tag(ebml_w);
}

fn encode_metadata(cx: @crate_ctxt, crate: @crate) -> str {

    let abbrevs = ty::new_ty_hash();
    let ecx = @{ccx: cx, type_abbrevs: abbrevs};

    let buf = io::mk_mem_buffer();
    let buf_w = io::mem_buffer_writer(buf);
    let ebml_w = ebml::create_writer(buf_w);

    encode_hash(ebml_w, cx.link_meta.extras_hash);

    let crate_attrs = synthesize_crate_attrs(ecx, crate);
    encode_attributes(ebml_w, crate_attrs);

    encode_crate_deps(ebml_w, cx.sess.cstore);

    // Encode and index the paths.
    ebml::start_tag(ebml_w, tag_paths);
    let paths_index = encode_item_paths(ebml_w, ecx, crate);
    let paths_buckets = create_index(paths_index, hash_path);
    encode_index(ebml_w, paths_buckets, write_str);
    ebml::end_tag(ebml_w);

    // Encode and index the items.
    ebml::start_tag(ebml_w, tag_items);
    let items_index = encode_info_for_items(ecx, ebml_w, crate.node.module);
    let items_buckets = create_index(items_index, hash_node_id);
    encode_index(ebml_w, items_buckets, write_int);
    ebml::end_tag(ebml_w);

    // Pad this, since something (LLVM, presumably) is cutting off the
    // remaining % 4 bytes.
    buf_w.write([0u8, 0u8, 0u8, 0u8]);
    io::mem_buffer_str(buf)
}

// Get the encoded string for a type
fn encoded_ty(tcx: ty::ctxt, t: ty::t) -> str {
    let cx = @{ds: def_to_str, tcx: tcx, abbrevs: tyencode::ac_no_abbrevs};
    let buf = io::mk_mem_buffer();
    tyencode::enc_ty(io::mem_buffer_writer(buf), cx, t);
    ret io::mem_buffer_str(buf);
}


// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
