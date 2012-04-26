// Metadata encoding

import std::{ebml, map, list};
import std::map::hashmap;
import io::writer_util;
import ebml::writer;
import syntax::ast::*;
import syntax::print::pprust;
import syntax::{ast_util, visit};
import syntax::ast_util::local_def;
import common::*;
import middle::trans::common::crate_ctxt;
import middle::ty;
import middle::ty::node_id_to_type;
import middle::ast_map;
import syntax::attr;
import driver::session::session;
import std::serialization::serializer;
import std::ebml::serializer;

export encode_metadata;
export encoded_ty;
export reachable;

// used by astencode:
export def_to_str;
export encode_ctxt;
export write_type;
export encode_def_id;

type abbrev_map = map::hashmap<ty::t, tyencode::ty_abbrev>;

type encode_ctxt = {ccx: @crate_ctxt,
                    type_abbrevs: abbrev_map};

fn reachable(ecx: @encode_ctxt, id: node_id) -> bool {
    ecx.ccx.reachable.contains_key(id)
}

// Path table encoding
fn encode_name(ebml_w: ebml::writer, name: str) {
    ebml_w.wr_tagged_str(tag_paths_data_name, name);
}

fn encode_def_id(ebml_w: ebml::writer, id: def_id) {
    ebml_w.wr_tagged_str(tag_def_id, def_to_str(id));
}

fn encode_region_param(ebml_w: ebml::writer, rp: region_param) {
    ebml_w.wr_tag(tag_region_param) {||
        serialize_region_param(ebml_w, rp)
    }
}

fn encode_named_def_id(ebml_w: ebml::writer, name: str, id: def_id) {
    ebml_w.wr_tag(tag_paths_data_item) {||
        encode_name(ebml_w, name);
        encode_def_id(ebml_w, id);
    }
}

fn encode_mutability(ebml_w: ebml::writer, mt: class_mutability) {
    ebml_w.wr_tag(tag_class_mut) {||
        ebml_w.writer.write([alt mt { class_immutable { 'i' }
                class_mutable { 'm' } } as u8]/&);
        }
}

type entry<T> = {val: T, pos: uint};

fn encode_enum_variant_paths(ebml_w: ebml::writer, variants: [variant],
                            path: [str], &index: [entry<str>]) {
    for variants.each {|variant|
        add_to_index(ebml_w, path, index, variant.node.name);
        ebml_w.wr_tag(tag_paths_data_item) {||
            encode_name(ebml_w, variant.node.name);
            encode_def_id(ebml_w, local_def(variant.node.id));
        }
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
    for nmod.items.each {|nitem|
        add_to_index(ebml_w, path, index, nitem.ident);
        encode_named_def_id(ebml_w, nitem.ident, local_def(nitem.id));
    }
}

fn encode_class_item_paths(ebml_w: ebml::writer,
     items: [@class_member], path: [str], &index: [entry<str>]) {
    for items.each {|it|
     alt ast_util::class_member_privacy(it) {
          priv { cont; }
          pub {
              let (id, ident) = alt it.node {
                 instance_var(v, _, _, vid, _) { (vid, v) }
                 class_method(it) { (it.id, it.ident) }
              };
              add_to_index(ebml_w, path, index, ident);
              encode_named_def_id(ebml_w, ident, local_def(id));
          }
       }
    }
}

fn encode_module_item_paths(ebml_w: ebml::writer, ecx: @encode_ctxt,
                            module: _mod, path: [str], &index: [entry<str>]) {
    // FIXME factor out add_to_index/start/encode_name/encode_def_id/end ops
    for module.items.each {|it|
        if !reachable(ecx, it.id) ||
           !ast_util::is_exported(it.ident, module) { cont; }
        alt it.node {
          item_const(_, _) {
            add_to_index(ebml_w, path, index, it.ident);
            encode_named_def_id(ebml_w, it.ident, local_def(it.id));
          }
          item_fn(_, tps, _) {
            add_to_index(ebml_w, path, index, it.ident);
            encode_named_def_id(ebml_w, it.ident, local_def(it.id));
          }
          item_mod(_mod) {
            add_to_index(ebml_w, path, index, it.ident);
            ebml_w.start_tag(tag_paths_data_mod);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            encode_module_item_paths(ebml_w, ecx, _mod, path + [it.ident],
                                     index);
            ebml_w.end_tag();
          }
          item_native_mod(nmod) {
            add_to_index(ebml_w, path, index, it.ident);
            ebml_w.start_tag(tag_paths_data_mod);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            encode_native_module_item_paths(ebml_w, nmod, path + [it.ident],
                                            index);
            ebml_w.end_tag();
          }
          item_ty(_, tps, _) {
            add_to_index(ebml_w, path, index, it.ident);
            ebml_w.start_tag(tag_paths_data_item);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            ebml_w.end_tag();
          }
          item_res(_, tps, _, _, ctor_id, _) {
            add_to_index(ebml_w, path, index, it.ident);
            ebml_w.start_tag(tag_paths_data_item);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(ctor_id));
            ebml_w.end_tag();
            add_to_index(ebml_w, path, index, it.ident);
            ebml_w.start_tag(tag_paths_data_item);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            ebml_w.end_tag();
          }
          item_class(_, _, items, ctor, _) {
            add_to_index(ebml_w, path, index, it.ident);
            ebml_w.start_tag(tag_paths_data_item);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            ebml_w.end_tag();
            ebml_w.start_tag(tag_paths);
            add_to_index(ebml_w, path, index, it.ident);
            #debug("ctor id: %d", ctor.node.id);
            encode_named_def_id(ebml_w, it.ident, local_def(ctor.node.id));
            encode_class_item_paths(ebml_w, items, path + [it.ident],
                                      index);
            ebml_w.end_tag();
          }
          item_enum(variants, _, _) {
            add_to_index(ebml_w, path, index, it.ident);
            ebml_w.start_tag(tag_paths_data_item);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            ebml_w.end_tag();
            encode_enum_variant_paths(ebml_w, variants, path, index);
          }
          item_iface(_, _) {
            add_to_index(ebml_w, path, index, it.ident);
            ebml_w.start_tag(tag_paths_data_item);
            encode_name(ebml_w, it.ident);
            encode_def_id(ebml_w, local_def(it.id));
            ebml_w.end_tag();
          }
          item_impl(_, _, _, _) {}
        }
    }
}

fn encode_item_paths(ebml_w: ebml::writer, ecx: @encode_ctxt, crate: @crate)
    -> [entry<str>] {
    let mut index: [entry<str>] = [];
    let mut path: [str] = [];
    ebml_w.start_tag(tag_paths);
    encode_module_item_paths(ebml_w, ecx, crate.node.module, path, index);
    encode_reexport_paths(ebml_w, ecx, index);
    ebml_w.end_tag();
    ret index;
}

fn encode_reexport_paths(ebml_w: ebml::writer,
                         ecx: @encode_ctxt, &index: [entry<str>]) {
    let tcx = ecx.ccx.tcx;
    for ecx.ccx.exp_map.each {|exp_id, defs|
        for defs.each {|def|
            if !def.reexp { cont; }
            let path = alt check tcx.items.get(exp_id) {
              ast_map::node_export(_, path) { ast_map::path_to_str(*path) }
            };
            index += [{val: path, pos: ebml_w.writer.tell()}];
            ebml_w.start_tag(tag_paths_data_item);
            encode_name(ebml_w, path);
            encode_def_id(ebml_w, def.id);
            ebml_w.end_tag();
        }
    }
}


// Item info table encoding
fn encode_family(ebml_w: ebml::writer, c: char) {
    ebml_w.start_tag(tag_items_data_item_family);
    ebml_w.writer.write([c as u8]/&);
    ebml_w.end_tag();
}

fn def_to_str(did: def_id) -> str { ret #fmt["%d:%d", did.crate, did.node]; }

fn encode_type_param_bounds(ebml_w: ebml::writer, ecx: @encode_ctxt,
                            params: [ty_param]) {
    let ty_str_ctxt = @{ds: def_to_str,
                        tcx: ecx.ccx.tcx,
                        reachable: reachable(ecx, _),
                        abbrevs: tyencode::ac_use_abbrevs(ecx.type_abbrevs)};
    for params.each {|param|
        ebml_w.start_tag(tag_items_data_item_ty_param_bounds);
        let bs = ecx.ccx.tcx.ty_param_bounds.get(param.id);
        tyencode::enc_bounds(ebml_w.writer, ty_str_ctxt, bs);
        ebml_w.end_tag();
    }
}

fn encode_variant_id(ebml_w: ebml::writer, vid: def_id) {
    ebml_w.start_tag(tag_items_data_item_variant);
    ebml_w.writer.write(str::bytes(def_to_str(vid)));
    ebml_w.end_tag();
}

fn write_type(ecx: @encode_ctxt, ebml_w: ebml::writer, typ: ty::t) {
    let ty_str_ctxt =
        @{ds: def_to_str,
          tcx: ecx.ccx.tcx,
          reachable: reachable(ecx, _),
          abbrevs: tyencode::ac_use_abbrevs(ecx.type_abbrevs)};
    tyencode::enc_ty(ebml_w.writer, ty_str_ctxt, typ);
}

fn encode_type(ecx: @encode_ctxt, ebml_w: ebml::writer, typ: ty::t) {
    ebml_w.start_tag(tag_items_data_item_type);
    write_type(ecx, ebml_w, typ);
    ebml_w.end_tag();
}

fn encode_symbol(ecx: @encode_ctxt, ebml_w: ebml::writer, id: node_id) {
    ebml_w.start_tag(tag_items_data_item_symbol);
    let sym = alt ecx.ccx.item_symbols.find(id) {
      some(x) { x }
      none { ecx.ccx.tcx.sess.bug(#fmt("encode_symbol: \
                    id not found %d", id)); }
    };
    ebml_w.writer.write(str::bytes(sym));
    ebml_w.end_tag();
}

fn encode_discriminant(ecx: @encode_ctxt, ebml_w: ebml::writer, id: node_id) {
    ebml_w.start_tag(tag_items_data_item_symbol);
    ebml_w.writer.write(str::bytes(ecx.ccx.discrim_symbols.get(id)));
    ebml_w.end_tag();
}

fn encode_disr_val(_ecx: @encode_ctxt, ebml_w: ebml::writer, disr_val: int) {
    ebml_w.start_tag(tag_disr_val);
    ebml_w.writer.write(str::bytes(int::to_str(disr_val,10u)));
    ebml_w.end_tag();
}

fn encode_parent_item(ebml_w: ebml::writer, id: def_id) {
    ebml_w.start_tag(tag_items_data_parent_item);
    ebml_w.writer.write(str::bytes(def_to_str(id)));
    ebml_w.end_tag();
}

fn encode_enum_variant_info(ecx: @encode_ctxt, ebml_w: ebml::writer,
                            id: node_id, variants: [variant],
                            path: ast_map::path, index: @mut [entry<int>],
                            ty_params: [ty_param]) {
    let mut disr_val = 0;
    let mut i = 0;
    let vi = ty::enum_variants(ecx.ccx.tcx, {crate: local_crate, node: id});
    for variants.each {|variant|
        *index += [{val: variant.node.id, pos: ebml_w.writer.tell()}];
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(variant.node.id));
        encode_family(ebml_w, 'v');
        encode_name(ebml_w, variant.node.name);
        encode_parent_item(ebml_w, local_def(id));
        encode_type(ecx, ebml_w,
                    node_id_to_type(ecx.ccx.tcx, variant.node.id));
        if vec::len(variant.node.args) > 0u && ty_params.len() == 0u {
            encode_symbol(ecx, ebml_w, variant.node.id);
        }
        encode_discriminant(ecx, ebml_w, variant.node.id);
        if vi[i].disr_val != disr_val {
            encode_disr_val(ecx, ebml_w, vi[i].disr_val);
            disr_val = vi[i].disr_val;
        }
        encode_type_param_bounds(ebml_w, ecx, ty_params);
        encode_path(ebml_w, path, ast_map::path_name(variant.node.name));
        ebml_w.end_tag();
        disr_val += 1;
        i += 1;
    }
}

fn encode_path(ebml_w: ebml::writer,
               path: ast_map::path,
               name: ast_map::path_elt) {
    fn encode_path_elt(ebml_w: ebml::writer, elt: ast_map::path_elt) {
        let (tag, name) = alt elt {
          ast_map::path_mod(name) { (tag_path_elt_mod, name) }
          ast_map::path_name(name) { (tag_path_elt_name, name) }
        };

        ebml_w.wr_tagged_str(tag, name);
    }

    ebml_w.wr_tag(tag_path) {||
        ebml_w.wr_tagged_u32(tag_path_len, (vec::len(path) + 1u) as u32);
        vec::iter(path) {|pe| encode_path_elt(ebml_w, pe); }
        encode_path_elt(ebml_w, name);
    }
}

fn encode_info_for_mod(ecx: @encode_ctxt, ebml_w: ebml::writer, md: _mod,
                       id: node_id, path: ast_map::path, name: ident) {
    ebml_w.start_tag(tag_items_data_item);
    encode_def_id(ebml_w, local_def(id));
    encode_family(ebml_w, 'm');
    encode_name(ebml_w, name);
    alt ecx.ccx.maps.impl_map.get(id) {
      list::cons(impls, @list::nil) {
        for vec::each(*impls) {|i|
            if ast_util::is_exported(i.ident, md) {
                ebml_w.wr_tagged_str(tag_mod_impl, def_to_str(i.did));
            }
        }
      }
      _ { ecx.ccx.tcx.sess.bug(#fmt("encode_info_for_mod: empty impl_map \
            entry for %?", path)); }
    }
    encode_path(ebml_w, path, ast_map::path_mod(name));
    ebml_w.end_tag();
}

fn encode_privacy(ebml_w: ebml::writer, privacy: privacy) {
    encode_family(ebml_w, alt privacy {
                pub { 'g' } priv { 'j' }});
}

/* Returns an index of items in this class */
fn encode_info_for_class(ecx: @encode_ctxt, ebml_w: ebml::writer,
                         id: node_id, path: ast_map::path,
                         class_tps: [ty_param],
                         items: [@class_member],
                         global_index: @mut[entry<int>])
 -> [entry<int>] {
    let index = @mut [];
    let tcx = ecx.ccx.tcx;
    for items.each {|ci|
     /* We encode both private and public fields -- need to include
        private fields to get the offsets right */
      alt ci.node {
        instance_var(nm, _, mt, id, pr) {
          *index += [{val: id, pos: ebml_w.writer.tell()}];
          ebml_w.start_tag(tag_items_data_item);
          #debug("encode_info_for_class: doing %s %d", nm, id);
          encode_privacy(ebml_w, pr);
          encode_name(ebml_w, nm);
          encode_path(ebml_w, path, ast_map::path_name(nm));
          encode_type(ecx, ebml_w, node_id_to_type(tcx, id));
          encode_mutability(ebml_w, mt);
          encode_def_id(ebml_w, local_def(id));
          ebml_w.end_tag();
        }
        class_method(m) {
           alt m.privacy {
              pub {
                *index += [{val: m.id, pos: ebml_w.writer.tell()}];
                /* Not sure whether we really need to have two indices,
                   but it works for now -- tjc */
                *global_index += [{val: m.id, pos: ebml_w.writer.tell()}];
                let impl_path = path + [ast_map::path_name(m.ident)];
                #debug("encode_info_for_class: doing %s %d", m.ident, m.id);
                encode_info_for_method(ecx, ebml_w, impl_path,
                                       should_inline(m.attrs), id, m,
                                       class_tps + m.tps);
            }
            _ { /* don't encode private methods */ }
          }
        }
      }
    };
    *index
}

fn encode_info_for_fn(ecx: @encode_ctxt, ebml_w: ebml::writer,
                      id: node_id, ident: ident, path: ast_map::path,
                      item: option<inlined_item>, tps: [ty_param],
                      decl: fn_decl) {
        ebml_w.start_tag(tag_items_data_item);
        encode_name(ebml_w, ident);
        encode_def_id(ebml_w, local_def(id));
        encode_family(ebml_w, purity_fn_family(decl.purity));
        encode_type_param_bounds(ebml_w, ecx, tps);
        let its_ty = node_id_to_type(ecx.ccx.tcx, id);
        #debug("fn name = %s ty = %s", ident,
               util::ppaux::ty_to_str(ecx.ccx.tcx, its_ty));
        encode_type(ecx, ebml_w, its_ty);
        encode_path(ebml_w, path, ast_map::path_name(ident));
        alt item {
           some(it) {
             astencode::encode_inlined_item(ecx, ebml_w, path, it);
           }
           none {
             encode_symbol(ecx, ebml_w, id);
           }
        }
        ebml_w.end_tag();
}

fn encode_info_for_method(ecx: @encode_ctxt, ebml_w: ebml::writer,
                          impl_path: ast_map::path, should_inline: bool,
                          parent_id: node_id,
                          m: @method, all_tps: [ty_param]) {
    #debug("encode_info_for_method: %d %s %u", m.id, m.ident, all_tps.len());
    ebml_w.start_tag(tag_items_data_item);
    encode_def_id(ebml_w, local_def(m.id));
    encode_family(ebml_w, purity_fn_family(m.decl.purity));
    encode_type_param_bounds(ebml_w, ecx, all_tps);
    encode_type(ecx, ebml_w, node_id_to_type(ecx.ccx.tcx, m.id));
    encode_name(ebml_w, m.ident);
    encode_path(ebml_w, impl_path, ast_map::path_name(m.ident));
    if all_tps.len() > 0u || should_inline {
        astencode::encode_inlined_item(
           ecx, ebml_w, impl_path,
           ii_method(local_def(parent_id), m));
    } else {
        encode_symbol(ecx, ebml_w, m.id);
    }
    ebml_w.end_tag();
}

fn purity_fn_family(p: purity) -> char {
    alt p {
      unsafe_fn { 'u' }
      pure_fn { 'p' }
      impure_fn { 'f' }
      crust_fn { 'c' }
    }
}


fn should_inline(attrs: [attribute]) -> bool {
    alt attr::find_inline_attr(attrs) {
        attr::ia_none { false }
        attr::ia_hint | attr::ia_always { true }
    }
}


fn encode_info_for_item(ecx: @encode_ctxt, ebml_w: ebml::writer, item: @item,
                        index: @mut [entry<int>], path: ast_map::path) {

    let tcx = ecx.ccx.tcx;
    let must_write =
        alt item.node { item_enum(_, _, _) { true } _ { false } };
    if !must_write && !reachable(ecx, item.id) { ret; }

    fn add_to_index_(item: @item, ebml_w: ebml::writer,
                     index: @mut [entry<int>]) {
        *index += [{val: item.id, pos: ebml_w.writer.tell()}];
    }
    let add_to_index = bind add_to_index_(item, ebml_w, index);

    alt item.node {
      item_const(_, _) {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'c');
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_symbol(ecx, ebml_w, item.id);
        encode_path(ebml_w, path, ast_map::path_name(item.ident));
        ebml_w.end_tag();
      }
      item_fn(decl, tps, _) {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, purity_fn_family(decl.purity));
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_path(ebml_w, path, ast_map::path_name(item.ident));
        if tps.len() > 0u || should_inline(item.attrs) {
            astencode::encode_inlined_item(ecx, ebml_w, path, ii_item(item));
        } else {
            encode_symbol(ecx, ebml_w, item.id);
        }
        ebml_w.end_tag();
      }
      item_mod(m) {
        add_to_index();
        encode_info_for_mod(ecx, ebml_w, m, item.id, path, item.ident);
      }
      item_native_mod(_) {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'n');
        encode_name(ebml_w, item.ident);
        encode_path(ebml_w, path, ast_map::path_name(item.ident));
        ebml_w.end_tag();
      }
      item_ty(_, tps, rp) {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'y');
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_name(ebml_w, item.ident);
        encode_path(ebml_w, path, ast_map::path_name(item.ident));
        encode_region_param(ebml_w, rp);
        ebml_w.end_tag();
      }
      item_enum(variants, tps, rp) {
        add_to_index();
        ebml_w.wr_tag(tag_items_data_item) {||
            encode_def_id(ebml_w, local_def(item.id));
            encode_family(ebml_w, 't');
            encode_type_param_bounds(ebml_w, ecx, tps);
            encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
            encode_name(ebml_w, item.ident);
            for variants.each {|v|
                encode_variant_id(ebml_w, local_def(v.node.id));
            }
            astencode::encode_inlined_item(ecx, ebml_w, path, ii_item(item));
            encode_path(ebml_w, path, ast_map::path_name(item.ident));
            encode_region_param(ebml_w, rp);
        }
        encode_enum_variant_info(ecx, ebml_w, item.id, variants,
                                 path, index, tps);
      }
      item_class(tps, _ifaces, items, ctor, rp) {
        /* First, encode the fields and methods
           These come first because we need to write them to make
           the index, and the index needs to be in the item for the
           class itself */
        let idx = encode_info_for_class(ecx, ebml_w, item.id, path, tps,
                                          items, index);
        /* Index the class*/
        add_to_index();
        /* Now, make an item for the class itself */
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'C');
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_name(ebml_w, item.ident);
        encode_path(ebml_w, path, ast_map::path_name(item.ident));
        encode_region_param(ebml_w, rp);
        /* FIXME: encode ifaces */
        /* Encode def_ids for each field and method
         for methods, write all the stuff get_iface_method
        needs to know*/
        let (fs,ms) = ast_util::split_class_items(items);
        for fs.each {|f|
           ebml_w.start_tag(tag_item_field);
           encode_privacy(ebml_w, f.privacy);
           encode_name(ebml_w, f.ident);
           encode_def_id(ebml_w, local_def(f.id));
           ebml_w.end_tag();
        }
        for ms.each {|m|
           alt m.privacy {
              priv { /* do nothing */ }
              pub {
                ebml_w.start_tag(tag_item_method);
                #debug("Writing %s %d", m.ident, m.id);
                encode_family(ebml_w, purity_fn_family(m.decl.purity));
                encode_name(ebml_w, m.ident);
                encode_type_param_bounds(ebml_w, ecx, tps + m.tps);
                encode_type(ecx, ebml_w, node_id_to_type(tcx, m.id));
                encode_def_id(ebml_w, local_def(m.id));
                ebml_w.end_tag();
              }
           }
        }
        /* Each class has its own index -- encode it */
        let bkts = create_index(idx, hash_node_id);
        encode_index(ebml_w, bkts, write_int);
        ebml_w.end_tag();
      }
      item_res(_, tps, _, _, ctor_id, rp) {
        add_to_index();
        let fn_ty = node_id_to_type(tcx, ctor_id);

        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(ctor_id));
        encode_family(ebml_w, 'y');
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, ty::ty_fn_ret(fn_ty));
        encode_name(ebml_w, item.ident);
        astencode::encode_inlined_item(ecx, ebml_w, path, ii_item(item));
        if (tps.len() == 0u) {
            encode_symbol(ecx, ebml_w, item.id);
        }
        encode_path(ebml_w, path, ast_map::path_name(item.ident));
        encode_region_param(ebml_w, rp);
        ebml_w.end_tag();

        *index += [{val: ctor_id, pos: ebml_w.writer.tell()}];
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(ctor_id));
        encode_family(ebml_w, 'f');
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, fn_ty);
        encode_parent_item(ebml_w, local_def(item.id));
        encode_path(ebml_w, path, ast_map::path_name(item.ident));
        ebml_w.end_tag();
      }
      item_impl(tps, ifce, _, methods) {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'i');
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_name(ebml_w, item.ident);
        for methods.each {|m|
            ebml_w.start_tag(tag_item_method);
            ebml_w.writer.write(str::bytes(def_to_str(local_def(m.id))));
            ebml_w.end_tag();
        }
        alt ifce {
          some(t) {
            let i_ty = ty::node_id_to_type(tcx, t.id);
            ebml_w.start_tag(tag_impl_iface);
            write_type(ecx, ebml_w, i_ty);
            ebml_w.end_tag();
          }
          _ {}
        }
        encode_path(ebml_w, path, ast_map::path_name(item.ident));
        ebml_w.end_tag();

        let impl_path = path + [ast_map::path_name(item.ident)];
        for methods.each {|m|
            *index += [{val: m.id, pos: ebml_w.writer.tell()}];
            encode_info_for_method(ecx, ebml_w, impl_path,
                   should_inline(m.attrs), item.id, m, tps + m.tps);
        }
      }
      item_iface(tps, ms) {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'I');
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_name(ebml_w, item.ident);
        let mut i = 0u;
        for vec::each(*ty::iface_methods(tcx, local_def(item.id))) {|mty|
            ebml_w.start_tag(tag_item_method);
            encode_name(ebml_w, mty.ident);
            encode_type_param_bounds(ebml_w, ecx, ms[i].tps);
            encode_type(ecx, ebml_w, ty::mk_fn(tcx, mty.fty));
            encode_family(ebml_w, purity_fn_family(mty.purity));
            ebml_w.end_tag();
            i += 1u;
        }
        encode_path(ebml_w, path, ast_map::path_name(item.ident));
        ebml_w.end_tag();
      }
    }
}

fn encode_info_for_native_item(ecx: @encode_ctxt, ebml_w: ebml::writer,
                               nitem: @native_item,
                               index: @mut [entry<int>],
                               path: ast_map::path, abi: native_abi) {
    if !reachable(ecx, nitem.id) { ret; }
    *index += [{val: nitem.id, pos: ebml_w.writer.tell()}];

    ebml_w.start_tag(tag_items_data_item);
    alt nitem.node {
      native_item_fn(fn_decl, tps) {
        encode_def_id(ebml_w, local_def(nitem.id));
        encode_family(ebml_w, purity_fn_family(fn_decl.purity));
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_type(ecx.ccx.tcx, nitem.id));
        if abi == native_abi_rust_intrinsic {
            astencode::encode_inlined_item(ecx, ebml_w, path,
                                           ii_native(nitem));
        } else {
            encode_symbol(ecx, ebml_w, nitem.id);
        }
        encode_path(ebml_w, path, ast_map::path_name(nitem.ident));
      }
    }
    ebml_w.end_tag();
}

fn encode_info_for_items(ecx: @encode_ctxt, ebml_w: ebml::writer,
                         crate: @crate) -> [entry<int>] {
    let index = @mut [];
    ebml_w.start_tag(tag_items_data);
    *index += [{val: crate_node_id, pos: ebml_w.writer.tell()}];
    encode_info_for_mod(ecx, ebml_w, crate.node.module,
                        crate_node_id, [], "");
    visit::visit_crate(*crate, (), visit::mk_vt(@{
        visit_expr: {|_e, _cx, _v|},
        visit_item: {|i, cx, v|
            visit::visit_item(i, cx, v);
            alt check ecx.ccx.tcx.items.get(i.id) {
              ast_map::node_item(_, pt) {
                encode_info_for_item(ecx, ebml_w, i, index, *pt);
                /* encode ctor, then encode items */
                alt i.node {
                  item_class(tps, _, _, ctor, _) {
                   /* this is assuming that ctors aren't inlined...
                      probably shouldn't assume that */
                   #debug("encoding info for ctor %s %d", i.ident,
                          ctor.node.id);
                   *index += [{val: ctor.node.id, pos: ebml_w.writer.tell()}];
                   encode_info_for_fn(ecx, ebml_w, ctor.node.id, i.ident,
                      *pt, if tps.len() > 0u {
                             some(ii_ctor(ctor, i.ident, tps,
                                          local_def(i.id))) }
                           else { none }, tps, ctor.node.dec)
                  }
                  _ {}
                }
              }
            }
        },
        visit_native_item: {|ni, cx, v|
            visit::visit_native_item(ni, cx, v);
            alt check ecx.ccx.tcx.items.get(ni.id) {
              ast_map::node_native_item(_, abi, pt) {
                encode_info_for_native_item(ecx, ebml_w, ni, index, *pt, abi);
              }
            }
        }
        with *visit::default_visitor()
    }));
    ebml_w.end_tag();
    ret *index;
}


// Path and definition ID indexing

fn create_index<T: copy>(index: [entry<T>], hash_fn: fn@(T) -> uint) ->
   [@[entry<T>]] {
    let mut buckets: [@mut [entry<T>]] = [];
    uint::range(0u, 256u) {|_i| buckets += [@mut []]; };
    for index.each {|elt|
        let h = hash_fn(elt.val);
        *buckets[h % 256u] += [elt];
    }

    let mut buckets_frozen = [];
    for buckets.each {|bucket|
        buckets_frozen += [@*bucket];
    }
    ret buckets_frozen;
}

fn encode_index<T>(ebml_w: ebml::writer, buckets: [@[entry<T>]],
                   write_fn: fn(io::writer, T)) {
    let writer = ebml_w.writer;
    ebml_w.start_tag(tag_index);
    let mut bucket_locs: [uint] = [];
    ebml_w.start_tag(tag_index_buckets);
    for buckets.each {|bucket|
        bucket_locs += [ebml_w.writer.tell()];
        ebml_w.start_tag(tag_index_buckets_bucket);
        for vec::each(*bucket) {|elt|
            ebml_w.start_tag(tag_index_buckets_bucket_elt);
            writer.write_be_uint(elt.pos, 4u);
            write_fn(writer, elt.val);
            ebml_w.end_tag();
        }
        ebml_w.end_tag();
    }
    ebml_w.end_tag();
    ebml_w.start_tag(tag_index_table);
    for bucket_locs.each {|pos| writer.write_be_uint(pos, 4u); }
    ebml_w.end_tag();
    ebml_w.end_tag();
}

fn write_str(writer: io::writer, &&s: str) { writer.write_str(s); }

fn write_int(writer: io::writer, &&n: int) {
    writer.write_be_uint(n as uint, 4u);
}

fn encode_meta_item(ebml_w: ebml::writer, mi: meta_item) {
    alt mi.node {
      meta_word(name) {
        ebml_w.start_tag(tag_meta_item_word);
        ebml_w.start_tag(tag_meta_item_name);
        ebml_w.writer.write(str::bytes(name));
        ebml_w.end_tag();
        ebml_w.end_tag();
      }
      meta_name_value(name, value) {
        alt value.node {
          lit_str(value) {
            ebml_w.start_tag(tag_meta_item_name_value);
            ebml_w.start_tag(tag_meta_item_name);
            ebml_w.writer.write(str::bytes(name));
            ebml_w.end_tag();
            ebml_w.start_tag(tag_meta_item_value);
            ebml_w.writer.write(str::bytes(value));
            ebml_w.end_tag();
            ebml_w.end_tag();
          }
          _ {/* FIXME (#611) */ }
        }
      }
      meta_list(name, items) {
        ebml_w.start_tag(tag_meta_item_list);
        ebml_w.start_tag(tag_meta_item_name);
        ebml_w.writer.write(str::bytes(name));
        ebml_w.end_tag();
        for items.each {|inner_item|
            encode_meta_item(ebml_w, *inner_item);
        }
        ebml_w.end_tag();
      }
    }
}

fn encode_attributes(ebml_w: ebml::writer, attrs: [attribute]) {
    ebml_w.start_tag(tag_attributes);
    for attrs.each {|attr|
        ebml_w.start_tag(tag_attribute);
        encode_meta_item(ebml_w, attr.node.value);
        ebml_w.end_tag();
    }
    ebml_w.end_tag();
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

    let mut attrs: [attribute] = [];
    let mut found_link_attr = false;
    for crate.node.attrs.each {|attr|
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

    fn get_ordered_deps(cstore: cstore::cstore) -> [decoder::crate_dep] {
        type hashkv = @{key: crate_num, val: cstore::crate_metadata};
        type numdep = decoder::crate_dep;

        // Pull the cnums and name,vers,hash out of cstore
        let mut deps: [mut numdep] = [mut];
        cstore::iter_crate_data(cstore) {|key, val|
            let dep = {cnum: key, name: val.name,
                       vers: decoder::get_crate_vers(val.data),
                       hash:  decoder::get_crate_hash(val.data)};
            deps += [mut dep];
        };

        // Sort by cnum
        fn lteq(kv1: numdep, kv2: numdep) -> bool { kv1.cnum <= kv2.cnum }
        std::sort::quick_sort(lteq, deps);

        // Sanity-check the crate numbers
        let mut expected_cnum = 1;
        for deps.each {|n|
            assert (n.cnum == expected_cnum);
            expected_cnum += 1;
        }

        // mut -> immutable hack for vec::map
        ret vec::slice(deps, 0u, vec::len(deps));
    }

    // We're just going to write a list of crate 'name-hash-version's, with
    // the assumption that they are numbered 1 to n.
    // FIXME: This is not nearly enough to support correct versioning
    // but is enough to get transitive crate dependencies working.
    ebml_w.start_tag(tag_crate_deps);
    for get_ordered_deps(cstore).each {|dep|
        encode_crate_dep(ebml_w, dep);
    }
    ebml_w.end_tag();
}

fn encode_crate_dep(ebml_w: ebml::writer, dep: decoder::crate_dep) {
    ebml_w.start_tag(tag_crate_dep);
    ebml_w.start_tag(tag_crate_dep_name);
    ebml_w.writer.write(str::bytes(dep.name));
    ebml_w.end_tag();
    ebml_w.start_tag(tag_crate_dep_vers);
    ebml_w.writer.write(str::bytes(dep.vers));
    ebml_w.end_tag();
    ebml_w.start_tag(tag_crate_dep_hash);
    ebml_w.writer.write(str::bytes(dep.hash));
    ebml_w.end_tag();
    ebml_w.end_tag();
}

fn encode_hash(ebml_w: ebml::writer, hash: str) {
    ebml_w.start_tag(tag_crate_hash);
    ebml_w.writer.write(str::bytes(hash));
    ebml_w.end_tag();
}

fn encode_metadata(cx: @crate_ctxt, crate: @crate) -> [u8] {
    let ecx = @{ccx: cx, type_abbrevs: ty::new_ty_hash()};

    let buf = io::mem_buffer();
    let buf_w = io::mem_buffer_writer(buf);
    let ebml_w = ebml::writer(buf_w);

    encode_hash(ebml_w, cx.link_meta.extras_hash);

    let crate_attrs = synthesize_crate_attrs(ecx, crate);
    encode_attributes(ebml_w, crate_attrs);

    encode_crate_deps(ebml_w, cx.sess.cstore);

    // Encode and index the paths.
    ebml_w.start_tag(tag_paths);
    let paths_index = encode_item_paths(ebml_w, ecx, crate);
    let paths_buckets = create_index(paths_index, hash_path);
    encode_index(ebml_w, paths_buckets, write_str);
    ebml_w.end_tag();

    // Encode and index the items.
    ebml_w.start_tag(tag_items);
    let items_index = encode_info_for_items(ecx, ebml_w, crate);
    let items_buckets = create_index(items_index, hash_node_id);
    encode_index(ebml_w, items_buckets, write_int);
    ebml_w.end_tag();

    // Pad this, since something (LLVM, presumably) is cutting off the
    // remaining % 4 bytes.
    buf_w.write([0u8, 0u8, 0u8, 0u8]/&);
    io::mem_buffer_buf(buf)
}

// Get the encoded string for a type
fn encoded_ty(tcx: ty::ctxt, t: ty::t) -> str {
    let cx = @{ds: def_to_str,
               tcx: tcx,
               reachable: {|_id| false},
               abbrevs: tyencode::ac_no_abbrevs};
    let buf = io::mem_buffer();
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
