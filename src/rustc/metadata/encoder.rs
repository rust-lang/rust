// Metadata encoding

import util::ppaux::ty_to_str;

import std::{ebml, map};
import std::map::hashmap;
import io::writer_util;
import ebml::writer;
import syntax::ast::*;
import syntax::print::pprust;
import syntax::{ast_util, visit};
import syntax::ast_util::*;
import common::*;
import middle::ty;
import middle::ty::node_id_to_type;
import syntax::ast_map;
import syntax::attr;
import std::serialization::serializer;
import std::ebml::serializer;
import syntax::ast;
import syntax::diagnostic::span_handler;

export encode_parms;
export encode_metadata;
export encoded_ty;
export reachable;
export encode_inlined_item;

// used by astencode:
export def_to_str;
export encode_ctxt;
export write_type;
export encode_def_id;

type abbrev_map = map::hashmap<ty::t, tyencode::ty_abbrev>;

type encode_inlined_item = fn@(ecx: @encode_ctxt,
                               ebml_w: ebml::writer,
                               path: ast_map::path,
                               ii: ast::inlined_item);

type encode_parms = {
    diag: span_handler,
    tcx: ty::ctxt,
    reachable: hashmap<ast::node_id, ()>,
    reexports: ~[(~str, def_id)],
    impl_map: fn@(ast::node_id) -> ~[(ident, def_id)],
    item_symbols: hashmap<ast::node_id, ~str>,
    discrim_symbols: hashmap<ast::node_id, ~str>,
    link_meta: link_meta,
    cstore: cstore::cstore,
    encode_inlined_item: encode_inlined_item
};

enum encode_ctxt = {
    diag: span_handler,
    tcx: ty::ctxt,
    reachable: hashmap<ast::node_id, ()>,
    reexports: ~[(~str, def_id)],
    impl_map: fn@(ast::node_id) -> ~[(ident, def_id)],
    item_symbols: hashmap<ast::node_id, ~str>,
    discrim_symbols: hashmap<ast::node_id, ~str>,
    link_meta: link_meta,
    cstore: cstore::cstore,
    encode_inlined_item: encode_inlined_item,
    type_abbrevs: abbrev_map
};

fn reachable(ecx: @encode_ctxt, id: node_id) -> bool {
    ecx.reachable.contains_key(id)
}

// Path table encoding
fn encode_name(ebml_w: ebml::writer, name: ident) {
    ebml_w.wr_tagged_str(tag_paths_data_name, *name);
}

fn encode_def_id(ebml_w: ebml::writer, id: def_id) {
    ebml_w.wr_tagged_str(tag_def_id, def_to_str(id));
}

/* Encodes the given name, then def_id as tagged strings */
fn encode_name_and_def_id(ebml_w: ebml::writer, nm: ident,
                          id: node_id) {
    encode_name(ebml_w, nm);
    encode_def_id(ebml_w, local_def(id));
}

fn encode_region_param(ecx: @encode_ctxt, ebml_w: ebml::writer,
                       it: @ast::item) {
    let rp = ecx.tcx.region_paramd_items.contains_key(it.id);
    if rp { do ebml_w.wr_tag(tag_region_param) { } }
}

fn encode_named_def_id(ebml_w: ebml::writer, name: ident, id: def_id) {
    do ebml_w.wr_tag(tag_paths_data_item) {
        encode_name(ebml_w, name);
        encode_def_id(ebml_w, id);
    }
}

fn encode_mutability(ebml_w: ebml::writer, mt: class_mutability) {
    do ebml_w.wr_tag(tag_class_mut) {
        ebml_w.writer.write(&[alt mt { class_immutable { 'i' }
                class_mutable { 'm' } } as u8]);
        }
}

type entry<T> = {val: T, pos: uint};

fn encode_enum_variant_paths(ebml_w: ebml::writer, variants: ~[variant],
                            path: ~[ident], &index: ~[entry<~str>]) {
    for variants.each |variant| {
        add_to_index(ebml_w, path, index, variant.node.name);
        do ebml_w.wr_tag(tag_paths_data_item) {
            encode_name(ebml_w, variant.node.name);
            encode_def_id(ebml_w, local_def(variant.node.id));
        }
    }
}

fn add_to_index(ebml_w: ebml::writer, path: &[ident], &index: ~[entry<~str>],
                name: ident) {
    let mut full_path = ~[];
    vec::push_all(full_path, path);
    vec::push(full_path, name);
    vec::push(index, {val: ast_util::path_name_i(full_path),
                      pos: ebml_w.writer.tell()});
}

fn encode_foreign_module_item_paths(ebml_w: ebml::writer, nmod: foreign_mod,
                                    path: ~[ident], &index: ~[entry<~str>]) {
    for nmod.items.each |nitem| {
      add_to_index(ebml_w, path, index, nitem.ident);
      do ebml_w.wr_tag(tag_paths_foreign_path) {
          encode_name(ebml_w, nitem.ident);
          encode_def_id(ebml_w, local_def(nitem.id));
      }
    }
}

fn encode_class_item_paths(ebml_w: ebml::writer,
     items: ~[@class_member], path: ~[ident], &index: ~[entry<~str>]) {
    for items.each |it| {
     alt ast_util::class_member_visibility(it) {
          private { again; }
          public {
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
                            module: _mod, path: ~[ident],
                            &index: ~[entry<~str>]) {
    for module.items.each |it| {
        if !reachable(ecx, it.id) ||
           !ast_util::is_exported(it.ident, module) { again; }
        if !ast_util::is_item_impl(it) {
            add_to_index(ebml_w, path, index, it.ident);
        }
        alt it.node {
          item_const(_, _) {
            encode_named_def_id(ebml_w, it.ident, local_def(it.id));
          }
          item_fn(_, tps, _) {
            encode_named_def_id(ebml_w, it.ident, local_def(it.id));
          }
          item_mod(_mod) {
            do ebml_w.wr_tag(tag_paths_data_mod) {
               encode_name_and_def_id(ebml_w, it.ident, it.id);
               encode_module_item_paths(ebml_w, ecx, _mod,
                                        vec::append_one(path, it.ident),
                                        index);
            }
          }
          item_foreign_mod(nmod) {
            do ebml_w.wr_tag(tag_paths_data_mod) {
              encode_name_and_def_id(ebml_w, it.ident, it.id);
              encode_foreign_module_item_paths(
                  ebml_w, nmod,
                  vec::append_one(path, it.ident), index);
            }
          }
          item_ty(_, tps) {
            do ebml_w.wr_tag(tag_paths_data_item) {
              encode_name_and_def_id(ebml_w, it.ident, it.id);
            }
          }
          item_class(_, _, items, ctor, m_dtor) {
            do ebml_w.wr_tag(tag_paths_data_item) {
                encode_name_and_def_id(ebml_w, it.ident, it.id);
            }
            do ebml_w.wr_tag(tag_paths) {
                // We add the same ident twice: for the
                // class and for its ctor
                add_to_index(ebml_w, path, index, it.ident);
                encode_named_def_id(ebml_w, it.ident,
                                    local_def(ctor.node.id));
                encode_class_item_paths(ebml_w, items,
                                        vec::append_one(path, it.ident),
                                        index);
            }
          }
          item_enum(variants, _) {
            do ebml_w.wr_tag(tag_paths_data_item) {
                  encode_name_and_def_id(ebml_w, it.ident, it.id);
              }
              encode_enum_variant_paths(ebml_w, variants, path, index);
          }
          item_trait(*) {
            do ebml_w.wr_tag(tag_paths_data_item) {
                  encode_name_and_def_id(ebml_w, it.ident, it.id);
              }
          }
          item_impl(*) {}
          item_mac(*) { fail ~"item macros unimplemented" }
        }
    }
}

fn encode_trait_ref(ebml_w: ebml::writer, ecx: @encode_ctxt, t: @trait_ref) {
    ebml_w.start_tag(tag_impl_trait);
    encode_type(ecx, ebml_w, node_id_to_type(ecx.tcx, t.ref_id));
    ebml_w.end_tag();
}

fn encode_item_paths(ebml_w: ebml::writer, ecx: @encode_ctxt, crate: @crate)
    -> ~[entry<~str>] {
    let mut index: ~[entry<~str>] = ~[];
    let mut path: ~[ident] = ~[];
    ebml_w.start_tag(tag_paths);
    encode_module_item_paths(ebml_w, ecx, crate.node.module, path, index);
    encode_reexport_paths(ebml_w, ecx, index);
    ebml_w.end_tag();
    ret index;
}

fn encode_reexport_paths(ebml_w: ebml::writer,
                         ecx: @encode_ctxt, &index: ~[entry<~str>]) {
    for ecx.reexports.each |reexport| {
        let (path, def_id) = reexport;
        vec::push(index, {val: path, pos: ebml_w.writer.tell()});
        // List metadata ignores tag_paths_foreign_path things, but
        // other things look at it.
        ebml_w.start_tag(tag_paths_foreign_path);
        encode_name(ebml_w, @path);
        encode_def_id(ebml_w, def_id);
        ebml_w.end_tag();
    }
}


// Item info table encoding
fn encode_family(ebml_w: ebml::writer, c: char) {
    ebml_w.start_tag(tag_items_data_item_family);
    ebml_w.writer.write(&[c as u8]);
    ebml_w.end_tag();
}

fn def_to_str(did: def_id) -> ~str { ret #fmt["%d:%d", did.crate, did.node]; }

fn encode_type_param_bounds(ebml_w: ebml::writer, ecx: @encode_ctxt,
                            params: ~[ty_param]) {
    let ty_str_ctxt = @{diag: ecx.diag,
                        ds: def_to_str,
                        tcx: ecx.tcx,
                        reachable: |a| reachable(ecx, a),
                        abbrevs: tyencode::ac_use_abbrevs(ecx.type_abbrevs)};
    for params.each |param| {
        ebml_w.start_tag(tag_items_data_item_ty_param_bounds);
        let bs = ecx.tcx.ty_param_bounds.get(param.id);
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
        @{diag: ecx.diag,
          ds: def_to_str,
          tcx: ecx.tcx,
          reachable: |a| reachable(ecx, a),
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
    let sym = alt ecx.item_symbols.find(id) {
      some(x) { x }
      none {
        ecx.diag.handler().bug(
            #fmt("encode_symbol: id not found %d", id));
      }
    };
    ebml_w.writer.write(str::bytes(sym));
    ebml_w.end_tag();
}

fn encode_discriminant(ecx: @encode_ctxt, ebml_w: ebml::writer, id: node_id) {
    ebml_w.start_tag(tag_items_data_item_symbol);
    ebml_w.writer.write(str::bytes(ecx.discrim_symbols.get(id)));
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
                            id: node_id, variants: ~[variant],
                            path: ast_map::path, index: @mut ~[entry<int>],
                            ty_params: ~[ty_param]) {
    let mut disr_val = 0;
    let mut i = 0;
    let vi = ty::enum_variants(ecx.tcx, {crate: local_crate, node: id});
    for variants.each |variant| {
        vec::push(*index, {val: variant.node.id, pos: ebml_w.writer.tell()});
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(variant.node.id));
        encode_family(ebml_w, 'v');
        encode_name(ebml_w, variant.node.name);
        encode_parent_item(ebml_w, local_def(id));
        encode_type(ecx, ebml_w,
                    node_id_to_type(ecx.tcx, variant.node.id));
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

        ebml_w.wr_tagged_str(tag, *name);
    }

    do ebml_w.wr_tag(tag_path) {
        ebml_w.wr_tagged_u32(tag_path_len, (vec::len(path) + 1u) as u32);
        do vec::iter(path) |pe| { encode_path_elt(ebml_w, pe); }
        encode_path_elt(ebml_w, name);
    }
}

fn encode_info_for_mod(ecx: @encode_ctxt, ebml_w: ebml::writer, md: _mod,
                       id: node_id, path: ast_map::path, name: ident) {
    ebml_w.start_tag(tag_items_data_item);
    encode_def_id(ebml_w, local_def(id));
    encode_family(ebml_w, 'm');
    encode_name(ebml_w, name);
    #debug("(encoding info for module) encoding info for module ID %d", id);
    // the impl map contains ref_ids
    let impls = ecx.impl_map(id);
    for impls.each |i| {
        let (ident, did) = i;
        #debug("(encoding info for module) ... encoding impl %s (%?/%?), \
                exported? %?",
               *ident,
               did,
               ast_map::node_id_to_str(ecx.tcx.items, did.node),
               ast_util::is_exported(ident, md));

        ebml_w.start_tag(tag_mod_impl);
        alt ecx.tcx.items.find(did.node) {
          some(ast_map::node_item(it@@{node: cl@item_class(*),_},_)) {
        /* If did stands for a trait
        ref, we need to map it to its parent class */
            ebml_w.wr_str(def_to_str(local_def(it.id)));
          }
          some(ast_map::node_item(@{node: item_impl(_,
                                               some(ifce),_,_),_},_)) {
            ebml_w.wr_str(def_to_str(did));
          }
          some(_) {
            ebml_w.wr_str(def_to_str(did));
          }
          none {
            // Must be a re-export, then!
            // ...or an iface ref
            ebml_w.wr_str(def_to_str(did));
          }
        };
        ebml_w.end_tag();
    } // for

    encode_path(ebml_w, path, ast_map::path_mod(name));
    ebml_w.end_tag();
}

fn encode_visibility(ebml_w: ebml::writer, visibility: visibility) {
    encode_family(ebml_w, alt visibility {
        public { 'g' } private { 'j' }
    });
}

/* Returns an index of items in this class */
fn encode_info_for_class(ecx: @encode_ctxt, ebml_w: ebml::writer,
                         id: node_id, path: ast_map::path,
                         class_tps: ~[ty_param],
                         items: ~[@class_member],
                         global_index: @mut~[entry<int>]) -> ~[entry<int>] {
    /* Each class has its own index, since different classes
       may have fields with the same name */
    let index = @mut ~[];
    let tcx = ecx.tcx;
    for items.each |ci| {
     /* We encode both private and public fields -- need to include
        private fields to get the offsets right */
      alt ci.node {
        instance_var(nm, _, mt, id, vis) {
          vec::push(*index, {val: id, pos: ebml_w.writer.tell()});
          vec::push(*global_index, {val: id, pos: ebml_w.writer.tell()});
          ebml_w.start_tag(tag_items_data_item);
          #debug("encode_info_for_class: doing %s %d", *nm, id);
          encode_visibility(ebml_w, vis);
          encode_name(ebml_w, nm);
          encode_path(ebml_w, path, ast_map::path_name(nm));
          encode_type(ecx, ebml_w, node_id_to_type(tcx, id));
          encode_mutability(ebml_w, mt);
          encode_def_id(ebml_w, local_def(id));
          ebml_w.end_tag();
        }
        class_method(m) {
           alt m.vis {
              public {
                vec::push(*index, {val: m.id, pos: ebml_w.writer.tell()});
                vec::push(*global_index,
                          {val: m.id, pos: ebml_w.writer.tell()});
                let impl_path = vec::append_one(path,
                                                ast_map::path_name(m.ident));
                #debug("encode_info_for_class: doing %s %d", *m.ident, m.id);
                encode_info_for_method(ecx, ebml_w, impl_path,
                                       should_inline(m.attrs), id, m,
                                       vec::append(class_tps, m.tps));
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
                      item: option<inlined_item>, tps: ~[ty_param],
                      decl: fn_decl) {
        ebml_w.start_tag(tag_items_data_item);
        encode_name(ebml_w, ident);
        encode_def_id(ebml_w, local_def(id));
        encode_family(ebml_w, purity_fn_family(decl.purity));
        encode_type_param_bounds(ebml_w, ecx, tps);
        let its_ty = node_id_to_type(ecx.tcx, id);
        #debug("fn name = %s ty = %s its node id = %d", *ident,
               util::ppaux::ty_to_str(ecx.tcx, its_ty), id);
        encode_type(ecx, ebml_w, its_ty);
        encode_path(ebml_w, path, ast_map::path_name(ident));
        alt item {
           some(it) {
             ecx.encode_inlined_item(ecx, ebml_w, path, it);
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
                          m: @method, all_tps: ~[ty_param]) {
    #debug("encode_info_for_method: %d %s %u", m.id, *m.ident, all_tps.len());
    ebml_w.start_tag(tag_items_data_item);
    encode_def_id(ebml_w, local_def(m.id));
    encode_family(ebml_w, purity_fn_family(m.decl.purity));
    encode_type_param_bounds(ebml_w, ecx, all_tps);
    encode_type(ecx, ebml_w, node_id_to_type(ecx.tcx, m.id));
    encode_name(ebml_w, m.ident);
    encode_path(ebml_w, impl_path, ast_map::path_name(m.ident));
    if all_tps.len() > 0u || should_inline {
        ecx.encode_inlined_item(
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
      extern_fn { 'F' }
    }
}


fn should_inline(attrs: ~[attribute]) -> bool {
    alt attr::find_inline_attr(attrs) {
        attr::ia_none { false }
        attr::ia_hint | attr::ia_always { true }
    }
}


fn encode_info_for_item(ecx: @encode_ctxt, ebml_w: ebml::writer, item: @item,
                        index: @mut ~[entry<int>], path: ast_map::path) {

    let tcx = ecx.tcx;
    let must_write =
        alt item.node {
            item_enum(_, _) | item_impl(*) | item_trait(*) | item_class(*) {
                true
            }
            _ {
                false
            }
        };
    if !must_write && !reachable(ecx, item.id) { ret; }

    fn add_to_index_(item: @item, ebml_w: ebml::writer,
                     index: @mut ~[entry<int>]) {
        vec::push(*index, {val: item.id, pos: ebml_w.writer.tell()});
    }
    let add_to_index = |copy ebml_w| add_to_index_(item, ebml_w, index);

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
            ecx.encode_inlined_item(ecx, ebml_w, path, ii_item(item));
        } else {
            encode_symbol(ecx, ebml_w, item.id);
        }
        ebml_w.end_tag();
      }
      item_mod(m) {
        add_to_index();
        encode_info_for_mod(ecx, ebml_w, m, item.id, path, item.ident);
      }
      item_foreign_mod(_) {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'n');
        encode_name(ebml_w, item.ident);
        encode_path(ebml_w, path, ast_map::path_name(item.ident));
        ebml_w.end_tag();
      }
      item_ty(_, tps) {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'y');
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_name(ebml_w, item.ident);
        encode_path(ebml_w, path, ast_map::path_name(item.ident));
        encode_region_param(ecx, ebml_w, item);
        ebml_w.end_tag();
      }
      item_enum(variants, tps) {
        add_to_index();
        do ebml_w.wr_tag(tag_items_data_item) {
            encode_def_id(ebml_w, local_def(item.id));
            encode_family(ebml_w, 't');
            encode_type_param_bounds(ebml_w, ecx, tps);
            encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
            encode_name(ebml_w, item.ident);
            for variants.each |v| {
                encode_variant_id(ebml_w, local_def(v.node.id));
            }
            ecx.encode_inlined_item(ecx, ebml_w, path, ii_item(item));
            encode_path(ebml_w, path, ast_map::path_name(item.ident));
            encode_region_param(ecx, ebml_w, item);
        }
        encode_enum_variant_info(ecx, ebml_w, item.id, variants,
                                 path, index, tps);
      }
      item_class(tps, traits, items, ctor, m_dtor) {
        /* First, encode the fields and methods
           These come first because we need to write them to make
           the index, and the index needs to be in the item for the
           class itself */
        let idx = encode_info_for_class(ecx, ebml_w, item.id, path, tps,
                                          items, index);
        /* Encode the dtor */
        do option::iter(m_dtor) |dtor| {
            vec::push(*index, {val: dtor.node.id, pos: ebml_w.writer.tell()});
          encode_info_for_fn(ecx, ebml_w, dtor.node.id, @(*item.ident
                             + ~"_dtor"), path, if tps.len() > 0u {
                               some(ii_dtor(dtor, item.ident, tps,
                                            local_def(item.id))) }
                             else { none }, tps, ast_util::dtor_dec());
        }

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
        encode_region_param(ecx, ebml_w, item);
        for traits.each |t| {
           encode_trait_ref(ebml_w, ecx, t);
        }
        /* Encode the dtor */
        /* Encode id for dtor */
        do option::iter(m_dtor) |dtor| {
            do ebml_w.wr_tag(tag_item_dtor) {
                encode_def_id(ebml_w, local_def(dtor.node.id));
            }
        };

        /* Encode def_ids for each field and method
         for methods, write all the stuff get_trait_method
        needs to know*/
        let (fs,ms) = ast_util::split_class_items(items);
        for fs.each |f| {
           ebml_w.start_tag(tag_item_field);
           encode_visibility(ebml_w, f.vis);
           encode_name(ebml_w, f.ident);
           encode_def_id(ebml_w, local_def(f.id));
           ebml_w.end_tag();
        }
        for ms.each |m| {
           alt m.vis {
              private { /* do nothing */ }
              public {
                /* Write the info that's needed when viewing this class
                   as a trait */
                ebml_w.start_tag(tag_item_trait_method);
                encode_family(ebml_w, purity_fn_family(m.decl.purity));
                encode_name(ebml_w, m.ident);
                encode_type_param_bounds(ebml_w, ecx, m.tps);
                encode_type(ecx, ebml_w, node_id_to_type(tcx, m.id));
                encode_def_id(ebml_w, local_def(m.id));
                ebml_w.end_tag();
                /* Write the info that's needed when viewing this class
                   as an impl (just the method def_id) */
                ebml_w.start_tag(tag_item_impl_method);
                ebml_w.writer.write(str::bytes(def_to_str(local_def(m.id))));
                ebml_w.end_tag();
              }
           }
        }
        /* Each class has its own index -- encode it */
        let bkts = create_index(idx, hash_node_id);
        encode_index(ebml_w, bkts, write_int);
        ebml_w.end_tag();
      }
      item_impl(tps, ifce, _, methods) {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'i');
        encode_region_param(ecx, ebml_w, item);
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_name(ebml_w, item.ident);
        for methods.each |m| {
            ebml_w.start_tag(tag_item_impl_method);
            ebml_w.writer.write(str::bytes(def_to_str(local_def(m.id))));
            ebml_w.end_tag();
        }
        do option::iter(ifce) |t| {
           encode_trait_ref(ebml_w, ecx, t)
        };
        encode_path(ebml_w, path, ast_map::path_name(item.ident));
        ebml_w.end_tag();

        let impl_path = vec::append_one(path,
                                        ast_map::path_name(item.ident));
        for methods.each |m| {
            vec::push(*index, {val: m.id, pos: ebml_w.writer.tell()});
            encode_info_for_method(ecx, ebml_w, impl_path,
                                   should_inline(m.attrs), item.id, m,
                                   vec::append(tps, m.tps));
        }
      }
      item_trait(tps, ms) {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'I');
        encode_region_param(ecx, ebml_w, item);
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_name(ebml_w, item.ident);
        let mut i = 0u;
        for vec::each(*ty::trait_methods(tcx, local_def(item.id))) |mty| {
            alt ms[i] {
              required(ty_m) {
                ebml_w.start_tag(tag_item_trait_method);
                encode_name(ebml_w, mty.ident);
                encode_type_param_bounds(ebml_w, ecx, ty_m.tps);
                encode_type(ecx, ebml_w, ty::mk_fn(tcx, mty.fty));
                encode_family(ebml_w, purity_fn_family(mty.purity));
                ebml_w.end_tag();
              }
              provided(m) {
                encode_info_for_method(ecx, ebml_w, path,
                                       should_inline(m.attrs), item.id,
                                       m, m.tps);
              }
            }
            i += 1u;
        }
        encode_path(ebml_w, path, ast_map::path_name(item.ident));
        ebml_w.end_tag();
      }
      item_mac(*) { fail ~"item macros unimplemented" }
    }
}

fn encode_info_for_foreign_item(ecx: @encode_ctxt, ebml_w: ebml::writer,
                                nitem: @foreign_item,
                                index: @mut ~[entry<int>],
                                path: ast_map::path, abi: foreign_abi) {
    if !reachable(ecx, nitem.id) { ret; }
    vec::push(*index, {val: nitem.id, pos: ebml_w.writer.tell()});

    ebml_w.start_tag(tag_items_data_item);
    alt nitem.node {
      foreign_item_fn(fn_decl, tps) {
        encode_def_id(ebml_w, local_def(nitem.id));
        encode_family(ebml_w, purity_fn_family(fn_decl.purity));
        encode_type_param_bounds(ebml_w, ecx, tps);
        encode_type(ecx, ebml_w, node_id_to_type(ecx.tcx, nitem.id));
        if abi == foreign_abi_rust_intrinsic {
            ecx.encode_inlined_item(ecx, ebml_w, path,
                                    ii_foreign(nitem));
        } else {
            encode_symbol(ecx, ebml_w, nitem.id);
        }
        encode_path(ebml_w, path, ast_map::path_name(nitem.ident));
      }
    }
    ebml_w.end_tag();
}

fn encode_info_for_items(ecx: @encode_ctxt, ebml_w: ebml::writer,
                         crate: @crate) -> ~[entry<int>] {
    let index = @mut ~[];
    ebml_w.start_tag(tag_items_data);
    vec::push(*index, {val: crate_node_id, pos: ebml_w.writer.tell()});
    encode_info_for_mod(ecx, ebml_w, crate.node.module,
                        crate_node_id, ~[], @~"");
    visit::visit_crate(*crate, (), visit::mk_vt(@{
        visit_expr: |_e, _cx, _v| { },
        visit_item: |i, cx, v, copy ebml_w| {
            visit::visit_item(i, cx, v);
            alt check ecx.tcx.items.get(i.id) {
              ast_map::node_item(_, pt) {
                encode_info_for_item(ecx, ebml_w, i, index, *pt);
                /* encode ctor, then encode items */
                alt i.node {
                   item_class(tps, _, _, ctor, m_dtor) {
                   #debug("encoding info for ctor %s %d", *i.ident,
                          ctor.node.id);
                   vec::push(*index,
                             {val: ctor.node.id, pos: ebml_w.writer.tell()});
                   encode_info_for_fn(ecx, ebml_w, ctor.node.id, i.ident,
                      *pt, if tps.len() > 0u {
                             some(ii_ctor(ctor, i.ident, tps,
                                          local_def(i.id))) }
                      else { none }, tps, ctor.node.dec);
                  }
                  _ {}
                }
              }
            }
        },
        visit_foreign_item: |ni, cx, v, copy ebml_w| {
            visit::visit_foreign_item(ni, cx, v);
            alt check ecx.tcx.items.get(ni.id) {
              ast_map::node_foreign_item(_, abi, pt) {
                encode_info_for_foreign_item(ecx, ebml_w, ni,
                                             index, *pt, abi);
              }
            }
        }
        with *visit::default_visitor()
    }));
    ebml_w.end_tag();
    ret *index;
}


// Path and definition ID indexing

fn create_index<T: copy>(index: ~[entry<T>], hash_fn: fn@(T) -> uint) ->
   ~[@~[entry<T>]] {
    let mut buckets: ~[@mut ~[entry<T>]] = ~[];
    for uint::range(0u, 256u) |_i| { vec::push(buckets, @mut ~[]); };
    for index.each |elt| {
        let h = hash_fn(elt.val);
        vec::push(*buckets[h % 256u], elt);
    }

    let mut buckets_frozen = ~[];
    for buckets.each |bucket| {
        vec::push(buckets_frozen, @*bucket);
    }
    ret buckets_frozen;
}

fn encode_index<T>(ebml_w: ebml::writer, buckets: ~[@~[entry<T>]],
                   write_fn: fn(io::writer, T)) {
    let writer = ebml_w.writer;
    ebml_w.start_tag(tag_index);
    let mut bucket_locs: ~[uint] = ~[];
    ebml_w.start_tag(tag_index_buckets);
    for buckets.each |bucket| {
        vec::push(bucket_locs, ebml_w.writer.tell());
        ebml_w.start_tag(tag_index_buckets_bucket);
        for vec::each(*bucket) |elt| {
            ebml_w.start_tag(tag_index_buckets_bucket_elt);
            writer.write_be_uint(elt.pos, 4u);
            write_fn(writer, elt.val);
            ebml_w.end_tag();
        }
        ebml_w.end_tag();
    }
    ebml_w.end_tag();
    ebml_w.start_tag(tag_index_table);
    for bucket_locs.each |pos| { writer.write_be_uint(pos, 4u); }
    ebml_w.end_tag();
    ebml_w.end_tag();
}

fn write_str(writer: io::writer, &&s: ~str) { writer.write_str(s); }

fn write_int(writer: io::writer, &&n: int) {
    writer.write_be_uint(n as uint, 4u);
}

fn encode_meta_item(ebml_w: ebml::writer, mi: meta_item) {
    alt mi.node {
      meta_word(name) {
        ebml_w.start_tag(tag_meta_item_word);
        ebml_w.start_tag(tag_meta_item_name);
        ebml_w.writer.write(str::bytes(*name));
        ebml_w.end_tag();
        ebml_w.end_tag();
      }
      meta_name_value(name, value) {
        alt value.node {
          lit_str(value) {
            ebml_w.start_tag(tag_meta_item_name_value);
            ebml_w.start_tag(tag_meta_item_name);
            ebml_w.writer.write(str::bytes(*name));
            ebml_w.end_tag();
            ebml_w.start_tag(tag_meta_item_value);
            ebml_w.writer.write(str::bytes(*value));
            ebml_w.end_tag();
            ebml_w.end_tag();
          }
          _ {/* FIXME (#623): encode other variants */ }
        }
      }
      meta_list(name, items) {
        ebml_w.start_tag(tag_meta_item_list);
        ebml_w.start_tag(tag_meta_item_name);
        ebml_w.writer.write(str::bytes(*name));
        ebml_w.end_tag();
        for items.each |inner_item| {
            encode_meta_item(ebml_w, *inner_item);
        }
        ebml_w.end_tag();
      }
    }
}

fn encode_attributes(ebml_w: ebml::writer, attrs: ~[attribute]) {
    ebml_w.start_tag(tag_attributes);
    for attrs.each |attr| {
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
fn synthesize_crate_attrs(ecx: @encode_ctxt, crate: @crate) -> ~[attribute] {

    fn synthesize_link_attr(ecx: @encode_ctxt, items: ~[@meta_item]) ->
       attribute {

        assert (*ecx.link_meta.name != ~"");
        assert (*ecx.link_meta.vers != ~"");

        let name_item =
            attr::mk_name_value_item_str(@~"name", *ecx.link_meta.name);
        let vers_item =
            attr::mk_name_value_item_str(@~"vers", *ecx.link_meta.vers);

        let other_items =
            {
                let tmp = attr::remove_meta_items_by_name(items, @~"name");
                attr::remove_meta_items_by_name(tmp, @~"vers")
            };

        let meta_items = vec::append(~[name_item, vers_item], other_items);
        let link_item = attr::mk_list_item(@~"link", meta_items);

        ret attr::mk_attr(link_item);
    }

    let mut attrs: ~[attribute] = ~[];
    let mut found_link_attr = false;
    for crate.node.attrs.each |attr| {
        vec::push(
            attrs,
            if *attr::get_attr_name(attr) != ~"link" {
                attr
            } else {
                alt attr.node.value.node {
                  meta_list(n, l) {
                    found_link_attr = true;;
                    synthesize_link_attr(ecx, l)
                  }
                  _ { attr }
                }
            });
    }

    if !found_link_attr { vec::push(attrs, synthesize_link_attr(ecx, ~[])); }

    ret attrs;
}

fn encode_crate_deps(ebml_w: ebml::writer, cstore: cstore::cstore) {

    fn get_ordered_deps(cstore: cstore::cstore) -> ~[decoder::crate_dep] {
        type hashkv = @{key: crate_num, val: cstore::crate_metadata};
        type numdep = decoder::crate_dep;

        // Pull the cnums and name,vers,hash out of cstore
        let mut deps: ~[mut numdep] = ~[mut];
        do cstore::iter_crate_data(cstore) |key, val| {
            let dep = {cnum: key, name: @val.name,
                       vers: decoder::get_crate_vers(val.data),
                       hash: decoder::get_crate_hash(val.data)};
            vec::push(deps, dep);
        };

        // Sort by cnum
        fn lteq(kv1: numdep, kv2: numdep) -> bool { kv1.cnum <= kv2.cnum }
        std::sort::quick_sort(lteq, deps);

        // Sanity-check the crate numbers
        let mut expected_cnum = 1;
        for deps.each |n| {
            assert (n.cnum == expected_cnum);
            expected_cnum += 1;
        }

        // mut -> immutable hack for vec::map
        ret vec::slice(deps, 0u, vec::len(deps));
    }

    // We're just going to write a list of crate 'name-hash-version's, with
    // the assumption that they are numbered 1 to n.
    // FIXME (#2166): This is not nearly enough to support correct versioning
    // but is enough to get transitive crate dependencies working.
    ebml_w.start_tag(tag_crate_deps);
    for get_ordered_deps(cstore).each |dep| {
        encode_crate_dep(ebml_w, dep);
    }
    ebml_w.end_tag();
}

fn encode_crate_dep(ebml_w: ebml::writer, dep: decoder::crate_dep) {
    ebml_w.start_tag(tag_crate_dep);
    ebml_w.start_tag(tag_crate_dep_name);
    ebml_w.writer.write(str::bytes(*dep.name));
    ebml_w.end_tag();
    ebml_w.start_tag(tag_crate_dep_vers);
    ebml_w.writer.write(str::bytes(*dep.vers));
    ebml_w.end_tag();
    ebml_w.start_tag(tag_crate_dep_hash);
    ebml_w.writer.write(str::bytes(*dep.hash));
    ebml_w.end_tag();
    ebml_w.end_tag();
}

fn encode_hash(ebml_w: ebml::writer, hash: ~str) {
    ebml_w.start_tag(tag_crate_hash);
    ebml_w.writer.write(str::bytes(hash));
    ebml_w.end_tag();
}

fn encode_metadata(parms: encode_parms, crate: @crate) -> ~[u8] {
    let ecx: @encode_ctxt = @encode_ctxt({
        diag: parms.diag,
        tcx: parms.tcx,
        reachable: parms.reachable,
        reexports: parms.reexports,
        impl_map: parms.impl_map,
        item_symbols: parms.item_symbols,
        discrim_symbols: parms.discrim_symbols,
        link_meta: parms.link_meta,
        cstore: parms.cstore,
        encode_inlined_item: parms.encode_inlined_item,
        type_abbrevs: ty::new_ty_hash()
     });

    let buf = io::mem_buffer();
    let buf_w = io::mem_buffer_writer(buf);
    let ebml_w = ebml::writer(buf_w);

    encode_hash(ebml_w, ecx.link_meta.extras_hash);

    let crate_attrs = synthesize_crate_attrs(ecx, crate);
    encode_attributes(ebml_w, crate_attrs);

    encode_crate_deps(ebml_w, ecx.cstore);

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
    buf_w.write(&[0u8, 0u8, 0u8, 0u8]);
    io::mem_buffer_buf(buf)
}

// Get the encoded string for a type
fn encoded_ty(tcx: ty::ctxt, t: ty::t) -> ~str {
    let cx = @{diag: tcx.diag,
               ds: def_to_str,
               tcx: tcx,
               reachable: |_id| false,
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
