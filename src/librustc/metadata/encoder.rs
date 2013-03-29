// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// Metadata encoding

use core::prelude::*;

use metadata::common::*;
use metadata::cstore;
use metadata::decoder;
use metadata::tyencode;
use middle::trans::reachable;
use middle::ty::node_id_to_type;
use middle::ty;
use middle;
use util::ppaux::ty_to_str;

use core::flate;
use core::hash::HashUtil;
use core::hashmap::linear::LinearMap;
use core::int;
use core::io::{Writer, WriterUtil};
use core::io;
use core::str;
use core::to_bytes::IterBytes;
use core::uint;
use core::vec;
use std::serialize::Encodable;
use std;
use syntax::ast::*;
use syntax::ast;
use syntax::ast_map;
use syntax::ast_util::*;
use syntax::attr;
use syntax::diagnostic::span_handler;
use syntax::parse::token::special_idents;
use syntax::{ast_util, visit};
use syntax::opt_vec::OptVec;
use syntax::opt_vec;
use syntax;
use writer = std::ebml::writer;

// used by astencode:
type abbrev_map = @mut LinearMap<ty::t, tyencode::ty_abbrev>;

pub type encode_inlined_item = @fn(ecx: @EncodeContext,
                                   ebml_w: writer::Encoder,
                                   path: &[ast_map::path_elt],
                                   ii: ast::inlined_item);

pub struct EncodeParams {
    diag: @span_handler,
    tcx: ty::ctxt,
    reachable: reachable::map,
    reexports2: middle::resolve::ExportMap2,
    item_symbols: @mut LinearMap<ast::node_id, ~str>,
    discrim_symbols: @mut LinearMap<ast::node_id, ~str>,
    link_meta: LinkMeta,
    cstore: @mut cstore::CStore,
    encode_inlined_item: encode_inlined_item
}

struct Stats {
    inline_bytes: uint,
    attr_bytes: uint,
    dep_bytes: uint,
    lang_item_bytes: uint,
    link_args_bytes: uint,
    item_bytes: uint,
    index_bytes: uint,
    zero_bytes: uint,
    total_bytes: uint,

    n_inlines: uint
}

pub struct EncodeContext {
    diag: @span_handler,
    tcx: ty::ctxt,
    stats: @mut Stats,
    reachable: reachable::map,
    reexports2: middle::resolve::ExportMap2,
    item_symbols: @mut LinearMap<ast::node_id, ~str>,
    discrim_symbols: @mut LinearMap<ast::node_id, ~str>,
    link_meta: LinkMeta,
    cstore: @mut cstore::CStore,
    encode_inlined_item: encode_inlined_item,
    type_abbrevs: abbrev_map
}

pub fn reachable(ecx: @EncodeContext, id: node_id) -> bool {
    ecx.reachable.contains(&id)
}

fn encode_name(ecx: @EncodeContext, ebml_w: writer::Encoder, name: ident) {
    ebml_w.wr_tagged_str(tag_paths_data_name, *ecx.tcx.sess.str_of(name));
}

fn encode_impl_type_basename(ecx: @EncodeContext, ebml_w: writer::Encoder,
                             name: ident) {
    ebml_w.wr_tagged_str(tag_item_impl_type_basename,
                         *ecx.tcx.sess.str_of(name));
}

pub fn encode_def_id(ebml_w: writer::Encoder, id: def_id) {
    ebml_w.wr_tagged_str(tag_def_id, def_to_str(id));
}

fn encode_region_param(ecx: @EncodeContext, ebml_w: writer::Encoder,
                       it: @ast::item) {
    let opt_rp = ecx.tcx.region_paramd_items.find(&it.id);
    for opt_rp.each |rp| {
        do ebml_w.wr_tag(tag_region_param) {
            (*rp).encode(&ebml_w);
        }
    }
}

fn encode_mutability(ebml_w: writer::Encoder, mt: struct_mutability) {
    do ebml_w.wr_tag(tag_struct_mut) {
        let val = match mt {
          struct_immutable => 'a',
          struct_mutable => 'm'
        };
        ebml_w.writer.write(&[val as u8]);
    }
}

struct entry<T> {
    val: T,
    pos: uint
}

fn add_to_index(ecx: @EncodeContext, ebml_w: writer::Encoder, path: &[ident],
                index: &mut ~[entry<~str>], name: ident) {
    let mut full_path = ~[];
    full_path.push_all(path);
    full_path.push(name);
    index.push(
        entry {
            val: ast_util::path_name_i(full_path,
                                       ecx.tcx.sess.parse_sess.interner),
            pos: ebml_w.writer.tell()
        });
}

fn encode_trait_ref(ebml_w: writer::Encoder, ecx: @EncodeContext,
                    t: @trait_ref) {
    ebml_w.start_tag(tag_impl_trait);
    encode_type(ecx, ebml_w, node_id_to_type(ecx.tcx, t.ref_id));
    ebml_w.end_tag();
}


// Item info table encoding
fn encode_family(ebml_w: writer::Encoder, c: char) {
    ebml_w.start_tag(tag_items_data_item_family);
    ebml_w.writer.write(&[c as u8]);
    ebml_w.end_tag();
}

pub fn def_to_str(did: def_id) -> ~str { fmt!("%d:%d", did.crate, did.node) }

fn encode_ty_type_param_bounds(ebml_w: writer::Encoder, ecx: @EncodeContext,
                               params: @~[ty::param_bounds]) {
    let ty_str_ctxt = @tyencode::ctxt {
        diag: ecx.diag,
        ds: def_to_str,
        tcx: ecx.tcx,
        reachable: |a| reachable(ecx, a),
        abbrevs: tyencode::ac_use_abbrevs(ecx.type_abbrevs)};
    for params.each |param| {
        ebml_w.start_tag(tag_items_data_item_ty_param_bounds);
        tyencode::enc_bounds(ebml_w.writer, ty_str_ctxt, *param);
        ebml_w.end_tag();
    }
}

fn encode_type_param_bounds(ebml_w: writer::Encoder,
                            ecx: @EncodeContext,
                            params: &OptVec<TyParam>) {
    let ty_param_bounds =
        @params.map_to_vec(|param| *ecx.tcx.ty_param_bounds.get(&param.id));
    encode_ty_type_param_bounds(ebml_w, ecx, ty_param_bounds);
}


fn encode_variant_id(ebml_w: writer::Encoder, vid: def_id) {
    ebml_w.start_tag(tag_items_data_item_variant);
    ebml_w.writer.write(str::to_bytes(def_to_str(vid)));
    ebml_w.end_tag();
}

pub fn write_type(ecx: @EncodeContext, ebml_w: writer::Encoder, typ: ty::t) {
    let ty_str_ctxt = @tyencode::ctxt {
        diag: ecx.diag,
        ds: def_to_str,
        tcx: ecx.tcx,
        reachable: |a| reachable(ecx, a),
        abbrevs: tyencode::ac_use_abbrevs(ecx.type_abbrevs)};
    tyencode::enc_ty(ebml_w.writer, ty_str_ctxt, typ);
}

pub fn write_vstore(ecx: @EncodeContext, ebml_w: writer::Encoder,
                    vstore: ty::vstore) {
    let ty_str_ctxt = @tyencode::ctxt {
        diag: ecx.diag,
        ds: def_to_str,
        tcx: ecx.tcx,
        reachable: |a| reachable(ecx, a),
        abbrevs: tyencode::ac_use_abbrevs(ecx.type_abbrevs)};
    tyencode::enc_vstore(ebml_w.writer, ty_str_ctxt, vstore);
}

fn encode_type(ecx: @EncodeContext, ebml_w: writer::Encoder, typ: ty::t) {
    ebml_w.start_tag(tag_items_data_item_type);
    write_type(ecx, ebml_w, typ);
    ebml_w.end_tag();
}

fn encode_symbol(ecx: @EncodeContext, ebml_w: writer::Encoder, id: node_id) {
    ebml_w.start_tag(tag_items_data_item_symbol);
    match ecx.item_symbols.find(&id) {
        Some(x) => {
            debug!("encode_symbol(id=%?, str=%s)", id, *x);
            ebml_w.writer.write(str::to_bytes(*x));
        }
        None => {
            ecx.diag.handler().bug(
                fmt!("encode_symbol: id not found %d", id));
        }
    }
    ebml_w.end_tag();
}

fn encode_discriminant(ecx: @EncodeContext, ebml_w: writer::Encoder,
                       id: node_id) {
    ebml_w.start_tag(tag_items_data_item_symbol);
    ebml_w.writer.write(str::to_bytes(*ecx.discrim_symbols.get(&id)));
    ebml_w.end_tag();
}

fn encode_disr_val(_ecx: @EncodeContext, ebml_w: writer::Encoder,
                   disr_val: int) {
    ebml_w.start_tag(tag_disr_val);
    ebml_w.writer.write(str::to_bytes(int::to_str(disr_val)));
    ebml_w.end_tag();
}

fn encode_parent_item(ebml_w: writer::Encoder, id: def_id) {
    ebml_w.start_tag(tag_items_data_parent_item);
    ebml_w.writer.write(str::to_bytes(def_to_str(id)));
    ebml_w.end_tag();
}

fn encode_enum_variant_info(ecx: @EncodeContext, ebml_w: writer::Encoder,
                            id: node_id, variants: &[variant],
                            path: &[ast_map::path_elt],
                            index: @mut ~[entry<int>],
                            generics: &ast::Generics) {
    debug!("encode_enum_variant_info(id=%?)", id);

    let mut disr_val = 0;
    let mut i = 0;
    let vi = ty::enum_variants(ecx.tcx,
                               ast::def_id { crate: local_crate, node: id });
    for variants.each |variant| {
        index.push(entry {val: variant.node.id, pos: ebml_w.writer.tell()});
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(variant.node.id));
        encode_family(ebml_w, 'v');
        encode_name(ecx, ebml_w, variant.node.name);
        encode_parent_item(ebml_w, local_def(id));
        encode_type(ecx, ebml_w,
                    node_id_to_type(ecx.tcx, variant.node.id));
        match variant.node.kind {
            ast::tuple_variant_kind(ref args)
                    if args.len() > 0 && generics.ty_params.len() == 0 => {
                encode_symbol(ecx, ebml_w, variant.node.id);
            }
            ast::tuple_variant_kind(_) | ast::struct_variant_kind(_) => {}
        }
        encode_discriminant(ecx, ebml_w, variant.node.id);
        if vi[i].disr_val != disr_val {
            encode_disr_val(ecx, ebml_w, vi[i].disr_val);
            disr_val = vi[i].disr_val;
        }
        encode_type_param_bounds(ebml_w, ecx, &generics.ty_params);
        encode_path(ecx, ebml_w, path,
                    ast_map::path_name(variant.node.name));
        ebml_w.end_tag();
        disr_val += 1;
        i += 1;
    }
}

fn encode_path(ecx: @EncodeContext, ebml_w: writer::Encoder,
               path: &[ast_map::path_elt], name: ast_map::path_elt) {
    fn encode_path_elt(ecx: @EncodeContext, ebml_w: writer::Encoder,
                       elt: ast_map::path_elt) {
        let (tag, name) = match elt {
          ast_map::path_mod(name) => (tag_path_elt_mod, name),
          ast_map::path_name(name) => (tag_path_elt_name, name)
        };

        ebml_w.wr_tagged_str(tag, *ecx.tcx.sess.str_of(name));
    }

    do ebml_w.wr_tag(tag_path) {
        ebml_w.wr_tagged_u32(tag_path_len, (path.len() + 1) as u32);
        for path.each |pe| {
            encode_path_elt(ecx, ebml_w, *pe);
        }
        encode_path_elt(ecx, ebml_w, name);
    }
}

fn encode_info_for_mod(ecx: @EncodeContext, ebml_w: writer::Encoder,
                       md: &_mod, id: node_id, path: &[ast_map::path_elt],
                       name: ident) {
    ebml_w.start_tag(tag_items_data_item);
    encode_def_id(ebml_w, local_def(id));
    encode_family(ebml_w, 'm');
    encode_name(ecx, ebml_w, name);
    debug!("(encoding info for module) encoding info for module ID %d", id);

    // Encode info about all the module children.
    for md.items.each |item| {
        match item.node {
            item_impl(*) => {
                let (ident, did) = (item.ident, item.id);
                debug!("(encoding info for module) ... encoding impl %s \
                        (%?/%?)",
                        *ecx.tcx.sess.str_of(ident),
                        did,
                        ast_map::node_id_to_str(ecx.tcx.items, did, ecx.tcx
                                                .sess.parse_sess.interner));

                ebml_w.start_tag(tag_mod_impl);
                ebml_w.wr_str(def_to_str(local_def(did)));
                ebml_w.end_tag();
            }
            _ => {} // FIXME #4573: Encode these too.
        }
    }

    encode_path(ecx, ebml_w, path, ast_map::path_mod(name));

    // Encode the reexports of this module.
    debug!("(encoding info for module) encoding reexports for %d", id);
    match ecx.reexports2.find(&id) {
        Some(ref exports) => {
            debug!("(encoding info for module) found reexports for %d", id);
            for exports.each |exp| {
                debug!("(encoding info for module) reexport '%s' for %d",
                       *exp.name, id);
                ebml_w.start_tag(tag_items_data_item_reexport);
                ebml_w.start_tag(tag_items_data_item_reexport_def_id);
                ebml_w.wr_str(def_to_str(exp.def_id));
                ebml_w.end_tag();
                ebml_w.start_tag(tag_items_data_item_reexport_name);
                ebml_w.wr_str(*exp.name);
                ebml_w.end_tag();
                ebml_w.end_tag();
            }
        }
        None => {
            debug!("(encoding info for module) found no reexports for %d",
                   id);
        }
    }

    ebml_w.end_tag();
}

fn encode_struct_field_family(ebml_w: writer::Encoder,
                              visibility: visibility) {
    encode_family(ebml_w, match visibility {
        public => 'g',
        private => 'j',
        inherited => 'N'
    });
}

fn encode_visibility(ebml_w: writer::Encoder, visibility: visibility) {
    ebml_w.start_tag(tag_items_data_item_visibility);
    let ch = match visibility {
        public => 'y',
        private => 'n',
        inherited => 'i',
    };
    ebml_w.wr_str(str::from_char(ch));
    ebml_w.end_tag();
}

fn encode_self_type(ebml_w: writer::Encoder, self_type: ast::self_ty_) {
    ebml_w.start_tag(tag_item_trait_method_self_ty);

    // Encode the base self type.
    match self_type {
        sty_static => {
            ebml_w.writer.write(&[ 's' as u8 ]);
        }
        sty_value => {
            ebml_w.writer.write(&[ 'v' as u8 ]);
        }
        sty_region(_, m) => {
            // FIXME(#4846) encode custom lifetime
            ebml_w.writer.write(&[ '&' as u8 ]);
            encode_mutability(ebml_w, m);
        }
        sty_box(m) => {
            ebml_w.writer.write(&[ '@' as u8 ]);
            encode_mutability(ebml_w, m);
        }
        sty_uniq(m) => {
            ebml_w.writer.write(&[ '~' as u8 ]);
            encode_mutability(ebml_w, m);
        }
    }

    ebml_w.end_tag();

    fn encode_mutability(ebml_w: writer::Encoder,
                         m: ast::mutability) {
        match m {
            m_imm => {
                ebml_w.writer.write(&[ 'i' as u8 ]);
            }
            m_mutbl => {
                ebml_w.writer.write(&[ 'm' as u8 ]);
            }
            m_const => {
                ebml_w.writer.write(&[ 'c' as u8 ]);
            }
        }
    }
}

fn encode_method_sort(ebml_w: writer::Encoder, sort: char) {
    ebml_w.start_tag(tag_item_trait_method_sort);
    ebml_w.writer.write(&[ sort as u8 ]);
    ebml_w.end_tag();
}

/* Returns an index of items in this class */
fn encode_info_for_struct(ecx: @EncodeContext, ebml_w: writer::Encoder,
                         path: &[ast_map::path_elt],
                         fields: &[@struct_field],
                         global_index: @mut~[entry<int>]) -> ~[entry<int>] {
    /* Each class has its own index, since different classes
       may have fields with the same name */
    let index = @mut ~[];
    let tcx = ecx.tcx;
     /* We encode both private and public fields -- need to include
        private fields to get the offsets right */
    for fields.each |field| {
        let (nm, mt, vis) = match field.node.kind {
            named_field(nm, mt, vis) => (nm, mt, vis),
            unnamed_field => (
                special_idents::unnamed_field,
                struct_immutable,
                inherited
            )
        };

        let id = field.node.id;
        index.push(entry {val: id, pos: ebml_w.writer.tell()});
        global_index.push(entry {val: id, pos: ebml_w.writer.tell()});
        ebml_w.start_tag(tag_items_data_item);
        debug!("encode_info_for_struct: doing %s %d",
               *tcx.sess.str_of(nm), id);
        encode_struct_field_family(ebml_w, vis);
        encode_name(ecx, ebml_w, nm);
        encode_path(ecx, ebml_w, path, ast_map::path_name(nm));
        encode_type(ecx, ebml_w, node_id_to_type(tcx, id));
        encode_mutability(ebml_w, mt);
        encode_def_id(ebml_w, local_def(id));
        ebml_w.end_tag();
    }
    /*bad*/copy *index
}

// This is for encoding info for ctors and dtors
fn encode_info_for_ctor(ecx: @EncodeContext,
                        ebml_w: writer::Encoder,
                        id: node_id,
                        ident: ident,
                        path: &[ast_map::path_elt],
                        item: Option<inlined_item>,
                        generics: &ast::Generics) {
        ebml_w.start_tag(tag_items_data_item);
        encode_name(ecx, ebml_w, ident);
        encode_def_id(ebml_w, local_def(id));
        encode_family(ebml_w, purity_fn_family(ast::impure_fn));
        encode_type_param_bounds(ebml_w, ecx, &generics.ty_params);
        let its_ty = node_id_to_type(ecx.tcx, id);
        debug!("fn name = %s ty = %s its node id = %d",
               *ecx.tcx.sess.str_of(ident),
               ty_to_str(ecx.tcx, its_ty), id);
        encode_type(ecx, ebml_w, its_ty);
        encode_path(ecx, ebml_w, path, ast_map::path_name(ident));
        match item {
           Some(ref it) => {
             (ecx.encode_inlined_item)(ecx, ebml_w, path, (*it));
           }
           None => {
             encode_symbol(ecx, ebml_w, id);
           }
        }
        ebml_w.end_tag();
}

fn encode_info_for_struct_ctor(ecx: @EncodeContext,
                               ebml_w: writer::Encoder,
                               path: &[ast_map::path_elt],
                               name: ast::ident,
                               ctor_id: node_id,
                               index: @mut ~[entry<int>]) {
    index.push(entry { val: ctor_id, pos: ebml_w.writer.tell() });

    ebml_w.start_tag(tag_items_data_item);
    encode_def_id(ebml_w, local_def(ctor_id));
    encode_family(ebml_w, 'f');
    encode_name(ecx, ebml_w, name);
    encode_type(ecx, ebml_w, node_id_to_type(ecx.tcx, ctor_id));
    encode_path(ecx, ebml_w, path, ast_map::path_name(name));

    if ecx.item_symbols.contains_key(&ctor_id) {
        encode_symbol(ecx, ebml_w, ctor_id);
    }

    ebml_w.end_tag();
}

fn encode_info_for_method(ecx: @EncodeContext,
                          ebml_w: writer::Encoder,
                          impl_path: &[ast_map::path_elt],
                          should_inline: bool,
                          parent_id: node_id,
                          m: @method,
                          parent_visibility: ast::visibility,
                          owner_generics: &ast::Generics,
                          method_generics: &ast::Generics) {
    debug!("encode_info_for_method: %d %s %u %u", m.id,
           *ecx.tcx.sess.str_of(m.ident),
           owner_generics.ty_params.len(),
           method_generics.ty_params.len());
    ebml_w.start_tag(tag_items_data_item);
    encode_def_id(ebml_w, local_def(m.id));

    match m.self_ty.node {
        ast::sty_static => {
            encode_family(ebml_w, purity_static_method_family(m.purity));
        }
        _ => encode_family(ebml_w, purity_fn_family(m.purity))
    }

    let mut combined_ty_params = opt_vec::Empty;
    combined_ty_params.push_all(&owner_generics.ty_params);
    combined_ty_params.push_all(&method_generics.ty_params);
    let len = combined_ty_params.len();
    encode_type_param_bounds(ebml_w, ecx, &combined_ty_params);

    encode_type(ecx, ebml_w, node_id_to_type(ecx.tcx, m.id));
    encode_name(ecx, ebml_w, m.ident);
    encode_path(ecx, ebml_w, impl_path, ast_map::path_name(m.ident));
    encode_self_type(ebml_w, m.self_ty.node);

    // Combine parent visibility and this visibility.
    let visibility = match m.vis {
        ast::inherited => parent_visibility,
        vis => vis,
    };
    encode_visibility(ebml_w, visibility);

    if len > 0u || should_inline {
        (ecx.encode_inlined_item)(
           ecx, ebml_w, impl_path,
           ii_method(local_def(parent_id), m));
    } else {
        encode_symbol(ecx, ebml_w, m.id);
    }
    ebml_w.end_tag();
}

fn purity_fn_family(p: purity) -> char {
    match p {
      unsafe_fn => 'u',
      pure_fn => 'p',
      impure_fn => 'f',
      extern_fn => 'e'
    }
}

fn purity_static_method_family(p: purity) -> char {
    match p {
      unsafe_fn => 'U',
      pure_fn => 'P',
      impure_fn => 'F',
      _ => fail!(~"extern fn can't be static")
    }
}


fn should_inline(attrs: &[attribute]) -> bool {
    match attr::find_inline_attr(attrs) {
        attr::ia_none | attr::ia_never  => false,
        attr::ia_hint | attr::ia_always => true
    }
}


fn encode_info_for_item(ecx: @EncodeContext, ebml_w: writer::Encoder,
                        item: @item, index: @mut ~[entry<int>],
                        path: &[ast_map::path_elt]) {

    let tcx = ecx.tcx;
    let must_write =
        match item.node {
          item_enum(_, _) | item_impl(*) | item_trait(*) | item_struct(*) |
          item_mod(*) | item_foreign_mod(*) | item_const(*) => true,
          _ => false
        };
    if !must_write && !reachable(ecx, item.id) { return; }

    fn add_to_index_(item: @item, ebml_w: writer::Encoder,
                     index: @mut ~[entry<int>]) {
        index.push(entry { val: item.id, pos: ebml_w.writer.tell() });
    }
    let add_to_index: &fn() = || add_to_index_(item, ebml_w, index);

    debug!("encoding info for item at %s",
           ecx.tcx.sess.codemap.span_to_str(item.span));

    match item.node {
      item_const(_, _) => {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'c');
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_symbol(ecx, ebml_w, item.id);
        encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
        (ecx.encode_inlined_item)(ecx, ebml_w, path, ii_item(item));
        ebml_w.end_tag();
      }
      item_fn(_, purity, ref generics, _) => {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, purity_fn_family(purity));
        let tps_len = generics.ty_params.len();
        encode_type_param_bounds(ebml_w, ecx, &generics.ty_params);
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
        encode_attributes(ebml_w, item.attrs);
        if tps_len > 0u || should_inline(item.attrs) {
            (ecx.encode_inlined_item)(ecx, ebml_w, path, ii_item(item));
        } else {
            encode_symbol(ecx, ebml_w, item.id);
        }
        ebml_w.end_tag();
      }
      item_mod(ref m) => {
        add_to_index();
        encode_info_for_mod(ecx, ebml_w, m, item.id, path, item.ident);
      }
      item_foreign_mod(_) => {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'n');
        encode_name(ecx, ebml_w, item.ident);
        encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
        ebml_w.end_tag();
      }
      item_ty(_, ref generics) => {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'y');
        encode_type_param_bounds(ebml_w, ecx, &generics.ty_params);
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_name(ecx, ebml_w, item.ident);
        encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
        encode_region_param(ecx, ebml_w, item);
        ebml_w.end_tag();
      }
      item_enum(ref enum_definition, ref generics) => {
        add_to_index();
        do ebml_w.wr_tag(tag_items_data_item) {
            encode_def_id(ebml_w, local_def(item.id));
            encode_family(ebml_w, 't');
            encode_type_param_bounds(ebml_w, ecx, &generics.ty_params);
            encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
            encode_name(ecx, ebml_w, item.ident);
            for (*enum_definition).variants.each |v| {
                encode_variant_id(ebml_w, local_def(v.node.id));
            }
            (ecx.encode_inlined_item)(ecx, ebml_w, path, ii_item(item));
            encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
            encode_region_param(ecx, ebml_w, item);
        }
        encode_enum_variant_info(ecx,
                                 ebml_w,
                                 item.id,
                                 (*enum_definition).variants,
                                 path,
                                 index,
                                 generics);
      }
      item_struct(struct_def, ref generics) => {
        /* First, encode the fields
           These come first because we need to write them to make
           the index, and the index needs to be in the item for the
           class itself */
        let idx = encode_info_for_struct(ecx, ebml_w, path,
                                         struct_def.fields, index);
        /* Encode the dtor */
        for struct_def.dtor.each |dtor| {
            index.push(entry {val: dtor.node.id, pos: ebml_w.writer.tell()});
          encode_info_for_ctor(ecx,
                               ebml_w,
                               dtor.node.id,
                               ecx.tcx.sess.ident_of(
                                   *ecx.tcx.sess.str_of(item.ident) +
                                   ~"_dtor"),
                               path,
                               if generics.ty_params.len() > 0u {
                                   Some(ii_dtor(copy *dtor,
                                                item.ident,
                                                copy *generics,
                                                local_def(item.id))) }
                               else {
                                   None
                               },
                               generics);
        }

        /* Index the class*/
        add_to_index();

        /* Now, make an item for the class itself */
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'S');
        encode_type_param_bounds(ebml_w, ecx, &generics.ty_params);
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));

        // If this is a tuple- or enum-like struct, encode the type of the
        // constructor.
        if struct_def.fields.len() > 0 &&
                struct_def.fields[0].node.kind == ast::unnamed_field {
            let ctor_id = match struct_def.ctor_id {
                Some(ctor_id) => ctor_id,
                None => ecx.tcx.sess.bug(~"struct def didn't have ctor id"),
            };

            encode_info_for_struct_ctor(ecx,
                                        ebml_w,
                                        path,
                                        item.ident,
                                        ctor_id,
                                        index);
        }

        encode_name(ecx, ebml_w, item.ident);
        encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
        encode_region_param(ecx, ebml_w, item);
        /* Encode the dtor */
        /* Encode id for dtor */
        for struct_def.dtor.each |dtor| {
            do ebml_w.wr_tag(tag_item_dtor) {
                encode_def_id(ebml_w, local_def(dtor.node.id));
            }
        };

        /* Encode def_ids for each field and method
         for methods, write all the stuff get_trait_method
        needs to know*/
        for struct_def.fields.each |f| {
            match f.node.kind {
                named_field(ident, _, vis) => {
                   ebml_w.start_tag(tag_item_field);
                   encode_struct_field_family(ebml_w, vis);
                   encode_name(ecx, ebml_w, ident);
                   encode_def_id(ebml_w, local_def(f.node.id));
                   ebml_w.end_tag();
                }
                unnamed_field => {
                    ebml_w.start_tag(tag_item_unnamed_field);
                    encode_def_id(ebml_w, local_def(f.node.id));
                    ebml_w.end_tag();
                }
            }
        }

        /* Each class has its own index -- encode it */
        let bkts = create_index(idx);
        encode_index(ebml_w, bkts, write_int);
        ebml_w.end_tag();
      }
      item_impl(ref generics, opt_trait, ty, ref methods) => {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'i');
        encode_region_param(ecx, ebml_w, item);
        encode_type_param_bounds(ebml_w, ecx, &generics.ty_params);
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_name(ecx, ebml_w, item.ident);
        encode_attributes(ebml_w, item.attrs);
        match ty.node {
            ast::ty_path(path, _) if path.idents.len() == 1 => {
                encode_impl_type_basename(ecx, ebml_w,
                                          ast_util::path_to_ident(path));
            }
            _ => {}
        }
        for methods.each |m| {
            ebml_w.start_tag(tag_item_impl_method);
            let method_def_id = local_def(m.id);
            ebml_w.writer.write(str::to_bytes(def_to_str(method_def_id)));
            ebml_w.end_tag();
        }
        for opt_trait.each |associated_trait| {
           encode_trait_ref(ebml_w, ecx, *associated_trait);
        }
        encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
        ebml_w.end_tag();

        // >:-<
        let mut impl_path = vec::append(~[], path);
        impl_path += ~[ast_map::path_name(item.ident)];

        // If there is a trait reference, treat the methods as always public.
        // This is to work around some incorrect behavior in privacy checking:
        // when the method belongs to a trait, it should acquire the privacy
        // from the trait, not the impl. Forcing the visibility to be public
        // makes things sorta work.
        let parent_visibility = if opt_trait.is_some() {
            ast::public
        } else {
            item.vis
        };

        for methods.each |m| {
            index.push(entry {val: m.id, pos: ebml_w.writer.tell()});
            encode_info_for_method(ecx,
                                   ebml_w,
                                   impl_path,
                                   should_inline(m.attrs),
                                   item.id,
                                   *m,
                                   parent_visibility,
                                   generics,
                                   &m.generics);
        }
      }
      item_trait(ref generics, ref traits, ref ms) => {
        let mut provided_methods = ~[];

        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, local_def(item.id));
        encode_family(ebml_w, 'I');
        encode_region_param(ecx, ebml_w, item);
        encode_type_param_bounds(ebml_w, ecx, &generics.ty_params);
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_name(ecx, ebml_w, item.ident);
        encode_attributes(ebml_w, item.attrs);
        let mut i = 0u;
        for vec::each(*ty::trait_methods(tcx, local_def(item.id))) |mty| {
            match (*ms)[i] {
              required(ref ty_m) => {
                ebml_w.start_tag(tag_item_trait_method);
                encode_def_id(ebml_w, local_def((*ty_m).id));
                encode_name(ecx, ebml_w, mty.ident);
                encode_type_param_bounds(ebml_w, ecx,
                                         &ty_m.generics.ty_params);
                encode_type(ecx, ebml_w,
                            ty::mk_bare_fn(tcx, copy mty.fty));
                encode_family(ebml_w, purity_fn_family(mty.fty.purity));
                encode_self_type(ebml_w, mty.self_ty);
                encode_method_sort(ebml_w, 'r');
                encode_visibility(ebml_w, ast::public);
                ebml_w.end_tag();
              }
              provided(m) => {
                provided_methods.push(m);

                ebml_w.start_tag(tag_item_trait_method);
                encode_def_id(ebml_w, local_def(m.id));
                encode_name(ecx, ebml_w, mty.ident);
                encode_type_param_bounds(ebml_w, ecx,
                                         &m.generics.ty_params);
                encode_type(ecx, ebml_w,
                            ty::mk_bare_fn(tcx, copy mty.fty));
                encode_family(ebml_w, purity_fn_family(mty.fty.purity));
                encode_self_type(ebml_w, mty.self_ty);
                encode_method_sort(ebml_w, 'p');
                encode_visibility(ebml_w, m.vis);
                ebml_w.end_tag();
              }
            }
            i += 1;
        }
        encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
        for traits.each |associated_trait| {
           encode_trait_ref(ebml_w, ecx, *associated_trait)
        }

        ebml_w.end_tag();

        // Now, output all of the static methods as items.  Note that for the
        // method info, we output static methods with type signatures as
        // written. Here, we output the *real* type signatures. I feel like
        // maybe we should only ever handle the real type signatures.
        for ms.each |m| {
            let ty_m = ast_util::trait_method_to_ty_method(m);
            if ty_m.self_ty.node != ast::sty_static { loop; }

            index.push(entry { val: ty_m.id, pos: ebml_w.writer.tell() });

            ebml_w.start_tag(tag_items_data_item);
            encode_def_id(ebml_w, local_def(ty_m.id));
            encode_parent_item(ebml_w, local_def(item.id));
            encode_name(ecx, ebml_w, ty_m.ident);
            encode_family(ebml_w,
                          purity_static_method_family(ty_m.purity));
            let polyty = ecx.tcx.tcache.get(&local_def(ty_m.id));
            encode_ty_type_param_bounds(ebml_w, ecx, polyty.bounds);
            encode_type(ecx, ebml_w, polyty.ty);
            let mut m_path = vec::append(~[], path); // :-(
            m_path += [ast_map::path_name(item.ident)];
            encode_path(ecx, ebml_w, m_path, ast_map::path_name(ty_m.ident));

            // For now, use the item visibility until trait methods can have
            // real visibility in the AST.
            encode_visibility(ebml_w, item.vis);

            ebml_w.end_tag();
        }

        // Finally, output all the provided methods as items.
        for provided_methods.each |m| {
            index.push(entry { val: m.id, pos: ebml_w.writer.tell() });

            // We do not concatenate the generics of the owning impl and that
            // of provided methods.  I am not sure why this is. -ndm
            let owner_generics = ast_util::empty_generics();

            encode_info_for_method(ecx,
                                   ebml_w,
                                   /*bad*/copy path,
                                   true,
                                   item.id,
                                   *m,
                                   item.vis,
                                   &owner_generics,
                                   &m.generics);
        }
      }
      item_mac(*) => fail!(~"item macros unimplemented")
    }
}

fn encode_info_for_foreign_item(ecx: @EncodeContext,
                                ebml_w: writer::Encoder,
                                nitem: @foreign_item,
                                index: @mut ~[entry<int>],
                                +path: ast_map::path,
                                abi: foreign_abi) {
    if !reachable(ecx, nitem.id) { return; }
    index.push(entry { val: nitem.id, pos: ebml_w.writer.tell() });

    ebml_w.start_tag(tag_items_data_item);
    match nitem.node {
      foreign_item_fn(_, purity, ref generics) => {
        encode_def_id(ebml_w, local_def(nitem.id));
        encode_family(ebml_w, purity_fn_family(purity));
        encode_type_param_bounds(ebml_w, ecx, &generics.ty_params);
        encode_type(ecx, ebml_w, node_id_to_type(ecx.tcx, nitem.id));
        if abi == foreign_abi_rust_intrinsic {
            (ecx.encode_inlined_item)(ecx, ebml_w, path, ii_foreign(nitem));
        } else {
            encode_symbol(ecx, ebml_w, nitem.id);
        }
        encode_path(ecx, ebml_w, path, ast_map::path_name(nitem.ident));
      }
      foreign_item_const(*) => {
        encode_def_id(ebml_w, local_def(nitem.id));
        encode_family(ebml_w, 'c');
        encode_type(ecx, ebml_w, node_id_to_type(ecx.tcx, nitem.id));
        encode_symbol(ecx, ebml_w, nitem.id);
        encode_path(ecx, ebml_w, path, ast_map::path_name(nitem.ident));
      }
    }
    ebml_w.end_tag();
}

fn encode_info_for_items(ecx: @EncodeContext, ebml_w: writer::Encoder,
                         crate: &crate) -> ~[entry<int>] {
    let index = @mut ~[];
    ebml_w.start_tag(tag_items_data);
    index.push(entry { val: crate_node_id, pos: ebml_w.writer.tell() });
    encode_info_for_mod(ecx, ebml_w, &crate.node.module,
                        crate_node_id, ~[],
                        syntax::parse::token::special_idents::invalid);
    visit::visit_crate(*crate, (), visit::mk_vt(@visit::Visitor {
        visit_expr: |_e, _cx, _v| { },
        visit_item: {
            let ebml_w = copy ebml_w;
            |i, cx, v| {
                visit::visit_item(i, cx, v);
                match *ecx.tcx.items.get(&i.id) {
                    ast_map::node_item(_, pt) => {
                        encode_info_for_item(ecx, ebml_w, i,
                                             index, *pt);
                    }
                    _ => fail!(~"bad item")
                }
            }
        },
        visit_foreign_item: {
            let ebml_w = copy ebml_w;
            |ni, cx, v| {
                visit::visit_foreign_item(ni, cx, v);
                match *ecx.tcx.items.get(&ni.id) {
                    ast_map::node_foreign_item(_, abi, _, pt) => {
                        encode_info_for_foreign_item(ecx, ebml_w, ni,
                                                     index, /*bad*/copy *pt,
                                                     abi);
                    }
                    // case for separate item and foreign-item tables
                    _ => fail!(~"bad foreign item")
                }
            }
        },
        ..*visit::default_visitor()
    }));
    ebml_w.end_tag();
    return /*bad*/copy *index;
}


// Path and definition ID indexing

fn create_index<T:Copy + Hash + IterBytes>(index: ~[entry<T>]) ->
   ~[@~[entry<T>]] {
    let mut buckets: ~[@mut ~[entry<T>]] = ~[];
    for uint::range(0u, 256u) |_i| { buckets.push(@mut ~[]); };
    for index.each |elt| {
        let h = elt.val.hash() as uint;
        buckets[h % 256].push(*elt);
    }

    let mut buckets_frozen = ~[];
    for buckets.each |bucket| {
        buckets_frozen.push(@/*bad*/copy **bucket);
    }
    return buckets_frozen;
}

fn encode_index<T>(ebml_w: writer::Encoder, buckets: ~[@~[entry<T>]],
                   write_fn: &fn(@io::Writer, T)) {
    let writer = ebml_w.writer;
    ebml_w.start_tag(tag_index);
    let mut bucket_locs: ~[uint] = ~[];
    ebml_w.start_tag(tag_index_buckets);
    for buckets.each |bucket| {
        bucket_locs.push(ebml_w.writer.tell());
        ebml_w.start_tag(tag_index_buckets_bucket);
        for vec::each(**bucket) |elt| {
            ebml_w.start_tag(tag_index_buckets_bucket_elt);
            fail_unless!(elt.pos < 0xffff_ffff);
            writer.write_be_u32(elt.pos as u32);
            write_fn(writer, elt.val);
            ebml_w.end_tag();
        }
        ebml_w.end_tag();
    }
    ebml_w.end_tag();
    ebml_w.start_tag(tag_index_table);
    for bucket_locs.each |pos| {
        fail_unless!(*pos < 0xffff_ffff);
        writer.write_be_u32(*pos as u32);
    }
    ebml_w.end_tag();
    ebml_w.end_tag();
}

fn write_str(writer: @io::Writer, &&s: ~str) { writer.write_str(s); }

fn write_int(writer: @io::Writer, &&n: int) {
    fail_unless!(n < 0x7fff_ffff);
    writer.write_be_u32(n as u32);
}

fn encode_meta_item(ebml_w: writer::Encoder, mi: @meta_item) {
    match mi.node {
      meta_word(name) => {
        ebml_w.start_tag(tag_meta_item_word);
        ebml_w.start_tag(tag_meta_item_name);
        ebml_w.writer.write(str::to_bytes(*name));
        ebml_w.end_tag();
        ebml_w.end_tag();
      }
      meta_name_value(name, value) => {
        match value.node {
          lit_str(value) => {
            ebml_w.start_tag(tag_meta_item_name_value);
            ebml_w.start_tag(tag_meta_item_name);
            ebml_w.writer.write(str::to_bytes(*name));
            ebml_w.end_tag();
            ebml_w.start_tag(tag_meta_item_value);
            ebml_w.writer.write(str::to_bytes(*value));
            ebml_w.end_tag();
            ebml_w.end_tag();
          }
          _ => {/* FIXME (#623): encode other variants */ }
        }
      }
      meta_list(name, ref items) => {
        ebml_w.start_tag(tag_meta_item_list);
        ebml_w.start_tag(tag_meta_item_name);
        ebml_w.writer.write(str::to_bytes(*name));
        ebml_w.end_tag();
        for items.each |inner_item| {
            encode_meta_item(ebml_w, *inner_item);
        }
        ebml_w.end_tag();
      }
    }
}

fn encode_attributes(ebml_w: writer::Encoder, attrs: &[attribute]) {
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
fn synthesize_crate_attrs(ecx: @EncodeContext,
                          crate: &crate) -> ~[attribute] {

    fn synthesize_link_attr(ecx: @EncodeContext, +items: ~[@meta_item]) ->
       attribute {

        fail_unless!(!ecx.link_meta.name.is_empty());
        fail_unless!(!ecx.link_meta.vers.is_empty());

        let name_item =
            attr::mk_name_value_item_str(@~"name",
                                         @ecx.link_meta.name.to_owned());
        let vers_item =
            attr::mk_name_value_item_str(@~"vers",
                                         @ecx.link_meta.vers.to_owned());

        let other_items =
            {
                let tmp = attr::remove_meta_items_by_name(items, ~"name");
                attr::remove_meta_items_by_name(tmp, ~"vers")
            };

        let meta_items = vec::append(~[name_item, vers_item], other_items);
        let link_item = attr::mk_list_item(@~"link", meta_items);

        return attr::mk_attr(link_item);
    }

    let mut attrs: ~[attribute] = ~[];
    let mut found_link_attr = false;
    for crate.node.attrs.each |attr| {
        attrs.push(
            if *attr::get_attr_name(attr) != ~"link" {
                /*bad*/copy *attr
            } else {
                match attr.node.value.node {
                  meta_list(_, ref l) => {
                    found_link_attr = true;;
                    synthesize_link_attr(ecx, /*bad*/copy *l)
                  }
                  _ => /*bad*/copy *attr
                }
            });
    }

    if !found_link_attr { attrs.push(synthesize_link_attr(ecx, ~[])); }

    return attrs;
}

fn encode_crate_deps(ecx: @EncodeContext,
                     ebml_w: writer::Encoder,
                     cstore: @mut cstore::CStore) {
    fn get_ordered_deps(ecx: @EncodeContext, cstore: @mut cstore::CStore)
                     -> ~[decoder::crate_dep] {
        type numdep = decoder::crate_dep;

        // Pull the cnums and name,vers,hash out of cstore
        let mut deps = ~[];
        do cstore::iter_crate_data(cstore) |key, val| {
            let dep = decoder::crate_dep {cnum: key,
                       name: ecx.tcx.sess.ident_of(/*bad*/ copy *val.name),
                       vers: decoder::get_crate_vers(val.data),
                       hash: decoder::get_crate_hash(val.data)};
            deps.push(dep);
        };

        // Sort by cnum
        std::sort::quick_sort(deps, |kv1, kv2| kv1.cnum <= kv2.cnum);

        // Sanity-check the crate numbers
        let mut expected_cnum = 1;
        for deps.each |n| {
            fail_unless!((n.cnum == expected_cnum));
            expected_cnum += 1;
        }

        // mut -> immutable hack for vec::map
        deps.slice(0, deps.len()).to_owned()
    }

    // We're just going to write a list of crate 'name-hash-version's, with
    // the assumption that they are numbered 1 to n.
    // FIXME (#2166): This is not nearly enough to support correct versioning
    // but is enough to get transitive crate dependencies working.
    ebml_w.start_tag(tag_crate_deps);
    for get_ordered_deps(ecx, cstore).each |dep| {
        encode_crate_dep(ecx, ebml_w, *dep);
    }
    ebml_w.end_tag();
}

fn encode_lang_items(ecx: @EncodeContext, ebml_w: writer::Encoder) {
    ebml_w.start_tag(tag_lang_items);

    for ecx.tcx.lang_items.each_item |def_id, i| {
        if def_id.crate != local_crate {
            loop;
        }

        ebml_w.start_tag(tag_lang_items_item);

        ebml_w.start_tag(tag_lang_items_item_id);
        ebml_w.writer.write_be_u32(i as u32);
        ebml_w.end_tag();   // tag_lang_items_item_id

        ebml_w.start_tag(tag_lang_items_item_node_id);
        ebml_w.writer.write_be_u32(def_id.node as u32);
        ebml_w.end_tag();   // tag_lang_items_item_node_id

        ebml_w.end_tag();   // tag_lang_items_item
    }

    ebml_w.end_tag();   // tag_lang_items
}

fn encode_link_args(ecx: @EncodeContext,
                    ebml_w: writer::Encoder) {
    ebml_w.start_tag(tag_link_args);

    let link_args = cstore::get_used_link_args(ecx.cstore);
    for link_args.each |link_arg| {
        ebml_w.start_tag(tag_link_args_arg);
        ebml_w.writer.write_str(link_arg.to_str());
        ebml_w.end_tag();
    }

    ebml_w.end_tag();
}

fn encode_crate_dep(ecx: @EncodeContext, ebml_w: writer::Encoder,
                    dep: decoder::crate_dep) {
    ebml_w.start_tag(tag_crate_dep);
    ebml_w.start_tag(tag_crate_dep_name);
    ebml_w.writer.write(str::to_bytes(*ecx.tcx.sess.str_of(dep.name)));
    ebml_w.end_tag();
    ebml_w.start_tag(tag_crate_dep_vers);
    ebml_w.writer.write(str::to_bytes(*dep.vers));
    ebml_w.end_tag();
    ebml_w.start_tag(tag_crate_dep_hash);
    ebml_w.writer.write(str::to_bytes(*dep.hash));
    ebml_w.end_tag();
    ebml_w.end_tag();
}

fn encode_hash(ebml_w: writer::Encoder, hash: &str) {
    ebml_w.start_tag(tag_crate_hash);
    ebml_w.writer.write(str::to_bytes(hash));
    ebml_w.end_tag();
}

// NB: Increment this as you change the metadata encoding version.
pub static metadata_encoding_version : &'static [u8] =
    &[0x72, //'r' as u8,
      0x75, //'u' as u8,
      0x73, //'s' as u8,
      0x74, //'t' as u8,
      0, 0, 0, 1 ];

pub fn encode_metadata(+parms: EncodeParams, crate: &crate) -> ~[u8] {
    let wr = @io::BytesWriter();
    let mut stats = Stats {
        inline_bytes: 0,
        attr_bytes: 0,
        dep_bytes: 0,
        lang_item_bytes: 0,
        link_args_bytes: 0,
        item_bytes: 0,
        index_bytes: 0,
        zero_bytes: 0,
        total_bytes: 0,
        n_inlines: 0
    };
    let EncodeParams{item_symbols, diag, tcx, reachable, reexports2,
                     discrim_symbols, cstore, encode_inlined_item,
                     link_meta, _} = parms;
    let ecx = @EncodeContext {
        diag: diag,
        tcx: tcx,
        stats: @mut stats,
        reachable: reachable,
        reexports2: reexports2,
        item_symbols: item_symbols,
        discrim_symbols: discrim_symbols,
        link_meta: link_meta,
        cstore: cstore,
        encode_inlined_item: encode_inlined_item,
        type_abbrevs: @mut LinearMap::new()
     };

    let ebml_w = writer::Encoder(wr as @io::Writer);

    encode_hash(ebml_w, ecx.link_meta.extras_hash);

    let mut i = wr.pos;
    let crate_attrs = synthesize_crate_attrs(ecx, crate);
    encode_attributes(ebml_w, crate_attrs);
    ecx.stats.attr_bytes = wr.pos - i;

    i = wr.pos;
    encode_crate_deps(ecx, ebml_w, ecx.cstore);
    ecx.stats.dep_bytes = wr.pos - i;

    // Encode the language items.
    i = wr.pos;
    encode_lang_items(ecx, ebml_w);
    ecx.stats.lang_item_bytes = wr.pos - i;

    // Encode the link args.
    i = wr.pos;
    encode_link_args(ecx, ebml_w);
    ecx.stats.link_args_bytes = wr.pos - i;

    // Encode and index the items.
    ebml_w.start_tag(tag_items);
    i = wr.pos;
    let items_index = encode_info_for_items(ecx, ebml_w, crate);
    ecx.stats.item_bytes = wr.pos - i;

    i = wr.pos;
    let items_buckets = create_index(items_index);
    encode_index(ebml_w, items_buckets, write_int);
    ecx.stats.index_bytes = wr.pos - i;
    ebml_w.end_tag();

    ecx.stats.total_bytes = wr.pos;

    if (tcx.sess.meta_stats()) {

        do wr.bytes.each |e| {
            if *e == 0 {
                ecx.stats.zero_bytes += 1;
            }
            true
        }

        io::println("metadata stats:");
        io::println(fmt!("    inline bytes: %u", ecx.stats.inline_bytes));
        io::println(fmt!(" attribute bytes: %u", ecx.stats.attr_bytes));
        io::println(fmt!("       dep bytes: %u", ecx.stats.dep_bytes));
        io::println(fmt!(" lang item bytes: %u", ecx.stats.lang_item_bytes));
        io::println(fmt!(" link args bytes: %u", ecx.stats.link_args_bytes));
        io::println(fmt!("      item bytes: %u", ecx.stats.item_bytes));
        io::println(fmt!("     index bytes: %u", ecx.stats.index_bytes));
        io::println(fmt!("      zero bytes: %u", ecx.stats.zero_bytes));
        io::println(fmt!("     total bytes: %u", ecx.stats.total_bytes));
    }

    // Pad this, since something (LLVM, presumably) is cutting off the
    // remaining % 4 bytes.
    wr.write(&[0u8, 0u8, 0u8, 0u8]);

    // FIXME #3396: weird bug here, for reasons unclear this emits random
    // looking bytes (mostly 0x1) if we use the version byte-array constant
    // above; so we use a string constant inline instead.
    //
    // Should be:
    //
    //   vec::from_slice(metadata_encoding_version) +

    (do str::as_bytes(&~"rust\x00\x00\x00\x01") |bytes| {
        vec::slice(*bytes, 0, 8).to_vec()
    }) + flate::deflate_bytes(wr.bytes)
}

// Get the encoded string for a type
pub fn encoded_ty(tcx: ty::ctxt, t: ty::t) -> ~str {
    let cx = @tyencode::ctxt {
        diag: tcx.diag,
        ds: def_to_str,
        tcx: tcx,
        reachable: |_id| false,
        abbrevs: tyencode::ac_no_abbrevs};
    do io::with_str_writer |wr| {
        tyencode::enc_ty(wr, cx, t);
    }
}


// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
