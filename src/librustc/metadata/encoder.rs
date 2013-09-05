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


use metadata::common::*;
use metadata::cstore;
use metadata::decoder;
use metadata::tyencode;
use middle::ty::{node_id_to_type, lookup_item_type};
use middle::ty;
use middle::typeck;
use middle::astencode;
use middle;

use std::hashmap::{HashMap, HashSet};
use std::io;
use std::str;
use std::vec;
use extra::flate;
use extra::serialize::Encodable;
use extra;
use syntax::abi::AbiSet;
use syntax::ast::*;
use syntax::ast;
use syntax::ast_map;
use syntax::ast_util::*;
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::diagnostic::span_handler;
use syntax::parse::token::special_idents;
use syntax::ast_util;
use syntax::visit::Visitor;
use syntax::visit;
use syntax::parse::token;
use syntax;
use writer = extra::ebml::writer;

use std::cast;

// used by astencode:
type abbrev_map = @mut HashMap<ty::t, tyencode::ty_abbrev>;

pub type encode_inlined_item<'self> = &'self fn(ecx: &EncodeContext,
                                   ebml_w: &mut writer::Encoder,
                                   path: &[ast_map::path_elt],
                                   ii: ast::inlined_item);

pub struct EncodeParams<'self> {
    diag: @mut span_handler,
    tcx: ty::ctxt,
    reexports2: middle::resolve::ExportMap2,
    item_symbols: &'self HashMap<ast::NodeId, ~str>,
    discrim_symbols: &'self HashMap<ast::NodeId, @str>,
    link_meta: &'self LinkMeta,
    cstore: @mut cstore::CStore,
    encode_inlined_item: encode_inlined_item<'self>,
    reachable: @mut HashSet<ast::NodeId>,
}

struct Stats {
    inline_bytes: uint,
    attr_bytes: uint,
    dep_bytes: uint,
    lang_item_bytes: uint,
    link_args_bytes: uint,
    impl_bytes: uint,
    misc_bytes: uint,
    item_bytes: uint,
    index_bytes: uint,
    zero_bytes: uint,
    total_bytes: uint,

    n_inlines: uint
}

pub struct EncodeContext<'self> {
    diag: @mut span_handler,
    tcx: ty::ctxt,
    stats: @mut Stats,
    reexports2: middle::resolve::ExportMap2,
    item_symbols: &'self HashMap<ast::NodeId, ~str>,
    discrim_symbols: &'self HashMap<ast::NodeId, @str>,
    link_meta: &'self LinkMeta,
    cstore: &'self cstore::CStore,
    encode_inlined_item: encode_inlined_item<'self>,
    type_abbrevs: abbrev_map,
    reachable: @mut HashSet<ast::NodeId>,
}

pub fn reachable(ecx: &EncodeContext, id: NodeId) -> bool {
    ecx.reachable.contains(&id)
}

fn encode_name(ecx: &EncodeContext,
               ebml_w: &mut writer::Encoder,
               name: Ident) {
    ebml_w.wr_tagged_str(tag_paths_data_name, ecx.tcx.sess.str_of(name));
}

fn encode_impl_type_basename(ecx: &EncodeContext,
                             ebml_w: &mut writer::Encoder,
                             name: Ident) {
    ebml_w.wr_tagged_str(tag_item_impl_type_basename,
                         ecx.tcx.sess.str_of(name));
}

pub fn encode_def_id(ebml_w: &mut writer::Encoder, id: DefId) {
    ebml_w.wr_tagged_str(tag_def_id, def_to_str(id));
}

fn encode_region_param(ecx: &EncodeContext,
                       ebml_w: &mut writer::Encoder,
                       it: @ast::item) {
    let opt_rp = ecx.tcx.region_paramd_items.find(&it.id);
    for rp in opt_rp.iter() {
        ebml_w.start_tag(tag_region_param);
        rp.encode(ebml_w);
        ebml_w.end_tag();
    }
}

#[deriving(Clone)]
struct entry<T> {
    val: T,
    pos: uint
}

fn add_to_index(ebml_w: &mut writer::Encoder,
                path: &[Ident],
                index: &mut ~[entry<~str>],
                name: Ident) {
    let mut full_path = ~[];
    full_path.push_all(path);
    full_path.push(name);
    index.push(
        entry {
            val: ast_util::path_name_i(full_path),
            pos: ebml_w.writer.tell()
        });
}

fn encode_trait_ref(ebml_w: &mut writer::Encoder,
                    ecx: &EncodeContext,
                    trait_ref: &ty::TraitRef,
                    tag: uint) {
    let ty_str_ctxt = @tyencode::ctxt {
        diag: ecx.diag,
        ds: def_to_str,
        tcx: ecx.tcx,
        abbrevs: tyencode::ac_use_abbrevs(ecx.type_abbrevs)
    };

    ebml_w.start_tag(tag);
    tyencode::enc_trait_ref(ebml_w.writer, ty_str_ctxt, trait_ref);
    ebml_w.end_tag();
}

fn encode_impl_vtables(ebml_w: &mut writer::Encoder,
                       ecx: &EncodeContext,
                       vtables: &typeck::impl_res) {
    ebml_w.start_tag(tag_item_impl_vtables);
    astencode::encode_vtable_res(ecx, ebml_w, vtables.trait_vtables);
    astencode::encode_vtable_param_res(ecx, ebml_w, vtables.self_vtables);
    ebml_w.end_tag();
}

// Item info table encoding
fn encode_family(ebml_w: &mut writer::Encoder, c: char) {
    ebml_w.start_tag(tag_items_data_item_family);
    ebml_w.writer.write(&[c as u8]);
    ebml_w.end_tag();
}

pub fn def_to_str(did: DefId) -> ~str {
    fmt!("%d:%d", did.crate, did.node)
}

fn encode_ty_type_param_defs(ebml_w: &mut writer::Encoder,
                             ecx: &EncodeContext,
                             params: @~[ty::TypeParameterDef],
                             tag: uint) {
    let ty_str_ctxt = @tyencode::ctxt {
        diag: ecx.diag,
        ds: def_to_str,
        tcx: ecx.tcx,
        abbrevs: tyencode::ac_use_abbrevs(ecx.type_abbrevs)
    };
    for param in params.iter() {
        ebml_w.start_tag(tag);
        tyencode::enc_type_param_def(ebml_w.writer, ty_str_ctxt, param);
        ebml_w.end_tag();
    }
}

fn encode_bounds_and_type(ebml_w: &mut writer::Encoder,
                          ecx: &EncodeContext,
                          tpt: &ty::ty_param_bounds_and_ty) {
    encode_ty_type_param_defs(ebml_w, ecx, tpt.generics.type_param_defs,
                              tag_items_data_item_ty_param_bounds);
    encode_type(ecx, ebml_w, tpt.ty);
}

fn encode_variant_id(ebml_w: &mut writer::Encoder, vid: DefId) {
    ebml_w.start_tag(tag_items_data_item_variant);
    let s = def_to_str(vid);
    ebml_w.writer.write(s.as_bytes());
    ebml_w.end_tag();
}

pub fn write_type(ecx: &EncodeContext,
                  ebml_w: &mut writer::Encoder,
                  typ: ty::t) {
    let ty_str_ctxt = @tyencode::ctxt {
        diag: ecx.diag,
        ds: def_to_str,
        tcx: ecx.tcx,
        abbrevs: tyencode::ac_use_abbrevs(ecx.type_abbrevs)
    };
    tyencode::enc_ty(ebml_w.writer, ty_str_ctxt, typ);
}

pub fn write_vstore(ecx: &EncodeContext,
                    ebml_w: &mut writer::Encoder,
                    vstore: ty::vstore) {
    let ty_str_ctxt = @tyencode::ctxt {
        diag: ecx.diag,
        ds: def_to_str,
        tcx: ecx.tcx,
        abbrevs: tyencode::ac_use_abbrevs(ecx.type_abbrevs)
    };
    tyencode::enc_vstore(ebml_w.writer, ty_str_ctxt, vstore);
}

fn encode_type(ecx: &EncodeContext,
               ebml_w: &mut writer::Encoder,
               typ: ty::t) {
    ebml_w.start_tag(tag_items_data_item_type);
    write_type(ecx, ebml_w, typ);
    ebml_w.end_tag();
}

fn encode_transformed_self_ty(ecx: &EncodeContext,
                              ebml_w: &mut writer::Encoder,
                              opt_typ: Option<ty::t>) {
    for &typ in opt_typ.iter() {
        ebml_w.start_tag(tag_item_method_transformed_self_ty);
        write_type(ecx, ebml_w, typ);
        ebml_w.end_tag();
    }
}

fn encode_method_fty(ecx: &EncodeContext,
                     ebml_w: &mut writer::Encoder,
                     typ: &ty::BareFnTy) {
    ebml_w.start_tag(tag_item_method_fty);

    let ty_str_ctxt = @tyencode::ctxt {
        diag: ecx.diag,
        ds: def_to_str,
        tcx: ecx.tcx,
        abbrevs: tyencode::ac_use_abbrevs(ecx.type_abbrevs)
    };
    tyencode::enc_bare_fn_ty(ebml_w.writer, ty_str_ctxt, typ);

    ebml_w.end_tag();
}

fn encode_symbol(ecx: &EncodeContext,
                 ebml_w: &mut writer::Encoder,
                 id: NodeId) {
    ebml_w.start_tag(tag_items_data_item_symbol);
    match ecx.item_symbols.find(&id) {
        Some(x) => {
            debug!("encode_symbol(id=%?, str=%s)", id, *x);
            ebml_w.writer.write(x.as_bytes());
        }
        None => {
            ecx.diag.handler().bug(
                fmt!("encode_symbol: id not found %d", id));
        }
    }
    ebml_w.end_tag();
}

fn encode_disr_val(_: &EncodeContext,
                   ebml_w: &mut writer::Encoder,
                   disr_val: ty::Disr) {
    ebml_w.start_tag(tag_disr_val);
    let s = disr_val.to_str();
    ebml_w.writer.write(s.as_bytes());
    ebml_w.end_tag();
}

fn encode_parent_item(ebml_w: &mut writer::Encoder, id: DefId) {
    ebml_w.start_tag(tag_items_data_parent_item);
    let s = def_to_str(id);
    ebml_w.writer.write(s.as_bytes());
    ebml_w.end_tag();
}

fn encode_enum_variant_info(ecx: &EncodeContext,
                            ebml_w: &mut writer::Encoder,
                            id: NodeId,
                            variants: &[variant],
                            path: &[ast_map::path_elt],
                            index: @mut ~[entry<i64>],
                            generics: &ast::Generics) {
    debug!("encode_enum_variant_info(id=%?)", id);

    let mut disr_val = 0;
    let mut i = 0;
    let vi = ty::enum_variants(ecx.tcx,
                               ast::DefId { crate: LOCAL_CRATE, node: id });
    for variant in variants.iter() {
        let def_id = local_def(variant.node.id);
        index.push(entry {val: variant.node.id as i64,
                          pos: ebml_w.writer.tell()});
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, def_id);
        encode_family(ebml_w, 'v');
        encode_name(ecx, ebml_w, variant.node.name);
        encode_parent_item(ebml_w, local_def(id));
        encode_visibility(ebml_w, variant.node.vis);
        encode_attributes(ebml_w, variant.node.attrs);
        match variant.node.kind {
            ast::tuple_variant_kind(ref args)
                    if args.len() > 0 && generics.ty_params.len() == 0 => {
                encode_symbol(ecx, ebml_w, variant.node.id);
            }
            ast::tuple_variant_kind(_) | ast::struct_variant_kind(_) => {}
        }
        if vi[i].disr_val != disr_val {
            encode_disr_val(ecx, ebml_w, vi[i].disr_val);
            disr_val = vi[i].disr_val;
        }
        encode_bounds_and_type(ebml_w, ecx,
                               &lookup_item_type(ecx.tcx, def_id));
        encode_path(ecx, ebml_w, path,
                    ast_map::path_name(variant.node.name));
        ebml_w.end_tag();
        disr_val += 1;
        i += 1;
    }
}

fn encode_path(ecx: &EncodeContext,
               ebml_w: &mut writer::Encoder,
               path: &[ast_map::path_elt],
               name: ast_map::path_elt) {
    fn encode_path_elt(ecx: &EncodeContext,
                       ebml_w: &mut writer::Encoder,
                       elt: ast_map::path_elt) {
        match elt {
            ast_map::path_mod(n) => {
                ebml_w.wr_tagged_str(tag_path_elt_mod, ecx.tcx.sess.str_of(n));
            }
            ast_map::path_name(n) => {
                ebml_w.wr_tagged_str(tag_path_elt_name, ecx.tcx.sess.str_of(n));
            }
            ast_map::path_pretty_name(n, extra) => {
                ebml_w.start_tag(tag_path_elt_pretty_name);
                ebml_w.wr_tagged_str(tag_path_elt_pretty_name_ident,
                                     ecx.tcx.sess.str_of(n));
                ebml_w.wr_tagged_u64(tag_path_elt_pretty_name_extra, extra);
                ebml_w.end_tag();
            }
        }
    }

    ebml_w.start_tag(tag_path);
    ebml_w.wr_tagged_u32(tag_path_len, (path.len() + 1) as u32);
    for pe in path.iter() {
        encode_path_elt(ecx, ebml_w, *pe);
    }
    encode_path_elt(ecx, ebml_w, name);
    ebml_w.end_tag();
}

fn encode_reexported_static_method(ecx: &EncodeContext,
                                   ebml_w: &mut writer::Encoder,
                                   exp: &middle::resolve::Export2,
                                   method_def_id: DefId,
                                   method_ident: Ident) {
    debug!("(encode reexported static method) %s::%s",
            exp.name, ecx.tcx.sess.str_of(method_ident));
    ebml_w.start_tag(tag_items_data_item_reexport);
    ebml_w.start_tag(tag_items_data_item_reexport_def_id);
    ebml_w.wr_str(def_to_str(method_def_id));
    ebml_w.end_tag();
    ebml_w.start_tag(tag_items_data_item_reexport_name);
    ebml_w.wr_str(fmt!("%s::%s", exp.name, ecx.tcx.sess.str_of(method_ident)));
    ebml_w.end_tag();
    ebml_w.end_tag();
}

fn encode_reexported_static_base_methods(ecx: &EncodeContext,
                                         ebml_w: &mut writer::Encoder,
                                         exp: &middle::resolve::Export2)
                                         -> bool {
    match ecx.tcx.inherent_impls.find(&exp.def_id) {
        Some(implementations) => {
            for &base_impl in implementations.iter() {
                for &m in base_impl.methods.iter() {
                    if m.explicit_self == ast::sty_static {
                        encode_reexported_static_method(ecx, ebml_w, exp,
                                                        m.def_id, m.ident);
                    }
                }
            }

            true
        }
        None => { false }
    }
}

fn encode_reexported_static_trait_methods(ecx: &EncodeContext,
                                          ebml_w: &mut writer::Encoder,
                                          exp: &middle::resolve::Export2)
                                          -> bool {
    match ecx.tcx.trait_methods_cache.find(&exp.def_id) {
        Some(methods) => {
            for &m in methods.iter() {
                if m.explicit_self == ast::sty_static {
                    encode_reexported_static_method(ecx, ebml_w, exp,
                                                    m.def_id, m.ident);
                }
            }

            true
        }
        None => { false }
    }
}

fn encode_reexported_static_methods(ecx: &EncodeContext,
                                    ebml_w: &mut writer::Encoder,
                                    mod_path: &[ast_map::path_elt],
                                    exp: &middle::resolve::Export2) {
    match ecx.tcx.items.find(&exp.def_id.node) {
        Some(&ast_map::node_item(item, path)) => {
            let original_name = ecx.tcx.sess.str_of(item.ident);

            //
            // We don't need to reexport static methods on items
            // declared in the same module as our `pub use ...` since
            // that's done when we encode the item itself.
            //
            // The only exception is when the reexport *changes* the
            // name e.g. `pub use Foo = self::Bar` -- we have
            // encoded metadata for static methods relative to Bar,
            // but not yet for Foo.
            //
            if mod_path != *path || exp.name != original_name {
                if !encode_reexported_static_base_methods(ecx, ebml_w, exp) {
                    if encode_reexported_static_trait_methods(ecx, ebml_w, exp) {
                        debug!(fmt!("(encode reexported static methods) %s \
                                    [trait]",
                                    original_name));
                    }
                }
                else {
                    debug!(fmt!("(encode reexported static methods) %s [base]",
                                original_name));
                }
            }
        }
        _ => {}
    }
}

/// Iterates through "auxiliary node IDs", which are node IDs that describe
/// top-level items that are sub-items of the given item. Specifically:
///
/// * For enums, iterates through the node IDs of the variants.
///
/// * For newtype structs, iterates through the node ID of the constructor.
fn each_auxiliary_node_id(item: @item, callback: &fn(NodeId) -> bool)
                          -> bool {
    let mut continue = true;
    match item.node {
        item_enum(ref enum_def, _) => {
            for variant in enum_def.variants.iter() {
                continue = callback(variant.node.id);
                if !continue {
                    break
                }
            }
        }
        item_struct(struct_def, _) => {
            // If this is a newtype struct, return the constructor.
            match struct_def.ctor_id {
                Some(ctor_id) if struct_def.fields.len() > 0 &&
                        struct_def.fields[0].node.kind ==
                        ast::unnamed_field => {
                    continue = callback(ctor_id);
                }
                _ => {}
            }
        }
        _ => {}
    }

    continue
}

fn encode_reexports(ecx: &EncodeContext,
                    ebml_w: &mut writer::Encoder,
                    id: NodeId,
                    path: &[ast_map::path_elt]) {
    debug!("(encoding info for module) encoding reexports for %d", id);
    match ecx.reexports2.find(&id) {
        Some(ref exports) => {
            debug!("(encoding info for module) found reexports for %d", id);
            for exp in exports.iter() {
                debug!("(encoding info for module) reexport '%s' (%d/%d) for \
                        %d",
                       exp.name,
                       exp.def_id.crate,
                       exp.def_id.node,
                       id);
                ebml_w.start_tag(tag_items_data_item_reexport);
                ebml_w.start_tag(tag_items_data_item_reexport_def_id);
                ebml_w.wr_str(def_to_str(exp.def_id));
                ebml_w.end_tag();
                ebml_w.start_tag(tag_items_data_item_reexport_name);
                ebml_w.wr_str(exp.name);
                ebml_w.end_tag();
                ebml_w.end_tag();
                encode_reexported_static_methods(ecx, ebml_w, path, exp);
            }
        }
        None => {
            debug!("(encoding info for module) found no reexports for %d",
                   id);
        }
    }
}

fn encode_info_for_mod(ecx: &EncodeContext,
                       ebml_w: &mut writer::Encoder,
                       md: &_mod,
                       id: NodeId,
                       path: &[ast_map::path_elt],
                       name: Ident,
                       vis: visibility) {
    ebml_w.start_tag(tag_items_data_item);
    encode_def_id(ebml_w, local_def(id));
    encode_family(ebml_w, 'm');
    encode_name(ecx, ebml_w, name);
    debug!("(encoding info for module) encoding info for module ID %d", id);

    // Encode info about all the module children.
    for item in md.items.iter() {
        ebml_w.start_tag(tag_mod_child);
        ebml_w.wr_str(def_to_str(local_def(item.id)));
        ebml_w.end_tag();

        do each_auxiliary_node_id(*item) |auxiliary_node_id| {
            ebml_w.start_tag(tag_mod_child);
            ebml_w.wr_str(def_to_str(local_def(auxiliary_node_id)));
            ebml_w.end_tag();
            true
        };

        match item.node {
            item_impl(*) => {
                let (ident, did) = (item.ident, item.id);
                debug!("(encoding info for module) ... encoding impl %s \
                        (%?/%?)",
                        ecx.tcx.sess.str_of(ident),
                        did,
                        ast_map::node_id_to_str(ecx.tcx.items, did, token::get_ident_interner()));

                ebml_w.start_tag(tag_mod_impl);
                ebml_w.wr_str(def_to_str(local_def(did)));
                ebml_w.end_tag();
            }
            _ => {}
        }
    }

    encode_path(ecx, ebml_w, path, ast_map::path_mod(name));

    // Encode the reexports of this module, if this module is public.
    if vis == public {
        debug!("(encoding info for module) encoding reexports for %d", id);
        encode_reexports(ecx, ebml_w, id, path);
    }

    ebml_w.end_tag();
}

fn encode_struct_field_family(ebml_w: &mut writer::Encoder,
                              visibility: visibility) {
    encode_family(ebml_w, match visibility {
        public => 'g',
        private => 'j',
        inherited => 'N'
    });
}

fn encode_visibility(ebml_w: &mut writer::Encoder, visibility: visibility) {
    ebml_w.start_tag(tag_items_data_item_visibility);
    let ch = match visibility {
        public => 'y',
        private => 'n',
        inherited => 'i',
    };
    ebml_w.wr_str(str::from_char(ch));
    ebml_w.end_tag();
}

fn encode_explicit_self(ebml_w: &mut writer::Encoder, explicit_self: ast::explicit_self_) {
    ebml_w.start_tag(tag_item_trait_method_explicit_self);

    // Encode the base self type.
    match explicit_self {
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
        sty_uniq => {
            ebml_w.writer.write(&[ '~' as u8 ]);
        }
    }

    ebml_w.end_tag();

    fn encode_mutability(ebml_w: &writer::Encoder,
                         m: ast::Mutability) {
        match m {
            MutImmutable => ebml_w.writer.write(&[ 'i' as u8 ]),
            MutMutable => ebml_w.writer.write(&[ 'm' as u8 ]),
        }
    }
}

fn encode_method_sort(ebml_w: &mut writer::Encoder, sort: char) {
    ebml_w.start_tag(tag_item_trait_method_sort);
    ebml_w.writer.write(&[ sort as u8 ]);
    ebml_w.end_tag();
}

fn encode_provided_source(ebml_w: &mut writer::Encoder,
                          source_opt: Option<DefId>) {
    for source in source_opt.iter() {
        ebml_w.start_tag(tag_item_method_provided_source);
        let s = def_to_str(*source);
        ebml_w.writer.write(s.as_bytes());
        ebml_w.end_tag();
    }
}

/* Returns an index of items in this class */
fn encode_info_for_struct(ecx: &EncodeContext,
                          ebml_w: &mut writer::Encoder,
                          path: &[ast_map::path_elt],
                          fields: &[@struct_field],
                          global_index: @mut ~[entry<i64>])
                          -> ~[entry<i64>] {
    /* Each class has its own index, since different classes
       may have fields with the same name */
    let mut index = ~[];
    let tcx = ecx.tcx;
     /* We encode both private and public fields -- need to include
        private fields to get the offsets right */
    for field in fields.iter() {
        let (nm, vis) = match field.node.kind {
            named_field(nm, vis) => (nm, vis),
            unnamed_field => (special_idents::unnamed_field, inherited)
        };

        let id = field.node.id;
        index.push(entry {val: id as i64, pos: ebml_w.writer.tell()});
        global_index.push(entry {val: id as i64, pos: ebml_w.writer.tell()});
        ebml_w.start_tag(tag_items_data_item);
        debug!("encode_info_for_struct: doing %s %d",
               tcx.sess.str_of(nm), id);
        encode_struct_field_family(ebml_w, vis);
        encode_name(ecx, ebml_w, nm);
        encode_path(ecx, ebml_w, path, ast_map::path_name(nm));
        encode_type(ecx, ebml_w, node_id_to_type(tcx, id));
        encode_def_id(ebml_w, local_def(id));
        ebml_w.end_tag();
    }
    index
}

fn encode_info_for_struct_ctor(ecx: &EncodeContext,
                               ebml_w: &mut writer::Encoder,
                               path: &[ast_map::path_elt],
                               name: ast::Ident,
                               ctor_id: NodeId,
                               index: @mut ~[entry<i64>]) {
    index.push(entry { val: ctor_id as i64, pos: ebml_w.writer.tell() });

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

fn encode_method_ty_fields(ecx: &EncodeContext,
                           ebml_w: &mut writer::Encoder,
                           method_ty: &ty::Method) {
    encode_def_id(ebml_w, method_ty.def_id);
    encode_name(ecx, ebml_w, method_ty.ident);
    encode_ty_type_param_defs(ebml_w, ecx,
                              method_ty.generics.type_param_defs,
                              tag_item_method_tps);
    encode_transformed_self_ty(ecx, ebml_w, method_ty.transformed_self_ty);
    encode_method_fty(ecx, ebml_w, &method_ty.fty);
    encode_visibility(ebml_w, method_ty.vis);
    encode_explicit_self(ebml_w, method_ty.explicit_self);
    let purity = method_ty.fty.purity;
    match method_ty.explicit_self {
        ast::sty_static => {
            encode_family(ebml_w, purity_static_method_family(purity));
        }
        _ => encode_family(ebml_w, purity_fn_family(purity))
    }
    encode_provided_source(ebml_w, method_ty.provided_source);
}

fn encode_info_for_method(ecx: &EncodeContext,
                          ebml_w: &mut writer::Encoder,
                          m: &ty::Method,
                          impl_path: &[ast_map::path_elt],
                          is_default_impl: bool,
                          parent_id: NodeId,
                          ast_method_opt: Option<@method>) {

    debug!("encode_info_for_method: %? %s", m.def_id,
           ecx.tcx.sess.str_of(m.ident));
    ebml_w.start_tag(tag_items_data_item);

    encode_method_ty_fields(ecx, ebml_w, m);
    encode_parent_item(ebml_w, local_def(parent_id));

    // The type for methods gets encoded twice, which is unfortunate.
    let tpt = lookup_item_type(ecx.tcx, m.def_id);
    encode_bounds_and_type(ebml_w, ecx, &tpt);

    encode_path(ecx, ebml_w, impl_path, ast_map::path_name(m.ident));

    for ast_method in ast_method_opt.iter() {
        let num_params = tpt.generics.type_param_defs.len();
        if num_params > 0u || is_default_impl
            || should_inline(ast_method.attrs) {
            (ecx.encode_inlined_item)(
                ecx, ebml_w, impl_path,
                ii_method(local_def(parent_id), false, *ast_method));
        } else {
            encode_symbol(ecx, ebml_w, m.def_id.node);
        }
    }

    ebml_w.end_tag();
}

fn purity_fn_family(p: purity) -> char {
    match p {
      unsafe_fn => 'u',
      impure_fn => 'f',
      extern_fn => 'e'
    }
}

fn purity_static_method_family(p: purity) -> char {
    match p {
      unsafe_fn => 'U',
      impure_fn => 'F',
      _ => fail!("extern fn can't be static")
    }
}


fn should_inline(attrs: &[Attribute]) -> bool {
    use syntax::attr::*;
    match find_inline_attr(attrs) {
        InlineNone | InlineNever  => false,
        InlineHint | InlineAlways => true
    }
}

// Encodes the inherent implementations of a structure, enumeration, or trait.
fn encode_inherent_implementations(ecx: &EncodeContext,
                                   ebml_w: &mut writer::Encoder,
                                   def_id: DefId) {
    match ecx.tcx.inherent_impls.find(&def_id) {
        None => {}
        Some(&implementations) => {
            for implementation in implementations.iter() {
                ebml_w.start_tag(tag_items_data_item_inherent_impl);
                encode_def_id(ebml_w, implementation.did);
                ebml_w.end_tag();
            }
        }
    }
}

// Encodes the implementations of a trait defined in this crate.
fn encode_extension_implementations(ecx: &EncodeContext,
                                    ebml_w: &mut writer::Encoder,
                                    trait_def_id: DefId) {
    match ecx.tcx.trait_impls.find(&trait_def_id) {
        None => {}
        Some(&implementations) => {
            for implementation in implementations.iter() {
                ebml_w.start_tag(tag_items_data_item_extension_impl);
                encode_def_id(ebml_w, implementation.did);
                ebml_w.end_tag();
            }
        }
    }
}

fn encode_info_for_item(ecx: &EncodeContext,
                        ebml_w: &mut writer::Encoder,
                        item: @item,
                        index: @mut ~[entry<i64>],
                        path: &[ast_map::path_elt]) {
    let tcx = ecx.tcx;

    fn add_to_index_(item: @item, ebml_w: &writer::Encoder,
                     index: @mut ~[entry<i64>]) {
        index.push(entry { val: item.id as i64, pos: ebml_w.writer.tell() });
    }
    let add_to_index: &fn() = || add_to_index_(item, ebml_w, index);

    debug!("encoding info for item at %s",
           ecx.tcx.sess.codemap.span_to_str(item.span));

    let def_id = local_def(item.id);
    match item.node {
      item_static(_, m, _) => {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, def_id);
        if m == ast::MutMutable {
            encode_family(ebml_w, 'b');
        } else {
            encode_family(ebml_w, 'c');
        }
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_symbol(ecx, ebml_w, item.id);
        encode_name(ecx, ebml_w, item.ident);
        encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
        (ecx.encode_inlined_item)(ecx, ebml_w, path, ii_item(item));
        ebml_w.end_tag();
      }
      item_fn(_, purity, _, ref generics, _) => {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, def_id);
        encode_family(ebml_w, purity_fn_family(purity));
        let tps_len = generics.ty_params.len();
        encode_bounds_and_type(ebml_w, ecx, &lookup_item_type(tcx, def_id));
        encode_name(ecx, ebml_w, item.ident);
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
        encode_info_for_mod(ecx,
                            ebml_w,
                            m,
                            item.id,
                            path,
                            item.ident,
                            item.vis);
      }
      item_foreign_mod(ref fm) => {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, def_id);
        encode_family(ebml_w, 'n');
        encode_name(ecx, ebml_w, item.ident);
        encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));

        // Encode all the items in this module.
        for foreign_item in fm.items.iter() {
            ebml_w.start_tag(tag_mod_child);
            ebml_w.wr_str(def_to_str(local_def(foreign_item.id)));
            ebml_w.end_tag();
        }

        ebml_w.end_tag();
      }
      item_ty(*) => {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, def_id);
        encode_family(ebml_w, 'y');
        encode_bounds_and_type(ebml_w, ecx, &lookup_item_type(tcx, def_id));
        encode_name(ecx, ebml_w, item.ident);
        encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
        encode_region_param(ecx, ebml_w, item);
        ebml_w.end_tag();
      }
      item_enum(ref enum_definition, ref generics) => {
        add_to_index();

        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, def_id);
        encode_family(ebml_w, 't');
        encode_bounds_and_type(ebml_w, ecx, &lookup_item_type(tcx, def_id));
        encode_name(ecx, ebml_w, item.ident);
        for v in (*enum_definition).variants.iter() {
            encode_variant_id(ebml_w, local_def(v.node.id));
        }
        (ecx.encode_inlined_item)(ecx, ebml_w, path, ii_item(item));
        encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
        encode_region_param(ecx, ebml_w, item);

        // Encode inherent implementations for this enumeration.
        encode_inherent_implementations(ecx, ebml_w, def_id);

        ebml_w.end_tag();

        encode_enum_variant_info(ecx,
                                 ebml_w,
                                 item.id,
                                 (*enum_definition).variants,
                                 path,
                                 index,
                                 generics);
      }
      item_struct(struct_def, _) => {
        /* First, encode the fields
           These come first because we need to write them to make
           the index, and the index needs to be in the item for the
           class itself */
        let idx = encode_info_for_struct(ecx, ebml_w, path,
                                         struct_def.fields, index);

        /* Index the class*/
        add_to_index();

        /* Now, make an item for the class itself */
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, def_id);
        encode_family(ebml_w, 'S');
        encode_bounds_and_type(ebml_w, ecx, &lookup_item_type(tcx, def_id));

        encode_name(ecx, ebml_w, item.ident);
        encode_attributes(ebml_w, item.attrs);
        encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
        encode_region_param(ecx, ebml_w, item);

        /* Encode def_ids for each field and method
         for methods, write all the stuff get_trait_method
        needs to know*/
        for f in struct_def.fields.iter() {
            match f.node.kind {
                named_field(ident, vis) => {
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

        // Encode inherent implementations for this structure.
        encode_inherent_implementations(ecx, ebml_w, def_id);

        /* Each class has its own index -- encode it */
        let bkts = create_index(idx);
        encode_index(ebml_w, bkts, write_i64);
        ebml_w.end_tag();

        // If this is a tuple- or enum-like struct, encode the type of the
        // constructor.
        if struct_def.fields.len() > 0 &&
                struct_def.fields[0].node.kind == ast::unnamed_field {
            let ctor_id = match struct_def.ctor_id {
                Some(ctor_id) => ctor_id,
                None => ecx.tcx.sess.bug("struct def didn't have ctor id"),
            };

            encode_info_for_struct_ctor(ecx,
                                        ebml_w,
                                        path,
                                        item.ident,
                                        ctor_id,
                                        index);
        }
      }
      item_impl(_, ref opt_trait, ref ty, ref ast_methods) => {
        // We need to encode information about the default methods we
        // have inherited, so we drive this based on the impl structure.
        let imp = tcx.impls.get(&def_id);

        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, def_id);
        encode_family(ebml_w, 'i');
        encode_region_param(ecx, ebml_w, item);
        encode_bounds_and_type(ebml_w, ecx, &lookup_item_type(tcx, def_id));
        encode_name(ecx, ebml_w, item.ident);
        encode_attributes(ebml_w, item.attrs);
        match ty.node {
            ast::ty_path(ref path, ref bounds, _) if path.segments
                                                         .len() == 1 => {
                assert!(bounds.is_none());
                encode_impl_type_basename(ecx, ebml_w,
                                          ast_util::path_to_ident(path));
            }
            _ => {}
        }
        for method in imp.methods.iter() {
            ebml_w.start_tag(tag_item_impl_method);
            let s = def_to_str(method.def_id);
            ebml_w.writer.write(s.as_bytes());
            ebml_w.end_tag();
        }
        for ast_trait_ref in opt_trait.iter() {
            let trait_ref = ty::node_id_to_trait_ref(
                tcx, ast_trait_ref.ref_id);
            encode_trait_ref(ebml_w, ecx, trait_ref, tag_item_trait_ref);
            let impl_vtables = ty::lookup_impl_vtables(tcx, def_id);
            encode_impl_vtables(ebml_w, ecx, &impl_vtables);
        }
        encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
        ebml_w.end_tag();

        // >:-<
        let mut impl_path = vec::append(~[], path);
        impl_path.push(ast_map::path_name(item.ident));

        // Iterate down the methods, emitting them. We rely on the
        // assumption that all of the actually implemented methods
        // appear first in the impl structure, in the same order they do
        // in the ast. This is a little sketchy.
        let num_implemented_methods = ast_methods.len();
        for (i, m) in imp.methods.iter().enumerate() {
            let ast_method = if i < num_implemented_methods {
                Some(ast_methods[i])
            } else { None };

            index.push(entry {val: m.def_id.node as i64,
                              pos: ebml_w.writer.tell()});
            encode_info_for_method(ecx,
                                   ebml_w,
                                   *m,
                                   impl_path,
                                   false,
                                   item.id,
                                   ast_method)
        }
      }
      item_trait(_, ref super_traits, ref ms) => {
        add_to_index();
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, def_id);
        encode_family(ebml_w, 'I');
        encode_region_param(ecx, ebml_w, item);
        let trait_def = ty::lookup_trait_def(tcx, def_id);
        encode_ty_type_param_defs(ebml_w, ecx,
                                  trait_def.generics.type_param_defs,
                                  tag_items_data_item_ty_param_bounds);
        encode_trait_ref(ebml_w, ecx, trait_def.trait_ref, tag_item_trait_ref);
        encode_name(ecx, ebml_w, item.ident);
        encode_attributes(ebml_w, item.attrs);
        for &method_def_id in ty::trait_method_def_ids(tcx, def_id).iter() {
            ebml_w.start_tag(tag_item_trait_method);
            encode_def_id(ebml_w, method_def_id);
            ebml_w.end_tag();

            ebml_w.start_tag(tag_mod_child);
            ebml_w.wr_str(def_to_str(method_def_id));
            ebml_w.end_tag();
        }
        encode_path(ecx, ebml_w, path, ast_map::path_name(item.ident));
        // FIXME(#8559): This should use the tcx's supertrait cache instead of
        // reading the AST's list, because the former has already filtered out
        // the builtin-kinds-as-supertraits. See corresponding fixme in decoder.
        for ast_trait_ref in super_traits.iter() {
            let trait_ref = ty::node_id_to_trait_ref(ecx.tcx, ast_trait_ref.ref_id);
            encode_trait_ref(ebml_w, ecx, trait_ref, tag_item_super_trait_ref);
        }

        // Encode the implementations of this trait.
        encode_extension_implementations(ecx, ebml_w, def_id);

        ebml_w.end_tag();

        // Now output the method info for each method.
        let r = ty::trait_method_def_ids(tcx, def_id);
        for (i, &method_def_id) in r.iter().enumerate() {
            assert_eq!(method_def_id.crate, ast::LOCAL_CRATE);

            let method_ty = ty::method(tcx, method_def_id);

            index.push(entry {val: method_def_id.node as i64,
                              pos: ebml_w.writer.tell()});

            ebml_w.start_tag(tag_items_data_item);

            encode_method_ty_fields(ecx, ebml_w, method_ty);

            encode_parent_item(ebml_w, def_id);

            let mut trait_path = vec::append(~[], path);
            trait_path.push(ast_map::path_name(item.ident));
            encode_path(ecx, ebml_w, trait_path, ast_map::path_name(method_ty.ident));

            match method_ty.explicit_self {
                sty_static => {
                    encode_family(ebml_w,
                                  purity_static_method_family(
                                      method_ty.fty.purity));

                    let tpt = ty::lookup_item_type(tcx, method_def_id);
                    encode_bounds_and_type(ebml_w, ecx, &tpt);
                }

                _ => {
                    encode_family(ebml_w,
                                  purity_fn_family(
                                      method_ty.fty.purity));
                }
            }

            match ms[i] {
                required(_) => {
                    encode_method_sort(ebml_w, 'r');
                }

                provided(m) => {
                    // If this is a static method, we've already encoded
                    // this.
                    if method_ty.explicit_self != sty_static {
                        // XXX: I feel like there is something funny going on.
                        let tpt = ty::lookup_item_type(tcx, method_def_id);
                        encode_bounds_and_type(ebml_w, ecx, &tpt);
                    }
                    encode_method_sort(ebml_w, 'p');
                    (ecx.encode_inlined_item)(
                        ecx, ebml_w, path,
                        ii_method(def_id, true, m));
                }
            }

            ebml_w.end_tag();
        }

        // Encode inherent implementations for this trait.
        encode_inherent_implementations(ecx, ebml_w, def_id);
      }
      item_mac(*) => fail!("item macros unimplemented")
    }
}

fn encode_info_for_foreign_item(ecx: &EncodeContext,
                                ebml_w: &mut writer::Encoder,
                                nitem: @foreign_item,
                                index: @mut ~[entry<i64>],
                                path: &ast_map::path,
                                abi: AbiSet) {
    index.push(entry { val: nitem.id as i64, pos: ebml_w.writer.tell() });

    ebml_w.start_tag(tag_items_data_item);
    match nitem.node {
      foreign_item_fn(*) => {
        encode_def_id(ebml_w, local_def(nitem.id));
        encode_family(ebml_w, purity_fn_family(impure_fn));
        encode_bounds_and_type(ebml_w, ecx,
                               &lookup_item_type(ecx.tcx,local_def(nitem.id)));
        encode_name(ecx, ebml_w, nitem.ident);
        if abi.is_intrinsic() {
            (ecx.encode_inlined_item)(ecx, ebml_w, *path, ii_foreign(nitem));
        } else {
            encode_symbol(ecx, ebml_w, nitem.id);
        }
        encode_path(ecx, ebml_w, *path, ast_map::path_name(nitem.ident));
      }
      foreign_item_static(_, mutbl) => {
        encode_def_id(ebml_w, local_def(nitem.id));
        if mutbl {
            encode_family(ebml_w, 'b');
        } else {
            encode_family(ebml_w, 'c');
        }
        encode_type(ecx, ebml_w, node_id_to_type(ecx.tcx, nitem.id));
        encode_symbol(ecx, ebml_w, nitem.id);
        encode_name(ecx, ebml_w, nitem.ident);
        encode_path(ecx, ebml_w, *path, ast_map::path_name(nitem.ident));
      }
    }
    ebml_w.end_tag();
}

fn my_visit_expr(_e:@Expr) { }

fn my_visit_item(i:@item, items: ast_map::map, ebml_w:&writer::Encoder,
                 ecx_ptr:*int, index: @mut ~[entry<i64>]) {
    match items.get_copy(&i.id) {
        ast_map::node_item(_, pt) => {
            let mut ebml_w = ebml_w.clone();
            // See above
            let ecx : &EncodeContext = unsafe { cast::transmute(ecx_ptr) };
            encode_info_for_item(ecx, &mut ebml_w, i, index, *pt);
        }
        _ => fail!("bad item")
    }
}

fn my_visit_foreign_item(ni:@foreign_item, items: ast_map::map, ebml_w:&writer::Encoder,
                         ecx_ptr:*int, index: @mut ~[entry<i64>]) {
    match items.get_copy(&ni.id) {
        ast_map::node_foreign_item(_, abi, _, pt) => {
            debug!("writing foreign item %s::%s",
                   ast_map::path_to_str(
                       *pt,
                       token::get_ident_interner()),
                   token::ident_to_str(&ni.ident));

            let mut ebml_w = ebml_w.clone();
            // See above
            let ecx : &EncodeContext = unsafe { cast::transmute(ecx_ptr) };
            encode_info_for_foreign_item(ecx,
                                         &mut ebml_w,
                                         ni,
                                         index,
                                         pt,
                                         abi);
        }
        // case for separate item and foreign-item tables
        _ => fail!("bad foreign item")
    }
}

struct EncodeVisitor {
    ebml_w_for_visit_item: writer::Encoder,
    ebml_w_for_visit_foreign_item: writer::Encoder,
    ecx_ptr:*int,
    items: ast_map::map,
    index: @mut ~[entry<i64>],
}

impl visit::Visitor<()> for EncodeVisitor {
    fn visit_expr(&mut self, ex:@Expr, _:()) {
        visit::walk_expr(self, ex, ());
        my_visit_expr(ex);
    }
    fn visit_item(&mut self, i:@item, _:()) {
        visit::walk_item(self, i, ());
        my_visit_item(i,
                      self.items,
                      &self.ebml_w_for_visit_item,
                      self.ecx_ptr,
                      self.index);
    }
    fn visit_foreign_item(&mut self, ni:@foreign_item, _:()) {
        visit::walk_foreign_item(self, ni, ());
        my_visit_foreign_item(ni,
                              self.items,
                              &self.ebml_w_for_visit_foreign_item,
                              self.ecx_ptr,
                              self.index);
    }
}

fn encode_info_for_items(ecx: &EncodeContext,
                         ebml_w: &mut writer::Encoder,
                         crate: &Crate)
                         -> ~[entry<i64>] {
    let index = @mut ~[];
    ebml_w.start_tag(tag_items_data);
    index.push(entry { val: CRATE_NODE_ID as i64, pos: ebml_w.writer.tell() });
    encode_info_for_mod(ecx,
                        ebml_w,
                        &crate.module,
                        CRATE_NODE_ID,
                        [],
                        syntax::parse::token::special_idents::invalid,
                        public);
    let items = ecx.tcx.items;

    // See comment in `encode_side_tables_for_ii` in astencode
    let ecx_ptr : *int = unsafe { cast::transmute(ecx) };
    let mut visitor = EncodeVisitor {
        index: index,
        items: items,
        ecx_ptr: ecx_ptr,
        ebml_w_for_visit_item: (*ebml_w).clone(),
        ebml_w_for_visit_foreign_item: (*ebml_w).clone(),
    };

    visit::walk_crate(&mut visitor, crate, ());

    ebml_w.end_tag();
    return /*bad*/(*index).clone();
}


// Path and definition ID indexing

fn create_index<T:Clone + Hash + IterBytes + 'static>(
                index: ~[entry<T>])
                -> ~[@~[entry<T>]] {
    let mut buckets: ~[@mut ~[entry<T>]] = ~[];
    for _ in range(0u, 256u) { buckets.push(@mut ~[]); };
    for elt in index.iter() {
        let h = elt.val.hash() as uint;
        buckets[h % 256].push((*elt).clone());
    }

    let mut buckets_frozen = ~[];
    for bucket in buckets.iter() {
        buckets_frozen.push(@/*bad*/(**bucket).clone());
    }
    return buckets_frozen;
}

fn encode_index<T:'static>(
                ebml_w: &mut writer::Encoder,
                buckets: ~[@~[entry<T>]],
                write_fn: &fn(@io::Writer, &T)) {
    let writer = ebml_w.writer;
    ebml_w.start_tag(tag_index);
    let mut bucket_locs: ~[uint] = ~[];
    ebml_w.start_tag(tag_index_buckets);
    for bucket in buckets.iter() {
        bucket_locs.push(ebml_w.writer.tell());
        ebml_w.start_tag(tag_index_buckets_bucket);
        for elt in (**bucket).iter() {
            ebml_w.start_tag(tag_index_buckets_bucket_elt);
            assert!(elt.pos < 0xffff_ffff);
            writer.write_be_u32(elt.pos as u32);
            write_fn(writer, &elt.val);
            ebml_w.end_tag();
        }
        ebml_w.end_tag();
    }
    ebml_w.end_tag();
    ebml_w.start_tag(tag_index_table);
    for pos in bucket_locs.iter() {
        assert!(*pos < 0xffff_ffff);
        writer.write_be_u32(*pos as u32);
    }
    ebml_w.end_tag();
    ebml_w.end_tag();
}

fn write_str(writer: @io::Writer, s: ~str) {
    writer.write_str(s);
}

fn write_i64(writer: @io::Writer, &n: &i64) {
    assert!(n < 0x7fff_ffff);
    writer.write_be_u32(n as u32);
}

fn encode_meta_item(ebml_w: &mut writer::Encoder, mi: @MetaItem) {
    match mi.node {
      MetaWord(name) => {
        ebml_w.start_tag(tag_meta_item_word);
        ebml_w.start_tag(tag_meta_item_name);
        ebml_w.writer.write(name.as_bytes());
        ebml_w.end_tag();
        ebml_w.end_tag();
      }
      MetaNameValue(name, value) => {
        match value.node {
          lit_str(value) => {
            ebml_w.start_tag(tag_meta_item_name_value);
            ebml_w.start_tag(tag_meta_item_name);
            ebml_w.writer.write(name.as_bytes());
            ebml_w.end_tag();
            ebml_w.start_tag(tag_meta_item_value);
            ebml_w.writer.write(value.as_bytes());
            ebml_w.end_tag();
            ebml_w.end_tag();
          }
          _ => {/* FIXME (#623): encode other variants */ }
        }
      }
      MetaList(name, ref items) => {
        ebml_w.start_tag(tag_meta_item_list);
        ebml_w.start_tag(tag_meta_item_name);
        ebml_w.writer.write(name.as_bytes());
        ebml_w.end_tag();
        for inner_item in items.iter() {
            encode_meta_item(ebml_w, *inner_item);
        }
        ebml_w.end_tag();
      }
    }
}

fn encode_attributes(ebml_w: &mut writer::Encoder, attrs: &[Attribute]) {
    ebml_w.start_tag(tag_attributes);
    for attr in attrs.iter() {
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
fn synthesize_crate_attrs(ecx: &EncodeContext,
                          crate: &Crate) -> ~[Attribute] {

    fn synthesize_link_attr(ecx: &EncodeContext, items: ~[@MetaItem]) ->
       Attribute {

        assert!(!ecx.link_meta.name.is_empty());
        assert!(!ecx.link_meta.vers.is_empty());

        let name_item =
            attr::mk_name_value_item_str(@"name",
                                         ecx.link_meta.name);
        let vers_item =
            attr::mk_name_value_item_str(@"vers",
                                         ecx.link_meta.vers);

        let mut meta_items = ~[name_item, vers_item];

        for &mi in items.iter().filter(|mi| "name" != mi.name() && "vers" != mi.name()) {
            meta_items.push(mi);
        }
        let link_item = attr::mk_list_item(@"link", meta_items);

        return attr::mk_attr(link_item);
    }

    let mut attrs = ~[];
    let mut found_link_attr = false;
    for attr in crate.attrs.iter() {
        attrs.push(
            if "link" != attr.name()  {
                *attr
            } else {
                match attr.meta_item_list() {
                  Some(l) => {
                    found_link_attr = true;;
                    synthesize_link_attr(ecx, l.to_owned())
                  }
                  _ => *attr
                }
            });
    }

    if !found_link_attr { attrs.push(synthesize_link_attr(ecx, ~[])); }

    return attrs;
}

fn encode_crate_deps(ecx: &EncodeContext,
                     ebml_w: &mut writer::Encoder,
                     cstore: &cstore::CStore) {
    fn get_ordered_deps(ecx: &EncodeContext, cstore: &cstore::CStore)
                     -> ~[decoder::CrateDep] {
        type numdep = decoder::CrateDep;

        // Pull the cnums and name,vers,hash out of cstore
        let mut deps = ~[];
        do cstore::iter_crate_data(cstore) |key, val| {
            let dep = decoder::CrateDep {cnum: key,
                       name: ecx.tcx.sess.ident_of(val.name),
                       vers: decoder::get_crate_vers(val.data),
                       hash: decoder::get_crate_hash(val.data)};
            deps.push(dep);
        };

        // Sort by cnum
        extra::sort::quick_sort(deps, |kv1, kv2| kv1.cnum <= kv2.cnum);

        // Sanity-check the crate numbers
        let mut expected_cnum = 1;
        for n in deps.iter() {
            assert_eq!(n.cnum, expected_cnum);
            expected_cnum += 1;
        }

        deps
    }

    // We're just going to write a list of crate 'name-hash-version's, with
    // the assumption that they are numbered 1 to n.
    // FIXME (#2166): This is not nearly enough to support correct versioning
    // but is enough to get transitive crate dependencies working.
    ebml_w.start_tag(tag_crate_deps);
    let r = get_ordered_deps(ecx, cstore);
    for dep in r.iter() {
        encode_crate_dep(ecx, ebml_w, *dep);
    }
    ebml_w.end_tag();
}

fn encode_lang_items(ecx: &EncodeContext, ebml_w: &mut writer::Encoder) {
    ebml_w.start_tag(tag_lang_items);

    do ecx.tcx.lang_items.each_item |def_id, i| {
        for id in def_id.iter() {
            if id.crate == LOCAL_CRATE {
                ebml_w.start_tag(tag_lang_items_item);

                ebml_w.start_tag(tag_lang_items_item_id);
                ebml_w.writer.write_be_u32(i as u32);
                ebml_w.end_tag();   // tag_lang_items_item_id

                ebml_w.start_tag(tag_lang_items_item_node_id);
                ebml_w.writer.write_be_u32(id.node as u32);
                ebml_w.end_tag();   // tag_lang_items_item_node_id

                ebml_w.end_tag();   // tag_lang_items_item
            }
        }
        true
    };

    ebml_w.end_tag();   // tag_lang_items
}

fn encode_link_args(ecx: &EncodeContext, ebml_w: &mut writer::Encoder) {
    ebml_w.start_tag(tag_link_args);

    let link_args = cstore::get_used_link_args(ecx.cstore);
    for link_arg in link_args.iter() {
        ebml_w.start_tag(tag_link_args_arg);
        ebml_w.writer.write_str(link_arg.to_str());
        ebml_w.end_tag();
    }

    ebml_w.end_tag();
}

struct ImplVisitor<'self> {
    ecx: &'self EncodeContext<'self>,
    ebml_w: &'self mut writer::Encoder,
}

impl<'self> Visitor<()> for ImplVisitor<'self> {
    fn visit_item(&mut self, item: @item, _: ()) {
        match item.node {
            item_impl(_, Some(ref trait_ref), _, _) => {
                let def_map = self.ecx.tcx.def_map;
                let trait_def = def_map.get_copy(&trait_ref.ref_id);
                let def_id = ast_util::def_id_of_def(trait_def);

                // Load eagerly if this is an implementation of the Drop trait
                // or if the trait is not defined in this crate.
                if def_id == self.ecx.tcx.lang_items.drop_trait().unwrap() ||
                        def_id.crate != LOCAL_CRATE {
                    self.ebml_w.start_tag(tag_impls_impl);
                    encode_def_id(self.ebml_w, local_def(item.id));
                    self.ebml_w.end_tag();
                }
            }
            _ => {}
        }
        visit::walk_item(self, item, ());
    }
}

/// Encodes implementations that are eagerly loaded.
///
/// None of this is necessary in theory; we can load all implementations
/// lazily. However, in two cases the optimizations to lazily load
/// implementations are not yet implemented. These two cases, which require us
/// to load implementations eagerly, are:
///
/// * Destructors (implementations of the Drop trait).
///
/// * Implementations of traits not defined in this crate.
fn encode_impls(ecx: &EncodeContext,
                crate: &Crate,
                ebml_w: &mut writer::Encoder) {
    ebml_w.start_tag(tag_impls);

    {
        let mut visitor = ImplVisitor {
            ecx: ecx,
            ebml_w: ebml_w,
        };
        visit::walk_crate(&mut visitor, crate, ());
    }

    ebml_w.end_tag();
}

fn encode_misc_info(ecx: &EncodeContext,
                    crate: &Crate,
                    ebml_w: &mut writer::Encoder) {
    ebml_w.start_tag(tag_misc_info);
    ebml_w.start_tag(tag_misc_info_crate_items);
    for &item in crate.module.items.iter() {
        ebml_w.start_tag(tag_mod_child);
        ebml_w.wr_str(def_to_str(local_def(item.id)));
        ebml_w.end_tag();

        do each_auxiliary_node_id(item) |auxiliary_node_id| {
            ebml_w.start_tag(tag_mod_child);
            ebml_w.wr_str(def_to_str(local_def(auxiliary_node_id)));
            ebml_w.end_tag();
            true
        };
    }

    // Encode reexports for the root module.
    encode_reexports(ecx, ebml_w, 0, []);

    ebml_w.end_tag();
    ebml_w.end_tag();
}

fn encode_crate_dep(ecx: &EncodeContext,
                    ebml_w: &mut writer::Encoder,
                    dep: decoder::CrateDep) {
    ebml_w.start_tag(tag_crate_dep);
    ebml_w.start_tag(tag_crate_dep_name);
    let s = ecx.tcx.sess.str_of(dep.name);
    ebml_w.writer.write(s.as_bytes());
    ebml_w.end_tag();
    ebml_w.start_tag(tag_crate_dep_vers);
    ebml_w.writer.write(dep.vers.as_bytes());
    ebml_w.end_tag();
    ebml_w.start_tag(tag_crate_dep_hash);
    ebml_w.writer.write(dep.hash.as_bytes());
    ebml_w.end_tag();
    ebml_w.end_tag();
}

fn encode_hash(ebml_w: &mut writer::Encoder, hash: &str) {
    ebml_w.start_tag(tag_crate_hash);
    ebml_w.writer.write(hash.as_bytes());
    ebml_w.end_tag();
}

// NB: Increment this as you change the metadata encoding version.
pub static metadata_encoding_version : &'static [u8] =
    &[0x72, //'r' as u8,
      0x75, //'u' as u8,
      0x73, //'s' as u8,
      0x74, //'t' as u8,
      0, 0, 0, 1 ];

pub fn encode_metadata(parms: EncodeParams, crate: &Crate) -> ~[u8] {
    let wr = @io::BytesWriter::new();
    let stats = Stats {
        inline_bytes: 0,
        attr_bytes: 0,
        dep_bytes: 0,
        lang_item_bytes: 0,
        link_args_bytes: 0,
        impl_bytes: 0,
        misc_bytes: 0,
        item_bytes: 0,
        index_bytes: 0,
        zero_bytes: 0,
        total_bytes: 0,
        n_inlines: 0
    };
    let EncodeParams {
        item_symbols,
        diag,
        tcx,
        reexports2,
        discrim_symbols,
        cstore,
        encode_inlined_item,
        link_meta,
        reachable,
        _
    } = parms;
    let type_abbrevs = @mut HashMap::new();
    let stats = @mut stats;
    let ecx = EncodeContext {
        diag: diag,
        tcx: tcx,
        stats: stats,
        reexports2: reexports2,
        item_symbols: item_symbols,
        discrim_symbols: discrim_symbols,
        link_meta: link_meta,
        cstore: cstore,
        encode_inlined_item: encode_inlined_item,
        type_abbrevs: type_abbrevs,
        reachable: reachable,
     };

    let mut ebml_w = writer::Encoder(wr as @io::Writer);

    encode_hash(&mut ebml_w, ecx.link_meta.extras_hash);

    let mut i = *wr.pos;
    let crate_attrs = synthesize_crate_attrs(&ecx, crate);
    encode_attributes(&mut ebml_w, crate_attrs);
    ecx.stats.attr_bytes = *wr.pos - i;

    i = *wr.pos;
    encode_crate_deps(&ecx, &mut ebml_w, ecx.cstore);
    ecx.stats.dep_bytes = *wr.pos - i;

    // Encode the language items.
    i = *wr.pos;
    encode_lang_items(&ecx, &mut ebml_w);
    ecx.stats.lang_item_bytes = *wr.pos - i;

    // Encode the link args.
    i = *wr.pos;
    encode_link_args(&ecx, &mut ebml_w);
    ecx.stats.link_args_bytes = *wr.pos - i;

    // Encode the def IDs of impls, for coherence checking.
    i = *wr.pos;
    encode_impls(&ecx, crate, &mut ebml_w);
    ecx.stats.impl_bytes = *wr.pos - i;

    // Encode miscellaneous info.
    i = *wr.pos;
    encode_misc_info(&ecx, crate, &mut ebml_w);
    ecx.stats.misc_bytes = *wr.pos - i;

    // Encode and index the items.
    ebml_w.start_tag(tag_items);
    i = *wr.pos;
    let items_index = encode_info_for_items(&ecx, &mut ebml_w, crate);
    ecx.stats.item_bytes = *wr.pos - i;

    i = *wr.pos;
    let items_buckets = create_index(items_index);
    encode_index(&mut ebml_w, items_buckets, write_i64);
    ecx.stats.index_bytes = *wr.pos - i;
    ebml_w.end_tag();

    ecx.stats.total_bytes = *wr.pos;

    if (tcx.sess.meta_stats()) {
        for e in wr.bytes.iter() {
            if *e == 0 {
                ecx.stats.zero_bytes += 1;
            }
        }

        io::println("metadata stats:");
        printfln!("    inline bytes: %u", ecx.stats.inline_bytes);
        printfln!(" attribute bytes: %u", ecx.stats.attr_bytes);
        printfln!("       dep bytes: %u", ecx.stats.dep_bytes);
        printfln!(" lang item bytes: %u", ecx.stats.lang_item_bytes);
        printfln!(" link args bytes: %u", ecx.stats.link_args_bytes);
        printfln!("      impl bytes: %u", ecx.stats.impl_bytes);
        printfln!("      misc bytes: %u", ecx.stats.misc_bytes);
        printfln!("      item bytes: %u", ecx.stats.item_bytes);
        printfln!("     index bytes: %u", ecx.stats.index_bytes);
        printfln!("      zero bytes: %u", ecx.stats.zero_bytes);
        printfln!("     total bytes: %u", ecx.stats.total_bytes);
    }

    // Pad this, since something (LLVM, presumably) is cutting off the
    // remaining % 4 bytes.
    wr.write(&[0u8, 0u8, 0u8, 0u8]);

    let writer_bytes: &mut ~[u8] = wr.bytes;

    metadata_encoding_version.to_owned() +
        flate::deflate_bytes(*writer_bytes)
}

// Get the encoded string for a type
pub fn encoded_ty(tcx: ty::ctxt, t: ty::t) -> ~str {
    let cx = @tyencode::ctxt {
        diag: tcx.diag,
        ds: def_to_str,
        tcx: tcx,
        abbrevs: tyencode::ac_no_abbrevs};
    do io::with_str_writer |wr| {
        tyencode::enc_ty(wr, cx, t);
    }
}
