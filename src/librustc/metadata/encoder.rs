// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Metadata encoding

#[allow(unused_must_use)]; // everything is just a MemWriter, can't fail
#[allow(non_camel_case_types)];

use back::svh::Svh;
use metadata::common::*;
use metadata::cstore;
use metadata::decoder;
use metadata::tyencode;
use middle::ty::{node_id_to_type, lookup_item_type};
use middle::astencode;
use middle::ty;
use middle::typeck;
use middle;
use util::nodemap::{NodeMap, NodeSet};

use serialize::Encodable;
use std::cast;
use std::cell::{Cell, RefCell};
use std::hash;
use std::hash::Hash;
use std::io::MemWriter;
use std::str;
use std::vec_ng::Vec;
use collections::HashMap;
use syntax::abi::AbiSet;
use syntax::ast::*;
use syntax::ast;
use syntax::ast_map::{PathElem, PathElems};
use syntax::ast_map;
use syntax::ast_util::*;
use syntax::ast_util;
use syntax::attr::AttrMetaMethods;
use syntax::attr;
use syntax::crateid::CrateId;
use syntax::diagnostic::SpanHandler;
use syntax::parse::token::InternedString;
use syntax::parse::token::special_idents;
use syntax::parse::token;
use syntax::visit::Visitor;
use syntax::visit;
use syntax;
use writer = serialize::ebml::writer;

// used by astencode:
pub type abbrev_map = @RefCell<HashMap<ty::t, tyencode::ty_abbrev>>;

/// A borrowed version of ast::InlinedItem.
pub enum InlinedItemRef<'a> {
    IIItemRef(&'a ast::Item),
    IIMethodRef(ast::DefId, bool, &'a ast::Method),
    IIForeignRef(&'a ast::ForeignItem)
}

pub type EncodeInlinedItem<'a> = 'a |ecx: &EncodeContext,
                                     ebml_w: &mut writer::Encoder,
                                     ii: InlinedItemRef|;

pub struct EncodeParams<'a> {
    diag: &'a SpanHandler,
    tcx: &'a ty::ctxt,
    reexports2: middle::resolve::ExportMap2,
    item_symbols: &'a RefCell<NodeMap<~str>>,
    non_inlineable_statics: &'a RefCell<NodeSet>,
    link_meta: &'a LinkMeta,
    cstore: &'a cstore::CStore,
    encode_inlined_item: EncodeInlinedItem<'a>,
}

pub struct Stats {
    inline_bytes: Cell<u64>,
    attr_bytes: Cell<u64>,
    dep_bytes: Cell<u64>,
    lang_item_bytes: Cell<u64>,
    native_lib_bytes: Cell<u64>,
    macro_registrar_fn_bytes: Cell<u64>,
    macro_defs_bytes: Cell<u64>,
    impl_bytes: Cell<u64>,
    misc_bytes: Cell<u64>,
    item_bytes: Cell<u64>,
    index_bytes: Cell<u64>,
    zero_bytes: Cell<u64>,
    total_bytes: Cell<u64>,
}

pub struct EncodeContext<'a> {
    diag: &'a SpanHandler,
    tcx: &'a ty::ctxt,
    stats: @Stats,
    reexports2: middle::resolve::ExportMap2,
    item_symbols: &'a RefCell<NodeMap<~str>>,
    non_inlineable_statics: &'a RefCell<NodeSet>,
    link_meta: &'a LinkMeta,
    cstore: &'a cstore::CStore,
    encode_inlined_item: EncodeInlinedItem<'a>,
    type_abbrevs: abbrev_map,
}

fn encode_name(ebml_w: &mut writer::Encoder, name: Name) {
    ebml_w.wr_tagged_str(tag_paths_data_name, token::get_name(name).get());
}

fn encode_impl_type_basename(ebml_w: &mut writer::Encoder, name: Ident) {
    ebml_w.wr_tagged_str(tag_item_impl_type_basename, token::get_ident(name).get());
}

pub fn encode_def_id(ebml_w: &mut writer::Encoder, id: DefId) {
    ebml_w.wr_tagged_str(tag_def_id, def_to_str(id));
}

#[deriving(Clone)]
struct entry<T> {
    val: T,
    pos: u64
}

fn encode_trait_ref(ebml_w: &mut writer::Encoder,
                    ecx: &EncodeContext,
                    trait_ref: &ty::TraitRef,
                    tag: uint) {
    let ty_str_ctxt = &tyencode::ctxt {
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
    format!("{}:{}", did.krate, did.node)
}

fn encode_ty_type_param_defs(ebml_w: &mut writer::Encoder,
                             ecx: &EncodeContext,
                             params: &[ty::TypeParameterDef],
                             tag: uint) {
    let ty_str_ctxt = &tyencode::ctxt {
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

fn encode_region_param_defs(ebml_w: &mut writer::Encoder,
                            params: &[ty::RegionParameterDef]) {
    for param in params.iter() {
        ebml_w.start_tag(tag_region_param_def);

        ebml_w.start_tag(tag_region_param_def_ident);
        encode_name(ebml_w, param.name);
        ebml_w.end_tag();

        ebml_w.wr_tagged_str(tag_region_param_def_def_id,
                             def_to_str(param.def_id));

        ebml_w.end_tag();
    }
}

fn encode_item_variances(ebml_w: &mut writer::Encoder,
                         ecx: &EncodeContext,
                         id: ast::NodeId) {
    let v = ty::item_variances(ecx.tcx, ast_util::local_def(id));
    ebml_w.start_tag(tag_item_variances);
    v.encode(ebml_w);
    ebml_w.end_tag();
}

fn encode_bounds_and_type(ebml_w: &mut writer::Encoder,
                          ecx: &EncodeContext,
                          tpt: &ty::ty_param_bounds_and_ty) {
    encode_ty_type_param_defs(ebml_w, ecx, tpt.generics.type_param_defs(),
                              tag_items_data_item_ty_param_bounds);
    encode_region_param_defs(ebml_w, tpt.generics.region_param_defs());
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
    let ty_str_ctxt = &tyencode::ctxt {
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
    let ty_str_ctxt = &tyencode::ctxt {
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

fn encode_method_fty(ecx: &EncodeContext,
                     ebml_w: &mut writer::Encoder,
                     typ: &ty::BareFnTy) {
    ebml_w.start_tag(tag_item_method_fty);

    let ty_str_ctxt = &tyencode::ctxt {
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
    let item_symbols = ecx.item_symbols.borrow();
    match item_symbols.get().find(&id) {
        Some(x) => {
            debug!("encode_symbol(id={:?}, str={})", id, *x);
            ebml_w.writer.write(x.as_bytes());
        }
        None => {
            ecx.diag.handler().bug(
                format!("encode_symbol: id not found {}", id));
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

fn encode_struct_fields(ebml_w: &mut writer::Encoder,
                        def: @StructDef) {
    for f in def.fields.iter() {
        match f.node.kind {
            NamedField(ident, vis) => {
               ebml_w.start_tag(tag_item_field);
               encode_struct_field_family(ebml_w, vis);
               encode_name(ebml_w, ident.name);
               encode_def_id(ebml_w, local_def(f.node.id));
               ebml_w.end_tag();
            }
            UnnamedField => {
                ebml_w.start_tag(tag_item_unnamed_field);
                encode_def_id(ebml_w, local_def(f.node.id));
                ebml_w.end_tag();
            }
        }
    }
}

fn encode_enum_variant_info(ecx: &EncodeContext,
                            ebml_w: &mut writer::Encoder,
                            id: NodeId,
                            variants: &[P<Variant>],
                            index: @RefCell<Vec<entry<i64>> >,
                            generics: &ast::Generics) {
    debug!("encode_enum_variant_info(id={:?})", id);

    let mut disr_val = 0;
    let mut i = 0;
    let vi = ty::enum_variants(ecx.tcx,
                               ast::DefId { krate: LOCAL_CRATE, node: id });
    for variant in variants.iter() {
        let def_id = local_def(variant.node.id);
        {
            let mut index = index.borrow_mut();
            index.get().push(entry {
                val: variant.node.id as i64,
                pos: ebml_w.writer.tell().unwrap(),
            });
        }
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, def_id);
        match variant.node.kind {
            ast::TupleVariantKind(_) => encode_family(ebml_w, 'v'),
            ast::StructVariantKind(_) => encode_family(ebml_w, 'V')
        }
        encode_name(ebml_w, variant.node.name.name);
        encode_parent_item(ebml_w, local_def(id));
        encode_visibility(ebml_w, variant.node.vis);
        encode_attributes(ebml_w, variant.node.attrs.as_slice());
        match variant.node.kind {
            ast::TupleVariantKind(ref args)
                    if args.len() > 0 && generics.ty_params.len() == 0 => {
                encode_symbol(ecx, ebml_w, variant.node.id);
            }
            ast::TupleVariantKind(_) => {},
            ast::StructVariantKind(def) => {
                let idx = encode_info_for_struct(ecx,
                                                 ebml_w,
                                                 def.fields.as_slice(),
                                                 index);
                encode_struct_fields(ebml_w, def);
                let bkts = create_index(idx);
                encode_index(ebml_w, bkts, write_i64);
            }
        }
        if vi.get(i).disr_val != disr_val {
            encode_disr_val(ecx, ebml_w, vi.get(i).disr_val);
            disr_val = vi.get(i).disr_val;
        }
        encode_bounds_and_type(ebml_w, ecx,
                               &lookup_item_type(ecx.tcx, def_id));

        ecx.tcx.map.with_path(variant.node.id, |path| encode_path(ebml_w, path));
        ebml_w.end_tag();
        disr_val += 1;
        i += 1;
    }
}

fn encode_path<PI: Iterator<PathElem> + Clone>(ebml_w: &mut writer::Encoder,
                                               mut path: PI) {
    ebml_w.start_tag(tag_path);
    ebml_w.wr_tagged_u32(tag_path_len, path.clone().len() as u32);
    for pe in path {
        let tag = match pe {
            ast_map::PathMod(_) => tag_path_elem_mod,
            ast_map::PathName(_) => tag_path_elem_name
        };
        ebml_w.wr_tagged_str(tag, token::get_name(pe.name()).get());
    }
    ebml_w.end_tag();
}

fn encode_reexported_static_method(ebml_w: &mut writer::Encoder,
                                   exp: &middle::resolve::Export2,
                                   method_def_id: DefId,
                                   method_ident: Ident) {
    debug!("(encode reexported static method) {}::{}",
            exp.name, token::get_ident(method_ident));
    ebml_w.start_tag(tag_items_data_item_reexport);
    ebml_w.start_tag(tag_items_data_item_reexport_def_id);
    ebml_w.wr_str(def_to_str(method_def_id));
    ebml_w.end_tag();
    ebml_w.start_tag(tag_items_data_item_reexport_name);
    ebml_w.wr_str(format!("{}::{}", exp.name, token::get_ident(method_ident)));
    ebml_w.end_tag();
    ebml_w.end_tag();
}

fn encode_reexported_static_base_methods(ecx: &EncodeContext,
                                         ebml_w: &mut writer::Encoder,
                                         exp: &middle::resolve::Export2)
                                         -> bool {
    let inherent_impls = ecx.tcx.inherent_impls.borrow();
    match inherent_impls.get().find(&exp.def_id) {
        Some(implementations) => {
            let implementations = implementations.borrow();
            for &base_impl in implementations.get().iter() {
                for &m in base_impl.methods.iter() {
                    if m.explicit_self == ast::SelfStatic {
                        encode_reexported_static_method(ebml_w, exp, m.def_id, m.ident);
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
    let trait_methods_cache = ecx.tcx.trait_methods_cache.borrow();
    match trait_methods_cache.get().find(&exp.def_id) {
        Some(methods) => {
            for &m in methods.iter() {
                if m.explicit_self == ast::SelfStatic {
                    encode_reexported_static_method(ebml_w, exp, m.def_id, m.ident);
                }
            }

            true
        }
        None => { false }
    }
}

fn encode_reexported_static_methods(ecx: &EncodeContext,
                                    ebml_w: &mut writer::Encoder,
                                    mod_path: PathElems,
                                    exp: &middle::resolve::Export2) {
    match ecx.tcx.map.find(exp.def_id.node) {
        Some(ast_map::NodeItem(item)) => {
            let original_name = token::get_ident(item.ident);

            let path_differs = ecx.tcx.map.with_path(exp.def_id.node, |path| {
                let (mut a, mut b) = (path, mod_path.clone());
                loop {
                    match (a.next(), b.next()) {
                        (None, None) => return true,
                        (None, _) | (_, None) => return false,
                        (Some(x), Some(y)) => if x != y { return false },
                    }
                }
            });

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
            if path_differs || original_name.get() != exp.name {
                if !encode_reexported_static_base_methods(ecx, ebml_w, exp) {
                    if encode_reexported_static_trait_methods(ecx, ebml_w, exp) {
                        debug!("(encode reexported static methods) {} \
                                 [trait]",
                                original_name);
                    }
                }
                else {
                    debug!("(encode reexported static methods) {} [base]",
                            original_name);
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
fn each_auxiliary_node_id(item: @Item, callback: |NodeId| -> bool) -> bool {
    let mut continue_ = true;
    match item.node {
        ItemEnum(ref enum_def, _) => {
            for variant in enum_def.variants.iter() {
                continue_ = callback(variant.node.id);
                if !continue_ {
                    break
                }
            }
        }
        ItemStruct(struct_def, _) => {
            // If this is a newtype struct, return the constructor.
            match struct_def.ctor_id {
                Some(ctor_id) if struct_def.fields.len() > 0 &&
                        struct_def.fields.get(0).node.kind ==
                        ast::UnnamedField => {
                    continue_ = callback(ctor_id);
                }
                _ => {}
            }
        }
        _ => {}
    }

    continue_
}

fn encode_reexports(ecx: &EncodeContext,
                    ebml_w: &mut writer::Encoder,
                    id: NodeId,
                    path: PathElems) {
    debug!("(encoding info for module) encoding reexports for {}", id);
    let reexports2 = ecx.reexports2.borrow();
    match reexports2.get().find(&id) {
        Some(ref exports) => {
            debug!("(encoding info for module) found reexports for {}", id);
            for exp in exports.iter() {
                debug!("(encoding info for module) reexport '{}' ({}/{}) for \
                        {}",
                       exp.name,
                       exp.def_id.krate,
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
                encode_reexported_static_methods(ecx, ebml_w, path.clone(), exp);
            }
        }
        None => {
            debug!("(encoding info for module) found no reexports for {}",
                   id);
        }
    }
}

fn encode_info_for_mod(ecx: &EncodeContext,
                       ebml_w: &mut writer::Encoder,
                       md: &Mod,
                       id: NodeId,
                       path: PathElems,
                       name: Ident,
                       vis: Visibility) {
    ebml_w.start_tag(tag_items_data_item);
    encode_def_id(ebml_w, local_def(id));
    encode_family(ebml_w, 'm');
    encode_name(ebml_w, name.name);
    debug!("(encoding info for module) encoding info for module ID {}", id);

    // Encode info about all the module children.
    for item in md.items.iter() {
        ebml_w.start_tag(tag_mod_child);
        ebml_w.wr_str(def_to_str(local_def(item.id)));
        ebml_w.end_tag();

        each_auxiliary_node_id(*item, |auxiliary_node_id| {
            ebml_w.start_tag(tag_mod_child);
            ebml_w.wr_str(def_to_str(local_def(auxiliary_node_id)));
            ebml_w.end_tag();
            true
        });

        match item.node {
            ItemImpl(..) => {
                let (ident, did) = (item.ident, item.id);
                debug!("(encoding info for module) ... encoding impl {} \
                        ({:?}/{:?})",
                        token::get_ident(ident),
                        did, ecx.tcx.map.node_to_str(did));

                ebml_w.start_tag(tag_mod_impl);
                ebml_w.wr_str(def_to_str(local_def(did)));
                ebml_w.end_tag();
            }
            _ => {}
        }
    }

    encode_path(ebml_w, path.clone());
    encode_visibility(ebml_w, vis);

    // Encode the reexports of this module, if this module is public.
    if vis == Public {
        debug!("(encoding info for module) encoding reexports for {}", id);
        encode_reexports(ecx, ebml_w, id, path);
    }

    ebml_w.end_tag();
}

fn encode_struct_field_family(ebml_w: &mut writer::Encoder,
                              visibility: Visibility) {
    encode_family(ebml_w, match visibility {
        Public => 'g',
        Private => 'j',
        Inherited => 'N'
    });
}

fn encode_visibility(ebml_w: &mut writer::Encoder, visibility: Visibility) {
    ebml_w.start_tag(tag_items_data_item_visibility);
    let ch = match visibility {
        Public => 'y',
        Private => 'n',
        Inherited => 'i',
    };
    ebml_w.wr_str(str::from_char(ch));
    ebml_w.end_tag();
}

fn encode_explicit_self(ebml_w: &mut writer::Encoder, explicit_self: ast::ExplicitSelf_) {
    ebml_w.start_tag(tag_item_trait_method_explicit_self);

    // Encode the base self type.
    match explicit_self {
        SelfStatic => { ebml_w.writer.write(&[ 's' as u8 ]); }
        SelfValue  => { ebml_w.writer.write(&[ 'v' as u8 ]); }
        SelfUniq   => { ebml_w.writer.write(&[ '~' as u8 ]); }
        SelfRegion(_, m) => {
            // FIXME(#4846) encode custom lifetime
            ebml_w.writer.write(&['&' as u8]);
            encode_mutability(ebml_w, m);
        }
    }

    ebml_w.end_tag();

    fn encode_mutability(ebml_w: &mut writer::Encoder,
                         m: ast::Mutability) {
        match m {
            MutImmutable => { ebml_w.writer.write(&[ 'i' as u8 ]); }
            MutMutable => { ebml_w.writer.write(&[ 'm' as u8 ]); }
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
                          fields: &[StructField],
                          global_index: @RefCell<Vec<entry<i64>> >)
                          -> Vec<entry<i64>> {
    /* Each class has its own index, since different classes
       may have fields with the same name */
    let mut index = Vec::new();
    let tcx = ecx.tcx;
     /* We encode both private and public fields -- need to include
        private fields to get the offsets right */
    for field in fields.iter() {
        let (nm, vis) = match field.node.kind {
            NamedField(nm, vis) => (nm, vis),
            UnnamedField => (special_idents::unnamed_field, Inherited)
        };

        let id = field.node.id;
        index.push(entry {val: id as i64, pos: ebml_w.writer.tell().unwrap()});
        {
            let mut global_index = global_index.borrow_mut();
            global_index.get().push(entry {
                val: id as i64,
                pos: ebml_w.writer.tell().unwrap(),
            });
        }
        ebml_w.start_tag(tag_items_data_item);
        debug!("encode_info_for_struct: doing {} {}",
               token::get_ident(nm), id);
        encode_struct_field_family(ebml_w, vis);
        encode_name(ebml_w, nm.name);
        encode_type(ecx, ebml_w, node_id_to_type(tcx, id));
        encode_def_id(ebml_w, local_def(id));
        ebml_w.end_tag();
    }
    index
}

fn encode_info_for_struct_ctor(ecx: &EncodeContext,
                               ebml_w: &mut writer::Encoder,
                               name: ast::Ident,
                               ctor_id: NodeId,
                               index: @RefCell<Vec<entry<i64>> >,
                               struct_id: NodeId) {
    {
        let mut index = index.borrow_mut();
        index.get().push(entry {
            val: ctor_id as i64,
            pos: ebml_w.writer.tell().unwrap(),
        });
    }

    ebml_w.start_tag(tag_items_data_item);
    encode_def_id(ebml_w, local_def(ctor_id));
    encode_family(ebml_w, 'f');
    encode_bounds_and_type(ebml_w, ecx,
                           &lookup_item_type(ecx.tcx, local_def(ctor_id)));
    encode_name(ebml_w, name.name);
    encode_type(ecx, ebml_w, node_id_to_type(ecx.tcx, ctor_id));
    ecx.tcx.map.with_path(ctor_id, |path| encode_path(ebml_w, path));
    encode_parent_item(ebml_w, local_def(struct_id));

    let item_symbols = ecx.item_symbols.borrow();
    if item_symbols.get().contains_key(&ctor_id) {
        encode_symbol(ecx, ebml_w, ctor_id);
    }

    // indicate that this is a tuple struct ctor, because downstream users will normally want
    // the tuple struct definition, but without this there is no way for them to tell that
    // they actually have a ctor rather than a normal function
    ebml_w.start_tag(tag_items_data_item_is_tuple_struct_ctor);
    ebml_w.end_tag();

    ebml_w.end_tag();
}

fn encode_method_ty_fields(ecx: &EncodeContext,
                           ebml_w: &mut writer::Encoder,
                           method_ty: &ty::Method) {
    encode_def_id(ebml_w, method_ty.def_id);
    encode_name(ebml_w, method_ty.ident.name);
    encode_ty_type_param_defs(ebml_w, ecx,
                              method_ty.generics.type_param_defs(),
                              tag_item_method_tps);
    encode_method_fty(ecx, ebml_w, &method_ty.fty);
    encode_visibility(ebml_w, method_ty.vis);
    encode_explicit_self(ebml_w, method_ty.explicit_self);
    let purity = method_ty.fty.purity;
    match method_ty.explicit_self {
        ast::SelfStatic => {
            encode_family(ebml_w, purity_static_method_family(purity));
        }
        _ => encode_family(ebml_w, purity_fn_family(purity))
    }
    encode_provided_source(ebml_w, method_ty.provided_source);
}

fn encode_info_for_method(ecx: &EncodeContext,
                          ebml_w: &mut writer::Encoder,
                          m: &ty::Method,
                          impl_path: PathElems,
                          is_default_impl: bool,
                          parent_id: NodeId,
                          ast_method_opt: Option<@Method>) {

    debug!("encode_info_for_method: {:?} {}", m.def_id,
           token::get_ident(m.ident));
    ebml_w.start_tag(tag_items_data_item);

    encode_method_ty_fields(ecx, ebml_w, m);
    encode_parent_item(ebml_w, local_def(parent_id));

    // The type for methods gets encoded twice, which is unfortunate.
    let tpt = lookup_item_type(ecx.tcx, m.def_id);
    encode_bounds_and_type(ebml_w, ecx, &tpt);

    let elem = ast_map::PathName(m.ident.name);
    encode_path(ebml_w, impl_path.chain(Some(elem).move_iter()));
    match ast_method_opt {
        Some(ast_method) => {
            encode_attributes(ebml_w, ast_method.attrs.as_slice())
        }
        None => ()
    }

    for &ast_method in ast_method_opt.iter() {
        let num_params = tpt.generics.type_param_defs().len();
        if num_params > 0u ||
                is_default_impl ||
                should_inline(ast_method.attrs.as_slice()) {
            (ecx.encode_inlined_item)(
                ecx, ebml_w, IIMethodRef(local_def(parent_id), false, ast_method));
        } else {
            encode_symbol(ecx, ebml_w, m.def_id.node);
        }
    }

    ebml_w.end_tag();
}

fn purity_fn_family(p: Purity) -> char {
    match p {
        UnsafeFn => 'u',
        ImpureFn => 'f',
        ExternFn => 'e'
    }
}

fn purity_static_method_family(p: Purity) -> char {
    match p {
        UnsafeFn => 'U',
        ImpureFn => 'F',
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
    let inherent_impls = ecx.tcx.inherent_impls.borrow();
    match inherent_impls.get().find(&def_id) {
        None => {}
        Some(&implementations) => {
            let implementations = implementations.borrow();
            for implementation in implementations.get().iter() {
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
    let trait_impls = ecx.tcx.trait_impls.borrow();
    match trait_impls.get().find(&trait_def_id) {
        None => {}
        Some(&implementations) => {
            let implementations = implementations.borrow();
            for implementation in implementations.get().iter() {
                ebml_w.start_tag(tag_items_data_item_extension_impl);
                encode_def_id(ebml_w, implementation.did);
                ebml_w.end_tag();
            }
        }
    }
}

fn encode_info_for_item(ecx: &EncodeContext,
                        ebml_w: &mut writer::Encoder,
                        item: &Item,
                        index: @RefCell<Vec<entry<i64>> >,
                        path: PathElems,
                        vis: ast::Visibility) {
    let tcx = ecx.tcx;

    fn add_to_index(item: &Item, ebml_w: &writer::Encoder,
                     index: @RefCell<Vec<entry<i64>> >) {
        let mut index = index.borrow_mut();
        index.get().push(entry {
            val: item.id as i64,
            pos: ebml_w.writer.tell().unwrap(),
        });
    }

    debug!("encoding info for item at {}",
           ecx.tcx.sess.codemap().span_to_str(item.span));

    let def_id = local_def(item.id);
    match item.node {
      ItemStatic(_, m, _) => {
        add_to_index(item, ebml_w, index);
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, def_id);
        if m == ast::MutMutable {
            encode_family(ebml_w, 'b');
        } else {
            encode_family(ebml_w, 'c');
        }
        encode_type(ecx, ebml_w, node_id_to_type(tcx, item.id));
        encode_symbol(ecx, ebml_w, item.id);
        encode_name(ebml_w, item.ident.name);
        encode_path(ebml_w, path);

        let inlineable = !ecx.non_inlineable_statics.borrow().get().contains(&item.id);

        if inlineable {
            (ecx.encode_inlined_item)(ecx, ebml_w, IIItemRef(item));
        }
        encode_visibility(ebml_w, vis);
        ebml_w.end_tag();
      }
      ItemFn(_, purity, _, ref generics, _) => {
        add_to_index(item, ebml_w, index);
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, def_id);
        encode_family(ebml_w, purity_fn_family(purity));
        let tps_len = generics.ty_params.len();
        encode_bounds_and_type(ebml_w, ecx, &lookup_item_type(tcx, def_id));
        encode_name(ebml_w, item.ident.name);
        encode_path(ebml_w, path);
        encode_attributes(ebml_w, item.attrs.as_slice());
        if tps_len > 0u || should_inline(item.attrs.as_slice()) {
            (ecx.encode_inlined_item)(ecx, ebml_w, IIItemRef(item));
        } else {
            encode_symbol(ecx, ebml_w, item.id);
        }
        encode_visibility(ebml_w, vis);
        ebml_w.end_tag();
      }
      ItemMod(ref m) => {
        add_to_index(item, ebml_w, index);
        encode_info_for_mod(ecx,
                            ebml_w,
                            m,
                            item.id,
                            path,
                            item.ident,
                            item.vis);
      }
      ItemForeignMod(ref fm) => {
        add_to_index(item, ebml_w, index);
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, def_id);
        encode_family(ebml_w, 'n');
        encode_name(ebml_w, item.ident.name);
        encode_path(ebml_w, path);

        // Encode all the items in this module.
        for foreign_item in fm.items.iter() {
            ebml_w.start_tag(tag_mod_child);
            ebml_w.wr_str(def_to_str(local_def(foreign_item.id)));
            ebml_w.end_tag();
        }
        encode_visibility(ebml_w, vis);
        ebml_w.end_tag();
      }
      ItemTy(..) => {
        add_to_index(item, ebml_w, index);
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, def_id);
        encode_family(ebml_w, 'y');
        encode_bounds_and_type(ebml_w, ecx, &lookup_item_type(tcx, def_id));
        encode_name(ebml_w, item.ident.name);
        encode_path(ebml_w, path);
        encode_visibility(ebml_w, vis);
        ebml_w.end_tag();
      }
      ItemEnum(ref enum_definition, ref generics) => {
        add_to_index(item, ebml_w, index);

        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, def_id);
        encode_family(ebml_w, 't');
        encode_item_variances(ebml_w, ecx, item.id);
        encode_bounds_and_type(ebml_w, ecx, &lookup_item_type(tcx, def_id));
        encode_name(ebml_w, item.ident.name);
        encode_attributes(ebml_w, item.attrs.as_slice());
        for v in (*enum_definition).variants.iter() {
            encode_variant_id(ebml_w, local_def(v.node.id));
        }
        (ecx.encode_inlined_item)(ecx, ebml_w, IIItemRef(item));
        encode_path(ebml_w, path);

        // Encode inherent implementations for this enumeration.
        encode_inherent_implementations(ecx, ebml_w, def_id);

        encode_visibility(ebml_w, vis);
        ebml_w.end_tag();

        encode_enum_variant_info(ecx,
                                 ebml_w,
                                 item.id,
                                 (*enum_definition).variants.as_slice(),
                                 index,
                                 generics);
      }
      ItemStruct(struct_def, _) => {
        /* First, encode the fields
           These come first because we need to write them to make
           the index, and the index needs to be in the item for the
           class itself */
        let idx = encode_info_for_struct(ecx, ebml_w,
                                         struct_def.fields.as_slice(), index);

        /* Index the class*/
        add_to_index(item, ebml_w, index);

        /* Now, make an item for the class itself */
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, def_id);
        encode_family(ebml_w, 'S');
        encode_bounds_and_type(ebml_w, ecx, &lookup_item_type(tcx, def_id));

        encode_item_variances(ebml_w, ecx, item.id);
        encode_name(ebml_w, item.ident.name);
        encode_attributes(ebml_w, item.attrs.as_slice());
        encode_path(ebml_w, path.clone());
        encode_visibility(ebml_w, vis);

        /* Encode def_ids for each field and method
         for methods, write all the stuff get_trait_method
        needs to know*/
        encode_struct_fields(ebml_w, struct_def);

        (ecx.encode_inlined_item)(ecx, ebml_w, IIItemRef(item));

        // Encode inherent implementations for this structure.
        encode_inherent_implementations(ecx, ebml_w, def_id);

        /* Each class has its own index -- encode it */
        let bkts = create_index(idx);
        encode_index(ebml_w, bkts, write_i64);
        ebml_w.end_tag();

        // If this is a tuple-like struct, encode the type of the constructor.
        match struct_def.ctor_id {
            Some(ctor_id) => {
                encode_info_for_struct_ctor(ecx, ebml_w, item.ident,
                                            ctor_id, index, def_id.node);
            }
            None => {}
        }
      }
      ItemImpl(_, ref opt_trait, ty, ref ast_methods) => {
        // We need to encode information about the default methods we
        // have inherited, so we drive this based on the impl structure.
        let impls = tcx.impls.borrow();
        let imp = impls.get().get(&def_id);

        add_to_index(item, ebml_w, index);
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, def_id);
        encode_family(ebml_w, 'i');
        encode_bounds_and_type(ebml_w, ecx, &lookup_item_type(tcx, def_id));
        encode_name(ebml_w, item.ident.name);
        encode_attributes(ebml_w, item.attrs.as_slice());
        match ty.node {
            ast::TyPath(ref path, ref bounds, _) if path.segments
                                                        .len() == 1 => {
                assert!(bounds.is_none());
                encode_impl_type_basename(ebml_w, ast_util::path_to_ident(path));
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
        encode_path(ebml_w, path.clone());
        ebml_w.end_tag();

        // Iterate down the methods, emitting them. We rely on the
        // assumption that all of the actually implemented methods
        // appear first in the impl structure, in the same order they do
        // in the ast. This is a little sketchy.
        let num_implemented_methods = ast_methods.len();
        for (i, m) in imp.methods.iter().enumerate() {
            let ast_method = if i < num_implemented_methods {
                Some(*ast_methods.get(i))
            } else { None };

            {
                let mut index = index.borrow_mut();
                index.get().push(entry {
                    val: m.def_id.node as i64,
                    pos: ebml_w.writer.tell().unwrap(),
                });
            }
            encode_info_for_method(ecx,
                                   ebml_w,
                                   *m,
                                   path.clone(),
                                   false,
                                   item.id,
                                   ast_method)
        }
      }
      ItemTrait(_, ref super_traits, ref ms) => {
        add_to_index(item, ebml_w, index);
        ebml_w.start_tag(tag_items_data_item);
        encode_def_id(ebml_w, def_id);
        encode_family(ebml_w, 'I');
        encode_item_variances(ebml_w, ecx, item.id);
        let trait_def = ty::lookup_trait_def(tcx, def_id);
        encode_ty_type_param_defs(ebml_w, ecx,
                                  trait_def.generics.type_param_defs(),
                                  tag_items_data_item_ty_param_bounds);
        encode_region_param_defs(ebml_w, trait_def.generics.region_param_defs());
        encode_trait_ref(ebml_w, ecx, trait_def.trait_ref, tag_item_trait_ref);
        encode_name(ebml_w, item.ident.name);
        encode_attributes(ebml_w, item.attrs.as_slice());
        encode_visibility(ebml_w, vis);
        for &method_def_id in ty::trait_method_def_ids(tcx, def_id).iter() {
            ebml_w.start_tag(tag_item_trait_method);
            encode_def_id(ebml_w, method_def_id);
            ebml_w.end_tag();

            ebml_w.start_tag(tag_mod_child);
            ebml_w.wr_str(def_to_str(method_def_id));
            ebml_w.end_tag();
        }
        encode_path(ebml_w, path.clone());
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
            assert_eq!(method_def_id.krate, ast::LOCAL_CRATE);

            let method_ty = ty::method(tcx, method_def_id);

            {
                let mut index = index.borrow_mut();
                index.get().push(entry {
                    val: method_def_id.node as i64,
                    pos: ebml_w.writer.tell().unwrap(),
                });
            }

            ebml_w.start_tag(tag_items_data_item);

            encode_method_ty_fields(ecx, ebml_w, method_ty);

            encode_parent_item(ebml_w, def_id);

            let elem = ast_map::PathName(method_ty.ident.name);
            encode_path(ebml_w, path.clone().chain(Some(elem).move_iter()));

            match method_ty.explicit_self {
                SelfStatic => {
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

            match ms.get(i) {
                &Required(ref tm) => {
                    encode_attributes(ebml_w, tm.attrs.as_slice());
                    encode_method_sort(ebml_w, 'r');
                }

                &Provided(m) => {
                    encode_attributes(ebml_w, m.attrs.as_slice());
                    // If this is a static method, we've already encoded
                    // this.
                    if method_ty.explicit_self != SelfStatic {
                        // FIXME: I feel like there is something funny going on.
                        let tpt = ty::lookup_item_type(tcx, method_def_id);
                        encode_bounds_and_type(ebml_w, ecx, &tpt);
                    }
                    encode_method_sort(ebml_w, 'p');
                    (ecx.encode_inlined_item)(
                        ecx, ebml_w, IIMethodRef(def_id, true, m));
                }
            }

            ebml_w.end_tag();
        }

        // Encode inherent implementations for this trait.
        encode_inherent_implementations(ecx, ebml_w, def_id);
      }
      ItemMac(..) => {
        // macros are encoded separately
      }
    }
}

fn encode_info_for_foreign_item(ecx: &EncodeContext,
                                ebml_w: &mut writer::Encoder,
                                nitem: &ForeignItem,
                                index: @RefCell<Vec<entry<i64>> >,
                                path: PathElems,
                                abi: AbiSet) {
    {
        let mut index = index.borrow_mut();
        index.get().push(entry {
            val: nitem.id as i64,
            pos: ebml_w.writer.tell().unwrap(),
        });
    }

    ebml_w.start_tag(tag_items_data_item);
    encode_def_id(ebml_w, local_def(nitem.id));
    match nitem.node {
      ForeignItemFn(..) => {
        encode_family(ebml_w, purity_fn_family(ImpureFn));
        encode_bounds_and_type(ebml_w, ecx,
                               &lookup_item_type(ecx.tcx,local_def(nitem.id)));
        encode_name(ebml_w, nitem.ident.name);
        if abi.is_intrinsic() {
            (ecx.encode_inlined_item)(ecx, ebml_w, IIForeignRef(nitem));
        } else {
            encode_symbol(ecx, ebml_w, nitem.id);
        }
      }
      ForeignItemStatic(_, mutbl) => {
        if mutbl {
            encode_family(ebml_w, 'b');
        } else {
            encode_family(ebml_w, 'c');
        }
        encode_type(ecx, ebml_w, node_id_to_type(ecx.tcx, nitem.id));
        encode_symbol(ecx, ebml_w, nitem.id);
        encode_name(ebml_w, nitem.ident.name);
      }
    }
    encode_path(ebml_w, path);
    ebml_w.end_tag();
}

fn my_visit_expr(_e: &Expr) { }

fn my_visit_item(i: &Item,
                 ebml_w: &mut writer::Encoder,
                 ecx_ptr: *int,
                 index: @RefCell<Vec<entry<i64>> >) {
    let mut ebml_w = unsafe { ebml_w.unsafe_clone() };
    // See above
    let ecx: &EncodeContext = unsafe { cast::transmute(ecx_ptr) };
    ecx.tcx.map.with_path(i.id, |path| {
        encode_info_for_item(ecx, &mut ebml_w, i, index, path, i.vis);
    });
}

fn my_visit_foreign_item(ni: &ForeignItem,
                         ebml_w: &mut writer::Encoder,
                         ecx_ptr:*int,
                         index: @RefCell<Vec<entry<i64>> >) {
    // See above
    let ecx: &EncodeContext = unsafe { cast::transmute(ecx_ptr) };
    debug!("writing foreign item {}::{}",
            ecx.tcx.map.path_to_str(ni.id),
            token::get_ident(ni.ident));

    let mut ebml_w = unsafe {
        ebml_w.unsafe_clone()
    };
    let abis = ecx.tcx.map.get_foreign_abis(ni.id);
    ecx.tcx.map.with_path(ni.id, |path| {
        encode_info_for_foreign_item(ecx, &mut ebml_w,
                                     ni, index,
                                     path, abis);
    });
}

struct EncodeVisitor<'a,'b> {
    ebml_w_for_visit_item: &'a mut writer::Encoder<'b>,
    ecx_ptr:*int,
    index: @RefCell<Vec<entry<i64>> >,
}

impl<'a,'b> visit::Visitor<()> for EncodeVisitor<'a,'b> {
    fn visit_expr(&mut self, ex: &Expr, _: ()) {
        visit::walk_expr(self, ex, ());
        my_visit_expr(ex);
    }
    fn visit_item(&mut self, i: &Item, _: ()) {
        visit::walk_item(self, i, ());
        my_visit_item(i,
                      self.ebml_w_for_visit_item,
                      self.ecx_ptr,
                      self.index);
    }
    fn visit_foreign_item(&mut self, ni: &ForeignItem, _: ()) {
        visit::walk_foreign_item(self, ni, ());
        my_visit_foreign_item(ni,
                              self.ebml_w_for_visit_item,
                              self.ecx_ptr,
                              self.index);
    }
}

fn encode_info_for_items(ecx: &EncodeContext,
                         ebml_w: &mut writer::Encoder,
                         krate: &Crate)
                         -> Vec<entry<i64>> {
    let index = @RefCell::new(Vec::new());
    ebml_w.start_tag(tag_items_data);
    {
        let mut index = index.borrow_mut();
        index.get().push(entry {
            val: CRATE_NODE_ID as i64,
            pos: ebml_w.writer.tell().unwrap(),
        });
    }
    encode_info_for_mod(ecx,
                        ebml_w,
                        &krate.module,
                        CRATE_NODE_ID,
                        ast_map::Values([].iter()).chain(None),
                        syntax::parse::token::special_idents::invalid,
                        Public);

    // See comment in `encode_side_tables_for_ii` in astencode
    let ecx_ptr: *int = unsafe { cast::transmute(ecx) };
    {
        let mut visitor = EncodeVisitor {
            index: index,
            ecx_ptr: ecx_ptr,
            ebml_w_for_visit_item: &mut *ebml_w,
        };

        visit::walk_crate(&mut visitor, krate, ());
    }

    ebml_w.end_tag();
    return /*bad*/(*index).get();
}


// Path and definition ID indexing

fn create_index<T:Clone + Hash + 'static>(
                index: Vec<entry<T>> )
                -> Vec<@Vec<entry<T>> > {
    let mut buckets: Vec<@RefCell<Vec<entry<T>> >> = Vec::new();
    for _ in range(0u, 256u) {
        buckets.push(@RefCell::new(Vec::new()));
    }
    for elt in index.iter() {
        let h = hash::hash(&elt.val) as uint;
        let mut bucket = buckets.get_mut(h % 256).borrow_mut();
        bucket.get().push((*elt).clone());
    }

    let mut buckets_frozen = Vec::new();
    for bucket in buckets.iter() {
        buckets_frozen.push(@/*bad*/(**bucket).get());
    }
    return buckets_frozen;
}

fn encode_index<T:'static>(
                ebml_w: &mut writer::Encoder,
                buckets: Vec<@Vec<entry<T>> > ,
                write_fn: |&mut MemWriter, &T|) {
    ebml_w.start_tag(tag_index);
    let mut bucket_locs = Vec::new();
    ebml_w.start_tag(tag_index_buckets);
    for bucket in buckets.iter() {
        bucket_locs.push(ebml_w.writer.tell().unwrap());
        ebml_w.start_tag(tag_index_buckets_bucket);
        for elt in (**bucket).iter() {
            ebml_w.start_tag(tag_index_buckets_bucket_elt);
            assert!(elt.pos < 0xffff_ffff);
            {
                let wr: &mut MemWriter = ebml_w.writer;
                wr.write_be_u32(elt.pos as u32);
            }
            write_fn(ebml_w.writer, &elt.val);
            ebml_w.end_tag();
        }
        ebml_w.end_tag();
    }
    ebml_w.end_tag();
    ebml_w.start_tag(tag_index_table);
    for pos in bucket_locs.iter() {
        assert!(*pos < 0xffff_ffff);
        let wr: &mut MemWriter = ebml_w.writer;
        wr.write_be_u32(*pos as u32);
    }
    ebml_w.end_tag();
    ebml_w.end_tag();
}

fn write_i64(writer: &mut MemWriter, &n: &i64) {
    let wr: &mut MemWriter = writer;
    assert!(n < 0x7fff_ffff);
    wr.write_be_u32(n as u32);
}

fn encode_meta_item(ebml_w: &mut writer::Encoder, mi: @MetaItem) {
    match mi.node {
      MetaWord(ref name) => {
        ebml_w.start_tag(tag_meta_item_word);
        ebml_w.start_tag(tag_meta_item_name);
        ebml_w.writer.write(name.get().as_bytes());
        ebml_w.end_tag();
        ebml_w.end_tag();
      }
      MetaNameValue(ref name, ref value) => {
        match value.node {
          LitStr(ref value, _) => {
            ebml_w.start_tag(tag_meta_item_name_value);
            ebml_w.start_tag(tag_meta_item_name);
            ebml_w.writer.write(name.get().as_bytes());
            ebml_w.end_tag();
            ebml_w.start_tag(tag_meta_item_value);
            ebml_w.writer.write(value.get().as_bytes());
            ebml_w.end_tag();
            ebml_w.end_tag();
          }
          _ => {/* FIXME (#623): encode other variants */ }
        }
      }
      MetaList(ref name, ref items) => {
        ebml_w.start_tag(tag_meta_item_list);
        ebml_w.start_tag(tag_meta_item_name);
        ebml_w.writer.write(name.get().as_bytes());
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

// So there's a special crate attribute called 'crate_id' which defines the
// metadata that Rust cares about for linking crates. If the user didn't
// provide it we will throw it in anyway with a default value.
fn synthesize_crate_attrs(ecx: &EncodeContext,
                          krate: &Crate) -> Vec<Attribute> {

    fn synthesize_crateid_attr(ecx: &EncodeContext) -> Attribute {
        assert!(!ecx.link_meta.crateid.name.is_empty());

        attr::mk_attr(
            attr::mk_name_value_item_str(
                InternedString::new("crate_id"),
                token::intern_and_get_ident(ecx.link_meta.crateid.to_str())))
    }

    let mut attrs = Vec::new();
    for attr in krate.attrs.iter() {
        if !attr.name().equiv(&("crate_id")) {
            attrs.push(*attr);
        }
    }
    attrs.push(synthesize_crateid_attr(ecx));

    attrs
}

fn encode_crate_deps(ebml_w: &mut writer::Encoder, cstore: &cstore::CStore) {
    fn get_ordered_deps(cstore: &cstore::CStore) -> Vec<decoder::CrateDep> {
        // Pull the cnums and name,vers,hash out of cstore
        let mut deps = Vec::new();
        cstore.iter_crate_data(|key, val| {
            let dep = decoder::CrateDep {
                cnum: key,
                crate_id: decoder::get_crate_id(val.data()),
                hash: decoder::get_crate_hash(val.data())
            };
            deps.push(dep);
        });

        // Sort by cnum
        deps.sort_by(|kv1, kv2| kv1.cnum.cmp(&kv2.cnum));

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
    let r = get_ordered_deps(cstore);
    for dep in r.iter() {
        encode_crate_dep(ebml_w, (*dep).clone());
    }
    ebml_w.end_tag();
}

fn encode_lang_items(ecx: &EncodeContext, ebml_w: &mut writer::Encoder) {
    ebml_w.start_tag(tag_lang_items);

    for (i, def_id) in ecx.tcx.lang_items.items() {
        for id in def_id.iter() {
            if id.krate == LOCAL_CRATE {
                ebml_w.start_tag(tag_lang_items_item);

                ebml_w.start_tag(tag_lang_items_item_id);
                {
                    let wr: &mut MemWriter = ebml_w.writer;
                    wr.write_be_u32(i as u32);
                }
                ebml_w.end_tag();   // tag_lang_items_item_id

                ebml_w.start_tag(tag_lang_items_item_node_id);
                {
                    let wr: &mut MemWriter = ebml_w.writer;
                    wr.write_be_u32(id.node as u32);
                }
                ebml_w.end_tag();   // tag_lang_items_item_node_id

                ebml_w.end_tag();   // tag_lang_items_item
            }
        }
    }

    ebml_w.end_tag();   // tag_lang_items
}

fn encode_native_libraries(ecx: &EncodeContext, ebml_w: &mut writer::Encoder) {
    ebml_w.start_tag(tag_native_libraries);

    let used_libraries = ecx.tcx.sess.cstore.get_used_libraries();
    let used_libraries = used_libraries.borrow();
    for &(ref lib, kind) in used_libraries.get().iter() {
        match kind {
            cstore::NativeStatic => {} // these libraries are not propagated
            cstore::NativeFramework | cstore::NativeUnknown => {
                ebml_w.start_tag(tag_native_libraries_lib);

                ebml_w.start_tag(tag_native_libraries_kind);
                ebml_w.writer.write_be_u32(kind as u32);
                ebml_w.end_tag();

                ebml_w.start_tag(tag_native_libraries_name);
                ebml_w.writer.write(lib.as_bytes());
                ebml_w.end_tag();

                ebml_w.end_tag();
            }
        }
    }

    ebml_w.end_tag();
}

fn encode_macro_registrar_fn(ecx: &EncodeContext, ebml_w: &mut writer::Encoder) {
    let ptr = ecx.tcx.sess.macro_registrar_fn.borrow();
    match *ptr.get() {
        Some(did) => {
            ebml_w.start_tag(tag_macro_registrar_fn);
            encode_def_id(ebml_w, did);
            ebml_w.end_tag();
        }
        None => {}
    }
}

struct MacroDefVisitor<'a, 'b> {
    ecx: &'a EncodeContext<'a>,
    ebml_w: &'a mut writer::Encoder<'b>
}

impl<'a, 'b> Visitor<()> for MacroDefVisitor<'a, 'b> {
    fn visit_item(&mut self, item: &Item, _: ()) {
        match item.node {
            ItemMac(..) => {
                let def = self.ecx.tcx.sess.codemap().span_to_snippet(item.span)
                    .expect("Unable to find source for macro");
                self.ebml_w.start_tag(tag_macro_def);
                self.ebml_w.wr_str(def);
                self.ebml_w.end_tag();
            }
            _ => {}
        }
        visit::walk_item(self, item, ());
    }
}

fn encode_macro_defs(ecx: &EncodeContext,
                     krate: &Crate,
                     ebml_w: &mut writer::Encoder) {
    ebml_w.start_tag(tag_exported_macros);
    {
        let mut visitor = MacroDefVisitor {
            ecx: ecx,
            ebml_w: ebml_w,
        };
        visit::walk_crate(&mut visitor, krate, ());
    }
    ebml_w.end_tag();
}

struct ImplVisitor<'a,'b> {
    ecx: &'a EncodeContext<'a>,
    ebml_w: &'a mut writer::Encoder<'b>,
}

impl<'a,'b> Visitor<()> for ImplVisitor<'a,'b> {
    fn visit_item(&mut self, item: &Item, _: ()) {
        match item.node {
            ItemImpl(_, Some(ref trait_ref), _, _) => {
                let def_map = self.ecx.tcx.def_map;
                let def_map = def_map.borrow();
                let trait_def = def_map.get().get_copy(&trait_ref.ref_id);
                let def_id = ast_util::def_id_of_def(trait_def);

                // Load eagerly if this is an implementation of the Drop trait
                // or if the trait is not defined in this crate.
                if Some(def_id) == self.ecx.tcx.lang_items.drop_trait() ||
                        def_id.krate != LOCAL_CRATE {
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
                krate: &Crate,
                ebml_w: &mut writer::Encoder) {
    ebml_w.start_tag(tag_impls);

    {
        let mut visitor = ImplVisitor {
            ecx: ecx,
            ebml_w: ebml_w,
        };
        visit::walk_crate(&mut visitor, krate, ());
    }

    ebml_w.end_tag();
}

fn encode_misc_info(ecx: &EncodeContext,
                    krate: &Crate,
                    ebml_w: &mut writer::Encoder) {
    ebml_w.start_tag(tag_misc_info);
    ebml_w.start_tag(tag_misc_info_crate_items);
    for &item in krate.module.items.iter() {
        ebml_w.start_tag(tag_mod_child);
        ebml_w.wr_str(def_to_str(local_def(item.id)));
        ebml_w.end_tag();

        each_auxiliary_node_id(item, |auxiliary_node_id| {
            ebml_w.start_tag(tag_mod_child);
            ebml_w.wr_str(def_to_str(local_def(auxiliary_node_id)));
            ebml_w.end_tag();
            true
        });
    }

    // Encode reexports for the root module.
    encode_reexports(ecx, ebml_w, 0, ast_map::Values([].iter()).chain(None));

    ebml_w.end_tag();
    ebml_w.end_tag();
}

fn encode_crate_dep(ebml_w: &mut writer::Encoder,
                    dep: decoder::CrateDep) {
    ebml_w.start_tag(tag_crate_dep);
    ebml_w.start_tag(tag_crate_dep_crateid);
    ebml_w.writer.write(dep.crate_id.to_str().as_bytes());
    ebml_w.end_tag();
    ebml_w.start_tag(tag_crate_dep_hash);
    ebml_w.writer.write(dep.hash.as_str().as_bytes());
    ebml_w.end_tag();
    ebml_w.end_tag();
}

fn encode_hash(ebml_w: &mut writer::Encoder, hash: &Svh) {
    ebml_w.start_tag(tag_crate_hash);
    ebml_w.writer.write(hash.as_str().as_bytes());
    ebml_w.end_tag();
}

fn encode_crate_id(ebml_w: &mut writer::Encoder, crate_id: &CrateId) {
    ebml_w.start_tag(tag_crate_crateid);
    ebml_w.writer.write(crate_id.to_str().as_bytes());
    ebml_w.end_tag();
}

// NB: Increment this as you change the metadata encoding version.
pub static metadata_encoding_version : &'static [u8] =
    &[0x72, //'r' as u8,
      0x75, //'u' as u8,
      0x73, //'s' as u8,
      0x74, //'t' as u8,
      0, 0, 0, 1 ];

pub fn encode_metadata(parms: EncodeParams, krate: &Crate) -> Vec<u8> {
    let mut wr = MemWriter::new();
    encode_metadata_inner(&mut wr, parms, krate);
    wr.unwrap().move_iter().collect()
}

fn encode_metadata_inner(wr: &mut MemWriter, parms: EncodeParams, krate: &Crate) {
    let stats = Stats {
        inline_bytes: Cell::new(0),
        attr_bytes: Cell::new(0),
        dep_bytes: Cell::new(0),
        lang_item_bytes: Cell::new(0),
        native_lib_bytes: Cell::new(0),
        macro_registrar_fn_bytes: Cell::new(0),
        macro_defs_bytes: Cell::new(0),
        impl_bytes: Cell::new(0),
        misc_bytes: Cell::new(0),
        item_bytes: Cell::new(0),
        index_bytes: Cell::new(0),
        zero_bytes: Cell::new(0),
        total_bytes: Cell::new(0),
    };
    let EncodeParams {
        item_symbols,
        diag,
        tcx,
        reexports2,
        cstore,
        encode_inlined_item,
        link_meta,
        non_inlineable_statics,
        ..
    } = parms;
    let type_abbrevs = @RefCell::new(HashMap::new());
    let stats = @stats;
    let ecx = EncodeContext {
        diag: diag,
        tcx: tcx,
        stats: stats,
        reexports2: reexports2,
        item_symbols: item_symbols,
        non_inlineable_statics: non_inlineable_statics,
        link_meta: link_meta,
        cstore: cstore,
        encode_inlined_item: encode_inlined_item,
        type_abbrevs: type_abbrevs,
     };

    let mut ebml_w = writer::Encoder(wr);

    encode_crate_id(&mut ebml_w, &ecx.link_meta.crateid);
    encode_hash(&mut ebml_w, &ecx.link_meta.crate_hash);

    let mut i = ebml_w.writer.tell().unwrap();
    let crate_attrs = synthesize_crate_attrs(&ecx, krate);
    encode_attributes(&mut ebml_w, crate_attrs.as_slice());
    ecx.stats.attr_bytes.set(ebml_w.writer.tell().unwrap() - i);

    i = ebml_w.writer.tell().unwrap();
    encode_crate_deps(&mut ebml_w, ecx.cstore);
    ecx.stats.dep_bytes.set(ebml_w.writer.tell().unwrap() - i);

    // Encode the language items.
    i = ebml_w.writer.tell().unwrap();
    encode_lang_items(&ecx, &mut ebml_w);
    ecx.stats.lang_item_bytes.set(ebml_w.writer.tell().unwrap() - i);

    // Encode the native libraries used
    i = ebml_w.writer.tell().unwrap();
    encode_native_libraries(&ecx, &mut ebml_w);
    ecx.stats.native_lib_bytes.set(ebml_w.writer.tell().unwrap() - i);

    // Encode the macro registrar function
    i = ebml_w.writer.tell().unwrap();
    encode_macro_registrar_fn(&ecx, &mut ebml_w);
    ecx.stats.macro_registrar_fn_bytes.set(ebml_w.writer.tell().unwrap() - i);

    // Encode macro definitions
    i = ebml_w.writer.tell().unwrap();
    encode_macro_defs(&ecx, krate, &mut ebml_w);
    ecx.stats.macro_defs_bytes.set(ebml_w.writer.tell().unwrap() - i);

    // Encode the def IDs of impls, for coherence checking.
    i = ebml_w.writer.tell().unwrap();
    encode_impls(&ecx, krate, &mut ebml_w);
    ecx.stats.impl_bytes.set(ebml_w.writer.tell().unwrap() - i);

    // Encode miscellaneous info.
    i = ebml_w.writer.tell().unwrap();
    encode_misc_info(&ecx, krate, &mut ebml_w);
    ecx.stats.misc_bytes.set(ebml_w.writer.tell().unwrap() - i);

    // Encode and index the items.
    ebml_w.start_tag(tag_items);
    i = ebml_w.writer.tell().unwrap();
    let items_index = encode_info_for_items(&ecx, &mut ebml_w, krate);
    ecx.stats.item_bytes.set(ebml_w.writer.tell().unwrap() - i);

    i = ebml_w.writer.tell().unwrap();
    let items_buckets = create_index(items_index);
    encode_index(&mut ebml_w, items_buckets, write_i64);
    ecx.stats.index_bytes.set(ebml_w.writer.tell().unwrap() - i);
    ebml_w.end_tag();

    ecx.stats.total_bytes.set(ebml_w.writer.tell().unwrap());

    if tcx.sess.meta_stats() {
        for e in ebml_w.writer.get_ref().iter() {
            if *e == 0 {
                ecx.stats.zero_bytes.set(ecx.stats.zero_bytes.get() + 1);
            }
        }

        println!("metadata stats:");
        println!("         inline bytes: {}", ecx.stats.inline_bytes.get());
        println!("      attribute bytes: {}", ecx.stats.attr_bytes.get());
        println!("            dep bytes: {}", ecx.stats.dep_bytes.get());
        println!("      lang item bytes: {}", ecx.stats.lang_item_bytes.get());
        println!("         native bytes: {}", ecx.stats.native_lib_bytes.get());
        println!("macro registrar bytes: {}", ecx.stats.macro_registrar_fn_bytes.get());
        println!("      macro def bytes: {}", ecx.stats.macro_defs_bytes.get());
        println!("           impl bytes: {}", ecx.stats.impl_bytes.get());
        println!("           misc bytes: {}", ecx.stats.misc_bytes.get());
        println!("           item bytes: {}", ecx.stats.item_bytes.get());
        println!("          index bytes: {}", ecx.stats.index_bytes.get());
        println!("           zero bytes: {}", ecx.stats.zero_bytes.get());
        println!("          total bytes: {}", ecx.stats.total_bytes.get());
    }
}

// Get the encoded string for a type
pub fn encoded_ty(tcx: &ty::ctxt, t: ty::t) -> ~str {
    let cx = &tyencode::ctxt {
        diag: tcx.sess.diagnostic(),
        ds: def_to_str,
        tcx: tcx,
        abbrevs: tyencode::ac_no_abbrevs};
    let mut wr = MemWriter::new();
    tyencode::enc_ty(&mut wr, cx, t);
    str::from_utf8_owned(wr.get_ref().to_owned()).unwrap()
}
