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

#![allow(unused_must_use)] // everything is just a MemWriter, can't fail
#![allow(non_camel_case_types)]

use back::svh::Svh;
use driver::config;
use metadata::common::*;
use metadata::cstore;
use metadata::decoder;
use metadata::tyencode;
use middle::ty::{lookup_item_type};
use middle::astencode;
use middle::ty;
use middle::typeck;
use middle::stability;
use middle;
use util::nodemap::{NodeMap, NodeSet};

use serialize::Encodable;
use std::cell::RefCell;
use std::gc::Gc;
use std::hash::Hash;
use std::hash;
use std::mem;
use std::collections::HashMap;
use syntax::abi;
use syntax::ast::*;
use syntax::ast;
use syntax::ast_map::{PathElem, PathElems};
use syntax::ast_map;
use syntax::ast_util::*;
use syntax::ast_util;
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::diagnostic::SpanHandler;
use syntax::parse::token::special_idents;
use syntax::parse::token;
use syntax::visit::Visitor;
use syntax::visit;
use syntax;
use rbml::writer;
use rbml::io::SeekableMemWriter;

/// A borrowed version of `ast::InlinedItem`.
pub enum InlinedItemRef<'a> {
    IIItemRef(&'a ast::Item),
    IITraitItemRef(ast::DefId, InlinedTraitItemRef<'a>),
    IIForeignRef(&'a ast::ForeignItem)
}

/// A borrowed version of `ast::InlinedTraitItem`.
pub enum InlinedTraitItemRef<'a> {
    ProvidedInlinedTraitItemRef(&'a Method),
    RequiredInlinedTraitItemRef(&'a Method),
}

pub type Encoder<'a> = writer::Encoder<'a, SeekableMemWriter>;

pub type EncodeInlinedItem<'a> = |ecx: &EncodeContext,
                                  rbml_w: &mut Encoder,
                                  ii: InlinedItemRef|: 'a;

pub struct EncodeParams<'a> {
    pub diag: &'a SpanHandler,
    pub tcx: &'a ty::ctxt,
    pub reexports2: &'a middle::resolve::ExportMap2,
    pub item_symbols: &'a RefCell<NodeMap<String>>,
    pub non_inlineable_statics: &'a RefCell<NodeSet>,
    pub link_meta: &'a LinkMeta,
    pub cstore: &'a cstore::CStore,
    pub encode_inlined_item: EncodeInlinedItem<'a>,
    pub reachable: &'a NodeSet,
}

pub struct EncodeContext<'a> {
    pub diag: &'a SpanHandler,
    pub tcx: &'a ty::ctxt,
    pub reexports2: &'a middle::resolve::ExportMap2,
    pub item_symbols: &'a RefCell<NodeMap<String>>,
    pub non_inlineable_statics: &'a RefCell<NodeSet>,
    pub link_meta: &'a LinkMeta,
    pub cstore: &'a cstore::CStore,
    pub encode_inlined_item: RefCell<EncodeInlinedItem<'a>>,
    pub type_abbrevs: tyencode::abbrev_map,
    pub reachable: &'a NodeSet,
}

fn encode_name(rbml_w: &mut Encoder, name: Name) {
    rbml_w.wr_tagged_str(tag_paths_data_name, token::get_name(name).get());
}

fn encode_impl_type_basename(rbml_w: &mut Encoder, name: Ident) {
    rbml_w.wr_tagged_str(tag_item_impl_type_basename, token::get_ident(name).get());
}

pub fn encode_def_id(rbml_w: &mut Encoder, id: DefId) {
    rbml_w.wr_tagged_str(tag_def_id, def_to_string(id).as_slice());
}

#[deriving(Clone)]
struct entry<T> {
    val: T,
    pos: u64
}

fn encode_trait_ref(rbml_w: &mut Encoder,
                    ecx: &EncodeContext,
                    trait_ref: &ty::TraitRef,
                    tag: uint) {
    let ty_str_ctxt = &tyencode::ctxt {
        diag: ecx.diag,
        ds: def_to_string,
        tcx: ecx.tcx,
        abbrevs: &ecx.type_abbrevs
    };

    rbml_w.start_tag(tag);
    tyencode::enc_trait_ref(rbml_w.writer, ty_str_ctxt, trait_ref);
    rbml_w.end_tag();
}

fn encode_impl_vtables(rbml_w: &mut Encoder,
                       ecx: &EncodeContext,
                       vtables: &typeck::vtable_res) {
    rbml_w.start_tag(tag_item_impl_vtables);
    astencode::encode_vtable_res(ecx, rbml_w, vtables);
    rbml_w.end_tag();
}

// Item info table encoding
fn encode_family(rbml_w: &mut Encoder, c: char) {
    rbml_w.start_tag(tag_items_data_item_family);
    rbml_w.writer.write(&[c as u8]);
    rbml_w.end_tag();
}

pub fn def_to_string(did: DefId) -> String {
    format!("{}:{}", did.krate, did.node)
}

fn encode_item_variances(rbml_w: &mut Encoder,
                         ecx: &EncodeContext,
                         id: ast::NodeId) {
    let v = ty::item_variances(ecx.tcx, ast_util::local_def(id));
    rbml_w.start_tag(tag_item_variances);
    v.encode(rbml_w);
    rbml_w.end_tag();
}

fn encode_bounds_and_type(rbml_w: &mut Encoder,
                          ecx: &EncodeContext,
                          pty: &ty::Polytype) {
    encode_generics(rbml_w, ecx, &pty.generics, tag_item_generics);
    encode_type(ecx, rbml_w, pty.ty);
}

fn encode_variant_id(rbml_w: &mut Encoder, vid: DefId) {
    rbml_w.start_tag(tag_items_data_item_variant);
    let s = def_to_string(vid);
    rbml_w.writer.write(s.as_bytes());
    rbml_w.end_tag();
}

pub fn write_closure_type(ecx: &EncodeContext,
                          rbml_w: &mut Encoder,
                          closure_type: &ty::ClosureTy) {
    let ty_str_ctxt = &tyencode::ctxt {
        diag: ecx.diag,
        ds: def_to_string,
        tcx: ecx.tcx,
        abbrevs: &ecx.type_abbrevs
    };
    tyencode::enc_closure_ty(rbml_w.writer, ty_str_ctxt, closure_type);
}

pub fn write_type(ecx: &EncodeContext,
                  rbml_w: &mut Encoder,
                  typ: ty::t) {
    let ty_str_ctxt = &tyencode::ctxt {
        diag: ecx.diag,
        ds: def_to_string,
        tcx: ecx.tcx,
        abbrevs: &ecx.type_abbrevs
    };
    tyencode::enc_ty(rbml_w.writer, ty_str_ctxt, typ);
}

pub fn write_region(ecx: &EncodeContext,
                    rbml_w: &mut Encoder,
                    r: ty::Region) {
    let ty_str_ctxt = &tyencode::ctxt {
        diag: ecx.diag,
        ds: def_to_string,
        tcx: ecx.tcx,
        abbrevs: &ecx.type_abbrevs
    };
    tyencode::enc_region(rbml_w.writer, ty_str_ctxt, r);
}

fn encode_bounds(rbml_w: &mut Encoder,
                 ecx: &EncodeContext,
                 bounds: &ty::ParamBounds,
                 tag: uint) {
    rbml_w.start_tag(tag);

    let ty_str_ctxt = &tyencode::ctxt { diag: ecx.diag,
                                        ds: def_to_string,
                                        tcx: ecx.tcx,
                                        abbrevs: &ecx.type_abbrevs };
    tyencode::enc_bounds(rbml_w.writer, ty_str_ctxt, bounds);

    rbml_w.end_tag();
}

fn encode_type(ecx: &EncodeContext,
               rbml_w: &mut Encoder,
               typ: ty::t) {
    rbml_w.start_tag(tag_items_data_item_type);
    write_type(ecx, rbml_w, typ);
    rbml_w.end_tag();
}

fn encode_region(ecx: &EncodeContext,
                 rbml_w: &mut Encoder,
                 r: ty::Region) {
    rbml_w.start_tag(tag_items_data_region);
    write_region(ecx, rbml_w, r);
    rbml_w.end_tag();
}

fn encode_method_fty(ecx: &EncodeContext,
                     rbml_w: &mut Encoder,
                     typ: &ty::BareFnTy) {
    rbml_w.start_tag(tag_item_method_fty);

    let ty_str_ctxt = &tyencode::ctxt {
        diag: ecx.diag,
        ds: def_to_string,
        tcx: ecx.tcx,
        abbrevs: &ecx.type_abbrevs
    };
    tyencode::enc_bare_fn_ty(rbml_w.writer, ty_str_ctxt, typ);

    rbml_w.end_tag();
}

fn encode_symbol(ecx: &EncodeContext,
                 rbml_w: &mut Encoder,
                 id: NodeId) {
    rbml_w.start_tag(tag_items_data_item_symbol);
    match ecx.item_symbols.borrow().find(&id) {
        Some(x) => {
            debug!("encode_symbol(id={:?}, str={})", id, *x);
            rbml_w.writer.write(x.as_bytes());
        }
        None => {
            ecx.diag.handler().bug(
                format!("encode_symbol: id not found {}", id).as_slice());
        }
    }
    rbml_w.end_tag();
}

fn encode_disr_val(_: &EncodeContext,
                   rbml_w: &mut Encoder,
                   disr_val: ty::Disr) {
    rbml_w.start_tag(tag_disr_val);
    let s = disr_val.to_string();
    rbml_w.writer.write(s.as_bytes());
    rbml_w.end_tag();
}

fn encode_parent_item(rbml_w: &mut Encoder, id: DefId) {
    rbml_w.start_tag(tag_items_data_parent_item);
    let s = def_to_string(id);
    rbml_w.writer.write(s.as_bytes());
    rbml_w.end_tag();
}

fn encode_struct_fields(rbml_w: &mut Encoder,
                        fields: &[ty::field_ty],
                        origin: DefId) {
    for f in fields.iter() {
        if f.name == special_idents::unnamed_field.name {
            rbml_w.start_tag(tag_item_unnamed_field);
        } else {
            rbml_w.start_tag(tag_item_field);
            encode_name(rbml_w, f.name);
        }
        encode_struct_field_family(rbml_w, f.vis);
        encode_def_id(rbml_w, f.id);
        rbml_w.start_tag(tag_item_field_origin);
        let s = def_to_string(origin);
        rbml_w.writer.write(s.as_bytes());
        rbml_w.end_tag();
        rbml_w.end_tag();
    }
}

fn encode_enum_variant_info(ecx: &EncodeContext,
                            rbml_w: &mut Encoder,
                            id: NodeId,
                            variants: &[P<Variant>],
                            index: &mut Vec<entry<i64>>) {
    debug!("encode_enum_variant_info(id={:?})", id);

    let mut disr_val = 0;
    let mut i = 0;
    let vi = ty::enum_variants(ecx.tcx,
                               ast::DefId { krate: LOCAL_CRATE, node: id });
    for variant in variants.iter() {
        let def_id = local_def(variant.node.id);
        index.push(entry {
            val: variant.node.id as i64,
            pos: rbml_w.writer.tell().unwrap(),
        });
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id(rbml_w, def_id);
        match variant.node.kind {
            ast::TupleVariantKind(_) => encode_family(rbml_w, 'v'),
            ast::StructVariantKind(_) => encode_family(rbml_w, 'V')
        }
        encode_name(rbml_w, variant.node.name.name);
        encode_parent_item(rbml_w, local_def(id));
        encode_visibility(rbml_w, variant.node.vis);
        encode_attributes(rbml_w, variant.node.attrs.as_slice());

        let stab = stability::lookup(ecx.tcx, ast_util::local_def(variant.node.id));
        encode_stability(rbml_w, stab);

        match variant.node.kind {
            ast::TupleVariantKind(_) => {},
            ast::StructVariantKind(_) => {
                let fields = ty::lookup_struct_fields(ecx.tcx, def_id);
                let idx = encode_info_for_struct(ecx,
                                                 rbml_w,
                                                 fields.as_slice(),
                                                 index);
                encode_struct_fields(rbml_w, fields.as_slice(), def_id);
                encode_index(rbml_w, idx, write_i64);
            }
        }
        if vi.get(i).disr_val != disr_val {
            encode_disr_val(ecx, rbml_w, vi.get(i).disr_val);
            disr_val = vi.get(i).disr_val;
        }
        encode_bounds_and_type(rbml_w, ecx,
                               &lookup_item_type(ecx.tcx, def_id));

        ecx.tcx.map.with_path(variant.node.id, |path| encode_path(rbml_w, path));
        rbml_w.end_tag();
        disr_val += 1;
        i += 1;
    }
}

fn encode_path<PI: Iterator<PathElem> + Clone>(rbml_w: &mut Encoder,
                                               mut path: PI) {
    rbml_w.start_tag(tag_path);
    rbml_w.wr_tagged_u32(tag_path_len, path.clone().count() as u32);
    for pe in path {
        let tag = match pe {
            ast_map::PathMod(_) => tag_path_elem_mod,
            ast_map::PathName(_) => tag_path_elem_name
        };
        rbml_w.wr_tagged_str(tag, token::get_name(pe.name()).get());
    }
    rbml_w.end_tag();
}

fn encode_reexported_static_method(rbml_w: &mut Encoder,
                                   exp: &middle::resolve::Export2,
                                   method_def_id: DefId,
                                   method_ident: Ident) {
    debug!("(encode reexported static method) {}::{}",
            exp.name, token::get_ident(method_ident));
    rbml_w.start_tag(tag_items_data_item_reexport);
    rbml_w.start_tag(tag_items_data_item_reexport_def_id);
    rbml_w.wr_str(def_to_string(method_def_id).as_slice());
    rbml_w.end_tag();
    rbml_w.start_tag(tag_items_data_item_reexport_name);
    rbml_w.wr_str(format!("{}::{}",
                          exp.name,
                          token::get_ident(method_ident)).as_slice());
    rbml_w.end_tag();
    rbml_w.end_tag();
}

fn encode_reexported_static_base_methods(ecx: &EncodeContext,
                                         rbml_w: &mut Encoder,
                                         exp: &middle::resolve::Export2)
                                         -> bool {
    let impl_items = ecx.tcx.impl_items.borrow();
    match ecx.tcx.inherent_impls.borrow().find(&exp.def_id) {
        Some(implementations) => {
            for base_impl_did in implementations.borrow().iter() {
                for &method_did in impl_items.get(base_impl_did).iter() {
                    let impl_item = ty::impl_or_trait_item(
                        ecx.tcx,
                        method_did.def_id());
                    match impl_item {
                        ty::MethodTraitItem(ref m) => {
                            if m.explicit_self ==
                                    ty::StaticExplicitSelfCategory {
                                encode_reexported_static_method(rbml_w,
                                                                exp,
                                                                m.def_id,
                                                                m.ident);
                            }
                        }
                    }
                }
            }

            true
        }
        None => { false }
    }
}

fn encode_reexported_static_trait_methods(ecx: &EncodeContext,
                                          rbml_w: &mut Encoder,
                                          exp: &middle::resolve::Export2)
                                          -> bool {
    match ecx.tcx.trait_items_cache.borrow().find(&exp.def_id) {
        Some(trait_items) => {
            for trait_item in trait_items.iter() {
                match *trait_item {
                    ty::MethodTraitItem(ref m) if m.explicit_self ==
                            ty::StaticExplicitSelfCategory => {
                        encode_reexported_static_method(rbml_w,
                                                        exp,
                                                        m.def_id,
                                                        m.ident);
                    }
                    _ => {}
                }
            }

            true
        }
        None => { false }
    }
}

fn encode_reexported_static_methods(ecx: &EncodeContext,
                                    rbml_w: &mut Encoder,
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
            if path_differs || original_name.get() != exp.name.as_slice() {
                if !encode_reexported_static_base_methods(ecx, rbml_w, exp) {
                    if encode_reexported_static_trait_methods(ecx, rbml_w, exp) {
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
fn each_auxiliary_node_id(item: Gc<Item>, callback: |NodeId| -> bool) -> bool {
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
                        struct_def.fields.get(0).node.kind.is_unnamed() => {
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
                    rbml_w: &mut Encoder,
                    id: NodeId,
                    path: PathElems) {
    debug!("(encoding info for module) encoding reexports for {}", id);
    match ecx.reexports2.borrow().find(&id) {
        Some(ref exports) => {
            debug!("(encoding info for module) found reexports for {}", id);
            for exp in exports.iter() {
                debug!("(encoding info for module) reexport '{}' ({}/{}) for \
                        {}",
                       exp.name,
                       exp.def_id.krate,
                       exp.def_id.node,
                       id);
                rbml_w.start_tag(tag_items_data_item_reexport);
                rbml_w.start_tag(tag_items_data_item_reexport_def_id);
                rbml_w.wr_str(def_to_string(exp.def_id).as_slice());
                rbml_w.end_tag();
                rbml_w.start_tag(tag_items_data_item_reexport_name);
                rbml_w.wr_str(exp.name.as_slice());
                rbml_w.end_tag();
                rbml_w.end_tag();
                encode_reexported_static_methods(ecx, rbml_w, path.clone(), exp);
            }
        }
        None => {
            debug!("(encoding info for module) found no reexports for {}",
                   id);
        }
    }
}

fn encode_info_for_mod(ecx: &EncodeContext,
                       rbml_w: &mut Encoder,
                       md: &Mod,
                       attrs: &[Attribute],
                       id: NodeId,
                       path: PathElems,
                       name: Ident,
                       vis: Visibility) {
    rbml_w.start_tag(tag_items_data_item);
    encode_def_id(rbml_w, local_def(id));
    encode_family(rbml_w, 'm');
    encode_name(rbml_w, name.name);
    debug!("(encoding info for module) encoding info for module ID {}", id);

    // Encode info about all the module children.
    for item in md.items.iter() {
        rbml_w.start_tag(tag_mod_child);
        rbml_w.wr_str(def_to_string(local_def(item.id)).as_slice());
        rbml_w.end_tag();

        each_auxiliary_node_id(*item, |auxiliary_node_id| {
            rbml_w.start_tag(tag_mod_child);
            rbml_w.wr_str(def_to_string(local_def(
                        auxiliary_node_id)).as_slice());
            rbml_w.end_tag();
            true
        });

        match item.node {
            ItemImpl(..) => {
                let (ident, did) = (item.ident, item.id);
                debug!("(encoding info for module) ... encoding impl {} \
                        ({:?}/{:?})",
                        token::get_ident(ident),
                        did, ecx.tcx.map.node_to_string(did));

                rbml_w.start_tag(tag_mod_impl);
                rbml_w.wr_str(def_to_string(local_def(did)).as_slice());
                rbml_w.end_tag();
            }
            _ => {}
        }
    }

    encode_path(rbml_w, path.clone());
    encode_visibility(rbml_w, vis);

    let stab = stability::lookup(ecx.tcx, ast_util::local_def(id));
    encode_stability(rbml_w, stab);

    // Encode the reexports of this module, if this module is public.
    if vis == Public {
        debug!("(encoding info for module) encoding reexports for {}", id);
        encode_reexports(ecx, rbml_w, id, path);
    }
    encode_attributes(rbml_w, attrs);

    rbml_w.end_tag();
}

fn encode_struct_field_family(rbml_w: &mut Encoder,
                              visibility: Visibility) {
    encode_family(rbml_w, match visibility {
        Public => 'g',
        Inherited => 'N'
    });
}

fn encode_visibility(rbml_w: &mut Encoder, visibility: Visibility) {
    rbml_w.start_tag(tag_items_data_item_visibility);
    let ch = match visibility {
        Public => 'y',
        Inherited => 'i',
    };
    rbml_w.wr_str(ch.to_string().as_slice());
    rbml_w.end_tag();
}

fn encode_unboxed_closure_kind(rbml_w: &mut Encoder,
                               kind: ty::UnboxedClosureKind) {
    rbml_w.start_tag(tag_unboxed_closure_kind);
    let ch = match kind {
        ty::FnUnboxedClosureKind => 'f',
        ty::FnMutUnboxedClosureKind => 'm',
        ty::FnOnceUnboxedClosureKind => 'o',
    };
    rbml_w.wr_str(ch.to_string().as_slice());
    rbml_w.end_tag();
}

fn encode_explicit_self(rbml_w: &mut Encoder,
                        explicit_self: &ty::ExplicitSelfCategory) {
    rbml_w.start_tag(tag_item_trait_method_explicit_self);

    // Encode the base self type.
    match *explicit_self {
        ty::StaticExplicitSelfCategory => {
            rbml_w.writer.write(&[ 's' as u8 ]);
        }
        ty::ByValueExplicitSelfCategory => {
            rbml_w.writer.write(&[ 'v' as u8 ]);
        }
        ty::ByBoxExplicitSelfCategory => {
            rbml_w.writer.write(&[ '~' as u8 ]);
        }
        ty::ByReferenceExplicitSelfCategory(_, m) => {
            // FIXME(#4846) encode custom lifetime
            rbml_w.writer.write(&['&' as u8]);
            encode_mutability(rbml_w, m);
        }
    }

    rbml_w.end_tag();

    fn encode_mutability(rbml_w: &mut Encoder,
                         m: ast::Mutability) {
        match m {
            MutImmutable => { rbml_w.writer.write(&[ 'i' as u8 ]); }
            MutMutable => { rbml_w.writer.write(&[ 'm' as u8 ]); }
        }
    }
}

fn encode_item_sort(rbml_w: &mut Encoder, sort: char) {
    rbml_w.start_tag(tag_item_trait_item_sort);
    rbml_w.writer.write(&[ sort as u8 ]);
    rbml_w.end_tag();
}

fn encode_parent_sort(rbml_w: &mut Encoder, sort: char) {
    rbml_w.start_tag(tag_item_trait_parent_sort);
    rbml_w.writer.write(&[ sort as u8 ]);
    rbml_w.end_tag();
}

fn encode_provided_source(rbml_w: &mut Encoder,
                          source_opt: Option<DefId>) {
    for source in source_opt.iter() {
        rbml_w.start_tag(tag_item_method_provided_source);
        let s = def_to_string(*source);
        rbml_w.writer.write(s.as_bytes());
        rbml_w.end_tag();
    }
}

/* Returns an index of items in this class */
fn encode_info_for_struct(ecx: &EncodeContext,
                          rbml_w: &mut Encoder,
                          fields: &[ty::field_ty],
                          global_index: &mut Vec<entry<i64>>)
                          -> Vec<entry<i64>> {
    /* Each class has its own index, since different classes
       may have fields with the same name */
    let mut index = Vec::new();
     /* We encode both private and public fields -- need to include
        private fields to get the offsets right */
    for field in fields.iter() {
        let nm = field.name;
        let id = field.id.node;

        index.push(entry {val: id as i64, pos: rbml_w.writer.tell().unwrap()});
        global_index.push(entry {
            val: id as i64,
            pos: rbml_w.writer.tell().unwrap(),
        });
        rbml_w.start_tag(tag_items_data_item);
        debug!("encode_info_for_struct: doing {} {}",
               token::get_name(nm), id);
        encode_struct_field_family(rbml_w, field.vis);
        encode_name(rbml_w, nm);
        encode_bounds_and_type(rbml_w, ecx,
                               &lookup_item_type(ecx.tcx, local_def(id)));
        encode_def_id(rbml_w, local_def(id));

        let stab = stability::lookup(ecx.tcx, field.id);
        encode_stability(rbml_w, stab);

        rbml_w.end_tag();
    }
    index
}

fn encode_info_for_struct_ctor(ecx: &EncodeContext,
                               rbml_w: &mut Encoder,
                               name: ast::Ident,
                               ctor_id: NodeId,
                               index: &mut Vec<entry<i64>>,
                               struct_id: NodeId) {
    index.push(entry {
        val: ctor_id as i64,
        pos: rbml_w.writer.tell().unwrap(),
    });

    rbml_w.start_tag(tag_items_data_item);
    encode_def_id(rbml_w, local_def(ctor_id));
    encode_family(rbml_w, 'f');
    encode_bounds_and_type(rbml_w, ecx,
                           &lookup_item_type(ecx.tcx, local_def(ctor_id)));
    encode_name(rbml_w, name.name);
    ecx.tcx.map.with_path(ctor_id, |path| encode_path(rbml_w, path));
    encode_parent_item(rbml_w, local_def(struct_id));

    if ecx.item_symbols.borrow().contains_key(&ctor_id) {
        encode_symbol(ecx, rbml_w, ctor_id);
    }

    let stab = stability::lookup(ecx.tcx, ast_util::local_def(ctor_id));
    encode_stability(rbml_w, stab);

    // indicate that this is a tuple struct ctor, because downstream users will normally want
    // the tuple struct definition, but without this there is no way for them to tell that
    // they actually have a ctor rather than a normal function
    rbml_w.start_tag(tag_items_data_item_is_tuple_struct_ctor);
    rbml_w.end_tag();

    rbml_w.end_tag();
}

fn encode_generics(rbml_w: &mut Encoder,
                   ecx: &EncodeContext,
                   generics: &ty::Generics,
                   tag: uint)
{
    rbml_w.start_tag(tag);

    // Type parameters
    let ty_str_ctxt = &tyencode::ctxt {
        diag: ecx.diag,
        ds: def_to_string,
        tcx: ecx.tcx,
        abbrevs: &ecx.type_abbrevs
    };
    for param in generics.types.iter() {
        rbml_w.start_tag(tag_type_param_def);
        tyencode::enc_type_param_def(rbml_w.writer, ty_str_ctxt, param);
        rbml_w.end_tag();
    }

    // Region parameters
    for param in generics.regions.iter() {
        rbml_w.start_tag(tag_region_param_def);

        rbml_w.start_tag(tag_region_param_def_ident);
        encode_name(rbml_w, param.name);
        rbml_w.end_tag();

        rbml_w.wr_tagged_str(tag_region_param_def_def_id,
                             def_to_string(param.def_id).as_slice());

        rbml_w.wr_tagged_u64(tag_region_param_def_space,
                             param.space.to_uint() as u64);

        rbml_w.wr_tagged_u64(tag_region_param_def_index,
                             param.index as u64);

        for &bound_region in param.bounds.iter() {
            encode_region(ecx, rbml_w, bound_region);
        }

        rbml_w.end_tag();
    }

    rbml_w.end_tag();
}

fn encode_method_ty_fields(ecx: &EncodeContext,
                           rbml_w: &mut Encoder,
                           method_ty: &ty::Method) {
    encode_def_id(rbml_w, method_ty.def_id);
    encode_name(rbml_w, method_ty.ident.name);
    encode_generics(rbml_w, ecx, &method_ty.generics,
                    tag_method_ty_generics);
    encode_method_fty(ecx, rbml_w, &method_ty.fty);
    encode_visibility(rbml_w, method_ty.vis);
    encode_explicit_self(rbml_w, &method_ty.explicit_self);
    let fn_style = method_ty.fty.fn_style;
    match method_ty.explicit_self {
        ty::StaticExplicitSelfCategory => {
            encode_family(rbml_w, fn_style_static_method_family(fn_style));
        }
        _ => encode_family(rbml_w, style_fn_family(fn_style))
    }
    encode_provided_source(rbml_w, method_ty.provided_source);
}

fn encode_info_for_method(ecx: &EncodeContext,
                          rbml_w: &mut Encoder,
                          m: &ty::Method,
                          impl_path: PathElems,
                          is_default_impl: bool,
                          parent_id: NodeId,
                          ast_method_opt: Option<Gc<Method>>) {

    debug!("encode_info_for_method: {:?} {}", m.def_id,
           token::get_ident(m.ident));
    rbml_w.start_tag(tag_items_data_item);

    encode_method_ty_fields(ecx, rbml_w, m);
    encode_parent_item(rbml_w, local_def(parent_id));
    encode_item_sort(rbml_w, 'r');

    let stab = stability::lookup(ecx.tcx, m.def_id);
    encode_stability(rbml_w, stab);

    // The type for methods gets encoded twice, which is unfortunate.
    let pty = lookup_item_type(ecx.tcx, m.def_id);
    encode_bounds_and_type(rbml_w, ecx, &pty);

    let elem = ast_map::PathName(m.ident.name);
    encode_path(rbml_w, impl_path.chain(Some(elem).move_iter()));
    match ast_method_opt {
        Some(ast_method) => {
            encode_attributes(rbml_w, ast_method.attrs.as_slice())
        }
        None => ()
    }

    for &ast_method in ast_method_opt.iter() {
        let any_types = !pty.generics.types.is_empty();
        if any_types || is_default_impl || should_inline(ast_method.attrs.as_slice()) {
            encode_inlined_item(ecx,
                                rbml_w,
                                IITraitItemRef(local_def(parent_id),
                                               RequiredInlinedTraitItemRef(
                                                   &*ast_method)));
        }
        if !any_types {
            encode_symbol(ecx, rbml_w, m.def_id.node);
        }
        encode_method_argument_names(rbml_w, &*ast_method.pe_fn_decl());
    }

    rbml_w.end_tag();
}

fn encode_method_argument_names(rbml_w: &mut Encoder,
                                decl: &ast::FnDecl) {
    rbml_w.start_tag(tag_method_argument_names);
    for arg in decl.inputs.iter() {
        rbml_w.start_tag(tag_method_argument_name);
        match arg.pat.node {
            ast::PatIdent(_, ref path1, _) => {
                let name = token::get_ident(path1.node);
                rbml_w.writer.write(name.get().as_bytes());
            }
            _ => {}
        }
        rbml_w.end_tag();
    }
    rbml_w.end_tag();
}

fn encode_inlined_item(ecx: &EncodeContext,
                       rbml_w: &mut Encoder,
                       ii: InlinedItemRef) {
    let mut eii = ecx.encode_inlined_item.borrow_mut();
    let eii: &mut EncodeInlinedItem = &mut *eii;
    (*eii)(ecx, rbml_w, ii)
}

fn style_fn_family(s: FnStyle) -> char {
    match s {
        UnsafeFn => 'u',
        NormalFn => 'f',
    }
}

fn fn_style_static_method_family(s: FnStyle) -> char {
    match s {
        UnsafeFn => 'U',
        NormalFn => 'F',
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
                                   rbml_w: &mut Encoder,
                                   def_id: DefId) {
    match ecx.tcx.inherent_impls.borrow().find(&def_id) {
        None => {}
        Some(implementations) => {
            for &impl_def_id in implementations.borrow().iter() {
                rbml_w.start_tag(tag_items_data_item_inherent_impl);
                encode_def_id(rbml_w, impl_def_id);
                rbml_w.end_tag();
            }
        }
    }
}

// Encodes the implementations of a trait defined in this crate.
fn encode_extension_implementations(ecx: &EncodeContext,
                                    rbml_w: &mut Encoder,
                                    trait_def_id: DefId) {
    match ecx.tcx.trait_impls.borrow().find(&trait_def_id) {
        None => {}
        Some(implementations) => {
            for &impl_def_id in implementations.borrow().iter() {
                rbml_w.start_tag(tag_items_data_item_extension_impl);
                encode_def_id(rbml_w, impl_def_id);
                rbml_w.end_tag();
            }
        }
    }
}

fn encode_stability(rbml_w: &mut Encoder, stab_opt: Option<attr::Stability>) {
    stab_opt.map(|stab| {
        rbml_w.start_tag(tag_items_data_item_stability);
        stab.encode(rbml_w).unwrap();
        rbml_w.end_tag();
    });
}

fn encode_info_for_item(ecx: &EncodeContext,
                        rbml_w: &mut Encoder,
                        item: &Item,
                        index: &mut Vec<entry<i64>>,
                        path: PathElems,
                        vis: ast::Visibility) {
    let tcx = ecx.tcx;

    fn add_to_index(item: &Item, rbml_w: &Encoder,
                    index: &mut Vec<entry<i64>>) {
        index.push(entry {
            val: item.id as i64,
            pos: rbml_w.writer.tell().unwrap(),
        });
    }

    debug!("encoding info for item at {}",
           tcx.sess.codemap().span_to_string(item.span));

    let def_id = local_def(item.id);
    let stab = stability::lookup(tcx, ast_util::local_def(item.id));

    match item.node {
      ItemStatic(_, m, _) => {
        add_to_index(item, rbml_w, index);
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id(rbml_w, def_id);
        if m == ast::MutMutable {
            encode_family(rbml_w, 'b');
        } else {
            encode_family(rbml_w, 'c');
        }
        encode_bounds_and_type(rbml_w, ecx, &lookup_item_type(tcx, def_id));
        encode_symbol(ecx, rbml_w, item.id);
        encode_name(rbml_w, item.ident.name);
        encode_path(rbml_w, path);

        let inlineable = !ecx.non_inlineable_statics.borrow().contains(&item.id);

        if inlineable {
            encode_inlined_item(ecx, rbml_w, IIItemRef(item));
        }
        encode_visibility(rbml_w, vis);
        encode_stability(rbml_w, stab);
        rbml_w.end_tag();
      }
      ItemFn(ref decl, fn_style, _, ref generics, _) => {
        add_to_index(item, rbml_w, index);
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id(rbml_w, def_id);
        encode_family(rbml_w, style_fn_family(fn_style));
        let tps_len = generics.ty_params.len();
        encode_bounds_and_type(rbml_w, ecx, &lookup_item_type(tcx, def_id));
        encode_name(rbml_w, item.ident.name);
        encode_path(rbml_w, path);
        encode_attributes(rbml_w, item.attrs.as_slice());
        if tps_len > 0u || should_inline(item.attrs.as_slice()) {
            encode_inlined_item(ecx, rbml_w, IIItemRef(item));
        }
        if tps_len == 0 {
            encode_symbol(ecx, rbml_w, item.id);
        }
        encode_visibility(rbml_w, vis);
        encode_stability(rbml_w, stab);
        encode_method_argument_names(rbml_w, &**decl);
        rbml_w.end_tag();
      }
      ItemMod(ref m) => {
        add_to_index(item, rbml_w, index);
        encode_info_for_mod(ecx,
                            rbml_w,
                            m,
                            item.attrs.as_slice(),
                            item.id,
                            path,
                            item.ident,
                            item.vis);
      }
      ItemForeignMod(ref fm) => {
        add_to_index(item, rbml_w, index);
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id(rbml_w, def_id);
        encode_family(rbml_w, 'n');
        encode_name(rbml_w, item.ident.name);
        encode_path(rbml_w, path);

        // Encode all the items in this module.
        for foreign_item in fm.items.iter() {
            rbml_w.start_tag(tag_mod_child);
            rbml_w.wr_str(def_to_string(local_def(foreign_item.id)).as_slice());
            rbml_w.end_tag();
        }
        encode_visibility(rbml_w, vis);
        encode_stability(rbml_w, stab);
        rbml_w.end_tag();
      }
      ItemTy(..) => {
        add_to_index(item, rbml_w, index);
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id(rbml_w, def_id);
        encode_family(rbml_w, 'y');
        encode_bounds_and_type(rbml_w, ecx, &lookup_item_type(tcx, def_id));
        encode_name(rbml_w, item.ident.name);
        encode_path(rbml_w, path);
        encode_visibility(rbml_w, vis);
        encode_stability(rbml_w, stab);
        rbml_w.end_tag();
      }
      ItemEnum(ref enum_definition, _) => {
        add_to_index(item, rbml_w, index);

        rbml_w.start_tag(tag_items_data_item);
        encode_def_id(rbml_w, def_id);
        encode_family(rbml_w, 't');
        encode_item_variances(rbml_w, ecx, item.id);
        encode_bounds_and_type(rbml_w, ecx, &lookup_item_type(tcx, def_id));
        encode_name(rbml_w, item.ident.name);
        encode_attributes(rbml_w, item.attrs.as_slice());
        for v in (*enum_definition).variants.iter() {
            encode_variant_id(rbml_w, local_def(v.node.id));
        }
        encode_inlined_item(ecx, rbml_w, IIItemRef(item));
        encode_path(rbml_w, path);

        // Encode inherent implementations for this enumeration.
        encode_inherent_implementations(ecx, rbml_w, def_id);

        encode_visibility(rbml_w, vis);
        encode_stability(rbml_w, stab);
        rbml_w.end_tag();

        encode_enum_variant_info(ecx,
                                 rbml_w,
                                 item.id,
                                 (*enum_definition).variants.as_slice(),
                                 index);
      }
      ItemStruct(struct_def, _) => {
        let fields = ty::lookup_struct_fields(tcx, def_id);

        /* First, encode the fields
           These come first because we need to write them to make
           the index, and the index needs to be in the item for the
           class itself */
        let idx = encode_info_for_struct(ecx,
                                         rbml_w,
                                         fields.as_slice(),
                                         index);

        /* Index the class*/
        add_to_index(item, rbml_w, index);

        /* Now, make an item for the class itself */
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id(rbml_w, def_id);
        encode_family(rbml_w, 'S');
        encode_bounds_and_type(rbml_w, ecx, &lookup_item_type(tcx, def_id));

        encode_item_variances(rbml_w, ecx, item.id);
        encode_name(rbml_w, item.ident.name);
        encode_attributes(rbml_w, item.attrs.as_slice());
        encode_path(rbml_w, path.clone());
        encode_stability(rbml_w, stab);
        encode_visibility(rbml_w, vis);

        /* Encode def_ids for each field and method
         for methods, write all the stuff get_trait_method
        needs to know*/
        encode_struct_fields(rbml_w, fields.as_slice(), def_id);

        encode_inlined_item(ecx, rbml_w, IIItemRef(item));

        // Encode inherent implementations for this structure.
        encode_inherent_implementations(ecx, rbml_w, def_id);

        /* Each class has its own index -- encode it */
        encode_index(rbml_w, idx, write_i64);
        rbml_w.end_tag();

        // If this is a tuple-like struct, encode the type of the constructor.
        match struct_def.ctor_id {
            Some(ctor_id) => {
                encode_info_for_struct_ctor(ecx, rbml_w, item.ident,
                                            ctor_id, index, def_id.node);
            }
            None => {}
        }
      }
      ItemImpl(_, ref opt_trait, ty, ref ast_items) => {
        // We need to encode information about the default methods we
        // have inherited, so we drive this based on the impl structure.
        let impl_items = tcx.impl_items.borrow();
        let items = impl_items.get(&def_id);

        add_to_index(item, rbml_w, index);
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id(rbml_w, def_id);
        encode_family(rbml_w, 'i');
        encode_bounds_and_type(rbml_w, ecx, &lookup_item_type(tcx, def_id));
        encode_name(rbml_w, item.ident.name);
        encode_attributes(rbml_w, item.attrs.as_slice());
        match ty.node {
            ast::TyPath(ref path, ref bounds, _) if path.segments
                                                        .len() == 1 => {
                let ident = path.segments.last().unwrap().identifier;
                assert!(bounds.is_none());
                encode_impl_type_basename(rbml_w, ident);
            }
            _ => {}
        }
        for &item_def_id in items.iter() {
            rbml_w.start_tag(tag_item_impl_item);
            match item_def_id {
                ty::MethodTraitItemId(item_def_id) => {
                    encode_def_id(rbml_w, item_def_id);
                    encode_item_sort(rbml_w, 'r');
                }
            }
            rbml_w.end_tag();
        }
        for ast_trait_ref in opt_trait.iter() {
            let trait_ref = ty::node_id_to_trait_ref(
                tcx, ast_trait_ref.ref_id);
            encode_trait_ref(rbml_w, ecx, &*trait_ref, tag_item_trait_ref);
            let impl_vtables = ty::lookup_impl_vtables(tcx, def_id);
            encode_impl_vtables(rbml_w, ecx, &impl_vtables);
        }
        encode_path(rbml_w, path.clone());
        encode_stability(rbml_w, stab);
        rbml_w.end_tag();

        // Iterate down the trait items, emitting them. We rely on the
        // assumption that all of the actually implemented trait items
        // appear first in the impl structure, in the same order they do
        // in the ast. This is a little sketchy.
        let num_implemented_methods = ast_items.len();
        for (i, &trait_item_def_id) in items.iter().enumerate() {
            let ast_item = if i < num_implemented_methods {
                Some(*ast_items.get(i))
            } else {
                None
            };

            index.push(entry {
                val: trait_item_def_id.def_id().node as i64,
                pos: rbml_w.writer.tell().unwrap(),
            });

            let trait_item_type =
                ty::impl_or_trait_item(tcx, trait_item_def_id.def_id());
            match (trait_item_type, ast_item) {
                (ty::MethodTraitItem(method_type),
                 Some(ast::MethodImplItem(ast_method))) => {
                    encode_info_for_method(ecx,
                                           rbml_w,
                                           &*method_type,
                                           path.clone(),
                                           false,
                                           item.id,
                                           Some(ast_method))
                }
                (ty::MethodTraitItem(method_type), None) => {
                    encode_info_for_method(ecx,
                                           rbml_w,
                                           &*method_type,
                                           path.clone(),
                                           false,
                                           item.id,
                                           None)
                }
            }
        }
      }
      ItemTrait(_, _, _, ref ms) => {
        add_to_index(item, rbml_w, index);
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id(rbml_w, def_id);
        encode_family(rbml_w, 'I');
        encode_item_variances(rbml_w, ecx, item.id);
        let trait_def = ty::lookup_trait_def(tcx, def_id);
        encode_generics(rbml_w, ecx, &trait_def.generics, tag_item_generics);
        encode_trait_ref(rbml_w, ecx, &*trait_def.trait_ref, tag_item_trait_ref);
        encode_name(rbml_w, item.ident.name);
        encode_attributes(rbml_w, item.attrs.as_slice());
        encode_visibility(rbml_w, vis);
        encode_stability(rbml_w, stab);
        for &method_def_id in ty::trait_item_def_ids(tcx, def_id).iter() {
            rbml_w.start_tag(tag_item_trait_item);
            match method_def_id {
                ty::MethodTraitItemId(method_def_id) => {
                    encode_def_id(rbml_w, method_def_id);
                    encode_item_sort(rbml_w, 'r');
                }
            }
            rbml_w.end_tag();

            rbml_w.start_tag(tag_mod_child);
            rbml_w.wr_str(def_to_string(method_def_id.def_id()).as_slice());
            rbml_w.end_tag();
        }
        encode_path(rbml_w, path.clone());

        encode_bounds(rbml_w, ecx, &trait_def.bounds, tag_trait_def_bounds);

        // Encode the implementations of this trait.
        encode_extension_implementations(ecx, rbml_w, def_id);

        rbml_w.end_tag();

        // Now output the trait item info for each trait item.
        let r = ty::trait_item_def_ids(tcx, def_id);
        for (i, &item_def_id) in r.iter().enumerate() {
            assert_eq!(item_def_id.def_id().krate, ast::LOCAL_CRATE);

            index.push(entry {
                val: item_def_id.def_id().node as i64,
                pos: rbml_w.writer.tell().unwrap(),
            });

            rbml_w.start_tag(tag_items_data_item);

            let trait_item_type =
                ty::impl_or_trait_item(tcx, item_def_id.def_id());
            match trait_item_type {
                 ty::MethodTraitItem(method_ty) => {
                    let method_def_id = item_def_id.def_id();

                    encode_method_ty_fields(ecx, rbml_w, &*method_ty);
                    encode_parent_item(rbml_w, def_id);

                    let stab = stability::lookup(tcx, method_def_id);
                    encode_stability(rbml_w, stab);

                    let elem = ast_map::PathName(method_ty.ident.name);
                    encode_path(rbml_w,
                                path.clone().chain(Some(elem).move_iter()));

                    match method_ty.explicit_self {
                        ty::StaticExplicitSelfCategory => {
                            encode_family(rbml_w,
                                          fn_style_static_method_family(
                                              method_ty.fty.fn_style));

                            let pty = ty::lookup_item_type(tcx,
                                                           method_def_id);
                            encode_bounds_and_type(rbml_w, ecx, &pty);
                        }

                        _ => {
                            encode_family(rbml_w,
                                          style_fn_family(
                                              method_ty.fty.fn_style));
                        }
                    }

                    match ms.get(i) {
                        &RequiredMethod(ref tm) => {
                            encode_attributes(rbml_w, tm.attrs.as_slice());
                            encode_item_sort(rbml_w, 'r');
                            encode_parent_sort(rbml_w, 't');
                            encode_method_argument_names(rbml_w, &*tm.decl);
                        }

                        &ProvidedMethod(m) => {
                            encode_attributes(rbml_w, m.attrs.as_slice());
                            // If this is a static method, we've already
                            // encoded this.
                            if method_ty.explicit_self !=
                                    ty::StaticExplicitSelfCategory {
                                // FIXME: I feel like there is something funny
                                // going on.
                                let pty = ty::lookup_item_type(tcx, method_def_id);
                                encode_bounds_and_type(rbml_w, ecx, &pty);
                            }
                            encode_item_sort(rbml_w, 'p');
                            encode_parent_sort(rbml_w, 't');
                            encode_inlined_item(
                                ecx,
                                rbml_w,
                                IITraitItemRef(
                                    def_id,
                                    ProvidedInlinedTraitItemRef(&*m)));
                            encode_method_argument_names(rbml_w,
                                                         &*m.pe_fn_decl());
                        }
                    }
                }
            }

            rbml_w.end_tag();
        }

        // Encode inherent implementations for this trait.
        encode_inherent_implementations(ecx, rbml_w, def_id);
      }
      ItemMac(..) => {
        // macros are encoded separately
      }
    }
}

fn encode_info_for_foreign_item(ecx: &EncodeContext,
                                rbml_w: &mut Encoder,
                                nitem: &ForeignItem,
                                index: &mut Vec<entry<i64>>,
                                path: PathElems,
                                abi: abi::Abi) {
    index.push(entry {
        val: nitem.id as i64,
        pos: rbml_w.writer.tell().unwrap(),
    });

    rbml_w.start_tag(tag_items_data_item);
    encode_def_id(rbml_w, local_def(nitem.id));
    encode_visibility(rbml_w, nitem.vis);
    match nitem.node {
      ForeignItemFn(..) => {
        encode_family(rbml_w, style_fn_family(NormalFn));
        encode_bounds_and_type(rbml_w, ecx,
                               &lookup_item_type(ecx.tcx,local_def(nitem.id)));
        encode_name(rbml_w, nitem.ident.name);
        if abi == abi::RustIntrinsic {
            encode_inlined_item(ecx, rbml_w, IIForeignRef(nitem));
        }
        encode_symbol(ecx, rbml_w, nitem.id);
      }
      ForeignItemStatic(_, mutbl) => {
        if mutbl {
            encode_family(rbml_w, 'b');
        } else {
            encode_family(rbml_w, 'c');
        }
        encode_bounds_and_type(rbml_w, ecx,
                               &lookup_item_type(ecx.tcx,local_def(nitem.id)));
        encode_symbol(ecx, rbml_w, nitem.id);
        encode_name(rbml_w, nitem.ident.name);
      }
    }
    encode_path(rbml_w, path);
    rbml_w.end_tag();
}

fn my_visit_expr(_e: &Expr) { }

fn my_visit_item(i: &Item,
                 rbml_w: &mut Encoder,
                 ecx_ptr: *const int,
                 index: &mut Vec<entry<i64>>) {
    let mut rbml_w = unsafe { rbml_w.unsafe_clone() };
    // See above
    let ecx: &EncodeContext = unsafe { mem::transmute(ecx_ptr) };
    ecx.tcx.map.with_path(i.id, |path| {
        encode_info_for_item(ecx, &mut rbml_w, i, index, path, i.vis);
    });
}

fn my_visit_foreign_item(ni: &ForeignItem,
                         rbml_w: &mut Encoder,
                         ecx_ptr:*const int,
                         index: &mut Vec<entry<i64>>) {
    // See above
    let ecx: &EncodeContext = unsafe { mem::transmute(ecx_ptr) };
    debug!("writing foreign item {}::{}",
            ecx.tcx.map.path_to_string(ni.id),
            token::get_ident(ni.ident));

    let mut rbml_w = unsafe {
        rbml_w.unsafe_clone()
    };
    let abi = ecx.tcx.map.get_foreign_abi(ni.id);
    ecx.tcx.map.with_path(ni.id, |path| {
        encode_info_for_foreign_item(ecx, &mut rbml_w,
                                     ni, index,
                                     path, abi);
    });
}

struct EncodeVisitor<'a,'b:'a> {
    rbml_w_for_visit_item: &'a mut Encoder<'b>,
    ecx_ptr:*const int,
    index: &'a mut Vec<entry<i64>>,
}

impl<'a,'b> visit::Visitor<()> for EncodeVisitor<'a,'b> {
    fn visit_expr(&mut self, ex: &Expr, _: ()) {
        visit::walk_expr(self, ex, ());
        my_visit_expr(ex);
    }
    fn visit_item(&mut self, i: &Item, _: ()) {
        visit::walk_item(self, i, ());
        my_visit_item(i,
                      self.rbml_w_for_visit_item,
                      self.ecx_ptr,
                      self.index);
    }
    fn visit_foreign_item(&mut self, ni: &ForeignItem, _: ()) {
        visit::walk_foreign_item(self, ni, ());
        my_visit_foreign_item(ni,
                              self.rbml_w_for_visit_item,
                              self.ecx_ptr,
                              self.index);
    }
}

fn encode_info_for_items(ecx: &EncodeContext,
                         rbml_w: &mut Encoder,
                         krate: &Crate)
                         -> Vec<entry<i64>> {
    let mut index = Vec::new();
    rbml_w.start_tag(tag_items_data);
    index.push(entry {
        val: CRATE_NODE_ID as i64,
        pos: rbml_w.writer.tell().unwrap(),
    });
    encode_info_for_mod(ecx,
                        rbml_w,
                        &krate.module,
                        &[],
                        CRATE_NODE_ID,
                        ast_map::Values([].iter()).chain(None),
                        syntax::parse::token::special_idents::invalid,
                        Public);

    // See comment in `encode_side_tables_for_ii` in astencode
    let ecx_ptr: *const int = unsafe { mem::transmute(ecx) };
    visit::walk_crate(&mut EncodeVisitor {
        index: &mut index,
        ecx_ptr: ecx_ptr,
        rbml_w_for_visit_item: &mut *rbml_w,
    }, krate, ());

    rbml_w.end_tag();
    index
}


// Path and definition ID indexing

fn encode_index<T: Hash>(rbml_w: &mut Encoder, index: Vec<entry<T>>,
                         write_fn: |&mut SeekableMemWriter, &T|) {
    let mut buckets: Vec<Vec<entry<T>>> = Vec::from_fn(256, |_| Vec::new());
    for elt in index.move_iter() {
        let h = hash::hash(&elt.val) as uint;
        buckets.get_mut(h % 256).push(elt);
    }

    rbml_w.start_tag(tag_index);
    let mut bucket_locs = Vec::new();
    rbml_w.start_tag(tag_index_buckets);
    for bucket in buckets.iter() {
        bucket_locs.push(rbml_w.writer.tell().unwrap());
        rbml_w.start_tag(tag_index_buckets_bucket);
        for elt in bucket.iter() {
            rbml_w.start_tag(tag_index_buckets_bucket_elt);
            assert!(elt.pos < 0xffff_ffff);
            {
                let wr: &mut SeekableMemWriter = rbml_w.writer;
                wr.write_be_u32(elt.pos as u32);
            }
            write_fn(rbml_w.writer, &elt.val);
            rbml_w.end_tag();
        }
        rbml_w.end_tag();
    }
    rbml_w.end_tag();
    rbml_w.start_tag(tag_index_table);
    for pos in bucket_locs.iter() {
        assert!(*pos < 0xffff_ffff);
        let wr: &mut SeekableMemWriter = rbml_w.writer;
        wr.write_be_u32(*pos as u32);
    }
    rbml_w.end_tag();
    rbml_w.end_tag();
}

fn write_i64(writer: &mut SeekableMemWriter, &n: &i64) {
    let wr: &mut SeekableMemWriter = writer;
    assert!(n < 0x7fff_ffff);
    wr.write_be_u32(n as u32);
}

fn encode_meta_item(rbml_w: &mut Encoder, mi: Gc<MetaItem>) {
    match mi.node {
      MetaWord(ref name) => {
        rbml_w.start_tag(tag_meta_item_word);
        rbml_w.start_tag(tag_meta_item_name);
        rbml_w.writer.write(name.get().as_bytes());
        rbml_w.end_tag();
        rbml_w.end_tag();
      }
      MetaNameValue(ref name, ref value) => {
        match value.node {
          LitStr(ref value, _) => {
            rbml_w.start_tag(tag_meta_item_name_value);
            rbml_w.start_tag(tag_meta_item_name);
            rbml_w.writer.write(name.get().as_bytes());
            rbml_w.end_tag();
            rbml_w.start_tag(tag_meta_item_value);
            rbml_w.writer.write(value.get().as_bytes());
            rbml_w.end_tag();
            rbml_w.end_tag();
          }
          _ => {/* FIXME (#623): encode other variants */ }
        }
      }
      MetaList(ref name, ref items) => {
        rbml_w.start_tag(tag_meta_item_list);
        rbml_w.start_tag(tag_meta_item_name);
        rbml_w.writer.write(name.get().as_bytes());
        rbml_w.end_tag();
        for inner_item in items.iter() {
            encode_meta_item(rbml_w, *inner_item);
        }
        rbml_w.end_tag();
      }
    }
}

fn encode_attributes(rbml_w: &mut Encoder, attrs: &[Attribute]) {
    rbml_w.start_tag(tag_attributes);
    for attr in attrs.iter() {
        rbml_w.start_tag(tag_attribute);
        rbml_w.wr_tagged_u8(tag_attribute_is_sugared_doc, attr.node.is_sugared_doc as u8);
        encode_meta_item(rbml_w, attr.node.value);
        rbml_w.end_tag();
    }
    rbml_w.end_tag();
}

fn encode_crate_deps(rbml_w: &mut Encoder, cstore: &cstore::CStore) {
    fn get_ordered_deps(cstore: &cstore::CStore) -> Vec<decoder::CrateDep> {
        // Pull the cnums and name,vers,hash out of cstore
        let mut deps = Vec::new();
        cstore.iter_crate_data(|key, val| {
            let dep = decoder::CrateDep {
                cnum: key,
                name: decoder::get_crate_name(val.data()),
                hash: decoder::get_crate_hash(val.data()),
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
    rbml_w.start_tag(tag_crate_deps);
    let r = get_ordered_deps(cstore);
    for dep in r.iter() {
        encode_crate_dep(rbml_w, (*dep).clone());
    }
    rbml_w.end_tag();
}

fn encode_lang_items(ecx: &EncodeContext, rbml_w: &mut Encoder) {
    rbml_w.start_tag(tag_lang_items);

    for (i, def_id) in ecx.tcx.lang_items.items() {
        for id in def_id.iter() {
            if id.krate == LOCAL_CRATE {
                rbml_w.start_tag(tag_lang_items_item);

                rbml_w.start_tag(tag_lang_items_item_id);
                {
                    let wr: &mut SeekableMemWriter = rbml_w.writer;
                    wr.write_be_u32(i as u32);
                }
                rbml_w.end_tag();   // tag_lang_items_item_id

                rbml_w.start_tag(tag_lang_items_item_node_id);
                {
                    let wr: &mut SeekableMemWriter = rbml_w.writer;
                    wr.write_be_u32(id.node as u32);
                }
                rbml_w.end_tag();   // tag_lang_items_item_node_id

                rbml_w.end_tag();   // tag_lang_items_item
            }
        }
    }

    for i in ecx.tcx.lang_items.missing.iter() {
        rbml_w.wr_tagged_u32(tag_lang_items_missing, *i as u32);
    }

    rbml_w.end_tag();   // tag_lang_items
}

fn encode_native_libraries(ecx: &EncodeContext, rbml_w: &mut Encoder) {
    rbml_w.start_tag(tag_native_libraries);

    for &(ref lib, kind) in ecx.tcx.sess.cstore.get_used_libraries()
                               .borrow().iter() {
        match kind {
            cstore::NativeStatic => {} // these libraries are not propagated
            cstore::NativeFramework | cstore::NativeUnknown => {
                rbml_w.start_tag(tag_native_libraries_lib);

                rbml_w.start_tag(tag_native_libraries_kind);
                rbml_w.writer.write_be_u32(kind as u32);
                rbml_w.end_tag();

                rbml_w.start_tag(tag_native_libraries_name);
                rbml_w.writer.write(lib.as_bytes());
                rbml_w.end_tag();

                rbml_w.end_tag();
            }
        }
    }

    rbml_w.end_tag();
}

fn encode_plugin_registrar_fn(ecx: &EncodeContext, rbml_w: &mut Encoder) {
    match ecx.tcx.sess.plugin_registrar_fn.get() {
        Some(id) => { rbml_w.wr_tagged_u32(tag_plugin_registrar_fn, id); }
        None => {}
    }
}

/// Given a span, write the text of that span into the output stream
/// as an exported macro
fn encode_macro_def(ecx: &EncodeContext,
                    rbml_w: &mut Encoder,
                    span: &syntax::codemap::Span) {
    let def = ecx.tcx.sess.codemap().span_to_snippet(*span)
        .expect("Unable to find source for macro");
    rbml_w.start_tag(tag_macro_def);
    rbml_w.wr_str(def.as_slice());
    rbml_w.end_tag();
}

/// Serialize the text of the exported macros
fn encode_macro_defs(ecx: &EncodeContext,
                     krate: &Crate,
                     rbml_w: &mut Encoder) {
    rbml_w.start_tag(tag_exported_macros);
    for item in krate.exported_macros.iter() {
        encode_macro_def(ecx, rbml_w, &item.span);
    }
    rbml_w.end_tag();
}

fn encode_unboxed_closures<'a>(
                           ecx: &'a EncodeContext,
                           rbml_w: &'a mut Encoder) {
    rbml_w.start_tag(tag_unboxed_closures);
    for (unboxed_closure_id, unboxed_closure) in ecx.tcx
                                                    .unboxed_closures
                                                    .borrow()
                                                    .iter() {
        if unboxed_closure_id.krate != LOCAL_CRATE {
            continue
        }

        rbml_w.start_tag(tag_unboxed_closure);
        encode_def_id(rbml_w, *unboxed_closure_id);
        rbml_w.start_tag(tag_unboxed_closure_type);
        write_closure_type(ecx, rbml_w, &unboxed_closure.closure_type);
        rbml_w.end_tag();
        encode_unboxed_closure_kind(rbml_w, unboxed_closure.kind);
        rbml_w.end_tag();
    }
    rbml_w.end_tag();
}

fn encode_struct_field_attrs(rbml_w: &mut Encoder, krate: &Crate) {
    struct StructFieldVisitor<'a, 'b:'a> {
        rbml_w: &'a mut Encoder<'b>,
    }

    impl<'a, 'b> Visitor<()> for StructFieldVisitor<'a, 'b> {
        fn visit_struct_field(&mut self, field: &ast::StructField, _: ()) {
            self.rbml_w.start_tag(tag_struct_field);
            self.rbml_w.wr_tagged_u32(tag_struct_field_id, field.node.id);
            encode_attributes(self.rbml_w, field.node.attrs.as_slice());
            self.rbml_w.end_tag();
        }
    }

    rbml_w.start_tag(tag_struct_fields);
    visit::walk_crate(&mut StructFieldVisitor {
        rbml_w: rbml_w
    }, krate, ());
    rbml_w.end_tag();
}



struct ImplVisitor<'a,'b:'a,'c:'a> {
    ecx: &'a EncodeContext<'b>,
    rbml_w: &'a mut Encoder<'c>,
}

impl<'a,'b,'c> Visitor<()> for ImplVisitor<'a,'b,'c> {
    fn visit_item(&mut self, item: &Item, _: ()) {
        match item.node {
            ItemImpl(_, Some(ref trait_ref), _, _) => {
                let def_map = &self.ecx.tcx.def_map;
                let trait_def = def_map.borrow().get_copy(&trait_ref.ref_id);
                let def_id = trait_def.def_id();

                // Load eagerly if this is an implementation of the Drop trait
                // or if the trait is not defined in this crate.
                if Some(def_id) == self.ecx.tcx.lang_items.drop_trait() ||
                        def_id.krate != LOCAL_CRATE {
                    self.rbml_w.start_tag(tag_impls_impl);
                    encode_def_id(self.rbml_w, local_def(item.id));
                    self.rbml_w.end_tag();
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
fn encode_impls<'a>(ecx: &'a EncodeContext,
                    krate: &Crate,
                    rbml_w: &'a mut Encoder) {
    rbml_w.start_tag(tag_impls);

    {
        let mut visitor = ImplVisitor {
            ecx: ecx,
            rbml_w: rbml_w,
        };
        visit::walk_crate(&mut visitor, krate, ());
    }

    rbml_w.end_tag();
}

fn encode_misc_info(ecx: &EncodeContext,
                    krate: &Crate,
                    rbml_w: &mut Encoder) {
    rbml_w.start_tag(tag_misc_info);
    rbml_w.start_tag(tag_misc_info_crate_items);
    for &item in krate.module.items.iter() {
        rbml_w.start_tag(tag_mod_child);
        rbml_w.wr_str(def_to_string(local_def(item.id)).as_slice());
        rbml_w.end_tag();

        each_auxiliary_node_id(item, |auxiliary_node_id| {
            rbml_w.start_tag(tag_mod_child);
            rbml_w.wr_str(def_to_string(local_def(
                        auxiliary_node_id)).as_slice());
            rbml_w.end_tag();
            true
        });
    }

    // Encode reexports for the root module.
    encode_reexports(ecx, rbml_w, 0, ast_map::Values([].iter()).chain(None));

    rbml_w.end_tag();
    rbml_w.end_tag();
}

fn encode_reachable_extern_fns(ecx: &EncodeContext, rbml_w: &mut Encoder) {
    rbml_w.start_tag(tag_reachable_extern_fns);

    for id in ecx.reachable.iter() {
        match ecx.tcx.map.find(*id) {
            Some(ast_map::NodeItem(i)) => {
                match i.node {
                    ast::ItemFn(_, _, abi, ref generics, _)
                                if abi != abi::Rust && !generics.is_type_parameterized() => {
                        rbml_w.wr_tagged_u32(tag_reachable_extern_fn_id, *id);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    rbml_w.end_tag();
}

fn encode_crate_dep(rbml_w: &mut Encoder,
                    dep: decoder::CrateDep) {
    rbml_w.start_tag(tag_crate_dep);
    rbml_w.start_tag(tag_crate_dep_crate_name);
    rbml_w.writer.write(dep.name.as_bytes());
    rbml_w.end_tag();
    rbml_w.start_tag(tag_crate_dep_hash);
    rbml_w.writer.write(dep.hash.as_str().as_bytes());
    rbml_w.end_tag();
    rbml_w.end_tag();
}

fn encode_hash(rbml_w: &mut Encoder, hash: &Svh) {
    rbml_w.start_tag(tag_crate_hash);
    rbml_w.writer.write(hash.as_str().as_bytes());
    rbml_w.end_tag();
}

fn encode_crate_name(rbml_w: &mut Encoder, crate_name: &str) {
    rbml_w.start_tag(tag_crate_crate_name);
    rbml_w.writer.write(crate_name.as_bytes());
    rbml_w.end_tag();
}

fn encode_crate_triple(rbml_w: &mut Encoder, triple: &str) {
    rbml_w.start_tag(tag_crate_triple);
    rbml_w.writer.write(triple.as_bytes());
    rbml_w.end_tag();
}

fn encode_dylib_dependency_formats(rbml_w: &mut Encoder, ecx: &EncodeContext) {
    rbml_w.start_tag(tag_dylib_dependency_formats);
    match ecx.tcx.dependency_formats.borrow().find(&config::CrateTypeDylib) {
        Some(arr) => {
            let s = arr.iter().enumerate().filter_map(|(i, slot)| {
                slot.map(|kind| (format!("{}:{}", i + 1, match kind {
                    cstore::RequireDynamic => "d",
                    cstore::RequireStatic => "s",
                })).to_string())
            }).collect::<Vec<String>>();
            rbml_w.writer.write(s.connect(",").as_bytes());
        }
        None => {}
    }
    rbml_w.end_tag();
}

// NB: Increment this as you change the metadata encoding version.
pub static metadata_encoding_version : &'static [u8] = &[b'r', b'u', b's', b't', 0, 0, 0, 1 ];

pub fn encode_metadata(parms: EncodeParams, krate: &Crate) -> Vec<u8> {
    let mut wr = SeekableMemWriter::new();
    encode_metadata_inner(&mut wr, parms, krate);
    wr.unwrap().move_iter().collect()
}

fn encode_metadata_inner(wr: &mut SeekableMemWriter, parms: EncodeParams, krate: &Crate) {
    struct Stats {
        attr_bytes: u64,
        dep_bytes: u64,
        lang_item_bytes: u64,
        native_lib_bytes: u64,
        plugin_registrar_fn_bytes: u64,
        macro_defs_bytes: u64,
        unboxed_closure_bytes: u64,
        impl_bytes: u64,
        misc_bytes: u64,
        item_bytes: u64,
        index_bytes: u64,
        zero_bytes: u64,
        total_bytes: u64,
    }
    let mut stats = Stats {
        attr_bytes: 0,
        dep_bytes: 0,
        lang_item_bytes: 0,
        native_lib_bytes: 0,
        plugin_registrar_fn_bytes: 0,
        macro_defs_bytes: 0,
        unboxed_closure_bytes: 0,
        impl_bytes: 0,
        misc_bytes: 0,
        item_bytes: 0,
        index_bytes: 0,
        zero_bytes: 0,
        total_bytes: 0,
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
        reachable,
        ..
    } = parms;
    let ecx = EncodeContext {
        diag: diag,
        tcx: tcx,
        reexports2: reexports2,
        item_symbols: item_symbols,
        non_inlineable_statics: non_inlineable_statics,
        link_meta: link_meta,
        cstore: cstore,
        encode_inlined_item: RefCell::new(encode_inlined_item),
        type_abbrevs: RefCell::new(HashMap::new()),
        reachable: reachable,
     };

    let mut rbml_w = writer::Encoder::new(wr);

    encode_crate_name(&mut rbml_w, ecx.link_meta.crate_name.as_slice());
    encode_crate_triple(&mut rbml_w,
                        tcx.sess
                           .targ_cfg
                           .target_strs
                           .target_triple
                           .as_slice());
    encode_hash(&mut rbml_w, &ecx.link_meta.crate_hash);
    encode_dylib_dependency_formats(&mut rbml_w, &ecx);

    let mut i = rbml_w.writer.tell().unwrap();
    encode_attributes(&mut rbml_w, krate.attrs.as_slice());
    stats.attr_bytes = rbml_w.writer.tell().unwrap() - i;

    i = rbml_w.writer.tell().unwrap();
    encode_crate_deps(&mut rbml_w, ecx.cstore);
    stats.dep_bytes = rbml_w.writer.tell().unwrap() - i;

    // Encode the language items.
    i = rbml_w.writer.tell().unwrap();
    encode_lang_items(&ecx, &mut rbml_w);
    stats.lang_item_bytes = rbml_w.writer.tell().unwrap() - i;

    // Encode the native libraries used
    i = rbml_w.writer.tell().unwrap();
    encode_native_libraries(&ecx, &mut rbml_w);
    stats.native_lib_bytes = rbml_w.writer.tell().unwrap() - i;

    // Encode the plugin registrar function
    i = rbml_w.writer.tell().unwrap();
    encode_plugin_registrar_fn(&ecx, &mut rbml_w);
    stats.plugin_registrar_fn_bytes = rbml_w.writer.tell().unwrap() - i;

    // Encode macro definitions
    i = rbml_w.writer.tell().unwrap();
    encode_macro_defs(&ecx, krate, &mut rbml_w);
    stats.macro_defs_bytes = rbml_w.writer.tell().unwrap() - i;

    // Encode the types of all unboxed closures in this crate.
    i = rbml_w.writer.tell().unwrap();
    encode_unboxed_closures(&ecx, &mut rbml_w);
    stats.unboxed_closure_bytes = rbml_w.writer.tell().unwrap() - i;

    // Encode the def IDs of impls, for coherence checking.
    i = rbml_w.writer.tell().unwrap();
    encode_impls(&ecx, krate, &mut rbml_w);
    stats.impl_bytes = rbml_w.writer.tell().unwrap() - i;

    // Encode miscellaneous info.
    i = rbml_w.writer.tell().unwrap();
    encode_misc_info(&ecx, krate, &mut rbml_w);
    encode_reachable_extern_fns(&ecx, &mut rbml_w);
    stats.misc_bytes = rbml_w.writer.tell().unwrap() - i;

    // Encode and index the items.
    rbml_w.start_tag(tag_items);
    i = rbml_w.writer.tell().unwrap();
    let items_index = encode_info_for_items(&ecx, &mut rbml_w, krate);
    stats.item_bytes = rbml_w.writer.tell().unwrap() - i;

    i = rbml_w.writer.tell().unwrap();
    encode_index(&mut rbml_w, items_index, write_i64);
    stats.index_bytes = rbml_w.writer.tell().unwrap() - i;
    rbml_w.end_tag();

    encode_struct_field_attrs(&mut rbml_w, krate);

    stats.total_bytes = rbml_w.writer.tell().unwrap();

    if tcx.sess.meta_stats() {
        for e in rbml_w.writer.get_ref().iter() {
            if *e == 0 {
                stats.zero_bytes += 1;
            }
        }

        println!("metadata stats:");
        println!("       attribute bytes: {}", stats.attr_bytes);
        println!("             dep bytes: {}", stats.dep_bytes);
        println!("       lang item bytes: {}", stats.lang_item_bytes);
        println!("          native bytes: {}", stats.native_lib_bytes);
        println!("plugin registrar bytes: {}", stats.plugin_registrar_fn_bytes);
        println!("       macro def bytes: {}", stats.macro_defs_bytes);
        println!(" unboxed closure bytes: {}", stats.unboxed_closure_bytes);
        println!("            impl bytes: {}", stats.impl_bytes);
        println!("            misc bytes: {}", stats.misc_bytes);
        println!("            item bytes: {}", stats.item_bytes);
        println!("           index bytes: {}", stats.index_bytes);
        println!("            zero bytes: {}", stats.zero_bytes);
        println!("           total bytes: {}", stats.total_bytes);
    }
}

// Get the encoded string for a type
pub fn encoded_ty(tcx: &ty::ctxt, t: ty::t) -> String {
    let mut wr = SeekableMemWriter::new();
    tyencode::enc_ty(&mut wr, &tyencode::ctxt {
        diag: tcx.sess.diagnostic(),
        ds: def_to_string,
        tcx: tcx,
        abbrevs: &RefCell::new(HashMap::new())
    }, t);
    String::from_utf8(wr.unwrap()).unwrap()
}
