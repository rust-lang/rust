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

pub use self::InlinedItemRef::*;

use back::svh::Svh;
use session::config;
use metadata::common::*;
use metadata::cstore;
use metadata::decoder;
use metadata::tyencode;
use middle::def;
use middle::ty::{lookup_item_type};
use middle::ty::{self, Ty};
use middle::stability;
use util::nodemap::{FnvHashMap, NodeMap, NodeSet};

use serialize::Encodable;
use std::cell::RefCell;
use std::hash::{Hash, Hasher, SipHasher};
use syntax::abi;
use syntax::ast::{self, DefId, NodeId};
use syntax::ast_map::{PathElem, PathElems};
use syntax::ast_map;
use syntax::ast_util::*;
use syntax::ast_util;
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::diagnostic::SpanHandler;
use syntax::parse::token::special_idents;
use syntax::parse::token;
use syntax::print::pprust;
use syntax::ptr::P;
use syntax::visit::Visitor;
use syntax::visit;
use syntax;
use rbml::writer;
use rbml::io::SeekableMemWriter;

/// A borrowed version of `ast::InlinedItem`.
pub enum InlinedItemRef<'a> {
    IIItemRef(&'a ast::Item),
    IITraitItemRef(DefId, &'a ast::TraitItem),
    IIImplItemRef(DefId, &'a ast::ImplItem),
    IIForeignRef(&'a ast::ForeignItem)
}

pub type Encoder<'a> = writer::Encoder<'a, SeekableMemWriter>;

pub type EncodeInlinedItem<'a> =
    Box<FnMut(&EncodeContext, &mut Encoder, InlinedItemRef) + 'a>;

pub struct EncodeParams<'a, 'tcx: 'a> {
    pub diag: &'a SpanHandler,
    pub tcx: &'a ty::ctxt<'tcx>,
    pub reexports: &'a def::ExportMap,
    pub item_symbols: &'a RefCell<NodeMap<String>>,
    pub link_meta: &'a LinkMeta,
    pub cstore: &'a cstore::CStore,
    pub encode_inlined_item: EncodeInlinedItem<'a>,
    pub reachable: &'a NodeSet,
}

pub struct EncodeContext<'a, 'tcx: 'a> {
    pub diag: &'a SpanHandler,
    pub tcx: &'a ty::ctxt<'tcx>,
    pub reexports: &'a def::ExportMap,
    pub item_symbols: &'a RefCell<NodeMap<String>>,
    pub link_meta: &'a LinkMeta,
    pub cstore: &'a cstore::CStore,
    pub encode_inlined_item: RefCell<EncodeInlinedItem<'a>>,
    pub type_abbrevs: tyencode::abbrev_map<'tcx>,
    pub reachable: &'a NodeSet,
}

fn encode_name(rbml_w: &mut Encoder, name: ast::Name) {
    rbml_w.wr_tagged_str(tag_paths_data_name, &token::get_name(name));
}

fn encode_impl_type_basename(rbml_w: &mut Encoder, name: ast::Ident) {
    rbml_w.wr_tagged_str(tag_item_impl_type_basename, &token::get_ident(name));
}

pub fn encode_def_id(rbml_w: &mut Encoder, id: DefId) {
    rbml_w.wr_tagged_str(tag_def_id, &def_to_string(id)[]);
}

#[derive(Clone)]
struct entry<T> {
    val: T,
    pos: u64
}

fn encode_trait_ref<'a, 'tcx>(rbml_w: &mut Encoder,
                              ecx: &EncodeContext<'a, 'tcx>,
                              trait_ref: &ty::TraitRef<'tcx>,
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

// Item info table encoding
fn encode_family(rbml_w: &mut Encoder, c: char) {
    rbml_w.start_tag(tag_items_data_item_family);
    rbml_w.writer.write_all(&[c as u8]);
    rbml_w.end_tag();
}

pub fn def_to_string(did: DefId) -> String {
    format!("{}:{}", did.krate, did.node)
}

fn encode_item_variances(rbml_w: &mut Encoder,
                         ecx: &EncodeContext,
                         id: NodeId) {
    let v = ty::item_variances(ecx.tcx, ast_util::local_def(id));
    rbml_w.start_tag(tag_item_variances);
    v.encode(rbml_w);
    rbml_w.end_tag();
}

fn encode_bounds_and_type_for_item<'a, 'tcx>(rbml_w: &mut Encoder,
                                             ecx: &EncodeContext<'a, 'tcx>,
                                             id: ast::NodeId) {
    encode_bounds_and_type(rbml_w,
                           ecx,
                           &ty::lookup_item_type(ecx.tcx, local_def(id)),
                           &ty::lookup_predicates(ecx.tcx, local_def(id)));
}

fn encode_bounds_and_type<'a, 'tcx>(rbml_w: &mut Encoder,
                                    ecx: &EncodeContext<'a, 'tcx>,
                                    scheme: &ty::TypeScheme<'tcx>,
                                    predicates: &ty::GenericPredicates<'tcx>) {
    encode_generics(rbml_w, ecx, &scheme.generics, &predicates, tag_item_generics);
    encode_type(ecx, rbml_w, scheme.ty);
}

fn encode_variant_id(rbml_w: &mut Encoder, vid: DefId) {
    rbml_w.start_tag(tag_items_data_item_variant);
    let s = def_to_string(vid);
    rbml_w.writer.write_all(s.as_bytes());
    rbml_w.end_tag();

    rbml_w.start_tag(tag_mod_child);
    rbml_w.wr_str(&s[..]);
    rbml_w.end_tag();
}

pub fn write_closure_type<'a, 'tcx>(ecx: &EncodeContext<'a, 'tcx>,
                                    rbml_w: &mut Encoder,
                                    closure_type: &ty::ClosureTy<'tcx>) {
    let ty_str_ctxt = &tyencode::ctxt {
        diag: ecx.diag,
        ds: def_to_string,
        tcx: ecx.tcx,
        abbrevs: &ecx.type_abbrevs
    };
    tyencode::enc_closure_ty(rbml_w.writer, ty_str_ctxt, closure_type);
}

pub fn write_type<'a, 'tcx>(ecx: &EncodeContext<'a, 'tcx>,
                            rbml_w: &mut Encoder,
                            typ: Ty<'tcx>) {
    let ty_str_ctxt = &tyencode::ctxt {
        diag: ecx.diag,
        ds: def_to_string,
        tcx: ecx.tcx,
        abbrevs: &ecx.type_abbrevs
    };
    tyencode::enc_ty(rbml_w.writer, ty_str_ctxt, typ);
}

pub fn write_trait_ref<'a, 'tcx>(ecx: &EncodeContext<'a, 'tcx>,
                                 rbml_w: &mut Encoder,
                                trait_ref: &ty::TraitRef<'tcx>) {
    let ty_str_ctxt = &tyencode::ctxt {
        diag: ecx.diag,
        ds: def_to_string,
        tcx: ecx.tcx,
        abbrevs: &ecx.type_abbrevs
    };
    tyencode::enc_trait_ref(rbml_w.writer, ty_str_ctxt, trait_ref);
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

fn encode_bounds<'a, 'tcx>(rbml_w: &mut Encoder,
                           ecx: &EncodeContext<'a, 'tcx>,
                           bounds: &ty::ParamBounds<'tcx>,
                           tag: uint) {
    rbml_w.start_tag(tag);

    let ty_str_ctxt = &tyencode::ctxt { diag: ecx.diag,
                                        ds: def_to_string,
                                        tcx: ecx.tcx,
                                        abbrevs: &ecx.type_abbrevs };
    tyencode::enc_bounds(rbml_w.writer, ty_str_ctxt, bounds);

    rbml_w.end_tag();
}

fn encode_type<'a, 'tcx>(ecx: &EncodeContext<'a, 'tcx>,
                         rbml_w: &mut Encoder,
                         typ: Ty<'tcx>) {
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

fn encode_method_fty<'a, 'tcx>(ecx: &EncodeContext<'a, 'tcx>,
                               rbml_w: &mut Encoder,
                               typ: &ty::BareFnTy<'tcx>) {
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
    match ecx.item_symbols.borrow().get(&id) {
        Some(x) => {
            debug!("encode_symbol(id={}, str={})", id, *x);
            rbml_w.writer.write_all(x.as_bytes());
        }
        None => {
            ecx.diag.handler().bug(
                &format!("encode_symbol: id not found {}", id)[]);
        }
    }
    rbml_w.end_tag();
}

fn encode_disr_val(_: &EncodeContext,
                   rbml_w: &mut Encoder,
                   disr_val: ty::Disr) {
    rbml_w.start_tag(tag_disr_val);
    let s = disr_val.to_string();
    rbml_w.writer.write_all(s.as_bytes());
    rbml_w.end_tag();
}

fn encode_parent_item(rbml_w: &mut Encoder, id: DefId) {
    rbml_w.start_tag(tag_items_data_parent_item);
    let s = def_to_string(id);
    rbml_w.writer.write_all(s.as_bytes());
    rbml_w.end_tag();
}

fn encode_struct_fields(rbml_w: &mut Encoder,
                        fields: &[ty::field_ty],
                        origin: DefId) {
    for f in fields {
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
        rbml_w.writer.write_all(s.as_bytes());
        rbml_w.end_tag();
        rbml_w.end_tag();
    }
}

fn encode_enum_variant_info(ecx: &EncodeContext,
                            rbml_w: &mut Encoder,
                            id: NodeId,
                            variants: &[P<ast::Variant>],
                            index: &mut Vec<entry<i64>>) {
    debug!("encode_enum_variant_info(id={})", id);

    let mut disr_val = 0;
    let mut i = 0;
    let vi = ty::enum_variants(ecx.tcx,
                               DefId { krate: ast::LOCAL_CRATE, node: id });
    for variant in variants {
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
        encode_attributes(rbml_w, &variant.node.attrs[]);
        encode_repr_attrs(rbml_w, ecx, &variant.node.attrs[]);

        let stab = stability::lookup(ecx.tcx, ast_util::local_def(variant.node.id));
        encode_stability(rbml_w, stab);

        match variant.node.kind {
            ast::TupleVariantKind(_) => {},
            ast::StructVariantKind(_) => {
                let fields = ty::lookup_struct_fields(ecx.tcx, def_id);
                let idx = encode_info_for_struct(ecx,
                                                 rbml_w,
                                                 &fields[..],
                                                 index);
                encode_struct_fields(rbml_w, &fields[..], def_id);
                encode_index(rbml_w, idx, write_i64);
            }
        }
        if (*vi)[i].disr_val != disr_val {
            encode_disr_val(ecx, rbml_w, (*vi)[i].disr_val);
            disr_val = (*vi)[i].disr_val;
        }
        encode_bounds_and_type_for_item(rbml_w, ecx, def_id.local_id());

        ecx.tcx.map.with_path(variant.node.id, |path| encode_path(rbml_w, path));
        rbml_w.end_tag();
        disr_val += 1;
        i += 1;
    }
}

fn encode_path<PI: Iterator<Item=PathElem>>(rbml_w: &mut Encoder, path: PI) {
    let path = path.collect::<Vec<_>>();
    rbml_w.start_tag(tag_path);
    rbml_w.wr_tagged_u32(tag_path_len, path.len() as u32);
    for pe in &path {
        let tag = match *pe {
            ast_map::PathMod(_) => tag_path_elem_mod,
            ast_map::PathName(_) => tag_path_elem_name
        };
        rbml_w.wr_tagged_str(tag, &token::get_name(pe.name()));
    }
    rbml_w.end_tag();
}

fn encode_reexported_static_method(rbml_w: &mut Encoder,
                                   exp: &def::Export,
                                   method_def_id: DefId,
                                   method_name: ast::Name) {
    debug!("(encode reexported static method) {}::{}",
            exp.name, token::get_name(method_name));
    rbml_w.start_tag(tag_items_data_item_reexport);
    rbml_w.start_tag(tag_items_data_item_reexport_def_id);
    rbml_w.wr_str(&def_to_string(method_def_id)[]);
    rbml_w.end_tag();
    rbml_w.start_tag(tag_items_data_item_reexport_name);
    rbml_w.wr_str(&format!("{}::{}",
                          exp.name,
                          token::get_name(method_name))[]);
    rbml_w.end_tag();
    rbml_w.end_tag();
}

fn encode_reexported_static_base_methods(ecx: &EncodeContext,
                                         rbml_w: &mut Encoder,
                                         exp: &def::Export)
                                         -> bool {
    let impl_items = ecx.tcx.impl_items.borrow();
    match ecx.tcx.inherent_impls.borrow().get(&exp.def_id) {
        Some(implementations) => {
            for base_impl_did in &**implementations {
                for &method_did in &*(*impl_items)[*base_impl_did] {
                    let impl_item = ty::impl_or_trait_item(
                        ecx.tcx,
                        method_did.def_id());
                    match impl_item {
                        ty::MethodTraitItem(ref m) => {
                            encode_reexported_static_method(rbml_w,
                                                            exp,
                                                            m.def_id,
                                                            m.name);
                        }
                        ty::TypeTraitItem(_) => {}
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
                                          exp: &def::Export)
                                          -> bool {
    match ecx.tcx.trait_items_cache.borrow().get(&exp.def_id) {
        Some(trait_items) => {
            for trait_item in &**trait_items {
                if let ty::MethodTraitItem(ref m) = *trait_item {
                    encode_reexported_static_method(rbml_w,
                                                    exp,
                                                    m.def_id,
                                                    m.name);
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
                                    exp: &def::Export) {
    if let Some(ast_map::NodeItem(item)) = ecx.tcx.map.find(exp.def_id.node) {
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
        if path_differs || item.ident.name != exp.name {
            if !encode_reexported_static_base_methods(ecx, rbml_w, exp) {
                if encode_reexported_static_trait_methods(ecx, rbml_w, exp) {
                    debug!("(encode reexported static methods) {} [trait]",
                           item.ident.name);
                }
            }
            else {
                debug!("(encode reexported static methods) {} [base]",
                       item.ident.name);
            }
        }
    }
}

/// Iterates through "auxiliary node IDs", which are node IDs that describe
/// top-level items that are sub-items of the given item. Specifically:
///
/// * For newtype structs, iterates through the node ID of the constructor.
fn each_auxiliary_node_id<F>(item: &ast::Item, callback: F) -> bool where
    F: FnOnce(NodeId) -> bool,
{
    let mut continue_ = true;
    match item.node {
        ast::ItemStruct(ref struct_def, _) => {
            // If this is a newtype struct, return the constructor.
            match struct_def.ctor_id {
                Some(ctor_id) if struct_def.fields.len() > 0 &&
                        struct_def.fields[0].node.kind.is_unnamed() => {
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
    match ecx.reexports.get(&id) {
        Some(exports) => {
            debug!("(encoding info for module) found reexports for {}", id);
            for exp in exports {
                debug!("(encoding info for module) reexport '{}' ({}/{}) for \
                        {}",
                       exp.name,
                       exp.def_id.krate,
                       exp.def_id.node,
                       id);
                rbml_w.start_tag(tag_items_data_item_reexport);
                rbml_w.start_tag(tag_items_data_item_reexport_def_id);
                rbml_w.wr_str(&def_to_string(exp.def_id)[]);
                rbml_w.end_tag();
                rbml_w.start_tag(tag_items_data_item_reexport_name);
                rbml_w.wr_str(exp.name.as_str());
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
                       md: &ast::Mod,
                       attrs: &[ast::Attribute],
                       id: NodeId,
                       path: PathElems,
                       name: ast::Ident,
                       vis: ast::Visibility) {
    rbml_w.start_tag(tag_items_data_item);
    encode_def_id(rbml_w, local_def(id));
    encode_family(rbml_w, 'm');
    encode_name(rbml_w, name.name);
    debug!("(encoding info for module) encoding info for module ID {}", id);

    // Encode info about all the module children.
    for item in &md.items {
        rbml_w.start_tag(tag_mod_child);
        rbml_w.wr_str(&def_to_string(local_def(item.id))[]);
        rbml_w.end_tag();

        each_auxiliary_node_id(&**item, |auxiliary_node_id| {
            rbml_w.start_tag(tag_mod_child);
            rbml_w.wr_str(&def_to_string(local_def(
                        auxiliary_node_id))[]);
            rbml_w.end_tag();
            true
        });

        if let ast::ItemImpl(..) = item.node {
            let (ident, did) = (item.ident, item.id);
            debug!("(encoding info for module) ... encoding impl {} ({}/{})",
                   token::get_ident(ident),
                   did, ecx.tcx.map.node_to_string(did));

            rbml_w.start_tag(tag_mod_impl);
            rbml_w.wr_str(&def_to_string(local_def(did))[]);
            rbml_w.end_tag();
        }
    }

    encode_path(rbml_w, path.clone());
    encode_visibility(rbml_w, vis);

    let stab = stability::lookup(ecx.tcx, ast_util::local_def(id));
    encode_stability(rbml_w, stab);

    // Encode the reexports of this module, if this module is public.
    if vis == ast::Public {
        debug!("(encoding info for module) encoding reexports for {}", id);
        encode_reexports(ecx, rbml_w, id, path);
    }
    encode_attributes(rbml_w, attrs);

    rbml_w.end_tag();
}

fn encode_struct_field_family(rbml_w: &mut Encoder,
                              visibility: ast::Visibility) {
    encode_family(rbml_w, match visibility {
        ast::Public => 'g',
        ast::Inherited => 'N'
    });
}

fn encode_visibility(rbml_w: &mut Encoder, visibility: ast::Visibility) {
    rbml_w.start_tag(tag_items_data_item_visibility);
    let ch = match visibility {
        ast::Public => 'y',
        ast::Inherited => 'i',
    };
    rbml_w.wr_str(&ch.to_string()[]);
    rbml_w.end_tag();
}

fn encode_explicit_self(rbml_w: &mut Encoder,
                        explicit_self: &ty::ExplicitSelfCategory) {
    rbml_w.start_tag(tag_item_trait_method_explicit_self);

    // Encode the base self type.
    match *explicit_self {
        ty::StaticExplicitSelfCategory => {
            rbml_w.writer.write_all(&[ 's' as u8 ]);
        }
        ty::ByValueExplicitSelfCategory => {
            rbml_w.writer.write_all(&[ 'v' as u8 ]);
        }
        ty::ByBoxExplicitSelfCategory => {
            rbml_w.writer.write_all(&[ '~' as u8 ]);
        }
        ty::ByReferenceExplicitSelfCategory(_, m) => {
            // FIXME(#4846) encode custom lifetime
            rbml_w.writer.write_all(&['&' as u8]);
            encode_mutability(rbml_w, m);
        }
    }

    rbml_w.end_tag();

    fn encode_mutability(rbml_w: &mut Encoder,
                         m: ast::Mutability) {
        match m {
            ast::MutImmutable => { rbml_w.writer.write_all(&[ 'i' as u8 ]); }
            ast::MutMutable => { rbml_w.writer.write_all(&[ 'm' as u8 ]); }
        }
    }
}

fn encode_item_sort(rbml_w: &mut Encoder, sort: char) {
    rbml_w.start_tag(tag_item_trait_item_sort);
    rbml_w.writer.write_all(&[ sort as u8 ]);
    rbml_w.end_tag();
}

fn encode_parent_sort(rbml_w: &mut Encoder, sort: char) {
    rbml_w.start_tag(tag_item_trait_parent_sort);
    rbml_w.writer.write_all(&[ sort as u8 ]);
    rbml_w.end_tag();
}

fn encode_provided_source(rbml_w: &mut Encoder,
                          source_opt: Option<DefId>) {
    if let Some(source) = source_opt {
        rbml_w.start_tag(tag_item_method_provided_source);
        let s = def_to_string(source);
        rbml_w.writer.write_all(s.as_bytes());
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
    for field in fields {
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
        encode_bounds_and_type_for_item(rbml_w, ecx, id);
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
    encode_family(rbml_w, 'o');
    encode_bounds_and_type_for_item(rbml_w, ecx, ctor_id);
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

fn encode_generics<'a, 'tcx>(rbml_w: &mut Encoder,
                             ecx: &EncodeContext<'a, 'tcx>,
                             generics: &ty::Generics<'tcx>,
                             predicates: &ty::GenericPredicates<'tcx>,
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
                             &def_to_string(param.def_id)[]);

        rbml_w.wr_tagged_u64(tag_region_param_def_space,
                             param.space.to_uint() as u64);

        rbml_w.wr_tagged_u64(tag_region_param_def_index,
                             param.index as u64);

        for &bound_region in &param.bounds {
            encode_region(ecx, rbml_w, bound_region);
        }

        rbml_w.end_tag();
    }

    for (space, _, predicate) in predicates.predicates.iter_enumerated() {
        rbml_w.start_tag(tag_predicate);

        rbml_w.wr_tagged_u8(tag_predicate_space, space as u8);

        rbml_w.start_tag(tag_predicate_data);
        tyencode::enc_predicate(rbml_w.writer, ty_str_ctxt, predicate);
        rbml_w.end_tag();

        rbml_w.end_tag();
    }

    rbml_w.end_tag();
}

fn encode_method_ty_fields<'a, 'tcx>(ecx: &EncodeContext<'a, 'tcx>,
                                     rbml_w: &mut Encoder,
                                     method_ty: &ty::Method<'tcx>) {
    encode_def_id(rbml_w, method_ty.def_id);
    encode_name(rbml_w, method_ty.name);
    encode_generics(rbml_w, ecx, &method_ty.generics, &method_ty.predicates,
                    tag_method_ty_generics);
    encode_method_fty(ecx, rbml_w, &method_ty.fty);
    encode_visibility(rbml_w, method_ty.vis);
    encode_explicit_self(rbml_w, &method_ty.explicit_self);
    match method_ty.explicit_self {
        ty::StaticExplicitSelfCategory => {
            encode_family(rbml_w, STATIC_METHOD_FAMILY);
        }
        _ => encode_family(rbml_w, METHOD_FAMILY)
    }
    encode_provided_source(rbml_w, method_ty.provided_source);
}

fn encode_info_for_method<'a, 'tcx>(ecx: &EncodeContext<'a, 'tcx>,
                                    rbml_w: &mut Encoder,
                                    m: &ty::Method<'tcx>,
                                    impl_path: PathElems,
                                    is_default_impl: bool,
                                    parent_id: NodeId,
                                    ast_item_opt: Option<&ast::ImplItem>) {

    debug!("encode_info_for_method: {:?} {:?}", m.def_id,
           token::get_name(m.name));
    rbml_w.start_tag(tag_items_data_item);

    encode_method_ty_fields(ecx, rbml_w, m);
    encode_parent_item(rbml_w, local_def(parent_id));
    encode_item_sort(rbml_w, 'r');

    let stab = stability::lookup(ecx.tcx, m.def_id);
    encode_stability(rbml_w, stab);

    // The type for methods gets encoded twice, which is unfortunate.
    encode_bounds_and_type_for_item(rbml_w, ecx, m.def_id.local_id());

    let elem = ast_map::PathName(m.name);
    encode_path(rbml_w, impl_path.chain(Some(elem).into_iter()));
    match ast_item_opt {
        Some(&ast::MethodImplItem(ref ast_method)) => {
            encode_attributes(rbml_w, &ast_method.attrs[]);
            let scheme = ty::lookup_item_type(ecx.tcx, m.def_id);
            let any_types = !scheme.generics.types.is_empty();
            if any_types || is_default_impl || should_inline(&ast_method.attrs[]) {
                encode_inlined_item(ecx, rbml_w, IIImplItemRef(local_def(parent_id),
                                                               ast_item_opt.unwrap()));
            }
            if !any_types {
                encode_symbol(ecx, rbml_w, m.def_id.node);
            }
            encode_method_argument_names(rbml_w, ast_method.pe_fn_decl());
        }
        Some(_) | None => {}
    }

    rbml_w.end_tag();
}

fn encode_info_for_associated_type(ecx: &EncodeContext,
                                   rbml_w: &mut Encoder,
                                   associated_type: &ty::AssociatedType,
                                   impl_path: PathElems,
                                   parent_id: NodeId,
                                   typedef_opt: Option<P<ast::Typedef>>) {
    debug!("encode_info_for_associated_type({:?},{:?})",
           associated_type.def_id,
           token::get_name(associated_type.name));

    rbml_w.start_tag(tag_items_data_item);

    encode_def_id(rbml_w, associated_type.def_id);
    encode_name(rbml_w, associated_type.name);
    encode_visibility(rbml_w, associated_type.vis);
    encode_family(rbml_w, 'y');
    encode_parent_item(rbml_w, local_def(parent_id));
    encode_item_sort(rbml_w, 't');

    encode_bounds_and_type_for_item(rbml_w, ecx, associated_type.def_id.local_id());

    let stab = stability::lookup(ecx.tcx, associated_type.def_id);
    encode_stability(rbml_w, stab);

    let elem = ast_map::PathName(associated_type.name);
    encode_path(rbml_w, impl_path.chain(Some(elem).into_iter()));

    match typedef_opt {
        None => {}
        Some(typedef) => {
            encode_attributes(rbml_w, &typedef.attrs[]);
            encode_type(ecx, rbml_w, ty::node_id_to_type(ecx.tcx,
                                                         typedef.id));
        }
    }

    rbml_w.end_tag();
}

fn encode_method_argument_names(rbml_w: &mut Encoder,
                                decl: &ast::FnDecl) {
    rbml_w.start_tag(tag_method_argument_names);
    for arg in &decl.inputs {
        rbml_w.start_tag(tag_method_argument_name);
        if let ast::PatIdent(_, ref path1, _) = arg.pat.node {
            let name = token::get_ident(path1.node);
            rbml_w.writer.write_all(name.as_bytes());
        }
        rbml_w.end_tag();
    }
    rbml_w.end_tag();
}

fn encode_repr_attrs(rbml_w: &mut Encoder,
                     ecx: &EncodeContext,
                     attrs: &[ast::Attribute]) {
    let mut repr_attrs = Vec::new();
    for attr in attrs {
        repr_attrs.extend(attr::find_repr_attrs(ecx.tcx.sess.diagnostic(),
                                                attr).into_iter());
    }
    rbml_w.start_tag(tag_items_data_item_repr);
    repr_attrs.encode(rbml_w);
    rbml_w.end_tag();
}

fn encode_inlined_item(ecx: &EncodeContext,
                       rbml_w: &mut Encoder,
                       ii: InlinedItemRef) {
    let mut eii = ecx.encode_inlined_item.borrow_mut();
    let eii: &mut EncodeInlinedItem = &mut *eii;
    eii(ecx, rbml_w, ii)
}

const FN_FAMILY: char = 'f';
const STATIC_METHOD_FAMILY: char = 'F';
const METHOD_FAMILY: char = 'h';

fn should_inline(attrs: &[ast::Attribute]) -> bool {
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
    match ecx.tcx.inherent_impls.borrow().get(&def_id) {
        None => {}
        Some(implementations) => {
            for &impl_def_id in &**implementations {
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
    match ecx.tcx.trait_impls.borrow().get(&trait_def_id) {
        None => {}
        Some(implementations) => {
            for &impl_def_id in &*implementations.borrow() {
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
                        item: &ast::Item,
                        index: &mut Vec<entry<i64>>,
                        path: PathElems,
                        vis: ast::Visibility) {
    let tcx = ecx.tcx;

    fn add_to_index(item: &ast::Item, rbml_w: &Encoder,
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
      ast::ItemStatic(_, m, _) => {
        add_to_index(item, rbml_w, index);
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id(rbml_w, def_id);
        if m == ast::MutMutable {
            encode_family(rbml_w, 'b');
        } else {
            encode_family(rbml_w, 'c');
        }
        encode_bounds_and_type_for_item(rbml_w, ecx, item.id);
        encode_symbol(ecx, rbml_w, item.id);
        encode_name(rbml_w, item.ident.name);
        encode_path(rbml_w, path);
        encode_visibility(rbml_w, vis);
        encode_stability(rbml_w, stab);
        encode_attributes(rbml_w, &item.attrs[]);
        rbml_w.end_tag();
      }
      ast::ItemConst(_, _) => {
        add_to_index(item, rbml_w, index);
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id(rbml_w, def_id);
        encode_family(rbml_w, 'C');
        encode_bounds_and_type_for_item(rbml_w, ecx, item.id);
        encode_name(rbml_w, item.ident.name);
        encode_path(rbml_w, path);
        encode_attributes(rbml_w, &item.attrs);
        encode_inlined_item(ecx, rbml_w, IIItemRef(item));
        encode_visibility(rbml_w, vis);
        encode_stability(rbml_w, stab);
        rbml_w.end_tag();
      }
      ast::ItemFn(ref decl, _, _, ref generics, _) => {
        add_to_index(item, rbml_w, index);
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id(rbml_w, def_id);
        encode_family(rbml_w, FN_FAMILY);
        let tps_len = generics.ty_params.len();
        encode_bounds_and_type_for_item(rbml_w, ecx, item.id);
        encode_name(rbml_w, item.ident.name);
        encode_path(rbml_w, path);
        encode_attributes(rbml_w, &item.attrs[]);
        if tps_len > 0 || should_inline(&item.attrs[]) {
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
      ast::ItemMod(ref m) => {
        add_to_index(item, rbml_w, index);
        encode_info_for_mod(ecx,
                            rbml_w,
                            m,
                            &item.attrs[],
                            item.id,
                            path,
                            item.ident,
                            item.vis);
      }
      ast::ItemForeignMod(ref fm) => {
        add_to_index(item, rbml_w, index);
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id(rbml_w, def_id);
        encode_family(rbml_w, 'n');
        encode_name(rbml_w, item.ident.name);
        encode_path(rbml_w, path);

        // Encode all the items in this module.
        for foreign_item in &fm.items {
            rbml_w.start_tag(tag_mod_child);
            rbml_w.wr_str(&def_to_string(local_def(foreign_item.id))[]);
            rbml_w.end_tag();
        }
        encode_visibility(rbml_w, vis);
        encode_stability(rbml_w, stab);
        rbml_w.end_tag();
      }
      ast::ItemTy(..) => {
        add_to_index(item, rbml_w, index);
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id(rbml_w, def_id);
        encode_family(rbml_w, 'y');
        encode_bounds_and_type_for_item(rbml_w, ecx, item.id);
        encode_name(rbml_w, item.ident.name);
        encode_path(rbml_w, path);
        encode_visibility(rbml_w, vis);
        encode_stability(rbml_w, stab);
        rbml_w.end_tag();
      }
      ast::ItemEnum(ref enum_definition, _) => {
        add_to_index(item, rbml_w, index);

        rbml_w.start_tag(tag_items_data_item);
        encode_def_id(rbml_w, def_id);
        encode_family(rbml_w, 't');
        encode_item_variances(rbml_w, ecx, item.id);
        encode_bounds_and_type_for_item(rbml_w, ecx, item.id);
        encode_name(rbml_w, item.ident.name);
        encode_attributes(rbml_w, &item.attrs[]);
        encode_repr_attrs(rbml_w, ecx, &item.attrs[]);
        for v in &enum_definition.variants {
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
                                 &(*enum_definition).variants[],
                                 index);
      }
      ast::ItemStruct(ref struct_def, _) => {
        let fields = ty::lookup_struct_fields(tcx, def_id);

        /* First, encode the fields
           These come first because we need to write them to make
           the index, and the index needs to be in the item for the
           class itself */
        let idx = encode_info_for_struct(ecx,
                                         rbml_w,
                                         &fields[..],
                                         index);

        /* Index the class*/
        add_to_index(item, rbml_w, index);

        /* Now, make an item for the class itself */
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id(rbml_w, def_id);
        encode_family(rbml_w, 'S');
        encode_bounds_and_type_for_item(rbml_w, ecx, item.id);

        encode_item_variances(rbml_w, ecx, item.id);
        encode_name(rbml_w, item.ident.name);
        encode_attributes(rbml_w, &item.attrs[]);
        encode_path(rbml_w, path.clone());
        encode_stability(rbml_w, stab);
        encode_visibility(rbml_w, vis);
        encode_repr_attrs(rbml_w, ecx, &item.attrs[]);

        /* Encode def_ids for each field and method
         for methods, write all the stuff get_trait_method
        needs to know*/
        encode_struct_fields(rbml_w, &fields[..], def_id);

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
      ast::ItemImpl(unsafety, polarity, _, ref opt_trait, ref ty, ref ast_items) => {
        // We need to encode information about the default methods we
        // have inherited, so we drive this based on the impl structure.
        let impl_items = tcx.impl_items.borrow();
        let items = &(*impl_items)[def_id];

        add_to_index(item, rbml_w, index);
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id(rbml_w, def_id);
        encode_family(rbml_w, 'i');
        encode_bounds_and_type_for_item(rbml_w, ecx, item.id);
        encode_name(rbml_w, item.ident.name);
        encode_attributes(rbml_w, &item.attrs[]);
        encode_unsafety(rbml_w, unsafety);
        encode_polarity(rbml_w, polarity);
        match ty.node {
            ast::TyPath(ref path, _) if path.segments.len() == 1 => {
                let ident = path.segments.last().unwrap().identifier;
                encode_impl_type_basename(rbml_w, ident);
            }
            _ => {}
        }
        for &item_def_id in items {
            rbml_w.start_tag(tag_item_impl_item);
            match item_def_id {
                ty::MethodTraitItemId(item_def_id) => {
                    encode_def_id(rbml_w, item_def_id);
                    encode_item_sort(rbml_w, 'r');
                }
                ty::TypeTraitItemId(item_def_id) => {
                    encode_def_id(rbml_w, item_def_id);
                    encode_item_sort(rbml_w, 't');
                }
            }
            rbml_w.end_tag();
        }
        if let Some(ref ast_trait_ref) = *opt_trait {
            let trait_ref = ty::node_id_to_trait_ref(
                tcx, ast_trait_ref.ref_id);
            encode_trait_ref(rbml_w, ecx, &*trait_ref, tag_item_trait_ref);
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
                Some(&ast_items[i])
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
                (ty::MethodTraitItem(ref method_type),
                 Some(&ast::MethodImplItem(_))) => {
                    encode_info_for_method(ecx,
                                           rbml_w,
                                           &**method_type,
                                           path.clone(),
                                           false,
                                           item.id,
                                           ast_item)
                }
                (ty::MethodTraitItem(ref method_type), _) => {
                    encode_info_for_method(ecx,
                                           rbml_w,
                                           &**method_type,
                                           path.clone(),
                                           false,
                                           item.id,
                                           None)
                }
                (ty::TypeTraitItem(ref associated_type),
                 Some(&ast::TypeImplItem(ref typedef))) => {
                    encode_info_for_associated_type(ecx,
                                                    rbml_w,
                                                    &**associated_type,
                                                    path.clone(),
                                                    item.id,
                                                    Some((*typedef).clone()))
                }
                (ty::TypeTraitItem(ref associated_type), _) => {
                    encode_info_for_associated_type(ecx,
                                                    rbml_w,
                                                    &**associated_type,
                                                    path.clone(),
                                                    item.id,
                                                    None)
                }
            }
        }
      }
      ast::ItemTrait(_, _, _, ref ms) => {
        add_to_index(item, rbml_w, index);
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id(rbml_w, def_id);
        encode_family(rbml_w, 'I');
        encode_item_variances(rbml_w, ecx, item.id);
        let trait_def = ty::lookup_trait_def(tcx, def_id);
        let trait_predicates = ty::lookup_predicates(tcx, def_id);
        encode_unsafety(rbml_w, trait_def.unsafety);
        encode_paren_sugar(rbml_w, trait_def.paren_sugar);
        encode_associated_type_names(rbml_w, &trait_def.associated_type_names);
        encode_generics(rbml_w, ecx, &trait_def.generics, &trait_predicates, tag_item_generics);
        encode_trait_ref(rbml_w, ecx, &*trait_def.trait_ref, tag_item_trait_ref);
        encode_name(rbml_w, item.ident.name);
        encode_attributes(rbml_w, &item.attrs[]);
        encode_visibility(rbml_w, vis);
        encode_stability(rbml_w, stab);
        for &method_def_id in &*ty::trait_item_def_ids(tcx, def_id) {
            rbml_w.start_tag(tag_item_trait_item);
            match method_def_id {
                ty::MethodTraitItemId(method_def_id) => {
                    encode_def_id(rbml_w, method_def_id);
                    encode_item_sort(rbml_w, 'r');
                }
                ty::TypeTraitItemId(type_def_id) => {
                    encode_def_id(rbml_w, type_def_id);
                    encode_item_sort(rbml_w, 't');
                }
            }
            rbml_w.end_tag();

            rbml_w.start_tag(tag_mod_child);
            rbml_w.wr_str(&def_to_string(method_def_id.def_id())[]);
            rbml_w.end_tag();
        }
        encode_path(rbml_w, path.clone());

        encode_bounds(rbml_w, ecx, &trait_def.bounds, tag_trait_def_bounds);

        // Encode the implementations of this trait.
        encode_extension_implementations(ecx, rbml_w, def_id);

        // Encode inherent implementations for this trait.
        encode_inherent_implementations(ecx, rbml_w, def_id);

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

            encode_parent_item(rbml_w, def_id);

            let stab = stability::lookup(tcx, item_def_id.def_id());
            encode_stability(rbml_w, stab);

            let trait_item_type =
                ty::impl_or_trait_item(tcx, item_def_id.def_id());
            let is_nonstatic_method;
            match trait_item_type {
                ty::MethodTraitItem(method_ty) => {
                    let method_def_id = item_def_id.def_id();

                    encode_method_ty_fields(ecx, rbml_w, &*method_ty);

                    let elem = ast_map::PathName(method_ty.name);
                    encode_path(rbml_w,
                                path.clone().chain(Some(elem).into_iter()));

                    match method_ty.explicit_self {
                        ty::StaticExplicitSelfCategory => {
                            encode_family(rbml_w,
                                          STATIC_METHOD_FAMILY);
                        }
                        _ => {
                            encode_family(rbml_w,
                                          METHOD_FAMILY);
                        }
                    }
                    encode_bounds_and_type_for_item(rbml_w, ecx, method_def_id.local_id());

                    is_nonstatic_method = method_ty.explicit_self !=
                        ty::StaticExplicitSelfCategory;
                }
                ty::TypeTraitItem(associated_type) => {
                    encode_name(rbml_w, associated_type.name);
                    encode_def_id(rbml_w, associated_type.def_id);

                    let elem = ast_map::PathName(associated_type.name);
                    encode_path(rbml_w,
                                path.clone().chain(Some(elem).into_iter()));

                    encode_item_sort(rbml_w, 't');
                    encode_family(rbml_w, 'y');

                    is_nonstatic_method = false;
                }
            }

            encode_parent_sort(rbml_w, 't');

            let trait_item = &ms[i];
            let encode_trait_item = |rbml_w: &mut Encoder| {
                // If this is a static method, we've already
                // encoded this.
                if is_nonstatic_method {
                    // FIXME: I feel like there is something funny
                    // going on.
                    encode_bounds_and_type_for_item(rbml_w, ecx, item_def_id.def_id().local_id());
                }
            };
            match trait_item {
                &ast::RequiredMethod(ref m) => {
                    encode_attributes(rbml_w, &m.attrs[]);
                    encode_trait_item(rbml_w);
                    encode_item_sort(rbml_w, 'r');
                    encode_method_argument_names(rbml_w, &*m.decl);
                }

                &ast::ProvidedMethod(ref m) => {
                    encode_attributes(rbml_w, &m.attrs[]);
                    encode_trait_item(rbml_w);
                    encode_item_sort(rbml_w, 'p');
                    encode_inlined_item(ecx, rbml_w, IITraitItemRef(def_id, trait_item));
                    encode_method_argument_names(rbml_w, &*m.pe_fn_decl());
                }

                &ast::TypeTraitItem(ref associated_type) => {
                    encode_attributes(rbml_w,
                                      &associated_type.attrs[]);
                    encode_item_sort(rbml_w, 't');
                }
            }

            rbml_w.end_tag();
        }
      }
      ast::ItemExternCrate(_) | ast::ItemUse(_) |ast::ItemMac(..) => {
        // these are encoded separately
      }
    }
}

fn encode_info_for_foreign_item(ecx: &EncodeContext,
                                rbml_w: &mut Encoder,
                                nitem: &ast::ForeignItem,
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
      ast::ForeignItemFn(ref fndecl, _) => {
        encode_family(rbml_w, FN_FAMILY);
        encode_bounds_and_type_for_item(rbml_w, ecx, nitem.id);
        encode_name(rbml_w, nitem.ident.name);
        if abi == abi::RustIntrinsic {
            encode_inlined_item(ecx, rbml_w, IIForeignRef(nitem));
        }
        encode_attributes(rbml_w, &*nitem.attrs);
        let stab = stability::lookup(ecx.tcx, ast_util::local_def(nitem.id));
        encode_stability(rbml_w, stab);
        encode_symbol(ecx, rbml_w, nitem.id);
        encode_method_argument_names(rbml_w, &*fndecl);
      }
      ast::ForeignItemStatic(_, mutbl) => {
        if mutbl {
            encode_family(rbml_w, 'b');
        } else {
            encode_family(rbml_w, 'c');
        }
        encode_bounds_and_type_for_item(rbml_w, ecx, nitem.id);
        encode_attributes(rbml_w, &*nitem.attrs);
        let stab = stability::lookup(ecx.tcx, ast_util::local_def(nitem.id));
        encode_stability(rbml_w, stab);
        encode_symbol(ecx, rbml_w, nitem.id);
        encode_name(rbml_w, nitem.ident.name);
      }
    }
    encode_path(rbml_w, path);
    rbml_w.end_tag();
}

fn my_visit_expr(_e: &ast::Expr) { }

fn my_visit_item(i: &ast::Item,
                 rbml_w: &mut Encoder,
                 ecx: &EncodeContext,
                 index: &mut Vec<entry<i64>>) {
    ecx.tcx.map.with_path(i.id, |path| {
        encode_info_for_item(ecx, rbml_w, i, index, path, i.vis);
    });
}

fn my_visit_foreign_item(ni: &ast::ForeignItem,
                         rbml_w: &mut Encoder,
                         ecx: &EncodeContext,
                         index: &mut Vec<entry<i64>>) {
    debug!("writing foreign item {}::{}",
            ecx.tcx.map.path_to_string(ni.id),
            token::get_ident(ni.ident));

    let abi = ecx.tcx.map.get_foreign_abi(ni.id);
    ecx.tcx.map.with_path(ni.id, |path| {
        encode_info_for_foreign_item(ecx, rbml_w,
                                     ni, index,
                                     path, abi);
    });
}

struct EncodeVisitor<'a, 'b:'a, 'c:'a, 'tcx:'c> {
    rbml_w_for_visit_item: &'a mut Encoder<'b>,
    ecx: &'a EncodeContext<'c,'tcx>,
    index: &'a mut Vec<entry<i64>>,
}

impl<'a, 'b, 'c, 'tcx, 'v> Visitor<'v> for EncodeVisitor<'a, 'b, 'c, 'tcx> {
    fn visit_expr(&mut self, ex: &ast::Expr) {
        visit::walk_expr(self, ex);
        my_visit_expr(ex);
    }
    fn visit_item(&mut self, i: &ast::Item) {
        visit::walk_item(self, i);
        my_visit_item(i,
                      self.rbml_w_for_visit_item,
                      self.ecx,
                      self.index);
    }
    fn visit_foreign_item(&mut self, ni: &ast::ForeignItem) {
        visit::walk_foreign_item(self, ni);
        my_visit_foreign_item(ni,
                              self.rbml_w_for_visit_item,
                              self.ecx,
                              self.index);
    }
}

fn encode_info_for_items(ecx: &EncodeContext,
                         rbml_w: &mut Encoder,
                         krate: &ast::Crate)
                         -> Vec<entry<i64>> {
    let mut index = Vec::new();
    rbml_w.start_tag(tag_items_data);
    index.push(entry {
        val: ast::CRATE_NODE_ID as i64,
        pos: rbml_w.writer.tell().unwrap(),
    });
    encode_info_for_mod(ecx,
                        rbml_w,
                        &krate.module,
                        &[],
                        ast::CRATE_NODE_ID,
                        [].iter().cloned().chain(None),
                        syntax::parse::token::special_idents::invalid,
                        ast::Public);

    visit::walk_crate(&mut EncodeVisitor {
        index: &mut index,
        ecx: ecx,
        rbml_w_for_visit_item: &mut *rbml_w,
    }, krate);

    rbml_w.end_tag();
    index
}


// Path and definition ID indexing

#[cfg(stage0)]
fn encode_index<T, F>(rbml_w: &mut Encoder, index: Vec<entry<T>>, mut write_fn: F) where
    F: FnMut(&mut SeekableMemWriter, &T),
    T: Hash<SipHasher>,
{
    let mut buckets: Vec<Vec<entry<T>>> = (0..256u16).map(|_| Vec::new()).collect();
    for elt in index {
        let mut s = SipHasher::new();
        elt.val.hash(&mut s);
        let h = s.finish() as uint;
        (&mut buckets[h % 256]).push(elt);
    }

    rbml_w.start_tag(tag_index);
    let mut bucket_locs = Vec::new();
    rbml_w.start_tag(tag_index_buckets);
    for bucket in &buckets {
        bucket_locs.push(rbml_w.writer.tell().unwrap());
        rbml_w.start_tag(tag_index_buckets_bucket);
        for elt in bucket {
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
    for pos in &bucket_locs {
        assert!(*pos < 0xffff_ffff);
        let wr: &mut SeekableMemWriter = rbml_w.writer;
        wr.write_be_u32(*pos as u32);
    }
    rbml_w.end_tag();
    rbml_w.end_tag();
}
#[cfg(not(stage0))]
fn encode_index<T, F>(rbml_w: &mut Encoder, index: Vec<entry<T>>, mut write_fn: F) where
    F: FnMut(&mut SeekableMemWriter, &T),
    T: Hash,
{
    let mut buckets: Vec<Vec<entry<T>>> = (0..256u16).map(|_| Vec::new()).collect();
    for elt in index {
        let mut s = SipHasher::new();
        elt.val.hash(&mut s);
        let h = s.finish() as uint;
        (&mut buckets[h % 256]).push(elt);
    }

    rbml_w.start_tag(tag_index);
    let mut bucket_locs = Vec::new();
    rbml_w.start_tag(tag_index_buckets);
    for bucket in &buckets {
        bucket_locs.push(rbml_w.writer.tell().unwrap());
        rbml_w.start_tag(tag_index_buckets_bucket);
        for elt in bucket {
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
    for pos in &bucket_locs {
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

fn encode_meta_item(rbml_w: &mut Encoder, mi: &ast::MetaItem) {
    match mi.node {
      ast::MetaWord(ref name) => {
        rbml_w.start_tag(tag_meta_item_word);
        rbml_w.start_tag(tag_meta_item_name);
        rbml_w.writer.write_all(name.as_bytes());
        rbml_w.end_tag();
        rbml_w.end_tag();
      }
      ast::MetaNameValue(ref name, ref value) => {
        match value.node {
          ast::LitStr(ref value, _) => {
            rbml_w.start_tag(tag_meta_item_name_value);
            rbml_w.start_tag(tag_meta_item_name);
            rbml_w.writer.write_all(name.as_bytes());
            rbml_w.end_tag();
            rbml_w.start_tag(tag_meta_item_value);
            rbml_w.writer.write_all(value.as_bytes());
            rbml_w.end_tag();
            rbml_w.end_tag();
          }
          _ => {/* FIXME (#623): encode other variants */ }
        }
      }
      ast::MetaList(ref name, ref items) => {
        rbml_w.start_tag(tag_meta_item_list);
        rbml_w.start_tag(tag_meta_item_name);
        rbml_w.writer.write_all(name.as_bytes());
        rbml_w.end_tag();
        for inner_item in items {
            encode_meta_item(rbml_w, &**inner_item);
        }
        rbml_w.end_tag();
      }
    }
}

fn encode_attributes(rbml_w: &mut Encoder, attrs: &[ast::Attribute]) {
    rbml_w.start_tag(tag_attributes);
    for attr in attrs {
        rbml_w.start_tag(tag_attribute);
        rbml_w.wr_tagged_u8(tag_attribute_is_sugared_doc, attr.node.is_sugared_doc as u8);
        encode_meta_item(rbml_w, &*attr.node.value);
        rbml_w.end_tag();
    }
    rbml_w.end_tag();
}

fn encode_unsafety(rbml_w: &mut Encoder, unsafety: ast::Unsafety) {
    let byte: u8 = match unsafety {
        ast::Unsafety::Normal => 0,
        ast::Unsafety::Unsafe => 1,
    };
    rbml_w.wr_tagged_u8(tag_unsafety, byte);
}

fn encode_paren_sugar(rbml_w: &mut Encoder, paren_sugar: bool) {
    let byte: u8 = if paren_sugar {1} else {0};
    rbml_w.wr_tagged_u8(tag_paren_sugar, byte);
}

fn encode_associated_type_names(rbml_w: &mut Encoder, names: &[ast::Name]) {
    rbml_w.start_tag(tag_associated_type_names);
    for &name in names {
        rbml_w.wr_tagged_str(tag_associated_type_name, &token::get_name(name));
    }
    rbml_w.end_tag();
}

fn encode_polarity(rbml_w: &mut Encoder, polarity: ast::ImplPolarity) {
    let byte: u8 = match polarity {
        ast::ImplPolarity::Positive => 0,
        ast::ImplPolarity::Negative => 1,
    };
    rbml_w.wr_tagged_u8(tag_polarity, byte);
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
        for n in &deps {
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
    for dep in &r {
        encode_crate_dep(rbml_w, (*dep).clone());
    }
    rbml_w.end_tag();
}

fn encode_lang_items(ecx: &EncodeContext, rbml_w: &mut Encoder) {
    rbml_w.start_tag(tag_lang_items);

    for (i, &def_id) in ecx.tcx.lang_items.items() {
        if let Some(id) = def_id {
            if id.krate == ast::LOCAL_CRATE {
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

    for i in &ecx.tcx.lang_items.missing {
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
                rbml_w.writer.write_all(lib.as_bytes());
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

/// Serialize the text of the exported macros
fn encode_macro_defs(rbml_w: &mut Encoder,
                     krate: &ast::Crate) {
    rbml_w.start_tag(tag_macro_defs);
    for def in &krate.exported_macros {
        rbml_w.start_tag(tag_macro_def);

        encode_name(rbml_w, def.ident.name);
        encode_attributes(rbml_w, &def.attrs[]);

        rbml_w.start_tag(tag_macro_def_body);
        rbml_w.wr_str(&pprust::tts_to_string(&def.body[])[]);
        rbml_w.end_tag();

        rbml_w.end_tag();
    }
    rbml_w.end_tag();
}

fn encode_struct_field_attrs(rbml_w: &mut Encoder, krate: &ast::Crate) {
    struct StructFieldVisitor<'a, 'b:'a> {
        rbml_w: &'a mut Encoder<'b>,
    }

    impl<'a, 'b, 'v> Visitor<'v> for StructFieldVisitor<'a, 'b> {
        fn visit_struct_field(&mut self, field: &ast::StructField) {
            self.rbml_w.start_tag(tag_struct_field);
            self.rbml_w.wr_tagged_u32(tag_struct_field_id, field.node.id);
            encode_attributes(self.rbml_w, &field.node.attrs[]);
            self.rbml_w.end_tag();
        }
    }

    rbml_w.start_tag(tag_struct_fields);
    visit::walk_crate(&mut StructFieldVisitor {
        rbml_w: rbml_w
    }, krate);
    rbml_w.end_tag();
}



struct ImplVisitor<'a, 'b:'a, 'c:'a, 'tcx:'b> {
    ecx: &'a EncodeContext<'b, 'tcx>,
    rbml_w: &'a mut Encoder<'c>,
}

impl<'a, 'b, 'c, 'tcx, 'v> Visitor<'v> for ImplVisitor<'a, 'b, 'c, 'tcx> {
    fn visit_item(&mut self, item: &ast::Item) {
        if let ast::ItemImpl(_, _, _, Some(ref trait_ref), _, _) = item.node {
            let def_map = &self.ecx.tcx.def_map;
            let trait_def = def_map.borrow()[trait_ref.ref_id].clone();
            let def_id = trait_def.def_id();

            // Load eagerly if this is an implementation of the Drop trait
            // or if the trait is not defined in this crate.
            if Some(def_id) == self.ecx.tcx.lang_items.drop_trait() ||
                    def_id.krate != ast::LOCAL_CRATE {
                self.rbml_w.start_tag(tag_impls_impl);
                encode_def_id(self.rbml_w, local_def(item.id));
                self.rbml_w.end_tag();
            }
        }
        visit::walk_item(self, item);
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
                    krate: &ast::Crate,
                    rbml_w: &'a mut Encoder) {
    rbml_w.start_tag(tag_impls);

    {
        let mut visitor = ImplVisitor {
            ecx: ecx,
            rbml_w: rbml_w,
        };
        visit::walk_crate(&mut visitor, krate);
    }

    rbml_w.end_tag();
}

fn encode_misc_info(ecx: &EncodeContext,
                    krate: &ast::Crate,
                    rbml_w: &mut Encoder) {
    rbml_w.start_tag(tag_misc_info);
    rbml_w.start_tag(tag_misc_info_crate_items);
    for item in &krate.module.items {
        rbml_w.start_tag(tag_mod_child);
        rbml_w.wr_str(&def_to_string(local_def(item.id))[]);
        rbml_w.end_tag();

        each_auxiliary_node_id(&**item, |auxiliary_node_id| {
            rbml_w.start_tag(tag_mod_child);
            rbml_w.wr_str(&def_to_string(local_def(
                        auxiliary_node_id))[]);
            rbml_w.end_tag();
            true
        });
    }

    // Encode reexports for the root module.
    encode_reexports(ecx, rbml_w, 0, [].iter().cloned().chain(None));

    rbml_w.end_tag();
    rbml_w.end_tag();
}

fn encode_reachable_extern_fns(ecx: &EncodeContext, rbml_w: &mut Encoder) {
    rbml_w.start_tag(tag_reachable_extern_fns);

    for id in ecx.reachable {
        if let Some(ast_map::NodeItem(i)) = ecx.tcx.map.find(*id) {
            if let ast::ItemFn(_, _, abi, ref generics, _) = i.node {
                if abi != abi::Rust && !generics.is_type_parameterized() {
                    rbml_w.wr_tagged_u32(tag_reachable_extern_fn_id, *id);
                }
            }
        }
    }

    rbml_w.end_tag();
}

fn encode_crate_dep(rbml_w: &mut Encoder,
                    dep: decoder::CrateDep) {
    rbml_w.start_tag(tag_crate_dep);
    rbml_w.start_tag(tag_crate_dep_crate_name);
    rbml_w.writer.write_all(dep.name.as_bytes());
    rbml_w.end_tag();
    rbml_w.start_tag(tag_crate_dep_hash);
    rbml_w.writer.write_all(dep.hash.as_str().as_bytes());
    rbml_w.end_tag();
    rbml_w.end_tag();
}

fn encode_hash(rbml_w: &mut Encoder, hash: &Svh) {
    rbml_w.start_tag(tag_crate_hash);
    rbml_w.writer.write_all(hash.as_str().as_bytes());
    rbml_w.end_tag();
}

fn encode_crate_name(rbml_w: &mut Encoder, crate_name: &str) {
    rbml_w.start_tag(tag_crate_crate_name);
    rbml_w.writer.write_all(crate_name.as_bytes());
    rbml_w.end_tag();
}

fn encode_crate_triple(rbml_w: &mut Encoder, triple: &str) {
    rbml_w.start_tag(tag_crate_triple);
    rbml_w.writer.write_all(triple.as_bytes());
    rbml_w.end_tag();
}

fn encode_dylib_dependency_formats(rbml_w: &mut Encoder, ecx: &EncodeContext) {
    rbml_w.start_tag(tag_dylib_dependency_formats);
    match ecx.tcx.dependency_formats.borrow().get(&config::CrateTypeDylib) {
        Some(arr) => {
            let s = arr.iter().enumerate().filter_map(|(i, slot)| {
                slot.map(|kind| (format!("{}:{}", i + 1, match kind {
                    cstore::RequireDynamic => "d",
                    cstore::RequireStatic => "s",
                })).to_string())
            }).collect::<Vec<String>>();
            rbml_w.writer.write_all(s.connect(",").as_bytes());
        }
        None => {}
    }
    rbml_w.end_tag();
}

// NB: Increment this as you change the metadata encoding version.
#[allow(non_upper_case_globals)]
pub const metadata_encoding_version : &'static [u8] = &[b'r', b'u', b's', b't', 0, 0, 0, 1 ];

pub fn encode_metadata(parms: EncodeParams, krate: &ast::Crate) -> Vec<u8> {
    let mut wr = SeekableMemWriter::new();
    encode_metadata_inner(&mut wr, parms, krate);
    let mut v = wr.unwrap();

    // And here we run into yet another obscure archive bug: in which metadata
    // loaded from archives may have trailing garbage bytes. Awhile back one of
    // our tests was failing sporadically on the OSX 64-bit builders (both nopt
    // and opt) by having rbml generate an out-of-bounds panic when looking at
    // metadata.
    //
    // Upon investigation it turned out that the metadata file inside of an rlib
    // (and ar archive) was being corrupted. Some compilations would generate a
    // metadata file which would end in a few extra bytes, while other
    // compilations would not have these extra bytes appended to the end. These
    // extra bytes were interpreted by rbml as an extra tag, so they ended up
    // being interpreted causing the out-of-bounds.
    //
    // The root cause of why these extra bytes were appearing was never
    // discovered, and in the meantime the solution we're employing is to insert
    // the length of the metadata to the start of the metadata. Later on this
    // will allow us to slice the metadata to the precise length that we just
    // generated regardless of trailing bytes that end up in it.
    let len = v.len() as u32;
    v.insert(0, (len >>  0) as u8);
    v.insert(0, (len >>  8) as u8);
    v.insert(0, (len >> 16) as u8);
    v.insert(0, (len >> 24) as u8);
    return v;
}

fn encode_metadata_inner(wr: &mut SeekableMemWriter,
                         parms: EncodeParams,
                         krate: &ast::Crate) {
    struct Stats {
        attr_bytes: u64,
        dep_bytes: u64,
        lang_item_bytes: u64,
        native_lib_bytes: u64,
        plugin_registrar_fn_bytes: u64,
        macro_defs_bytes: u64,
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
        reexports,
        cstore,
        encode_inlined_item,
        link_meta,
        reachable,
        ..
    } = parms;
    let ecx = EncodeContext {
        diag: diag,
        tcx: tcx,
        reexports: reexports,
        item_symbols: item_symbols,
        link_meta: link_meta,
        cstore: cstore,
        encode_inlined_item: RefCell::new(encode_inlined_item),
        type_abbrevs: RefCell::new(FnvHashMap()),
        reachable: reachable,
     };

    let mut rbml_w = writer::Encoder::new(wr);

    encode_crate_name(&mut rbml_w, &ecx.link_meta.crate_name[]);
    encode_crate_triple(&mut rbml_w,
                        &tcx.sess
                           .opts
                           .target_triple
                           []);
    encode_hash(&mut rbml_w, &ecx.link_meta.crate_hash);
    encode_dylib_dependency_formats(&mut rbml_w, &ecx);

    let mut i = rbml_w.writer.tell().unwrap();
    encode_attributes(&mut rbml_w, &krate.attrs[]);
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
    encode_macro_defs(&mut rbml_w, krate);
    stats.macro_defs_bytes = rbml_w.writer.tell().unwrap() - i;

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
        for e in rbml_w.writer.get_ref() {
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
        println!("            impl bytes: {}", stats.impl_bytes);
        println!("            misc bytes: {}", stats.misc_bytes);
        println!("            item bytes: {}", stats.item_bytes);
        println!("           index bytes: {}", stats.index_bytes);
        println!("            zero bytes: {}", stats.zero_bytes);
        println!("           total bytes: {}", stats.total_bytes);
    }
}

// Get the encoded string for a type
pub fn encoded_ty<'tcx>(tcx: &ty::ctxt<'tcx>, t: Ty<'tcx>) -> String {
    let mut wr = SeekableMemWriter::new();
    tyencode::enc_ty(&mut wr, &tyencode::ctxt {
        diag: tcx.sess.diagnostic(),
        ds: def_to_string,
        tcx: tcx,
        abbrevs: &RefCell::new(FnvHashMap())
    }, t);
    String::from_utf8(wr.unwrap()).unwrap()
}
