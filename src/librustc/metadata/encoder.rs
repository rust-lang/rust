// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
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
use session::config;
use metadata::common::*;
use metadata::cstore;
use metadata::cstore::LOCAL_CRATE;
use metadata::decoder;
use metadata::tyencode;
use metadata::index::{self, IndexData};
use metadata::inline::InlinedItemRef;
use middle::def;
use middle::def_id::{CRATE_DEF_INDEX, DefId};
use middle::dependency_format::Linkage;
use middle::stability;
use middle::subst;
use middle::ty::{self, Ty};
use util::nodemap::{FnvHashMap, NodeMap, NodeSet};

use serialize::Encodable;
use std::cell::RefCell;
use std::io::prelude::*;
use std::io::{Cursor, SeekFrom};
use std::rc::Rc;
use std::u32;
use syntax::abi;
use syntax::ast::{self, NodeId, Name, CRATE_NODE_ID, CrateNum};
use syntax::attr;
use syntax::attr::AttrMetaMethods;
use syntax::diagnostic::SpanHandler;
use syntax::parse::token::special_idents;
use syntax;
use rbml::writer::Encoder;

use rustc_front::hir;
use rustc_front::visit::Visitor;
use rustc_front::visit;
use front::map::{LinkedPath, PathElem, PathElems};
use front::map as ast_map;

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

impl<'a, 'tcx> EncodeContext<'a,'tcx> {
    fn local_id(&self, def_id: DefId) -> NodeId {
        self.tcx.map.as_local_node_id(def_id).unwrap()
    }
}

/// "interned" entries referenced by id
#[derive(PartialEq, Eq, Hash)]
pub enum XRef<'tcx> { Predicate(ty::Predicate<'tcx>) }

struct CrateIndex<'tcx> {
    items: IndexData,
    xrefs: FnvHashMap<XRef<'tcx>, u32>, // sequentially-assigned
}

impl<'tcx> CrateIndex<'tcx> {
    fn record(&mut self, id: DefId, rbml_w: &mut Encoder) {
        let position = rbml_w.mark_stable_position();
        self.items.record(id, position);
    }

    fn add_xref(&mut self, xref: XRef<'tcx>) -> u32 {
        let old_len = self.xrefs.len() as u32;
        *self.xrefs.entry(xref).or_insert(old_len)
    }
}

fn encode_name(rbml_w: &mut Encoder, name: Name) {
    rbml_w.wr_tagged_str(tag_paths_data_name, &name.as_str());
}

fn encode_def_id(rbml_w: &mut Encoder, id: DefId) {
    rbml_w.wr_tagged_u64(tag_def_id, def_to_u64(id));
}

/// For every DefId that we create a metadata item for, we include a
/// serialized copy of its DefKey, which allows us to recreate a path.
fn encode_def_id_and_key(ecx: &EncodeContext,
                         rbml_w: &mut Encoder,
                         def_id: DefId)
{
    encode_def_id(rbml_w, def_id);
    encode_def_key(ecx, rbml_w, def_id);
}

fn encode_def_key(ecx: &EncodeContext,
                  rbml_w: &mut Encoder,
                  def_id: DefId)
{
    rbml_w.start_tag(tag_def_key);
    let def_key = ecx.tcx.map.def_key(def_id);
    def_key.encode(rbml_w);
    rbml_w.end_tag();
}

fn encode_trait_ref<'a, 'tcx>(rbml_w: &mut Encoder,
                              ecx: &EncodeContext<'a, 'tcx>,
                              trait_ref: ty::TraitRef<'tcx>,
                              tag: usize) {
    let ty_str_ctxt = &tyencode::ctxt {
        diag: ecx.diag,
        ds: def_to_string,
        tcx: ecx.tcx,
        abbrevs: &ecx.type_abbrevs
    };

    rbml_w.start_tag(tag);
    tyencode::enc_trait_ref(rbml_w, ty_str_ctxt, trait_ref);
    rbml_w.end_tag();
}

// Item info table encoding
fn encode_family(rbml_w: &mut Encoder, c: char) {
    rbml_w.wr_tagged_u8(tag_items_data_item_family, c as u8);
}

pub fn def_to_u64(did: DefId) -> u64 {
    assert!(did.index.as_u32() < u32::MAX);
    (did.krate as u64) << 32 | (did.index.as_usize() as u64)
}

pub fn def_to_string(did: DefId) -> String {
    format!("{}:{}", did.krate, did.index.as_usize())
}

fn encode_item_variances(rbml_w: &mut Encoder,
                         ecx: &EncodeContext,
                         id: NodeId) {
    let v = ecx.tcx.item_variances(ecx.tcx.map.local_def_id(id));
    rbml_w.start_tag(tag_item_variances);
    v.encode(rbml_w);
    rbml_w.end_tag();
}

fn encode_bounds_and_type_for_item<'a, 'tcx>(rbml_w: &mut Encoder,
                                             ecx: &EncodeContext<'a, 'tcx>,
                                             index: &mut CrateIndex<'tcx>,
                                             id: NodeId) {
    encode_bounds_and_type(rbml_w,
                           ecx,
                           index,
                           &ecx.tcx.lookup_item_type(ecx.tcx.map.local_def_id(id)),
                           &ecx.tcx.lookup_predicates(ecx.tcx.map.local_def_id(id)));
}

fn encode_bounds_and_type<'a, 'tcx>(rbml_w: &mut Encoder,
                                    ecx: &EncodeContext<'a, 'tcx>,
                                    index: &mut CrateIndex<'tcx>,
                                    scheme: &ty::TypeScheme<'tcx>,
                                    predicates: &ty::GenericPredicates<'tcx>) {
    encode_generics(rbml_w, ecx, index,
                    &scheme.generics, &predicates, tag_item_generics);
    encode_type(ecx, rbml_w, scheme.ty);
}

fn encode_variant_id(rbml_w: &mut Encoder, vid: DefId) {
    let id = def_to_u64(vid);
    rbml_w.wr_tagged_u64(tag_items_data_item_variant, id);
    rbml_w.wr_tagged_u64(tag_mod_child, id);
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
    tyencode::enc_closure_ty(rbml_w, ty_str_ctxt, closure_type);
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
    tyencode::enc_ty(rbml_w, ty_str_ctxt, typ);
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
    tyencode::enc_trait_ref(rbml_w, ty_str_ctxt, *trait_ref);
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
    tyencode::enc_region(rbml_w, ty_str_ctxt, r);
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

fn encode_symbol(ecx: &EncodeContext,
                 rbml_w: &mut Encoder,
                 id: NodeId) {
    match ecx.item_symbols.borrow().get(&id) {
        Some(x) => {
            debug!("encode_symbol(id={}, str={})", id, *x);
            rbml_w.wr_tagged_str(tag_items_data_item_symbol, x);
        }
        None => {
            ecx.diag.handler().bug(
                &format!("encode_symbol: id not found {}", id));
        }
    }
}

fn encode_disr_val(_: &EncodeContext,
                   rbml_w: &mut Encoder,
                   disr_val: ty::Disr) {
    rbml_w.wr_tagged_str(tag_disr_val, &disr_val.to_string());
}

fn encode_parent_item(rbml_w: &mut Encoder, id: DefId) {
    rbml_w.wr_tagged_u64(tag_items_data_parent_item, def_to_u64(id));
}

fn encode_struct_fields(rbml_w: &mut Encoder,
                        variant: ty::VariantDef) {
    for f in &variant.fields {
        if f.name == special_idents::unnamed_field.name {
            rbml_w.start_tag(tag_item_unnamed_field);
        } else {
            rbml_w.start_tag(tag_item_field);
            encode_name(rbml_w, f.name);
        }
        encode_struct_field_family(rbml_w, f.vis);
        encode_def_id(rbml_w, f.did);
        rbml_w.end_tag();
    }
}

fn encode_enum_variant_info<'a, 'tcx>(ecx: &EncodeContext<'a, 'tcx>,
                                      rbml_w: &mut Encoder,
                                      id: NodeId,
                                      vis: hir::Visibility,
                                      index: &mut CrateIndex<'tcx>) {
    debug!("encode_enum_variant_info(id={})", id);

    let mut disr_val = 0;
    let def = ecx.tcx.lookup_adt_def(ecx.tcx.map.local_def_id(id));
    for variant in &def.variants {
        let vid = variant.did;
        let variant_node_id = ecx.local_id(vid);

        if let ty::VariantKind::Struct = variant.kind() {
            // tuple-like enum variant fields aren't really items so
            // don't try to encode them.
            for field in &variant.fields {
                encode_field(ecx, rbml_w, field, index);
            }
        }

        index.record(vid, rbml_w);
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id_and_key(ecx, rbml_w, vid);
        encode_family(rbml_w, match variant.kind() {
            ty::VariantKind::Unit | ty::VariantKind::Tuple => 'v',
            ty::VariantKind::Struct => 'V'
        });
        encode_name(rbml_w, variant.name);
        encode_parent_item(rbml_w, ecx.tcx.map.local_def_id(id));
        encode_visibility(rbml_w, vis);

        let attrs = ecx.tcx.get_attrs(vid);
        encode_attributes(rbml_w, &attrs);
        encode_repr_attrs(rbml_w, ecx, &attrs);

        let stab = stability::lookup(ecx.tcx, vid);
        encode_stability(rbml_w, stab);

        encode_struct_fields(rbml_w, variant);

        let specified_disr_val = variant.disr_val;
        if specified_disr_val != disr_val {
            encode_disr_val(ecx, rbml_w, specified_disr_val);
            disr_val = specified_disr_val;
        }
        encode_bounds_and_type_for_item(rbml_w, ecx, index, variant_node_id);

        ecx.tcx.map.with_path(variant_node_id, |path| encode_path(rbml_w, path));
        rbml_w.end_tag();
        disr_val = disr_val.wrapping_add(1);
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
        rbml_w.wr_tagged_str(tag, &pe.name().as_str());
    }
    rbml_w.end_tag();
}

/// Iterates through "auxiliary node IDs", which are node IDs that describe
/// top-level items that are sub-items of the given item. Specifically:
///
/// * For newtype structs, iterates through the node ID of the constructor.
fn each_auxiliary_node_id<F>(item: &hir::Item, callback: F) -> bool where
    F: FnOnce(NodeId) -> bool,
{
    let mut continue_ = true;
    match item.node {
        hir::ItemStruct(ref struct_def, _) => {
            // If this is a newtype struct, return the constructor.
            if struct_def.is_tuple() {
                continue_ = callback(struct_def.id());
            }
        }
        _ => {}
    }

    continue_
}

fn encode_reexports(ecx: &EncodeContext,
                    rbml_w: &mut Encoder,
                    id: NodeId) {
    debug!("(encoding info for module) encoding reexports for {}", id);
    match ecx.reexports.get(&id) {
        Some(exports) => {
            debug!("(encoding info for module) found reexports for {}", id);
            for exp in exports {
                debug!("(encoding info for module) reexport '{}' ({:?}) for \
                        {}",
                       exp.name,
                       exp.def_id,
                       id);
                rbml_w.start_tag(tag_items_data_item_reexport);
                rbml_w.wr_tagged_u64(tag_items_data_item_reexport_def_id,
                                     def_to_u64(exp.def_id));
                rbml_w.wr_tagged_str(tag_items_data_item_reexport_name,
                                     &exp.name.as_str());
                rbml_w.end_tag();
            }
        },
        None => debug!("(encoding info for module) found no reexports for {}", id),
    }
}

fn encode_info_for_mod(ecx: &EncodeContext,
                       rbml_w: &mut Encoder,
                       md: &hir::Mod,
                       attrs: &[ast::Attribute],
                       id: NodeId,
                       path: PathElems,
                       name: Name,
                       vis: hir::Visibility) {
    rbml_w.start_tag(tag_items_data_item);
    encode_def_id_and_key(ecx, rbml_w, ecx.tcx.map.local_def_id(id));
    encode_family(rbml_w, 'm');
    encode_name(rbml_w, name);
    debug!("(encoding info for module) encoding info for module ID {}", id);

    // Encode info about all the module children.
    for item in &md.items {
        rbml_w.wr_tagged_u64(tag_mod_child,
                             def_to_u64(ecx.tcx.map.local_def_id(item.id)));

        each_auxiliary_node_id(&**item, |auxiliary_node_id| {
            rbml_w.wr_tagged_u64(tag_mod_child,
                                 def_to_u64(ecx.tcx.map.local_def_id(auxiliary_node_id)));
            true
        });
    }

    encode_path(rbml_w, path.clone());
    encode_visibility(rbml_w, vis);

    let stab = stability::lookup(ecx.tcx, ecx.tcx.map.local_def_id(id));
    encode_stability(rbml_w, stab);

    // Encode the reexports of this module, if this module is public.
    if vis == hir::Public {
        debug!("(encoding info for module) encoding reexports for {}", id);
        encode_reexports(ecx, rbml_w, id);
    }
    encode_attributes(rbml_w, attrs);

    rbml_w.end_tag();
}

fn encode_struct_field_family(rbml_w: &mut Encoder,
                              visibility: hir::Visibility) {
    encode_family(rbml_w, match visibility {
        hir::Public => 'g',
        hir::Inherited => 'N'
    });
}

fn encode_visibility(rbml_w: &mut Encoder, visibility: hir::Visibility) {
    let ch = match visibility {
        hir::Public => 'y',
        hir::Inherited => 'i',
    };
    rbml_w.wr_tagged_u8(tag_items_data_item_visibility, ch as u8);
}

fn encode_constness(rbml_w: &mut Encoder, constness: hir::Constness) {
    rbml_w.start_tag(tag_items_data_item_constness);
    let ch = match constness {
        hir::Constness::Const => 'c',
        hir::Constness::NotConst => 'n',
    };
    rbml_w.wr_str(&ch.to_string());
    rbml_w.end_tag();
}

fn encode_explicit_self(rbml_w: &mut Encoder,
                        explicit_self: &ty::ExplicitSelfCategory) {
    let tag = tag_item_trait_method_explicit_self;

    // Encode the base self type.
    match *explicit_self {
        ty::StaticExplicitSelfCategory => {
            rbml_w.wr_tagged_bytes(tag, &['s' as u8]);
        }
        ty::ByValueExplicitSelfCategory => {
            rbml_w.wr_tagged_bytes(tag, &['v' as u8]);
        }
        ty::ByBoxExplicitSelfCategory => {
            rbml_w.wr_tagged_bytes(tag, &['~' as u8]);
        }
        ty::ByReferenceExplicitSelfCategory(_, m) => {
            // FIXME(#4846) encode custom lifetime
            let ch = encode_mutability(m);
            rbml_w.wr_tagged_bytes(tag, &['&' as u8, ch]);
        }
    }

    fn encode_mutability(m: hir::Mutability) -> u8 {
        match m {
            hir::MutImmutable => 'i' as u8,
            hir::MutMutable => 'm' as u8,
        }
    }
}

fn encode_item_sort(rbml_w: &mut Encoder, sort: char) {
    rbml_w.wr_tagged_u8(tag_item_trait_item_sort, sort as u8);
}

fn encode_field<'a, 'tcx>(ecx: &EncodeContext<'a, 'tcx>,
                          rbml_w: &mut Encoder,
                          field: ty::FieldDef<'tcx>,
                          index: &mut CrateIndex<'tcx>) {
    let nm = field.name;
    let id = ecx.local_id(field.did);

    index.record(field.did, rbml_w);
    rbml_w.start_tag(tag_items_data_item);
    debug!("encode_field: encoding {} {}", nm, id);
    encode_struct_field_family(rbml_w, field.vis);
    encode_name(rbml_w, nm);
    encode_bounds_and_type_for_item(rbml_w, ecx, index, id);
    encode_def_id_and_key(ecx, rbml_w, field.did);

    let stab = stability::lookup(ecx.tcx, field.did);
    encode_stability(rbml_w, stab);

    rbml_w.end_tag();
}

fn encode_info_for_struct_ctor<'a, 'tcx>(ecx: &EncodeContext<'a, 'tcx>,
                                         rbml_w: &mut Encoder,
                                         name: Name,
                                         ctor_id: NodeId,
                                         index: &mut CrateIndex<'tcx>,
                                         struct_id: NodeId) {
    let ctor_def_id = ecx.tcx.map.local_def_id(ctor_id);

    index.record(ctor_def_id, rbml_w);
    rbml_w.start_tag(tag_items_data_item);
    encode_def_id_and_key(ecx, rbml_w, ctor_def_id);
    encode_family(rbml_w, 'o');
    encode_bounds_and_type_for_item(rbml_w, ecx, index, ctor_id);
    encode_name(rbml_w, name);
    ecx.tcx.map.with_path(ctor_id, |path| encode_path(rbml_w, path));
    encode_parent_item(rbml_w, ecx.tcx.map.local_def_id(struct_id));

    if ecx.item_symbols.borrow().contains_key(&ctor_id) {
        encode_symbol(ecx, rbml_w, ctor_id);
    }

    let stab = stability::lookup(ecx.tcx, ecx.tcx.map.local_def_id(ctor_id));
    encode_stability(rbml_w, stab);

    // indicate that this is a tuple struct ctor, because downstream users will normally want
    // the tuple struct definition, but without this there is no way for them to tell that
    // they actually have a ctor rather than a normal function
    rbml_w.wr_tagged_bytes(tag_items_data_item_is_tuple_struct_ctor, &[]);

    rbml_w.end_tag();
}

fn encode_generics<'a, 'tcx>(rbml_w: &mut Encoder,
                             ecx: &EncodeContext<'a, 'tcx>,
                             index: &mut CrateIndex<'tcx>,
                             generics: &ty::Generics<'tcx>,
                             predicates: &ty::GenericPredicates<'tcx>,
                             tag: usize)
{
    rbml_w.start_tag(tag);

    // Type parameters
    let ty_str_ctxt = &tyencode::ctxt {
        diag: ecx.diag,
        ds: def_to_string,
        tcx: ecx.tcx,
        abbrevs: &ecx.type_abbrevs
    };

    for param in &generics.types {
        rbml_w.start_tag(tag_type_param_def);
        tyencode::enc_type_param_def(rbml_w, ty_str_ctxt, param);
        rbml_w.end_tag();
    }

    // Region parameters
    for param in &generics.regions {
        rbml_w.start_tag(tag_region_param_def);

        rbml_w.start_tag(tag_region_param_def_ident);
        encode_name(rbml_w, param.name);
        rbml_w.end_tag();

        rbml_w.wr_tagged_u64(tag_region_param_def_def_id,
                             def_to_u64(param.def_id));

        rbml_w.wr_tagged_u64(tag_region_param_def_space,
                             param.space.to_uint() as u64);

        rbml_w.wr_tagged_u64(tag_region_param_def_index,
                             param.index as u64);

        for &bound_region in &param.bounds {
            encode_region(ecx, rbml_w, bound_region);
        }

        rbml_w.end_tag();
    }

    encode_predicates_in_current_doc(rbml_w, ecx, index, predicates);

    rbml_w.end_tag();
}

fn encode_predicates_in_current_doc<'a,'tcx>(rbml_w: &mut Encoder,
                                             _ecx: &EncodeContext<'a,'tcx>,
                                             index: &mut CrateIndex<'tcx>,
                                             predicates: &ty::GenericPredicates<'tcx>)
{
    for (space, _, predicate) in predicates.predicates.iter_enumerated() {
        let tag = match space {
            subst::TypeSpace => tag_type_predicate,
            subst::SelfSpace => tag_self_predicate,
            subst::FnSpace => tag_fn_predicate
        };

        rbml_w.wr_tagged_u32(tag,
            index.add_xref(XRef::Predicate(predicate.clone())));
    }
}

fn encode_predicates<'a,'tcx>(rbml_w: &mut Encoder,
                              ecx: &EncodeContext<'a,'tcx>,
                              index: &mut CrateIndex<'tcx>,
                              predicates: &ty::GenericPredicates<'tcx>,
                              tag: usize)
{
    rbml_w.start_tag(tag);
    encode_predicates_in_current_doc(rbml_w, ecx, index, predicates);
    rbml_w.end_tag();
}

fn encode_method_ty_fields<'a, 'tcx>(ecx: &EncodeContext<'a, 'tcx>,
                                     rbml_w: &mut Encoder,
                                     index: &mut CrateIndex<'tcx>,
                                     method_ty: &ty::Method<'tcx>) {
    encode_def_id_and_key(ecx, rbml_w, method_ty.def_id);
    encode_name(rbml_w, method_ty.name);
    encode_generics(rbml_w, ecx, index,
                    &method_ty.generics, &method_ty.predicates,
                    tag_method_ty_generics);
    encode_visibility(rbml_w, method_ty.vis);
    encode_explicit_self(rbml_w, &method_ty.explicit_self);
    match method_ty.explicit_self {
        ty::StaticExplicitSelfCategory => {
            encode_family(rbml_w, STATIC_METHOD_FAMILY);
        }
        _ => encode_family(rbml_w, METHOD_FAMILY)
    }
}

fn encode_info_for_associated_const<'a, 'tcx>(ecx: &EncodeContext<'a, 'tcx>,
                                              rbml_w: &mut Encoder,
                                              index: &mut CrateIndex<'tcx>,
                                              associated_const: &ty::AssociatedConst,
                                              impl_path: PathElems,
                                              parent_id: NodeId,
                                              impl_item_opt: Option<&hir::ImplItem>) {
    debug!("encode_info_for_associated_const({:?},{:?})",
           associated_const.def_id,
           associated_const.name);

    index.record(associated_const.def_id, rbml_w);
    rbml_w.start_tag(tag_items_data_item);

    encode_def_id_and_key(ecx, rbml_w, associated_const.def_id);
    encode_name(rbml_w, associated_const.name);
    encode_visibility(rbml_w, associated_const.vis);
    encode_family(rbml_w, 'C');

    encode_parent_item(rbml_w, ecx.tcx.map.local_def_id(parent_id));
    encode_item_sort(rbml_w, 'C');

    encode_bounds_and_type_for_item(rbml_w, ecx, index,
                                    ecx.local_id(associated_const.def_id));

    let stab = stability::lookup(ecx.tcx, associated_const.def_id);
    encode_stability(rbml_w, stab);

    let elem = ast_map::PathName(associated_const.name);
    encode_path(rbml_w, impl_path.chain(Some(elem)));

    if let Some(ii) = impl_item_opt {
        encode_attributes(rbml_w, &ii.attrs);
        encode_inlined_item(ecx,
                            rbml_w,
                            InlinedItemRef::ImplItem(ecx.tcx.map.local_def_id(parent_id),
                                                     ii));
    }

    rbml_w.end_tag();
}

fn encode_info_for_method<'a, 'tcx>(ecx: &EncodeContext<'a, 'tcx>,
                                    rbml_w: &mut Encoder,
                                    index: &mut CrateIndex<'tcx>,
                                    m: &ty::Method<'tcx>,
                                    impl_path: PathElems,
                                    is_default_impl: bool,
                                    parent_id: NodeId,
                                    impl_item_opt: Option<&hir::ImplItem>) {

    debug!("encode_info_for_method: {:?} {:?}", m.def_id,
           m.name);
    index.record(m.def_id, rbml_w);
    rbml_w.start_tag(tag_items_data_item);

    encode_method_ty_fields(ecx, rbml_w, index, m);
    encode_parent_item(rbml_w, ecx.tcx.map.local_def_id(parent_id));
    encode_item_sort(rbml_w, 'r');

    let stab = stability::lookup(ecx.tcx, m.def_id);
    encode_stability(rbml_w, stab);

    let m_node_id = ecx.local_id(m.def_id);
    encode_bounds_and_type_for_item(rbml_w, ecx, index, m_node_id);

    let elem = ast_map::PathName(m.name);
    encode_path(rbml_w, impl_path.chain(Some(elem)));
    if let Some(impl_item) = impl_item_opt {
        if let hir::MethodImplItem(ref sig, _) = impl_item.node {
            encode_attributes(rbml_w, &impl_item.attrs);
            let scheme = ecx.tcx.lookup_item_type(m.def_id);
            let any_types = !scheme.generics.types.is_empty();
            let needs_inline = any_types || is_default_impl ||
                               attr::requests_inline(&impl_item.attrs);
            if needs_inline || sig.constness == hir::Constness::Const {
                encode_inlined_item(ecx,
                                    rbml_w,
                                    InlinedItemRef::ImplItem(ecx.tcx.map.local_def_id(parent_id),
                                                             impl_item));
            }
            encode_constness(rbml_w, sig.constness);
            if !any_types {
                let m_id = ecx.local_id(m.def_id);
                encode_symbol(ecx, rbml_w, m_id);
            }
            encode_method_argument_names(rbml_w, &sig.decl);
        }
    }

    rbml_w.end_tag();
}

fn encode_info_for_associated_type<'a, 'tcx>(ecx: &EncodeContext<'a, 'tcx>,
                                             rbml_w: &mut Encoder,
                                             index: &mut CrateIndex<'tcx>,
                                             associated_type: &ty::AssociatedType<'tcx>,
                                             impl_path: PathElems,
                                             parent_id: NodeId,
                                             impl_item_opt: Option<&hir::ImplItem>) {
    debug!("encode_info_for_associated_type({:?},{:?})",
           associated_type.def_id,
           associated_type.name);

    index.record(associated_type.def_id, rbml_w);
    rbml_w.start_tag(tag_items_data_item);

    encode_def_id_and_key(ecx, rbml_w, associated_type.def_id);
    encode_name(rbml_w, associated_type.name);
    encode_visibility(rbml_w, associated_type.vis);
    encode_family(rbml_w, 'y');
    encode_parent_item(rbml_w, ecx.tcx.map.local_def_id(parent_id));
    encode_item_sort(rbml_w, 't');

    let stab = stability::lookup(ecx.tcx, associated_type.def_id);
    encode_stability(rbml_w, stab);

    let elem = ast_map::PathName(associated_type.name);
    encode_path(rbml_w, impl_path.chain(Some(elem)));

    if let Some(ii) = impl_item_opt {
        encode_attributes(rbml_w, &ii.attrs);
    } else {
        encode_predicates(rbml_w, ecx, index,
                          &ecx.tcx.lookup_predicates(associated_type.def_id),
                          tag_item_generics);
    }

    if let Some(ty) = associated_type.ty {
        encode_type(ecx, rbml_w, ty);
    }

    rbml_w.end_tag();
}

fn encode_method_argument_names(rbml_w: &mut Encoder,
                                decl: &hir::FnDecl) {
    rbml_w.start_tag(tag_method_argument_names);
    for arg in &decl.inputs {
        let tag = tag_method_argument_name;
        if let hir::PatIdent(_, ref path1, _) = arg.pat.node {
            let name = path1.node.name.as_str();
            rbml_w.wr_tagged_bytes(tag, name.as_bytes());
        } else {
            rbml_w.wr_tagged_bytes(tag, &[]);
        }
    }
    rbml_w.end_tag();
}

fn encode_repr_attrs(rbml_w: &mut Encoder,
                     ecx: &EncodeContext,
                     attrs: &[ast::Attribute]) {
    let mut repr_attrs = Vec::new();
    for attr in attrs {
        repr_attrs.extend(attr::find_repr_attrs(ecx.tcx.sess.diagnostic(),
                                                attr));
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

// Encodes the inherent implementations of a structure, enumeration, or trait.
fn encode_inherent_implementations(ecx: &EncodeContext,
                                   rbml_w: &mut Encoder,
                                   def_id: DefId) {
    match ecx.tcx.inherent_impls.borrow().get(&def_id) {
        None => {}
        Some(implementations) => {
            for &impl_def_id in implementations.iter() {
                rbml_w.start_tag(tag_items_data_item_inherent_impl);
                encode_def_id(rbml_w, impl_def_id);
                rbml_w.end_tag();
            }
        }
    }
}

fn encode_stability(rbml_w: &mut Encoder, stab_opt: Option<&attr::Stability>) {
    stab_opt.map(|stab| {
        rbml_w.start_tag(tag_items_data_item_stability);
        stab.encode(rbml_w).unwrap();
        rbml_w.end_tag();
    });
}

fn encode_xrefs<'a, 'tcx>(ecx: &EncodeContext<'a, 'tcx>,
                          rbml_w: &mut Encoder,
                          xrefs: FnvHashMap<XRef<'tcx>, u32>)
{
    let ty_str_ctxt = &tyencode::ctxt {
        diag: ecx.diag,
        ds: def_to_string,
        tcx: ecx.tcx,
        abbrevs: &ecx.type_abbrevs
    };

    let mut xref_positions = vec![0; xrefs.len()];
    rbml_w.start_tag(tag_xref_data);
    for (xref, id) in xrefs.into_iter() {
        xref_positions[id as usize] = rbml_w.mark_stable_position() as u32;
        match xref {
            XRef::Predicate(p) => {
                tyencode::enc_predicate(rbml_w, ty_str_ctxt, &p)
            }
        }
    }
    rbml_w.end_tag();

    rbml_w.start_tag(tag_xref_index);
    index::write_dense_index(xref_positions, rbml_w.writer);
    rbml_w.end_tag();
}

fn encode_info_for_item<'a, 'tcx>(ecx: &EncodeContext<'a, 'tcx>,
                                  rbml_w: &mut Encoder,
                                  item: &hir::Item,
                                  index: &mut CrateIndex<'tcx>,
                                  path: PathElems,
                                  vis: hir::Visibility) {
    let tcx = ecx.tcx;

    debug!("encoding info for item at {}",
           tcx.sess.codemap().span_to_string(item.span));

    let def_id = ecx.tcx.map.local_def_id(item.id);
    let stab = stability::lookup(tcx, ecx.tcx.map.local_def_id(item.id));

    match item.node {
      hir::ItemStatic(_, m, _) => {
        index.record(def_id, rbml_w);
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id_and_key(ecx, rbml_w, def_id);
        if m == hir::MutMutable {
            encode_family(rbml_w, 'b');
        } else {
            encode_family(rbml_w, 'c');
        }
        encode_bounds_and_type_for_item(rbml_w, ecx, index, item.id);
        encode_symbol(ecx, rbml_w, item.id);
        encode_name(rbml_w, item.name);
        encode_path(rbml_w, path);
        encode_visibility(rbml_w, vis);
        encode_stability(rbml_w, stab);
        encode_attributes(rbml_w, &item.attrs);
        rbml_w.end_tag();
      }
      hir::ItemConst(_, _) => {
        index.record(def_id, rbml_w);
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id_and_key(ecx, rbml_w, def_id);
        encode_family(rbml_w, 'C');
        encode_bounds_and_type_for_item(rbml_w, ecx, index, item.id);
        encode_name(rbml_w, item.name);
        encode_path(rbml_w, path);
        encode_attributes(rbml_w, &item.attrs);
        encode_inlined_item(ecx, rbml_w, InlinedItemRef::Item(item));
        encode_visibility(rbml_w, vis);
        encode_stability(rbml_w, stab);
        rbml_w.end_tag();
      }
      hir::ItemFn(ref decl, _, constness, _, ref generics, _) => {
        index.record(def_id, rbml_w);
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id_and_key(ecx, rbml_w, def_id);
        encode_family(rbml_w, FN_FAMILY);
        let tps_len = generics.ty_params.len();
        encode_bounds_and_type_for_item(rbml_w, ecx, index, item.id);
        encode_name(rbml_w, item.name);
        encode_path(rbml_w, path);
        encode_attributes(rbml_w, &item.attrs);
        let needs_inline = tps_len > 0 || attr::requests_inline(&item.attrs);
        if needs_inline || constness == hir::Constness::Const {
            encode_inlined_item(ecx, rbml_w, InlinedItemRef::Item(item));
        }
        if tps_len == 0 {
            encode_symbol(ecx, rbml_w, item.id);
        }
        encode_constness(rbml_w, constness);
        encode_visibility(rbml_w, vis);
        encode_stability(rbml_w, stab);
        encode_method_argument_names(rbml_w, &**decl);
        rbml_w.end_tag();
      }
      hir::ItemMod(ref m) => {
        index.record(def_id, rbml_w);
        encode_info_for_mod(ecx,
                            rbml_w,
                            m,
                            &item.attrs,
                            item.id,
                            path,
                            item.name,
                            item.vis);
      }
      hir::ItemForeignMod(ref fm) => {
        index.record(def_id, rbml_w);
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id_and_key(ecx, rbml_w, def_id);
        encode_family(rbml_w, 'n');
        encode_name(rbml_w, item.name);
        encode_path(rbml_w, path);

        // Encode all the items in this module.
        for foreign_item in &fm.items {
            rbml_w.wr_tagged_u64(tag_mod_child,
                                 def_to_u64(ecx.tcx.map.local_def_id(foreign_item.id)));
        }
        encode_visibility(rbml_w, vis);
        encode_stability(rbml_w, stab);
        rbml_w.end_tag();
      }
      hir::ItemTy(..) => {
        index.record(def_id, rbml_w);
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id_and_key(ecx, rbml_w, def_id);
        encode_family(rbml_w, 'y');
        encode_bounds_and_type_for_item(rbml_w, ecx, index, item.id);
        encode_name(rbml_w, item.name);
        encode_path(rbml_w, path);
        encode_visibility(rbml_w, vis);
        encode_stability(rbml_w, stab);
        rbml_w.end_tag();
      }
      hir::ItemEnum(ref enum_definition, _) => {
        index.record(def_id, rbml_w);

        rbml_w.start_tag(tag_items_data_item);
        encode_def_id_and_key(ecx, rbml_w, def_id);
        encode_family(rbml_w, 't');
        encode_item_variances(rbml_w, ecx, item.id);
        encode_bounds_and_type_for_item(rbml_w, ecx, index, item.id);
        encode_name(rbml_w, item.name);
        encode_attributes(rbml_w, &item.attrs);
        encode_repr_attrs(rbml_w, ecx, &item.attrs);
        for v in &enum_definition.variants {
            encode_variant_id(rbml_w, ecx.tcx.map.local_def_id(v.node.data.id()));
        }
        encode_inlined_item(ecx, rbml_w, InlinedItemRef::Item(item));
        encode_path(rbml_w, path);

        // Encode inherent implementations for this enumeration.
        encode_inherent_implementations(ecx, rbml_w, def_id);

        encode_visibility(rbml_w, vis);
        encode_stability(rbml_w, stab);
        rbml_w.end_tag();

        encode_enum_variant_info(ecx,
                                 rbml_w,
                                 item.id,
                                 vis,
                                 index);
      }
      hir::ItemStruct(ref struct_def, _) => {
        let def = ecx.tcx.lookup_adt_def(def_id);
        let variant = def.struct_variant();

        /* Index the class*/
        index.record(def_id, rbml_w);

        /* Now, make an item for the class itself */
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id_and_key(ecx, rbml_w, def_id);
        encode_family(rbml_w, 'S');
        encode_bounds_and_type_for_item(rbml_w, ecx, index, item.id);

        encode_item_variances(rbml_w, ecx, item.id);
        encode_name(rbml_w, item.name);
        encode_attributes(rbml_w, &item.attrs);
        encode_path(rbml_w, path.clone());
        encode_stability(rbml_w, stab);
        encode_visibility(rbml_w, vis);
        encode_repr_attrs(rbml_w, ecx, &item.attrs);

        /* Encode def_ids for each field and method
         for methods, write all the stuff get_trait_method
        needs to know*/
        encode_struct_fields(rbml_w, variant);

        encode_inlined_item(ecx, rbml_w, InlinedItemRef::Item(item));

        // Encode inherent implementations for this structure.
        encode_inherent_implementations(ecx, rbml_w, def_id);

        if !struct_def.is_struct() {
            let ctor_did = ecx.tcx.map.local_def_id(struct_def.id());
            rbml_w.wr_tagged_u64(tag_items_data_item_struct_ctor,
                                 def_to_u64(ctor_did));
        }

        rbml_w.end_tag();

        for field in &variant.fields {
            encode_field(ecx, rbml_w, field, index);
        }

        // If this is a tuple-like struct, encode the type of the constructor.
        if !struct_def.is_struct() {
            encode_info_for_struct_ctor(ecx, rbml_w, item.name, struct_def.id(), index, item.id);
        }
      }
      hir::ItemDefaultImpl(unsafety, _) => {
          index.record(def_id, rbml_w);
          rbml_w.start_tag(tag_items_data_item);
          encode_def_id_and_key(ecx, rbml_w, def_id);
          encode_family(rbml_w, 'd');
          encode_name(rbml_w, item.name);
          encode_unsafety(rbml_w, unsafety);

          let trait_ref = tcx.impl_trait_ref(ecx.tcx.map.local_def_id(item.id)).unwrap();
          encode_trait_ref(rbml_w, ecx, trait_ref, tag_item_trait_ref);
          rbml_w.end_tag();
      }
      hir::ItemImpl(unsafety, polarity, _, _, _, ref ast_items) => {
        // We need to encode information about the default methods we
        // have inherited, so we drive this based on the impl structure.
        let impl_items = tcx.impl_items.borrow();
        let items = impl_items.get(&def_id).unwrap();

        index.record(def_id, rbml_w);
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id_and_key(ecx, rbml_w, def_id);
        encode_family(rbml_w, 'i');
        encode_bounds_and_type_for_item(rbml_w, ecx, index, item.id);
        encode_name(rbml_w, item.name);
        encode_attributes(rbml_w, &item.attrs);
        encode_unsafety(rbml_w, unsafety);
        encode_polarity(rbml_w, polarity);

        match tcx.custom_coerce_unsized_kinds.borrow().get(&ecx.tcx.map.local_def_id(item.id)) {
            Some(&kind) => {
                rbml_w.start_tag(tag_impl_coerce_unsized_kind);
                kind.encode(rbml_w);
                rbml_w.end_tag();
            }
            None => {}
        }

        for &item_def_id in items {
            rbml_w.start_tag(tag_item_impl_item);
            match item_def_id {
                ty::ConstTraitItemId(item_def_id) => {
                    encode_def_id(rbml_w, item_def_id);
                    encode_item_sort(rbml_w, 'C');
                }
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
        if let Some(trait_ref) = tcx.impl_trait_ref(ecx.tcx.map.local_def_id(item.id)) {
            encode_trait_ref(rbml_w, ecx, trait_ref, tag_item_trait_ref);
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
                Some(&*ast_items[i])
            } else {
                None
            };

            match tcx.impl_or_trait_item(trait_item_def_id.def_id()) {
                ty::ConstTraitItem(ref associated_const) => {
                    encode_info_for_associated_const(ecx,
                                                     rbml_w,
                                                     index,
                                                     &*associated_const,
                                                     path.clone(),
                                                     item.id,
                                                     ast_item)
                }
                ty::MethodTraitItem(ref method_type) => {
                    encode_info_for_method(ecx,
                                           rbml_w,
                                           index,
                                           &**method_type,
                                           path.clone(),
                                           false,
                                           item.id,
                                           ast_item)
                }
                ty::TypeTraitItem(ref associated_type) => {
                    encode_info_for_associated_type(ecx,
                                                    rbml_w,
                                                    index,
                                                    &**associated_type,
                                                    path.clone(),
                                                    item.id,
                                                    ast_item)
                }
            }
        }
      }
      hir::ItemTrait(_, _, _, ref ms) => {
        index.record(def_id, rbml_w);
        rbml_w.start_tag(tag_items_data_item);
        encode_def_id_and_key(ecx, rbml_w, def_id);
        encode_family(rbml_w, 'I');
        encode_item_variances(rbml_w, ecx, item.id);
        let trait_def = tcx.lookup_trait_def(def_id);
        let trait_predicates = tcx.lookup_predicates(def_id);
        encode_unsafety(rbml_w, trait_def.unsafety);
        encode_paren_sugar(rbml_w, trait_def.paren_sugar);
        encode_defaulted(rbml_w, tcx.trait_has_default_impl(def_id));
        encode_associated_type_names(rbml_w, &trait_def.associated_type_names);
        encode_generics(rbml_w, ecx, index,
                        &trait_def.generics, &trait_predicates,
                        tag_item_generics);
        encode_predicates(rbml_w, ecx, index,
                          &tcx.lookup_super_predicates(def_id),
                          tag_item_super_predicates);
        encode_trait_ref(rbml_w, ecx, trait_def.trait_ref, tag_item_trait_ref);
        encode_name(rbml_w, item.name);
        encode_attributes(rbml_w, &item.attrs);
        encode_visibility(rbml_w, vis);
        encode_stability(rbml_w, stab);
        for &method_def_id in tcx.trait_item_def_ids(def_id).iter() {
            rbml_w.start_tag(tag_item_trait_item);
            match method_def_id {
                ty::ConstTraitItemId(const_def_id) => {
                    encode_def_id(rbml_w, const_def_id);
                    encode_item_sort(rbml_w, 'C');
                }
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

            rbml_w.wr_tagged_u64(tag_mod_child,
                                 def_to_u64(method_def_id.def_id()));
        }
        encode_path(rbml_w, path.clone());

        // Encode inherent implementations for this trait.
        encode_inherent_implementations(ecx, rbml_w, def_id);

        rbml_w.end_tag();

        // Now output the trait item info for each trait item.
        let r = tcx.trait_item_def_ids(def_id);
        for (i, &item_def_id) in r.iter().enumerate() {
            assert_eq!(item_def_id.def_id().krate, LOCAL_CRATE);

            index.record(item_def_id.def_id(), rbml_w);
            rbml_w.start_tag(tag_items_data_item);

            encode_parent_item(rbml_w, def_id);

            let stab = stability::lookup(tcx, item_def_id.def_id());
            encode_stability(rbml_w, stab);

            let trait_item_type =
                tcx.impl_or_trait_item(item_def_id.def_id());
            let is_nonstatic_method;
            match trait_item_type {
                ty::ConstTraitItem(associated_const) => {
                    encode_name(rbml_w, associated_const.name);
                    encode_def_id_and_key(ecx, rbml_w, associated_const.def_id);
                    encode_visibility(rbml_w, associated_const.vis);

                    let elem = ast_map::PathName(associated_const.name);
                    encode_path(rbml_w,
                                path.clone().chain(Some(elem)));

                    encode_family(rbml_w, 'C');

                    encode_bounds_and_type_for_item(rbml_w, ecx, index,
                                                    ecx.local_id(associated_const.def_id));

                    is_nonstatic_method = false;
                }
                ty::MethodTraitItem(method_ty) => {
                    let method_def_id = item_def_id.def_id();

                    encode_method_ty_fields(ecx, rbml_w, index, &*method_ty);

                    let elem = ast_map::PathName(method_ty.name);
                    encode_path(rbml_w,
                                path.clone().chain(Some(elem)));

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
                    encode_bounds_and_type_for_item(rbml_w, ecx, index,
                                                    ecx.local_id(method_def_id));

                    is_nonstatic_method = method_ty.explicit_self !=
                        ty::StaticExplicitSelfCategory;
                }
                ty::TypeTraitItem(associated_type) => {
                    encode_name(rbml_w, associated_type.name);
                    encode_def_id_and_key(ecx, rbml_w, associated_type.def_id);

                    let elem = ast_map::PathName(associated_type.name);
                    encode_path(rbml_w,
                                path.clone().chain(Some(elem)));

                    encode_item_sort(rbml_w, 't');
                    encode_family(rbml_w, 'y');

                    if let Some(ty) = associated_type.ty {
                        encode_type(ecx, rbml_w, ty);
                    }

                    is_nonstatic_method = false;
                }
            }

            let trait_item = &*ms[i];
            encode_attributes(rbml_w, &trait_item.attrs);
            match trait_item.node {
                hir::ConstTraitItem(_, ref default) => {
                    if default.is_some() {
                        encode_item_sort(rbml_w, 'C');
                    } else {
                        encode_item_sort(rbml_w, 'c');
                    }

                    encode_inlined_item(ecx, rbml_w,
                                        InlinedItemRef::TraitItem(def_id, trait_item));
                }
                hir::MethodTraitItem(ref sig, ref body) => {
                    // If this is a static method, we've already
                    // encoded this.
                    if is_nonstatic_method {
                        // FIXME: I feel like there is something funny
                        // going on.
                        encode_bounds_and_type_for_item(rbml_w, ecx, index,
                                                        ecx.local_id(item_def_id.def_id()));
                    }

                    if body.is_some() {
                        encode_item_sort(rbml_w, 'p');
                        encode_inlined_item(ecx, rbml_w,
                                            InlinedItemRef::TraitItem(def_id, trait_item));
                    } else {
                        encode_item_sort(rbml_w, 'r');
                    }
                    encode_method_argument_names(rbml_w, &sig.decl);
                }

                hir::TypeTraitItem(..) => {}
            }

            rbml_w.end_tag();
        }
      }
      hir::ItemExternCrate(_) | hir::ItemUse(_) => {
        // these are encoded separately
      }
    }
}

fn encode_info_for_foreign_item<'a, 'tcx>(ecx: &EncodeContext<'a, 'tcx>,
                                          rbml_w: &mut Encoder,
                                          nitem: &hir::ForeignItem,
                                          index: &mut CrateIndex<'tcx>,
                                          path: PathElems,
                                          abi: abi::Abi) {
    let def_id = ecx.tcx.map.local_def_id(nitem.id);

    index.record(def_id, rbml_w);
    rbml_w.start_tag(tag_items_data_item);
    encode_def_id_and_key(ecx, rbml_w, def_id);
    encode_visibility(rbml_w, nitem.vis);
    match nitem.node {
      hir::ForeignItemFn(ref fndecl, _) => {
        encode_family(rbml_w, FN_FAMILY);
        encode_bounds_and_type_for_item(rbml_w, ecx, index, nitem.id);
        encode_name(rbml_w, nitem.name);
        if abi == abi::RustIntrinsic || abi == abi::PlatformIntrinsic {
            encode_inlined_item(ecx, rbml_w, InlinedItemRef::Foreign(nitem));
        }
        encode_attributes(rbml_w, &*nitem.attrs);
        let stab = stability::lookup(ecx.tcx, ecx.tcx.map.local_def_id(nitem.id));
        encode_stability(rbml_w, stab);
        encode_symbol(ecx, rbml_w, nitem.id);
        encode_method_argument_names(rbml_w, &*fndecl);
      }
      hir::ForeignItemStatic(_, mutbl) => {
        if mutbl {
            encode_family(rbml_w, 'b');
        } else {
            encode_family(rbml_w, 'c');
        }
        encode_bounds_and_type_for_item(rbml_w, ecx, index, nitem.id);
        encode_attributes(rbml_w, &*nitem.attrs);
        let stab = stability::lookup(ecx.tcx, ecx.tcx.map.local_def_id(nitem.id));
        encode_stability(rbml_w, stab);
        encode_symbol(ecx, rbml_w, nitem.id);
        encode_name(rbml_w, nitem.name);
      }
    }
    encode_path(rbml_w, path);
    rbml_w.end_tag();
}

fn my_visit_expr(expr: &hir::Expr,
                 rbml_w: &mut Encoder,
                 ecx: &EncodeContext,
                 index: &mut CrateIndex) {
    match expr.node {
        hir::ExprClosure(..) => {
            let def_id = ecx.tcx.map.local_def_id(expr.id);

            index.record(def_id, rbml_w);

            rbml_w.start_tag(tag_items_data_item);
            encode_def_id_and_key(ecx, rbml_w, def_id);

            rbml_w.start_tag(tag_items_closure_ty);
            write_closure_type(ecx, rbml_w, &ecx.tcx.tables.borrow().closure_tys[&def_id]);
            rbml_w.end_tag();

            rbml_w.start_tag(tag_items_closure_kind);
            ecx.tcx.closure_kind(def_id).encode(rbml_w).unwrap();
            rbml_w.end_tag();

            ecx.tcx.map.with_path(expr.id, |path| encode_path(rbml_w, path));

            rbml_w.end_tag();
        }
        _ => { }
    }
}

fn my_visit_item<'a, 'tcx>(i: &hir::Item,
                           rbml_w: &mut Encoder,
                           ecx: &EncodeContext<'a, 'tcx>,
                           index: &mut CrateIndex<'tcx>) {
    ecx.tcx.map.with_path(i.id, |path| {
        encode_info_for_item(ecx, rbml_w, i, index, path, i.vis);
    });
}

fn my_visit_foreign_item<'a, 'tcx>(ni: &hir::ForeignItem,
                                   rbml_w: &mut Encoder,
                                   ecx: &EncodeContext<'a, 'tcx>,
                                   index: &mut CrateIndex<'tcx>) {
    debug!("writing foreign item {}::{}",
            ecx.tcx.map.path_to_string(ni.id),
            ni.name);

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
    index: &'a mut CrateIndex<'tcx>,
}

impl<'a, 'b, 'c, 'tcx, 'v> Visitor<'v> for EncodeVisitor<'a, 'b, 'c, 'tcx> {
    fn visit_expr(&mut self, ex: &hir::Expr) {
        visit::walk_expr(self, ex);
        my_visit_expr(ex, self.rbml_w_for_visit_item, self.ecx, self.index);
    }
    fn visit_item(&mut self, i: &hir::Item) {
        visit::walk_item(self, i);
        my_visit_item(i, self.rbml_w_for_visit_item, self.ecx, self.index);
    }
    fn visit_foreign_item(&mut self, ni: &hir::ForeignItem) {
        visit::walk_foreign_item(self, ni);
        my_visit_foreign_item(ni, self.rbml_w_for_visit_item, self.ecx, self.index);
    }
}

fn encode_info_for_items<'a, 'tcx>(ecx: &EncodeContext<'a, 'tcx>,
                                   rbml_w: &mut Encoder,
                                   krate: &hir::Crate)
                                   -> CrateIndex<'tcx> {
    let mut index = CrateIndex {
        items: IndexData::new(ecx.tcx.map.num_local_def_ids()),
        xrefs: FnvHashMap()
    };
    rbml_w.start_tag(tag_items_data);

    index.record(DefId::local(CRATE_DEF_INDEX), rbml_w);
    encode_info_for_mod(ecx,
                        rbml_w,
                        &krate.module,
                        &[],
                        CRATE_NODE_ID,
                        [].iter().cloned().chain(LinkedPath::empty()),
                        syntax::parse::token::special_idents::invalid.name,
                        hir::Public);

    visit::walk_crate(&mut EncodeVisitor {
        index: &mut index,
        ecx: ecx,
        rbml_w_for_visit_item: &mut *rbml_w,
    }, krate);

    rbml_w.end_tag();
    index
}

fn encode_item_index(rbml_w: &mut Encoder, index: IndexData) {
    rbml_w.start_tag(tag_index);
    index.write_index(rbml_w.writer);
    rbml_w.end_tag();
}

fn encode_meta_item(rbml_w: &mut Encoder, mi: &ast::MetaItem) {
    match mi.node {
      ast::MetaWord(ref name) => {
        rbml_w.start_tag(tag_meta_item_word);
        rbml_w.wr_tagged_str(tag_meta_item_name, name);
        rbml_w.end_tag();
      }
      ast::MetaNameValue(ref name, ref value) => {
        match value.node {
          ast::LitStr(ref value, _) => {
            rbml_w.start_tag(tag_meta_item_name_value);
            rbml_w.wr_tagged_str(tag_meta_item_name, name);
            rbml_w.wr_tagged_str(tag_meta_item_value, value);
            rbml_w.end_tag();
          }
          _ => {/* FIXME (#623): encode other variants */ }
        }
      }
      ast::MetaList(ref name, ref items) => {
        rbml_w.start_tag(tag_meta_item_list);
        rbml_w.wr_tagged_str(tag_meta_item_name, name);
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

fn encode_unsafety(rbml_w: &mut Encoder, unsafety: hir::Unsafety) {
    let byte: u8 = match unsafety {
        hir::Unsafety::Normal => 0,
        hir::Unsafety::Unsafe => 1,
    };
    rbml_w.wr_tagged_u8(tag_unsafety, byte);
}

fn encode_paren_sugar(rbml_w: &mut Encoder, paren_sugar: bool) {
    let byte: u8 = if paren_sugar {1} else {0};
    rbml_w.wr_tagged_u8(tag_paren_sugar, byte);
}

fn encode_defaulted(rbml_w: &mut Encoder, is_defaulted: bool) {
    let byte: u8 = if is_defaulted {1} else {0};
    rbml_w.wr_tagged_u8(tag_defaulted_trait, byte);
}

fn encode_associated_type_names(rbml_w: &mut Encoder, names: &[Name]) {
    rbml_w.start_tag(tag_associated_type_names);
    for &name in names {
        rbml_w.wr_tagged_str(tag_associated_type_name, &name.as_str());
    }
    rbml_w.end_tag();
}

fn encode_polarity(rbml_w: &mut Encoder, polarity: hir::ImplPolarity) {
    let byte: u8 = match polarity {
        hir::ImplPolarity::Positive => 0,
        hir::ImplPolarity::Negative => 1,
    };
    rbml_w.wr_tagged_u8(tag_polarity, byte);
}

fn encode_crate_deps(rbml_w: &mut Encoder, cstore: &cstore::CStore) {
    fn get_ordered_deps(cstore: &cstore::CStore)
                        -> Vec<(CrateNum, Rc<cstore::crate_metadata>)> {
        // Pull the cnums and name,vers,hash out of cstore
        let mut deps = Vec::new();
        cstore.iter_crate_data(|cnum, val| {
            deps.push((cnum, val.clone()));
        });

        // Sort by cnum
        deps.sort_by(|kv1, kv2| kv1.0.cmp(&kv2.0));

        // Sanity-check the crate numbers
        let mut expected_cnum = 1;
        for &(n, _) in &deps {
            assert_eq!(n, expected_cnum);
            expected_cnum += 1;
        }

        deps
    }

    // We're just going to write a list of crate 'name-hash-version's, with
    // the assumption that they are numbered 1 to n.
    // FIXME (#2166): This is not nearly enough to support correct versioning
    // but is enough to get transitive crate dependencies working.
    rbml_w.start_tag(tag_crate_deps);
    for (_cnum, dep) in get_ordered_deps(cstore) {
        encode_crate_dep(rbml_w, &dep);
    }
    rbml_w.end_tag();
}

fn encode_lang_items(ecx: &EncodeContext, rbml_w: &mut Encoder) {
    rbml_w.start_tag(tag_lang_items);

    for (i, &opt_def_id) in ecx.tcx.lang_items.items() {
        if let Some(def_id) = opt_def_id {
            if def_id.is_local() {
                rbml_w.start_tag(tag_lang_items_item);
                rbml_w.wr_tagged_u32(tag_lang_items_item_id, i as u32);
                rbml_w.wr_tagged_u32(tag_lang_items_item_index, def_id.index.as_u32());
                rbml_w.end_tag();
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
                rbml_w.wr_tagged_u32(tag_native_libraries_kind, kind as u32);
                rbml_w.wr_tagged_str(tag_native_libraries_name, lib);
                rbml_w.end_tag();
            }
        }
    }

    rbml_w.end_tag();
}

fn encode_plugin_registrar_fn(ecx: &EncodeContext, rbml_w: &mut Encoder) {
    match ecx.tcx.sess.plugin_registrar_fn.get() {
        Some(id) => {
            let def_id = ecx.tcx.map.local_def_id(id);
            rbml_w.wr_tagged_u32(tag_plugin_registrar_fn, def_id.index.as_u32());
        }
        None => {}
    }
}

fn encode_codemap(ecx: &EncodeContext, rbml_w: &mut Encoder) {
    rbml_w.start_tag(tag_codemap);
    let codemap = ecx.tcx.sess.codemap();

    for filemap in &codemap.files.borrow()[..] {

        if filemap.lines.borrow().is_empty() || filemap.is_imported() {
            // No need to export empty filemaps, as they can't contain spans
            // that need translation.
            // Also no need to re-export imported filemaps, as any downstream
            // crate will import them from their original source.
            continue;
        }

        rbml_w.start_tag(tag_codemap_filemap);
        filemap.encode(rbml_w);
        rbml_w.end_tag();
    }

    rbml_w.end_tag();
}

/// Serialize the text of the exported macros
fn encode_macro_defs(rbml_w: &mut Encoder,
                     krate: &hir::Crate) {
    rbml_w.start_tag(tag_macro_defs);
    for def in &krate.exported_macros {
        rbml_w.start_tag(tag_macro_def);

        encode_name(rbml_w, def.name);
        encode_attributes(rbml_w, &def.attrs);

        rbml_w.wr_tagged_str(tag_macro_def_body,
                             &::syntax::print::pprust::tts_to_string(&def.body));

        rbml_w.end_tag();
    }
    rbml_w.end_tag();
}

fn encode_struct_field_attrs(ecx: &EncodeContext,
                             rbml_w: &mut Encoder,
                             krate: &hir::Crate) {
    struct StructFieldVisitor<'a, 'b:'a, 'c:'a, 'tcx:'b> {
        ecx: &'a EncodeContext<'b, 'tcx>,
        rbml_w: &'a mut Encoder<'c>,
    }

    impl<'a, 'b, 'c, 'tcx, 'v> Visitor<'v> for StructFieldVisitor<'a, 'b, 'c, 'tcx> {
        fn visit_struct_field(&mut self, field: &hir::StructField) {
            self.rbml_w.start_tag(tag_struct_field);
            let def_id = self.ecx.tcx.map.local_def_id(field.node.id);
            encode_def_id(self.rbml_w, def_id);
            encode_attributes(self.rbml_w, &field.node.attrs);
            self.rbml_w.end_tag();
        }
    }

    rbml_w.start_tag(tag_struct_fields);
    visit::walk_crate(&mut StructFieldVisitor { ecx: ecx, rbml_w: rbml_w }, krate);
    rbml_w.end_tag();
}



struct ImplVisitor<'a, 'tcx:'a> {
    tcx: &'a ty::ctxt<'tcx>,
    impls: FnvHashMap<DefId, Vec<DefId>>
}

impl<'a, 'tcx, 'v> Visitor<'v> for ImplVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        if let hir::ItemImpl(..) = item.node {
            let impl_id = self.tcx.map.local_def_id(item.id);
            if let Some(trait_ref) = self.tcx.impl_trait_ref(impl_id) {
                self.impls.entry(trait_ref.def_id)
                    .or_insert(vec![])
                    .push(impl_id);
            }
        }
        visit::walk_item(self, item);
    }
}

/// Encodes an index, mapping each trait to its (local) implementations.
fn encode_impls<'a>(ecx: &'a EncodeContext,
                    krate: &hir::Crate,
                    rbml_w: &'a mut Encoder) {
    let mut visitor = ImplVisitor {
        tcx: ecx.tcx,
        impls: FnvHashMap()
    };
    visit::walk_crate(&mut visitor, krate);

    rbml_w.start_tag(tag_impls);
    for (trait_, trait_impls) in visitor.impls {
        rbml_w.start_tag(tag_impls_trait);
        encode_def_id(rbml_w, trait_);
        for impl_ in trait_impls {
            rbml_w.wr_tagged_u64(tag_impls_trait_impl, def_to_u64(impl_));
        }
        rbml_w.end_tag();
    }
    rbml_w.end_tag();
}

fn encode_misc_info(ecx: &EncodeContext,
                    krate: &hir::Crate,
                    rbml_w: &mut Encoder) {
    rbml_w.start_tag(tag_misc_info);
    rbml_w.start_tag(tag_misc_info_crate_items);
    for item in &krate.module.items {
        rbml_w.wr_tagged_u64(tag_mod_child,
                             def_to_u64(ecx.tcx.map.local_def_id(item.id)));

        each_auxiliary_node_id(&**item, |auxiliary_node_id| {
            rbml_w.wr_tagged_u64(tag_mod_child,
                                 def_to_u64(ecx.tcx.map.local_def_id(auxiliary_node_id)));
            true
        });
    }

    // Encode reexports for the root module.
    encode_reexports(ecx, rbml_w, 0);

    rbml_w.end_tag();
    rbml_w.end_tag();
}

// Encodes all reachable symbols in this crate into the metadata.
//
// This pass is seeded off the reachability list calculated in the
// middle::reachable module but filters out items that either don't have a
// symbol associated with them (they weren't translated) or if they're an FFI
// definition (as that's not defined in this crate).
fn encode_reachable(ecx: &EncodeContext, rbml_w: &mut Encoder) {
    rbml_w.start_tag(tag_reachable_ids);
    for &id in ecx.reachable {
        let def_id = ecx.tcx.map.local_def_id(id);
        rbml_w.wr_tagged_u32(tag_reachable_id, def_id.index.as_u32());
    }
    rbml_w.end_tag();
}

fn encode_crate_dep(rbml_w: &mut Encoder,
                    dep: &cstore::crate_metadata) {
    rbml_w.start_tag(tag_crate_dep);
    rbml_w.wr_tagged_str(tag_crate_dep_crate_name, &dep.name());
    let hash = decoder::get_crate_hash(dep.data());
    rbml_w.wr_tagged_str(tag_crate_dep_hash, hash.as_str());
    rbml_w.wr_tagged_u8(tag_crate_dep_explicitly_linked,
                        dep.explicitly_linked.get() as u8);
    rbml_w.end_tag();
}

fn encode_hash(rbml_w: &mut Encoder, hash: &Svh) {
    rbml_w.wr_tagged_str(tag_crate_hash, hash.as_str());
}

fn encode_rustc_version(rbml_w: &mut Encoder) {
    rbml_w.wr_tagged_str(tag_rustc_version, &rustc_version());
}

fn encode_crate_name(rbml_w: &mut Encoder, crate_name: &str) {
    rbml_w.wr_tagged_str(tag_crate_crate_name, crate_name);
}

fn encode_crate_triple(rbml_w: &mut Encoder, triple: &str) {
    rbml_w.wr_tagged_str(tag_crate_triple, triple);
}

fn encode_dylib_dependency_formats(rbml_w: &mut Encoder, ecx: &EncodeContext) {
    let tag = tag_dylib_dependency_formats;
    match ecx.tcx.sess.dependency_formats.borrow().get(&config::CrateTypeDylib) {
        Some(arr) => {
            let s = arr.iter().enumerate().filter_map(|(i, slot)| {
                let kind = match *slot {
                    Linkage::NotLinked |
                    Linkage::IncludedFromDylib => return None,
                    Linkage::Dynamic => "d",
                    Linkage::Static => "s",
                };
                Some(format!("{}:{}", i + 1, kind))
            }).collect::<Vec<String>>();
            rbml_w.wr_tagged_str(tag, &s.join(","));
        }
        None => {
            rbml_w.wr_tagged_str(tag, "");
        }
    }
}

// NB: Increment this as you change the metadata encoding version.
#[allow(non_upper_case_globals)]
pub const metadata_encoding_version : &'static [u8] = &[b'r', b'u', b's', b't', 0, 0, 0, 2 ];

pub fn encode_metadata(parms: EncodeParams, krate: &hir::Crate) -> Vec<u8> {
    let mut wr = Cursor::new(Vec::new());
    encode_metadata_inner(&mut wr, parms, krate);

    // RBML compacts the encoded bytes whenever appropriate,
    // so there are some garbages left after the end of the data.
    let metalen = wr.seek(SeekFrom::Current(0)).unwrap() as usize;
    let mut v = wr.into_inner();
    v.truncate(metalen);
    assert_eq!(v.len(), metalen);

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

fn encode_metadata_inner(wr: &mut Cursor<Vec<u8>>,
                         parms: EncodeParams,
                         krate: &hir::Crate) {
    struct Stats {
        attr_bytes: u64,
        dep_bytes: u64,
        lang_item_bytes: u64,
        native_lib_bytes: u64,
        plugin_registrar_fn_bytes: u64,
        codemap_bytes: u64,
        macro_defs_bytes: u64,
        impl_bytes: u64,
        misc_bytes: u64,
        item_bytes: u64,
        index_bytes: u64,
        xref_bytes: u64,
        zero_bytes: u64,
        total_bytes: u64,
    }
    let mut stats = Stats {
        attr_bytes: 0,
        dep_bytes: 0,
        lang_item_bytes: 0,
        native_lib_bytes: 0,
        plugin_registrar_fn_bytes: 0,
        codemap_bytes: 0,
        macro_defs_bytes: 0,
        impl_bytes: 0,
        misc_bytes: 0,
        item_bytes: 0,
        index_bytes: 0,
        xref_bytes: 0,
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

    let mut rbml_w = Encoder::new(wr);

    encode_rustc_version(&mut rbml_w);
    encode_crate_name(&mut rbml_w, &ecx.link_meta.crate_name);
    encode_crate_triple(&mut rbml_w, &tcx.sess.opts.target_triple);
    encode_hash(&mut rbml_w, &ecx.link_meta.crate_hash);
    encode_dylib_dependency_formats(&mut rbml_w, &ecx);

    let mut i = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_attributes(&mut rbml_w, &krate.attrs);
    stats.attr_bytes = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    i = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_crate_deps(&mut rbml_w, ecx.cstore);
    stats.dep_bytes = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    // Encode the language items.
    i = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_lang_items(&ecx, &mut rbml_w);
    stats.lang_item_bytes = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    // Encode the native libraries used
    i = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_native_libraries(&ecx, &mut rbml_w);
    stats.native_lib_bytes = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    // Encode the plugin registrar function
    i = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_plugin_registrar_fn(&ecx, &mut rbml_w);
    stats.plugin_registrar_fn_bytes = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    // Encode codemap
    i = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_codemap(&ecx, &mut rbml_w);
    stats.codemap_bytes = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    // Encode macro definitions
    i = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_macro_defs(&mut rbml_w, krate);
    stats.macro_defs_bytes = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    // Encode the def IDs of impls, for coherence checking.
    i = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_impls(&ecx, krate, &mut rbml_w);
    stats.impl_bytes = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    // Encode miscellaneous info.
    i = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_misc_info(&ecx, krate, &mut rbml_w);
    encode_reachable(&ecx, &mut rbml_w);
    stats.misc_bytes = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    // Encode and index the items.
    rbml_w.start_tag(tag_items);
    i = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap();
    let index = encode_info_for_items(&ecx, &mut rbml_w, krate);
    stats.item_bytes = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap() - i;
    rbml_w.end_tag();

    i = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_item_index(&mut rbml_w, index.items);
    stats.index_bytes = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    i = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_xrefs(&ecx, &mut rbml_w, index.xrefs);
    stats.xref_bytes = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    encode_struct_field_attrs(&ecx, &mut rbml_w, krate);

    stats.total_bytes = rbml_w.writer.seek(SeekFrom::Current(0)).unwrap();

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
        println!("         codemap bytes: {}", stats.codemap_bytes);
        println!("       macro def bytes: {}", stats.macro_defs_bytes);
        println!("            impl bytes: {}", stats.impl_bytes);
        println!("            misc bytes: {}", stats.misc_bytes);
        println!("            item bytes: {}", stats.item_bytes);
        println!("           index bytes: {}", stats.index_bytes);
        println!("            xref bytes: {}", stats.xref_bytes);
        println!("            zero bytes: {}", stats.zero_bytes);
        println!("           total bytes: {}", stats.total_bytes);
    }
}

// Get the encoded string for a type
pub fn encoded_ty<'tcx>(tcx: &ty::ctxt<'tcx>, t: Ty<'tcx>) -> Vec<u8> {
    let mut wr = Cursor::new(Vec::new());
    tyencode::enc_ty(&mut Encoder::new(&mut wr), &tyencode::ctxt {
        diag: tcx.sess.diagnostic(),
        ds: def_to_string,
        tcx: tcx,
        abbrevs: &RefCell::new(FnvHashMap())
    }, t);
    wr.into_inner()
}
