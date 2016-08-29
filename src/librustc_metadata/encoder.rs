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

use astencode::encode_inlined_item;
use common::*;
use cstore;
use decoder;
use def_key;
use tyencode;
use index::{self, IndexData};

use middle::cstore::{InlinedItemRef, LinkMeta};
use rustc::hir::def;
use rustc::hir::def_id::{CRATE_DEF_INDEX, DefId};
use middle::dependency_format::Linkage;
use rustc::dep_graph::DepNode;
use rustc::traits::specialization_graph;
use rustc::ty::{self, Ty, TyCtxt};

use rustc::hir::svh::Svh;
use rustc::mir::mir_map::MirMap;
use rustc::session::config::{self, PanicStrategy, CrateTypeRustcMacro};
use rustc::util::nodemap::{FnvHashMap, NodeSet};

use rustc_serialize::Encodable;
use std::cell::RefCell;
use std::io::prelude::*;
use std::io::{Cursor, SeekFrom};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::u32;
use syntax::ast::{self, NodeId, Name, CRATE_NODE_ID, CrateNum};
use syntax::attr;
use syntax;
use syntax_pos::BytePos;
use rbml;

use rustc::hir::{self, PatKind};
use rustc::hir::intravisit::Visitor;
use rustc::hir::intravisit;
use rustc::hir::map::DefKey;

use super::index_builder::{FromId, IndexBuilder, ItemContentBuilder, Untracked, XRef};

pub struct EncodeContext<'a, 'tcx: 'a> {
    pub rbml_w: rbml::writer::Encoder,
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
    pub reexports: &'a def::ExportMap,
    pub link_meta: &'a LinkMeta,
    pub cstore: &'a cstore::CStore,
    pub type_abbrevs: &'a tyencode::abbrev_map<'tcx>,
    pub reachable: &'a NodeSet,
    pub mir_map: &'a MirMap<'tcx>,
}

impl<'a, 'tcx> Deref for EncodeContext<'a, 'tcx> {
    type Target = rbml::writer::Encoder;
    fn deref(&self) -> &Self::Target {
        &self.rbml_w
    }
}

impl<'a, 'tcx> DerefMut for EncodeContext<'a, 'tcx> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.rbml_w
    }
}

fn encode_name(ecx: &mut EncodeContext, name: Name) {
    ecx.wr_tagged_str(tag_paths_data_name, &name.as_str());
}

fn encode_def_id(ecx: &mut EncodeContext, id: DefId) {
    ecx.wr_tagged_u64(tag_def_id, def_to_u64(id));
}

fn encode_def_key(ecx: &mut EncodeContext, key: DefKey) {
    let simple_key = def_key::simplify_def_key(key);
    ecx.start_tag(tag_def_key);
    simple_key.encode(ecx);
    ecx.end_tag();
}

/// For every DefId that we create a metadata item for, we include a
/// serialized copy of its DefKey, which allows us to recreate a path.
fn encode_def_id_and_key(ecx: &mut EncodeContext, def_id: DefId) {
    encode_def_id(ecx, def_id);
    let def_key = ecx.tcx.map.def_key(def_id);
    encode_def_key(ecx, def_key);
}

fn encode_trait_ref<'a, 'tcx>(ecx: &mut EncodeContext<'a, 'tcx>,
                              trait_ref: ty::TraitRef<'tcx>,
                              tag: usize) {
    let cx = ecx.ty_str_ctxt();
    ecx.start_tag(tag);
    tyencode::enc_trait_ref(&mut ecx.writer, &cx, trait_ref);
    ecx.mark_stable_position();
    ecx.end_tag();
}

// Item info table encoding
fn encode_family(ecx: &mut EncodeContext, c: char) {
    ecx.wr_tagged_u8(tag_items_data_item_family, c as u8);
}

pub fn def_to_u64(did: DefId) -> u64 {
    assert!(did.index.as_u32() < u32::MAX);
    (did.krate as u64) << 32 | (did.index.as_usize() as u64)
}

pub fn def_to_string(_tcx: TyCtxt, did: DefId) -> String {
    format!("{}:{}", did.krate, did.index.as_usize())
}

fn encode_item_variances(ecx: &mut EncodeContext, id: NodeId) {
    let v = ecx.tcx.item_variances(ecx.tcx.map.local_def_id(id));
    ecx.start_tag(tag_item_variances);
    v.encode(ecx);
    ecx.end_tag();
}

impl<'a, 'b, 'tcx> ItemContentBuilder<'a, 'b, 'tcx> {
    fn encode_bounds_and_type_for_item(&mut self, def_id: DefId) {
        let tcx = self.tcx;
        self.encode_bounds_and_type(&tcx.lookup_item_type(def_id),
                                    &tcx.lookup_predicates(def_id));
    }

    fn encode_bounds_and_type(&mut self,
                              scheme: &ty::TypeScheme<'tcx>,
                              predicates: &ty::GenericPredicates<'tcx>) {
        self.encode_generics(&scheme.generics, &predicates);
        self.encode_type(scheme.ty);
    }
}

fn encode_variant_id(ecx: &mut EncodeContext, vid: DefId) {
    let id = def_to_u64(vid);
    ecx.wr_tagged_u64(tag_items_data_item_variant, id);
    ecx.wr_tagged_u64(tag_mod_child, id);
}

fn write_closure_type<'a, 'tcx>(ecx: &mut EncodeContext<'a, 'tcx>,
                                closure_type: &ty::ClosureTy<'tcx>) {
    let cx = ecx.ty_str_ctxt();
    tyencode::enc_closure_ty(&mut ecx.writer, &cx, closure_type);
    ecx.mark_stable_position();
}

impl<'a, 'b, 'tcx> ItemContentBuilder<'a, 'b, 'tcx> {
    fn encode_type(&mut self, typ: Ty<'tcx>) {
        let cx = self.ty_str_ctxt();
        self.start_tag(tag_items_data_item_type);
        tyencode::enc_ty(&mut self.writer, &cx, typ);
        self.mark_stable_position();
        self.end_tag();
    }

    fn encode_disr_val(&mut self,
                       disr_val: ty::Disr) {
        // convert to u64 so just the number is printed, without any type info
        self.wr_tagged_str(tag_disr_val, &disr_val.to_u64_unchecked().to_string());
    }

    fn encode_parent_item(&mut self, id: DefId) {
        self.wr_tagged_u64(tag_items_data_parent_item, def_to_u64(id));
    }

    fn encode_struct_fields(&mut self,
                            variant: ty::VariantDef) {
        for f in &variant.fields {
            if variant.kind == ty::VariantKind::Tuple {
                self.start_tag(tag_item_unnamed_field);
            } else {
                self.start_tag(tag_item_field);
                encode_name(self, f.name);
            }
            self.encode_struct_field_family(f.vis);
            encode_def_id(self, f.did);
            self.end_tag();
        }
    }
}

impl<'a, 'b, 'tcx> IndexBuilder<'a, 'b, 'tcx> {
    fn encode_enum_variant_infos(&mut self, enum_did: DefId) {
        debug!("encode_enum_variant_info(enum_did={:?})", enum_did);
        let def = self.tcx.lookup_adt_def(enum_did);
        self.encode_fields(enum_did);
        for (i, variant) in def.variants.iter().enumerate() {
            self.record(variant.did,
                        ItemContentBuilder::encode_enum_variant_info,
                        (enum_did, Untracked(i)));
        }
    }
}

impl<'a, 'b, 'tcx> ItemContentBuilder<'a, 'b, 'tcx> {
    /// Encode data for the given variant of the given ADT. The
    /// index of the variant is untracked: this is ok because we
    /// will have to lookup the adt-def by its id, and that gives us
    /// the right to access any information in the adt-def (including,
    /// e.g., the length of the various vectors).
    fn encode_enum_variant_info(&mut self,
                                (enum_did, Untracked(index)):
                                (DefId, Untracked<usize>)) {
        let tcx = self.tcx;
        let def = tcx.lookup_adt_def(enum_did);
        let variant = &def.variants[index];
        let vid = variant.did;
        encode_def_id_and_key(self, vid);
        encode_family(self, match variant.kind {
            ty::VariantKind::Struct => 'V',
            ty::VariantKind::Tuple => 'v',
            ty::VariantKind::Unit => 'w',
        });
        encode_name(self, variant.name);
        self.encode_parent_item(enum_did);

        let enum_id = tcx.map.as_local_node_id(enum_did).unwrap();
        let enum_vis = &tcx.map.expect_item(enum_id).vis;
        self.encode_visibility(enum_vis);

        let attrs = tcx.get_attrs(vid);
        encode_attributes(self, &attrs);
        self.encode_repr_attrs(&attrs);

        let stab = tcx.lookup_stability(vid);
        let depr = tcx.lookup_deprecation(vid);
        encode_stability(self, stab);
        encode_deprecation(self, depr);

        self.encode_struct_fields(variant);
        self.encode_disr_val(variant.disr_val);
        self.encode_bounds_and_type_for_item(vid);
    }
}

fn encode_reexports(ecx: &mut EncodeContext, id: NodeId) {
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
                ecx.start_tag(tag_items_data_item_reexport);
                ecx.wr_tagged_u64(tag_items_data_item_reexport_def_id,
                                  def_to_u64(exp.def_id));
                ecx.wr_tagged_str(tag_items_data_item_reexport_name,
                                  &exp.name.as_str());
                ecx.end_tag();
            }
        },
        None => debug!("(encoding info for module) found no reexports for {}", id),
    }
}

impl<'a, 'b, 'tcx> ItemContentBuilder<'a, 'b, 'tcx> {
    fn encode_info_for_mod(&mut self,
                           FromId(id, (md, attrs, name, vis)):
                           FromId<(&hir::Mod, &[ast::Attribute], Name, &hir::Visibility)>) {
        let tcx = self.tcx;

        encode_def_id_and_key(self, tcx.map.local_def_id(id));
        encode_family(self, 'm');
        encode_name(self, name);
        debug!("(encoding info for module) encoding info for module ID {}", id);

        // Encode info about all the module children.
        for item_id in &md.item_ids {
            self.wr_tagged_u64(tag_mod_child,
                               def_to_u64(tcx.map.local_def_id(item_id.id)));
        }

        self.encode_visibility(vis);

        let stab = tcx.lookup_stability(tcx.map.local_def_id(id));
        let depr = tcx.lookup_deprecation(tcx.map.local_def_id(id));
        encode_stability(self, stab);
        encode_deprecation(self, depr);

        // Encode the reexports of this module, if this module is public.
        if *vis == hir::Public {
            debug!("(encoding info for module) encoding reexports for {}", id);
            encode_reexports(self, id);
        }
        encode_attributes(self, attrs);
    }

    fn encode_struct_field_family(&mut self,
                                  visibility: ty::Visibility) {
        encode_family(self, if visibility.is_public() { 'g' } else { 'N' });
    }

    fn encode_visibility<T: HasVisibility>(&mut self, visibility: T) {
        let ch = if visibility.is_public() { 'y' } else { 'i' };
        self.wr_tagged_u8(tag_items_data_item_visibility, ch as u8);
    }
}

trait HasVisibility: Sized {
    fn is_public(self) -> bool;
}

impl<'a> HasVisibility for &'a hir::Visibility {
    fn is_public(self) -> bool {
        *self == hir::Public
    }
}

impl HasVisibility for ty::Visibility {
    fn is_public(self) -> bool {
        self == ty::Visibility::Public
    }
}

fn encode_constness(ecx: &mut EncodeContext, constness: hir::Constness) {
    ecx.start_tag(tag_items_data_item_constness);
    let ch = match constness {
        hir::Constness::Const => 'c',
        hir::Constness::NotConst => 'n',
    };
    ecx.wr_str(&ch.to_string());
    ecx.end_tag();
}

fn encode_defaultness(ecx: &mut EncodeContext, defaultness: hir::Defaultness) {
    let ch = match defaultness {
        hir::Defaultness::Default => 'd',
        hir::Defaultness::Final => 'f',
    };
    ecx.wr_tagged_u8(tag_items_data_item_defaultness, ch as u8);
}

fn encode_explicit_self(ecx: &mut EncodeContext,
                        explicit_self: &ty::ExplicitSelfCategory) {
    let tag = tag_item_trait_method_explicit_self;

    // Encode the base self type.
    match *explicit_self {
        ty::ExplicitSelfCategory::Static => {
            ecx.wr_tagged_bytes(tag, &['s' as u8]);
        }
        ty::ExplicitSelfCategory::ByValue => {
            ecx.wr_tagged_bytes(tag, &['v' as u8]);
        }
        ty::ExplicitSelfCategory::ByBox => {
            ecx.wr_tagged_bytes(tag, &['~' as u8]);
        }
        ty::ExplicitSelfCategory::ByReference(_, m) => {
            // FIXME(#4846) encode custom lifetime
            let ch = encode_mutability(m);
            ecx.wr_tagged_bytes(tag, &['&' as u8, ch]);
        }
    }

    fn encode_mutability(m: hir::Mutability) -> u8 {
        match m {
            hir::MutImmutable => 'i' as u8,
            hir::MutMutable => 'm' as u8,
        }
    }
}

fn encode_item_sort(ecx: &mut EncodeContext, sort: char) {
    ecx.wr_tagged_u8(tag_item_trait_item_sort, sort as u8);
}

impl<'a, 'b, 'tcx> IndexBuilder<'a, 'b, 'tcx> {
    fn encode_fields(&mut self,
                     adt_def_id: DefId) {
        let def = self.tcx.lookup_adt_def(adt_def_id);
        for (variant_index, variant) in def.variants.iter().enumerate() {
            for (field_index, field) in variant.fields.iter().enumerate() {
                self.record(field.did,
                            ItemContentBuilder::encode_field,
                            (adt_def_id, Untracked((variant_index, field_index))));
            }
        }
    }
}

impl<'a, 'b, 'tcx> ItemContentBuilder<'a, 'b, 'tcx> {
    /// Encode data for the given field of the given variant of the
    /// given ADT. The indices of the variant/field are untracked:
    /// this is ok because we will have to lookup the adt-def by its
    /// id, and that gives us the right to access any information in
    /// the adt-def (including, e.g., the length of the various
    /// vectors).
    fn encode_field(&mut self,
                    (adt_def_id, Untracked((variant_index, field_index))):
                    (DefId, Untracked<(usize, usize)>)) {
        let tcx = self.tcx;
        let def = tcx.lookup_adt_def(adt_def_id);
        let variant = &def.variants[variant_index];
        let field = &variant.fields[field_index];

        let nm = field.name;
        debug!("encode_field: encoding {} {:?}", nm, field.did);

        self.encode_struct_field_family(field.vis);
        encode_name(self, nm);
        self.encode_bounds_and_type_for_item(field.did);
        encode_def_id_and_key(self, field.did);

        let stab = tcx.lookup_stability(field.did);
        let depr = tcx.lookup_deprecation(field.did);
        encode_stability(self, stab);
        encode_deprecation(self, depr);
    }

    fn encode_struct_ctor(&mut self,
                          (struct_def_id, struct_node_id, ctor_node_id):
                          (DefId, ast::NodeId, ast::NodeId)) {
        let tcx = self.tcx;
        let def = tcx.lookup_adt_def(struct_def_id);
        let variant = def.struct_variant();
        let item = tcx.map.expect_item(struct_node_id);
        let ctor_def_id = tcx.map.local_def_id(ctor_node_id);
        encode_def_id_and_key(self, ctor_def_id);
        encode_family(self, match variant.kind {
            ty::VariantKind::Struct => 'S',
            ty::VariantKind::Tuple => 's',
            ty::VariantKind::Unit => 'u',
        });
        self.encode_bounds_and_type_for_item(ctor_def_id);
        encode_name(self, item.name);
        self.encode_parent_item(struct_def_id);

        let stab = tcx.lookup_stability(ctor_def_id);
        let depr = tcx.lookup_deprecation(ctor_def_id);
        encode_stability(self, stab);
        encode_deprecation(self, depr);

        // indicate that this is a tuple struct ctor, because
        // downstream users will normally want the tuple struct
        // definition, but without this there is no way for them
        // to tell that they actually have a ctor rather than a
        // normal function
        self.wr_tagged_bytes(tag_items_data_item_is_tuple_struct_ctor, &[]);
    }

    fn encode_generics(&mut self,
                       generics: &ty::Generics<'tcx>,
                       predicates: &ty::GenericPredicates<'tcx>)
    {
        let cx = self.ty_str_ctxt();
        self.start_tag(tag_item_generics);
        tyencode::enc_generics(&mut self.writer, &cx, generics);
        self.mark_stable_position();
        self.end_tag();
        self.encode_predicates(predicates, tag_item_predicates);
    }

    fn encode_predicates(&mut self,
                         predicates: &ty::GenericPredicates<'tcx>,
                         tag: usize) {
        self.start_tag(tag);
        if let Some(def_id) = predicates.parent {
            self.wr_tagged_u64(tag_items_data_parent_item, def_to_u64(def_id));
        }
        for predicate in &predicates.predicates {
            let xref = self.add_xref(XRef::Predicate(predicate.clone()));
            self.wr_tagged_u32(tag_predicate, xref);
        }
        self.end_tag();
    }

    fn encode_method_ty_fields(&mut self, method_ty: &ty::Method<'tcx>) {
        encode_def_id_and_key(self, method_ty.def_id);
        encode_name(self, method_ty.name);
        self.encode_generics(&method_ty.generics, &method_ty.predicates);
        self.encode_visibility(method_ty.vis);
        encode_explicit_self(self, &method_ty.explicit_self);
        match method_ty.explicit_self {
            ty::ExplicitSelfCategory::Static => {
                encode_family(self, STATIC_METHOD_FAMILY);
            }
            _ => encode_family(self, METHOD_FAMILY)
        }
    }

    fn encode_info_for_trait_item(&mut self,
                                  (trait_def_id, item_def_id, trait_item):
                                  (DefId, DefId, &hir::TraitItem)) {
        let tcx = self.tcx;

        self.encode_parent_item(trait_def_id);

        let stab = tcx.lookup_stability(item_def_id);
        let depr = tcx.lookup_deprecation(item_def_id);
        encode_stability(self, stab);
        encode_deprecation(self, depr);

        let trait_item_type =
            tcx.impl_or_trait_item(item_def_id);
        let is_nonstatic_method;
        match trait_item_type {
            ty::ConstTraitItem(associated_const) => {
                encode_name(self, associated_const.name);
                encode_def_id_and_key(self, associated_const.def_id);
                self.encode_visibility(associated_const.vis);

                encode_family(self, 'C');

                self.encode_bounds_and_type_for_item(associated_const.def_id);

                is_nonstatic_method = false;
            }
            ty::MethodTraitItem(method_ty) => {
                let method_def_id = item_def_id;

                self.encode_method_ty_fields(&method_ty);

                match method_ty.explicit_self {
                    ty::ExplicitSelfCategory::Static => {
                        encode_family(self, STATIC_METHOD_FAMILY);
                    }
                    _ => {
                        encode_family(self, METHOD_FAMILY);
                    }
                }
                self.encode_bounds_and_type_for_item(method_def_id);

                is_nonstatic_method = method_ty.explicit_self !=
                    ty::ExplicitSelfCategory::Static;
            }
            ty::TypeTraitItem(associated_type) => {
                encode_name(self, associated_type.name);
                encode_def_id_and_key(self, associated_type.def_id);
                encode_item_sort(self, 't');
                encode_family(self, 'y');

                if let Some(ty) = associated_type.ty {
                    self.encode_type(ty);
                }

                is_nonstatic_method = false;
            }
        }

        encode_attributes(self, &trait_item.attrs);
        match trait_item.node {
            hir::ConstTraitItem(_, ref default) => {
                if default.is_some() {
                    encode_item_sort(self, 'C');
                } else {
                    encode_item_sort(self, 'c');
                }

                encode_inlined_item(self,
                                    InlinedItemRef::TraitItem(trait_def_id, trait_item));
                self.encode_mir(item_def_id);
            }
            hir::MethodTraitItem(ref sig, ref body) => {
                // If this is a static method, we've already
                // encoded self.
                if is_nonstatic_method {
                    self.encode_bounds_and_type_for_item(item_def_id);
                }

                if body.is_some() {
                    encode_item_sort(self, 'p');
                    self.encode_mir(item_def_id);
                } else {
                    encode_item_sort(self, 'r');
                }
                self.encode_method_argument_names(&sig.decl);
            }

            hir::TypeTraitItem(..) => {}
        }
    }

    fn encode_info_for_impl_item(&mut self,
                                 (impl_id, impl_item_def_id, ast_item):
                                 (NodeId, DefId, Option<&hir::ImplItem>)) {
        match self.tcx.impl_or_trait_item(impl_item_def_id) {
            ty::ConstTraitItem(ref associated_const) => {
                self.encode_info_for_associated_const(&associated_const,
                                                      impl_id,
                                                      ast_item)
            }
            ty::MethodTraitItem(ref method_type) => {
                self.encode_info_for_method(&method_type,
                                            false,
                                            impl_id,
                                            ast_item)
            }
            ty::TypeTraitItem(ref associated_type) => {
                self.encode_info_for_associated_type(&associated_type,
                                                     impl_id,
                                                     ast_item)
            }
        }
    }

    fn encode_info_for_associated_const(&mut self,
                                        associated_const: &ty::AssociatedConst,
                                        parent_id: NodeId,
                                        impl_item_opt: Option<&hir::ImplItem>) {
        let tcx = self.tcx;
        debug!("encode_info_for_associated_const({:?},{:?})",
               associated_const.def_id,
               associated_const.name);

        encode_def_id_and_key(self, associated_const.def_id);
        encode_name(self, associated_const.name);
        self.encode_visibility(associated_const.vis);
        encode_family(self, 'C');

        self.encode_parent_item(tcx.map.local_def_id(parent_id));
        encode_item_sort(self, 'C');

        self.encode_bounds_and_type_for_item(associated_const.def_id);

        let stab = tcx.lookup_stability(associated_const.def_id);
        let depr = tcx.lookup_deprecation(associated_const.def_id);
        encode_stability(self, stab);
        encode_deprecation(self, depr);

        if let Some(ii) = impl_item_opt {
            encode_attributes(self, &ii.attrs);
            encode_defaultness(self, ii.defaultness);
            encode_inlined_item(self,
                                InlinedItemRef::ImplItem(tcx.map.local_def_id(parent_id),
                                                         ii));
            self.encode_mir(associated_const.def_id);
        }
    }

    fn encode_info_for_method(&mut self,
                              m: &ty::Method<'tcx>,
                              is_default_impl: bool,
                              parent_id: NodeId,
                              impl_item_opt: Option<&hir::ImplItem>) {
        let tcx = self.tcx;

        debug!("encode_info_for_method: {:?} {:?}", m.def_id,
               m.name);
        self.encode_method_ty_fields(m);
        self.encode_parent_item(tcx.map.local_def_id(parent_id));
        encode_item_sort(self, 'r');

        let stab = tcx.lookup_stability(m.def_id);
        let depr = tcx.lookup_deprecation(m.def_id);
        encode_stability(self, stab);
        encode_deprecation(self, depr);

        self.encode_bounds_and_type_for_item(m.def_id);

        if let Some(impl_item) = impl_item_opt {
            if let hir::ImplItemKind::Method(ref sig, _) = impl_item.node {
                encode_attributes(self, &impl_item.attrs);
                let generics = tcx.lookup_generics(m.def_id);
                let types = generics.parent_types as usize + generics.types.len();
                let needs_inline = types > 0 || is_default_impl ||
                    attr::requests_inline(&impl_item.attrs);
                if sig.constness == hir::Constness::Const {
                    encode_inlined_item(
                        self,
                        InlinedItemRef::ImplItem(tcx.map.local_def_id(parent_id),
                                                 impl_item));
                }
                if needs_inline || sig.constness == hir::Constness::Const {
                    self.encode_mir(m.def_id);
                }
                encode_constness(self, sig.constness);
                encode_defaultness(self, impl_item.defaultness);
                self.encode_method_argument_names(&sig.decl);
            }
        }
    }

    fn encode_info_for_associated_type(&mut self,
                                       associated_type: &ty::AssociatedType<'tcx>,
                                       parent_id: NodeId,
                                       impl_item_opt: Option<&hir::ImplItem>) {
        let tcx = self.tcx;
        debug!("encode_info_for_associated_type({:?},{:?})",
               associated_type.def_id,
               associated_type.name);

        encode_def_id_and_key(self, associated_type.def_id);
        encode_name(self, associated_type.name);
        self.encode_visibility(associated_type.vis);
        encode_family(self, 'y');
        self.encode_parent_item(tcx.map.local_def_id(parent_id));
        encode_item_sort(self, 't');

        let stab = tcx.lookup_stability(associated_type.def_id);
        let depr = tcx.lookup_deprecation(associated_type.def_id);
        encode_stability(self, stab);
        encode_deprecation(self, depr);

        if let Some(ii) = impl_item_opt {
            encode_attributes(self, &ii.attrs);
            encode_defaultness(self, ii.defaultness);
        }

        if let Some(ty) = associated_type.ty {
            self.encode_type(ty);
        }
    }

    fn encode_method_argument_names(&mut self,
                                    decl: &hir::FnDecl) {
        self.start_tag(tag_method_argument_names);
        for arg in &decl.inputs {
            let tag = tag_method_argument_name;
            if let PatKind::Binding(_, ref path1, _) = arg.pat.node {
                let name = path1.node.as_str();
                self.wr_tagged_bytes(tag, name.as_bytes());
            } else {
                self.wr_tagged_bytes(tag, &[]);
            }
        }
        self.end_tag();
    }

    fn encode_repr_attrs(&mut self,
                         attrs: &[ast::Attribute]) {
        let mut repr_attrs = Vec::new();
        for attr in attrs {
            repr_attrs.extend(attr::find_repr_attrs(self.tcx.sess.diagnostic(),
                                                    attr));
        }
        self.start_tag(tag_items_data_item_repr);
        repr_attrs.encode(self.ecx);
        self.end_tag();
    }

    fn encode_mir(&mut self, def_id: DefId) {
        if let Some(mir) = self.mir_map.map.get(&def_id) {
            self.start_tag(tag_mir as usize);
            mir.encode(self.ecx);
            self.end_tag();
        }
    }
}

const FN_FAMILY: char = 'f';
const STATIC_METHOD_FAMILY: char = 'F';
const METHOD_FAMILY: char = 'h';

// Encodes the inherent implementations of a structure, enumeration, or trait.
fn encode_inherent_implementations(ecx: &mut EncodeContext,
                                   def_id: DefId) {
    match ecx.tcx.inherent_impls.borrow().get(&def_id) {
        None => {}
        Some(implementations) => {
            for &impl_def_id in implementations.iter() {
                ecx.start_tag(tag_items_data_item_inherent_impl);
                encode_def_id(ecx, impl_def_id);
                ecx.end_tag();
            }
        }
    }
}

fn encode_stability(ecx: &mut EncodeContext, stab_opt: Option<&attr::Stability>) {
    stab_opt.map(|stab| {
        ecx.start_tag(tag_items_data_item_stability);
        stab.encode(ecx).unwrap();
        ecx.end_tag();
    });
}

fn encode_deprecation(ecx: &mut EncodeContext, depr_opt: Option<attr::Deprecation>) {
    depr_opt.map(|depr| {
        ecx.start_tag(tag_items_data_item_deprecation);
        depr.encode(ecx).unwrap();
        ecx.end_tag();
    });
}

fn encode_parent_impl(ecx: &mut EncodeContext, parent_opt: Option<DefId>) {
    parent_opt.map(|parent| {
        ecx.wr_tagged_u64(tag_items_data_parent_impl, def_to_u64(parent));
    });
}

fn encode_xrefs<'a, 'tcx>(ecx: &mut EncodeContext<'a, 'tcx>,
                          xrefs: FnvHashMap<XRef<'tcx>, u32>)
{
    let mut xref_positions = vec![0; xrefs.len()];
    let cx = ecx.ty_str_ctxt();

    // Encode XRefs sorted by their ID
    let mut sorted_xrefs: Vec<_> = xrefs.into_iter().collect();
    sorted_xrefs.sort_by_key(|&(_, id)| id);

    ecx.start_tag(tag_xref_data);
    for (xref, id) in sorted_xrefs.into_iter() {
        xref_positions[id as usize] = ecx.mark_stable_position() as u32;
        match xref {
            XRef::Predicate(p) => {
                tyencode::enc_predicate(&mut ecx.writer, &cx, &p)
            }
        }
    }
    ecx.mark_stable_position();
    ecx.end_tag();

    ecx.start_tag(tag_xref_index);
    index::write_dense_index(xref_positions, &mut ecx.writer);
    ecx.end_tag();
}

impl<'a, 'b, 'tcx> ItemContentBuilder<'a, 'b, 'tcx> {
    fn encode_info_for_item(&mut self,
                            (def_id, item): (DefId, &hir::Item)) {
        let tcx = self.tcx;

        debug!("encoding info for item at {}",
               tcx.sess.codemap().span_to_string(item.span));

        let vis = &item.vis;

        let (stab, depr) = tcx.dep_graph.with_task(DepNode::MetaData(def_id), || {
            (tcx.lookup_stability(tcx.map.local_def_id(item.id)),
             tcx.lookup_deprecation(tcx.map.local_def_id(item.id)))
        });

        match item.node {
            hir::ItemStatic(_, m, _) => {
                encode_def_id_and_key(self, def_id);
                if m == hir::MutMutable {
                    encode_family(self, 'b');
                } else {
                    encode_family(self, 'c');
                }
                self.encode_bounds_and_type_for_item(def_id);
                encode_name(self, item.name);
                self.encode_visibility(vis);
                encode_stability(self, stab);
                encode_deprecation(self, depr);
                encode_attributes(self, &item.attrs);
            }
            hir::ItemConst(..) => {
                encode_def_id_and_key(self, def_id);
                encode_family(self, 'C');
                self.encode_bounds_and_type_for_item(def_id);
                encode_name(self, item.name);
                encode_attributes(self, &item.attrs);
                encode_inlined_item(self, InlinedItemRef::Item(def_id, item));
                self.encode_mir(def_id);
                self.encode_visibility(vis);
                encode_stability(self, stab);
                encode_deprecation(self, depr);
            }
            hir::ItemFn(ref decl, _, constness, _, ref generics, _) => {
                encode_def_id_and_key(self, def_id);
                encode_family(self, FN_FAMILY);
                let tps_len = generics.ty_params.len();
                self.encode_bounds_and_type_for_item(def_id);
                encode_name(self, item.name);
                encode_attributes(self, &item.attrs);
                let needs_inline = tps_len > 0 || attr::requests_inline(&item.attrs);
                if constness == hir::Constness::Const {
                    encode_inlined_item(self, InlinedItemRef::Item(def_id, item));
                }
                if needs_inline || constness == hir::Constness::Const {
                    self.encode_mir(def_id);
                }
                encode_constness(self, constness);
                self.encode_visibility(vis);
                encode_stability(self, stab);
                encode_deprecation(self, depr);
                self.encode_method_argument_names(&decl);
            }
            hir::ItemMod(ref m) => {
                self.encode_info_for_mod(FromId(item.id, (m, &item.attrs, item.name, &item.vis)));
            }
            hir::ItemForeignMod(ref fm) => {
                encode_def_id_and_key(self, def_id);
                encode_family(self, 'n');
                encode_name(self, item.name);

                // Encode all the items in self module.
                for foreign_item in &fm.items {
                    self.wr_tagged_u64(
                        tag_mod_child,
                        def_to_u64(tcx.map.local_def_id(foreign_item.id)));
                }
                self.encode_visibility(vis);
                encode_stability(self, stab);
                encode_deprecation(self, depr);
            }
            hir::ItemTy(..) => {
                encode_def_id_and_key(self, def_id);
                encode_family(self, 'y');
                self.encode_bounds_and_type_for_item(def_id);
                encode_name(self, item.name);
                self.encode_visibility(vis);
                encode_stability(self, stab);
                encode_deprecation(self, depr);
            }
            hir::ItemEnum(ref enum_definition, _) => {
                encode_def_id_and_key(self, def_id);
                encode_family(self, 't');
                encode_item_variances(self, item.id);
                self.encode_bounds_and_type_for_item(def_id);
                encode_name(self, item.name);
                encode_attributes(self, &item.attrs);
                self.encode_repr_attrs(&item.attrs);
                for v in &enum_definition.variants {
                    encode_variant_id(self, tcx.map.local_def_id(v.node.data.id()));
                }

                // Encode inherent implementations for self enumeration.
                encode_inherent_implementations(self, def_id);

                self.encode_visibility(vis);
                encode_stability(self, stab);
                encode_deprecation(self, depr);
            }
            hir::ItemStruct(ref struct_def, _) => {
                /* Index the class*/
                let def = tcx.lookup_adt_def(def_id);
                let variant = def.struct_variant();

                /* Now, make an item for the class itself */
                encode_def_id_and_key(self, def_id);
                encode_family(self, match *struct_def {
                    hir::VariantData::Struct(..) => 'S',
                    hir::VariantData::Tuple(..) => 's',
                    hir::VariantData::Unit(..) => 'u',
                });
                self.encode_bounds_and_type_for_item(def_id);

                encode_item_variances(self, item.id);
                encode_name(self, item.name);
                encode_attributes(self, &item.attrs);
                encode_stability(self, stab);
                encode_deprecation(self, depr);
                self.encode_visibility(vis);
                self.encode_repr_attrs(&item.attrs);

                /* Encode def_ids for each field and method
                for methods, write all the stuff get_trait_method
                needs to know*/
                self.encode_struct_fields(variant);

                // Encode inherent implementations for self structure.
                encode_inherent_implementations(self, def_id);

                if !struct_def.is_struct() {
                    let ctor_did = tcx.map.local_def_id(struct_def.id());
                    self.wr_tagged_u64(tag_items_data_item_struct_ctor,
                                       def_to_u64(ctor_did));
                }
            }
            hir::ItemUnion(..) => {
                let def = self.tcx.lookup_adt_def(def_id);
                let variant = def.struct_variant();

                encode_def_id_and_key(self, def_id);
                encode_family(self, 'U');
                self.encode_bounds_and_type_for_item(def_id);

                encode_item_variances(self, item.id);
                encode_name(self, item.name);
                encode_attributes(self, &item.attrs);
                encode_stability(self, stab);
                encode_deprecation(self, depr);
                self.encode_visibility(vis);
                self.encode_repr_attrs(&item.attrs);

                /* Encode def_ids for each field and method
                for methods, write all the stuff get_trait_method
                needs to know*/
                self.encode_struct_fields(variant);

                encode_inlined_item(self, InlinedItemRef::Item(def_id, item));
                self.encode_mir(def_id);

                // Encode inherent implementations for self union.
                encode_inherent_implementations(self, def_id);
            }
            hir::ItemDefaultImpl(unsafety, _) => {
                encode_def_id_and_key(self, def_id);
                encode_family(self, 'd');
                encode_name(self, item.name);
                encode_unsafety(self, unsafety);

                let trait_ref = tcx.impl_trait_ref(tcx.map.local_def_id(item.id)).unwrap();
                encode_trait_ref(self, trait_ref, tag_item_trait_ref);
            }
            hir::ItemImpl(unsafety, polarity, ..) => {
                // We need to encode information about the default methods we
                // have inherited, so we drive self based on the impl structure.
                let impl_items = tcx.impl_items.borrow();
                let items = &impl_items[&def_id];

                encode_def_id_and_key(self, def_id);
                encode_family(self, 'i');
                self.encode_bounds_and_type_for_item(def_id);
                encode_name(self, item.name);
                encode_attributes(self, &item.attrs);
                encode_unsafety(self, unsafety);
                encode_polarity(self, polarity);

                match
                    tcx.custom_coerce_unsized_kinds
                       .borrow()
                       .get(&tcx.map.local_def_id(item.id))
                {
                    Some(&kind) => {
                        self.start_tag(tag_impl_coerce_unsized_kind);
                        kind.encode(self.ecx);
                        self.end_tag();
                    }
                    None => {}
                }

                for &item_def_id in items {
                    self.start_tag(tag_item_impl_item);
                    match item_def_id {
                        ty::ConstTraitItemId(item_def_id) => {
                            encode_def_id(self, item_def_id);
                            encode_item_sort(self, 'C');
                        }
                        ty::MethodTraitItemId(item_def_id) => {
                            encode_def_id(self, item_def_id);
                            encode_item_sort(self, 'r');
                        }
                        ty::TypeTraitItemId(item_def_id) => {
                            encode_def_id(self, item_def_id);
                            encode_item_sort(self, 't');
                        }
                    }
                    self.end_tag();
                }

                let did = tcx.map.local_def_id(item.id);
                if let Some(trait_ref) = tcx.impl_trait_ref(did) {
                    encode_trait_ref(self, trait_ref, tag_item_trait_ref);

                    let trait_def = tcx.lookup_trait_def(trait_ref.def_id);
                    let parent = trait_def.ancestors(did)
                                          .skip(1)
                                          .next()
                                          .and_then(|node| match node {
                                              specialization_graph::Node::Impl(parent) =>
                                                  Some(parent),
                                              _ => None,
                                          });
                    encode_parent_impl(self, parent);
                }
                encode_stability(self, stab);
                encode_deprecation(self, depr);
            }
            hir::ItemTrait(..) => {
                encode_def_id_and_key(self, def_id);
                encode_family(self, 'I');
                encode_item_variances(self, item.id);
                let trait_def = tcx.lookup_trait_def(def_id);
                let trait_predicates = tcx.lookup_predicates(def_id);
                encode_unsafety(self, trait_def.unsafety);
                encode_paren_sugar(self, trait_def.paren_sugar);
                encode_defaulted(self, tcx.trait_has_default_impl(def_id));
                encode_associated_type_names(self, &trait_def.associated_type_names);
                self.encode_generics(&trait_def.generics, &trait_predicates);
                self.encode_predicates(&tcx.lookup_super_predicates(def_id),
                                       tag_item_super_predicates);
                encode_trait_ref(self, trait_def.trait_ref, tag_item_trait_ref);
                encode_name(self, item.name);
                encode_attributes(self, &item.attrs);
                self.encode_visibility(vis);
                encode_stability(self, stab);
                encode_deprecation(self, depr);
                for &method_def_id in tcx.trait_item_def_ids(def_id).iter() {
                    self.start_tag(tag_item_trait_item);
                    match method_def_id {
                        ty::ConstTraitItemId(const_def_id) => {
                            encode_def_id(self, const_def_id);
                            encode_item_sort(self, 'C');
                        }
                        ty::MethodTraitItemId(method_def_id) => {
                            encode_def_id(self, method_def_id);
                            encode_item_sort(self, 'r');
                        }
                        ty::TypeTraitItemId(type_def_id) => {
                            encode_def_id(self, type_def_id);
                            encode_item_sort(self, 't');
                        }
                    }
                    self.end_tag();

                    self.wr_tagged_u64(tag_mod_child,
                                       def_to_u64(method_def_id.def_id()));
                }

                // Encode inherent implementations for self trait.
                encode_inherent_implementations(self, def_id);
            }
            hir::ItemExternCrate(_) | hir::ItemUse(_) => {
                bug!("cannot encode info for item {:?}", item)
            }
        }
    }
}

impl<'a, 'b, 'tcx> IndexBuilder<'a, 'b, 'tcx> {
    /// In some cases, along with the item itself, we also
    /// encode some sub-items. Usually we want some info from the item
    /// so it's easier to do that here then to wait until we would encounter
    /// normally in the visitor walk.
    fn encode_addl_info_for_item(&mut self,
                                 item: &hir::Item) {
        let def_id = self.tcx.map.local_def_id(item.id);
        match item.node {
            hir::ItemStatic(..) |
            hir::ItemConst(..) |
            hir::ItemFn(..) |
            hir::ItemMod(..) |
            hir::ItemForeignMod(..) |
            hir::ItemExternCrate(..) |
            hir::ItemUse(..) |
            hir::ItemDefaultImpl(..) |
            hir::ItemTy(..) => {
                // no sub-item recording needed in these cases
            }
            hir::ItemEnum(..) => {
                self.encode_enum_variant_infos(def_id);
            }
            hir::ItemStruct(ref struct_def, _) => {
                self.encode_addl_struct_info(def_id, struct_def.id(), item);
            }
            hir::ItemUnion(..) => {
                self.encode_addl_union_info(def_id);
            }
            hir::ItemImpl(.., ref ast_items) => {
                self.encode_addl_impl_info(def_id, item.id, ast_items);
            }
            hir::ItemTrait(.., ref trait_items) => {
                self.encode_addl_trait_info(def_id, trait_items);
            }
        }
    }

    fn encode_addl_struct_info(&mut self,
                               def_id: DefId,
                               struct_node_id: ast::NodeId,
                               item: &hir::Item) {
        let def = self.tcx.lookup_adt_def(def_id);
        let variant = def.struct_variant();

        self.encode_fields(def_id);

        // If this is a tuple-like struct, encode the type of the constructor.
        match variant.kind {
            ty::VariantKind::Struct => {
                // no value for structs like struct Foo { ... }
            }
            ty::VariantKind::Tuple | ty::VariantKind::Unit => {
                // there is a value for structs like `struct
                // Foo()` and `struct Foo`
                let ctor_def_id = self.tcx.map.local_def_id(struct_node_id);
                self.record(ctor_def_id,
                            ItemContentBuilder::encode_struct_ctor,
                            (def_id, item.id, struct_node_id));
            }
        }
    }

    fn encode_addl_union_info(&mut self, def_id: DefId) {
        self.encode_fields(def_id);
    }

    fn encode_addl_impl_info(&mut self,
                             def_id: DefId,
                             impl_id: ast::NodeId,
                             ast_items: &[hir::ImplItem]) {
        let impl_items = self.tcx.impl_items.borrow();
        let items = &impl_items[&def_id];

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

            let trait_item_def_id = trait_item_def_id.def_id();
            self.record(trait_item_def_id,
                        ItemContentBuilder::encode_info_for_impl_item,
                        (impl_id, trait_item_def_id, ast_item));
        }
    }

    fn encode_addl_trait_info(&mut self,
                              def_id: DefId,
                              trait_items: &[hir::TraitItem]) {
        // Now output the trait item info for each trait item.
        let tcx = self.tcx;
        let r = tcx.trait_item_def_ids(def_id);
        for (item_def_id, trait_item) in r.iter().zip(trait_items) {
            let item_def_id = item_def_id.def_id();
            assert!(item_def_id.is_local());
            self.record(item_def_id,
                        ItemContentBuilder::encode_info_for_trait_item,
                        (def_id, item_def_id, trait_item));
        }
    }
}

impl<'a, 'b, 'tcx> ItemContentBuilder<'a, 'b, 'tcx> {
    fn encode_info_for_foreign_item(&mut self,
                                    (def_id, nitem): (DefId, &hir::ForeignItem)) {
        let tcx = self.tcx;

        debug!("writing foreign item {}", tcx.node_path_str(nitem.id));

        encode_def_id_and_key(self, def_id);
        let parent_id = tcx.map.get_parent(nitem.id);
        self.encode_parent_item(tcx.map.local_def_id(parent_id));
        self.encode_visibility(&nitem.vis);
        match nitem.node {
            hir::ForeignItemFn(ref fndecl, _) => {
                encode_family(self, FN_FAMILY);
                self.encode_bounds_and_type_for_item(def_id);
                encode_name(self, nitem.name);
                encode_attributes(self, &nitem.attrs);
                let stab = tcx.lookup_stability(tcx.map.local_def_id(nitem.id));
                let depr = tcx.lookup_deprecation(tcx.map.local_def_id(nitem.id));
                encode_stability(self, stab);
                encode_deprecation(self, depr);
                self.encode_method_argument_names(&fndecl);
            }
            hir::ForeignItemStatic(_, mutbl) => {
                if mutbl {
                    encode_family(self, 'b');
                } else {
                    encode_family(self, 'c');
                }
                self.encode_bounds_and_type_for_item(def_id);
                encode_attributes(self, &nitem.attrs);
                let stab = tcx.lookup_stability(tcx.map.local_def_id(nitem.id));
                let depr = tcx.lookup_deprecation(tcx.map.local_def_id(nitem.id));
                encode_stability(self, stab);
                encode_deprecation(self, depr);
                encode_name(self, nitem.name);
            }
        }
    }
}

struct EncodeVisitor<'a, 'b: 'a, 'tcx: 'b> {
    index: IndexBuilder<'a, 'b, 'tcx>,
}

impl<'a, 'b, 'tcx> Visitor<'tcx> for EncodeVisitor<'a, 'b, 'tcx> {
    fn visit_expr(&mut self, ex: &'tcx hir::Expr) {
        intravisit::walk_expr(self, ex);
        self.index.encode_info_for_expr(ex);
    }
    fn visit_item(&mut self, item: &'tcx hir::Item) {
        intravisit::walk_item(self, item);
        let def_id = self.index.tcx.map.local_def_id(item.id);
        match item.node {
            hir::ItemExternCrate(_) | hir::ItemUse(_) => (), // ignore these
            _ => self.index.record(def_id,
                                   ItemContentBuilder::encode_info_for_item,
                                   (def_id, item)),
        }
        self.index.encode_addl_info_for_item(item);
    }
    fn visit_foreign_item(&mut self, ni: &'tcx hir::ForeignItem) {
        intravisit::walk_foreign_item(self, ni);
        let def_id = self.index.tcx.map.local_def_id(ni.id);
        self.index.record(def_id,
                          ItemContentBuilder::encode_info_for_foreign_item,
                          (def_id, ni));
    }
    fn visit_ty(&mut self, ty: &'tcx hir::Ty) {
        intravisit::walk_ty(self, ty);
        self.index.encode_info_for_ty(ty);
    }
}

impl<'a, 'b, 'tcx> IndexBuilder<'a, 'b, 'tcx> {
    fn encode_info_for_ty(&mut self, ty: &hir::Ty) {
        if let hir::TyImplTrait(_) = ty.node {
            let def_id = self.tcx.map.local_def_id(ty.id);
            self.record(def_id,
                        ItemContentBuilder::encode_info_for_anon_ty,
                        def_id);
        }
    }

    fn encode_info_for_expr(&mut self, expr: &hir::Expr) {
        match expr.node {
            hir::ExprClosure(..) => {
                let def_id = self.tcx.map.local_def_id(expr.id);
                self.record(def_id,
                            ItemContentBuilder::encode_info_for_closure,
                            def_id);
            }
            _ => { }
        }
    }
}

impl<'a, 'b, 'tcx> ItemContentBuilder<'a, 'b, 'tcx> {
    fn encode_info_for_anon_ty(&mut self, def_id: DefId) {
        encode_def_id_and_key(self, def_id);
        encode_family(self, 'y');
        self.encode_bounds_and_type_for_item(def_id);
    }

    fn encode_info_for_closure(&mut self, def_id: DefId) {
        let tcx = self.tcx;
        encode_def_id_and_key(self, def_id);
        encode_name(self, syntax::parse::token::intern("<closure>"));

        self.start_tag(tag_items_closure_ty);
        write_closure_type(self,
                           &tcx.tables.borrow().closure_tys[&def_id]);
        self.end_tag();

        self.start_tag(tag_items_closure_kind);
        tcx.closure_kind(def_id).encode(self.ecx).unwrap();
        self.end_tag();

        assert!(self.mir_map.map.contains_key(&def_id));
        self.encode_mir(def_id);
    }
}

fn encode_info_for_items<'a, 'tcx>(ecx: &mut EncodeContext<'a, 'tcx>)
                                   -> (IndexData, FnvHashMap<XRef<'tcx>, u32>) {
    let krate = ecx.tcx.map.krate();

    ecx.start_tag(tag_items_data);

    let fields = {
        let mut index = IndexBuilder::new(ecx);
        index.record(DefId::local(CRATE_DEF_INDEX),
                     ItemContentBuilder::encode_info_for_mod,
                     FromId(CRATE_NODE_ID, (&krate.module,
                                            &[],
                                            syntax::parse::token::intern(&ecx.link_meta.crate_name),
                                            &hir::Public)));
        let mut visitor = EncodeVisitor {
            index: index,
        };
        krate.visit_all_items(&mut visitor);
        visitor.index.into_fields()
    };

    ecx.end_tag();

    fields
}

fn encode_item_index(ecx: &mut EncodeContext, index: IndexData) {
    ecx.start_tag(tag_index);
    index.write_index(&mut ecx.writer);
    ecx.end_tag();
}

fn encode_attributes(ecx: &mut EncodeContext, attrs: &[ast::Attribute]) {
    ecx.start_tag(tag_attributes);
    ecx.emit_opaque(|opaque_encoder| {
        attrs.encode(opaque_encoder)
    }).unwrap();
    ecx.end_tag();
}

fn encode_unsafety(ecx: &mut EncodeContext, unsafety: hir::Unsafety) {
    let byte: u8 = match unsafety {
        hir::Unsafety::Normal => 0,
        hir::Unsafety::Unsafe => 1,
    };
    ecx.wr_tagged_u8(tag_unsafety, byte);
}

fn encode_paren_sugar(ecx: &mut EncodeContext, paren_sugar: bool) {
    let byte: u8 = if paren_sugar {1} else {0};
    ecx.wr_tagged_u8(tag_paren_sugar, byte);
}

fn encode_defaulted(ecx: &mut EncodeContext, is_defaulted: bool) {
    let byte: u8 = if is_defaulted {1} else {0};
    ecx.wr_tagged_u8(tag_defaulted_trait, byte);
}

fn encode_associated_type_names(ecx: &mut EncodeContext, names: &[Name]) {
    ecx.start_tag(tag_associated_type_names);
    for &name in names {
        ecx.wr_tagged_str(tag_associated_type_name, &name.as_str());
    }
    ecx.end_tag();
}

fn encode_polarity(ecx: &mut EncodeContext, polarity: hir::ImplPolarity) {
    let byte: u8 = match polarity {
        hir::ImplPolarity::Positive => 0,
        hir::ImplPolarity::Negative => 1,
    };
    ecx.wr_tagged_u8(tag_polarity, byte);
}

fn encode_crate_deps(ecx: &mut EncodeContext, cstore: &cstore::CStore) {
    fn get_ordered_deps(cstore: &cstore::CStore)
                        -> Vec<(CrateNum, Rc<cstore::CrateMetadata>)> {
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
    ecx.start_tag(tag_crate_deps);
    for (_cnum, dep) in get_ordered_deps(cstore) {
        encode_crate_dep(ecx, &dep);
    }
    ecx.end_tag();
}

fn encode_lang_items(ecx: &mut EncodeContext) {
    ecx.start_tag(tag_lang_items);

    for (i, &opt_def_id) in ecx.tcx.lang_items.items().iter().enumerate() {
        if let Some(def_id) = opt_def_id {
            if def_id.is_local() {
                ecx.start_tag(tag_lang_items_item);
                ecx.wr_tagged_u32(tag_lang_items_item_id, i as u32);
                ecx.wr_tagged_u32(tag_lang_items_item_index, def_id.index.as_u32());
                ecx.end_tag();
            }
        }
    }

    for i in &ecx.tcx.lang_items.missing {
        ecx.wr_tagged_u32(tag_lang_items_missing, *i as u32);
    }

    ecx.end_tag();   // tag_lang_items
}

fn encode_native_libraries(ecx: &mut EncodeContext) {
    ecx.start_tag(tag_native_libraries);

    for &(ref lib, kind) in ecx.tcx.sess.cstore.used_libraries().iter() {
        match kind {
            cstore::NativeStatic => {} // these libraries are not propagated
            cstore::NativeFramework | cstore::NativeUnknown => {
                ecx.start_tag(tag_native_libraries_lib);
                ecx.wr_tagged_u32(tag_native_libraries_kind, kind as u32);
                ecx.wr_tagged_str(tag_native_libraries_name, lib);
                ecx.end_tag();
            }
        }
    }

    ecx.end_tag();
}

fn encode_plugin_registrar_fn(ecx: &mut EncodeContext) {
    match ecx.tcx.sess.plugin_registrar_fn.get() {
        Some(id) => {
            let def_id = ecx.tcx.map.local_def_id(id);
            ecx.wr_tagged_u32(tag_plugin_registrar_fn, def_id.index.as_u32());
        }
        None => {}
    }
}

fn encode_codemap(ecx: &mut EncodeContext) {
    ecx.start_tag(tag_codemap);
    let codemap = ecx.tcx.sess.codemap();

    for filemap in &codemap.files.borrow()[..] {

        if filemap.lines.borrow().is_empty() || filemap.is_imported() {
            // No need to export empty filemaps, as they can't contain spans
            // that need translation.
            // Also no need to re-export imported filemaps, as any downstream
            // crate will import them from their original source.
            continue;
        }

        ecx.start_tag(tag_codemap_filemap);
        ecx.emit_opaque(|opaque_encoder| {
            filemap.encode(opaque_encoder)
        }).unwrap();
        ecx.end_tag();
    }

    ecx.end_tag();
}

/// Serialize the text of the exported macros
fn encode_macro_defs(ecx: &mut EncodeContext,
                     krate: &hir::Crate) {
    ecx.start_tag(tag_macro_defs);
    for def in &krate.exported_macros {
        ecx.start_tag(tag_macro_def);

        encode_name(ecx, def.name);
        encode_attributes(ecx, &def.attrs);
        let &BytePos(lo) = &def.span.lo;
        let &BytePos(hi) = &def.span.hi;
        ecx.wr_tagged_u32(tag_macro_def_span_lo, lo);
        ecx.wr_tagged_u32(tag_macro_def_span_hi, hi);

        ecx.wr_tagged_str(tag_macro_def_body,
                          &::syntax::print::pprust::tts_to_string(&def.body));

        ecx.end_tag();
    }
    ecx.end_tag();

    if ecx.tcx.sess.crate_types.borrow().contains(&CrateTypeRustcMacro) {
        let id = ecx.tcx.sess.derive_registrar_fn.get().unwrap();
        let did = ecx.tcx.map.local_def_id(id);
        ecx.wr_tagged_u32(tag_macro_derive_registrar, did.index.as_u32());
    }
}

fn encode_struct_field_attrs(ecx: &mut EncodeContext, krate: &hir::Crate) {
    struct StructFieldVisitor<'a, 'b:'a, 'tcx:'b> {
        ecx: &'a mut EncodeContext<'b, 'tcx>
    }

    impl<'a, 'b, 'tcx, 'v> Visitor<'v> for StructFieldVisitor<'a, 'b, 'tcx> {
        fn visit_struct_field(&mut self, field: &hir::StructField) {
            self.ecx.start_tag(tag_struct_field);
            let def_id = self.ecx.tcx.map.local_def_id(field.id);
            encode_def_id(self.ecx, def_id);
            encode_attributes(self.ecx, &field.attrs);
            self.ecx.end_tag();
        }
    }

    ecx.start_tag(tag_struct_fields);
    krate.visit_all_items(&mut StructFieldVisitor { ecx: ecx });
    ecx.end_tag();
}



struct ImplVisitor<'a, 'tcx:'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
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
    }
}

/// Encodes an index, mapping each trait to its (local) implementations.
fn encode_impls(ecx: &mut EncodeContext, krate: &hir::Crate) {
    let mut visitor = ImplVisitor {
        tcx: ecx.tcx,
        impls: FnvHashMap()
    };
    krate.visit_all_items(&mut visitor);

    ecx.start_tag(tag_impls);
    for (trait_, trait_impls) in visitor.impls {
        ecx.start_tag(tag_impls_trait);
        encode_def_id(ecx, trait_);
        for impl_ in trait_impls {
            ecx.wr_tagged_u64(tag_impls_trait_impl, def_to_u64(impl_));
        }
        ecx.end_tag();
    }
    ecx.end_tag();
}

// Encodes all reachable symbols in this crate into the metadata.
//
// This pass is seeded off the reachability list calculated in the
// middle::reachable module but filters out items that either don't have a
// symbol associated with them (they weren't translated) or if they're an FFI
// definition (as that's not defined in this crate).
fn encode_reachable(ecx: &mut EncodeContext) {
    ecx.start_tag(tag_reachable_ids);
    for &id in ecx.reachable {
        let def_id = ecx.tcx.map.local_def_id(id);
        ecx.wr_tagged_u32(tag_reachable_id, def_id.index.as_u32());
    }
    ecx.end_tag();
}

fn encode_crate_dep(ecx: &mut EncodeContext,
                    dep: &cstore::CrateMetadata) {
    ecx.start_tag(tag_crate_dep);
    ecx.wr_tagged_str(tag_crate_dep_crate_name, &dep.name());
    let hash = decoder::get_crate_hash(dep.data());
    ecx.wr_tagged_u64(tag_crate_dep_hash, hash.as_u64());
    ecx.wr_tagged_u8(tag_crate_dep_explicitly_linked,
                     dep.explicitly_linked.get() as u8);
    ecx.end_tag();
}

fn encode_hash(ecx: &mut EncodeContext, hash: &Svh) {
    ecx.wr_tagged_u64(tag_crate_hash, hash.as_u64());
}

fn encode_rustc_version(ecx: &mut EncodeContext) {
    ecx.wr_tagged_str(tag_rustc_version, &rustc_version());
}

fn encode_crate_name(ecx: &mut EncodeContext, crate_name: &str) {
    ecx.wr_tagged_str(tag_crate_crate_name, crate_name);
}

fn encode_crate_disambiguator(ecx: &mut EncodeContext, crate_disambiguator: &str) {
    ecx.wr_tagged_str(tag_crate_disambiguator, crate_disambiguator);
}

fn encode_crate_triple(ecx: &mut EncodeContext, triple: &str) {
    ecx.wr_tagged_str(tag_crate_triple, triple);
}

fn encode_dylib_dependency_formats(ecx: &mut EncodeContext) {
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
            ecx.wr_tagged_str(tag, &s.join(","));
        }
        None => {
            ecx.wr_tagged_str(tag, "");
        }
    }
}

fn encode_panic_strategy(ecx: &mut EncodeContext) {
    match ecx.tcx.sess.opts.cg.panic {
        PanicStrategy::Unwind => {
            ecx.wr_tagged_u8(tag_panic_strategy, b'U');
        }
        PanicStrategy::Abort => {
            ecx.wr_tagged_u8(tag_panic_strategy, b'A');
        }
    }
}

pub fn encode_metadata(mut ecx: EncodeContext, krate: &hir::Crate) -> Vec<u8> {
    encode_metadata_inner(&mut ecx, krate);

    // RBML compacts the encoded bytes whenever appropriate,
    // so there are some garbages left after the end of the data.
    let metalen = ecx.rbml_w.writer.seek(SeekFrom::Current(0)).unwrap() as usize;
    let mut v = ecx.rbml_w.writer.into_inner();
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
    //
    // We also need to store the metadata encoding version here, because
    // rlibs don't have it. To get older versions of rustc to ignore
    // this metadata, there are 4 zero bytes at the start, which are
    // treated as a length of 0 by old compilers.

    let len = v.len();
    let mut result = vec![];
    result.push(0);
    result.push(0);
    result.push(0);
    result.push(0);
    result.extend(metadata_encoding_version.iter().cloned());
    result.push((len >> 24) as u8);
    result.push((len >> 16) as u8);
    result.push((len >>  8) as u8);
    result.push((len >>  0) as u8);
    result.extend(v);
    result
}

fn encode_metadata_inner(ecx: &mut EncodeContext, krate: &hir::Crate) {
    struct Stats {
        attr_bytes: u64,
        dep_bytes: u64,
        lang_item_bytes: u64,
        native_lib_bytes: u64,
        plugin_registrar_fn_bytes: u64,
        codemap_bytes: u64,
        macro_defs_bytes: u64,
        impl_bytes: u64,
        reachable_bytes: u64,
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
        reachable_bytes: 0,
        item_bytes: 0,
        index_bytes: 0,
        xref_bytes: 0,
        zero_bytes: 0,
        total_bytes: 0,
    };

    encode_rustc_version(ecx);

    let tcx = ecx.tcx;
    let link_meta = ecx.link_meta;
    encode_crate_name(ecx, &link_meta.crate_name);
    encode_crate_triple(ecx, &tcx.sess.opts.target_triple);
    encode_hash(ecx, &link_meta.crate_hash);
    encode_crate_disambiguator(ecx, &tcx.sess.local_crate_disambiguator());
    encode_dylib_dependency_formats(ecx);
    encode_panic_strategy(ecx);

    let mut i = ecx.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_attributes(ecx, &krate.attrs);
    stats.attr_bytes = ecx.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    i = ecx.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_crate_deps(ecx, ecx.cstore);
    stats.dep_bytes = ecx.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    // Encode the language items.
    i = ecx.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_lang_items(ecx);
    stats.lang_item_bytes = ecx.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    // Encode the native libraries used
    i = ecx.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_native_libraries(ecx);
    stats.native_lib_bytes = ecx.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    // Encode the plugin registrar function
    i = ecx.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_plugin_registrar_fn(ecx);
    stats.plugin_registrar_fn_bytes = ecx.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    // Encode codemap
    i = ecx.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_codemap(ecx);
    stats.codemap_bytes = ecx.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    // Encode macro definitions
    i = ecx.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_macro_defs(ecx, krate);
    stats.macro_defs_bytes = ecx.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    // Encode the def IDs of impls, for coherence checking.
    i = ecx.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_impls(ecx, krate);
    stats.impl_bytes = ecx.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    // Encode reachability info.
    i = ecx.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_reachable(ecx);
    stats.reachable_bytes = ecx.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    // Encode and index the items.
    ecx.start_tag(tag_items);
    i = ecx.writer.seek(SeekFrom::Current(0)).unwrap();
    let (items, xrefs) = encode_info_for_items(ecx);
    stats.item_bytes = ecx.writer.seek(SeekFrom::Current(0)).unwrap() - i;
    ecx.end_tag();

    i = ecx.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_item_index(ecx, items);
    stats.index_bytes = ecx.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    i = ecx.writer.seek(SeekFrom::Current(0)).unwrap();
    encode_xrefs(ecx, xrefs);
    stats.xref_bytes = ecx.writer.seek(SeekFrom::Current(0)).unwrap() - i;

    encode_struct_field_attrs(ecx, krate);

    stats.total_bytes = ecx.writer.seek(SeekFrom::Current(0)).unwrap();

    if ecx.tcx.sess.meta_stats() {
        for e in ecx.writer.get_ref() {
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
        println!("       reachable bytes: {}", stats.reachable_bytes);
        println!("            item bytes: {}", stats.item_bytes);
        println!("           index bytes: {}", stats.index_bytes);
        println!("            xref bytes: {}", stats.xref_bytes);
        println!("            zero bytes: {}", stats.zero_bytes);
        println!("           total bytes: {}", stats.total_bytes);
    }
}

// Get the encoded string for a type
pub fn encoded_ty<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            t: Ty<'tcx>,
                            def_id_to_string: for<'b> fn(TyCtxt<'b, 'tcx, 'tcx>, DefId) -> String)
                            -> Vec<u8> {
    let mut wr = Cursor::new(Vec::new());
    tyencode::enc_ty(&mut wr, &tyencode::ctxt {
        ds: def_id_to_string,
        tcx: tcx,
        abbrevs: &RefCell::new(FnvHashMap())
    }, t);
    wr.into_inner()
}
