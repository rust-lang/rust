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
use index::{self, IndexData};

use middle::cstore::{InlinedItemRef, LinkMeta, LinkagePreference};
use rustc::hir::def;
use rustc::hir::def_id::{CrateNum, CRATE_DEF_INDEX, DefId};
use middle::dependency_format::Linkage;
use rustc::dep_graph::DepNode;
use rustc::traits::specialization_graph;
use rustc::ty::{self, Ty, TyCtxt};

use rustc::mir::mir_map::MirMap;
use rustc::session::config::{self, CrateTypeRustcMacro};
use rustc::util::nodemap::{FnvHashMap, NodeSet};

use rustc_serialize::{Encodable, Encoder, SpecializedEncoder, opaque};
use std::cell::RefCell;
use std::intrinsics;
use std::io::prelude::*;
use std::io::Cursor;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::u32;
use syntax::ast::{self, NodeId, Name, CRATE_NODE_ID};
use syntax::attr;
use syntax;
use rbml;

use rustc::hir::{self, PatKind};
use rustc::hir::intravisit::Visitor;
use rustc::hir::intravisit;
use rustc::hir::map::DefKey;

use super::index_builder::{FromId, IndexBuilder, Untracked};

pub struct EncodeContext<'a, 'tcx: 'a> {
    rbml_w: rbml::writer::Encoder<'a>,
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
    reexports: &'a def::ExportMap,
    link_meta: &'a LinkMeta,
    cstore: &'a cstore::CStore,
    reachable: &'a NodeSet,
    mir_map: &'a MirMap<'tcx>,

    type_shorthands: RefCell<FnvHashMap<Ty<'tcx>, usize>>,
    xrefs: FnvHashMap<XRef<'tcx>, u32>, // sequentially-assigned
}

/// "interned" entries referenced by id
#[derive(PartialEq, Eq, Hash)]
enum XRef<'tcx> { Predicate(ty::Predicate<'tcx>) }

impl<'a, 'tcx> Deref for EncodeContext<'a, 'tcx> {
    type Target = rbml::writer::Encoder<'a>;
    fn deref(&self) -> &Self::Target {
        &self.rbml_w
    }
}

impl<'a, 'tcx> DerefMut for EncodeContext<'a, 'tcx> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.rbml_w
    }
}

macro_rules! encoder_methods {
    ($($name:ident($ty:ty);)*) => {
        $(fn $name(&mut self, value: $ty) -> Result<(), Self::Error> {
            self.opaque.$name(value)
        })*
    }
}

impl<'a, 'tcx> Encoder for EncodeContext<'a, 'tcx> {
    type Error = <opaque::Encoder<'a> as Encoder>::Error;

    fn emit_nil(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    encoder_methods! {
        emit_usize(usize);
        emit_u64(u64);
        emit_u32(u32);
        emit_u16(u16);
        emit_u8(u8);

        emit_isize(isize);
        emit_i64(i64);
        emit_i32(i32);
        emit_i16(i16);
        emit_i8(i8);

        emit_bool(bool);
        emit_f64(f64);
        emit_f32(f32);
        emit_char(char);
        emit_str(&str);
    }
}

impl<'a, 'tcx> SpecializedEncoder<Ty<'tcx>> for EncodeContext<'a, 'tcx> {
    fn specialized_encode(&mut self, ty: &Ty<'tcx>) -> Result<(), Self::Error> {
        let existing_shorthand = self.type_shorthands.borrow().get(ty).cloned();
        if let Some(shorthand) = existing_shorthand {
            return self.emit_usize(shorthand);
        }

        let start = self.mark_stable_position();
        ty.sty.encode(self)?;
        let len = self.mark_stable_position() - start;

        // The shorthand encoding uses the same usize as the
        // discriminant, with an offset so they can't conflict.
        let discriminant = unsafe { intrinsics::discriminant_value(&ty.sty) };
        assert!(discriminant < TYPE_SHORTHAND_OFFSET as u64);
        let shorthand = start + TYPE_SHORTHAND_OFFSET;

        // Get the number of bits that leb128 could fit
        // in the same space as the fully encoded type.
        let leb128_bits = len * 7;

        // Check that the shorthand is a not longer than the
        // full encoding itself, i.e. it's an obvious win.
        if leb128_bits >= 64 || (shorthand as u64) < (1 << leb128_bits) {
            self.type_shorthands.borrow_mut().insert(*ty, shorthand);
        }

        Ok(())
    }
}

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
    fn seq<I, F, T>(&mut self, iter: I, mut f: F)
    where I: IntoIterator,
          I::IntoIter: ExactSizeIterator,
          F: FnMut(&mut Self, I::Item) -> T,
          T: Encodable {
        let iter = iter.into_iter();
        self.emit_seq(iter.len(), move |ecx| {
            for (i, elem) in iter.enumerate() {
                ecx.emit_seq_elt(i, |ecx| {
                    f(ecx, elem).encode(ecx)
                })?;
            }
            Ok(())
        }).unwrap();
    }
}

fn encode_name(ecx: &mut EncodeContext, name: Name) {
    ecx.start_tag(tag_paths_data_name);
    name.encode(ecx).unwrap();
    ecx.end_tag();
}

fn encode_def_id(ecx: &mut EncodeContext, def_id: DefId) {
    assert!(def_id.is_local());
    ecx.start_tag(tag_def_index);
    def_id.index.encode(ecx).unwrap();
    ecx.end_tag();
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
    ecx.start_tag(tag);
    trait_ref.encode(ecx).unwrap();
    ecx.end_tag();
}

// Item info table encoding
fn encode_family(ecx: &mut EncodeContext, f: Family) {
    ecx.start_tag(tag_items_data_item_family);
    f.encode(ecx).unwrap();
    ecx.end_tag();
}

fn encode_item_variances(ecx: &mut EncodeContext, id: NodeId) {
    let v = ecx.tcx.item_variances(ecx.tcx.map.local_def_id(id));
    ecx.start_tag(tag_item_variances);
    v.encode(ecx);
    ecx.end_tag();
}

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
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

    fn encode_type(&mut self, typ: Ty<'tcx>) {
        self.start_tag(tag_items_data_item_type);
        typ.encode(self).unwrap();
        self.end_tag();
    }

    fn encode_disr_val(&mut self,
                       disr_val: ty::Disr) {
        self.start_tag(tag_disr_val);
        disr_val.to_u64_unchecked().encode(self).unwrap();
        self.end_tag();
    }

    fn encode_parent_item(&mut self, id: DefId) {
        self.start_tag(tag_items_data_parent_item);
        id.encode(self).unwrap();
        self.end_tag();
    }

    fn encode_variant_fields(&mut self,
                             variant: ty::VariantDef) {
        self.start_tag(tag_item_fields);
        self.seq(&variant.fields, |_, f| f.did);
        self.end_tag();
    }
}

impl<'a, 'b, 'tcx> IndexBuilder<'a, 'b, 'tcx> {
    fn encode_enum_variant_infos(&mut self, enum_did: DefId) {
        debug!("encode_enum_variant_info(enum_did={:?})", enum_did);
        let def = self.tcx.lookup_adt_def(enum_did);
        self.encode_fields(enum_did);
        for (i, variant) in def.variants.iter().enumerate() {
            self.record(variant.did,
                        EncodeContext::encode_enum_variant_info,
                        (enum_did, Untracked(i)));
        }
    }
}

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
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
        encode_family(self, Family::Variant(variant.kind));
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

        self.encode_variant_fields(variant);
        self.encode_disr_val(variant.disr_val);
        self.encode_bounds_and_type_for_item(vid);
    }
}

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
    fn encode_info_for_mod(&mut self,
                           FromId(id, (md, attrs, name, vis)):
                           FromId<(&hir::Mod, &[ast::Attribute], Name, &hir::Visibility)>) {
        let tcx = self.tcx;

        encode_def_id_and_key(self, tcx.map.local_def_id(id));
        encode_family(self, Family::Mod);
        encode_name(self, name);
        debug!("(encoding info for module) encoding info for module ID {}", id);

        // Encode info about all the module children.
        self.start_tag(tag_mod_children);
        self.seq(&md.item_ids, |_, item_id| {
            tcx.map.local_def_id(item_id.id)
        });

        // Encode the reexports of this module, if this module is public.
        match self.reexports.get(&id) {
            Some(exports) if *vis == hir::Public => exports.encode(self).unwrap(),
            _ => <[def::Export]>::encode(&[], self).unwrap()
        }
        self.end_tag();

        self.encode_visibility(vis);

        let stab = tcx.lookup_stability(tcx.map.local_def_id(id));
        let depr = tcx.lookup_deprecation(tcx.map.local_def_id(id));
        encode_stability(self, stab);
        encode_deprecation(self, depr);

        encode_attributes(self, attrs);
    }

    fn encode_struct_field_family(&mut self,
                                  visibility: ty::Visibility) {
        encode_family(self, if visibility.is_public() {
            Family::PublicField
        } else {
            Family::InheritedField
        });
    }

    fn encode_visibility<T: HasVisibility>(&mut self, visibility: T) {
        let vis = if visibility.is_public() {
            ty::Visibility::Public
        } else {
            ty::Visibility::PrivateExternal
        };
        self.start_tag(tag_items_data_item_visibility);
        vis.encode(self).unwrap();
        self.end_tag();
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
    constness.encode(ecx).unwrap();
    ecx.end_tag();
}

fn encode_defaultness(ecx: &mut EncodeContext, defaultness: hir::Defaultness) {
    ecx.start_tag(tag_items_data_item_defaultness);
    defaultness.encode(ecx).unwrap();
    ecx.end_tag();
}

impl<'a, 'b, 'tcx> IndexBuilder<'a, 'b, 'tcx> {
    fn encode_fields(&mut self,
                     adt_def_id: DefId) {
        let def = self.tcx.lookup_adt_def(adt_def_id);
        for (variant_index, variant) in def.variants.iter().enumerate() {
            for (field_index, field) in variant.fields.iter().enumerate() {
                self.record(field.did,
                            EncodeContext::encode_field,
                            (adt_def_id, Untracked((variant_index, field_index))));
            }
        }
    }
}

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
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

        let variant_id = tcx.map.as_local_node_id(variant.did).unwrap();
        let variant_data = tcx.map.expect_variant_data(variant_id);
        encode_attributes(self, &variant_data.fields()[field_index].attrs);

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
        encode_family(self, Family::Struct(variant.kind));
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
        self.start_tag(tag_items_data_item_is_tuple_struct_ctor);
        true.encode(self).unwrap();
        self.end_tag();
    }

    fn encode_generics(&mut self,
                       generics: &ty::Generics<'tcx>,
                       predicates: &ty::GenericPredicates<'tcx>)
    {
        self.start_tag(tag_item_generics);
        generics.encode(self).unwrap();
        self.end_tag();
        self.encode_predicates(predicates, tag_item_predicates);
    }

    fn encode_predicates(&mut self,
                         predicates: &ty::GenericPredicates<'tcx>,
                         tag: usize) {
        self.start_tag(tag);
        predicates.parent.encode(self).unwrap();
        self.seq(&predicates.predicates, |ecx, predicate| {
            ecx.add_xref(XRef::Predicate(predicate.clone()))
        });
        self.end_tag();
    }

    fn encode_method_ty_fields(&mut self, method_ty: &ty::Method<'tcx>) {
        encode_def_id_and_key(self, method_ty.def_id);
        encode_name(self, method_ty.name);
        self.encode_generics(&method_ty.generics, &method_ty.predicates);
        self.encode_visibility(method_ty.vis);

        self.start_tag(tag_item_trait_method_explicit_self);
        method_ty.explicit_self.encode(self).unwrap();
        self.end_tag();

        encode_family(self, Family::Method);
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

        match tcx.impl_or_trait_item(item_def_id) {
            ty::ConstTraitItem(associated_const) => {
                encode_name(self, associated_const.name);
                encode_def_id_and_key(self, item_def_id);
                self.encode_visibility(associated_const.vis);

                encode_family(self, Family::AssociatedConst);
                self.encode_bounds_and_type_for_item(item_def_id);
            }
            ty::MethodTraitItem(method_ty) => {
                self.encode_method_ty_fields(&method_ty);
                self.encode_bounds_and_type_for_item(item_def_id);
            }
            ty::TypeTraitItem(associated_type) => {
                encode_name(self, associated_type.name);
                encode_def_id_and_key(self, item_def_id);
                encode_family(self, Family::AssociatedType);

                if let Some(ty) = associated_type.ty {
                    self.encode_type(ty);
                }
            }
        }

        encode_attributes(self, &trait_item.attrs);
        match trait_item.node {
            hir::ConstTraitItem(_, ref default) => {
                self.start_tag(tag_item_trait_item_has_body);
                default.is_some().encode(self).unwrap();
                self.end_tag();

                encode_inlined_item(self,
                                    InlinedItemRef::TraitItem(trait_def_id, trait_item));
                self.encode_mir(item_def_id);
            }
            hir::MethodTraitItem(ref sig, ref body) => {
                self.start_tag(tag_item_trait_item_has_body);
                body.is_some().encode(self).unwrap();
                self.end_tag();

                self.encode_mir(item_def_id);
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
        encode_family(self, Family::AssociatedConst);

        self.encode_parent_item(tcx.map.local_def_id(parent_id));

        self.start_tag(tag_item_trait_item_has_body);
        true.encode(self).unwrap();
        self.end_tag();

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
                              parent_id: NodeId,
                              impl_item_opt: Option<&hir::ImplItem>) {
        let tcx = self.tcx;

        debug!("encode_info_for_method: {:?} {:?}", m.def_id,
               m.name);
        self.encode_method_ty_fields(m);
        self.encode_parent_item(tcx.map.local_def_id(parent_id));

        self.start_tag(tag_item_trait_item_has_body);
        true.encode(self).unwrap();
        self.end_tag();

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
                let needs_inline = types > 0 || attr::requests_inline(&impl_item.attrs);
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
        encode_family(self, Family::AssociatedType);
        self.encode_parent_item(tcx.map.local_def_id(parent_id));

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

        self.seq(&decl.inputs, |_, arg| {
            if let PatKind::Binding(_, ref path1, _) = arg.pat.node {
                path1.node
            } else {
                syntax::parse::token::intern("")
            }
        });

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
        repr_attrs.encode(self);
        self.end_tag();
    }

    fn encode_mir(&mut self, def_id: DefId) {
        if let Some(mir) = self.mir_map.map.get(&def_id) {
            self.start_tag(tag_mir as usize);
            mir.encode(self);
            self.end_tag();
        }
    }
}

// Encodes the inherent implementations of a structure, enumeration, or trait.
fn encode_inherent_implementations(ecx: &mut EncodeContext,
                                   def_id: DefId) {
    ecx.start_tag(tag_items_data_item_inherent_impls);
    match ecx.tcx.inherent_impls.borrow().get(&def_id) {
        None => <[DefId]>::encode(&[], ecx).unwrap(),
        Some(implementations) => implementations.encode(ecx).unwrap()
    }
    ecx.end_tag();
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

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
    fn add_xref(&mut self, xref: XRef<'tcx>) -> u32 {
        let old_len = self.xrefs.len() as u32;
        *self.xrefs.entry(xref).or_insert(old_len)
    }

    fn encode_xrefs(&mut self) {
        let xrefs = mem::replace(&mut self.xrefs, Default::default());
        let mut xref_positions = vec![0; xrefs.len()];

        // Encode XRefs sorted by their ID
        let mut sorted_xrefs: Vec<_> = xrefs.into_iter().collect();
        sorted_xrefs.sort_by_key(|&(_, id)| id);

        self.start_tag(tag_xref_data);
        for (xref, id) in sorted_xrefs.into_iter() {
            xref_positions[id as usize] = self.mark_stable_position() as u32;
            match xref {
                XRef::Predicate(p) => p.encode(self).unwrap()
            }
        }
        self.mark_stable_position();
        self.end_tag();

        self.start_tag(tag_xref_index);
        index::write_dense_index(xref_positions, &mut self.opaque.cursor);
        self.end_tag();
    }

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
                encode_family(self, if m == hir::MutMutable {
                    Family::MutStatic
                } else {
                    Family::ImmStatic
                });
                self.encode_bounds_and_type_for_item(def_id);
                encode_name(self, item.name);
                self.encode_visibility(vis);
                encode_stability(self, stab);
                encode_deprecation(self, depr);
                encode_attributes(self, &item.attrs);
            }
            hir::ItemConst(..) => {
                encode_def_id_and_key(self, def_id);
                encode_family(self, Family::Const);
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
                encode_family(self, Family::Fn);
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
                encode_family(self, Family::ForeignMod);
                encode_name(self, item.name);

                // Encode all the items in self module.
                self.start_tag(tag_mod_children);
                self.seq(&fm.items, |_, foreign_item| {
                    tcx.map.local_def_id(foreign_item.id)
                });
                <[def::Export]>::encode(&[], self).unwrap();
                self.end_tag();

                self.encode_visibility(vis);
                encode_stability(self, stab);
                encode_deprecation(self, depr);
            }
            hir::ItemTy(..) => {
                encode_def_id_and_key(self, def_id);
                encode_family(self, Family::Type);
                self.encode_bounds_and_type_for_item(def_id);
                encode_name(self, item.name);
                self.encode_visibility(vis);
                encode_stability(self, stab);
                encode_deprecation(self, depr);
            }
            hir::ItemEnum(ref enum_definition, _) => {
                encode_def_id_and_key(self, def_id);
                encode_family(self, Family::Enum);
                encode_item_variances(self, item.id);
                self.encode_bounds_and_type_for_item(def_id);
                encode_name(self, item.name);
                encode_attributes(self, &item.attrs);
                self.encode_repr_attrs(&item.attrs);

                self.start_tag(tag_mod_children);
                self.seq(&enum_definition.variants, |_, v| {
                    tcx.map.local_def_id(v.node.data.id())
                });
                <[def::Export]>::encode(&[], self).unwrap();
                self.end_tag();

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
                encode_family(self, Family::Struct(variant.kind));
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
                self.encode_variant_fields(variant);

                // Encode inherent implementations for self structure.
                encode_inherent_implementations(self, def_id);

                if !struct_def.is_struct() {
                    let ctor_did = tcx.map.local_def_id(struct_def.id());
                    self.start_tag(tag_items_data_item_struct_ctor);
                    ctor_did.encode(self).unwrap();
                    self.end_tag();
                }
            }
            hir::ItemUnion(..) => {
                let def = self.tcx.lookup_adt_def(def_id);
                let variant = def.struct_variant();

                encode_def_id_and_key(self, def_id);
                encode_family(self, Family::Union);
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
                self.encode_variant_fields(variant);

                encode_inlined_item(self, InlinedItemRef::Item(def_id, item));
                self.encode_mir(def_id);

                // Encode inherent implementations for self union.
                encode_inherent_implementations(self, def_id);
            }
            hir::ItemDefaultImpl(..) => {
                encode_def_id_and_key(self, def_id);
                encode_family(self, Family::DefaultImpl);
                encode_name(self, item.name);

                let trait_ref = tcx.impl_trait_ref(tcx.map.local_def_id(item.id)).unwrap();
                encode_trait_ref(self, trait_ref, tag_item_trait_ref);
            }
            hir::ItemImpl(_, polarity, ..) => {
                encode_def_id_and_key(self, def_id);
                encode_family(self, Family::Impl);
                self.encode_bounds_and_type_for_item(def_id);
                encode_name(self, item.name);
                encode_attributes(self, &item.attrs);

                self.start_tag(tag_polarity);
                polarity.encode(self).unwrap();
                self.end_tag();

                match
                    tcx.custom_coerce_unsized_kinds
                       .borrow()
                       .get(&tcx.map.local_def_id(item.id))
                {
                    Some(&kind) => {
                        self.start_tag(tag_impl_coerce_unsized_kind);
                        kind.encode(self);
                        self.end_tag();
                    }
                    None => {}
                }

                self.start_tag(tag_mod_children);
                tcx.impl_or_trait_items(def_id).encode(self).unwrap();
                <[def::Export]>::encode(&[], self).unwrap();
                self.end_tag();

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
                    parent.map(|parent| {
                        self.start_tag(tag_items_data_parent_impl);
                        parent.encode(self).unwrap();
                        self.end_tag();
                    });
                }
                encode_stability(self, stab);
                encode_deprecation(self, depr);
            }
            hir::ItemTrait(..) => {
                encode_def_id_and_key(self, def_id);
                encode_family(self, Family::Trait);
                encode_item_variances(self, item.id);
                let trait_def = tcx.lookup_trait_def(def_id);
                let trait_predicates = tcx.lookup_predicates(def_id);

                self.start_tag(tag_unsafety);
                trait_def.unsafety.encode(self).unwrap();
                self.end_tag();

                self.start_tag(tag_paren_sugar);
                trait_def.paren_sugar.encode(self).unwrap();
                self.end_tag();

                self.start_tag(tag_defaulted_trait);
                tcx.trait_has_default_impl(def_id).encode(self).unwrap();
                self.end_tag();

                self.encode_generics(&trait_def.generics, &trait_predicates);
                self.encode_predicates(&tcx.lookup_super_predicates(def_id),
                                       tag_item_super_predicates);
                encode_trait_ref(self, trait_def.trait_ref, tag_item_trait_ref);
                encode_name(self, item.name);
                encode_attributes(self, &item.attrs);
                self.encode_visibility(vis);
                encode_stability(self, stab);
                encode_deprecation(self, depr);

                self.start_tag(tag_mod_children);
                tcx.impl_or_trait_items(def_id).encode(self).unwrap();
                <[def::Export]>::encode(&[], self).unwrap();
                self.end_tag();

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
                            EncodeContext::encode_struct_ctor,
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
        let items = self.tcx.impl_or_trait_items(def_id);

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

            self.record(trait_item_def_id,
                        EncodeContext::encode_info_for_impl_item,
                        (impl_id, trait_item_def_id, ast_item));
        }
    }

    fn encode_addl_trait_info(&mut self,
                              def_id: DefId,
                              trait_items: &[hir::TraitItem]) {
        // Now output the trait item info for each trait item.
        let r = self.tcx.impl_or_trait_items(def_id);
        for (&item_def_id, trait_item) in r.iter().zip(trait_items) {
            assert!(item_def_id.is_local());
            self.record(item_def_id,
                        EncodeContext::encode_info_for_trait_item,
                        (def_id, item_def_id, trait_item));
        }
    }
}

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
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
                encode_family(self, Family::Fn);
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
                encode_family(self, if mutbl {
                    Family::MutStatic
                } else {
                    Family::ImmStatic
                });
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
                                   EncodeContext::encode_info_for_item,
                                   (def_id, item)),
        }
        self.index.encode_addl_info_for_item(item);
    }
    fn visit_foreign_item(&mut self, ni: &'tcx hir::ForeignItem) {
        intravisit::walk_foreign_item(self, ni);
        let def_id = self.index.tcx.map.local_def_id(ni.id);
        self.index.record(def_id,
                          EncodeContext::encode_info_for_foreign_item,
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
                        EncodeContext::encode_info_for_anon_ty,
                        def_id);
        }
    }

    fn encode_info_for_expr(&mut self, expr: &hir::Expr) {
        match expr.node {
            hir::ExprClosure(..) => {
                let def_id = self.tcx.map.local_def_id(expr.id);
                self.record(def_id,
                            EncodeContext::encode_info_for_closure,
                            def_id);
            }
            _ => { }
        }
    }
}

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
    fn encode_info_for_anon_ty(&mut self, def_id: DefId) {
        encode_def_id_and_key(self, def_id);
        self.encode_bounds_and_type_for_item(def_id);
    }

    fn encode_info_for_closure(&mut self, def_id: DefId) {
        let tcx = self.tcx;
        encode_def_id_and_key(self, def_id);
        encode_name(self, syntax::parse::token::intern("<closure>"));

        self.start_tag(tag_items_closure_ty);
        tcx.tables.borrow().closure_tys[&def_id].encode(self).unwrap();
        self.end_tag();

        self.start_tag(tag_items_closure_kind);
        tcx.closure_kind(def_id).encode(self).unwrap();
        self.end_tag();

        assert!(self.mir_map.map.contains_key(&def_id));
        self.encode_mir(def_id);
    }
}

fn encode_info_for_items(ecx: &mut EncodeContext) -> IndexData {
    let krate = ecx.tcx.map.krate();

    ecx.start_tag(tag_items_data);

    let items = {
        let mut index = IndexBuilder::new(ecx);
        index.record(DefId::local(CRATE_DEF_INDEX),
                     EncodeContext::encode_info_for_mod,
                     FromId(CRATE_NODE_ID, (&krate.module,
                                            &krate.attrs,
                                            syntax::parse::token::intern(&ecx.link_meta.crate_name),
                                            &hir::Public)));
        let mut visitor = EncodeVisitor {
            index: index,
        };
        krate.visit_all_items(&mut visitor);
        visitor.index.into_items()
    };

    ecx.end_tag();

    items
}

fn encode_item_index(ecx: &mut EncodeContext, index: IndexData) {
    ecx.start_tag(tag_index);
    index.write_index(&mut ecx.opaque.cursor);
    ecx.end_tag();
}

fn encode_attributes(ecx: &mut EncodeContext, attrs: &[ast::Attribute]) {
    ecx.start_tag(tag_attributes);
    attrs.encode(ecx).unwrap();
    ecx.end_tag();
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
            assert_eq!(n, CrateNum::new(expected_cnum));
            expected_cnum += 1;
        }

        deps
    }

    // We're just going to write a list of crate 'name-hash-version's, with
    // the assumption that they are numbered 1 to n.
    // FIXME (#2166): This is not nearly enough to support correct versioning
    // but is enough to get transitive crate dependencies working.
    ecx.start_tag(tag_crate_deps);
    ecx.seq(&get_ordered_deps(cstore), |_, &(_, ref dep)| {
        (dep.name(), decoder::get_crate_hash(dep.data()),
         dep.explicitly_linked.get())
    });
    ecx.end_tag();
}

fn encode_lang_items(ecx: &mut EncodeContext) {
    let tcx = ecx.tcx;
    let lang_items = || {
        tcx.lang_items.items().iter().enumerate().filter_map(|(i, &opt_def_id)| {
            if let Some(def_id) = opt_def_id {
                if def_id.is_local() {
                    return Some((def_id.index, i));
                }
            }
            None
        })
    };

    let count = lang_items().count();
    let mut lang_items = lang_items();

    ecx.start_tag(tag_lang_items);
    ecx.seq(0..count, |_, _| lang_items.next().unwrap());
    ecx.end_tag();

    ecx.start_tag(tag_lang_items_missing);
    tcx.lang_items.missing.encode(ecx).unwrap();
    ecx.end_tag();
}

fn encode_native_libraries(ecx: &mut EncodeContext) {
    let used_libraries = ecx.tcx.sess.cstore.used_libraries();
    let libs = || {
        used_libraries.iter().filter_map(|&(ref lib, kind)| {
            match kind {
                cstore::NativeStatic => None, // these libraries are not propagated
                cstore::NativeFramework | cstore::NativeUnknown => {
                    Some((kind, lib))
                }
            }
        })
    };

    let count = libs().count();
    let mut libs = libs();

    ecx.start_tag(tag_native_libraries);
    ecx.seq(0..count, |_, _| libs.next().unwrap());
    ecx.end_tag();
}

fn encode_plugin_registrar_fn(ecx: &mut EncodeContext) {
    match ecx.tcx.sess.plugin_registrar_fn.get() {
        Some(id) => {
            let def_id = ecx.tcx.map.local_def_id(id);
            ecx.start_tag(tag_plugin_registrar_fn);
            def_id.index.encode(ecx).unwrap();
            ecx.end_tag();
        }
        None => {}
    }
}

fn encode_codemap(ecx: &mut EncodeContext) {
    let codemap = ecx.tcx.sess.codemap();
    let all_filemaps = codemap.files.borrow();
    let filemaps = || {
        // No need to export empty filemaps, as they can't contain spans
        // that need translation.
        // Also no need to re-export imported filemaps, as any downstream
        // crate will import them from their original source.
        all_filemaps.iter().filter(|filemap| {
            !filemap.lines.borrow().is_empty() && !filemap.is_imported()
        })
    };

    let count = filemaps().count();
    let mut filemaps = filemaps();

    ecx.start_tag(tag_codemap);
    ecx.seq(0..count, |_, _| filemaps.next().unwrap());
    ecx.end_tag();
}

/// Serialize the text of the exported macros
fn encode_macro_defs(ecx: &mut EncodeContext) {
    let tcx = ecx.tcx;
    ecx.start_tag(tag_macro_defs);
    ecx.seq(&tcx.map.krate().exported_macros, |_, def| {
        let body = ::syntax::print::pprust::tts_to_string(&def.body);
        (def.name, &def.attrs, def.span, body)
    });
    ecx.end_tag();

    if ecx.tcx.sess.crate_types.borrow().contains(&CrateTypeRustcMacro) {
        let id = ecx.tcx.sess.derive_registrar_fn.get().unwrap();
        let did = ecx.tcx.map.local_def_id(id);

        ecx.start_tag(tag_macro_derive_registrar);
        did.index.encode(ecx).unwrap();
        ecx.end_tag();
    }
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
fn encode_impls(ecx: &mut EncodeContext) {
    let mut visitor = ImplVisitor {
        tcx: ecx.tcx,
        impls: FnvHashMap()
    };
    ecx.tcx.map.krate().visit_all_items(&mut visitor);

    ecx.start_tag(tag_impls);
    for (trait_def_id, trait_impls) in visitor.impls {
        // FIXME(eddyb) Avoid wrapping the entries in docs.
        ecx.start_tag(0);
        (trait_def_id.krate.as_u32(), trait_def_id.index).encode(ecx).unwrap();
        trait_impls.encode(ecx).unwrap();
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

    let reachable = ecx.reachable;
    ecx.seq(reachable, |ecx, &id| ecx.tcx.map.local_def_id(id).index);

    ecx.end_tag();
}

fn encode_dylib_dependency_formats(ecx: &mut EncodeContext) {
    ecx.start_tag(tag_dylib_dependency_formats);
    match ecx.tcx.sess.dependency_formats.borrow().get(&config::CrateTypeDylib) {
        Some(arr) => {
            ecx.seq(arr, |_, slot| {
                match *slot {
                    Linkage::NotLinked |
                    Linkage::IncludedFromDylib => None,

                    Linkage::Dynamic => Some(LinkagePreference::RequireDynamic),
                    Linkage::Static => Some(LinkagePreference::RequireStatic),
                }
            });
        }
        None => {
            <[Option<LinkagePreference>]>::encode(&[], ecx).unwrap();
        }
    }
    ecx.end_tag();
}

pub fn encode_metadata<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                 cstore: &cstore::CStore,
                                 reexports: &def::ExportMap,
                                 link_meta: &LinkMeta,
                                 reachable: &NodeSet,
                                 mir_map: &MirMap<'tcx>) -> Vec<u8> {
    let mut cursor = Cursor::new(vec![]);
    cursor.write_all(&[0, 0, 0, 0]).unwrap();
    cursor.write_all(metadata_encoding_version).unwrap();
    // Will be filed with the length after encoding the crate.
    cursor.write_all(&[0, 0, 0, 0]).unwrap();

    encode_metadata_inner(&mut EncodeContext {
        rbml_w: rbml::writer::Encoder::new(&mut cursor),
        tcx: tcx,
        reexports: reexports,
        link_meta: link_meta,
        cstore: cstore,
        reachable: reachable,
        mir_map: mir_map,
        type_shorthands: Default::default(),
        xrefs: Default::default()
    });

    // RBML compacts the encoded bytes whenever appropriate,
    // so there are some garbages left after the end of the data.
    let meta_len = cursor.position() as usize;
    cursor.get_mut().truncate(meta_len);

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

    let meta_start = 8 + ::common::metadata_encoding_version.len();
    let len = meta_len - meta_start;
    let mut result = cursor.into_inner();
    result[meta_start - 4] = (len >> 24) as u8;
    result[meta_start - 3] = (len >> 16) as u8;
    result[meta_start - 2] = (len >>  8) as u8;
    result[meta_start - 1] = (len >>  0) as u8;
    result
}

fn encode_metadata_inner(ecx: &mut EncodeContext) {
    ecx.wr_tagged_str(tag_rustc_version, &rustc_version());

    let tcx = ecx.tcx;
    let link_meta = ecx.link_meta;

    ecx.start_tag(tag_crate_crate_name);
    link_meta.crate_name.encode(ecx).unwrap();
    ecx.end_tag();

    ecx.start_tag(tag_crate_triple);
    tcx.sess.opts.target_triple.encode(ecx).unwrap();
    ecx.end_tag();

    ecx.start_tag(tag_crate_hash);
    link_meta.crate_hash.encode(ecx).unwrap();
    ecx.end_tag();

    ecx.start_tag(tag_crate_disambiguator);
    tcx.sess.local_crate_disambiguator().encode(ecx).unwrap();
    ecx.end_tag();

    encode_dylib_dependency_formats(ecx);

    ecx.start_tag(tag_panic_strategy);
    ecx.tcx.sess.opts.cg.panic.encode(ecx);
    ecx.end_tag();

    let mut i = ecx.position();
    encode_crate_deps(ecx, ecx.cstore);
    let dep_bytes = ecx.position() - i;

    // Encode the language items.
    i = ecx.position();
    encode_lang_items(ecx);
    let lang_item_bytes = ecx.position() - i;

    // Encode the native libraries used
    i = ecx.position();
    encode_native_libraries(ecx);
    let native_lib_bytes = ecx.position() - i;

    // Encode the plugin registrar function
    i = ecx.position();
    encode_plugin_registrar_fn(ecx);
    let plugin_registrar_fn_bytes = ecx.position() - i;

    // Encode codemap
    i = ecx.position();
    encode_codemap(ecx);
    let codemap_bytes = ecx.position() - i;

    // Encode macro definitions
    i = ecx.position();
    encode_macro_defs(ecx);
    let macro_defs_bytes = ecx.position() - i;

    // Encode the def IDs of impls, for coherence checking.
    i = ecx.position();
    encode_impls(ecx);
    let impl_bytes = ecx.position() - i;

    // Encode reachability info.
    i = ecx.position();
    encode_reachable(ecx);
    let reachable_bytes = ecx.position() - i;

    // Encode and index the items.
    ecx.start_tag(tag_items);
    i = ecx.position();
    let items = encode_info_for_items(ecx);
    let item_bytes = ecx.position() - i;
    ecx.end_tag();

    i = ecx.position();
    encode_item_index(ecx, items);
    let index_bytes = ecx.position() - i;

    i = ecx.position();
    ecx.encode_xrefs();
    let xref_bytes = ecx.position() - i;

    let total_bytes = ecx.position();

    if ecx.tcx.sess.meta_stats() {
        let mut zero_bytes = 0;
        for e in ecx.opaque.cursor.get_ref() {
            if *e == 0 {
                zero_bytes += 1;
            }
        }

        println!("metadata stats:");
        println!("             dep bytes: {}", dep_bytes);
        println!("       lang item bytes: {}", lang_item_bytes);
        println!("          native bytes: {}", native_lib_bytes);
        println!("plugin registrar bytes: {}", plugin_registrar_fn_bytes);
        println!("         codemap bytes: {}", codemap_bytes);
        println!("       macro def bytes: {}", macro_defs_bytes);
        println!("            impl bytes: {}", impl_bytes);
        println!("       reachable bytes: {}", reachable_bytes);
        println!("            item bytes: {}", item_bytes);
        println!("           index bytes: {}", index_bytes);
        println!("            xref bytes: {}", xref_bytes);
        println!("            zero bytes: {}", zero_bytes);
        println!("           total bytes: {}", total_bytes);
    }
}
