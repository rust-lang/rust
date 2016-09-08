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
use index::IndexData;

use rustc::middle::cstore::{InlinedItemRef, LinkMeta, LinkagePreference};
use rustc::hir::def;
use rustc::hir::def_id::{CrateNum, CRATE_DEF_INDEX, DefIndex, DefId};
use rustc::middle::dependency_format::Linkage;
use rustc::traits::specialization_graph;
use rustc::ty::{self, Ty, TyCtxt};

use rustc::mir::mir_map::MirMap;
use rustc::session::config::{self, CrateTypeRustcMacro};
use rustc::util::nodemap::{FnvHashMap, NodeSet};

use rustc_serialize::{Encodable, Encoder, SpecializedEncoder, opaque};
use std::hash::Hash;
use std::intrinsics;
use std::io::prelude::*;
use std::io::Cursor;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::u32;
use syntax::ast::{self, CRATE_NODE_ID};
use syntax::attr;
use syntax;
use rbml;

use rustc::hir::{self, PatKind};
use rustc::hir::intravisit::Visitor;
use rustc::hir::intravisit;

use super::index_builder::{FromId, IndexBuilder, Untracked};

pub struct EncodeContext<'a, 'tcx: 'a> {
    rbml_w: rbml::writer::Encoder<'a>,
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
    reexports: &'a def::ExportMap,
    link_meta: &'a LinkMeta,
    cstore: &'a cstore::CStore,
    reachable: &'a NodeSet,
    mir_map: &'a MirMap<'tcx>,

    type_shorthands: FnvHashMap<Ty<'tcx>, usize>,
    predicate_shorthands: FnvHashMap<ty::Predicate<'tcx>, usize>,
}

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
        self.encode_with_shorthand(ty, &ty.sty, |ecx| &mut ecx.type_shorthands)
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

    /// Encode the given value or a previously cached shorthand.
    fn encode_with_shorthand<T, U, M>(&mut self, value: &T, variant: &U, map: M)
                                      -> Result<(), <Self as Encoder>::Error>
    where M: for<'b> Fn(&'b mut Self) -> &'b mut FnvHashMap<T, usize>,
          T: Clone + Eq + Hash,
          U: Encodable {
        let existing_shorthand = map(self).get(value).cloned();
        if let Some(shorthand) = existing_shorthand {
            return self.emit_usize(shorthand);
        }

        let start = self.mark_stable_position();
        variant.encode(self)?;
        let len = self.mark_stable_position() - start;

        // The shorthand encoding uses the same usize as the
        // discriminant, with an offset so they can't conflict.
        let discriminant = unsafe {
            intrinsics::discriminant_value(variant)
        };
        assert!(discriminant < SHORTHAND_OFFSET as u64);
        let shorthand = start + SHORTHAND_OFFSET;

        // Get the number of bits that leb128 could fit
        // in the same space as the fully encoded type.
        let leb128_bits = len * 7;

        // Check that the shorthand is a not longer than the
        // full encoding itself, i.e. it's an obvious win.
        if leb128_bits >= 64 || (shorthand as u64) < (1 << leb128_bits) {
            map(self).insert(value.clone(), shorthand);
        }

        Ok(())
    }

    /// For every DefId that we create a metadata item for, we include a
    /// serialized copy of its DefKey, which allows us to recreate a path.
    fn encode_def_key(&mut self, def_id: DefId) {
        self.start_tag(item_tag::def_key);
        self.tcx.map.def_key(def_id).encode(self);
        self.end_tag();
    }

    // Item info table encoding
    fn encode_family(&mut self, f: Family) {
        self.start_tag(item_tag::family);
        f.encode(self).unwrap();
        self.end_tag();
    }

    fn encode_item_variances(&mut self, def_id: DefId) {
        let v = self.tcx.item_variances(def_id);
        self.start_tag(item_tag::variances);
        v.encode(self);
        self.end_tag();
    }

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
        self.start_tag(item_tag::ty);
        typ.encode(self).unwrap();
        self.end_tag();
    }

    fn encode_variant(&mut self, variant: ty::VariantDef,
                      struct_ctor: Option<DefIndex>)
                      -> EntryData {
        self.start_tag(item_tag::children);
        self.seq(&variant.fields, |_, f| {
            assert!(f.did.is_local());
            f.did.index
        });
        self.end_tag();

        EntryData::Variant(VariantData {
            kind: variant.kind,
            disr: variant.disr_val.to_u64_unchecked(),
            struct_ctor: struct_ctor
        })
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
        self.encode_def_key(vid);
        self.encode_family(Family::Variant);

        let enum_id = tcx.map.as_local_node_id(enum_did).unwrap();
        let enum_vis = &tcx.map.expect_item(enum_id).vis;
        self.encode_visibility(enum_vis);

        let attrs = tcx.get_attrs(vid);
        encode_attributes(self, &attrs);
        encode_stability(self, vid);

        let data = self.encode_variant(variant, None);

        self.start_tag(item_tag::data);
        data.encode(self).unwrap();
        self.end_tag();

        self.start_tag(item_tag::typed_data);
        EntryTypedData::Other.encode(self).unwrap();
        self.end_tag();

        self.encode_bounds_and_type_for_item(vid);
    }
}

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
    fn encode_info_for_mod(&mut self,
                           FromId(id, (md, attrs, vis)):
                           FromId<(&hir::Mod, &[ast::Attribute], &hir::Visibility)>) {
        let tcx = self.tcx;

        let def_id = tcx.map.local_def_id(id);
        self.encode_def_key(def_id);
        self.encode_family(Family::Mod);
        self.encode_visibility(vis);
        encode_stability(self, def_id);
        encode_attributes(self, attrs);
        debug!("(encoding info for module) encoding info for module ID {}", id);

        // Encode info about all the module children.
        self.start_tag(item_tag::children);
        self.seq(&md.item_ids, |_, item_id| {
            tcx.map.local_def_id(item_id.id).index
        });
        self.end_tag();

        // Encode the reexports of this module, if this module is public.
        let reexports = match self.reexports.get(&id) {
            Some(exports) if *vis == hir::Public => exports.clone(),
            _ => vec![]
        };

        self.start_tag(item_tag::data);
        EntryData::Mod(ModData {
            reexports: reexports
        }).encode(self).unwrap();
        self.end_tag();

        self.start_tag(item_tag::typed_data);
        EntryTypedData::Other.encode(self).unwrap();
        self.end_tag();
    }

    fn encode_visibility<T: HasVisibility>(&mut self, visibility: T) {
        let vis = if visibility.is_public() {
            ty::Visibility::Public
        } else {
            ty::Visibility::PrivateExternal
        };
        self.start_tag(item_tag::visibility);
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

        self.encode_family(Family::Field);
        self.encode_visibility(field.vis);
        self.encode_bounds_and_type_for_item(field.did);
        self.encode_def_key(field.did);

        let variant_id = tcx.map.as_local_node_id(variant.did).unwrap();
        let variant_data = tcx.map.expect_variant_data(variant_id);
        encode_attributes(self, &variant_data.fields()[field_index].attrs);
        encode_stability(self, field.did);
    }

    fn encode_struct_ctor(&mut self, ctor_def_id: DefId) {
        self.encode_def_key(ctor_def_id);
        self.encode_family(Family::Struct);
        self.encode_bounds_and_type_for_item(ctor_def_id);

        encode_stability(self, ctor_def_id);
    }

    fn encode_generics(&mut self,
                       generics: &ty::Generics<'tcx>,
                       predicates: &ty::GenericPredicates<'tcx>)
    {
        self.start_tag(item_tag::generics);
        generics.encode(self).unwrap();
        self.end_tag();
        self.encode_predicates(predicates, item_tag::predicates);
    }

    fn encode_predicates(&mut self,
                         predicates: &ty::GenericPredicates<'tcx>,
                         tag: usize) {
        self.start_tag(tag);
        predicates.parent.encode(self).unwrap();
        self.seq(&predicates.predicates, |ecx, predicate| {
            ecx.encode_with_shorthand(predicate, predicate,
                                      |ecx| &mut ecx.predicate_shorthands).unwrap()
        });
        self.end_tag();
    }

    fn encode_info_for_trait_item(&mut self, def_id: DefId) {
        let tcx = self.tcx;

        let node_id = tcx.map.as_local_node_id(def_id).unwrap();
        let ast_item = tcx.map.expect_trait_item(node_id);
        let trait_item = tcx.impl_or_trait_item(def_id);
        let (family, has_default, typed_data) = match trait_item {
            ty::ConstTraitItem(ref associated_const) => {
                self.encode_bounds_and_type_for_item(def_id);

                let trait_def_id = trait_item.container().id();
                encode_inlined_item(self,
                                    InlinedItemRef::TraitItem(trait_def_id, ast_item));

                (Family::AssociatedConst,
                associated_const.has_value,
                 EntryTypedData::Other)
            }
            ty::MethodTraitItem(ref method_ty) => {
                self.encode_bounds_and_type_for_item(def_id);

                (Family::Method,
                 method_ty.has_body,
                 EntryTypedData::Method(MethodTypedData {
                    explicit_self: method_ty.explicit_self
                 }))
            }
            ty::TypeTraitItem(ref associated_type) => {
                if let Some(ty) = associated_type.ty {
                    self.encode_type(ty);
                }

                (Family::AssociatedType, false, EntryTypedData::Other)
            }
        };

        self.encode_def_key(def_id);
        self.encode_family(family);
        self.encode_visibility(trait_item.vis());

        encode_stability(self, def_id);
        encode_attributes(self, &ast_item.attrs);
        if let hir::MethodTraitItem(ref sig, _) = ast_item.node {
            self.encode_fn_arg_names(&sig.decl);
        };

        self.start_tag(item_tag::data);
        EntryData::TraitAssociated(TraitAssociatedData {
            has_default: has_default
        }).encode(self).unwrap();
        self.end_tag();

        self.start_tag(item_tag::typed_data);
        typed_data.encode(self).unwrap();
        self.end_tag();

        self.encode_mir(def_id);
    }

    fn encode_info_for_impl_item(&mut self, def_id: DefId) {
        let node_id = self.tcx.map.as_local_node_id(def_id).unwrap();
        let ast_item = self.tcx.map.expect_impl_item(node_id);
        let impl_item = self.tcx.impl_or_trait_item(def_id);
        let impl_def_id = impl_item.container().id();
        let (family, typed_data) = match impl_item {
            ty::ConstTraitItem(_) => {
                self.encode_bounds_and_type_for_item(def_id);

                encode_inlined_item(self,
                                    InlinedItemRef::ImplItem(impl_def_id, ast_item));
                self.encode_mir(def_id);

                (Family::AssociatedConst, EntryTypedData::Other)
            }
            ty::MethodTraitItem(ref method_type) => {
                self.encode_bounds_and_type_for_item(def_id);

                (Family::Method,
                 EntryTypedData::Method(MethodTypedData {
                    explicit_self: method_type.explicit_self
                 }))
            }
            ty::TypeTraitItem(ref associated_type) => {
                if let Some(ty) = associated_type.ty {
                    self.encode_type(ty);
                }

                (Family::AssociatedType, EntryTypedData::Other)
            }
        };

        self.encode_def_key(def_id);
        self.encode_family(family);
        self.encode_visibility(impl_item.vis());
        encode_attributes(self, &ast_item.attrs);
        encode_stability(self, def_id);

        let constness = if let hir::ImplItemKind::Method(ref sig, _) = ast_item.node {
            if sig.constness == hir::Constness::Const {
                encode_inlined_item(
                    self,
                    InlinedItemRef::ImplItem(impl_def_id, ast_item));
            }

            let generics = self.tcx.lookup_generics(def_id);
            let types = generics.parent_types as usize + generics.types.len();
            let needs_inline = types > 0 || attr::requests_inline(&ast_item.attrs);
            if needs_inline || sig.constness == hir::Constness::Const {
                self.encode_mir(def_id);
            }
            self.encode_fn_arg_names(&sig.decl);
            sig.constness
        } else {
            hir::Constness::NotConst
        };

        self.start_tag(item_tag::data);
        EntryData::ImplAssociated(ImplAssociatedData {
            defaultness: ast_item.defaultness,
            constness:constness
        }).encode(self).unwrap();
        self.end_tag();

        self.start_tag(item_tag::typed_data);
        typed_data.encode(self).unwrap();
        self.end_tag();
    }

    fn encode_fn_arg_names(&mut self,
                                    decl: &hir::FnDecl) {
        self.start_tag(item_tag::fn_arg_names);

        self.seq(&decl.inputs, |_, arg| {
            if let PatKind::Binding(_, ref path1, _) = arg.pat.node {
                path1.node
            } else {
                syntax::parse::token::intern("")
            }
        });

        self.end_tag();
    }

    fn encode_mir(&mut self, def_id: DefId) {
        if let Some(mir) = self.mir_map.map.get(&def_id) {
            self.start_tag(item_tag::mir as usize);
            mir.encode(self);
            self.end_tag();
        }
    }
}

// Encodes the inherent implementations of a structure, enumeration, or trait.
fn encode_inherent_implementations(ecx: &mut EncodeContext,
                                   def_id: DefId) {
    ecx.start_tag(item_tag::inherent_impls);
    match ecx.tcx.inherent_impls.borrow().get(&def_id) {
        None => <[DefId]>::encode(&[], ecx).unwrap(),
        Some(implementations) => implementations.encode(ecx).unwrap()
    }
    ecx.end_tag();
}

fn encode_stability(ecx: &mut EncodeContext, def_id: DefId) {
    ecx.tcx.lookup_stability(def_id).map(|stab| {
        ecx.start_tag(item_tag::stability);
        stab.encode(ecx).unwrap();
        ecx.end_tag();
    });
    ecx.tcx.lookup_deprecation(def_id).map(|depr| {
        ecx.start_tag(item_tag::deprecation);
        depr.encode(ecx).unwrap();
        ecx.end_tag();
    });
}

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
    fn encode_info_for_item(&mut self,
                            (def_id, item): (DefId, &hir::Item)) {
        let tcx = self.tcx;

        debug!("encoding info for item at {}",
               tcx.sess.codemap().span_to_string(item.span));

        let (family, data, typed_data) = match item.node {
            hir::ItemStatic(_, m, _) => {
                self.encode_bounds_and_type_for_item(def_id);

                if m == hir::MutMutable {
                    (Family::MutStatic, EntryData::Other, EntryTypedData::Other)
                } else {
                    (Family::ImmStatic, EntryData::Other, EntryTypedData::Other)
                }
            }
            hir::ItemConst(..) => {
                self.encode_bounds_and_type_for_item(def_id);
                encode_inlined_item(self, InlinedItemRef::Item(def_id, item));
                self.encode_mir(def_id);

                (Family::Const, EntryData::Other, EntryTypedData::Other)
            }
            hir::ItemFn(ref decl, _, constness, _, ref generics, _) => {
                let tps_len = generics.ty_params.len();
                self.encode_bounds_and_type_for_item(def_id);
                let needs_inline = tps_len > 0 || attr::requests_inline(&item.attrs);
                if constness == hir::Constness::Const {
                    encode_inlined_item(self, InlinedItemRef::Item(def_id, item));
                }
                if needs_inline || constness == hir::Constness::Const {
                    self.encode_mir(def_id);
                }
                self.encode_fn_arg_names(&decl);

                (Family::Fn, EntryData::Fn(FnData {
                    constness: constness
                 }), EntryTypedData::Other)
            }
            hir::ItemMod(ref m) => {
                self.encode_info_for_mod(FromId(item.id, (m, &item.attrs, &item.vis)));
                return;
            }
            hir::ItemForeignMod(ref fm) => {
                // Encode all the items in self module.
                self.start_tag(item_tag::children);
                self.seq(&fm.items, |_, foreign_item| {
                    tcx.map.local_def_id(foreign_item.id).index
                });
                self.end_tag();

                (Family::ForeignMod, EntryData::Other, EntryTypedData::Other)
            }
            hir::ItemTy(..) => {
                self.encode_bounds_and_type_for_item(def_id);

                (Family::Type, EntryData::Other, EntryTypedData::Other)
            }
            hir::ItemEnum(ref enum_definition, _) => {
                self.encode_item_variances(def_id);
                self.encode_bounds_and_type_for_item(def_id);

                self.start_tag(item_tag::children);
                self.seq(&enum_definition.variants, |_, v| {
                    tcx.map.local_def_id(v.node.data.id()).index
                });
                self.end_tag();

                // Encode inherent implementations for self enumeration.
                encode_inherent_implementations(self, def_id);

                (Family::Enum, EntryData::Other, EntryTypedData::Other)
            }
            hir::ItemStruct(ref struct_def, _) => {
                let def = tcx.lookup_adt_def(def_id);
                let variant = def.struct_variant();

                self.encode_bounds_and_type_for_item(def_id);

                self.encode_item_variances(def_id);

                /* Encode def_ids for each field and method
                for methods, write all the stuff get_trait_method
                needs to know*/
                let struct_ctor = if !struct_def.is_struct() {
                    Some(tcx.map.local_def_id(struct_def.id()).index)
                } else {
                    None
                };
                let data = self.encode_variant(variant, struct_ctor);

                // Encode inherent implementations for self structure.
                encode_inherent_implementations(self, def_id);

                (Family::Struct, data, EntryTypedData::Other)
            }
            hir::ItemUnion(..) => {
                self.encode_bounds_and_type_for_item(def_id);

                self.encode_item_variances(def_id);

                /* Encode def_ids for each field and method
                for methods, write all the stuff get_trait_method
                needs to know*/
                let def = self.tcx.lookup_adt_def(def_id);
                let data = self.encode_variant(def.struct_variant(), None);

                // Encode inherent implementations for self union.
                encode_inherent_implementations(self, def_id);

                (Family::Union, data, EntryTypedData::Other)
            }
            hir::ItemDefaultImpl(..) => {
                (Family::DefaultImpl, EntryData::Other,
                 EntryTypedData::Impl(ImplTypedData {
                    trait_ref: tcx.impl_trait_ref(def_id)
                 }))
            }
            hir::ItemImpl(_, polarity, ..) => {
                self.encode_bounds_and_type_for_item(def_id);

                let trait_ref = tcx.impl_trait_ref(def_id);
                let parent = if let Some(trait_ref) = trait_ref {
                    let trait_def = tcx.lookup_trait_def(trait_ref.def_id);
                    trait_def.ancestors(def_id).skip(1).next().and_then(|node| {
                        match node {
                            specialization_graph::Node::Impl(parent) => Some(parent),
                            _ => None,
                        }
                    })
                } else {
                    None
                };

                self.start_tag(item_tag::children);
                self.seq(&tcx.impl_or_trait_items(def_id)[..], |_, &def_id| {
                    assert!(def_id.is_local());
                    def_id.index
                });
                self.end_tag();

                (Family::Impl,
                 EntryData::Impl(ImplData {
                    polarity: polarity,
                    parent_impl: parent,
                    coerce_unsized_kind: tcx.custom_coerce_unsized_kinds.borrow()
                                            .get(&def_id).cloned()
                 }),
                 EntryTypedData::Impl(ImplTypedData {
                    trait_ref: trait_ref
                 }))
            }
            hir::ItemTrait(..) => {
                self.encode_item_variances(def_id);
                let trait_def = tcx.lookup_trait_def(def_id);
                let trait_predicates = tcx.lookup_predicates(def_id);

                self.encode_generics(&trait_def.generics, &trait_predicates);
                self.encode_predicates(&tcx.lookup_super_predicates(def_id),
                                       item_tag::super_predicates);

                self.start_tag(item_tag::children);
                self.seq(&tcx.impl_or_trait_items(def_id)[..], |_, &def_id| {
                    assert!(def_id.is_local());
                    def_id.index
                });
                self.end_tag();

                // Encode inherent implementations for self trait.
                encode_inherent_implementations(self, def_id);

                (Family::Trait,
                 EntryData::Trait(TraitData {
                    unsafety: trait_def.unsafety,
                    paren_sugar: trait_def.paren_sugar,
                    has_default_impl: tcx.trait_has_default_impl(def_id)
                 }),
                 EntryTypedData::Trait(TraitTypedData {
                    trait_ref: trait_def.trait_ref
                 }))
            }
            hir::ItemExternCrate(_) | hir::ItemUse(_) => {
                bug!("cannot encode info for item {:?}", item)
            }
        };

        self.encode_family(family);
        self.encode_def_key(def_id);
        self.encode_visibility(&item.vis);
        encode_attributes(self, &item.attrs);
        encode_stability(self, def_id);

        self.start_tag(item_tag::data);
        data.encode(self).unwrap();
        self.end_tag();

        self.start_tag(item_tag::typed_data);
        typed_data.encode(self).unwrap();
        self.end_tag();
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
                self.encode_fields(def_id);

                let def = self.tcx.lookup_adt_def(def_id);
                for (i, variant) in def.variants.iter().enumerate() {
                    self.record(variant.did,
                                EncodeContext::encode_enum_variant_info,
                                (def_id, Untracked(i)));
                }
            }
            hir::ItemStruct(ref struct_def, _) => {
                self.encode_fields(def_id);

                // If this is a tuple-like struct, encode the type of the constructor.
                match self.tcx.lookup_adt_def(def_id).struct_variant().kind {
                    ty::VariantKind::Struct => {
                        // no value for structs like struct Foo { ... }
                    }
                    ty::VariantKind::Tuple | ty::VariantKind::Unit => {
                        // there is a value for structs like `struct
                        // Foo()` and `struct Foo`
                        let ctor_def_id = self.tcx.map.local_def_id(struct_def.id());
                        self.record(ctor_def_id,
                                    EncodeContext::encode_struct_ctor,
                                    ctor_def_id);
                    }
                }
            }
            hir::ItemUnion(..) => {
                self.encode_fields(def_id);
            }
            hir::ItemImpl(..) => {
                for &trait_item_def_id in &self.tcx.impl_or_trait_items(def_id)[..] {
                    self.record(trait_item_def_id,
                                EncodeContext::encode_info_for_impl_item,
                                trait_item_def_id);
                }
            }
            hir::ItemTrait(..) => {
                for &item_def_id in &self.tcx.impl_or_trait_items(def_id)[..] {
                    self.record(item_def_id,
                                EncodeContext::encode_info_for_trait_item,
                                item_def_id);
                }
            }
        }
    }
}

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
    fn encode_info_for_foreign_item(&mut self,
                                    (def_id, nitem): (DefId, &hir::ForeignItem)) {
        let tcx = self.tcx;

        debug!("writing foreign item {}", tcx.node_path_str(nitem.id));

        self.encode_def_key(def_id);
        self.encode_visibility(&nitem.vis);
        self.encode_bounds_and_type_for_item(def_id);
        let family = match nitem.node {
            hir::ForeignItemFn(ref fndecl, _) => {
                self.encode_fn_arg_names(&fndecl);

                Family::ForeignFn
            }
            hir::ForeignItemStatic(_, true) => Family::ForeignMutStatic,
            hir::ForeignItemStatic(_, false) => Family::ForeignImmStatic
        };
        self.encode_family(family);

        self.start_tag(item_tag::data);
        EntryData::Other.encode(self).unwrap();
        self.end_tag();

        self.start_tag(item_tag::typed_data);
        EntryTypedData::Other.encode(self).unwrap();
        self.end_tag();

        encode_attributes(self, &nitem.attrs);
        encode_stability(self, def_id);
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
        self.encode_def_key(def_id);
        self.encode_bounds_and_type_for_item(def_id);
    }

    fn encode_info_for_closure(&mut self, def_id: DefId) {
        let tcx = self.tcx;
        self.encode_def_key(def_id);
        self.encode_family(Family::Closure);

        self.start_tag(item_tag::data);
        EntryData::Closure(ClosureData {
            kind: tcx.closure_kind(def_id)
        }).encode(self).unwrap();
        self.end_tag();

        self.start_tag(item_tag::typed_data);
        EntryTypedData::Closure(ClosureTypedData {
            ty: tcx.tables.borrow().closure_tys[&def_id].clone()
        }).encode(self).unwrap();
        self.end_tag();

        assert!(self.mir_map.map.contains_key(&def_id));
        self.encode_mir(def_id);
    }
}

fn encode_info_for_items(ecx: &mut EncodeContext) -> IndexData {
    let krate = ecx.tcx.map.krate();

    // FIXME(eddyb) Avoid wrapping the items in a doc.
    ecx.start_tag(0).unwrap();

    let items = {
        let mut index = IndexBuilder::new(ecx);
        index.record(DefId::local(CRATE_DEF_INDEX),
                     EncodeContext::encode_info_for_mod,
                     FromId(CRATE_NODE_ID, (&krate.module, &krate.attrs, &hir::Public)));
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
    ecx.start_tag(root_tag::index);
    index.write_index(&mut ecx.opaque.cursor);
    ecx.end_tag();
}

fn encode_attributes(ecx: &mut EncodeContext, attrs: &[ast::Attribute]) {
    ecx.start_tag(item_tag::attributes);
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
    ecx.start_tag(root_tag::crate_deps);
    ecx.seq(&get_ordered_deps(cstore), |_, &(_, ref dep)| {
        (dep.name(), dep.hash(), dep.explicitly_linked.get())
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

    ecx.start_tag(root_tag::lang_items);
    ecx.seq(0..count, |_, _| lang_items.next().unwrap());
    ecx.end_tag();

    ecx.start_tag(root_tag::lang_items_missing);
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

    ecx.start_tag(root_tag::native_libraries);
    ecx.seq(0..count, |_, _| libs.next().unwrap());
    ecx.end_tag();
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

    ecx.start_tag(root_tag::codemap);
    ecx.seq(0..count, |_, _| filemaps.next().unwrap());
    ecx.end_tag();
}

/// Serialize the text of the exported macros
fn encode_macro_defs(ecx: &mut EncodeContext) {
    let tcx = ecx.tcx;
    ecx.start_tag(root_tag::macro_defs);
    ecx.seq(&tcx.map.krate().exported_macros, |_, def| {
        let body = ::syntax::print::pprust::tts_to_string(&def.body);
        (def.name, &def.attrs, def.span, body)
    });
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
fn encode_impls(ecx: &mut EncodeContext) {
    let mut visitor = ImplVisitor {
        tcx: ecx.tcx,
        impls: FnvHashMap()
    };
    ecx.tcx.map.krate().visit_all_items(&mut visitor);

    ecx.start_tag(root_tag::impls);
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
    ecx.start_tag(root_tag::reachable_ids);

    let reachable = ecx.reachable;
    ecx.seq(reachable, |ecx, &id| ecx.tcx.map.local_def_id(id).index);

    ecx.end_tag();
}

fn encode_dylib_dependency_formats(ecx: &mut EncodeContext) {
    ecx.start_tag(root_tag::dylib_dependency_formats);
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
        predicate_shorthands: Default::default()
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
    ecx.wr_tagged_str(root_tag::rustc_version, &rustc_version());

    let tcx = ecx.tcx;
    let link_meta = ecx.link_meta;

    ecx.start_tag(root_tag::crate_info);
    let is_rustc_macro = tcx.sess.crate_types.borrow().contains(&CrateTypeRustcMacro);
    CrateInfo {
        name: link_meta.crate_name.clone(),
        triple: tcx.sess.opts.target_triple.clone(),
        hash: link_meta.crate_hash,
        disambiguator: tcx.sess.local_crate_disambiguator().to_string(),
        panic_strategy: tcx.sess.opts.cg.panic.clone(),
        plugin_registrar_fn: tcx.sess.plugin_registrar_fn.get().map(|id| {
            tcx.map.local_def_id(id).index
        }),
        macro_derive_registrar: if is_rustc_macro {
            let id = tcx.sess.derive_registrar_fn.get().unwrap();
            Some(tcx.map.local_def_id(id).index)
        } else {
            None
        }
    }.encode(ecx).unwrap();
    ecx.end_tag();

    let mut i = ecx.position();
    encode_crate_deps(ecx, ecx.cstore);
    encode_dylib_dependency_formats(ecx);
    let dep_bytes = ecx.position() - i;

    // Encode the language items.
    i = ecx.position();
    encode_lang_items(ecx);
    let lang_item_bytes = ecx.position() - i;

    // Encode the native libraries used
    i = ecx.position();
    encode_native_libraries(ecx);
    let native_lib_bytes = ecx.position() - i;

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
    i = ecx.position();
    let items = encode_info_for_items(ecx);
    let item_bytes = ecx.position() - i;

    i = ecx.position();
    encode_item_index(ecx, items);
    let index_bytes = ecx.position() - i;

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
        println!("         codemap bytes: {}", codemap_bytes);
        println!("       macro def bytes: {}", macro_defs_bytes);
        println!("            impl bytes: {}", impl_bytes);
        println!("       reachable bytes: {}", reachable_bytes);
        println!("            item bytes: {}", item_bytes);
        println!("           index bytes: {}", index_bytes);
        println!("            zero bytes: {}", zero_bytes);
        println!("           total bytes: {}", total_bytes);
    }
}
