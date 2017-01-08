// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cstore;
use index::Index;
use schema::*;

use rustc::middle::cstore::{LinkMeta, LinkagePreference, NativeLibrary};
use rustc::hir::def;
use rustc::hir::def_id::{CrateNum, CRATE_DEF_INDEX, DefIndex, DefId};
use rustc::hir::map::definitions::DefPathTable;
use rustc::middle::dependency_format::Linkage;
use rustc::middle::lang_items;
use rustc::mir;
use rustc::traits::specialization_graph;
use rustc::ty::{self, Ty, TyCtxt};

use rustc::session::config::{self, CrateTypeProcMacro};
use rustc::util::nodemap::{FxHashMap, NodeSet};

use rustc_serialize::{Encodable, Encoder, SpecializedEncoder, opaque};
use std::hash::Hash;
use std::intrinsics;
use std::io::prelude::*;
use std::io::Cursor;
use std::rc::Rc;
use std::u32;
use syntax::ast::{self, CRATE_NODE_ID};
use syntax::codemap::Spanned;
use syntax::attr;
use syntax::symbol::Symbol;
use syntax_pos;

use rustc::hir::{self, PatKind};
use rustc::hir::itemlikevisit::ItemLikeVisitor;
use rustc::hir::intravisit::{Visitor, NestedVisitorMap};
use rustc::hir::intravisit;

use rustc_i128::{u128, i128};

use super::index_builder::{FromId, IndexBuilder, Untracked};

pub struct EncodeContext<'a, 'tcx: 'a> {
    opaque: opaque::Encoder<'a>,
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
    reexports: &'a def::ExportMap,
    link_meta: &'a LinkMeta,
    cstore: &'a cstore::CStore,
    exported_symbols: &'a NodeSet,

    lazy_state: LazyState,
    type_shorthands: FxHashMap<Ty<'tcx>, usize>,
    predicate_shorthands: FxHashMap<ty::Predicate<'tcx>, usize>,
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
        emit_u128(u128);
        emit_u64(u64);
        emit_u32(u32);
        emit_u16(u16);
        emit_u8(u8);

        emit_isize(isize);
        emit_i128(i128);
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

impl<'a, 'tcx, T> SpecializedEncoder<Lazy<T>> for EncodeContext<'a, 'tcx> {
    fn specialized_encode(&mut self, lazy: &Lazy<T>) -> Result<(), Self::Error> {
        self.emit_lazy_distance(lazy.position, Lazy::<T>::min_size())
    }
}

impl<'a, 'tcx, T> SpecializedEncoder<LazySeq<T>> for EncodeContext<'a, 'tcx> {
    fn specialized_encode(&mut self, seq: &LazySeq<T>) -> Result<(), Self::Error> {
        self.emit_usize(seq.len)?;
        if seq.len == 0 {
            return Ok(());
        }
        self.emit_lazy_distance(seq.position, LazySeq::<T>::min_size(seq.len))
    }
}

impl<'a, 'tcx> SpecializedEncoder<Ty<'tcx>> for EncodeContext<'a, 'tcx> {
    fn specialized_encode(&mut self, ty: &Ty<'tcx>) -> Result<(), Self::Error> {
        self.encode_with_shorthand(ty, &ty.sty, |ecx| &mut ecx.type_shorthands)
    }
}

impl<'a, 'tcx> SpecializedEncoder<ty::GenericPredicates<'tcx>> for EncodeContext<'a, 'tcx> {
    fn specialized_encode(&mut self,
                          predicates: &ty::GenericPredicates<'tcx>)
                          -> Result<(), Self::Error> {
        predicates.parent.encode(self)?;
        predicates.predicates.len().encode(self)?;
        for predicate in &predicates.predicates {
            self.encode_with_shorthand(predicate, predicate, |ecx| &mut ecx.predicate_shorthands)?
        }
        Ok(())
    }
}

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
    pub fn position(&self) -> usize {
        self.opaque.position()
    }

    fn emit_node<F: FnOnce(&mut Self, usize) -> R, R>(&mut self, f: F) -> R {
        assert_eq!(self.lazy_state, LazyState::NoNode);
        let pos = self.position();
        self.lazy_state = LazyState::NodeStart(pos);
        let r = f(self, pos);
        self.lazy_state = LazyState::NoNode;
        r
    }

    fn emit_lazy_distance(&mut self,
                          position: usize,
                          min_size: usize)
                          -> Result<(), <Self as Encoder>::Error> {
        let min_end = position + min_size;
        let distance = match self.lazy_state {
            LazyState::NoNode => bug!("emit_lazy_distance: outside of a metadata node"),
            LazyState::NodeStart(start) => {
                assert!(min_end <= start);
                start - min_end
            }
            LazyState::Previous(last_min_end) => {
                assert!(last_min_end <= position);
                position - last_min_end
            }
        };
        self.lazy_state = LazyState::Previous(min_end);
        self.emit_usize(distance)
    }

    pub fn lazy<T: Encodable>(&mut self, value: &T) -> Lazy<T> {
        self.emit_node(|ecx, pos| {
            value.encode(ecx).unwrap();

            assert!(pos + Lazy::<T>::min_size() <= ecx.position());
            Lazy::with_position(pos)
        })
    }

    fn lazy_seq<I, T>(&mut self, iter: I) -> LazySeq<T>
        where I: IntoIterator<Item = T>,
              T: Encodable
    {
        self.emit_node(|ecx, pos| {
            let len = iter.into_iter().map(|value| value.encode(ecx).unwrap()).count();

            assert!(pos + LazySeq::<T>::min_size(len) <= ecx.position());
            LazySeq::with_position_and_length(pos, len)
        })
    }

    fn lazy_seq_ref<'b, I, T>(&mut self, iter: I) -> LazySeq<T>
        where I: IntoIterator<Item = &'b T>,
              T: 'b + Encodable
    {
        self.emit_node(|ecx, pos| {
            let len = iter.into_iter().map(|value| value.encode(ecx).unwrap()).count();

            assert!(pos + LazySeq::<T>::min_size(len) <= ecx.position());
            LazySeq::with_position_and_length(pos, len)
        })
    }

    /// Encode the given value or a previously cached shorthand.
    fn encode_with_shorthand<T, U, M>(&mut self,
                                      value: &T,
                                      variant: &U,
                                      map: M)
                                      -> Result<(), <Self as Encoder>::Error>
        where M: for<'b> Fn(&'b mut Self) -> &'b mut FxHashMap<T, usize>,
              T: Clone + Eq + Hash,
              U: Encodable
    {
        let existing_shorthand = map(self).get(value).cloned();
        if let Some(shorthand) = existing_shorthand {
            return self.emit_usize(shorthand);
        }

        let start = self.position();
        variant.encode(self)?;
        let len = self.position() - start;

        // The shorthand encoding uses the same usize as the
        // discriminant, with an offset so they can't conflict.
        let discriminant = unsafe { intrinsics::discriminant_value(variant) };
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

    fn encode_item_variances(&mut self, def_id: DefId) -> LazySeq<ty::Variance> {
        let tcx = self.tcx;
        self.lazy_seq(tcx.item_variances(def_id).iter().cloned())
    }

    fn encode_item_type(&mut self, def_id: DefId) -> Lazy<Ty<'tcx>> {
        let tcx = self.tcx;
        self.lazy(&tcx.item_type(def_id))
    }

    /// Encode data for the given variant of the given ADT. The
    /// index of the variant is untracked: this is ok because we
    /// will have to lookup the adt-def by its id, and that gives us
    /// the right to access any information in the adt-def (including,
    /// e.g., the length of the various vectors).
    fn encode_enum_variant_info(&mut self,
                                (enum_did, Untracked(index)): (DefId, Untracked<usize>))
                                -> Entry<'tcx> {
        let tcx = self.tcx;
        let def = tcx.lookup_adt_def(enum_did);
        let variant = &def.variants[index];
        let def_id = variant.did;

        let data = VariantData {
            ctor_kind: variant.ctor_kind,
            disr: variant.disr_val.to_u128_unchecked(),
            struct_ctor: None,
        };

        let enum_id = tcx.map.as_local_node_id(enum_did).unwrap();
        let enum_vis = &tcx.map.expect_item(enum_id).vis;

        Entry {
            kind: EntryKind::Variant(self.lazy(&data)),
            visibility: self.lazy(&ty::Visibility::from_hir(enum_vis, enum_id, tcx)),
            span: self.lazy(&tcx.def_span(def_id)),
            attributes: self.encode_attributes(&tcx.get_attrs(def_id)),
            children: self.lazy_seq(variant.fields.iter().map(|f| {
                assert!(f.did.is_local());
                f.did.index
            })),
            stability: self.encode_stability(def_id),
            deprecation: self.encode_deprecation(def_id),

            ty: Some(self.encode_item_type(def_id)),
            inherent_impls: LazySeq::empty(),
            variances: LazySeq::empty(),
            generics: Some(self.encode_generics(def_id)),
            predicates: Some(self.encode_predicates(def_id)),

            ast: None,
            mir: None,
        }
    }

    fn encode_info_for_mod(&mut self,
                           FromId(id, (md, attrs, vis)): FromId<(&hir::Mod,
                                                                 &[ast::Attribute],
                                                                 &hir::Visibility)>)
                           -> Entry<'tcx> {
        let tcx = self.tcx;
        let def_id = tcx.map.local_def_id(id);

        let data = ModData {
            reexports: match self.reexports.get(&id) {
                Some(exports) if *vis == hir::Public => self.lazy_seq_ref(exports),
                _ => LazySeq::empty(),
            },
        };

        Entry {
            kind: EntryKind::Mod(self.lazy(&data)),
            visibility: self.lazy(&ty::Visibility::from_hir(vis, id, tcx)),
            span: self.lazy(&md.inner),
            attributes: self.encode_attributes(attrs),
            children: self.lazy_seq(md.item_ids.iter().map(|item_id| {
                tcx.map.local_def_id(item_id.id).index
            })),
            stability: self.encode_stability(def_id),
            deprecation: self.encode_deprecation(def_id),

            ty: None,
            inherent_impls: LazySeq::empty(),
            variances: LazySeq::empty(),
            generics: None,
            predicates: None,

            ast: None,
            mir: None
        }
    }
}

impl<'a, 'b, 'tcx> IndexBuilder<'a, 'b, 'tcx> {
    fn encode_fields(&mut self, adt_def_id: DefId) {
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
                    (adt_def_id, Untracked((variant_index, field_index))): (DefId,
                                                                            Untracked<(usize,
                                                                                       usize)>))
                    -> Entry<'tcx> {
        let tcx = self.tcx;
        let variant = &tcx.lookup_adt_def(adt_def_id).variants[variant_index];
        let field = &variant.fields[field_index];

        let def_id = field.did;
        let variant_id = tcx.map.as_local_node_id(variant.did).unwrap();
        let variant_data = tcx.map.expect_variant_data(variant_id);

        Entry {
            kind: EntryKind::Field,
            visibility: self.lazy(&field.vis),
            span: self.lazy(&tcx.def_span(def_id)),
            attributes: self.encode_attributes(&variant_data.fields()[field_index].attrs),
            children: LazySeq::empty(),
            stability: self.encode_stability(def_id),
            deprecation: self.encode_deprecation(def_id),

            ty: Some(self.encode_item_type(def_id)),
            inherent_impls: LazySeq::empty(),
            variances: LazySeq::empty(),
            generics: Some(self.encode_generics(def_id)),
            predicates: Some(self.encode_predicates(def_id)),

            ast: None,
            mir: None,
        }
    }

    fn encode_struct_ctor(&mut self, (adt_def_id, def_id): (DefId, DefId)) -> Entry<'tcx> {
        let tcx = self.tcx;
        let variant = tcx.lookup_adt_def(adt_def_id).struct_variant();

        let data = VariantData {
            ctor_kind: variant.ctor_kind,
            disr: variant.disr_val.to_u128_unchecked(),
            struct_ctor: Some(def_id.index),
        };

        let struct_id = tcx.map.as_local_node_id(adt_def_id).unwrap();
        let struct_vis = &tcx.map.expect_item(struct_id).vis;

        Entry {
            kind: EntryKind::Struct(self.lazy(&data)),
            visibility: self.lazy(&ty::Visibility::from_hir(struct_vis, struct_id, tcx)),
            span: self.lazy(&tcx.def_span(def_id)),
            attributes: LazySeq::empty(),
            children: LazySeq::empty(),
            stability: self.encode_stability(def_id),
            deprecation: self.encode_deprecation(def_id),

            ty: Some(self.encode_item_type(def_id)),
            inherent_impls: LazySeq::empty(),
            variances: LazySeq::empty(),
            generics: Some(self.encode_generics(def_id)),
            predicates: Some(self.encode_predicates(def_id)),

            ast: None,
            mir: None,
        }
    }

    fn encode_generics(&mut self, def_id: DefId) -> Lazy<ty::Generics<'tcx>> {
        let tcx = self.tcx;
        self.lazy(tcx.item_generics(def_id))
    }

    fn encode_predicates(&mut self, def_id: DefId) -> Lazy<ty::GenericPredicates<'tcx>> {
        let tcx = self.tcx;
        self.lazy(&tcx.item_predicates(def_id))
    }

    fn encode_info_for_trait_item(&mut self, def_id: DefId) -> Entry<'tcx> {
        let tcx = self.tcx;

        let node_id = tcx.map.as_local_node_id(def_id).unwrap();
        let ast_item = tcx.map.expect_trait_item(node_id);
        let trait_item = tcx.associated_item(def_id);

        let container = match trait_item.defaultness {
            hir::Defaultness::Default { has_value: true } =>
                AssociatedContainer::TraitWithDefault,
            hir::Defaultness::Default { has_value: false } =>
                AssociatedContainer::TraitRequired,
            hir::Defaultness::Final =>
                span_bug!(ast_item.span, "traits cannot have final items"),
        };

        let kind = match trait_item.kind {
            ty::AssociatedKind::Const => EntryKind::AssociatedConst(container),
            ty::AssociatedKind::Method => {
                let fn_data = if let hir::TraitItemKind::Method(_, ref m) = ast_item.node {
                    let arg_names = match *m {
                        hir::TraitMethod::Required(ref names) => {
                            self.encode_fn_arg_names(names)
                        }
                        hir::TraitMethod::Provided(body) => {
                            self.encode_fn_arg_names_for_body(body)
                        }
                    };
                    FnData {
                        constness: hir::Constness::NotConst,
                        arg_names: arg_names
                    }
                } else {
                    bug!()
                };
                EntryKind::Method(self.lazy(&MethodData {
                    fn_data: fn_data,
                    container: container,
                    has_self: trait_item.method_has_self_argument,
                }))
            }
            ty::AssociatedKind::Type => EntryKind::AssociatedType(container),
        };

        Entry {
            kind: kind,
            visibility: self.lazy(&trait_item.vis),
            span: self.lazy(&ast_item.span),
            attributes: self.encode_attributes(&ast_item.attrs),
            children: LazySeq::empty(),
            stability: self.encode_stability(def_id),
            deprecation: self.encode_deprecation(def_id),

            ty: match trait_item.kind {
                ty::AssociatedKind::Const |
                ty::AssociatedKind::Method => {
                    Some(self.encode_item_type(def_id))
                }
                ty::AssociatedKind::Type => {
                    if trait_item.defaultness.has_value() {
                        Some(self.encode_item_type(def_id))
                    } else {
                        None
                    }
                }
            },
            inherent_impls: LazySeq::empty(),
            variances: LazySeq::empty(),
            generics: Some(self.encode_generics(def_id)),
            predicates: Some(self.encode_predicates(def_id)),

            ast: if let hir::TraitItemKind::Const(_, Some(body)) = ast_item.node {
                Some(self.encode_body(body))
            } else {
                None
            },
            mir: self.encode_mir(def_id),
        }
    }

    fn encode_info_for_impl_item(&mut self, def_id: DefId) -> Entry<'tcx> {
        let node_id = self.tcx.map.as_local_node_id(def_id).unwrap();
        let ast_item = self.tcx.map.expect_impl_item(node_id);
        let impl_item = self.tcx.associated_item(def_id);

        let container = match impl_item.defaultness {
            hir::Defaultness::Default { has_value: true } => AssociatedContainer::ImplDefault,
            hir::Defaultness::Final => AssociatedContainer::ImplFinal,
            hir::Defaultness::Default { has_value: false } =>
                span_bug!(ast_item.span, "impl items always have values (currently)"),
        };

        let kind = match impl_item.kind {
            ty::AssociatedKind::Const => EntryKind::AssociatedConst(container),
            ty::AssociatedKind::Method => {
                let fn_data = if let hir::ImplItemKind::Method(ref sig, body) = ast_item.node {
                    FnData {
                        constness: sig.constness,
                        arg_names: self.encode_fn_arg_names_for_body(body),
                    }
                } else {
                    bug!()
                };
                EntryKind::Method(self.lazy(&MethodData {
                    fn_data: fn_data,
                    container: container,
                    has_self: impl_item.method_has_self_argument,
                }))
            }
            ty::AssociatedKind::Type => EntryKind::AssociatedType(container)
        };

        let (ast, mir) = if let hir::ImplItemKind::Const(_, body) = ast_item.node {
            (Some(body), true)
        } else if let hir::ImplItemKind::Method(ref sig, body) = ast_item.node {
            let generics = self.tcx.item_generics(def_id);
            let types = generics.parent_types as usize + generics.types.len();
            let needs_inline = types > 0 || attr::requests_inline(&ast_item.attrs);
            let is_const_fn = sig.constness == hir::Constness::Const;
            let ast = if is_const_fn { Some(body) } else { None };
            let always_encode_mir = self.tcx.sess.opts.debugging_opts.always_encode_mir;
            (ast, needs_inline || is_const_fn || always_encode_mir)
        } else {
            (None, false)
        };

        Entry {
            kind: kind,
            visibility: self.lazy(&impl_item.vis),
            span: self.lazy(&ast_item.span),
            attributes: self.encode_attributes(&ast_item.attrs),
            children: LazySeq::empty(),
            stability: self.encode_stability(def_id),
            deprecation: self.encode_deprecation(def_id),

            ty: Some(self.encode_item_type(def_id)),
            inherent_impls: LazySeq::empty(),
            variances: LazySeq::empty(),
            generics: Some(self.encode_generics(def_id)),
            predicates: Some(self.encode_predicates(def_id)),

            ast: ast.map(|body| self.encode_body(body)),
            mir: if mir { self.encode_mir(def_id) } else { None },
        }
    }

    fn encode_fn_arg_names_for_body(&mut self, body_id: hir::BodyId)
                                    -> LazySeq<ast::Name> {
        let _ignore = self.tcx.dep_graph.in_ignore();
        let body = self.tcx.map.body(body_id);
        self.lazy_seq(body.arguments.iter().map(|arg| {
            match arg.pat.node {
                PatKind::Binding(_, _, name, _) => name.node,
                _ => Symbol::intern("")
            }
        }))
    }

    fn encode_fn_arg_names(&mut self, names: &[Spanned<ast::Name>])
                           -> LazySeq<ast::Name> {
        self.lazy_seq(names.iter().map(|name| name.node))
    }

    fn encode_mir(&mut self, def_id: DefId) -> Option<Lazy<mir::Mir<'tcx>>> {
        self.tcx.mir_map.borrow().get(&def_id).map(|mir| self.lazy(&*mir.borrow()))
    }

    // Encodes the inherent implementations of a structure, enumeration, or trait.
    fn encode_inherent_implementations(&mut self, def_id: DefId) -> LazySeq<DefIndex> {
        match self.tcx.inherent_impls.borrow().get(&def_id) {
            None => LazySeq::empty(),
            Some(implementations) => {
                self.lazy_seq(implementations.iter().map(|&def_id| {
                    assert!(def_id.is_local());
                    def_id.index
                }))
            }
        }
    }

    fn encode_stability(&mut self, def_id: DefId) -> Option<Lazy<attr::Stability>> {
        self.tcx.lookup_stability(def_id).map(|stab| self.lazy(stab))
    }

    fn encode_deprecation(&mut self, def_id: DefId) -> Option<Lazy<attr::Deprecation>> {
        self.tcx.lookup_deprecation(def_id).map(|depr| self.lazy(&depr))
    }

    fn encode_info_for_item(&mut self, (def_id, item): (DefId, &'tcx hir::Item)) -> Entry<'tcx> {
        let tcx = self.tcx;

        debug!("encoding info for item at {}",
               tcx.sess.codemap().span_to_string(item.span));

        let kind = match item.node {
            hir::ItemStatic(_, hir::MutMutable, _) => EntryKind::MutStatic,
            hir::ItemStatic(_, hir::MutImmutable, _) => EntryKind::ImmStatic,
            hir::ItemConst(..) => EntryKind::Const,
            hir::ItemFn(_, _, constness, .., body) => {
                let data = FnData {
                    constness: constness,
                    arg_names: self.encode_fn_arg_names_for_body(body),
                };

                EntryKind::Fn(self.lazy(&data))
            }
            hir::ItemMod(ref m) => {
                return self.encode_info_for_mod(FromId(item.id, (m, &item.attrs, &item.vis)));
            }
            hir::ItemForeignMod(_) => EntryKind::ForeignMod,
            hir::ItemTy(..) => EntryKind::Type,
            hir::ItemEnum(..) => EntryKind::Enum,
            hir::ItemStruct(ref struct_def, _) => {
                let variant = tcx.lookup_adt_def(def_id).struct_variant();

                // Encode def_ids for each field and method
                // for methods, write all the stuff get_trait_method
                // needs to know
                let struct_ctor = if !struct_def.is_struct() {
                    Some(tcx.map.local_def_id(struct_def.id()).index)
                } else {
                    None
                };
                EntryKind::Struct(self.lazy(&VariantData {
                    ctor_kind: variant.ctor_kind,
                    disr: variant.disr_val.to_u128_unchecked(),
                    struct_ctor: struct_ctor,
                }))
            }
            hir::ItemUnion(..) => {
                let variant = tcx.lookup_adt_def(def_id).struct_variant();

                EntryKind::Union(self.lazy(&VariantData {
                    ctor_kind: variant.ctor_kind,
                    disr: variant.disr_val.to_u128_unchecked(),
                    struct_ctor: None,
                }))
            }
            hir::ItemDefaultImpl(..) => {
                let data = ImplData {
                    polarity: hir::ImplPolarity::Positive,
                    parent_impl: None,
                    coerce_unsized_kind: None,
                    trait_ref: tcx.impl_trait_ref(def_id).map(|trait_ref| self.lazy(&trait_ref)),
                };

                EntryKind::DefaultImpl(self.lazy(&data))
            }
            hir::ItemImpl(_, polarity, ..) => {
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

                let data = ImplData {
                    polarity: polarity,
                    parent_impl: parent,
                    coerce_unsized_kind: tcx.custom_coerce_unsized_kinds
                        .borrow()
                        .get(&def_id)
                        .cloned(),
                    trait_ref: trait_ref.map(|trait_ref| self.lazy(&trait_ref)),
                };

                EntryKind::Impl(self.lazy(&data))
            }
            hir::ItemTrait(..) => {
                let trait_def = tcx.lookup_trait_def(def_id);
                let data = TraitData {
                    unsafety: trait_def.unsafety,
                    paren_sugar: trait_def.paren_sugar,
                    has_default_impl: tcx.trait_has_default_impl(def_id),
                    super_predicates: self.lazy(&tcx.item_super_predicates(def_id)),
                };

                EntryKind::Trait(self.lazy(&data))
            }
            hir::ItemExternCrate(_) |
            hir::ItemUse(..) => bug!("cannot encode info for item {:?}", item),
        };

        Entry {
            kind: kind,
            visibility: self.lazy(&ty::Visibility::from_hir(&item.vis, item.id, tcx)),
            span: self.lazy(&item.span),
            attributes: self.encode_attributes(&item.attrs),
            children: match item.node {
                hir::ItemForeignMod(ref fm) => {
                    self.lazy_seq(fm.items
                        .iter()
                        .map(|foreign_item| tcx.map.local_def_id(foreign_item.id).index))
                }
                hir::ItemEnum(..) => {
                    let def = self.tcx.lookup_adt_def(def_id);
                    self.lazy_seq(def.variants.iter().map(|v| {
                        assert!(v.did.is_local());
                        v.did.index
                    }))
                }
                hir::ItemStruct(..) |
                hir::ItemUnion(..) => {
                    let def = self.tcx.lookup_adt_def(def_id);
                    self.lazy_seq(def.struct_variant().fields.iter().map(|f| {
                        assert!(f.did.is_local());
                        f.did.index
                    }))
                }
                hir::ItemImpl(..) |
                hir::ItemTrait(..) => {
                    self.lazy_seq(tcx.associated_item_def_ids(def_id).iter().map(|&def_id| {
                        assert!(def_id.is_local());
                        def_id.index
                    }))
                }
                _ => LazySeq::empty(),
            },
            stability: self.encode_stability(def_id),
            deprecation: self.encode_deprecation(def_id),

            ty: match item.node {
                hir::ItemStatic(..) |
                hir::ItemConst(..) |
                hir::ItemFn(..) |
                hir::ItemTy(..) |
                hir::ItemEnum(..) |
                hir::ItemStruct(..) |
                hir::ItemUnion(..) |
                hir::ItemImpl(..) => Some(self.encode_item_type(def_id)),
                _ => None,
            },
            inherent_impls: self.encode_inherent_implementations(def_id),
            variances: match item.node {
                hir::ItemEnum(..) |
                hir::ItemStruct(..) |
                hir::ItemUnion(..) |
                hir::ItemTrait(..) => self.encode_item_variances(def_id),
                _ => LazySeq::empty(),
            },
            generics: match item.node {
                hir::ItemStatic(..) |
                hir::ItemConst(..) |
                hir::ItemFn(..) |
                hir::ItemTy(..) |
                hir::ItemEnum(..) |
                hir::ItemStruct(..) |
                hir::ItemUnion(..) |
                hir::ItemImpl(..) |
                hir::ItemTrait(..) => Some(self.encode_generics(def_id)),
                _ => None,
            },
            predicates: match item.node {
                hir::ItemStatic(..) |
                hir::ItemConst(..) |
                hir::ItemFn(..) |
                hir::ItemTy(..) |
                hir::ItemEnum(..) |
                hir::ItemStruct(..) |
                hir::ItemUnion(..) |
                hir::ItemImpl(..) |
                hir::ItemTrait(..) => Some(self.encode_predicates(def_id)),
                _ => None,
            },

            ast: match item.node {
                hir::ItemConst(_, body) |
                hir::ItemFn(_, _, hir::Constness::Const, _, _, body) => {
                    Some(self.encode_body(body))
                }
                _ => None,
            },
            mir: match item.node {
                hir::ItemStatic(..) if self.tcx.sess.opts.debugging_opts.always_encode_mir => {
                    self.encode_mir(def_id)
                }
                hir::ItemConst(..) => self.encode_mir(def_id),
                hir::ItemFn(_, _, constness, _, ref generics, _) => {
                    let tps_len = generics.ty_params.len();
                    let needs_inline = tps_len > 0 || attr::requests_inline(&item.attrs);
                    let always_encode_mir = self.tcx.sess.opts.debugging_opts.always_encode_mir;
                    if needs_inline || constness == hir::Constness::Const || always_encode_mir {
                        self.encode_mir(def_id)
                    } else {
                        None
                    }
                }
                _ => None,
            },
        }
    }

    /// Serialize the text of exported macros
    fn encode_info_for_macro_def(&mut self, macro_def: &hir::MacroDef) -> Entry<'tcx> {
        Entry {
            kind: EntryKind::MacroDef(self.lazy(&MacroDef {
                body: ::syntax::print::pprust::tts_to_string(&macro_def.body)
            })),
            visibility: self.lazy(&ty::Visibility::Public),
            span: self.lazy(&macro_def.span),

            attributes: self.encode_attributes(&macro_def.attrs),
            children: LazySeq::empty(),
            stability: None,
            deprecation: None,
            ty: None,
            inherent_impls: LazySeq::empty(),
            variances: LazySeq::empty(),
            generics: None,
            predicates: None,
            ast: None,
            mir: None,
        }
    }
}

impl<'a, 'b, 'tcx> IndexBuilder<'a, 'b, 'tcx> {
    /// In some cases, along with the item itself, we also
    /// encode some sub-items. Usually we want some info from the item
    /// so it's easier to do that here then to wait until we would encounter
    /// normally in the visitor walk.
    fn encode_addl_info_for_item(&mut self, item: &hir::Item) {
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

                // If the struct has a constructor, encode it.
                if !struct_def.is_struct() {
                    let ctor_def_id = self.tcx.map.local_def_id(struct_def.id());
                    self.record(ctor_def_id,
                                EncodeContext::encode_struct_ctor,
                                (def_id, ctor_def_id));
                }
            }
            hir::ItemUnion(..) => {
                self.encode_fields(def_id);
            }
            hir::ItemImpl(..) => {
                for &trait_item_def_id in &self.tcx.associated_item_def_ids(def_id)[..] {
                    self.record(trait_item_def_id,
                                EncodeContext::encode_info_for_impl_item,
                                trait_item_def_id);
                }
            }
            hir::ItemTrait(..) => {
                for &item_def_id in &self.tcx.associated_item_def_ids(def_id)[..] {
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
                                    (def_id, nitem): (DefId, &hir::ForeignItem))
                                    -> Entry<'tcx> {
        let tcx = self.tcx;

        debug!("writing foreign item {}", tcx.node_path_str(nitem.id));

        let kind = match nitem.node {
            hir::ForeignItemFn(_, ref names, _) => {
                let data = FnData {
                    constness: hir::Constness::NotConst,
                    arg_names: self.encode_fn_arg_names(names),
                };
                EntryKind::ForeignFn(self.lazy(&data))
            }
            hir::ForeignItemStatic(_, true) => EntryKind::ForeignMutStatic,
            hir::ForeignItemStatic(_, false) => EntryKind::ForeignImmStatic,
        };

        Entry {
            kind: kind,
            visibility: self.lazy(&ty::Visibility::from_hir(&nitem.vis, nitem.id, tcx)),
            span: self.lazy(&nitem.span),
            attributes: self.encode_attributes(&nitem.attrs),
            children: LazySeq::empty(),
            stability: self.encode_stability(def_id),
            deprecation: self.encode_deprecation(def_id),

            ty: Some(self.encode_item_type(def_id)),
            inherent_impls: LazySeq::empty(),
            variances: LazySeq::empty(),
            generics: Some(self.encode_generics(def_id)),
            predicates: Some(self.encode_predicates(def_id)),

            ast: None,
            mir: None,
        }
    }
}

struct EncodeVisitor<'a, 'b: 'a, 'tcx: 'b> {
    index: IndexBuilder<'a, 'b, 'tcx>,
}

impl<'a, 'b, 'tcx> Visitor<'tcx> for EncodeVisitor<'a, 'b, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.index.tcx.map)
    }
    fn visit_expr(&mut self, ex: &'tcx hir::Expr) {
        intravisit::walk_expr(self, ex);
        self.index.encode_info_for_expr(ex);
    }
    fn visit_item(&mut self, item: &'tcx hir::Item) {
        intravisit::walk_item(self, item);
        let def_id = self.index.tcx.map.local_def_id(item.id);
        match item.node {
            hir::ItemExternCrate(_) |
            hir::ItemUse(..) => (), // ignore these
            _ => self.index.record(def_id, EncodeContext::encode_info_for_item, (def_id, item)),
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
    fn visit_macro_def(&mut self, macro_def: &'tcx hir::MacroDef) {
        let def_id = self.index.tcx.map.local_def_id(macro_def.id);
        self.index.record(def_id, EncodeContext::encode_info_for_macro_def, macro_def);
    }
}

impl<'a, 'b, 'tcx> IndexBuilder<'a, 'b, 'tcx> {
    fn encode_info_for_ty(&mut self, ty: &hir::Ty) {
        if let hir::TyImplTrait(_) = ty.node {
            let def_id = self.tcx.map.local_def_id(ty.id);
            self.record(def_id, EncodeContext::encode_info_for_anon_ty, def_id);
        }
    }

    fn encode_info_for_expr(&mut self, expr: &hir::Expr) {
        match expr.node {
            hir::ExprClosure(..) => {
                let def_id = self.tcx.map.local_def_id(expr.id);
                self.record(def_id, EncodeContext::encode_info_for_closure, def_id);
            }
            _ => {}
        }
    }
}

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
    fn encode_info_for_anon_ty(&mut self, def_id: DefId) -> Entry<'tcx> {
        let tcx = self.tcx;
        Entry {
            kind: EntryKind::Type,
            visibility: self.lazy(&ty::Visibility::Public),
            span: self.lazy(&tcx.def_span(def_id)),
            attributes: LazySeq::empty(),
            children: LazySeq::empty(),
            stability: None,
            deprecation: None,

            ty: Some(self.encode_item_type(def_id)),
            inherent_impls: LazySeq::empty(),
            variances: LazySeq::empty(),
            generics: Some(self.encode_generics(def_id)),
            predicates: Some(self.encode_predicates(def_id)),

            ast: None,
            mir: None,
        }
    }

    fn encode_info_for_closure(&mut self, def_id: DefId) -> Entry<'tcx> {
        let tcx = self.tcx;

        let data = ClosureData {
            kind: tcx.closure_kind(def_id),
            ty: self.lazy(&tcx.closure_tys.borrow()[&def_id]),
        };

        Entry {
            kind: EntryKind::Closure(self.lazy(&data)),
            visibility: self.lazy(&ty::Visibility::Public),
            span: self.lazy(&tcx.def_span(def_id)),
            attributes: self.encode_attributes(&tcx.get_attrs(def_id)),
            children: LazySeq::empty(),
            stability: None,
            deprecation: None,

            ty: Some(self.encode_item_type(def_id)),
            inherent_impls: LazySeq::empty(),
            variances: LazySeq::empty(),
            generics: Some(self.encode_generics(def_id)),
            predicates: None,

            ast: None,
            mir: self.encode_mir(def_id),
        }
    }

    fn encode_info_for_items(&mut self) -> Index {
        let krate = self.tcx.map.krate();
        let mut index = IndexBuilder::new(self);
        index.record(DefId::local(CRATE_DEF_INDEX),
                     EncodeContext::encode_info_for_mod,
                     FromId(CRATE_NODE_ID, (&krate.module, &krate.attrs, &hir::Public)));
        let mut visitor = EncodeVisitor { index: index };
        krate.visit_all_item_likes(&mut visitor.as_deep_visitor());
        for macro_def in &krate.exported_macros {
            visitor.visit_macro_def(macro_def);
        }
        visitor.index.into_items()
    }

    fn encode_attributes(&mut self, attrs: &[ast::Attribute]) -> LazySeq<ast::Attribute> {
        self.lazy_seq_ref(attrs)
    }

    fn encode_crate_deps(&mut self) -> LazySeq<CrateDep> {
        fn get_ordered_deps(cstore: &cstore::CStore) -> Vec<(CrateNum, Rc<cstore::CrateMetadata>)> {
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
        let deps = get_ordered_deps(self.cstore);
        self.lazy_seq(deps.iter().map(|&(_, ref dep)| {
            CrateDep {
                name: dep.name(),
                hash: dep.hash(),
                kind: dep.dep_kind.get(),
            }
        }))
    }

    fn encode_lang_items(&mut self) -> (LazySeq<(DefIndex, usize)>, LazySeq<lang_items::LangItem>) {
        let tcx = self.tcx;
        let lang_items = tcx.lang_items.items().iter();
        (self.lazy_seq(lang_items.enumerate().filter_map(|(i, &opt_def_id)| {
            if let Some(def_id) = opt_def_id {
                if def_id.is_local() {
                    return Some((def_id.index, i));
                }
            }
            None
        })),
         self.lazy_seq_ref(&tcx.lang_items.missing))
    }

    fn encode_native_libraries(&mut self) -> LazySeq<NativeLibrary> {
        let used_libraries = self.tcx.sess.cstore.used_libraries();
        self.lazy_seq(used_libraries)
    }

    fn encode_codemap(&mut self) -> LazySeq<syntax_pos::FileMap> {
        let codemap = self.tcx.sess.codemap();
        let all_filemaps = codemap.files.borrow();
        self.lazy_seq_ref(all_filemaps.iter()
            .filter(|filemap| {
                // No need to re-export imported filemaps, as any downstream
                // crate will import them from their original source.
                !filemap.is_imported()
            })
            .map(|filemap| &**filemap))
    }

    fn encode_def_path_table(&mut self) -> Lazy<DefPathTable> {
        let definitions = self.tcx.map.definitions();
        self.lazy(definitions.def_path_table())
    }
}

struct ImplVisitor<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    impls: FxHashMap<DefId, Vec<DefIndex>>,
}

impl<'a, 'tcx, 'v> ItemLikeVisitor<'v> for ImplVisitor<'a, 'tcx> {
    fn visit_item(&mut self, item: &hir::Item) {
        if let hir::ItemImpl(..) = item.node {
            let impl_id = self.tcx.map.local_def_id(item.id);
            if let Some(trait_ref) = self.tcx.impl_trait_ref(impl_id) {
                self.impls
                    .entry(trait_ref.def_id)
                    .or_insert(vec![])
                    .push(impl_id.index);
            }
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &'v hir::TraitItem) {}

    fn visit_impl_item(&mut self, _impl_item: &'v hir::ImplItem) {
        // handled in `visit_item` above
    }
}

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
    /// Encodes an index, mapping each trait to its (local) implementations.
    fn encode_impls(&mut self) -> LazySeq<TraitImpls> {
        let mut visitor = ImplVisitor {
            tcx: self.tcx,
            impls: FxHashMap(),
        };
        self.tcx.map.krate().visit_all_item_likes(&mut visitor);

        let all_impls: Vec<_> = visitor.impls
            .into_iter()
            .map(|(trait_def_id, impls)| {
                TraitImpls {
                    trait_id: (trait_def_id.krate.as_u32(), trait_def_id.index),
                    impls: self.lazy_seq(impls),
                }
            })
            .collect();

        self.lazy_seq(all_impls)
    }

    // Encodes all symbols exported from this crate into the metadata.
    //
    // This pass is seeded off the reachability list calculated in the
    // middle::reachable module but filters out items that either don't have a
    // symbol associated with them (they weren't translated) or if they're an FFI
    // definition (as that's not defined in this crate).
    fn encode_exported_symbols(&mut self) -> LazySeq<DefIndex> {
        let exported_symbols = self.exported_symbols;
        let tcx = self.tcx;
        self.lazy_seq(exported_symbols.iter().map(|&id| tcx.map.local_def_id(id).index))
    }

    fn encode_dylib_dependency_formats(&mut self) -> LazySeq<Option<LinkagePreference>> {
        match self.tcx.sess.dependency_formats.borrow().get(&config::CrateTypeDylib) {
            Some(arr) => {
                self.lazy_seq(arr.iter().map(|slot| {
                    match *slot {
                        Linkage::NotLinked |
                        Linkage::IncludedFromDylib => None,

                        Linkage::Dynamic => Some(LinkagePreference::RequireDynamic),
                        Linkage::Static => Some(LinkagePreference::RequireStatic),
                    }
                }))
            }
            None => LazySeq::empty(),
        }
    }

    fn encode_crate_root(&mut self) -> Lazy<CrateRoot> {
        let mut i = self.position();
        let crate_deps = self.encode_crate_deps();
        let dylib_dependency_formats = self.encode_dylib_dependency_formats();
        let dep_bytes = self.position() - i;

        // Encode the language items.
        i = self.position();
        let (lang_items, lang_items_missing) = self.encode_lang_items();
        let lang_item_bytes = self.position() - i;

        // Encode the native libraries used
        i = self.position();
        let native_libraries = self.encode_native_libraries();
        let native_lib_bytes = self.position() - i;

        // Encode codemap
        i = self.position();
        let codemap = self.encode_codemap();
        let codemap_bytes = self.position() - i;

        // Encode DefPathTable
        i = self.position();
        let def_path_table = self.encode_def_path_table();
        let def_path_table_bytes = self.position() - i;

        // Encode the def IDs of impls, for coherence checking.
        i = self.position();
        let impls = self.encode_impls();
        let impl_bytes = self.position() - i;

        // Encode exported symbols info.
        i = self.position();
        let exported_symbols = self.encode_exported_symbols();
        let exported_symbols_bytes = self.position() - i;

        // Encode and index the items.
        i = self.position();
        let items = self.encode_info_for_items();
        let item_bytes = self.position() - i;

        i = self.position();
        let index = items.write_index(&mut self.opaque.cursor);
        let index_bytes = self.position() - i;

        let tcx = self.tcx;
        let link_meta = self.link_meta;
        let is_proc_macro = tcx.sess.crate_types.borrow().contains(&CrateTypeProcMacro);
        let root = self.lazy(&CrateRoot {
            name: link_meta.crate_name,
            triple: tcx.sess.opts.target_triple.clone(),
            hash: link_meta.crate_hash,
            disambiguator: tcx.sess.local_crate_disambiguator(),
            panic_strategy: tcx.sess.panic_strategy(),
            plugin_registrar_fn: tcx.sess
                .plugin_registrar_fn
                .get()
                .map(|id| tcx.map.local_def_id(id).index),
            macro_derive_registrar: if is_proc_macro {
                let id = tcx.sess.derive_registrar_fn.get().unwrap();
                Some(tcx.map.local_def_id(id).index)
            } else {
                None
            },

            crate_deps: crate_deps,
            dylib_dependency_formats: dylib_dependency_formats,
            lang_items: lang_items,
            lang_items_missing: lang_items_missing,
            native_libraries: native_libraries,
            codemap: codemap,
            def_path_table: def_path_table,
            impls: impls,
            exported_symbols: exported_symbols,
            index: index,
        });

        let total_bytes = self.position();

        if self.tcx.sess.meta_stats() {
            let mut zero_bytes = 0;
            for e in self.opaque.cursor.get_ref() {
                if *e == 0 {
                    zero_bytes += 1;
                }
            }

            println!("metadata stats:");
            println!("             dep bytes: {}", dep_bytes);
            println!("       lang item bytes: {}", lang_item_bytes);
            println!("          native bytes: {}", native_lib_bytes);
            println!("         codemap bytes: {}", codemap_bytes);
            println!("            impl bytes: {}", impl_bytes);
            println!("    exp. symbols bytes: {}", exported_symbols_bytes);
            println!("  def-path table bytes: {}", def_path_table_bytes);
            println!("            item bytes: {}", item_bytes);
            println!("           index bytes: {}", index_bytes);
            println!("            zero bytes: {}", zero_bytes);
            println!("           total bytes: {}", total_bytes);
        }

        root
    }
}

// NOTE(eddyb) The following comment was preserved for posterity, even
// though it's no longer relevant as EBML (which uses nested & tagged
// "documents") was replaced with a scheme that can't go out of bounds.
//
// And here we run into yet another obscure archive bug: in which metadata
// loaded from archives may have trailing garbage bytes. Awhile back one of
// our tests was failing sporadically on the OSX 64-bit builders (both nopt
// and opt) by having ebml generate an out-of-bounds panic when looking at
// metadata.
//
// Upon investigation it turned out that the metadata file inside of an rlib
// (and ar archive) was being corrupted. Some compilations would generate a
// metadata file which would end in a few extra bytes, while other
// compilations would not have these extra bytes appended to the end. These
// extra bytes were interpreted by ebml as an extra tag, so they ended up
// being interpreted causing the out-of-bounds.
//
// The root cause of why these extra bytes were appearing was never
// discovered, and in the meantime the solution we're employing is to insert
// the length of the metadata to the start of the metadata. Later on this
// will allow us to slice the metadata to the precise length that we just
// generated regardless of trailing bytes that end up in it.

pub fn encode_metadata<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                 cstore: &cstore::CStore,
                                 reexports: &def::ExportMap,
                                 link_meta: &LinkMeta,
                                 exported_symbols: &NodeSet)
                                 -> Vec<u8> {
    let mut cursor = Cursor::new(vec![]);
    cursor.write_all(METADATA_HEADER).unwrap();

    // Will be filed with the root position after encoding everything.
    cursor.write_all(&[0, 0, 0, 0]).unwrap();

    let root = {
        let mut ecx = EncodeContext {
            opaque: opaque::Encoder::new(&mut cursor),
            tcx: tcx,
            reexports: reexports,
            link_meta: link_meta,
            cstore: cstore,
            exported_symbols: exported_symbols,
            lazy_state: LazyState::NoNode,
            type_shorthands: Default::default(),
            predicate_shorthands: Default::default(),
        };

        // Encode the rustc version string in a predictable location.
        rustc_version().encode(&mut ecx).unwrap();

        // Encode all the entries and extra information in the crate,
        // culminating in the `CrateRoot` which points to all of it.
        ecx.encode_crate_root()
    };
    let mut result = cursor.into_inner();

    // Encode the root position.
    let header = METADATA_HEADER.len();
    let pos = root.position;
    result[header + 0] = (pos >> 24) as u8;
    result[header + 1] = (pos >> 16) as u8;
    result[header + 2] = (pos >> 8) as u8;
    result[header + 3] = (pos >> 0) as u8;

    result
}
