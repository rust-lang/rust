// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Decoding metadata from a single crate's metadata

use cstore::{self, CrateMetadata, MetadataBlob, NativeLibrary};
use schema::*;

use rustc::hir::map::{DefKey, DefPath, DefPathData};
use rustc::hir;

use rustc::middle::cstore::LinkagePreference;
use rustc::hir::def::{self, Def, CtorKind};
use rustc::hir::def_id::{CrateNum, DefId, DefIndex, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc::middle::lang_items;
use rustc::session::Session;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::subst::Substs;

use rustc_const_math::ConstInt;

use rustc::mir::Mir;

use std::borrow::Cow;
use std::cell::Ref;
use std::collections::BTreeMap;
use std::io;
use std::mem;
use std::str;
use std::u32;

use rustc_serialize::{Decodable, Decoder, SpecializedDecoder, opaque};
use syntax::attr;
use syntax::ast;
use syntax::codemap;
use syntax_pos::{self, Span, BytePos, Pos, DUMMY_SP};
use rustc_i128::{u128, i128};

pub struct DecodeContext<'a, 'tcx: 'a> {
    opaque: opaque::Decoder<'a>,
    cdata: Option<&'a CrateMetadata>,
    sess: Option<&'a Session>,
    tcx: Option<TyCtxt<'a, 'tcx, 'tcx>>,

    // Cache the last used filemap for translating spans as an optimization.
    last_filemap_index: usize,

    lazy_state: LazyState,
}

/// Abstract over the various ways one can create metadata decoders.
pub trait Metadata<'a, 'tcx>: Copy {
    fn raw_bytes(self) -> &'a [u8];
    fn cdata(self) -> Option<&'a CrateMetadata> { None }
    fn sess(self) -> Option<&'a Session> { None }
    fn tcx(self) -> Option<TyCtxt<'a, 'tcx, 'tcx>> { None }

    fn decoder(self, pos: usize) -> DecodeContext<'a, 'tcx> {
        let tcx = self.tcx();
        DecodeContext {
            opaque: opaque::Decoder::new(self.raw_bytes(), pos),
            cdata: self.cdata(),
            sess: self.sess().or(tcx.map(|tcx| tcx.sess)),
            tcx: tcx,
            last_filemap_index: 0,
            lazy_state: LazyState::NoNode,
        }
    }
}

impl<'a, 'tcx> Metadata<'a, 'tcx> for &'a MetadataBlob {
    fn raw_bytes(self) -> &'a [u8] {
        match *self {
            MetadataBlob::Inflated(ref vec) => vec,
            MetadataBlob::Archive(ref ar) => ar.as_slice(),
            MetadataBlob::Raw(ref vec) => vec,
        }
    }
}

impl<'a, 'tcx> Metadata<'a, 'tcx> for &'a CrateMetadata {
    fn raw_bytes(self) -> &'a [u8] {
        self.blob.raw_bytes()
    }
    fn cdata(self) -> Option<&'a CrateMetadata> {
        Some(self)
    }
}

impl<'a, 'tcx> Metadata<'a, 'tcx> for (&'a CrateMetadata, &'a Session) {
    fn raw_bytes(self) -> &'a [u8] {
        self.0.raw_bytes()
    }
    fn cdata(self) -> Option<&'a CrateMetadata> {
        Some(self.0)
    }
    fn sess(self) -> Option<&'a Session> {
        Some(&self.1)
    }
}

impl<'a, 'tcx> Metadata<'a, 'tcx> for (&'a CrateMetadata, TyCtxt<'a, 'tcx, 'tcx>) {
    fn raw_bytes(self) -> &'a [u8] {
        self.0.raw_bytes()
    }
    fn cdata(self) -> Option<&'a CrateMetadata> {
        Some(self.0)
    }
    fn tcx(self) -> Option<TyCtxt<'a, 'tcx, 'tcx>> {
        Some(self.1)
    }
}

impl<'a, 'tcx: 'a, T: Decodable> Lazy<T> {
    pub fn decode<M: Metadata<'a, 'tcx>>(self, meta: M) -> T {
        let mut dcx = meta.decoder(self.position);
        dcx.lazy_state = LazyState::NodeStart(self.position);
        T::decode(&mut dcx).unwrap()
    }
}

impl<'a, 'tcx: 'a, T: Decodable> LazySeq<T> {
    pub fn decode<M: Metadata<'a, 'tcx>>(self, meta: M) -> impl Iterator<Item = T> + 'a {
        let mut dcx = meta.decoder(self.position);
        dcx.lazy_state = LazyState::NodeStart(self.position);
        (0..self.len).map(move |_| T::decode(&mut dcx).unwrap())
    }
}

impl<'a, 'tcx> DecodeContext<'a, 'tcx> {
    pub fn tcx(&self) -> TyCtxt<'a, 'tcx, 'tcx> {
        self.tcx.expect("missing TyCtxt in DecodeContext")
    }

    pub fn cdata(&self) -> &'a CrateMetadata {
        self.cdata.expect("missing CrateMetadata in DecodeContext")
    }

    fn with_position<F: FnOnce(&mut Self) -> R, R>(&mut self, pos: usize, f: F) -> R {
        let new_opaque = opaque::Decoder::new(self.opaque.data, pos);
        let old_opaque = mem::replace(&mut self.opaque, new_opaque);
        let old_state = mem::replace(&mut self.lazy_state, LazyState::NoNode);
        let r = f(self);
        self.opaque = old_opaque;
        self.lazy_state = old_state;
        r
    }

    fn read_lazy_distance(&mut self, min_size: usize) -> Result<usize, <Self as Decoder>::Error> {
        let distance = self.read_usize()?;
        let position = match self.lazy_state {
            LazyState::NoNode => bug!("read_lazy_distance: outside of a metadata node"),
            LazyState::NodeStart(start) => {
                assert!(distance + min_size <= start);
                start - distance - min_size
            }
            LazyState::Previous(last_min_end) => last_min_end + distance,
        };
        self.lazy_state = LazyState::Previous(position + min_size);
        Ok(position)
    }
}

macro_rules! decoder_methods {
    ($($name:ident -> $ty:ty;)*) => {
        $(fn $name(&mut self) -> Result<$ty, Self::Error> {
            self.opaque.$name()
        })*
    }
}

impl<'doc, 'tcx> Decoder for DecodeContext<'doc, 'tcx> {
    type Error = <opaque::Decoder<'doc> as Decoder>::Error;

    decoder_methods! {
        read_nil -> ();

        read_u128 -> u128;
        read_u64 -> u64;
        read_u32 -> u32;
        read_u16 -> u16;
        read_u8 -> u8;
        read_usize -> usize;

        read_i128 -> i128;
        read_i64 -> i64;
        read_i32 -> i32;
        read_i16 -> i16;
        read_i8 -> i8;
        read_isize -> isize;

        read_bool -> bool;
        read_f64 -> f64;
        read_f32 -> f32;
        read_char -> char;
        read_str -> Cow<str>;
    }

    fn error(&mut self, err: &str) -> Self::Error {
        self.opaque.error(err)
    }
}

impl<'a, 'tcx, T> SpecializedDecoder<Lazy<T>> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<Lazy<T>, Self::Error> {
        Ok(Lazy::with_position(self.read_lazy_distance(Lazy::<T>::min_size())?))
    }
}

impl<'a, 'tcx, T> SpecializedDecoder<LazySeq<T>> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<LazySeq<T>, Self::Error> {
        let len = self.read_usize()?;
        let position = if len == 0 {
            0
        } else {
            self.read_lazy_distance(LazySeq::<T>::min_size(len))?
        };
        Ok(LazySeq::with_position_and_length(position, len))
    }
}

impl<'a, 'tcx> SpecializedDecoder<CrateNum> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<CrateNum, Self::Error> {
        let cnum = CrateNum::from_u32(u32::decode(self)?);
        if cnum == LOCAL_CRATE {
            Ok(self.cdata().cnum)
        } else {
            Ok(self.cdata().cnum_map.borrow()[cnum])
        }
    }
}

impl<'a, 'tcx> SpecializedDecoder<Span> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<Span, Self::Error> {
        let lo = BytePos::decode(self)?;
        let hi = BytePos::decode(self)?;

        let sess = if let Some(sess) = self.sess {
            sess
        } else {
            return Ok(syntax_pos::mk_sp(lo, hi));
        };

        let (lo, hi) = if lo > hi {
            // Currently macro expansion sometimes produces invalid Span values
            // where lo > hi. In order not to crash the compiler when trying to
            // translate these values, let's transform them into something we
            // can handle (and which will produce useful debug locations at
            // least some of the time).
            // This workaround is only necessary as long as macro expansion is
            // not fixed. FIXME(#23480)
            (lo, lo)
        } else {
            (lo, hi)
        };

        let imported_filemaps = self.cdata().imported_filemaps(&sess.codemap());
        let filemap = {
            // Optimize for the case that most spans within a translated item
            // originate from the same filemap.
            let last_filemap = &imported_filemaps[self.last_filemap_index];

            if lo >= last_filemap.original_start_pos && lo <= last_filemap.original_end_pos &&
               hi >= last_filemap.original_start_pos &&
               hi <= last_filemap.original_end_pos {
                last_filemap
            } else {
                let mut a = 0;
                let mut b = imported_filemaps.len();

                while b - a > 1 {
                    let m = (a + b) / 2;
                    if imported_filemaps[m].original_start_pos > lo {
                        b = m;
                    } else {
                        a = m;
                    }
                }

                self.last_filemap_index = a;
                &imported_filemaps[a]
            }
        };

        let lo = (lo - filemap.original_start_pos) + filemap.translated_filemap.start_pos;
        let hi = (hi - filemap.original_start_pos) + filemap.translated_filemap.start_pos;

        Ok(syntax_pos::mk_sp(lo, hi))
    }
}

// FIXME(#36588) These impls are horribly unsound as they allow
// the caller to pick any lifetime for 'tcx, including 'static,
// by using the unspecialized proxies to them.

impl<'a, 'tcx> SpecializedDecoder<Ty<'tcx>> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<Ty<'tcx>, Self::Error> {
        let tcx = self.tcx();

        // Handle shorthands first, if we have an usize > 0x80.
        if self.opaque.data[self.opaque.position()] & 0x80 != 0 {
            let pos = self.read_usize()?;
            assert!(pos >= SHORTHAND_OFFSET);
            let key = ty::CReaderCacheKey {
                cnum: self.cdata().cnum,
                pos: pos - SHORTHAND_OFFSET,
            };
            if let Some(ty) = tcx.rcache.borrow().get(&key).cloned() {
                return Ok(ty);
            }

            let ty = self.with_position(key.pos, Ty::decode)?;
            tcx.rcache.borrow_mut().insert(key, ty);
            Ok(ty)
        } else {
            Ok(tcx.mk_ty(ty::TypeVariants::decode(self)?))
        }
    }
}


impl<'a, 'tcx> SpecializedDecoder<ty::GenericPredicates<'tcx>> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<ty::GenericPredicates<'tcx>, Self::Error> {
        Ok(ty::GenericPredicates {
            parent: Decodable::decode(self)?,
            predicates: (0..self.read_usize()?).map(|_| {
                    // Handle shorthands first, if we have an usize > 0x80.
                    if self.opaque.data[self.opaque.position()] & 0x80 != 0 {
                        let pos = self.read_usize()?;
                        assert!(pos >= SHORTHAND_OFFSET);
                        let pos = pos - SHORTHAND_OFFSET;

                        self.with_position(pos, ty::Predicate::decode)
                    } else {
                        ty::Predicate::decode(self)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl<'a, 'tcx> SpecializedDecoder<&'tcx Substs<'tcx>> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<&'tcx Substs<'tcx>, Self::Error> {
        Ok(self.tcx().mk_substs((0..self.read_usize()?).map(|_| Decodable::decode(self)))?)
    }
}

impl<'a, 'tcx> SpecializedDecoder<&'tcx ty::Region> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<&'tcx ty::Region, Self::Error> {
        Ok(self.tcx().mk_region(Decodable::decode(self)?))
    }
}

impl<'a, 'tcx> SpecializedDecoder<&'tcx ty::Slice<Ty<'tcx>>> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<&'tcx ty::Slice<Ty<'tcx>>, Self::Error> {
        Ok(self.tcx().mk_type_list((0..self.read_usize()?).map(|_| Decodable::decode(self)))?)
    }
}

impl<'a, 'tcx> SpecializedDecoder<&'tcx ty::BareFnTy<'tcx>> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<&'tcx ty::BareFnTy<'tcx>, Self::Error> {
        Ok(self.tcx().mk_bare_fn(Decodable::decode(self)?))
    }
}

impl<'a, 'tcx> SpecializedDecoder<&'tcx ty::AdtDef> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<&'tcx ty::AdtDef, Self::Error> {
        let def_id = DefId::decode(self)?;
        Ok(self.tcx().lookup_adt_def(def_id))
    }
}

impl<'a, 'tcx> SpecializedDecoder<&'tcx ty::Slice<ty::ExistentialPredicate<'tcx>>>
    for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self)
        -> Result<&'tcx ty::Slice<ty::ExistentialPredicate<'tcx>>, Self::Error> {
        Ok(self.tcx().mk_existential_predicates((0..self.read_usize()?)
                                                .map(|_| Decodable::decode(self)))?)
    }
}

impl<'a, 'tcx> MetadataBlob {
    pub fn is_compatible(&self) -> bool {
        self.raw_bytes().starts_with(METADATA_HEADER)
    }

    pub fn get_rustc_version(&self) -> String {
        Lazy::with_position(METADATA_HEADER.len() + 4).decode(self)
    }

    pub fn get_root(&self) -> CrateRoot {
        let slice = self.raw_bytes();
        let offset = METADATA_HEADER.len();
        let pos = (((slice[offset + 0] as u32) << 24) | ((slice[offset + 1] as u32) << 16) |
                   ((slice[offset + 2] as u32) << 8) |
                   ((slice[offset + 3] as u32) << 0)) as usize;
        Lazy::with_position(pos).decode(self)
    }

    pub fn list_crate_metadata(&self, out: &mut io::Write) -> io::Result<()> {
        write!(out, "=External Dependencies=\n")?;
        let root = self.get_root();
        for (i, dep) in root.crate_deps.decode(self).enumerate() {
            write!(out, "{} {}-{}\n", i + 1, dep.name, dep.hash)?;
        }
        write!(out, "\n")?;
        Ok(())
    }
}

impl<'tcx> EntryKind<'tcx> {
    fn to_def(&self, did: DefId) -> Option<Def> {
        Some(match *self {
            EntryKind::Const => Def::Const(did),
            EntryKind::AssociatedConst(_) => Def::AssociatedConst(did),
            EntryKind::ImmStatic |
            EntryKind::ForeignImmStatic => Def::Static(did, false),
            EntryKind::MutStatic |
            EntryKind::ForeignMutStatic => Def::Static(did, true),
            EntryKind::Struct(_) => Def::Struct(did),
            EntryKind::Union(_) => Def::Union(did),
            EntryKind::Fn(_) |
            EntryKind::ForeignFn(_) => Def::Fn(did),
            EntryKind::Method(_) => Def::Method(did),
            EntryKind::Type => Def::TyAlias(did),
            EntryKind::AssociatedType(_) => Def::AssociatedTy(did),
            EntryKind::Mod(_) => Def::Mod(did),
            EntryKind::Variant(_) => Def::Variant(did),
            EntryKind::Trait(_) => Def::Trait(did),
            EntryKind::Enum => Def::Enum(did),
            EntryKind::MacroDef(_) => Def::Macro(did),

            EntryKind::ForeignMod |
            EntryKind::Impl(_) |
            EntryKind::DefaultImpl(_) |
            EntryKind::Field |
            EntryKind::Closure(_) => return None,
        })
    }
}

impl<'a, 'tcx> CrateMetadata {
    fn is_proc_macro(&self, id: DefIndex) -> bool {
        self.proc_macros.is_some() && id != CRATE_DEF_INDEX
    }

    fn maybe_entry(&self, item_id: DefIndex) -> Option<Lazy<Entry<'tcx>>> {
        assert!(!self.is_proc_macro(item_id));
        self.root.index.lookup(self.blob.raw_bytes(), item_id)
    }

    fn entry(&self, item_id: DefIndex) -> Entry<'tcx> {
        match self.maybe_entry(item_id) {
            None => {
                bug!("entry: id not found: {:?} in crate {:?} with number {}",
                     item_id,
                     self.name,
                     self.cnum)
            }
            Some(d) => d.decode(self),
        }
    }

    fn local_def_id(&self, index: DefIndex) -> DefId {
        DefId {
            krate: self.cnum,
            index: index,
        }
    }

    fn item_name(&self, item_index: DefIndex) -> ast::Name {
        self.def_key(item_index)
            .disambiguated_data
            .data
            .get_opt_name()
            .expect("no name in item_name")
    }

    pub fn get_def(&self, index: DefIndex) -> Option<Def> {
        match self.is_proc_macro(index) {
            true => Some(Def::Macro(self.local_def_id(index))),
            false => self.entry(index).kind.to_def(self.local_def_id(index)),
        }
    }

    pub fn get_span(&self, index: DefIndex, sess: &Session) -> Span {
        match self.is_proc_macro(index) {
            true => DUMMY_SP,
            false => self.entry(index).span.decode((self, sess)),
        }
    }

    pub fn get_trait_def(&self,
                         item_id: DefIndex,
                         tcx: TyCtxt<'a, 'tcx, 'tcx>)
                         -> ty::TraitDef {
        let data = match self.entry(item_id).kind {
            EntryKind::Trait(data) => data.decode(self),
            _ => bug!(),
        };

        ty::TraitDef::new(self.local_def_id(item_id),
                          data.unsafety,
                          data.paren_sugar,
                          self.def_path(item_id).deterministic_hash(tcx))
    }

    fn get_variant(&self,
                   item: &Entry<'tcx>,
                   index: DefIndex)
                   -> (ty::VariantDef, Option<DefIndex>) {
        let data = match item.kind {
            EntryKind::Variant(data) |
            EntryKind::Struct(data) |
            EntryKind::Union(data) => data.decode(self),
            _ => bug!(),
        };

        (ty::VariantDef {
            did: self.local_def_id(data.struct_ctor.unwrap_or(index)),
            name: self.item_name(index),
            fields: item.children.decode(self).map(|index| {
                let f = self.entry(index);
                ty::FieldDef {
                    did: self.local_def_id(index),
                    name: self.item_name(index),
                    vis: f.visibility.decode(self)
                }
            }).collect(),
            disr_val: ConstInt::Infer(data.disr),
            ctor_kind: data.ctor_kind,
        }, data.struct_ctor)
    }

    pub fn get_adt_def(&self,
                       item_id: DefIndex,
                       tcx: TyCtxt<'a, 'tcx, 'tcx>)
                       -> &'tcx ty::AdtDef {
        let item = self.entry(item_id);
        let did = self.local_def_id(item_id);
        let mut ctor_index = None;
        let variants = if let EntryKind::Enum = item.kind {
            item.children
                .decode(self)
                .map(|index| {
                    let (variant, struct_ctor) = self.get_variant(&self.entry(index), index);
                    assert_eq!(struct_ctor, None);
                    variant
                })
                .collect()
        } else {
            let (variant, struct_ctor) = self.get_variant(&item, item_id);
            ctor_index = struct_ctor;
            vec![variant]
        };
        let kind = match item.kind {
            EntryKind::Enum => ty::AdtKind::Enum,
            EntryKind::Struct(_) => ty::AdtKind::Struct,
            EntryKind::Union(_) => ty::AdtKind::Union,
            _ => bug!("get_adt_def called on a non-ADT {:?}", did),
        };

        let adt = tcx.alloc_adt_def(did, kind, variants);
        if let Some(ctor_index) = ctor_index {
            // Make adt definition available through constructor id as well.
            tcx.adt_defs.borrow_mut().insert(self.local_def_id(ctor_index), adt);
        }

        adt
    }

    pub fn get_predicates(&self,
                          item_id: DefIndex,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>)
                          -> ty::GenericPredicates<'tcx> {
        self.entry(item_id).predicates.unwrap().decode((self, tcx))
    }

    pub fn get_super_predicates(&self,
                                item_id: DefIndex,
                                tcx: TyCtxt<'a, 'tcx, 'tcx>)
                                -> ty::GenericPredicates<'tcx> {
        match self.entry(item_id).kind {
            EntryKind::Trait(data) => data.decode(self).super_predicates.decode((self, tcx)),
            _ => bug!(),
        }
    }

    pub fn get_generics(&self,
                        item_id: DefIndex,
                        tcx: TyCtxt<'a, 'tcx, 'tcx>)
                        -> ty::Generics<'tcx> {
        self.entry(item_id).generics.unwrap().decode((self, tcx))
    }

    pub fn get_type(&self, id: DefIndex, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Ty<'tcx> {
        self.entry(id).ty.unwrap().decode((self, tcx))
    }

    pub fn get_stability(&self, id: DefIndex) -> Option<attr::Stability> {
        match self.is_proc_macro(id) {
            true => None,
            false => self.entry(id).stability.map(|stab| stab.decode(self)),
        }
    }

    pub fn get_deprecation(&self, id: DefIndex) -> Option<attr::Deprecation> {
        match self.is_proc_macro(id) {
            true => None,
            false => self.entry(id).deprecation.map(|depr| depr.decode(self)),
        }
    }

    pub fn get_visibility(&self, id: DefIndex) -> ty::Visibility {
        match self.is_proc_macro(id) {
            true => ty::Visibility::Public,
            false => self.entry(id).visibility.decode(self),
        }
    }

    fn get_impl_data(&self, id: DefIndex) -> ImplData<'tcx> {
        match self.entry(id).kind {
            EntryKind::Impl(data) => data.decode(self),
            _ => bug!(),
        }
    }

    pub fn get_parent_impl(&self, id: DefIndex) -> Option<DefId> {
        self.get_impl_data(id).parent_impl
    }

    pub fn get_impl_polarity(&self, id: DefIndex) -> hir::ImplPolarity {
        self.get_impl_data(id).polarity
    }

    pub fn get_custom_coerce_unsized_kind(&self,
                                          id: DefIndex)
                                          -> Option<ty::adjustment::CustomCoerceUnsized> {
        self.get_impl_data(id).coerce_unsized_kind
    }

    pub fn get_impl_trait(&self,
                          id: DefIndex,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>)
                          -> Option<ty::TraitRef<'tcx>> {
        self.get_impl_data(id).trait_ref.map(|tr| tr.decode((self, tcx)))
    }

    /// Iterates over the language items in the given crate.
    pub fn get_lang_items(&self) -> Vec<(DefIndex, usize)> {
        self.root.lang_items.decode(self).collect()
    }

    /// Iterates over each child of the given item.
    pub fn each_child_of_item<F>(&self, id: DefIndex, mut callback: F)
        where F: FnMut(def::Export)
    {
        if let Some(ref proc_macros) = self.proc_macros {
            if id == CRATE_DEF_INDEX {
                for (id, &(name, _)) in proc_macros.iter().enumerate() {
                    let def = Def::Macro(DefId { krate: self.cnum, index: DefIndex::new(id + 1) });
                    callback(def::Export { name: name, def: def });
                }
            }
            return
        }

        // Find the item.
        let item = match self.maybe_entry(id) {
            None => return,
            Some(item) => item.decode(self),
        };

        // Iterate over all children.
        let macros_only = self.dep_kind.get().macros_only();
        for child_index in item.children.decode(self) {
            if macros_only {
                continue
            }

            // Get the item.
            if let Some(child) = self.maybe_entry(child_index) {
                let child = child.decode(self);
                match child.kind {
                    EntryKind::MacroDef(..) => {}
                    _ if macros_only => continue,
                    _ => {}
                }

                // Hand off the item to the callback.
                match child.kind {
                    // FIXME(eddyb) Don't encode these in children.
                    EntryKind::ForeignMod => {
                        for child_index in child.children.decode(self) {
                            if let Some(def) = self.get_def(child_index) {
                                callback(def::Export {
                                    def: def,
                                    name: self.item_name(child_index),
                                });
                            }
                        }
                        continue;
                    }
                    EntryKind::Impl(_) |
                    EntryKind::DefaultImpl(_) => continue,

                    _ => {}
                }

                let def_key = self.def_key(child_index);
                if let (Some(def), Some(name)) =
                    (self.get_def(child_index), def_key.disambiguated_data.data.get_opt_name()) {
                    callback(def::Export {
                        def: def,
                        name: name,
                    });
                    // For non-reexport structs and variants add their constructors to children.
                    // Reexport lists automatically contain constructors when necessary.
                    match def {
                        Def::Struct(..) => {
                            if let Some(ctor_def_id) = self.get_struct_ctor_def_id(child_index) {
                                let ctor_kind = self.get_ctor_kind(child_index);
                                let ctor_def = Def::StructCtor(ctor_def_id, ctor_kind);
                                callback(def::Export {
                                    def: ctor_def,
                                    name: name,
                                });
                            }
                        }
                        Def::Variant(def_id) => {
                            // Braced variants, unlike structs, generate unusable names in
                            // value namespace, they are reserved for possible future use.
                            let ctor_kind = self.get_ctor_kind(child_index);
                            let ctor_def = Def::VariantCtor(def_id, ctor_kind);
                            callback(def::Export {
                                def: ctor_def,
                                name: name,
                            });
                        }
                        _ => {}
                    }
                }
            }
        }

        if let EntryKind::Mod(data) = item.kind {
            for exp in data.decode(self).reexports.decode(self) {
                match exp.def {
                    Def::Macro(..) => {}
                    _ if macros_only => continue,
                    _ => {}
                }
                callback(exp);
            }
        }
    }

    pub fn maybe_get_item_body(&self,
                               tcx: TyCtxt<'a, 'tcx, 'tcx>,
                               id: DefIndex)
                               -> Option<&'tcx hir::Body> {
        if self.is_proc_macro(id) { return None; }
        self.entry(id).ast.map(|ast| {
            let def_id = self.local_def_id(id);
            let ast = ast.decode(self);

            let tables = ast.tables.decode((self, tcx));
            tcx.tables.borrow_mut().insert(def_id, tcx.alloc_tables(tables));

            let body = ast.body.decode((self, tcx));
            tcx.map.intern_inlined_body(def_id, body)
        })
    }

    pub fn item_body_nested_bodies(&self, id: DefIndex) -> BTreeMap<hir::BodyId, hir::Body> {
        self.entry(id).ast.into_iter().flat_map(|ast| {
            ast.decode(self).nested_bodies.decode(self).map(|body| (body.id(), body))
        }).collect()
    }

    pub fn const_is_rvalue_promotable_to_static(&self, id: DefIndex) -> bool {
        self.entry(id).ast.expect("const item missing `ast`")
            .decode(self).rvalue_promotable_to_static
    }

    pub fn is_item_mir_available(&self, id: DefIndex) -> bool {
        !self.is_proc_macro(id) &&
        self.maybe_entry(id).and_then(|item| item.decode(self).mir).is_some()
    }

    pub fn maybe_get_item_mir(&self,
                              tcx: TyCtxt<'a, 'tcx, 'tcx>,
                              id: DefIndex)
                              -> Option<Mir<'tcx>> {
        match self.is_proc_macro(id) {
            true => None,
            false => self.entry(id).mir.map(|mir| mir.decode((self, tcx))),
        }
    }

    pub fn get_associated_item(&self, id: DefIndex) -> Option<ty::AssociatedItem> {
        let item = self.entry(id);
        let parent_and_name = || {
            let def_key = self.def_key(id);
            (self.local_def_id(def_key.parent.unwrap()),
             def_key.disambiguated_data.data.get_opt_name().unwrap())
        };

        Some(match item.kind {
            EntryKind::AssociatedConst(container) => {
                let (parent, name) = parent_and_name();
                ty::AssociatedItem {
                    name: name,
                    kind: ty::AssociatedKind::Const,
                    vis: item.visibility.decode(self),
                    defaultness: container.defaultness(),
                    def_id: self.local_def_id(id),
                    container: container.with_def_id(parent),
                    method_has_self_argument: false
                }
            }
            EntryKind::Method(data) => {
                let (parent, name) = parent_and_name();
                let data = data.decode(self);
                ty::AssociatedItem {
                    name: name,
                    kind: ty::AssociatedKind::Method,
                    vis: item.visibility.decode(self),
                    defaultness: data.container.defaultness(),
                    def_id: self.local_def_id(id),
                    container: data.container.with_def_id(parent),
                    method_has_self_argument: data.has_self
                }
            }
            EntryKind::AssociatedType(container) => {
                let (parent, name) = parent_and_name();
                ty::AssociatedItem {
                    name: name,
                    kind: ty::AssociatedKind::Type,
                    vis: item.visibility.decode(self),
                    defaultness: container.defaultness(),
                    def_id: self.local_def_id(id),
                    container: container.with_def_id(parent),
                    method_has_self_argument: false
                }
            }
            _ => return None,
        })
    }

    pub fn get_item_variances(&self, id: DefIndex) -> Vec<ty::Variance> {
        self.entry(id).variances.decode(self).collect()
    }

    pub fn get_ctor_kind(&self, node_id: DefIndex) -> CtorKind {
        match self.entry(node_id).kind {
            EntryKind::Struct(data) |
            EntryKind::Union(data) |
            EntryKind::Variant(data) => data.decode(self).ctor_kind,
            _ => CtorKind::Fictive,
        }
    }

    pub fn get_struct_ctor_def_id(&self, node_id: DefIndex) -> Option<DefId> {
        match self.entry(node_id).kind {
            EntryKind::Struct(data) => {
                data.decode(self).struct_ctor.map(|index| self.local_def_id(index))
            }
            _ => None,
        }
    }

    pub fn get_item_attrs(&self, node_id: DefIndex) -> Vec<ast::Attribute> {
        if self.is_proc_macro(node_id) {
            return Vec::new();
        }
        // The attributes for a tuple struct are attached to the definition, not the ctor;
        // we assume that someone passing in a tuple struct ctor is actually wanting to
        // look at the definition
        let mut item = self.entry(node_id);
        let def_key = self.def_key(node_id);
        if def_key.disambiguated_data.data == DefPathData::StructCtor {
            item = self.entry(def_key.parent.unwrap());
        }
        self.get_attributes(&item)
    }

    pub fn get_struct_field_names(&self, id: DefIndex) -> Vec<ast::Name> {
        self.entry(id)
            .children
            .decode(self)
            .map(|index| self.item_name(index))
            .collect()
    }

    fn get_attributes(&self, item: &Entry<'tcx>) -> Vec<ast::Attribute> {
        item.attributes
            .decode(self)
            .map(|mut attr| {
                // Need new unique IDs: old thread-local IDs won't map to new threads.
                attr.id = attr::mk_attr_id();
                attr
            })
            .collect()
    }

    // Translate a DefId from the current compilation environment to a DefId
    // for an external crate.
    fn reverse_translate_def_id(&self, did: DefId) -> Option<DefId> {
        for (local, &global) in self.cnum_map.borrow().iter_enumerated() {
            if global == did.krate {
                return Some(DefId {
                    krate: local,
                    index: did.index,
                });
            }
        }

        None
    }

    pub fn get_inherent_implementations_for_type(&self, id: DefIndex) -> Vec<DefId> {
        self.entry(id)
            .inherent_impls
            .decode(self)
            .map(|index| self.local_def_id(index))
            .collect()
    }

    pub fn get_implementations_for_trait(&self, filter: Option<DefId>, result: &mut Vec<DefId>) {
        // Do a reverse lookup beforehand to avoid touching the crate_num
        // hash map in the loop below.
        let filter = match filter.map(|def_id| self.reverse_translate_def_id(def_id)) {
            Some(Some(def_id)) => Some((def_id.krate.as_u32(), def_id.index)),
            Some(None) => return,
            None if self.proc_macros.is_some() => return,
            None => None,
        };

        // FIXME(eddyb) Make this O(1) instead of O(n).
        for trait_impls in self.root.impls.decode(self) {
            if filter.is_some() && filter != Some(trait_impls.trait_id) {
                continue;
            }

            result.extend(trait_impls.impls.decode(self).map(|index| self.local_def_id(index)));

            if filter.is_some() {
                break;
            }
        }
    }

    pub fn get_trait_of_item(&self, id: DefIndex) -> Option<DefId> {
        self.def_key(id).parent.and_then(|parent_index| {
            match self.entry(parent_index).kind {
                EntryKind::Trait(_) => Some(self.local_def_id(parent_index)),
                _ => None,
            }
        })
    }


    pub fn get_native_libraries(&self) -> Vec<NativeLibrary> {
        self.root.native_libraries.decode(self).collect()
    }

    pub fn get_dylib_dependency_formats(&self) -> Vec<(CrateNum, LinkagePreference)> {
        self.root
            .dylib_dependency_formats
            .decode(self)
            .enumerate()
            .flat_map(|(i, link)| {
                let cnum = CrateNum::new(i + 1);
                link.map(|link| (self.cnum_map.borrow()[cnum], link))
            })
            .collect()
    }

    pub fn get_missing_lang_items(&self) -> Vec<lang_items::LangItem> {
        self.root.lang_items_missing.decode(self).collect()
    }

    pub fn get_fn_arg_names(&self, id: DefIndex) -> Vec<ast::Name> {
        let arg_names = match self.entry(id).kind {
            EntryKind::Fn(data) |
            EntryKind::ForeignFn(data) => data.decode(self).arg_names,
            EntryKind::Method(data) => data.decode(self).fn_data.arg_names,
            _ => LazySeq::empty(),
        };
        arg_names.decode(self).collect()
    }

    pub fn get_exported_symbols(&self) -> Vec<DefId> {
        self.exported_symbols.iter().map(|&index| self.local_def_id(index)).collect()
    }

    pub fn get_macro(&self, id: DefIndex) -> (ast::Name, MacroDef) {
        let entry = self.entry(id);
        match entry.kind {
            EntryKind::MacroDef(macro_def) => (self.item_name(id), macro_def.decode(self)),
            _ => bug!(),
        }
    }

    pub fn is_const_fn(&self, id: DefIndex) -> bool {
        let constness = match self.entry(id).kind {
            EntryKind::Method(data) => data.decode(self).fn_data.constness,
            EntryKind::Fn(data) => data.decode(self).constness,
            _ => hir::Constness::NotConst,
        };
        constness == hir::Constness::Const
    }

    pub fn is_foreign_item(&self, id: DefIndex) -> bool {
        match self.entry(id).kind {
            EntryKind::ForeignImmStatic |
            EntryKind::ForeignMutStatic |
            EntryKind::ForeignFn(_) => true,
            _ => false,
        }
    }

    pub fn is_dllimport_foreign_item(&self, id: DefIndex) -> bool {
        self.dllimport_foreign_items.contains(&id)
    }

    pub fn is_defaulted_trait(&self, trait_id: DefIndex) -> bool {
        match self.entry(trait_id).kind {
            EntryKind::Trait(data) => data.decode(self).has_default_impl,
            _ => bug!(),
        }
    }

    pub fn is_default_impl(&self, impl_id: DefIndex) -> bool {
        match self.entry(impl_id).kind {
            EntryKind::DefaultImpl(_) => true,
            _ => false,
        }
    }

    pub fn closure_kind(&self, closure_id: DefIndex) -> ty::ClosureKind {
        match self.entry(closure_id).kind {
            EntryKind::Closure(data) => data.decode(self).kind,
            _ => bug!(),
        }
    }

    pub fn closure_ty(&self,
                      closure_id: DefIndex,
                      tcx: TyCtxt<'a, 'tcx, 'tcx>)
                      -> ty::ClosureTy<'tcx> {
        match self.entry(closure_id).kind {
            EntryKind::Closure(data) => data.decode(self).ty.decode((self, tcx)),
            _ => bug!(),
        }
    }

    pub fn def_key(&self, index: DefIndex) -> DefKey {
        self.def_path_table.def_key(index)
    }

    // Returns the path leading to the thing with this `id`.
    pub fn def_path(&self, id: DefIndex) -> DefPath {
        debug!("def_path(id={:?})", id);
        DefPath::make(self.cnum, id, |parent| self.def_path_table.def_key(parent))
    }

    /// Imports the codemap from an external crate into the codemap of the crate
    /// currently being compiled (the "local crate").
    ///
    /// The import algorithm works analogous to how AST items are inlined from an
    /// external crate's metadata:
    /// For every FileMap in the external codemap an 'inline' copy is created in the
    /// local codemap. The correspondence relation between external and local
    /// FileMaps is recorded in the `ImportedFileMap` objects returned from this
    /// function. When an item from an external crate is later inlined into this
    /// crate, this correspondence information is used to translate the span
    /// information of the inlined item so that it refers the correct positions in
    /// the local codemap (see `<decoder::DecodeContext as SpecializedDecoder<Span>>`).
    ///
    /// The import algorithm in the function below will reuse FileMaps already
    /// existing in the local codemap. For example, even if the FileMap of some
    /// source file of libstd gets imported many times, there will only ever be
    /// one FileMap object for the corresponding file in the local codemap.
    ///
    /// Note that imported FileMaps do not actually contain the source code of the
    /// file they represent, just information about length, line breaks, and
    /// multibyte characters. This information is enough to generate valid debuginfo
    /// for items inlined from other crates.
    pub fn imported_filemaps(&'a self,
                             local_codemap: &codemap::CodeMap)
                             -> Ref<'a, Vec<cstore::ImportedFileMap>> {
        {
            let filemaps = self.codemap_import_info.borrow();
            if !filemaps.is_empty() {
                return filemaps;
            }
        }

        let external_codemap = self.root.codemap.decode(self);

        let imported_filemaps = external_codemap.map(|filemap_to_import| {
                // Try to find an existing FileMap that can be reused for the filemap to
                // be imported. A FileMap is reusable if it is exactly the same, just
                // positioned at a different offset within the codemap.
                let reusable_filemap = {
                    local_codemap.files
                        .borrow()
                        .iter()
                        .find(|fm| are_equal_modulo_startpos(&fm, &filemap_to_import))
                        .map(|rc| rc.clone())
                };

                match reusable_filemap {
                    Some(fm) => {

                        debug!("CrateMetaData::imported_filemaps reuse \
                                filemap {:?} original (start_pos {:?} end_pos {:?}) \
                                translated (start_pos {:?} end_pos {:?})",
                               filemap_to_import.name,
                               filemap_to_import.start_pos, filemap_to_import.end_pos,
                               fm.start_pos, fm.end_pos);

                        cstore::ImportedFileMap {
                            original_start_pos: filemap_to_import.start_pos,
                            original_end_pos: filemap_to_import.end_pos,
                            translated_filemap: fm,
                        }
                    }
                    None => {
                        // We can't reuse an existing FileMap, so allocate a new one
                        // containing the information we need.
                        let syntax_pos::FileMap { name,
                                                  abs_path,
                                                  start_pos,
                                                  end_pos,
                                                  lines,
                                                  multibyte_chars,
                                                  .. } = filemap_to_import;

                        let source_length = (end_pos - start_pos).to_usize();

                        // Translate line-start positions and multibyte character
                        // position into frame of reference local to file.
                        // `CodeMap::new_imported_filemap()` will then translate those
                        // coordinates to their new global frame of reference when the
                        // offset of the FileMap is known.
                        let mut lines = lines.into_inner();
                        for pos in &mut lines {
                            *pos = *pos - start_pos;
                        }
                        let mut multibyte_chars = multibyte_chars.into_inner();
                        for mbc in &mut multibyte_chars {
                            mbc.pos = mbc.pos - start_pos;
                        }

                        let local_version = local_codemap.new_imported_filemap(name,
                                                                               abs_path,
                                                                               source_length,
                                                                               lines,
                                                                               multibyte_chars);
                        debug!("CrateMetaData::imported_filemaps alloc \
                                filemap {:?} original (start_pos {:?} end_pos {:?}) \
                                translated (start_pos {:?} end_pos {:?})",
                               local_version.name, start_pos, end_pos,
                               local_version.start_pos, local_version.end_pos);

                        cstore::ImportedFileMap {
                            original_start_pos: start_pos,
                            original_end_pos: end_pos,
                            translated_filemap: local_version,
                        }
                    }
                }
            })
            .collect();

        // This shouldn't borrow twice, but there is no way to downgrade RefMut to Ref.
        *self.codemap_import_info.borrow_mut() = imported_filemaps;
        self.codemap_import_info.borrow()
    }
}

fn are_equal_modulo_startpos(fm1: &syntax_pos::FileMap, fm2: &syntax_pos::FileMap) -> bool {
    if fm1.byte_length() != fm2.byte_length() {
        return false;
    }

    if fm1.name != fm2.name {
        return false;
    }

    let lines1 = fm1.lines.borrow();
    let lines2 = fm2.lines.borrow();

    if lines1.len() != lines2.len() {
        return false;
    }

    for (&line1, &line2) in lines1.iter().zip(lines2.iter()) {
        if (line1 - fm1.start_pos) != (line2 - fm2.start_pos) {
            return false;
        }
    }

    let multibytes1 = fm1.multibyte_chars.borrow();
    let multibytes2 = fm2.multibyte_chars.borrow();

    if multibytes1.len() != multibytes2.len() {
        return false;
    }

    for (mb1, mb2) in multibytes1.iter().zip(multibytes2.iter()) {
        if (mb1.bytes != mb2.bytes) || ((mb1.pos - fm1.start_pos) != (mb2.pos - fm2.start_pos)) {
            return false;
        }
    }

    true
}
