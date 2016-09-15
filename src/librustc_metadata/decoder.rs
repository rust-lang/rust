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

use astencode::decode_inlined_item;
use cstore::{CrateMetadata, MetadataBlob, NativeLibraryKind};
use common::*;
use index;

use rustc::hir::svh::Svh;
use rustc::hir::map as hir_map;
use rustc::hir::map::{DefKey, DefPathData};
use rustc::util::nodemap::FnvHashMap;
use rustc::hir;
use rustc::hir::intravisit::IdRange;

use rustc::middle::cstore::{InlinedItem, LinkagePreference};
use rustc::hir::def::{self, Def};
use rustc::hir::def_id::{CrateNum, DefId, DefIndex, LOCAL_CRATE};
use rustc::middle::lang_items;
use rustc::ty::{ImplContainer, TraitContainer};
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::subst::Substs;

use rustc_const_math::ConstInt;

use rustc::mir::repr::Mir;

use std::io;
use std::mem;
use std::rc::Rc;
use std::str;
use std::u32;

use rbml;
use rustc_serialize::{Decodable, Decoder, SpecializedDecoder, opaque};
use syntax::attr;
use syntax::ast::{self, NodeId};
use syntax::parse::token;
use syntax_pos::{self, Span, BytePos};

pub struct DecodeContext<'a, 'tcx: 'a> {
    pub opaque: opaque::Decoder<'a>,
    tcx: Option<TyCtxt<'a, 'tcx, 'tcx>>,
    cdata: Option<&'a CrateMetadata>,
    pub from_id_range: IdRange,
    pub to_id_range: IdRange,
    // Cache the last used filemap for translating spans as an optimization.
    last_filemap_index: usize,
}

impl<'a, 'tcx> DecodeContext<'a, 'tcx> {
    pub fn new(doc: rbml::Doc<'a>, cdata: Option<&'a CrateMetadata>)
               -> DecodeContext<'a, 'tcx> {
        let id_range = IdRange {
            min: NodeId::from_u32(u32::MIN),
            max: NodeId::from_u32(u32::MAX)
        };
        DecodeContext {
            opaque: opaque::Decoder::new(doc.data, doc.start),
            cdata: cdata,
            tcx: None,
            from_id_range: id_range,
            to_id_range: id_range,
            last_filemap_index: 0
        }
    }

    pub fn tcx(&self) -> TyCtxt<'a, 'tcx, 'tcx> {
        self.tcx.expect("missing TyCtxt in DecodeContext")
    }

    pub fn cdata(&self) -> &'a CrateMetadata {
        self.cdata.expect("missing CrateMetadata in DecodeContext")
    }

    pub fn decode<T: Decodable>(&mut self) -> T {
        T::decode(self).unwrap()
    }

    pub fn typed(mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Self {
        self.tcx = Some(tcx);
        self
    }

    /// Iterate over the indices of a sequence.
    /// This will work solely because of `serialize::opaque`'s
    /// simple encoding of `n: usize` followed by `n` elements.
    pub fn seq<T: Decodable>(mut self) -> impl Iterator<Item=T> {
        (0..self.read_usize().unwrap()).map(move |_| {
            self.decode()
        })
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

        read_u64 -> u64;
        read_u32 -> u32;
        read_u16 -> u16;
        read_u8 -> u8;
        read_usize -> usize;

        read_i64 -> i64;
        read_i32 -> i32;
        read_i16 -> i16;
        read_i8 -> i8;
        read_isize -> isize;

        read_bool -> bool;
        read_f64 -> f64;
        read_f32 -> f32;
        read_char -> char;
        read_str -> String;
    }

    fn error(&mut self, err: &str) -> Self::Error {
        self.opaque.error(err)
    }
}

impl<'a, 'tcx> SpecializedDecoder<NodeId> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<NodeId, Self::Error> {
        let id = u32::decode(self)?;

        // from_id_range should be non-empty
        assert!(!self.from_id_range.empty());
        // Make sure that translating the NodeId will actually yield a
        // meaningful result
        if !self.from_id_range.contains(NodeId::from_u32(id)) {
            bug!("NodeId::decode: {} out of DecodeContext range ({:?} -> {:?})",
                 id, self.from_id_range, self.to_id_range);
        }

        // Use wrapping arithmetic because otherwise it introduces control flow.
        // Maybe we should just have the control flow? -- aatch
        Ok(NodeId::from_u32(id.wrapping_sub(self.from_id_range.min.as_u32())
                              .wrapping_add(self.to_id_range.min.as_u32())))
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

        let tcx = if let Some(tcx) = self.tcx {
            tcx
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

        let imported_filemaps = self.cdata().imported_filemaps(&tcx.sess.codemap());
        let filemap = {
            // Optimize for the case that most spans within a translated item
            // originate from the same filemap.
            let last_filemap = &imported_filemaps[self.last_filemap_index];

            if lo >= last_filemap.original_start_pos &&
            lo <= last_filemap.original_end_pos &&
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

        let lo = (lo - filemap.original_start_pos) +
                  filemap.translated_filemap.start_pos;
        let hi = (hi - filemap.original_start_pos) +
                  filemap.translated_filemap.start_pos;

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
                pos: pos - SHORTHAND_OFFSET
            };
            if let Some(ty) = tcx.rcache.borrow().get(&key).cloned() {
                return Ok(ty);
            }

            let new = opaque::Decoder::new(self.opaque.data, key.pos);
            let old = mem::replace(&mut self.opaque, new);
            let ty = Ty::decode(self)?;
            self.opaque = old;
            tcx.rcache.borrow_mut().insert(key, ty);
            return Ok(ty);
        }

        Ok(tcx.mk_ty(ty::TypeVariants::decode(self)?))
    }
}

impl<'a, 'tcx> SpecializedDecoder<&'tcx Substs<'tcx>> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<&'tcx Substs<'tcx>, Self::Error> {
        Ok(self.tcx().mk_substs(Decodable::decode(self)?))
    }
}

impl<'a, 'tcx> SpecializedDecoder<&'tcx ty::Region> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<&'tcx ty::Region, Self::Error> {
        Ok(self.tcx().mk_region(Decodable::decode(self)?))
    }
}

impl<'a, 'tcx> SpecializedDecoder<&'tcx ty::Slice<Ty<'tcx>>> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<&'tcx ty::Slice<Ty<'tcx>>, Self::Error> {
        Ok(self.tcx().mk_type_list(Decodable::decode(self)?))
    }
}

impl<'a, 'tcx> SpecializedDecoder<&'tcx ty::BareFnTy<'tcx>> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<&'tcx ty::BareFnTy<'tcx>, Self::Error> {
        Ok(self.tcx().mk_bare_fn(Decodable::decode(self)?))
    }
}

impl<'a, 'tcx> SpecializedDecoder<ty::AdtDef<'tcx>> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<ty::AdtDef<'tcx>, Self::Error> {
        let def_id = DefId::decode(self)?;
        Ok(self.tcx().lookup_adt_def(def_id))
    }
}

#[derive(Clone)]
pub struct CrateDep {
    pub cnum: CrateNum,
    pub name: String,
    pub hash: Svh,
    pub explicitly_linked: bool,
}

impl<'a, 'tcx> MetadataBlob {
    fn root(&self) -> rbml::Doc {
        rbml::Doc::new(self.as_slice())
    }

    fn child_at(&'a self, pos: usize, tag: usize) -> DecodeContext<'a, 'tcx> {
        DecodeContext::new(rbml::Doc::at(self.as_slice(), pos).child(tag), None)
    }

    fn get(&'a self, tag: usize) -> DecodeContext<'a, 'tcx> {
        DecodeContext::new(self.root().child(tag), None)
    }

    pub fn load_index(&self) -> index::Index {
        index::Index::from_rbml(self.root().child(root_tag::index))
    }

    pub fn crate_rustc_version(&self) -> Option<String> {
        self.root().maybe_child(root_tag::rustc_version).map(|s| {
            str::from_utf8(&s.data[s.start..s.end]).unwrap().to_string()
        })
    }

    // Go through each item in the metadata and create a map from that
    // item's def-key to the item's DefIndex.
    pub fn load_key_map(&self) -> FnvHashMap<DefKey, DefIndex> {
        self.load_index().iter_enumerated(self.as_slice()).map(|(index, pos)| {
            (self.child_at(pos as usize, item_tag::def_key).decode(), index)
        }).collect()
    }

    pub fn get_crate_deps(&self) -> Vec<CrateDep> {
        let dcx = self.get(root_tag::crate_deps);

        dcx.seq().enumerate().map(|(crate_num, (name, hash, explicitly_linked))| {
            CrateDep {
                cnum: CrateNum::new(crate_num + 1),
                name: name,
                hash: hash,
                explicitly_linked: explicitly_linked,
            }
        }).collect()
    }

    pub fn get_crate_info(&self) -> CrateInfo {
        self.get(root_tag::crate_info).decode()
    }

    pub fn list_crate_metadata(&self, out: &mut io::Write) -> io::Result<()> {
        write!(out, "=External Dependencies=\n")?;
        for dep in &self.get_crate_deps() {
            write!(out, "{} {}-{}\n", dep.cnum, dep.name, dep.hash)?;
        }
        write!(out, "\n")?;
        Ok(())
    }

    pub fn get_imported_filemaps(&self) -> Vec<syntax_pos::FileMap> {
        self.get(root_tag::codemap).decode()
    }

    pub fn each_exported_macro<F>(&self, mut f: F) where
        F: FnMut(ast::Name, Vec<ast::Attribute>, Span, String) -> bool,
    {
        for (name, attrs, span, body) in self.get(root_tag::macro_defs).seq() {
            if !f(name, attrs, span, body) {
                break;
            }
        }
    }
}

impl Family {
    fn to_def(&self, did: DefId) -> Option<Def> {
        Some(match *self {
            Family::Const  => Def::Const(did),
            Family::AssociatedConst => Def::AssociatedConst(did),
            Family::ImmStatic | Family::ForeignImmStatic => Def::Static(did, false),
            Family::MutStatic | Family::ForeignMutStatic => Def::Static(did, true),
            Family::Struct => Def::Struct(did),
            Family::Union => Def::Union(did),
            Family::Fn | Family::ForeignFn  => Def::Fn(did),
            Family::Method => Def::Method(did),
            Family::Type => Def::TyAlias(did),
            Family::AssociatedType => Def::AssociatedTy(did),
            Family::Mod => Def::Mod(did),
            Family::Variant => Def::Variant(did),
            Family::Trait => Def::Trait(did),
            Family::Enum => Def::Enum(did),

            Family::ForeignMod |
            Family::Impl |
            Family::DefaultImpl |
            Family::Field |
            Family::Closure => {
                return None
            }
        })
    }
}

impl<'a, 'tcx> CrateMetadata {
    fn maybe_get(&'a self, item: rbml::Doc<'a>, tag: usize)
                 -> Option<DecodeContext<'a, 'tcx>> {
        item.maybe_child(tag).map(|child| {
            DecodeContext::new(child, Some(self))
        })
    }

    fn get(&'a self, item: rbml::Doc<'a>, tag: usize) -> DecodeContext<'a, 'tcx> {
        match self.maybe_get(item, tag) {
            Some(dcx) => dcx,
            None => bug!("failed to find child with tag {}", tag)
        }
    }

    fn item_family(&self, item: rbml::Doc) -> Family {
        self.get(item, item_tag::family).decode()
    }

    fn item_visibility(&self, item: rbml::Doc) -> ty::Visibility {
        self.get(item, item_tag::visibility).decode()
    }

    fn item_def_key(&self, item: rbml::Doc) -> hir_map::DefKey {
        self.get(item, item_tag::def_key).decode()
    }

    fn item_name(&self, item: rbml::Doc) -> ast::Name {
        self.maybe_item_name(item).expect("no item in item_name")
    }

    fn maybe_item_name(&self, item: rbml::Doc) -> Option<ast::Name> {
        let name = match self.item_def_key(item).disambiguated_data.data {
            DefPathData::TypeNs(name) |
            DefPathData::ValueNs(name) |
            DefPathData::Module(name) |
            DefPathData::MacroDef(name) |
            DefPathData::TypeParam(name) |
            DefPathData::LifetimeDef(name) |
            DefPathData::EnumVariant(name) |
            DefPathData::Field(name) |
            DefPathData::Binding(name) => Some(name),

            DefPathData::InlinedRoot(_) => bug!("unexpected DefPathData"),

            DefPathData::CrateRoot |
            DefPathData::Misc |
            DefPathData::Impl |
            DefPathData::ClosureExpr |
            DefPathData::StructCtor |
            DefPathData::Initializer |
            DefPathData::ImplTrait => None
        };

        name.map(|s| token::intern(&s))
    }

    fn maybe_entry(&self, item_id: DefIndex) -> Option<rbml::Doc> {
        self.index.lookup_item(self.data.as_slice(), item_id).map(|pos| {
            rbml::Doc::at(self.data.as_slice(), pos as usize)
        })
    }

    fn entry(&self, item_id: DefIndex) -> rbml::Doc {
        match self.maybe_entry(item_id) {
            None => bug!("entry: id not found: {:?} in crate {:?} with number {}",
                         item_id,
                         self.name,
                         self.cnum),
            Some(d) => d
        }
    }

    fn local_def_id(&self, index: DefIndex) -> DefId {
        DefId {
            krate: self.cnum,
            index: index
        }
    }

    fn entry_data(&self, doc: rbml::Doc) -> EntryData {
        self.get(doc, item_tag::data).decode()
    }

    fn entry_typed_data(&self, doc: rbml::Doc, tcx: TyCtxt<'a, 'tcx, 'tcx>)
                        -> EntryTypedData<'tcx> {
        self.get(doc, item_tag::typed_data).typed(tcx).decode()
    }

    fn item_parent_item(&self, d: rbml::Doc) -> Option<DefId> {
        self.item_def_key(d).parent.map(|index| self.local_def_id(index))
    }

    fn doc_type(&self, doc: rbml::Doc, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Ty<'tcx> {
        self.maybe_doc_type(doc, tcx).expect("missing item_tag::ty")
    }

    fn maybe_doc_type(&self, doc: rbml::Doc, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Option<Ty<'tcx>> {
        self.maybe_get(doc, item_tag::ty).map(|dcx| dcx.typed(tcx).decode())
    }

    pub fn get_def(&self, index: DefIndex) -> Option<Def> {
        self.item_family(self.entry(index)).to_def(self.local_def_id(index))
    }

    pub fn get_trait_def(&self,
                         item_id: DefIndex,
                         tcx: TyCtxt<'a, 'tcx, 'tcx>) -> ty::TraitDef<'tcx> {
        let item_doc = self.entry(item_id);
        let generics = self.doc_generics(item_doc, tcx);

        let data = match self.entry_data(item_doc) {
            EntryData::Trait(data) => data,
            _ => bug!()
        };
        let typed_data = match self.entry_typed_data(item_doc, tcx) {
            EntryTypedData::Trait(data) => data,
            _ => bug!()
        };

        ty::TraitDef::new(data.unsafety, data.paren_sugar, generics, typed_data.trait_ref,
                          self.def_path(item_id).unwrap().deterministic_hash(tcx)))
    }

    fn get_variant(&self, item: rbml::Doc, index: DefIndex)
                  -> (ty::VariantDefData<'tcx, 'tcx>, Option<DefIndex>) {
        let data = match self.entry_data(item) {
            EntryData::Variant(data) => data,
            _ => bug!()
        };

        let fields = self.get(item, item_tag::children).seq().map(|index| {
            let f = self.entry(index);
            ty::FieldDefData::new(self.local_def_id(index),
                                  self.item_name(f),
                                  self.item_visibility(f))
        }).collect();

        (ty::VariantDefData {
            did: self.local_def_id(data.struct_ctor.unwrap_or(index)),
            name: self.item_name(item),
            fields: fields,
            disr_val: ConstInt::Infer(data.disr),
            kind: data.kind,
        }, data.struct_ctor)
    }

    pub fn get_adt_def(&self, item_id: DefIndex, tcx: TyCtxt<'a, 'tcx, 'tcx>)
                       -> ty::AdtDefMaster<'tcx> {
        let doc = self.entry(item_id);
        let did = self.local_def_id(item_id);
        let mut ctor_index = None;
        let family = self.item_family(doc);
        let variants = if family == Family::Enum {
            self.get(doc, item_tag::children).seq().map(|index| {
                let (variant, struct_ctor) = self.get_variant(self.entry(index), index);
                assert_eq!(struct_ctor, None);
                variant
            }).collect()
        } else{
            let (variant, struct_ctor) = self.get_variant(doc, item_id);
            ctor_index = struct_ctor;
            vec![variant]
        };
        let kind = match family {
            Family::Enum => ty::AdtKind::Enum,
            Family::Struct => ty::AdtKind::Struct,
            Family::Union => ty::AdtKind::Union,
            _ => bug!("get_adt_def called on a non-ADT {:?} - {:?}",
                      family, did)
        };

        let adt = tcx.intern_adt_def(did, kind, variants);
        if let Some(ctor_index) = ctor_index {
            // Make adt definition available through constructor id as well.
            tcx.insert_adt_def(self.local_def_id(ctor_index), adt);
        }

        // this needs to be done *after* the variant is interned,
        // to support recursive structures
        for variant in &adt.variants {
            for field in &variant.fields {
                debug!("evaluating the type of {:?}::{:?}", variant.name, field.name);
                let ty = self.get_type(field.did.index, tcx);
                field.fulfill_ty(ty);
                debug!("evaluating the type of {:?}::{:?}: {:?}",
                       variant.name, field.name, ty);
            }
        }

        adt
    }

    pub fn get_predicates(&self, item_id: DefIndex, tcx: TyCtxt<'a, 'tcx, 'tcx>)
                          -> ty::GenericPredicates<'tcx> {
        self.doc_predicates(self.entry(item_id), tcx, item_tag::predicates)
    }

    pub fn get_super_predicates(&self, item_id: DefIndex, tcx: TyCtxt<'a, 'tcx, 'tcx>)
                                -> ty::GenericPredicates<'tcx> {
        self.doc_predicates(self.entry(item_id), tcx, item_tag::super_predicates)
    }

    pub fn get_generics(&self, item_id: DefIndex, tcx: TyCtxt<'a, 'tcx, 'tcx>)
                        -> &'tcx ty::Generics<'tcx> {
        self.doc_generics(self.entry(item_id), tcx)
    }

    pub fn get_type(&self, id: DefIndex, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> Ty<'tcx> {
        self.doc_type(self.entry(id), tcx)
    }

    pub fn get_stability(&self, id: DefIndex) -> Option<attr::Stability> {
        self.maybe_get(self.entry(id), item_tag::stability).map(|mut dcx| {
            dcx.decode()
        })
    }

    pub fn get_deprecation(&self, id: DefIndex) -> Option<attr::Deprecation> {
        self.maybe_get(self.entry(id), item_tag::deprecation).map(|mut dcx| {
            dcx.decode()
        })
    }

    pub fn get_visibility(&self, id: DefIndex) -> ty::Visibility {
        self.item_visibility(self.entry(id))
    }

    fn get_impl_data(&self, id: DefIndex) -> ImplData {
        match self.entry_data(self.entry(id)) {
            EntryData::Impl(data) => data,
            _ => bug!()
        }
    }

    pub fn get_parent_impl(&self, id: DefIndex) -> Option<DefId> {
        self.get_impl_data(id).parent_impl
    }

    pub fn get_impl_polarity(&self, id: DefIndex) -> hir::ImplPolarity {
        self.get_impl_data(id).polarity
    }

    pub fn get_custom_coerce_unsized_kind(&self, id: DefIndex)
                                          -> Option<ty::adjustment::CustomCoerceUnsized> {
        self.get_impl_data(id).coerce_unsized_kind
    }

    pub fn get_impl_trait(&self,
                          id: DefIndex,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>)
                          -> Option<ty::TraitRef<'tcx>> {
        match self.entry_typed_data(self.entry(id), tcx) {
            EntryTypedData::Impl(data) => data.trait_ref,
            _ => bug!()
        }
    }

    /// Iterates over the language items in the given crate.
    pub fn get_lang_items(&self) -> Vec<(DefIndex, usize)> {
        self.get(self.data.root(), root_tag::lang_items).decode()
    }

    /// Iterates over each child of the given item.
    pub fn each_child_of_item<F>(&self, id: DefIndex, mut callback: F)
        where F: FnMut(def::Export)
    {
        // Find the item.
        let item_doc = match self.maybe_entry(id) {
            None => return,
            Some(item_doc) => item_doc,
        };

        let dcx = match self.maybe_get(item_doc, item_tag::children) {
            Some(dcx) => dcx,
            None => return
        };

        // Iterate over all children.
        for child_index in dcx.seq() {
            // Get the item.
            if let Some(child) = self.maybe_entry(child_index) {
                // Hand off the item to the callback.
                match self.item_family(child) {
                    // FIXME(eddyb) Don't encode these in children.
                    Family::ForeignMod => {
                        for child_index in self.get(child, item_tag::children).seq() {
                            callback(def::Export {
                                def_id: self.local_def_id(child_index),
                                name: self.item_name(self.entry(child_index))
                            });
                        }
                        continue;
                    }
                    Family::Impl | Family::DefaultImpl => continue,

                    _ => {}
                }

                if let Some(name) = self.maybe_item_name(child) {
                    callback(def::Export {
                        def_id: self.local_def_id(child_index),
                        name: name
                    });
                }
            }
        }

        let reexports = match self.entry_data(item_doc) {
            EntryData::Mod(data) => data.reexports,
            _ => return
        };
        for exp in reexports {
            callback(exp);
        }
    }

    pub fn maybe_get_item_name(&self, id: DefIndex) -> Option<ast::Name> {
        self.maybe_item_name(self.entry(id))
    }

    pub fn maybe_get_item_ast(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, id: DefIndex)
                              -> Option<&'tcx InlinedItem> {
        debug!("Looking up item: {:?}", id);
        let item_doc = self.entry(id);
        let item_did = self.local_def_id(id);
        let parent_def_id = self.local_def_id(self.def_key(id).parent.unwrap());
        let mut parent_def_path = self.def_path(id).unwrap();
        parent_def_path.data.pop();
        item_doc.maybe_child(item_tag::ast).map(|ast_doc| {
            decode_inlined_item(self, tcx, parent_def_path, parent_def_id, ast_doc, item_did)
        })
    }

    pub fn is_item_mir_available(&self, id: DefIndex) -> bool {
        if let Some(item_doc) = self.maybe_entry(id) {
            return item_doc.maybe_child(item_tag::mir).is_some();
        }

        false
    }

    pub fn maybe_get_item_mir(&self, tcx: TyCtxt<'a, 'tcx, 'tcx>, id: DefIndex)
                              -> Option<Mir<'tcx>> {
        self.maybe_get(self.entry(id), item_tag::mir).map(|dcx| {
            dcx.typed(tcx).decode()
        })
    }

    pub fn get_impl_or_trait_item(&self, id: DefIndex, tcx: TyCtxt<'a, 'tcx, 'tcx>)
                                  -> Option<ty::ImplOrTraitItem<'tcx>> {
        let item_doc = self.entry(id);
        let family = self.item_family(item_doc);

        match family {
            Family::AssociatedConst |
            Family::Method |
            Family::AssociatedType => {}

            _ => return None
        }

        let def_id = self.local_def_id(id);

        let container_id = self.item_parent_item(item_doc).unwrap();
        let container = match self.item_family(self.entry(container_id.index)) {
            Family::Trait => TraitContainer(container_id),
            _ => ImplContainer(container_id),
        };

        let name = self.item_name(item_doc);
        let vis = self.item_visibility(item_doc);

        let (defaultness, has_body) = match self.entry_data(item_doc) {
            EntryData::TraitAssociated(data) => {
                (hir::Defaultness::Default, data.has_default)
            }
            EntryData::ImplAssociated(data) => {
                (data.defaultness, true)
            }
            _ => bug!()
        };

        Some(match family {
            Family::AssociatedConst => {
                ty::ConstTraitItem(Rc::new(ty::AssociatedConst {
                    name: name,
                    ty: self.doc_type(item_doc, tcx),
                    vis: vis,
                    defaultness: defaultness,
                    def_id: def_id,
                    container: container,
                    has_value: has_body,
                }))
            }
            Family::Method => {
                let generics = self.doc_generics(item_doc, tcx);
                let predicates = self.doc_predicates(item_doc, tcx, item_tag::predicates);
                let ity = tcx.lookup_item_type(def_id).ty;
                let fty = match ity.sty {
                    ty::TyFnDef(.., fty) => fty,
                    _ => bug!(
                        "the type {:?} of the method {:?} is not a function?",
                        ity, name)
                };

                let explicit_self = match self.entry_typed_data(item_doc, tcx) {
                    EntryTypedData::Method(data) => data.explicit_self,
                    _ => bug!()
                };
                ty::MethodTraitItem(Rc::new(ty::Method {
                    name: name,
                    generics: generics,
                    predicates: predicates,
                    fty: fty,
                    explicit_self: explicit_self,
                    vis: vis,
                    defaultness: defaultness,
                    has_body: has_body,
                    def_id: def_id,
                    container: container,
                }))
            }
            Family::AssociatedType => {
                ty::TypeTraitItem(Rc::new(ty::AssociatedType {
                    name: name,
                    ty: self.maybe_doc_type(item_doc, tcx),
                    vis: vis,
                    defaultness: defaultness,
                    def_id: def_id,
                    container: container,
                }))
            }
            _ => bug!()
        })
    }

    pub fn get_item_variances(&self, id: DefIndex) -> Vec<ty::Variance> {
        let item_doc = self.entry(id);
        self.get(item_doc, item_tag::variances).decode()
    }

    pub fn get_struct_ctor_def_id(&self, node_id: DefIndex) -> Option<DefId> {
        let data = match self.entry_data(self.entry(node_id)) {
            EntryData::Variant(data) => data,
            _ => bug!()
        };

        data.struct_ctor.map(|index| self.local_def_id(index))
    }

    pub fn get_item_attrs(&self, node_id: DefIndex) -> Vec<ast::Attribute> {
        // The attributes for a tuple struct are attached to the definition, not the ctor;
        // we assume that someone passing in a tuple struct ctor is actually wanting to
        // look at the definition
        let mut item = self.entry(node_id);
        let def_key = self.item_def_key(item);
        if def_key.disambiguated_data.data == DefPathData::StructCtor {
            item = self.entry(def_key.parent.unwrap());
        }
        self.get_attributes(item)
    }

    pub fn get_struct_field_names(&self, id: DefIndex) -> Vec<ast::Name> {
        self.get(self.entry(id), item_tag::children).seq().map(|index| {
            self.item_name(self.entry(index))
        }).collect()
    }

    fn get_attributes(&self, md: rbml::Doc) -> Vec<ast::Attribute> {
        self.maybe_get(md, item_tag::attributes).map_or(vec![], |mut dcx| {
            let mut attrs = dcx.decode::<Vec<ast::Attribute>>();

            // Need new unique IDs: old thread-local IDs won't map to new threads.
            for attr in attrs.iter_mut() {
                attr.node.id = attr::mk_attr_id();
            }

            attrs
        })
    }

    // Translate a DefId from the current compilation environment to a DefId
    // for an external crate.
    fn reverse_translate_def_id(&self, did: DefId) -> Option<DefId> {
        for (local, &global) in self.cnum_map.borrow().iter_enumerated() {
            if global == did.krate {
                return Some(DefId { krate: local, index: did.index });
            }
        }

        None
    }

    pub fn each_inherent_implementation_for_type<F>(&self, id: DefIndex, mut callback: F)
        where F: FnMut(DefId),
    {
        for impl_def_id in self.get(self.entry(id), item_tag::inherent_impls).seq() {
            callback(impl_def_id);
        }
    }

    pub fn each_implementation_for_trait<F>(&self,
                                            filter: Option<DefId>,
                                            mut callback: F) where
        F: FnMut(DefId),
    {
        // Do a reverse lookup beforehand to avoid touching the crate_num
        // hash map in the loop below.
        let filter = match filter.map(|def_id| self.reverse_translate_def_id(def_id)) {
            Some(Some(def_id)) => Some(def_id),
            Some(None) => return,
            None => None
        };

        // FIXME(eddyb) Make this O(1) instead of O(n).
        for trait_doc in self.data.root().children_of(root_tag::impls) {
            let mut dcx = DecodeContext::new(trait_doc, Some(self));

            let (krate, index) = dcx.decode();
            if let Some(local_did) = filter {
                if (local_did.krate.as_u32(), local_did.index) != (krate, index) {
                    continue;
                }
            }

            for impl_def_id in dcx.seq() {
                callback(impl_def_id);
            }
        }
    }

    pub fn get_trait_of_item(&self, id: DefIndex) -> Option<DefId> {
        let item_doc = self.entry(id);
        let parent_item_id = match self.item_parent_item(item_doc) {
            None => return None,
            Some(item_id) => item_id,
        };
        match self.item_family(self.entry(parent_item_id.index)) {
            Family::Trait => Some(parent_item_id),
            _ => None
        }
    }


    pub fn get_native_libraries(&self) -> Vec<(NativeLibraryKind, String)> {
        self.get(self.data.root(), root_tag::native_libraries).decode()
    }

    pub fn get_dylib_dependency_formats(&self) -> Vec<(CrateNum, LinkagePreference)> {
        let dcx = self.get(self.data.root(), root_tag::dylib_dependency_formats);

        dcx.seq::<Option<_>>().enumerate().flat_map(|(i, link)| {
            let cnum = CrateNum::new(i + 1);
            link.map(|link| (self.cnum_map.borrow()[cnum], link))
        }).collect()
    }

    pub fn get_missing_lang_items(&self) -> Vec<lang_items::LangItem> {
        self.get(self.data.root(), root_tag::lang_items_missing).decode()
    }

    pub fn get_fn_arg_names(&self, id: DefIndex) -> Vec<String> {
        self.maybe_get(self.entry(id), item_tag::fn_arg_names)
            .map_or(vec![], |mut dcx| dcx.decode())
    }

    pub fn get_reachable_ids(&self) -> Vec<DefId> {
        let dcx = self.get(self.data.root(), root_tag::reachable_ids);

        dcx.seq().map(|index| self.local_def_id(index)).collect()
    }

    pub fn is_const_fn(&self, id: DefIndex) -> bool {
        let constness = match self.entry_data(self.entry(id)) {
            EntryData::ImplAssociated(data) => data.constness,
            EntryData::Fn(data) => data.constness,
            _ => hir::Constness::NotConst
        };
        constness == hir::Constness::Const
    }

    pub fn is_extern_item(&self, id: DefIndex, tcx: TyCtxt<'a, 'tcx, 'tcx>) -> bool {
        let item_doc = match self.maybe_entry(id) {
            Some(doc) => doc,
            None => return false,
        };
        let applicable = match self.item_family(item_doc) {
            Family::ImmStatic |
            Family::MutStatic |
            Family::ForeignImmStatic |
            Family::ForeignMutStatic => true,

            Family::Fn | Family::ForeignFn => {
                self.get_generics(id, tcx).types.is_empty()
            }

            _ => false,
        };

        if applicable {
            attr::contains_extern_indicator(tcx.sess.diagnostic(),
                                            &self.get_attributes(item_doc))
        } else {
            false
        }
    }

    pub fn is_foreign_item(&self, id: DefIndex) -> bool {
        match self.item_family(self.entry(id)) {
            Family::ForeignImmStatic |
            Family::ForeignMutStatic |
            Family::ForeignFn => true,
            _ => false
        }
    }

    fn doc_generics(&self, base_doc: rbml::Doc, tcx: TyCtxt<'a, 'tcx, 'tcx>)
                    -> &'tcx ty::Generics<'tcx> {
        let generics = self.get(base_doc, item_tag::generics).typed(tcx).decode();
        tcx.alloc_generics(generics)
    }

    fn doc_predicates(&self, base_doc: rbml::Doc, tcx: TyCtxt<'a, 'tcx, 'tcx>, tag: usize)
                      -> ty::GenericPredicates<'tcx> {
        let mut dcx = self.get(base_doc, tag).typed(tcx);

        ty::GenericPredicates {
            parent: dcx.decode(),
            predicates: (0..dcx.decode::<usize>()).map(|_| {
                // Handle shorthands first, if we have an usize > 0x80.
                if dcx.opaque.data[dcx.opaque.position()] & 0x80 != 0 {
                    let pos = dcx.decode::<usize>();
                    assert!(pos >= SHORTHAND_OFFSET);
                    let pos = pos - SHORTHAND_OFFSET;

                    let data = self.data.as_slice();
                    let doc = rbml::Doc {
                        data: data,
                        start: pos,
                        end: data.len(),
                    };
                    DecodeContext::new(doc, Some(self)).typed(tcx).decode()
                } else {
                    dcx.decode()
                }
            }).collect()
        }
    }

    pub fn is_defaulted_trait(&self, trait_id: DefIndex) -> bool {
        match self.entry_data(self.entry(trait_id)) {
            EntryData::Trait(data) => data.has_default_impl,
            _ => bug!()
        }
    }

    pub fn is_default_impl(&self, impl_id: DefIndex) -> bool {
        self.item_family(self.entry(impl_id)) == Family::DefaultImpl
    }

    pub fn closure_kind(&self, closure_id: DefIndex) -> ty::ClosureKind {
        match self.entry_data(self.entry(closure_id)) {
            EntryData::Closure(data) => data.kind,
            _ => bug!()
        }
    }

    pub fn closure_ty(&self, closure_id: DefIndex, tcx: TyCtxt<'a, 'tcx, 'tcx>)
                      -> ty::ClosureTy<'tcx> {
        match self.entry_typed_data(self.entry(closure_id), tcx) {
            EntryTypedData::Closure(data) => data.ty,
            _ => bug!()
        }
    }

    pub fn def_key(&self, id: DefIndex) -> hir_map::DefKey {
        debug!("def_key: id={:?}", id);
        self.item_def_key(self.entry(id))
    }

    // Returns the path leading to the thing with this `id`. Note that
    // some def-ids don't wind up in the metadata, so `def_path` sometimes
    // returns `None`
    pub fn def_path(&self, id: DefIndex) -> Option<hir_map::DefPath> {
        debug!("def_path(id={:?})", id);
        if self.maybe_entry(id).is_some() {
            Some(hir_map::DefPath::make(self.cnum, id, |parent| self.def_key(parent)))
        } else {
            None
        }
    }
}
