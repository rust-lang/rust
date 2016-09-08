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

#![allow(non_camel_case_types)]

use astencode::decode_inlined_item;
use cstore::{self, CrateMetadata};
use common::*;
use index;

use rustc::hir::def_id::CRATE_DEF_INDEX;
use rustc::hir::svh::Svh;
use rustc::hir::map as hir_map;
use rustc::hir::map::{DefKey, DefPathData};
use rustc::util::nodemap::FnvHashMap;
use rustc::hir;
use rustc::hir::intravisit::IdRange;

use rustc::middle::cstore::{InlinedItem, LinkagePreference};
use rustc::hir::def::Def;
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

use rbml::reader;
use rbml;
use rustc_serialize::{Decodable, Decoder, SpecializedDecoder, opaque};
use syntax::attr;
use syntax::ast::{self, NodeId};
use syntax::parse::token;
use syntax_pos::{self, Span, BytePos};

pub struct DecodeContext<'a, 'tcx: 'a> {
    pub opaque: opaque::Decoder<'a>,
    pub tcx: Option<TyCtxt<'a, 'tcx, 'tcx>>,
    pub cdata: Option<&'a cstore::CrateMetadata>,
    pub from_id_range: IdRange,
    pub to_id_range: IdRange,
    // Cache the last used filemap for translating spans as an optimization.
    last_filemap_index: usize,
}

impl<'doc> rbml::Doc<'doc> {
    pub fn decoder<'tcx>(self) -> DecodeContext<'doc, 'tcx> {
        let id_range = IdRange {
            min: NodeId::from_u32(u32::MIN),
            max: NodeId::from_u32(u32::MAX)
        };
        DecodeContext {
            opaque: opaque::Decoder::new(self.data, self.start),
            cdata: None,
            tcx: None,
            from_id_range: id_range,
            to_id_range: id_range,
            last_filemap_index: 0
        }
    }
}

impl<'a, 'tcx> DecodeContext<'a, 'tcx> {
    pub fn tcx(&self) -> TyCtxt<'a, 'tcx, 'tcx> {
        self.tcx.expect("missing TyCtxt in DecodeContext")
    }

    pub fn cdata(&self) -> &'a cstore::CrateMetadata {
        self.cdata.expect("missing CrateMetadata in DecodeContext")
    }

    pub fn decode<T: Decodable>(&mut self) -> T {
        T::decode(self).unwrap()
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

pub type Cmd<'a> = &'a CrateMetadata;

impl CrateMetadata {
    fn get_item(&self, item_id: DefIndex) -> Option<rbml::Doc> {
        self.index.lookup_item(self.data(), item_id).map(|pos| {
            rbml::Doc::at(self.data(), pos as usize)
        })
    }

    fn lookup_item(&self, item_id: DefIndex) -> rbml::Doc {
        match self.get_item(item_id) {
            None => bug!("lookup_item: id not found: {:?} in crate {:?} with number {}",
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
}

pub fn load_index(data: &[u8]) -> index::Index {
    index::Index::from_rbml(rbml::Doc::new(data).get(root_tag::index))
}

pub fn crate_rustc_version(data: &[u8]) -> Option<String> {
    let doc = rbml::Doc::new(data);
    reader::maybe_get_doc(doc, root_tag::rustc_version).map(|s| {
        str::from_utf8(&s.data[s.start..s.end]).unwrap().to_string()
    })
}

// Go through each item in the metadata and create a map from that
// item's def-key to the item's DefIndex.
pub fn load_key_map(data: &[u8]) -> FnvHashMap<DefKey, DefIndex> {
    load_index(data).iter_enumerated(data).map(|(index, pos)| {
        // load def-key from item
        let key = item_def_key(rbml::Doc::at(data, pos as usize));
        (key, index)
    }).collect()
}

fn item_family(item: rbml::Doc) -> Family {
    item.get(item_tag::family).decoder().decode()
}

fn item_visibility(item: rbml::Doc) -> ty::Visibility {
    item.get(item_tag::visibility).decoder().decode()
}

fn entry_data(doc: rbml::Doc, cdata: Cmd) -> EntryData {
    let mut dcx = doc.get(item_tag::data).decoder();
    dcx.cdata = Some(cdata);

    dcx.decode()
}

fn entry_typed_data<'a, 'tcx>(doc: rbml::Doc, tcx: TyCtxt<'a, 'tcx, 'tcx>, cdata: Cmd)
                              -> EntryTypedData<'tcx> {
    let mut dcx = doc.get(item_tag::typed_data).decoder();
    dcx.cdata = Some(cdata);
    dcx.tcx = Some(tcx);

    dcx.decode()
}

fn item_parent_item(cdata: Cmd, d: rbml::Doc) -> Option<DefId> {
    item_def_key(d).parent.map(|index| cdata.local_def_id(index))
}

fn doc_type<'a, 'tcx>(doc: rbml::Doc, tcx: TyCtxt<'a, 'tcx, 'tcx>, cdata: Cmd) -> Ty<'tcx> {
    maybe_doc_type(doc, tcx, cdata).expect("missing item_tag::ty")
}

fn maybe_doc_type<'a, 'tcx>(doc: rbml::Doc, tcx: TyCtxt<'a, 'tcx, 'tcx>, cdata: Cmd)
                            -> Option<Ty<'tcx>> {
    reader::maybe_get_doc(doc, item_tag::ty).map(|tp| {
        let mut dcx = tp.decoder();
        dcx.tcx = Some(tcx);
        dcx.cdata = Some(cdata);
        dcx.decode()
    })
}

fn item_name(item: rbml::Doc) -> ast::Name {
    maybe_item_name(item).expect("no item in item_name")
}

fn maybe_item_name(item: rbml::Doc) -> Option<ast::Name> {
    let name = match item_def_key(item).disambiguated_data.data {
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

pub fn get_trait_def<'a, 'tcx>(cdata: Cmd,
                               item_id: DefIndex,
                               tcx: TyCtxt<'a, 'tcx, 'tcx>) -> ty::TraitDef<'tcx>
{
    let item_doc = cdata.lookup_item(item_id);
    let generics = doc_generics(item_doc, tcx, cdata);

    let data = match entry_data(item_doc, cdata) {
        EntryData::Trait(data) => data,
        _ => bug!()
    };
    let typed_data = match entry_typed_data(item_doc, tcx, cdata) {
        EntryTypedData::Trait(data) => data,
        _ => bug!()
    };

    ty::TraitDef::new(data.unsafety, data.paren_sugar, generics, typed_data.trait_ref,
                      def_path(cdata, item_id).unwrap().deterministic_hash(tcx)))
}

fn get_variant<'tcx>(cdata: Cmd,
                     item: rbml::Doc,
                     index: DefIndex)
                     -> (ty::VariantDefData<'tcx, 'tcx>, Option<DefIndex>) {
    let data = match entry_data(item, cdata) {
        EntryData::Variant(data) => data,
        _ => bug!()
    };

    let mut dcx = item.get(item_tag::children).decoder();
    dcx.cdata = Some(cdata);

    let fields = dcx.seq().map(|index| {
        let f = cdata.lookup_item(index);
        ty::FieldDefData::new(cdata.local_def_id(index),
                              item_name(f),
                              item_visibility(f))
    }).collect();

    (ty::VariantDefData {
        did: cdata.local_def_id(data.struct_ctor.unwrap_or(index)),
        name: item_name(item),
        fields: fields,
        disr_val: ConstInt::Infer(data.disr),
        kind: data.kind,
    }, data.struct_ctor)
}

pub fn get_adt_def<'a, 'tcx>(cdata: Cmd,
                             item_id: DefIndex,
                             tcx: TyCtxt<'a, 'tcx, 'tcx>)
                             -> ty::AdtDefMaster<'tcx>
{
    let doc = cdata.lookup_item(item_id);
    let did = cdata.local_def_id(item_id);
    let mut ctor_index = None;
    let family = item_family(doc);
    let variants = if family == Family::Enum {
        let mut dcx = doc.get(item_tag::children).decoder();
        dcx.cdata = Some(cdata);

        dcx.seq().map(|index| {
            let (variant, struct_ctor) = get_variant(cdata, cdata.lookup_item(index), index);
            assert_eq!(struct_ctor, None);
            variant
        }).collect()
    } else{
        let (variant, struct_ctor) = get_variant(cdata, doc, item_id);
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
        tcx.insert_adt_def(cdata.local_def_id(ctor_index), adt);
    }

    // this needs to be done *after* the variant is interned,
    // to support recursive structures
    for variant in &adt.variants {
        for field in &variant.fields {
            debug!("evaluating the type of {:?}::{:?}", variant.name, field.name);
            let ty = get_type(cdata, field.did.index, tcx);
            field.fulfill_ty(ty);
            debug!("evaluating the type of {:?}::{:?}: {:?}",
                   variant.name, field.name, ty);
        }
    }

    adt
}

pub fn get_predicates<'a, 'tcx>(cdata: Cmd,
                                item_id: DefIndex,
                                tcx: TyCtxt<'a, 'tcx, 'tcx>)
                                -> ty::GenericPredicates<'tcx>
{
    let item_doc = cdata.lookup_item(item_id);
    doc_predicates(item_doc, tcx, cdata, item_tag::predicates)
}

pub fn get_super_predicates<'a, 'tcx>(cdata: Cmd,
                                      item_id: DefIndex,
                                      tcx: TyCtxt<'a, 'tcx, 'tcx>)
                                      -> ty::GenericPredicates<'tcx>
{
    let item_doc = cdata.lookup_item(item_id);
    doc_predicates(item_doc, tcx, cdata, item_tag::super_predicates)
}

pub fn get_generics<'a, 'tcx>(cdata: Cmd,
                              item_id: DefIndex,
                              tcx: TyCtxt<'a, 'tcx, 'tcx>)
                              -> &'tcx ty::Generics<'tcx>
{
    let item_doc = cdata.lookup_item(item_id);
    doc_generics(item_doc, tcx, cdata)
}

pub fn get_type<'a, 'tcx>(cdata: Cmd, id: DefIndex, tcx: TyCtxt<'a, 'tcx, 'tcx>)
                          -> Ty<'tcx>
{
    let item_doc = cdata.lookup_item(id);
    doc_type(item_doc, tcx, cdata)
}

pub fn get_stability(cdata: Cmd, id: DefIndex) -> Option<attr::Stability> {
    let item = cdata.lookup_item(id);
    reader::maybe_get_doc(item, item_tag::stability).map(|doc| {
        doc.decoder().decode()
    })
}

pub fn get_deprecation(cdata: Cmd, id: DefIndex) -> Option<attr::Deprecation> {
    let item = cdata.lookup_item(id);
    reader::maybe_get_doc(item, item_tag::deprecation).map(|doc| {
        doc.decoder().decode()
    })
}

pub fn get_visibility(cdata: Cmd, id: DefIndex) -> ty::Visibility {
    item_visibility(cdata.lookup_item(id))
}

fn get_impl_data(cdata: Cmd, id: DefIndex) -> ImplData {
    match entry_data(cdata.lookup_item(id), cdata) {
        EntryData::Impl(data) => data,
        _ => bug!()
    }
}

pub fn get_parent_impl(cdata: Cmd, id: DefIndex) -> Option<DefId> {
    get_impl_data(cdata, id).parent_impl
}

pub fn get_impl_polarity(cdata: Cmd, id: DefIndex) -> hir::ImplPolarity {
    get_impl_data(cdata, id).polarity
}

pub fn get_custom_coerce_unsized_kind(
    cdata: Cmd,
    id: DefIndex)
    -> Option<ty::adjustment::CustomCoerceUnsized>
{
    get_impl_data(cdata, id).coerce_unsized_kind
}

pub fn get_impl_trait<'a, 'tcx>(cdata: Cmd,
                                id: DefIndex,
                                tcx: TyCtxt<'a, 'tcx, 'tcx>)
                                -> Option<ty::TraitRef<'tcx>>
{
    match entry_typed_data(cdata.lookup_item(id), tcx, cdata) {
        EntryTypedData::Impl(data) => data.trait_ref,
        _ => bug!()
    }
}

/// Iterates over the language items in the given crate.
pub fn get_lang_items(cdata: Cmd) -> Vec<(DefIndex, usize)> {
    rbml::Doc::new(cdata.data()).get(root_tag::lang_items).decoder().decode()
}


/// Iterates over each child of the given item.
pub fn each_child_of_item<F, G>(cdata: Cmd, id: DefIndex,
                                mut get_crate_data: &mut G,
                                mut callback: &mut F)
    where F: FnMut(Def, ast::Name, ty::Visibility),
          G: FnMut(CrateNum) -> Rc<CrateMetadata>,
{
    // Find the item.
    let item_doc = match cdata.get_item(id) {
        None => return,
        Some(item_doc) => item_doc,
    };

    let mut dcx = match reader::maybe_get_doc(item_doc, item_tag::children) {
        Some(doc) => doc.decoder(),
        None => return
    };
    dcx.cdata = Some(cdata);

    // Iterate over all children.
    for child_index in dcx.seq::<DefIndex>() {
        // Get the item.
        if let Some(child) = cdata.get_item(child_index) {
            // Hand off the item to the callback.
            let family = item_family(child);
            if let Family::ForeignMod = family {
                each_child_of_item(cdata, child_index, get_crate_data, callback);
            } else if let Some(def) = family.to_def(cdata.local_def_id(child_index)) {
                callback(def, item_name(child), item_visibility(child));
            }
        }
    }

    let reexports = match entry_data(item_doc, cdata) {
        EntryData::Mod(data) => data.reexports,
        _ => return
    };
    for exp in reexports {
        // This reexport may be in yet another crate.
        let crate_data = if exp.def_id.krate == cdata.cnum {
            None
        } else {
            Some(get_crate_data(exp.def_id.krate))
        };
        let crate_data = match crate_data {
            Some(ref cdata) => &**cdata,
            None => cdata
        };

        // Get the item.
        if let Some(child) = crate_data.get_item(exp.def_id.index) {
            // Hand off the item to the callback.
            if let Some(def) = item_family(child).to_def(exp.def_id) {
                // These items have a public visibility because they're part of
                // a public re-export.
                callback(def, exp.name, ty::Visibility::Public);
            }
        }
    }
}

pub fn maybe_get_item_name(cdata: Cmd, id: DefIndex) -> Option<ast::Name> {
    maybe_item_name(cdata.lookup_item(id))
}

pub fn maybe_get_item_ast<'a, 'tcx>(cdata: Cmd, tcx: TyCtxt<'a, 'tcx, 'tcx>, id: DefIndex)
                                    -> Option<&'tcx InlinedItem> {
    debug!("Looking up item: {:?}", id);
    let item_doc = cdata.lookup_item(id);
    let item_did = cdata.local_def_id(id);
    let parent_def_id = cdata.local_def_id(def_key(cdata, id).parent.unwrap());
    let mut parent_def_path = def_path(cdata, id).unwrap();
    parent_def_path.data.pop();
    reader::maybe_get_doc(item_doc, item_tag::ast).map(|ast_doc| {
        decode_inlined_item(cdata, tcx, parent_def_path, parent_def_id, ast_doc, item_did)
    })
}

pub fn is_item_mir_available<'tcx>(cdata: Cmd, id: DefIndex) -> bool {
    if let Some(item_doc) = cdata.get_item(id) {
        return reader::maybe_get_doc(item_doc, item_tag::mir as usize).is_some();
    }

    false
}

pub fn maybe_get_item_mir<'a, 'tcx>(cdata: Cmd,
                                    tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                    id: DefIndex)
                                    -> Option<Mir<'tcx>> {
    let item_doc = cdata.lookup_item(id);

    reader::maybe_get_doc(item_doc, item_tag::mir).map(|mir_doc| {
        let mut dcx = mir_doc.decoder();
        dcx.tcx = Some(tcx);
        dcx.cdata = Some(cdata);
        dcx.decode()
    })
}

pub fn get_impl_or_trait_item<'a, 'tcx>(cdata: Cmd, id: DefIndex, tcx: TyCtxt<'a, 'tcx, 'tcx>)
                                        -> Option<ty::ImplOrTraitItem<'tcx>> {
    let item_doc = cdata.lookup_item(id);
    let family = item_family(item_doc);

    match family {
        Family::AssociatedConst |
        Family::Method |
        Family::AssociatedType => {}

        _ => return None
    }

    let def_id = cdata.local_def_id(id);

    let container_id = item_parent_item(cdata, item_doc).unwrap();
    let container = match item_family(cdata.lookup_item(container_id.index)) {
        Family::Trait => TraitContainer(container_id),
        _ => ImplContainer(container_id),
    };

    let name = item_name(item_doc);
    let vis = item_visibility(item_doc);

    let (defaultness, has_body) = match entry_data(item_doc, cdata) {
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
            let ty = doc_type(item_doc, tcx, cdata);
            ty::ConstTraitItem(Rc::new(ty::AssociatedConst {
                name: name,
                ty: ty,
                vis: vis,
                defaultness: defaultness,
                def_id: def_id,
                container: container,
                has_value: has_body,
            }))
        }
        Family::Method => {
            let generics = doc_generics(item_doc, tcx, cdata);
            let predicates = doc_predicates(item_doc, tcx, cdata, item_tag::predicates);
            let ity = tcx.lookup_item_type(def_id).ty;
            let fty = match ity.sty {
                ty::TyFnDef(.., fty) => fty,
                _ => bug!(
                    "the type {:?} of the method {:?} is not a function?",
                    ity, name)
            };

            let explicit_self = match entry_typed_data(item_doc, tcx, cdata) {
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
            let ty = maybe_doc_type(item_doc, tcx, cdata);
            ty::TypeTraitItem(Rc::new(ty::AssociatedType {
                name: name,
                ty: ty,
                vis: vis,
                defaultness: defaultness,
                def_id: def_id,
                container: container,
            }))
        }
        _ => bug!()
    })
}

pub fn get_item_variances(cdata: Cmd, id: DefIndex) -> Vec<ty::Variance> {
    let item_doc = cdata.lookup_item(id);
    item_doc.get(item_tag::variances).decoder().decode()
}

pub fn get_struct_ctor_def_id(cdata: Cmd, node_id: DefIndex) -> Option<DefId>
{
    let data = match entry_data(cdata.lookup_item(node_id), cdata) {
        EntryData::Variant(data) => data,
        _ => bug!()
    };

    data.struct_ctor.map(|index| cdata.local_def_id(index))
}

pub fn get_item_attrs(cdata: Cmd,
                      node_id: DefIndex)
                      -> Vec<ast::Attribute> {
    // The attributes for a tuple struct are attached to the definition, not the ctor;
    // we assume that someone passing in a tuple struct ctor is actually wanting to
    // look at the definition
    let mut item = cdata.lookup_item(node_id);
    let def_key = item_def_key(item);
    if def_key.disambiguated_data.data == DefPathData::StructCtor {
        item = cdata.lookup_item(def_key.parent.unwrap());
    }
    get_attributes(item)
}

pub fn get_struct_field_names(cdata: Cmd, id: DefIndex) -> Vec<ast::Name> {
    let mut dcx = cdata.lookup_item(id).get(item_tag::children).decoder();
    dcx.cdata = Some(cdata);

    dcx.seq().map(|index| item_name(cdata.lookup_item(index))).collect()
}

fn get_attributes(md: rbml::Doc) -> Vec<ast::Attribute> {
    reader::maybe_get_doc(md, item_tag::attributes).map_or(vec![], |attrs_doc| {
        let mut attrs = attrs_doc.decoder().decode::<Vec<ast::Attribute>>();

        // Need new unique IDs: old thread-local IDs won't map to new threads.
        for attr in attrs.iter_mut() {
            attr.node.id = attr::mk_attr_id();
        }

        attrs
    })
}

#[derive(Clone)]
pub struct CrateDep {
    pub cnum: CrateNum,
    pub name: String,
    pub hash: Svh,
    pub explicitly_linked: bool,
}

pub fn get_crate_deps(data: &[u8]) -> Vec<CrateDep> {
    let dcx = rbml::Doc::new(data).get(root_tag::crate_deps).decoder();

    dcx.seq().enumerate().map(|(crate_num, (name, hash, explicitly_linked))| {
        CrateDep {
            cnum: CrateNum::new(crate_num + 1),
            name: name,
            hash: hash,
            explicitly_linked: explicitly_linked,
        }
    }).collect()
}

fn list_crate_deps(data: &[u8], out: &mut io::Write) -> io::Result<()> {
    write!(out, "=External Dependencies=\n")?;
    for dep in &get_crate_deps(data) {
        write!(out, "{} {}-{}\n", dep.cnum, dep.name, dep.hash)?;
    }
    write!(out, "\n")?;
    Ok(())
}

pub fn get_crate_info(data: &[u8]) -> CrateInfo {
    rbml::Doc::new(data).get(root_tag::crate_info).decoder().decode()
}

pub fn list_crate_metadata(bytes: &[u8], out: &mut io::Write) -> io::Result<()> {
    list_crate_deps(bytes, out)
}

// Translate a DefId from the current compilation environment to a DefId
// for an external crate.
fn reverse_translate_def_id(cdata: Cmd, did: DefId) -> Option<DefId> {
    for (local, &global) in cdata.cnum_map.borrow().iter_enumerated() {
        if global == did.krate {
            return Some(DefId { krate: local, index: did.index });
        }
    }

    None
}

pub fn each_inherent_implementation_for_type<F>(cdata: Cmd,
                                                id: DefIndex,
                                                mut callback: F)
    where F: FnMut(DefId),
{
    let item_doc = cdata.lookup_item(id);
    let mut dcx = item_doc.get(item_tag::inherent_impls).decoder();
    dcx.cdata = Some(cdata);

    for impl_def_id in dcx.seq() {
        callback(impl_def_id);
    }
}

pub fn each_implementation_for_trait<F>(cdata: Cmd,
                                        filter: Option<DefId>,
                                        mut callback: F) where
    F: FnMut(DefId),
{
    // Do a reverse lookup beforehand to avoid touching the crate_num
    // hash map in the loop below.
    let filter = match filter.map(|def_id| reverse_translate_def_id(cdata, def_id)) {
        Some(Some(def_id)) => Some(def_id),
        Some(None) => return,
        None => None
    };

    // FIXME(eddyb) Make this O(1) instead of O(n).
    for trait_doc in rbml::Doc::new(cdata.data()).get(root_tag::impls).children() {
        let mut dcx = trait_doc.decoder();
        dcx.cdata = Some(cdata);

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

pub fn get_trait_of_item(cdata: Cmd, id: DefIndex) -> Option<DefId> {
    let item_doc = cdata.lookup_item(id);
    let parent_item_id = match item_parent_item(cdata, item_doc) {
        None => return None,
        Some(item_id) => item_id,
    };
    match item_family(cdata.lookup_item(parent_item_id.index)) {
        Family::Trait => Some(parent_item_id),
        _ => None
    }
}


pub fn get_native_libraries(cdata: Cmd)
                            -> Vec<(cstore::NativeLibraryKind, String)> {
    rbml::Doc::new(cdata.data()).get(root_tag::native_libraries).decoder().decode()
}

pub fn each_exported_macro<F>(data: &[u8], mut f: F) where
    F: FnMut(ast::Name, Vec<ast::Attribute>, Span, String) -> bool,
{
    let dcx = rbml::Doc::new(data).get(root_tag::macro_defs).decoder();
    for (name, attrs, span, body) in dcx.seq() {
        if !f(name, attrs, span, body) {
            break;
        }
    }
}

pub fn get_dylib_dependency_formats(cdata: Cmd)
    -> Vec<(CrateNum, LinkagePreference)>
{
    let dcx = rbml::Doc::new(cdata.data()).get(root_tag::dylib_dependency_formats).decoder();

    dcx.seq::<Option<_>>().enumerate().flat_map(|(i, link)| {
        let cnum = CrateNum::new(i + 1);
        link.map(|link| (cdata.cnum_map.borrow()[cnum], link))
    }).collect()
}

pub fn get_missing_lang_items(cdata: Cmd) -> Vec<lang_items::LangItem> {
    rbml::Doc::new(cdata.data()).get(root_tag::lang_items_missing).decoder().decode()
}

pub fn get_fn_arg_names(cdata: Cmd, id: DefIndex) -> Vec<String> {
    let method_doc = cdata.lookup_item(id);
    match reader::maybe_get_doc(method_doc, item_tag::fn_arg_names) {
        Some(args_doc) => args_doc.decoder().decode(),
        None => vec![],
    }
}

pub fn get_reachable_ids(cdata: Cmd) -> Vec<DefId> {
    let dcx = rbml::Doc::new(cdata.data()).get(root_tag::reachable_ids).decoder();

    dcx.seq().map(|index| cdata.local_def_id(index)).collect()
}

pub fn is_const_fn(cdata: Cmd, id: DefIndex) -> bool {
    let constness = match entry_data(cdata.lookup_item(id), cdata) {
        EntryData::ImplAssociated(data) => data.constness,
        EntryData::Fn(data) => data.constness,
        _ => hir::Constness::NotConst
    };
    constness == hir::Constness::Const
}

pub fn is_extern_item<'a, 'tcx>(cdata: Cmd,
                                id: DefIndex,
                                tcx: TyCtxt<'a, 'tcx, 'tcx>)
                                -> bool {
    let item_doc = match cdata.get_item(id) {
        Some(doc) => doc,
        None => return false,
    };
    let applicable = match item_family(item_doc) {
        Family::ImmStatic |
        Family::MutStatic |
        Family::ForeignImmStatic |
        Family::ForeignMutStatic => true,

        Family::Fn | Family::ForeignFn => {
            get_generics(cdata, id, tcx).types.is_empty()
        }

        _ => false,
    };

    if applicable {
        attr::contains_extern_indicator(tcx.sess.diagnostic(),
                                        &get_attributes(item_doc))
    } else {
        false
    }
}

pub fn is_foreign_item(cdata: Cmd, id: DefIndex) -> bool {
    match item_family(cdata.lookup_item(id)) {
        Family::ForeignImmStatic |
        Family::ForeignMutStatic |
        Family::ForeignFn => true,
        _ => false
    }
}

fn doc_generics<'a, 'tcx>(base_doc: rbml::Doc,
                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                          cdata: Cmd)
                          -> &'tcx ty::Generics<'tcx>
{
    let mut dcx = base_doc.get(item_tag::generics).decoder();
    dcx.tcx = Some(tcx);
    dcx.cdata = Some(cdata);
    tcx.alloc_generics(dcx.decode())
}

fn doc_predicates<'a, 'tcx>(base_doc: rbml::Doc,
                            tcx: TyCtxt<'a, 'tcx, 'tcx>,
                            cdata: Cmd,
                            tag: usize)
                            -> ty::GenericPredicates<'tcx>
{
    let mut dcx = base_doc.get(tag).decoder();
    dcx.cdata = Some(cdata);
    dcx.tcx = Some(tcx);

    ty::GenericPredicates {
        parent: dcx.decode(),
        predicates: (0..dcx.decode::<usize>()).map(|_| {
            // Handle shorthands first, if we have an usize > 0x80.
            if dcx.opaque.data[dcx.opaque.position()] & 0x80 != 0 {
                let pos = dcx.decode::<usize>();
                assert!(pos >= SHORTHAND_OFFSET);
                let pos = pos - SHORTHAND_OFFSET;

                let mut dcx = rbml::Doc {
                    data: cdata.data(),
                    start: pos,
                    end: cdata.data().len(),
                }.decoder();
                dcx.tcx = Some(tcx);
                dcx.cdata = Some(cdata);
                dcx.decode()
            } else {
                dcx.decode()
            }
        }).collect()
    }
}

pub fn is_defaulted_trait(cdata: Cmd, trait_id: DefIndex) -> bool {
    match entry_data(cdata.lookup_item(trait_id), cdata) {
        EntryData::Trait(data) => data.has_default_impl,
        _ => bug!()
    }
}

pub fn is_default_impl(cdata: Cmd, impl_id: DefIndex) -> bool {
    item_family(cdata.lookup_item(impl_id)) == Family::DefaultImpl
}

pub fn get_imported_filemaps(metadata: &[u8]) -> Vec<syntax_pos::FileMap> {
    rbml::Doc::new(metadata).get(root_tag::codemap).decoder().decode()
}

pub fn closure_kind(cdata: Cmd, closure_id: DefIndex) -> ty::ClosureKind {
    match entry_data(cdata.lookup_item(closure_id), cdata) {
        EntryData::Closure(data) => data.kind,
        _ => bug!()
    }
}

pub fn closure_ty<'a, 'tcx>(cdata: Cmd, closure_id: DefIndex, tcx: TyCtxt<'a, 'tcx, 'tcx>)
                            -> ty::ClosureTy<'tcx> {
    match entry_typed_data(cdata.lookup_item(closure_id), tcx, cdata) {
        EntryTypedData::Closure(data) => data.ty,
        _ => bug!()
    }
}

pub fn def_key(cdata: Cmd, id: DefIndex) -> hir_map::DefKey {
    debug!("def_key: id={:?}", id);
    item_def_key(cdata.lookup_item(id))
}

fn item_def_key(item_doc: rbml::Doc) -> hir_map::DefKey {
    item_doc.get(item_tag::def_key).decoder().decode()
}

// Returns the path leading to the thing with this `id`. Note that
// some def-ids don't wind up in the metadata, so `def_path` sometimes
// returns `None`
pub fn def_path(cdata: Cmd, id: DefIndex) -> Option<hir_map::DefPath> {
    debug!("def_path(id={:?})", id);
    if cdata.get_item(id).is_some() {
        Some(hir_map::DefPath::make(cdata.cnum, id, |parent| def_key(cdata, parent)))
    } else {
        None
    }
}
