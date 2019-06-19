// Decoding metadata from a single crate's metadata

use crate::cstore::{self, CrateMetadata, MetadataBlob, NativeLibrary, ForeignModule};
use crate::schema::*;

use rustc_data_structures::sync::{Lrc, ReadGuard};
use rustc::hir::map::{DefKey, DefPath, DefPathData, DefPathHash, Definitions};
use rustc::hir;
use rustc::middle::cstore::LinkagePreference;
use rustc::middle::exported_symbols::{ExportedSymbol, SymbolExportLevel};
use rustc::hir::def::{self, Res, DefKind, CtorOf, CtorKind};
use rustc::hir::def_id::{CrateNum, DefId, DefIndex, LocalDefId, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc::hir::map::definitions::DefPathTable;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc::middle::lang_items;
use rustc::mir::{self, interpret};
use rustc::mir::interpret::AllocDecodingSession;
use rustc::session::Session;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::ty::codec::TyDecoder;
use rustc::mir::Body;
use rustc::util::captures::Captures;

use std::io;
use std::mem;
use std::u32;

use rustc_serialize::{Decodable, Decoder, SpecializedDecoder, opaque};
use syntax::attr;
use syntax::ast::{self, Ident};
use syntax::source_map;
use syntax::symbol::{Symbol, sym};
use syntax::ext::base::{MacroKind, SyntaxExtension};
use syntax::ext::hygiene::Mark;
use syntax_pos::{self, Span, BytePos, Pos, DUMMY_SP, NO_EXPANSION};
use log::debug;

pub struct DecodeContext<'a, 'tcx> {
    opaque: opaque::Decoder<'a>,
    cdata: Option<&'a CrateMetadata>,
    sess: Option<&'tcx Session>,
    tcx: Option<TyCtxt<'tcx>>,

    // Cache the last used source_file for translating spans as an optimization.
    last_source_file_index: usize,

    lazy_state: LazyState,

    // Used for decoding interpret::AllocIds in a cached & thread-safe manner.
    alloc_decoding_session: Option<AllocDecodingSession<'a>>,
}

/// Abstract over the various ways one can create metadata decoders.
pub trait Metadata<'a, 'tcx>: Copy {
    fn raw_bytes(self) -> &'a [u8];
    fn cdata(self) -> Option<&'a CrateMetadata> { None }
    fn sess(self) -> Option<&'tcx Session> { None }
    fn tcx(self) -> Option<TyCtxt<'tcx>> { None }

    fn decoder(self, pos: usize) -> DecodeContext<'a, 'tcx> {
        let tcx = self.tcx();
        DecodeContext {
            opaque: opaque::Decoder::new(self.raw_bytes(), pos),
            cdata: self.cdata(),
            sess: self.sess().or(tcx.map(|tcx| tcx.sess)),
            tcx,
            last_source_file_index: 0,
            lazy_state: LazyState::NoNode,
            alloc_decoding_session: self.cdata().map(|cdata| {
                cdata.alloc_decoding_state.new_decoding_session()
            }),
        }
    }
}

impl<'a, 'tcx> Metadata<'a, 'tcx> for &'a MetadataBlob {
    fn raw_bytes(self) -> &'a [u8] {
        &self.0
    }
}


impl<'a, 'tcx> Metadata<'a, 'tcx> for (&'a MetadataBlob, &'tcx Session) {
    fn raw_bytes(self) -> &'a [u8] {
        let (blob, _) = self;
        &blob.0
    }

    fn sess(self) -> Option<&'tcx Session> {
        let (_, sess) = self;
        Some(sess)
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

impl<'a, 'tcx> Metadata<'a, 'tcx> for (&'a CrateMetadata, &'tcx Session) {
    fn raw_bytes(self) -> &'a [u8] {
        self.0.raw_bytes()
    }
    fn cdata(self) -> Option<&'a CrateMetadata> {
        Some(self.0)
    }
    fn sess(self) -> Option<&'tcx Session> {
        Some(&self.1)
    }
}

impl<'a, 'tcx> Metadata<'a, 'tcx> for (&'a CrateMetadata, TyCtxt<'tcx>) {
    fn raw_bytes(self) -> &'a [u8] {
        self.0.raw_bytes()
    }
    fn cdata(self) -> Option<&'a CrateMetadata> {
        Some(self.0)
    }
    fn tcx(self) -> Option<TyCtxt<'tcx>> {
        Some(self.1)
    }
}

impl<'a, 'tcx, T: Decodable> Lazy<T> {
    pub fn decode<M: Metadata<'a, 'tcx>>(self, meta: M) -> T {
        let mut dcx = meta.decoder(self.position);
        dcx.lazy_state = LazyState::NodeStart(self.position);
        T::decode(&mut dcx).unwrap()
    }
}

impl<'a: 'x, 'tcx: 'x, 'x, T: Decodable> LazySeq<T> {
    pub fn decode<M: Metadata<'a, 'tcx>>(
        self,
        meta: M,
    ) -> impl Iterator<Item = T> + Captures<'a> + Captures<'tcx> + 'x {
        let mut dcx = meta.decoder(self.position);
        dcx.lazy_state = LazyState::NodeStart(self.position);
        (0..self.len).map(move |_| T::decode(&mut dcx).unwrap())
    }
}

impl<'a, 'tcx> DecodeContext<'a, 'tcx> {
    pub fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx.expect("missing TyCtxt in DecodeContext")
    }

    pub fn cdata(&self) -> &'a CrateMetadata {
        self.cdata.expect("missing CrateMetadata in DecodeContext")
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

impl<'a, 'tcx> TyDecoder<'tcx> for DecodeContext<'a, 'tcx> {
    #[inline]
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx.expect("missing TyCtxt in DecodeContext")
    }

    #[inline]
    fn peek_byte(&self) -> u8 {
        self.opaque.data[self.opaque.position()]
    }

    #[inline]
    fn position(&self) -> usize {
        self.opaque.position()
    }

    fn cached_ty_for_shorthand<F>(&mut self,
                                  shorthand: usize,
                                  or_insert_with: F)
                                  -> Result<Ty<'tcx>, Self::Error>
        where F: FnOnce(&mut Self) -> Result<Ty<'tcx>, Self::Error>
    {
        let tcx = self.tcx();

        let key = ty::CReaderCacheKey {
            cnum: self.cdata().cnum,
            pos: shorthand,
        };

        if let Some(&ty) = tcx.rcache.borrow().get(&key) {
            return Ok(ty);
        }

        let ty = or_insert_with(self)?;
        tcx.rcache.borrow_mut().insert(key, ty);
        Ok(ty)
    }

    fn with_position<F, R>(&mut self, pos: usize, f: F) -> R
        where F: FnOnce(&mut Self) -> R
    {
        let new_opaque = opaque::Decoder::new(self.opaque.data, pos);
        let old_opaque = mem::replace(&mut self.opaque, new_opaque);
        let old_state = mem::replace(&mut self.lazy_state, LazyState::NoNode);
        let r = f(self);
        self.opaque = old_opaque;
        self.lazy_state = old_state;
        r
    }

    fn map_encoded_cnum_to_current(&self, cnum: CrateNum) -> CrateNum {
        if cnum == LOCAL_CRATE {
            self.cdata().cnum
        } else {
            self.cdata().cnum_map[cnum]
        }
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


impl<'a, 'tcx> SpecializedDecoder<DefId> for DecodeContext<'a, 'tcx> {
    #[inline]
    fn specialized_decode(&mut self) -> Result<DefId, Self::Error> {
        let krate = CrateNum::decode(self)?;
        let index = DefIndex::decode(self)?;

        Ok(DefId {
            krate,
            index,
        })
    }
}

impl<'a, 'tcx> SpecializedDecoder<DefIndex> for DecodeContext<'a, 'tcx> {
    #[inline]
    fn specialized_decode(&mut self) -> Result<DefIndex, Self::Error> {
        Ok(DefIndex::from_u32(self.read_u32()?))
    }
}

impl<'a, 'tcx> SpecializedDecoder<LocalDefId> for DecodeContext<'a, 'tcx> {
    #[inline]
    fn specialized_decode(&mut self) -> Result<LocalDefId, Self::Error> {
        self.specialized_decode().map(|i| LocalDefId::from_def_id(i))
    }
}

impl<'a, 'tcx> SpecializedDecoder<interpret::AllocId> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<interpret::AllocId, Self::Error> {
        if let Some(alloc_decoding_session) = self.alloc_decoding_session {
            alloc_decoding_session.decode_alloc_id(self)
        } else {
            bug!("Attempting to decode interpret::AllocId without CrateMetadata")
        }
    }
}

impl<'a, 'tcx> SpecializedDecoder<Span> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<Span, Self::Error> {
        let tag = u8::decode(self)?;

        if tag == TAG_INVALID_SPAN {
            return Ok(DUMMY_SP)
        }

        debug_assert_eq!(tag, TAG_VALID_SPAN);

        let lo = BytePos::decode(self)?;
        let len = BytePos::decode(self)?;
        let hi = lo + len;

        let sess = if let Some(sess) = self.sess {
            sess
        } else {
            bug!("Cannot decode Span without Session.")
        };

        let imported_source_files = self.cdata().imported_source_files(&sess.source_map());
        let source_file = {
            // Optimize for the case that most spans within a translated item
            // originate from the same source_file.
            let last_source_file = &imported_source_files[self.last_source_file_index];

            if lo >= last_source_file.original_start_pos &&
               lo <= last_source_file.original_end_pos {
                last_source_file
            } else {
                let mut a = 0;
                let mut b = imported_source_files.len();

                while b - a > 1 {
                    let m = (a + b) / 2;
                    if imported_source_files[m].original_start_pos > lo {
                        b = m;
                    } else {
                        a = m;
                    }
                }

                self.last_source_file_index = a;
                &imported_source_files[a]
            }
        };

        // Make sure our binary search above is correct.
        debug_assert!(lo >= source_file.original_start_pos &&
                      lo <= source_file.original_end_pos);

        // Make sure we correctly filtered out invalid spans during encoding
        debug_assert!(hi >= source_file.original_start_pos &&
                      hi <= source_file.original_end_pos);

        let lo = (lo + source_file.translated_source_file.start_pos)
                 - source_file.original_start_pos;
        let hi = (hi + source_file.translated_source_file.start_pos)
                 - source_file.original_start_pos;

        Ok(Span::new(lo, hi, NO_EXPANSION))
    }
}

impl<'a, 'tcx> SpecializedDecoder<Fingerprint> for DecodeContext<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<Fingerprint, Self::Error> {
        Fingerprint::decode_opaque(&mut self.opaque)
    }
}

impl<'a, 'tcx, T: Decodable> SpecializedDecoder<mir::ClearCrossCrate<T>>
for DecodeContext<'a, 'tcx> {
    #[inline]
    fn specialized_decode(&mut self) -> Result<mir::ClearCrossCrate<T>, Self::Error> {
        Ok(mir::ClearCrossCrate::Clear)
    }
}

implement_ty_decoder!( DecodeContext<'a, 'tcx> );

impl<'tcx> MetadataBlob {
    pub fn is_compatible(&self) -> bool {
        self.raw_bytes().starts_with(METADATA_HEADER)
    }

    pub fn get_rustc_version(&self) -> String {
        Lazy::with_position(METADATA_HEADER.len() + 4).decode(self)
    }

    pub fn get_root(&self) -> CrateRoot<'tcx> {
        let slice = self.raw_bytes();
        let offset = METADATA_HEADER.len();
        let pos = (((slice[offset + 0] as u32) << 24) | ((slice[offset + 1] as u32) << 16) |
                   ((slice[offset + 2] as u32) << 8) |
                   ((slice[offset + 3] as u32) << 0)) as usize;
        Lazy::with_position(pos).decode(self)
    }

    pub fn list_crate_metadata(&self,
                               out: &mut dyn io::Write) -> io::Result<()> {
        write!(out, "=External Dependencies=\n")?;
        let root = self.get_root();
        for (i, dep) in root.crate_deps
                            .decode(self)
                            .enumerate() {
            write!(out, "{} {}{}\n", i + 1, dep.name, dep.extra_filename)?;
        }
        write!(out, "\n")?;
        Ok(())
    }
}

impl<'tcx> EntryKind<'tcx> {
    fn def_kind(&self) -> Option<DefKind> {
        Some(match *self {
            EntryKind::Const(..) => DefKind::Const,
            EntryKind::AssocConst(..) => DefKind::AssocConst,
            EntryKind::ImmStatic |
            EntryKind::MutStatic |
            EntryKind::ForeignImmStatic |
            EntryKind::ForeignMutStatic => DefKind::Static,
            EntryKind::Struct(_, _) => DefKind::Struct,
            EntryKind::Union(_, _) => DefKind::Union,
            EntryKind::Fn(_) |
            EntryKind::ForeignFn(_) => DefKind::Fn,
            EntryKind::Method(_) => DefKind::Method,
            EntryKind::Type => DefKind::TyAlias,
            EntryKind::TypeParam => DefKind::TyParam,
            EntryKind::ConstParam => DefKind::ConstParam,
            EntryKind::Existential => DefKind::Existential,
            EntryKind::AssocType(_) => DefKind::AssocTy,
            EntryKind::AssocExistential(_) => DefKind::AssocExistential,
            EntryKind::Mod(_) => DefKind::Mod,
            EntryKind::Variant(_) => DefKind::Variant,
            EntryKind::Trait(_) => DefKind::Trait,
            EntryKind::TraitAlias(_) => DefKind::TraitAlias,
            EntryKind::Enum(..) => DefKind::Enum,
            EntryKind::MacroDef(_) => DefKind::Macro(MacroKind::Bang),
            EntryKind::ForeignType => DefKind::ForeignTy,

            EntryKind::ForeignMod |
            EntryKind::GlobalAsm |
            EntryKind::Impl(_) |
            EntryKind::Field |
            EntryKind::Generator(_) |
            EntryKind::Closure(_) => return None,
        })
    }
}

/// Creates the "fake" DefPathTable for a given proc macro crate.
///
/// The DefPathTable is as follows:
///
/// CRATE_ROOT (DefIndex 0:0)
///  |- GlobalMetaDataKind data (DefIndex 1:0 .. DefIndex 1:N)
///  |- proc macro #0 (DefIndex 1:N)
///  |- proc macro #1 (DefIndex 1:N+1)
///  \- ...
crate fn proc_macro_def_path_table(crate_root: &CrateRoot<'_>,
                                   proc_macros: &[(ast::Name, Lrc<SyntaxExtension>)])
                                   -> DefPathTable
{
    let mut definitions = Definitions::default();

    let name = crate_root.name.as_str();
    let disambiguator = crate_root.disambiguator;
    debug!("creating proc macro def path table for {:?}/{:?}", name, disambiguator);
    let crate_root = definitions.create_root_def(&name, disambiguator);
    for (index, (name, _)) in proc_macros.iter().enumerate() {
        let def_index = definitions.create_def_with_parent(
            crate_root,
            ast::DUMMY_NODE_ID,
            DefPathData::MacroNs(name.as_interned_str()),
            Mark::root(),
            DUMMY_SP);
        debug!("definition for {:?} is {:?}", name, def_index);
        assert_eq!(def_index, DefIndex::from_proc_macro_index(index));
    }

    definitions.def_path_table().clone()
}

impl<'a, 'tcx> CrateMetadata {
    fn is_proc_macro(&self, id: DefIndex) -> bool {
        self.proc_macros.is_some() && id != CRATE_DEF_INDEX
    }

    fn maybe_entry(&self, item_id: DefIndex) -> Option<Lazy<Entry<'tcx>>> {
        assert!(!self.is_proc_macro(item_id));
        self.root.entries_index.lookup(self.blob.raw_bytes(), item_id)
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
            index,
        }
    }

    pub fn item_name(&self, item_index: DefIndex) -> Symbol {
        self.def_key(item_index)
            .disambiguated_data
            .data
            .get_opt_name()
            .expect("no name in item_name")
            .as_symbol()
    }

    pub fn def_kind(&self, index: DefIndex) -> Option<DefKind> {
        if !self.is_proc_macro(index) {
            self.entry(index).kind.def_kind()
        } else {
            Some(DefKind::Macro(
                self.proc_macros.as_ref().unwrap()[index.to_proc_macro_index()].1.macro_kind()
            ))
        }
    }

    pub fn get_span(&self, index: DefIndex, sess: &Session) -> Span {
        match self.is_proc_macro(index) {
            true => DUMMY_SP,
            false => self.entry(index).span.decode((self, sess)),
        }
    }

    pub fn get_trait_def(&self, item_id: DefIndex, sess: &Session) -> ty::TraitDef {
        match self.entry(item_id).kind {
            EntryKind::Trait(data) => {
                let data = data.decode((self, sess));
                ty::TraitDef::new(self.local_def_id(item_id),
                                  data.unsafety,
                                  data.paren_sugar,
                                  data.has_auto_impl,
                                  data.is_marker,
                                  self.def_path_table.def_path_hash(item_id))
            },
            EntryKind::TraitAlias(_) => {
                ty::TraitDef::new(self.local_def_id(item_id),
                                  hir::Unsafety::Normal,
                                  false,
                                  false,
                                  false,
                                  self.def_path_table.def_path_hash(item_id))
            },
            _ => bug!("def-index does not refer to trait or trait alias"),
        }
    }

    fn get_variant(
        &self,
        tcx: TyCtxt<'tcx>,
        item: &Entry<'_>,
        index: DefIndex,
        parent_did: DefId,
        adt_kind: ty::AdtKind,
    ) -> ty::VariantDef {
        let data = match item.kind {
            EntryKind::Variant(data) |
            EntryKind::Struct(data, _) |
            EntryKind::Union(data, _) => data.decode(self),
            _ => bug!(),
        };

        let variant_did = if adt_kind == ty::AdtKind::Enum {
            Some(self.local_def_id(index))
        } else {
            None
        };
        let ctor_did = data.ctor.map(|index| self.local_def_id(index));

        ty::VariantDef::new(
            tcx,
            Ident::with_empty_ctxt(self.item_name(index)),
            variant_did,
            ctor_did,
            data.discr,
            item.children.decode(self).map(|index| {
                let f = self.entry(index);
                ty::FieldDef {
                    did: self.local_def_id(index),
                    ident: Ident::with_empty_ctxt(self.item_name(index)),
                    vis: f.visibility.decode(self)
                }
            }).collect(),
            data.ctor_kind,
            adt_kind,
            parent_did,
            false,
        )
    }

    pub fn get_adt_def(&self, item_id: DefIndex, tcx: TyCtxt<'tcx>) -> &'tcx ty::AdtDef {
        let item = self.entry(item_id);
        let did = self.local_def_id(item_id);

        let (kind, repr) = match item.kind {
            EntryKind::Enum(repr) => (ty::AdtKind::Enum, repr),
            EntryKind::Struct(_, repr) => (ty::AdtKind::Struct, repr),
            EntryKind::Union(_, repr) => (ty::AdtKind::Union, repr),
            _ => bug!("get_adt_def called on a non-ADT {:?}", did),
        };

        let variants = if let ty::AdtKind::Enum = kind {
            item.children
                .decode(self)
                .map(|index| {
                    self.get_variant(tcx, &self.entry(index), index, did, kind)
                })
                .collect()
        } else {
            std::iter::once(self.get_variant(tcx, &item, item_id, did, kind)).collect()
        };

        tcx.alloc_adt_def(did, kind, variants, repr)
    }

    pub fn get_predicates(
        &self,
        item_id: DefIndex,
        tcx: TyCtxt<'tcx>,
    ) -> ty::GenericPredicates<'tcx> {
        self.entry(item_id).predicates.unwrap().decode((self, tcx))
}

    pub fn get_predicates_defined_on(
        &self,
        item_id: DefIndex,
        tcx: TyCtxt<'tcx>,
    ) -> ty::GenericPredicates<'tcx> {
        self.entry(item_id).predicates_defined_on.unwrap().decode((self, tcx))
    }

    pub fn get_super_predicates(
        &self,
        item_id: DefIndex,
        tcx: TyCtxt<'tcx>,
    ) -> ty::GenericPredicates<'tcx> {
        let super_predicates = match self.entry(item_id).kind {
            EntryKind::Trait(data) => data.decode(self).super_predicates,
            EntryKind::TraitAlias(data) => data.decode(self).super_predicates,
            _ => bug!("def-index does not refer to trait or trait alias"),
        };

        super_predicates.decode((self, tcx))
    }

    pub fn get_generics(&self,
                        item_id: DefIndex,
                        sess: &Session)
                        -> ty::Generics {
        self.entry(item_id).generics.unwrap().decode((self, sess))
    }

    pub fn get_type(&self, id: DefIndex, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        self.entry(id).ty.unwrap().decode((self, tcx))
    }

    pub fn get_stability(&self, id: DefIndex) -> Option<attr::Stability> {
        match self.is_proc_macro(id) {
            true => self.root.proc_macro_stability.clone(),
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

    pub fn get_impl_defaultness(&self, id: DefIndex) -> hir::Defaultness {
        self.get_impl_data(id).defaultness
    }

    pub fn get_coerce_unsized_info(&self,
                                   id: DefIndex)
                                   -> Option<ty::adjustment::CoerceUnsizedInfo> {
        self.get_impl_data(id).coerce_unsized_info
    }

    pub fn get_impl_trait(&self, id: DefIndex, tcx: TyCtxt<'tcx>) -> Option<ty::TraitRef<'tcx>> {
        self.get_impl_data(id).trait_ref.map(|tr| tr.decode((self, tcx)))
    }

    /// Iterates over all the stability attributes in the given crate.
    pub fn get_lib_features(&self, tcx: TyCtxt<'tcx>) -> &'tcx [(ast::Name, Option<ast::Name>)] {
        // FIXME: For a proc macro crate, not sure whether we should return the "host"
        // features or an empty Vec. Both don't cause ICEs.
        tcx.arena.alloc_from_iter(self.root
            .lib_features
            .decode(self))
    }

    /// Iterates over the language items in the given crate.
    pub fn get_lang_items(&self, tcx: TyCtxt<'tcx>) -> &'tcx [(DefId, usize)] {
        if self.proc_macros.is_some() {
            // Proc macro crates do not export any lang-items to the target.
            &[]
        } else {
            tcx.arena.alloc_from_iter(self.root
                .lang_items
                .decode(self)
                .map(|(def_index, index)| (self.local_def_id(def_index), index)))
        }
    }

    /// Iterates over each child of the given item.
    pub fn each_child_of_item<F>(&self, id: DefIndex, mut callback: F, sess: &Session)
        where F: FnMut(def::Export<hir::HirId>)
    {
        if let Some(ref proc_macros) = self.proc_macros {
            /* If we are loading as a proc macro, we want to return the view of this crate
             * as a proc macro crate, not as a Rust crate. See `proc_macro_def_path_table`
             * for the DefPathTable we are corresponding to.
             */
            if id == CRATE_DEF_INDEX {
                for (id, &(name, ref ext)) in proc_macros.iter().enumerate() {
                    let res = Res::Def(
                        DefKind::Macro(ext.macro_kind()),
                        self.local_def_id(DefIndex::from_proc_macro_index(id)),
                    );
                    let ident = Ident::with_empty_ctxt(name);
                    callback(def::Export {
                        ident: ident,
                        res: res,
                        vis: ty::Visibility::Public,
                        span: DUMMY_SP,
                    });
                }
            }
            return
        }

        // Find the item.
        let item = match self.maybe_entry(id) {
            None => return,
            Some(item) => item.decode((self, sess)),
        };

        // Iterate over all children.
        let macros_only = self.dep_kind.lock().macros_only();
        for child_index in item.children.decode((self, sess)) {
            if macros_only {
                continue
            }

            // Get the item.
            if let Some(child) = self.maybe_entry(child_index) {
                let child = child.decode((self, sess));
                match child.kind {
                    EntryKind::MacroDef(..) => {}
                    _ if macros_only => continue,
                    _ => {}
                }

                // Hand off the item to the callback.
                match child.kind {
                    // FIXME(eddyb) Don't encode these in children.
                    EntryKind::ForeignMod => {
                        for child_index in child.children.decode((self, sess)) {
                            if let Some(kind) = self.def_kind(child_index) {
                                callback(def::Export {
                                    res: Res::Def(kind, self.local_def_id(child_index)),
                                    ident: Ident::with_empty_ctxt(self.item_name(child_index)),
                                    vis: self.get_visibility(child_index),
                                    span: self.entry(child_index).span.decode((self, sess)),
                                });
                            }
                        }
                        continue;
                    }
                    EntryKind::Impl(_) => continue,

                    _ => {}
                }

                let def_key = self.def_key(child_index);
                let span = child.span.decode((self, sess));
                if let (Some(kind), Some(name)) =
                    (self.def_kind(child_index), def_key.disambiguated_data.data.get_opt_name()) {
                    let ident = Ident::from_interned_str(name);
                    let vis = self.get_visibility(child_index);
                    let def_id = self.local_def_id(child_index);
                    let res = Res::Def(kind, def_id);
                    callback(def::Export { res, ident, vis, span });
                    // For non-re-export structs and variants add their constructors to children.
                    // Re-export lists automatically contain constructors when necessary.
                    match kind {
                        DefKind::Struct => {
                            if let Some(ctor_def_id) = self.get_ctor_def_id(child_index) {
                                let ctor_kind = self.get_ctor_kind(child_index);
                                let ctor_res = Res::Def(
                                    DefKind::Ctor(CtorOf::Struct, ctor_kind),
                                    ctor_def_id,
                                );
                                let vis = self.get_visibility(ctor_def_id.index);
                                callback(def::Export { res: ctor_res, vis, ident, span });
                            }
                        }
                        DefKind::Variant => {
                            // Braced variants, unlike structs, generate unusable names in
                            // value namespace, they are reserved for possible future use.
                            // It's ok to use the variant's id as a ctor id since an
                            // error will be reported on any use of such resolution anyway.
                            let ctor_def_id = self.get_ctor_def_id(child_index).unwrap_or(def_id);
                            let ctor_kind = self.get_ctor_kind(child_index);
                            let ctor_res = Res::Def(
                                DefKind::Ctor(CtorOf::Variant, ctor_kind),
                                ctor_def_id,
                            );
                            let mut vis = self.get_visibility(ctor_def_id.index);
                            if ctor_def_id == def_id && vis == ty::Visibility::Public {
                                // For non-exhaustive variants lower the constructor visibility to
                                // within the crate. We only need this for fictive constructors,
                                // for other constructors correct visibilities
                                // were already encoded in metadata.
                                let attrs = self.get_item_attrs(def_id.index, sess);
                                if attr::contains_name(&attrs, sym::non_exhaustive) {
                                    let crate_def_id = self.local_def_id(CRATE_DEF_INDEX);
                                    vis = ty::Visibility::Restricted(crate_def_id);
                                }
                            }
                            callback(def::Export { res: ctor_res, ident, vis, span });
                        }
                        _ => {}
                    }
                }
            }
        }

        if let EntryKind::Mod(data) = item.kind {
            for exp in data.decode((self, sess)).reexports.decode((self, sess)) {
                match exp.res {
                    Res::Def(DefKind::Macro(..), _) => {}
                    _ if macros_only => continue,
                    _ => {}
                }
                callback(exp);
            }
        }
    }

    pub fn const_is_rvalue_promotable_to_static(&self, id: DefIndex) -> bool {
        match self.entry(id).kind {
            EntryKind::AssocConst(_, data, _) |
            EntryKind::Const(data, _) => data.ast_promotable,
            _ => bug!(),
        }
    }

    pub fn is_item_mir_available(&self, id: DefIndex) -> bool {
        !self.is_proc_macro(id) &&
        self.maybe_entry(id).and_then(|item| item.decode(self).mir).is_some()
    }

    pub fn maybe_get_optimized_mir(&self, tcx: TyCtxt<'tcx>, id: DefIndex) -> Option<Body<'tcx>> {
        match self.is_proc_macro(id) {
            true => None,
            false => self.entry(id).mir.map(|mir| mir.decode((self, tcx))),
        }
    }

    pub fn mir_const_qualif(&self, id: DefIndex) -> u8 {
        match self.entry(id).kind {
            EntryKind::Const(qualif, _) |
            EntryKind::AssocConst(AssocContainer::ImplDefault, qualif, _) |
            EntryKind::AssocConst(AssocContainer::ImplFinal, qualif, _) => {
                qualif.mir
            }
            _ => bug!(),
        }
    }

    pub fn get_associated_item(&self, id: DefIndex) -> ty::AssocItem {
        let item = self.entry(id);
        let def_key = self.def_key(id);
        let parent = self.local_def_id(def_key.parent.unwrap());
        let name = def_key.disambiguated_data.data.get_opt_name().unwrap();

        let (kind, container, has_self) = match item.kind {
            EntryKind::AssocConst(container, _, _) => {
                (ty::AssocKind::Const, container, false)
            }
            EntryKind::Method(data) => {
                let data = data.decode(self);
                (ty::AssocKind::Method, data.container, data.has_self)
            }
            EntryKind::AssocType(container) => {
                (ty::AssocKind::Type, container, false)
            }
            EntryKind::AssocExistential(container) => {
                (ty::AssocKind::Existential, container, false)
            }
            _ => bug!("cannot get associated-item of `{:?}`", def_key)
        };

        ty::AssocItem {
            ident: Ident::from_interned_str(name),
            kind,
            vis: item.visibility.decode(self),
            defaultness: container.defaultness(),
            def_id: self.local_def_id(id),
            container: container.with_def_id(parent),
            method_has_self_argument: has_self
        }
    }

    pub fn get_item_variances(&self, id: DefIndex) -> Vec<ty::Variance> {
        self.entry(id).variances.decode(self).collect()
    }

    pub fn get_ctor_kind(&self, node_id: DefIndex) -> CtorKind {
        match self.entry(node_id).kind {
            EntryKind::Struct(data, _) |
            EntryKind::Union(data, _) |
            EntryKind::Variant(data) => data.decode(self).ctor_kind,
            _ => CtorKind::Fictive,
        }
    }

    pub fn get_ctor_def_id(&self, node_id: DefIndex) -> Option<DefId> {
        match self.entry(node_id).kind {
            EntryKind::Struct(data, _) => {
                data.decode(self).ctor.map(|index| self.local_def_id(index))
            }
            EntryKind::Variant(data) => {
                data.decode(self).ctor.map(|index| self.local_def_id(index))
            }
            _ => None,
        }
    }

    pub fn get_item_attrs(&self, node_id: DefIndex, sess: &Session) -> Lrc<[ast::Attribute]> {
        if self.is_proc_macro(node_id) {
            return Lrc::new([]);
        }

        // The attributes for a tuple struct/variant are attached to the definition, not the ctor;
        // we assume that someone passing in a tuple struct ctor is actually wanting to
        // look at the definition
        let def_key = self.def_key(node_id);
        let item_id = if def_key.disambiguated_data.data == DefPathData::Ctor {
            def_key.parent.unwrap()
        } else {
            node_id
        };

        let item = self.entry(item_id);
        Lrc::from(self.get_attributes(&item, sess))
    }

    pub fn get_struct_field_names(&self, id: DefIndex) -> Vec<ast::Name> {
        self.entry(id)
            .children
            .decode(self)
            .map(|index| self.item_name(index))
            .collect()
    }

    fn get_attributes(&self, item: &Entry<'tcx>, sess: &Session) -> Vec<ast::Attribute> {
        item.attributes
            .decode((self, sess))
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
        for (local, &global) in self.cnum_map.iter_enumerated() {
            if global == did.krate {
                return Some(DefId {
                    krate: local,
                    index: did.index,
                });
            }
        }

        None
    }

    pub fn get_inherent_implementations_for_type(
        &self,
        tcx: TyCtxt<'tcx>,
        id: DefIndex,
    ) -> &'tcx [DefId] {
        tcx.arena.alloc_from_iter(self.entry(id)
                                      .inherent_impls
                                      .decode(self)
                                      .map(|index| self.local_def_id(index)))
    }

    pub fn get_implementations_for_trait(
        &self,
        tcx: TyCtxt<'tcx>,
        filter: Option<DefId>,
    ) -> &'tcx [DefId] {
        if self.proc_macros.is_some() {
            // proc-macro crates export no trait impls.
            return &[]
        }

        // Do a reverse lookup beforehand to avoid touching the crate_num
        // hash map in the loop below.
        let filter = match filter.map(|def_id| self.reverse_translate_def_id(def_id)) {
            Some(Some(def_id)) => Some((def_id.krate.as_u32(), def_id.index)),
            Some(None) => return &[],
            None => None,
        };

        if let Some(filter) = filter {
            if let Some(impls) = self.trait_impls.get(&filter) {
                tcx.arena.alloc_from_iter(impls.decode(self).map(|idx| self.local_def_id(idx)))
            } else {
                &[]
            }
        } else {
            tcx.arena.alloc_from_iter(self.trait_impls.values().flat_map(|impls| {
                impls.decode(self).map(|idx| self.local_def_id(idx))
            }))
        }
    }

    pub fn get_trait_of_item(&self, id: DefIndex) -> Option<DefId> {
        let def_key = self.def_key(id);
        match def_key.disambiguated_data.data {
            DefPathData::TypeNs(..) | DefPathData::ValueNs(..) => (),
            // Not an associated item
            _ => return None,
        }
        def_key.parent.and_then(|parent_index| {
            match self.entry(parent_index).kind {
                EntryKind::Trait(_) |
                EntryKind::TraitAlias(_) => Some(self.local_def_id(parent_index)),
                _ => None,
            }
        })
    }


    pub fn get_native_libraries(&self, sess: &Session) -> Vec<NativeLibrary> {
        if self.proc_macros.is_some() {
            // Proc macro crates do not have any *target* native libraries.
            vec![]
        } else {
            self.root.native_libraries.decode((self, sess)).collect()
        }
    }

    pub fn get_foreign_modules(&self, tcx: TyCtxt<'tcx>) -> &'tcx [ForeignModule] {
        if self.proc_macros.is_some() {
            // Proc macro crates do not have any *target* foreign modules.
            &[]
        } else {
            tcx.arena.alloc_from_iter(self.root.foreign_modules.decode((self, tcx.sess)))
        }
    }

    pub fn get_dylib_dependency_formats(
        &self,
        tcx: TyCtxt<'tcx>,
    ) -> &'tcx [(CrateNum, LinkagePreference)] {
        tcx.arena.alloc_from_iter(self.root
            .dylib_dependency_formats
            .decode(self)
            .enumerate()
            .flat_map(|(i, link)| {
                let cnum = CrateNum::new(i + 1);
                link.map(|link| (self.cnum_map[cnum], link))
            }))
    }

    pub fn get_missing_lang_items(&self, tcx: TyCtxt<'tcx>) -> &'tcx [lang_items::LangItem] {
        if self.proc_macros.is_some() {
            // Proc macro crates do not depend on any target weak lang-items.
            &[]
        } else {
            tcx.arena.alloc_from_iter(self.root
                .lang_items_missing
                .decode(self))
        }
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

    pub fn exported_symbols(
        &self,
        tcx: TyCtxt<'tcx>,
    ) -> Vec<(ExportedSymbol<'tcx>, SymbolExportLevel)> {
        if self.proc_macros.is_some() {
            // If this crate is a custom derive crate, then we're not even going to
            // link those in so we skip those crates.
            vec![]
        } else {
            self.root.exported_symbols.decode((self, tcx)).collect()
        }
    }

    pub fn get_rendered_const(&self, id: DefIndex) -> String {
        match self.entry(id).kind {
            EntryKind::Const(_, data) |
            EntryKind::AssocConst(_, _, data) => data.decode(self).0,
            _ => bug!(),
        }
    }

    pub fn get_macro(&self, id: DefIndex) -> MacroDef {
        let entry = self.entry(id);
        match entry.kind {
            EntryKind::MacroDef(macro_def) => macro_def.decode(self),
            _ => bug!(),
        }
    }

    crate fn is_const_fn_raw(&self, id: DefIndex) -> bool {
        let constness = match self.entry(id).kind {
            EntryKind::Method(data) => data.decode(self).fn_data.constness,
            EntryKind::Fn(data) => data.decode(self).constness,
            EntryKind::Variant(..) | EntryKind::Struct(..) => hir::Constness::Const,
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

    crate fn static_mutability(&self, id: DefIndex) -> Option<hir::Mutability> {
        match self.entry(id).kind {
            EntryKind::ImmStatic |
            EntryKind::ForeignImmStatic => Some(hir::MutImmutable),
            EntryKind::MutStatic |
            EntryKind::ForeignMutStatic => Some(hir::MutMutable),
            _ => None,
        }
    }

    pub fn fn_sig(&self, id: DefIndex, tcx: TyCtxt<'tcx>) -> ty::PolyFnSig<'tcx> {
        let sig = match self.entry(id).kind {
            EntryKind::Fn(data) |
            EntryKind::ForeignFn(data) => data.decode(self).sig,
            EntryKind::Method(data) => data.decode(self).fn_data.sig,
            EntryKind::Variant(data) |
            EntryKind::Struct(data, _) => data.decode(self).ctor_sig.unwrap(),
            EntryKind::Closure(data) => data.decode(self).sig,
            _ => bug!(),
        };
        sig.decode((self, tcx))
    }

    #[inline]
    pub fn def_key(&self, index: DefIndex) -> DefKey {
        self.def_path_table.def_key(index)
    }

    // Returns the path leading to the thing with this `id`.
    pub fn def_path(&self, id: DefIndex) -> DefPath {
        debug!("def_path(cnum={:?}, id={:?})", self.cnum, id);
        DefPath::make(self.cnum, id, |parent| self.def_path_table.def_key(parent))
    }

    #[inline]
    pub fn def_path_hash(&self, index: DefIndex) -> DefPathHash {
        self.def_path_table.def_path_hash(index)
    }

    /// Imports the source_map from an external crate into the source_map of the crate
    /// currently being compiled (the "local crate").
    ///
    /// The import algorithm works analogous to how AST items are inlined from an
    /// external crate's metadata:
    /// For every SourceFile in the external source_map an 'inline' copy is created in the
    /// local source_map. The correspondence relation between external and local
    /// SourceFiles is recorded in the `ImportedSourceFile` objects returned from this
    /// function. When an item from an external crate is later inlined into this
    /// crate, this correspondence information is used to translate the span
    /// information of the inlined item so that it refers the correct positions in
    /// the local source_map (see `<decoder::DecodeContext as SpecializedDecoder<Span>>`).
    ///
    /// The import algorithm in the function below will reuse SourceFiles already
    /// existing in the local source_map. For example, even if the SourceFile of some
    /// source file of libstd gets imported many times, there will only ever be
    /// one SourceFile object for the corresponding file in the local source_map.
    ///
    /// Note that imported SourceFiles do not actually contain the source code of the
    /// file they represent, just information about length, line breaks, and
    /// multibyte characters. This information is enough to generate valid debuginfo
    /// for items inlined from other crates.
    ///
    /// Proc macro crates don't currently export spans, so this function does not have
    /// to work for them.
    pub fn imported_source_files(&'a self,
                                 local_source_map: &source_map::SourceMap)
                                 -> ReadGuard<'a, Vec<cstore::ImportedSourceFile>> {
        {
            let source_files = self.source_map_import_info.borrow();
            if !source_files.is_empty() {
                return source_files;
            }
        }

        // Lock the source_map_import_info to ensure this only happens once
        let mut source_map_import_info = self.source_map_import_info.borrow_mut();

        if !source_map_import_info.is_empty() {
            drop(source_map_import_info);
            return self.source_map_import_info.borrow();
        }

        let external_source_map = self.root.source_map.decode(self);

        let imported_source_files = external_source_map.map(|source_file_to_import| {
            // We can't reuse an existing SourceFile, so allocate a new one
            // containing the information we need.
            let syntax_pos::SourceFile { name,
                                      name_was_remapped,
                                      src_hash,
                                      start_pos,
                                      end_pos,
                                      mut lines,
                                      mut multibyte_chars,
                                      mut non_narrow_chars,
                                      name_hash,
                                      .. } = source_file_to_import;

            let source_length = (end_pos - start_pos).to_usize();

            // Translate line-start positions and multibyte character
            // position into frame of reference local to file.
            // `SourceMap::new_imported_source_file()` will then translate those
            // coordinates to their new global frame of reference when the
            // offset of the SourceFile is known.
            for pos in &mut lines {
                *pos = *pos - start_pos;
            }
            for mbc in &mut multibyte_chars {
                mbc.pos = mbc.pos - start_pos;
            }
            for swc in &mut non_narrow_chars {
                *swc = *swc - start_pos;
            }

            let local_version = local_source_map.new_imported_source_file(name,
                                                                   name_was_remapped,
                                                                   self.cnum.as_u32(),
                                                                   src_hash,
                                                                   name_hash,
                                                                   source_length,
                                                                   lines,
                                                                   multibyte_chars,
                                                                   non_narrow_chars);
            debug!("CrateMetaData::imported_source_files alloc \
                    source_file {:?} original (start_pos {:?} end_pos {:?}) \
                    translated (start_pos {:?} end_pos {:?})",
                   local_version.name, start_pos, end_pos,
                   local_version.start_pos, local_version.end_pos);

            cstore::ImportedSourceFile {
                original_start_pos: start_pos,
                original_end_pos: end_pos,
                translated_source_file: local_version,
            }
        }).collect();

        *source_map_import_info = imported_source_files;
        drop(source_map_import_info);

        // This shouldn't borrow twice, but there is no way to downgrade RefMut to Ref.
        self.source_map_import_info.borrow()
    }
}
