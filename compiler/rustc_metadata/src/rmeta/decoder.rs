// Decoding metadata from a single crate's metadata

use crate::creader::{CStore, CrateMetadataRef};
use crate::rmeta::*;

use rustc_ast as ast;
use rustc_data_structures::captures::Captures;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::svh::Svh;
use rustc_data_structures::sync::{Lock, LockGuard, Lrc, OnceCell};
use rustc_data_structures::unhash::UnhashMap;
use rustc_expand::base::{SyntaxExtension, SyntaxExtensionKind};
use rustc_expand::proc_macro::{AttrProcMacro, BangProcMacro, DeriveProcMacro};
use rustc_hir::def::{CtorKind, DefKind, Res};
use rustc_hir::def_id::{CrateNum, DefId, DefIndex, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc_hir::definitions::{DefKey, DefPath, DefPathData, DefPathHash};
use rustc_hir::diagnostic_items::DiagnosticItems;
use rustc_index::vec::{Idx, IndexVec};
use rustc_middle::metadata::ModChild;
use rustc_middle::middle::exported_symbols::{ExportedSymbol, SymbolExportInfo};
use rustc_middle::mir::interpret::{AllocDecodingSession, AllocDecodingState};
use rustc_middle::ty::codec::TyDecoder;
use rustc_middle::ty::fast_reject::SimplifiedType;
use rustc_middle::ty::GeneratorDiagnosticData;
use rustc_middle::ty::{self, ParameterizedOverTcx, Ty, TyCtxt, Visibility};
use rustc_serialize::opaque::MemDecoder;
use rustc_serialize::{Decodable, Decoder};
use rustc_session::cstore::{
    CrateSource, ExternCrate, ForeignModule, LinkagePreference, NativeLib,
};
use rustc_session::Session;
use rustc_span::hygiene::ExpnIndex;
use rustc_span::source_map::{respan, Spanned};
use rustc_span::symbol::{kw, Ident, Symbol};
use rustc_span::{self, BytePos, ExpnId, Pos, Span, SyntaxContext, DUMMY_SP};

use proc_macro::bridge::client::ProcMacro;
use std::iter::TrustedLen;
use std::num::NonZeroUsize;
use std::path::Path;
use std::{io, iter, mem};

pub(super) use cstore_impl::provide;
pub use cstore_impl::provide_extern;
use rustc_span::hygiene::HygieneDecodeContext;

mod cstore_impl;

/// A reference to the raw binary version of crate metadata.
/// A `MetadataBlob` internally is just a reference counted pointer to
/// the actual data, so cloning it is cheap.
#[derive(Clone)]
pub(crate) struct MetadataBlob(Lrc<MetadataRef>);

// This is needed so we can create an OwningRef into the blob.
// The data behind a `MetadataBlob` has a stable address because it is
// contained within an Rc/Arc.
unsafe impl rustc_data_structures::owning_ref::StableAddress for MetadataBlob {}

// This is needed so we can create an OwningRef into the blob.
impl std::ops::Deref for MetadataBlob {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        &self.0[..]
    }
}

/// A map from external crate numbers (as decoded from some crate file) to
/// local crate numbers (as generated during this session). Each external
/// crate may refer to types in other external crates, and each has their
/// own crate numbers.
pub(crate) type CrateNumMap = IndexVec<CrateNum, CrateNum>;

pub(crate) struct CrateMetadata {
    /// The primary crate data - binary metadata blob.
    blob: MetadataBlob,

    // --- Some data pre-decoded from the metadata blob, usually for performance ---
    root: CrateRoot,
    /// Trait impl data.
    /// FIXME: Used only from queries and can use query cache,
    /// so pre-decoding can probably be avoided.
    trait_impls: FxHashMap<(u32, DefIndex), LazyArray<(DefIndex, Option<SimplifiedType>)>>,
    /// Inherent impls which do not follow the normal coherence rules.
    ///
    /// These can be introduced using either `#![rustc_coherence_is_core]`
    /// or `#[rustc_allow_incoherent_impl]`.
    incoherent_impls: FxHashMap<SimplifiedType, LazyArray<DefIndex>>,
    /// Proc macro descriptions for this crate, if it's a proc macro crate.
    raw_proc_macros: Option<&'static [ProcMacro]>,
    /// Source maps for code from the crate.
    source_map_import_info: Lock<Vec<Option<ImportedSourceFile>>>,
    /// For every definition in this crate, maps its `DefPathHash` to its `DefIndex`.
    def_path_hash_map: DefPathHashMapRef<'static>,
    /// Likewise for ExpnHash.
    expn_hash_map: OnceCell<UnhashMap<ExpnHash, ExpnIndex>>,
    /// Used for decoding interpret::AllocIds in a cached & thread-safe manner.
    alloc_decoding_state: AllocDecodingState,
    /// Caches decoded `DefKey`s.
    def_key_cache: Lock<FxHashMap<DefIndex, DefKey>>,
    /// Caches decoded `DefPathHash`es.
    def_path_hash_cache: Lock<FxHashMap<DefIndex, DefPathHash>>,

    // --- Other significant crate properties ---
    /// ID of this crate, from the current compilation session's point of view.
    cnum: CrateNum,
    /// Maps crate IDs as they are were seen from this crate's compilation sessions into
    /// IDs as they are seen from the current compilation session.
    cnum_map: CrateNumMap,
    /// Same ID set as `cnum_map` plus maybe some injected crates like panic runtime.
    dependencies: Lock<Vec<CrateNum>>,
    /// How to link (or not link) this crate to the currently compiled crate.
    dep_kind: Lock<CrateDepKind>,
    /// Filesystem location of this crate.
    source: Lrc<CrateSource>,
    /// Whether or not this crate should be consider a private dependency
    /// for purposes of the 'exported_private_dependencies' lint
    private_dep: bool,
    /// The hash for the host proc macro. Used to support `-Z dual-proc-macro`.
    host_hash: Option<Svh>,

    /// Additional data used for decoding `HygieneData` (e.g. `SyntaxContext`
    /// and `ExpnId`).
    /// Note that we store a `HygieneDecodeContext` for each `CrateMetadat`. This is
    /// because `SyntaxContext` ids are not globally unique, so we need
    /// to track which ids we've decoded on a per-crate basis.
    hygiene_context: HygieneDecodeContext,

    // --- Data used only for improving diagnostics ---
    /// Information about the `extern crate` item or path that caused this crate to be loaded.
    /// If this is `None`, then the crate was injected (e.g., by the allocator).
    extern_crate: Lock<Option<ExternCrate>>,
}

/// Holds information about a rustc_span::SourceFile imported from another crate.
/// See `imported_source_file()` for more information.
#[derive(Clone)]
struct ImportedSourceFile {
    /// This SourceFile's byte-offset within the source_map of its original crate
    original_start_pos: rustc_span::BytePos,
    /// The end of this SourceFile within the source_map of its original crate
    original_end_pos: rustc_span::BytePos,
    /// The imported SourceFile's representation within the local source_map
    translated_source_file: Lrc<rustc_span::SourceFile>,
}

pub(super) struct DecodeContext<'a, 'tcx> {
    opaque: MemDecoder<'a>,
    cdata: Option<CrateMetadataRef<'a>>,
    blob: &'a MetadataBlob,
    sess: Option<&'tcx Session>,
    tcx: Option<TyCtxt<'tcx>>,

    lazy_state: LazyState,

    // Used for decoding interpret::AllocIds in a cached & thread-safe manner.
    alloc_decoding_session: Option<AllocDecodingSession<'a>>,
}

/// Abstract over the various ways one can create metadata decoders.
pub(super) trait Metadata<'a, 'tcx>: Copy {
    fn blob(self) -> &'a MetadataBlob;

    fn cdata(self) -> Option<CrateMetadataRef<'a>> {
        None
    }
    fn sess(self) -> Option<&'tcx Session> {
        None
    }
    fn tcx(self) -> Option<TyCtxt<'tcx>> {
        None
    }

    fn decoder(self, pos: usize) -> DecodeContext<'a, 'tcx> {
        let tcx = self.tcx();
        DecodeContext {
            opaque: MemDecoder::new(self.blob(), pos),
            cdata: self.cdata(),
            blob: self.blob(),
            sess: self.sess().or(tcx.map(|tcx| tcx.sess)),
            tcx,
            lazy_state: LazyState::NoNode,
            alloc_decoding_session: self
                .cdata()
                .map(|cdata| cdata.cdata.alloc_decoding_state.new_decoding_session()),
        }
    }
}

impl<'a, 'tcx> Metadata<'a, 'tcx> for &'a MetadataBlob {
    #[inline]
    fn blob(self) -> &'a MetadataBlob {
        self
    }
}

impl<'a, 'tcx> Metadata<'a, 'tcx> for (&'a MetadataBlob, &'tcx Session) {
    #[inline]
    fn blob(self) -> &'a MetadataBlob {
        self.0
    }

    #[inline]
    fn sess(self) -> Option<&'tcx Session> {
        let (_, sess) = self;
        Some(sess)
    }
}

impl<'a, 'tcx> Metadata<'a, 'tcx> for CrateMetadataRef<'a> {
    #[inline]
    fn blob(self) -> &'a MetadataBlob {
        &self.cdata.blob
    }
    #[inline]
    fn cdata(self) -> Option<CrateMetadataRef<'a>> {
        Some(self)
    }
}

impl<'a, 'tcx> Metadata<'a, 'tcx> for (CrateMetadataRef<'a>, &'tcx Session) {
    #[inline]
    fn blob(self) -> &'a MetadataBlob {
        &self.0.cdata.blob
    }
    #[inline]
    fn cdata(self) -> Option<CrateMetadataRef<'a>> {
        Some(self.0)
    }
    #[inline]
    fn sess(self) -> Option<&'tcx Session> {
        Some(self.1)
    }
}

impl<'a, 'tcx> Metadata<'a, 'tcx> for (CrateMetadataRef<'a>, TyCtxt<'tcx>) {
    #[inline]
    fn blob(self) -> &'a MetadataBlob {
        &self.0.cdata.blob
    }
    #[inline]
    fn cdata(self) -> Option<CrateMetadataRef<'a>> {
        Some(self.0)
    }
    #[inline]
    fn tcx(self) -> Option<TyCtxt<'tcx>> {
        Some(self.1)
    }
}

impl<T: ParameterizedOverTcx> LazyValue<T> {
    fn decode<'a, 'tcx, M: Metadata<'a, 'tcx>>(self, metadata: M) -> T::Value<'tcx>
    where
        T::Value<'tcx>: Decodable<DecodeContext<'a, 'tcx>>,
    {
        let mut dcx = metadata.decoder(self.position.get());
        dcx.lazy_state = LazyState::NodeStart(self.position);
        T::Value::decode(&mut dcx)
    }
}

struct DecodeIterator<'a, 'tcx, T> {
    elem_counter: std::ops::Range<usize>,
    dcx: DecodeContext<'a, 'tcx>,
    _phantom: PhantomData<fn() -> T>,
}

impl<'a, 'tcx, T: Decodable<DecodeContext<'a, 'tcx>>> Iterator for DecodeIterator<'a, 'tcx, T> {
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.elem_counter.next().map(|_| T::decode(&mut self.dcx))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.elem_counter.size_hint()
    }
}

impl<'a, 'tcx, T: Decodable<DecodeContext<'a, 'tcx>>> ExactSizeIterator
    for DecodeIterator<'a, 'tcx, T>
{
    fn len(&self) -> usize {
        self.elem_counter.len()
    }
}

unsafe impl<'a, 'tcx, T: Decodable<DecodeContext<'a, 'tcx>>> TrustedLen
    for DecodeIterator<'a, 'tcx, T>
{
}

impl<T: ParameterizedOverTcx> LazyArray<T> {
    fn decode<'a, 'tcx, M: Metadata<'a, 'tcx>>(
        self,
        metadata: M,
    ) -> DecodeIterator<'a, 'tcx, T::Value<'tcx>>
    where
        T::Value<'tcx>: Decodable<DecodeContext<'a, 'tcx>>,
    {
        let mut dcx = metadata.decoder(self.position.get());
        dcx.lazy_state = LazyState::NodeStart(self.position);
        DecodeIterator { elem_counter: (0..self.num_elems), dcx, _phantom: PhantomData }
    }
}

impl<'a, 'tcx> DecodeContext<'a, 'tcx> {
    #[inline]
    fn tcx(&self) -> TyCtxt<'tcx> {
        debug_assert!(self.tcx.is_some(), "missing TyCtxt in DecodeContext");
        self.tcx.unwrap()
    }

    #[inline]
    pub fn blob(&self) -> &'a MetadataBlob {
        self.blob
    }

    #[inline]
    pub fn cdata(&self) -> CrateMetadataRef<'a> {
        debug_assert!(self.cdata.is_some(), "missing CrateMetadata in DecodeContext");
        self.cdata.unwrap()
    }

    #[inline]
    fn map_encoded_cnum_to_current(&self, cnum: CrateNum) -> CrateNum {
        self.cdata().map_encoded_cnum_to_current(cnum)
    }

    #[inline]
    fn read_lazy_offset_then<T>(&mut self, f: impl Fn(NonZeroUsize) -> T) -> T {
        let distance = self.read_usize();
        let position = match self.lazy_state {
            LazyState::NoNode => bug!("read_lazy_with_meta: outside of a metadata node"),
            LazyState::NodeStart(start) => {
                let start = start.get();
                assert!(distance <= start);
                start - distance
            }
            LazyState::Previous(last_pos) => last_pos.get() + distance,
        };
        let position = NonZeroUsize::new(position).unwrap();
        self.lazy_state = LazyState::Previous(position);
        f(position)
    }

    fn read_lazy<T>(&mut self) -> LazyValue<T> {
        self.read_lazy_offset_then(|pos| LazyValue::from_position(pos))
    }

    fn read_lazy_array<T>(&mut self, len: usize) -> LazyArray<T> {
        self.read_lazy_offset_then(|pos| LazyArray::from_position_and_num_elems(pos, len))
    }

    fn read_lazy_table<I, T>(&mut self, len: usize) -> LazyTable<I, T> {
        self.read_lazy_offset_then(|pos| LazyTable::from_position_and_encoded_size(pos, len))
    }

    #[inline]
    pub fn read_raw_bytes(&mut self, len: usize) -> &[u8] {
        self.opaque.read_raw_bytes(len)
    }
}

impl<'a, 'tcx> TyDecoder for DecodeContext<'a, 'tcx> {
    const CLEAR_CROSS_CRATE: bool = true;

    type I = TyCtxt<'tcx>;

    #[inline]
    fn interner(&self) -> Self::I {
        self.tcx()
    }

    #[inline]
    fn peek_byte(&self) -> u8 {
        self.opaque.data[self.opaque.position()]
    }

    #[inline]
    fn position(&self) -> usize {
        self.opaque.position()
    }

    fn cached_ty_for_shorthand<F>(&mut self, shorthand: usize, or_insert_with: F) -> Ty<'tcx>
    where
        F: FnOnce(&mut Self) -> Ty<'tcx>,
    {
        let tcx = self.tcx();

        let key = ty::CReaderCacheKey { cnum: Some(self.cdata().cnum), pos: shorthand };

        if let Some(&ty) = tcx.ty_rcache.borrow().get(&key) {
            return ty;
        }

        let ty = or_insert_with(self);
        tcx.ty_rcache.borrow_mut().insert(key, ty);
        ty
    }

    fn with_position<F, R>(&mut self, pos: usize, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let new_opaque = MemDecoder::new(self.opaque.data, pos);
        let old_opaque = mem::replace(&mut self.opaque, new_opaque);
        let old_state = mem::replace(&mut self.lazy_state, LazyState::NoNode);
        let r = f(self);
        self.opaque = old_opaque;
        self.lazy_state = old_state;
        r
    }

    fn decode_alloc_id(&mut self) -> rustc_middle::mir::interpret::AllocId {
        if let Some(alloc_decoding_session) = self.alloc_decoding_session {
            alloc_decoding_session.decode_alloc_id(self)
        } else {
            bug!("Attempting to decode interpret::AllocId without CrateMetadata")
        }
    }
}

impl<'a, 'tcx> Decodable<DecodeContext<'a, 'tcx>> for CrateNum {
    fn decode(d: &mut DecodeContext<'a, 'tcx>) -> CrateNum {
        let cnum = CrateNum::from_u32(d.read_u32());
        d.map_encoded_cnum_to_current(cnum)
    }
}

impl<'a, 'tcx> Decodable<DecodeContext<'a, 'tcx>> for DefIndex {
    fn decode(d: &mut DecodeContext<'a, 'tcx>) -> DefIndex {
        DefIndex::from_u32(d.read_u32())
    }
}

impl<'a, 'tcx> Decodable<DecodeContext<'a, 'tcx>> for ExpnIndex {
    fn decode(d: &mut DecodeContext<'a, 'tcx>) -> ExpnIndex {
        ExpnIndex::from_u32(d.read_u32())
    }
}

impl<'a, 'tcx> Decodable<DecodeContext<'a, 'tcx>> for ast::AttrId {
    fn decode(d: &mut DecodeContext<'a, 'tcx>) -> ast::AttrId {
        let sess = d.sess.expect("can't decode AttrId without Session");
        sess.parse_sess.attr_id_generator.mk_attr_id()
    }
}

impl<'a, 'tcx> Decodable<DecodeContext<'a, 'tcx>> for SyntaxContext {
    fn decode(decoder: &mut DecodeContext<'a, 'tcx>) -> SyntaxContext {
        let cdata = decoder.cdata();
        let sess = decoder.sess.unwrap();
        let cname = cdata.root.name;
        rustc_span::hygiene::decode_syntax_context(decoder, &cdata.hygiene_context, |_, id| {
            debug!("SpecializedDecoder<SyntaxContext>: decoding {}", id);
            cdata
                .root
                .syntax_contexts
                .get(cdata, id)
                .unwrap_or_else(|| panic!("Missing SyntaxContext {id:?} for crate {cname:?}"))
                .decode((cdata, sess))
        })
    }
}

impl<'a, 'tcx> Decodable<DecodeContext<'a, 'tcx>> for ExpnId {
    fn decode(decoder: &mut DecodeContext<'a, 'tcx>) -> ExpnId {
        let local_cdata = decoder.cdata();
        let sess = decoder.sess.unwrap();

        let cnum = CrateNum::decode(decoder);
        let index = u32::decode(decoder);

        let expn_id = rustc_span::hygiene::decode_expn_id(cnum, index, |expn_id| {
            let ExpnId { krate: cnum, local_id: index } = expn_id;
            // Lookup local `ExpnData`s in our own crate data. Foreign `ExpnData`s
            // are stored in the owning crate, to avoid duplication.
            debug_assert_ne!(cnum, LOCAL_CRATE);
            let crate_data = if cnum == local_cdata.cnum {
                local_cdata
            } else {
                local_cdata.cstore.get_crate_data(cnum)
            };
            let expn_data = crate_data
                .root
                .expn_data
                .get(crate_data, index)
                .unwrap()
                .decode((crate_data, sess));
            let expn_hash = crate_data
                .root
                .expn_hashes
                .get(crate_data, index)
                .unwrap()
                .decode((crate_data, sess));
            (expn_data, expn_hash)
        });
        expn_id
    }
}

impl<'a, 'tcx> Decodable<DecodeContext<'a, 'tcx>> for Span {
    fn decode(decoder: &mut DecodeContext<'a, 'tcx>) -> Span {
        let ctxt = SyntaxContext::decode(decoder);
        let tag = u8::decode(decoder);

        if tag == TAG_PARTIAL_SPAN {
            return DUMMY_SP.with_ctxt(ctxt);
        }

        debug_assert!(tag == TAG_VALID_SPAN_LOCAL || tag == TAG_VALID_SPAN_FOREIGN);

        let lo = BytePos::decode(decoder);
        let len = BytePos::decode(decoder);
        let hi = lo + len;

        let Some(sess) = decoder.sess else {
            bug!("Cannot decode Span without Session.")
        };

        // Index of the file in the corresponding crate's list of encoded files.
        let metadata_index = u32::decode(decoder);

        // There are two possibilities here:
        // 1. This is a 'local span', which is located inside a `SourceFile`
        // that came from this crate. In this case, we use the source map data
        // encoded in this crate. This branch should be taken nearly all of the time.
        // 2. This is a 'foreign span', which is located inside a `SourceFile`
        // that came from a *different* crate (some crate upstream of the one
        // whose metadata we're looking at). For example, consider this dependency graph:
        //
        // A -> B -> C
        //
        // Suppose that we're currently compiling crate A, and start deserializing
        // metadata from crate B. When we deserialize a Span from crate B's metadata,
        // there are two possibilities:
        //
        // 1. The span references a file from crate B. This makes it a 'local' span,
        // which means that we can use crate B's serialized source map information.
        // 2. The span references a file from crate C. This makes it a 'foreign' span,
        // which means we need to use Crate *C* (not crate B) to determine the source
        // map information. We only record source map information for a file in the
        // crate that 'owns' it, so deserializing a Span may require us to look at
        // a transitive dependency.
        //
        // When we encode a foreign span, we adjust its 'lo' and 'high' values
        // to be based on the *foreign* crate (e.g. crate C), not the crate
        // we are writing metadata for (e.g. crate B). This allows us to
        // treat the 'local' and 'foreign' cases almost identically during deserialization:
        // we can call `imported_source_file` for the proper crate, and binary search
        // through the returned slice using our span.
        let source_file = if tag == TAG_VALID_SPAN_LOCAL {
            decoder.cdata().imported_source_file(metadata_index, sess)
        } else {
            // When we encode a proc-macro crate, all `Span`s should be encoded
            // with `TAG_VALID_SPAN_LOCAL`
            if decoder.cdata().root.is_proc_macro_crate() {
                // Decode `CrateNum` as u32 - using `CrateNum::decode` will ICE
                // since we don't have `cnum_map` populated.
                let cnum = u32::decode(decoder);
                panic!(
                    "Decoding of crate {:?} tried to access proc-macro dep {:?}",
                    decoder.cdata().root.name,
                    cnum
                );
            }
            // tag is TAG_VALID_SPAN_FOREIGN, checked by `debug_assert` above
            let cnum = CrateNum::decode(decoder);
            debug!(
                "SpecializedDecoder<Span>::specialized_decode: loading source files from cnum {:?}",
                cnum
            );

            let foreign_data = decoder.cdata().cstore.get_crate_data(cnum);
            foreign_data.imported_source_file(metadata_index, sess)
        };

        // Make sure our span is well-formed.
        debug_assert!(
            lo + source_file.original_start_pos <= source_file.original_end_pos,
            "Malformed encoded span: lo={:?} source_file.original_start_pos={:?} source_file.original_end_pos={:?}",
            lo,
            source_file.original_start_pos,
            source_file.original_end_pos
        );

        // Make sure we correctly filtered out invalid spans during encoding.
        debug_assert!(
            hi + source_file.original_start_pos <= source_file.original_end_pos,
            "Malformed encoded span: hi={:?} source_file.original_start_pos={:?} source_file.original_end_pos={:?}",
            hi,
            source_file.original_start_pos,
            source_file.original_end_pos
        );

        let lo = lo + source_file.translated_source_file.start_pos;
        let hi = hi + source_file.translated_source_file.start_pos;

        // Do not try to decode parent for foreign spans.
        Span::new(lo, hi, ctxt, None)
    }
}

impl<'a, 'tcx> Decodable<DecodeContext<'a, 'tcx>> for Symbol {
    fn decode(d: &mut DecodeContext<'a, 'tcx>) -> Self {
        let tag = d.read_u8();

        match tag {
            SYMBOL_STR => {
                let s = d.read_str();
                Symbol::intern(s)
            }
            SYMBOL_OFFSET => {
                // read str offset
                let pos = d.read_usize();
                let old_pos = d.opaque.position();

                // move to str ofset and read
                d.opaque.set_position(pos);
                let s = d.read_str();
                let sym = Symbol::intern(s);

                // restore position
                d.opaque.set_position(old_pos);

                sym
            }
            SYMBOL_PREINTERNED => {
                let symbol_index = d.read_u32();
                Symbol::new_from_decoded(symbol_index)
            }
            _ => unreachable!(),
        }
    }
}

impl<'a, 'tcx> Decodable<DecodeContext<'a, 'tcx>> for &'tcx [(ty::Predicate<'tcx>, Span)] {
    fn decode(d: &mut DecodeContext<'a, 'tcx>) -> Self {
        ty::codec::RefDecodable::decode(d)
    }
}

impl<'a, 'tcx, T> Decodable<DecodeContext<'a, 'tcx>> for LazyValue<T> {
    fn decode(decoder: &mut DecodeContext<'a, 'tcx>) -> Self {
        decoder.read_lazy()
    }
}

impl<'a, 'tcx, T> Decodable<DecodeContext<'a, 'tcx>> for LazyArray<T> {
    fn decode(decoder: &mut DecodeContext<'a, 'tcx>) -> Self {
        let len = decoder.read_usize();
        if len == 0 { LazyArray::default() } else { decoder.read_lazy_array(len) }
    }
}

impl<'a, 'tcx, I: Idx, T> Decodable<DecodeContext<'a, 'tcx>> for LazyTable<I, T> {
    fn decode(decoder: &mut DecodeContext<'a, 'tcx>) -> Self {
        let len = decoder.read_usize();
        decoder.read_lazy_table(len)
    }
}

implement_ty_decoder!(DecodeContext<'a, 'tcx>);

impl MetadataBlob {
    pub(crate) fn new(metadata_ref: MetadataRef) -> MetadataBlob {
        MetadataBlob(Lrc::new(metadata_ref))
    }

    pub(crate) fn is_compatible(&self) -> bool {
        self.blob().starts_with(METADATA_HEADER)
    }

    pub(crate) fn get_rustc_version(&self) -> String {
        LazyValue::<String>::from_position(NonZeroUsize::new(METADATA_HEADER.len() + 4).unwrap())
            .decode(self)
    }

    pub(crate) fn get_root(&self) -> CrateRoot {
        let slice = &self.blob()[..];
        let offset = METADATA_HEADER.len();

        let pos_bytes = slice[offset..][..4].try_into().unwrap();
        let pos = u32::from_be_bytes(pos_bytes) as usize;

        LazyValue::<CrateRoot>::from_position(NonZeroUsize::new(pos).unwrap()).decode(self)
    }

    pub(crate) fn list_crate_metadata(&self, out: &mut dyn io::Write) -> io::Result<()> {
        let root = self.get_root();
        writeln!(out, "Crate info:")?;
        writeln!(out, "name {}{}", root.name, root.extra_filename)?;
        writeln!(out, "hash {} stable_crate_id {:?}", root.hash, root.stable_crate_id)?;
        writeln!(out, "proc_macro {:?}", root.proc_macro_data.is_some())?;
        writeln!(out, "=External Dependencies=")?;

        for (i, dep) in root.crate_deps.decode(self).enumerate() {
            let CrateDep { name, extra_filename, hash, host_hash, kind } = dep;
            let number = i + 1;

            writeln!(
                out,
                "{number} {name}{extra_filename} hash {hash} host_hash {host_hash:?} kind {kind:?}"
            )?;
        }
        write!(out, "\n")?;
        Ok(())
    }
}

impl CrateRoot {
    pub(crate) fn is_proc_macro_crate(&self) -> bool {
        self.proc_macro_data.is_some()
    }

    pub(crate) fn name(&self) -> Symbol {
        self.name
    }

    pub(crate) fn hash(&self) -> Svh {
        self.hash
    }

    pub(crate) fn stable_crate_id(&self) -> StableCrateId {
        self.stable_crate_id
    }

    pub(crate) fn triple(&self) -> &TargetTriple {
        &self.triple
    }

    pub(crate) fn decode_crate_deps<'a>(
        &self,
        metadata: &'a MetadataBlob,
    ) -> impl ExactSizeIterator<Item = CrateDep> + Captures<'a> {
        self.crate_deps.decode(metadata)
    }
}

impl<'a, 'tcx> CrateMetadataRef<'a> {
    fn raw_proc_macro(self, id: DefIndex) -> &'a ProcMacro {
        // DefIndex's in root.proc_macro_data have a one-to-one correspondence
        // with items in 'raw_proc_macros'.
        let pos = self
            .root
            .proc_macro_data
            .as_ref()
            .unwrap()
            .macros
            .decode(self)
            .position(|i| i == id)
            .unwrap();
        &self.raw_proc_macros.unwrap()[pos]
    }

    fn opt_item_name(self, item_index: DefIndex) -> Option<Symbol> {
        let def_key = self.def_key(item_index);
        def_key.disambiguated_data.data.get_opt_name().or_else(|| {
            if def_key.disambiguated_data.data == DefPathData::Ctor {
                let parent_index = def_key.parent.expect("no parent for a constructor");
                self.def_key(parent_index).disambiguated_data.data.get_opt_name()
            } else {
                None
            }
        })
    }

    fn item_name(self, item_index: DefIndex) -> Symbol {
        self.opt_item_name(item_index).expect("no encoded ident for item")
    }

    fn opt_item_ident(self, item_index: DefIndex, sess: &Session) -> Option<Ident> {
        let name = self.opt_item_name(item_index)?;
        let span =
            self.root.tables.def_ident_span.get(self, item_index).unwrap().decode((self, sess));
        Some(Ident::new(name, span))
    }

    fn item_ident(self, item_index: DefIndex, sess: &Session) -> Ident {
        self.opt_item_ident(item_index, sess).expect("no encoded ident for item")
    }

    #[inline]
    pub(super) fn map_encoded_cnum_to_current(self, cnum: CrateNum) -> CrateNum {
        if cnum == LOCAL_CRATE { self.cnum } else { self.cnum_map[cnum] }
    }

    fn def_kind(self, item_id: DefIndex) -> DefKind {
        self.root.tables.opt_def_kind.get(self, item_id).unwrap_or_else(|| {
            bug!(
                "CrateMetadata::def_kind({:?}): id not found, in crate {:?} with number {}",
                item_id,
                self.root.name,
                self.cnum,
            )
        })
    }

    fn get_span(self, index: DefIndex, sess: &Session) -> Span {
        self.root
            .tables
            .def_span
            .get(self, index)
            .unwrap_or_else(|| panic!("Missing span for {index:?}"))
            .decode((self, sess))
    }

    fn load_proc_macro(self, id: DefIndex, sess: &Session) -> SyntaxExtension {
        let (name, kind, helper_attrs) = match *self.raw_proc_macro(id) {
            ProcMacro::CustomDerive { trait_name, attributes, client } => {
                let helper_attrs =
                    attributes.iter().cloned().map(Symbol::intern).collect::<Vec<_>>();
                (
                    trait_name,
                    SyntaxExtensionKind::Derive(Box::new(DeriveProcMacro { client })),
                    helper_attrs,
                )
            }
            ProcMacro::Attr { name, client } => {
                (name, SyntaxExtensionKind::Attr(Box::new(AttrProcMacro { client })), Vec::new())
            }
            ProcMacro::Bang { name, client } => {
                (name, SyntaxExtensionKind::Bang(Box::new(BangProcMacro { client })), Vec::new())
            }
        };

        let attrs: Vec<_> = self.get_item_attrs(id, sess).collect();
        SyntaxExtension::new(
            sess,
            kind,
            self.get_span(id, sess),
            helper_attrs,
            self.root.edition,
            Symbol::intern(name),
            &attrs,
        )
    }

    fn get_variant(self, kind: &DefKind, index: DefIndex, parent_did: DefId) -> ty::VariantDef {
        let adt_kind = match kind {
            DefKind::Variant => ty::AdtKind::Enum,
            DefKind::Struct => ty::AdtKind::Struct,
            DefKind::Union => ty::AdtKind::Union,
            _ => bug!(),
        };

        let data = self.root.tables.variant_data.get(self, index).unwrap().decode(self);

        let variant_did =
            if adt_kind == ty::AdtKind::Enum { Some(self.local_def_id(index)) } else { None };
        let ctor = data.ctor.map(|(kind, index)| (kind, self.local_def_id(index)));

        ty::VariantDef::new(
            self.item_name(index),
            variant_did,
            ctor,
            data.discr,
            self.root
                .tables
                .children
                .get(self, index)
                .expect("fields are not encoded for a variant")
                .decode(self)
                .map(|index| ty::FieldDef {
                    did: self.local_def_id(index),
                    name: self.item_name(index),
                    vis: self.get_visibility(index),
                })
                .collect(),
            adt_kind,
            parent_did,
            false,
            data.is_non_exhaustive,
        )
    }

    fn get_adt_def(self, item_id: DefIndex, tcx: TyCtxt<'tcx>) -> ty::AdtDef<'tcx> {
        let kind = self.def_kind(item_id);
        let did = self.local_def_id(item_id);

        let adt_kind = match kind {
            DefKind::Enum => ty::AdtKind::Enum,
            DefKind::Struct => ty::AdtKind::Struct,
            DefKind::Union => ty::AdtKind::Union,
            _ => bug!("get_adt_def called on a non-ADT {:?}", did),
        };
        let repr = self.root.tables.repr_options.get(self, item_id).unwrap().decode(self);

        let variants = if let ty::AdtKind::Enum = adt_kind {
            self.root
                .tables
                .children
                .get(self, item_id)
                .expect("variants are not encoded for an enum")
                .decode(self)
                .filter_map(|index| {
                    let kind = self.def_kind(index);
                    match kind {
                        DefKind::Ctor(..) => None,
                        _ => Some(self.get_variant(&kind, index, did)),
                    }
                })
                .collect()
        } else {
            std::iter::once(self.get_variant(&kind, item_id, did)).collect()
        };

        tcx.alloc_adt_def(did, adt_kind, variants, repr)
    }

    fn get_generics(self, item_id: DefIndex, sess: &Session) -> ty::Generics {
        self.root.tables.generics_of.get(self, item_id).unwrap().decode((self, sess))
    }

    fn get_visibility(self, id: DefIndex) -> ty::Visibility<DefId> {
        self.root
            .tables
            .visibility
            .get(self, id)
            .unwrap()
            .decode(self)
            .map_id(|index| self.local_def_id(index))
    }

    fn get_trait_item_def_id(self, id: DefIndex) -> Option<DefId> {
        self.root.tables.trait_item_def_id.get(self, id).map(|d| d.decode_from_cdata(self))
    }

    fn get_expn_that_defined(self, id: DefIndex, sess: &Session) -> ExpnId {
        self.root.tables.expn_that_defined.get(self, id).unwrap().decode((self, sess))
    }

    fn get_debugger_visualizers(self) -> Vec<rustc_span::DebuggerVisualizerFile> {
        self.root.debugger_visualizers.decode(self).collect::<Vec<_>>()
    }

    /// Iterates over all the stability attributes in the given crate.
    fn get_lib_features(self, tcx: TyCtxt<'tcx>) -> &'tcx [(Symbol, Option<Symbol>)] {
        tcx.arena.alloc_from_iter(self.root.lib_features.decode(self))
    }

    /// Iterates over the stability implications in the given crate (when a `#[unstable]` attribute
    /// has an `implied_by` meta item, then the mapping from the implied feature to the actual
    /// feature is a stability implication).
    fn get_stability_implications(self, tcx: TyCtxt<'tcx>) -> &'tcx [(Symbol, Symbol)] {
        tcx.arena.alloc_from_iter(self.root.stability_implications.decode(self))
    }

    /// Iterates over the language items in the given crate.
    fn get_lang_items(self, tcx: TyCtxt<'tcx>) -> &'tcx [(DefId, LangItem)] {
        tcx.arena.alloc_from_iter(
            self.root
                .lang_items
                .decode(self)
                .map(move |(def_index, index)| (self.local_def_id(def_index), index)),
        )
    }

    /// Iterates over the diagnostic items in the given crate.
    fn get_diagnostic_items(self) -> DiagnosticItems {
        let mut id_to_name = FxHashMap::default();
        let name_to_id = self
            .root
            .diagnostic_items
            .decode(self)
            .map(|(name, def_index)| {
                let id = self.local_def_id(def_index);
                id_to_name.insert(id, name);
                (name, id)
            })
            .collect();
        DiagnosticItems { id_to_name, name_to_id }
    }

    fn get_mod_child(self, id: DefIndex, sess: &Session) -> ModChild {
        let ident = self.item_ident(id, sess);
        let kind = self.def_kind(id);
        let def_id = self.local_def_id(id);
        let res = Res::Def(kind, def_id);
        let vis = self.get_visibility(id);
        let span = self.get_span(id, sess);
        let macro_rules = match kind {
            DefKind::Macro(..) => self.root.tables.is_macro_rules.get(self, id),
            _ => false,
        };

        ModChild { ident, res, vis, span, macro_rules }
    }

    /// Iterates over all named children of the given module,
    /// including both proper items and reexports.
    /// Module here is understood in name resolution sense - it can be a `mod` item,
    /// or a crate root, or an enum, or a trait.
    fn get_module_children(
        self,
        id: DefIndex,
        sess: &'a Session,
    ) -> impl Iterator<Item = ModChild> + 'a {
        iter::from_generator(move || {
            if let Some(data) = &self.root.proc_macro_data {
                // If we are loading as a proc macro, we want to return
                // the view of this crate as a proc macro crate.
                if id == CRATE_DEF_INDEX {
                    for child_index in data.macros.decode(self) {
                        yield self.get_mod_child(child_index, sess);
                    }
                }
            } else {
                // Iterate over all children.
                for child_index in self.root.tables.children.get(self, id).unwrap().decode(self) {
                    yield self.get_mod_child(child_index, sess);
                }

                if let Some(reexports) = self.root.tables.module_reexports.get(self, id) {
                    for reexport in reexports.decode((self, sess)) {
                        yield reexport;
                    }
                }
            }
        })
    }

    fn is_ctfe_mir_available(self, id: DefIndex) -> bool {
        self.root.tables.mir_for_ctfe.get(self, id).is_some()
    }

    fn is_item_mir_available(self, id: DefIndex) -> bool {
        self.root.tables.optimized_mir.get(self, id).is_some()
    }

    fn module_expansion(self, id: DefIndex, sess: &Session) -> ExpnId {
        match self.def_kind(id) {
            DefKind::Mod | DefKind::Enum | DefKind::Trait => self.get_expn_that_defined(id, sess),
            _ => panic!("Expected module, found {:?}", self.local_def_id(id)),
        }
    }

    fn get_fn_has_self_parameter(self, id: DefIndex, sess: &'a Session) -> bool {
        self.root
            .tables
            .fn_arg_names
            .get(self, id)
            .expect("argument names not encoded for a function")
            .decode((self, sess))
            .nth(0)
            .map_or(false, |ident| ident.name == kw::SelfLower)
    }

    fn get_associated_item_def_ids(
        self,
        id: DefIndex,
        sess: &'a Session,
    ) -> impl Iterator<Item = DefId> + 'a {
        self.root
            .tables
            .children
            .get(self, id)
            .expect("associated items not encoded for an item")
            .decode((self, sess))
            .map(move |child_index| self.local_def_id(child_index))
    }

    fn get_associated_item(self, id: DefIndex, sess: &'a Session) -> ty::AssocItem {
        let name = self.item_name(id);

        let (kind, has_self) = match self.def_kind(id) {
            DefKind::AssocConst => (ty::AssocKind::Const, false),
            DefKind::AssocFn => (ty::AssocKind::Fn, self.get_fn_has_self_parameter(id, sess)),
            DefKind::AssocTy => (ty::AssocKind::Type, false),
            _ => bug!("cannot get associated-item of `{:?}`", self.def_key(id)),
        };
        let container = self.root.tables.assoc_container.get(self, id).unwrap();

        ty::AssocItem {
            name,
            kind,
            def_id: self.local_def_id(id),
            trait_item_def_id: self.get_trait_item_def_id(id),
            container,
            fn_has_self_parameter: has_self,
        }
    }

    fn get_ctor(self, node_id: DefIndex) -> Option<(CtorKind, DefId)> {
        match self.def_kind(node_id) {
            DefKind::Struct | DefKind::Variant => {
                let vdata = self.root.tables.variant_data.get(self, node_id).unwrap().decode(self);
                vdata.ctor.map(|(kind, index)| (kind, self.local_def_id(index)))
            }
            _ => None,
        }
    }

    fn get_item_attrs(
        self,
        id: DefIndex,
        sess: &'a Session,
    ) -> impl Iterator<Item = ast::Attribute> + 'a {
        self.root
            .tables
            .attributes
            .get(self, id)
            .unwrap_or_else(|| {
                // Structure and variant constructors don't have any attributes encoded for them,
                // but we assume that someone passing a constructor ID actually wants to look at
                // the attributes on the corresponding struct or variant.
                let def_key = self.def_key(id);
                assert_eq!(def_key.disambiguated_data.data, DefPathData::Ctor);
                let parent_id = def_key.parent.expect("no parent for a constructor");
                self.root
                    .tables
                    .attributes
                    .get(self, parent_id)
                    .expect("no encoded attributes for a structure or variant")
            })
            .decode((self, sess))
    }

    fn get_struct_field_names(
        self,
        id: DefIndex,
        sess: &'a Session,
    ) -> impl Iterator<Item = Spanned<Symbol>> + 'a {
        self.root
            .tables
            .children
            .get(self, id)
            .expect("fields not encoded for a struct")
            .decode(self)
            .map(move |index| respan(self.get_span(index, sess), self.item_name(index)))
    }

    fn get_struct_field_visibilities(
        self,
        id: DefIndex,
    ) -> impl Iterator<Item = Visibility<DefId>> + 'a {
        self.root
            .tables
            .children
            .get(self, id)
            .expect("fields not encoded for a struct")
            .decode(self)
            .map(move |field_index| self.get_visibility(field_index))
    }

    fn get_inherent_implementations_for_type(
        self,
        tcx: TyCtxt<'tcx>,
        id: DefIndex,
    ) -> &'tcx [DefId] {
        tcx.arena.alloc_from_iter(
            self.root
                .tables
                .inherent_impls
                .get(self, id)
                .decode(self)
                .map(|index| self.local_def_id(index)),
        )
    }

    /// Decodes all inherent impls in the crate (for rustdoc).
    fn get_inherent_impls(self) -> impl Iterator<Item = (DefId, DefId)> + 'a {
        (0..self.root.tables.inherent_impls.size()).flat_map(move |i| {
            let ty_index = DefIndex::from_usize(i);
            let ty_def_id = self.local_def_id(ty_index);
            self.root
                .tables
                .inherent_impls
                .get(self, ty_index)
                .decode(self)
                .map(move |impl_index| (ty_def_id, self.local_def_id(impl_index)))
        })
    }

    /// Decodes all traits in the crate (for rustdoc and rustc diagnostics).
    fn get_traits(self) -> impl Iterator<Item = DefId> + 'a {
        self.root.traits.decode(self).map(move |index| self.local_def_id(index))
    }

    /// Decodes all trait impls in the crate (for rustdoc).
    fn get_trait_impls(self) -> impl Iterator<Item = (DefId, DefId, Option<SimplifiedType>)> + 'a {
        self.cdata.trait_impls.iter().flat_map(move |(&(trait_cnum_raw, trait_index), impls)| {
            let trait_def_id = DefId {
                krate: self.cnum_map[CrateNum::from_u32(trait_cnum_raw)],
                index: trait_index,
            };
            impls.decode(self).map(move |(impl_index, simplified_self_ty)| {
                (trait_def_id, self.local_def_id(impl_index), simplified_self_ty)
            })
        })
    }

    fn get_all_incoherent_impls(self) -> impl Iterator<Item = DefId> + 'a {
        self.cdata
            .incoherent_impls
            .values()
            .flat_map(move |impls| impls.decode(self).map(move |idx| self.local_def_id(idx)))
    }

    fn get_incoherent_impls(self, tcx: TyCtxt<'tcx>, simp: SimplifiedType) -> &'tcx [DefId] {
        if let Some(impls) = self.cdata.incoherent_impls.get(&simp) {
            tcx.arena.alloc_from_iter(impls.decode(self).map(|idx| self.local_def_id(idx)))
        } else {
            &[]
        }
    }

    fn get_implementations_of_trait(
        self,
        tcx: TyCtxt<'tcx>,
        trait_def_id: DefId,
    ) -> &'tcx [(DefId, Option<SimplifiedType>)] {
        if self.trait_impls.is_empty() {
            return &[];
        }

        // Do a reverse lookup beforehand to avoid touching the crate_num
        // hash map in the loop below.
        let key = match self.reverse_translate_def_id(trait_def_id) {
            Some(def_id) => (def_id.krate.as_u32(), def_id.index),
            None => return &[],
        };

        if let Some(impls) = self.trait_impls.get(&key) {
            tcx.arena.alloc_from_iter(
                impls
                    .decode(self)
                    .map(|(idx, simplified_self_ty)| (self.local_def_id(idx), simplified_self_ty)),
            )
        } else {
            &[]
        }
    }

    fn get_native_libraries(self, sess: &'a Session) -> impl Iterator<Item = NativeLib> + 'a {
        self.root.native_libraries.decode((self, sess))
    }

    fn get_proc_macro_quoted_span(self, index: usize, sess: &Session) -> Span {
        self.root
            .tables
            .proc_macro_quoted_spans
            .get(self, index)
            .unwrap_or_else(|| panic!("Missing proc macro quoted span: {index:?}"))
            .decode((self, sess))
    }

    fn get_foreign_modules(self, sess: &'a Session) -> impl Iterator<Item = ForeignModule> + '_ {
        self.root.foreign_modules.decode((self, sess))
    }

    fn get_dylib_dependency_formats(
        self,
        tcx: TyCtxt<'tcx>,
    ) -> &'tcx [(CrateNum, LinkagePreference)] {
        tcx.arena.alloc_from_iter(
            self.root.dylib_dependency_formats.decode(self).enumerate().flat_map(|(i, link)| {
                let cnum = CrateNum::new(i + 1);
                link.map(|link| (self.cnum_map[cnum], link))
            }),
        )
    }

    fn get_missing_lang_items(self, tcx: TyCtxt<'tcx>) -> &'tcx [LangItem] {
        tcx.arena.alloc_from_iter(self.root.lang_items_missing.decode(self))
    }

    fn exported_symbols(
        self,
        tcx: TyCtxt<'tcx>,
    ) -> &'tcx [(ExportedSymbol<'tcx>, SymbolExportInfo)] {
        tcx.arena.alloc_from_iter(self.root.exported_symbols.decode((self, tcx)))
    }

    fn get_macro(self, id: DefIndex, sess: &Session) -> ast::MacroDef {
        match self.def_kind(id) {
            DefKind::Macro(_) => {
                let macro_rules = self.root.tables.is_macro_rules.get(self, id);
                let body =
                    self.root.tables.macro_definition.get(self, id).unwrap().decode((self, sess));
                ast::MacroDef { macro_rules, body: ast::ptr::P(body) }
            }
            _ => bug!(),
        }
    }

    fn is_foreign_item(self, id: DefIndex) -> bool {
        if let Some(parent) = self.def_key(id).parent {
            matches!(self.def_kind(parent), DefKind::ForeignMod)
        } else {
            false
        }
    }

    #[inline]
    fn def_key(self, index: DefIndex) -> DefKey {
        *self
            .def_key_cache
            .lock()
            .entry(index)
            .or_insert_with(|| self.root.tables.def_keys.get(self, index).unwrap().decode(self))
    }

    // Returns the path leading to the thing with this `id`.
    fn def_path(self, id: DefIndex) -> DefPath {
        debug!("def_path(cnum={:?}, id={:?})", self.cnum, id);
        DefPath::make(self.cnum, id, |parent| self.def_key(parent))
    }

    fn def_path_hash_unlocked(
        self,
        index: DefIndex,
        def_path_hashes: &mut FxHashMap<DefIndex, DefPathHash>,
    ) -> DefPathHash {
        *def_path_hashes
            .entry(index)
            .or_insert_with(|| self.root.tables.def_path_hashes.get(self, index))
    }

    #[inline]
    fn def_path_hash(self, index: DefIndex) -> DefPathHash {
        let mut def_path_hashes = self.def_path_hash_cache.lock();
        self.def_path_hash_unlocked(index, &mut def_path_hashes)
    }

    #[inline]
    fn def_path_hash_to_def_index(self, hash: DefPathHash) -> DefIndex {
        self.def_path_hash_map.def_path_hash_to_def_index(&hash)
    }

    fn expn_hash_to_expn_id(self, sess: &Session, index_guess: u32, hash: ExpnHash) -> ExpnId {
        debug_assert_eq!(ExpnId::from_hash(hash), None);
        let index_guess = ExpnIndex::from_u32(index_guess);
        let old_hash = self.root.expn_hashes.get(self, index_guess).map(|lazy| lazy.decode(self));

        let index = if old_hash == Some(hash) {
            // Fast path: the expn and its index is unchanged from the
            // previous compilation session. There is no need to decode anything
            // else.
            index_guess
        } else {
            // Slow path: We need to find out the new `DefIndex` of the provided
            // `DefPathHash`, if its still exists. This requires decoding every `DefPathHash`
            // stored in this crate.
            let map = self.cdata.expn_hash_map.get_or_init(|| {
                let end_id = self.root.expn_hashes.size() as u32;
                let mut map =
                    UnhashMap::with_capacity_and_hasher(end_id as usize, Default::default());
                for i in 0..end_id {
                    let i = ExpnIndex::from_u32(i);
                    if let Some(hash) = self.root.expn_hashes.get(self, i) {
                        map.insert(hash.decode(self), i);
                    }
                }
                map
            });
            map[&hash]
        };

        let data = self.root.expn_data.get(self, index).unwrap().decode((self, sess));
        rustc_span::hygiene::register_expn_id(self.cnum, index, data, hash)
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
    fn imported_source_file(self, source_file_index: u32, sess: &Session) -> ImportedSourceFile {
        fn filter<'a>(sess: &Session, path: Option<&'a Path>) -> Option<&'a Path> {
            path.filter(|_| {
                // Only spend time on further checks if we have what to translate *to*.
                sess.opts.real_rust_source_base_dir.is_some()
                // Some tests need the translation to be always skipped.
                && sess.opts.unstable_opts.translate_remapped_path_to_local_path
            })
            .filter(|virtual_dir| {
                // Don't translate away `/rustc/$hash` if we're still remapping to it,
                // since that means we're still building `std`/`rustc` that need it,
                // and we don't want the real path to leak into codegen/debuginfo.
                !sess.opts.remap_path_prefix.iter().any(|(_from, to)| to == virtual_dir)
            })
        }

        // Translate the virtual `/rustc/$hash` prefix back to a real directory
        // that should hold actual sources, where possible.
        //
        // NOTE: if you update this, you might need to also update bootstrap's code for generating
        // the `rust-src` component in `Src::run` in `src/bootstrap/dist.rs`.
        let virtual_rust_source_base_dir = [
            filter(sess, option_env!("CFG_VIRTUAL_RUST_SOURCE_BASE_DIR").map(Path::new)),
            filter(sess, sess.opts.unstable_opts.simulate_remapped_rust_src_base.as_deref()),
        ];

        let try_to_translate_virtual_to_real = |name: &mut rustc_span::FileName| {
            debug!(
                "try_to_translate_virtual_to_real(name={:?}): \
                 virtual_rust_source_base_dir={:?}, real_rust_source_base_dir={:?}",
                name, virtual_rust_source_base_dir, sess.opts.real_rust_source_base_dir,
            );

            for virtual_dir in virtual_rust_source_base_dir.iter().flatten() {
                if let Some(real_dir) = &sess.opts.real_rust_source_base_dir {
                    if let rustc_span::FileName::Real(old_name) = name {
                        if let rustc_span::RealFileName::Remapped { local_path: _, virtual_name } =
                            old_name
                        {
                            if let Ok(rest) = virtual_name.strip_prefix(virtual_dir) {
                                let virtual_name = virtual_name.clone();

                                // The std library crates are in
                                // `$sysroot/lib/rustlib/src/rust/library`, whereas other crates
                                // may be in `$sysroot/lib/rustlib/src/rust/` directly. So we
                                // detect crates from the std libs and handle them specially.
                                const STD_LIBS: &[&str] = &[
                                    "core",
                                    "alloc",
                                    "std",
                                    "test",
                                    "term",
                                    "unwind",
                                    "proc_macro",
                                    "panic_abort",
                                    "panic_unwind",
                                    "profiler_builtins",
                                    "rtstartup",
                                    "rustc-std-workspace-core",
                                    "rustc-std-workspace-alloc",
                                    "rustc-std-workspace-std",
                                    "backtrace",
                                ];
                                let is_std_lib = STD_LIBS.iter().any(|l| rest.starts_with(l));

                                let new_path = if is_std_lib {
                                    real_dir.join("library").join(rest)
                                } else {
                                    real_dir.join(rest)
                                };

                                debug!(
                                    "try_to_translate_virtual_to_real: `{}` -> `{}`",
                                    virtual_name.display(),
                                    new_path.display(),
                                );
                                let new_name = rustc_span::RealFileName::Remapped {
                                    local_path: Some(new_path),
                                    virtual_name,
                                };
                                *old_name = new_name;
                            }
                        }
                    }
                }
            }
        };

        let mut import_info = self.cdata.source_map_import_info.lock();
        for _ in import_info.len()..=(source_file_index as usize) {
            import_info.push(None);
        }
        import_info[source_file_index as usize]
            .get_or_insert_with(|| {
                let source_file_to_import = self
                    .root
                    .source_map
                    .get(self, source_file_index)
                    .expect("missing source file")
                    .decode(self);

                // We can't reuse an existing SourceFile, so allocate a new one
                // containing the information we need.
                let rustc_span::SourceFile {
                    mut name,
                    src_hash,
                    start_pos,
                    end_pos,
                    lines,
                    multibyte_chars,
                    non_narrow_chars,
                    normalized_pos,
                    name_hash,
                    ..
                } = source_file_to_import;

                // If this file is under $sysroot/lib/rustlib/src/ but has not been remapped
                // during rust bootstrapping by `remap-debuginfo = true`, and the user
                // wish to simulate that behaviour by -Z simulate-remapped-rust-src-base,
                // then we change `name` to a similar state as if the rust was bootstrapped
                // with `remap-debuginfo = true`.
                // This is useful for testing so that tests about the effects of
                // `try_to_translate_virtual_to_real` don't have to worry about how the
                // compiler is bootstrapped.
                if let Some(virtual_dir) = &sess.opts.unstable_opts.simulate_remapped_rust_src_base
                {
                    if let Some(real_dir) = &sess.opts.real_rust_source_base_dir {
                        for subdir in ["library", "compiler"] {
                            if let rustc_span::FileName::Real(ref mut old_name) = name {
                                if let rustc_span::RealFileName::LocalPath(local) = old_name {
                                    if let Ok(rest) = local.strip_prefix(real_dir.join(subdir)) {
                                        *old_name = rustc_span::RealFileName::Remapped {
                                            local_path: None,
                                            virtual_name: virtual_dir.join(subdir).join(rest),
                                        };
                                    }
                                }
                            }
                        }
                    }
                }

                // If this file's path has been remapped to `/rustc/$hash`,
                // we might be able to reverse that (also see comments above,
                // on `try_to_translate_virtual_to_real`).
                try_to_translate_virtual_to_real(&mut name);

                let source_length = (end_pos - start_pos).to_usize();

                let local_version = sess.source_map().new_imported_source_file(
                    name,
                    src_hash,
                    name_hash,
                    source_length,
                    self.cnum,
                    lines,
                    multibyte_chars,
                    non_narrow_chars,
                    normalized_pos,
                    start_pos,
                    source_file_index,
                );
                debug!(
                    "CrateMetaData::imported_source_files alloc \
                         source_file {:?} original (start_pos {:?} end_pos {:?}) \
                         translated (start_pos {:?} end_pos {:?})",
                    local_version.name,
                    start_pos,
                    end_pos,
                    local_version.start_pos,
                    local_version.end_pos
                );

                ImportedSourceFile {
                    original_start_pos: start_pos,
                    original_end_pos: end_pos,
                    translated_source_file: local_version,
                }
            })
            .clone()
    }

    fn get_generator_diagnostic_data(
        self,
        tcx: TyCtxt<'tcx>,
        id: DefIndex,
    ) -> Option<GeneratorDiagnosticData<'tcx>> {
        self.root
            .tables
            .generator_diagnostic_data
            .get(self, id)
            .map(|param| param.decode((self, tcx)))
            .map(|generator_data| GeneratorDiagnosticData {
                generator_interior_types: generator_data.generator_interior_types,
                hir_owner: generator_data.hir_owner,
                nodes_types: generator_data.nodes_types,
                adjustments: generator_data.adjustments,
            })
    }

    fn get_attr_flags(self, index: DefIndex) -> AttrFlags {
        self.root.tables.attr_flags.get(self, index)
    }

    fn get_is_intrinsic(self, index: DefIndex) -> bool {
        self.root.tables.is_intrinsic.get(self, index)
    }
}

impl CrateMetadata {
    pub(crate) fn new(
        sess: &Session,
        cstore: &CStore,
        blob: MetadataBlob,
        root: CrateRoot,
        raw_proc_macros: Option<&'static [ProcMacro]>,
        cnum: CrateNum,
        cnum_map: CrateNumMap,
        dep_kind: CrateDepKind,
        source: CrateSource,
        private_dep: bool,
        host_hash: Option<Svh>,
    ) -> CrateMetadata {
        let trait_impls = root
            .impls
            .decode((&blob, sess))
            .map(|trait_impls| (trait_impls.trait_id, trait_impls.impls))
            .collect();
        let alloc_decoding_state =
            AllocDecodingState::new(root.interpret_alloc_index.decode(&blob).collect());
        let dependencies = Lock::new(cnum_map.iter().cloned().collect());

        // Pre-decode the DefPathHash->DefIndex table. This is a cheap operation
        // that does not copy any data. It just does some data verification.
        let def_path_hash_map = root.def_path_hash_map.decode(&blob);

        let mut cdata = CrateMetadata {
            blob,
            root,
            trait_impls,
            incoherent_impls: Default::default(),
            raw_proc_macros,
            source_map_import_info: Lock::new(Vec::new()),
            def_path_hash_map,
            expn_hash_map: Default::default(),
            alloc_decoding_state,
            cnum,
            cnum_map,
            dependencies,
            dep_kind: Lock::new(dep_kind),
            source: Lrc::new(source),
            private_dep,
            host_hash,
            extern_crate: Lock::new(None),
            hygiene_context: Default::default(),
            def_key_cache: Default::default(),
            def_path_hash_cache: Default::default(),
        };

        // Need `CrateMetadataRef` to decode `DefId`s in simplified types.
        cdata.incoherent_impls = cdata
            .root
            .incoherent_impls
            .decode(CrateMetadataRef { cdata: &cdata, cstore })
            .map(|incoherent_impls| (incoherent_impls.self_ty, incoherent_impls.impls))
            .collect();

        cdata
    }

    pub(crate) fn dependencies(&self) -> LockGuard<'_, Vec<CrateNum>> {
        self.dependencies.borrow()
    }

    pub(crate) fn add_dependency(&self, cnum: CrateNum) {
        self.dependencies.borrow_mut().push(cnum);
    }

    pub(crate) fn update_extern_crate(&self, new_extern_crate: ExternCrate) -> bool {
        let mut extern_crate = self.extern_crate.borrow_mut();
        let update = Some(new_extern_crate.rank()) > extern_crate.as_ref().map(ExternCrate::rank);
        if update {
            *extern_crate = Some(new_extern_crate);
        }
        update
    }

    pub(crate) fn source(&self) -> &CrateSource {
        &*self.source
    }

    pub(crate) fn dep_kind(&self) -> CrateDepKind {
        *self.dep_kind.lock()
    }

    pub(crate) fn update_dep_kind(&self, f: impl FnOnce(CrateDepKind) -> CrateDepKind) {
        self.dep_kind.with_lock(|dep_kind| *dep_kind = f(*dep_kind))
    }

    pub(crate) fn required_panic_strategy(&self) -> Option<PanicStrategy> {
        self.root.required_panic_strategy
    }

    pub(crate) fn needs_panic_runtime(&self) -> bool {
        self.root.needs_panic_runtime
    }

    pub(crate) fn is_panic_runtime(&self) -> bool {
        self.root.panic_runtime
    }

    pub(crate) fn is_profiler_runtime(&self) -> bool {
        self.root.profiler_runtime
    }

    pub(crate) fn needs_allocator(&self) -> bool {
        self.root.needs_allocator
    }

    pub(crate) fn has_global_allocator(&self) -> bool {
        self.root.has_global_allocator
    }

    pub(crate) fn has_alloc_error_handler(&self) -> bool {
        self.root.has_alloc_error_handler
    }

    pub(crate) fn has_default_lib_allocator(&self) -> bool {
        self.root.has_default_lib_allocator
    }

    pub(crate) fn is_proc_macro_crate(&self) -> bool {
        self.root.is_proc_macro_crate()
    }

    pub(crate) fn name(&self) -> Symbol {
        self.root.name
    }

    pub(crate) fn stable_crate_id(&self) -> StableCrateId {
        self.root.stable_crate_id
    }

    pub(crate) fn hash(&self) -> Svh {
        self.root.hash
    }

    fn num_def_ids(&self) -> usize {
        self.root.tables.def_keys.size()
    }

    fn local_def_id(&self, index: DefIndex) -> DefId {
        DefId { krate: self.cnum, index }
    }

    // Translate a DefId from the current compilation environment to a DefId
    // for an external crate.
    fn reverse_translate_def_id(&self, did: DefId) -> Option<DefId> {
        for (local, &global) in self.cnum_map.iter_enumerated() {
            if global == did.krate {
                return Some(DefId { krate: local, index: did.index });
            }
        }

        None
    }
}
