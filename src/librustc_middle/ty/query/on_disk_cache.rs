use crate::dep_graph::{DepNodeIndex, SerializedDepNodeIndex};
use crate::mir::interpret;
use crate::mir::interpret::{AllocDecodingSession, AllocDecodingState};
use crate::ty::codec::{self as ty_codec, TyDecoder, TyEncoder};
use crate::ty::context::TyCtxt;
use crate::ty::{self, Ty};
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::{FxHashMap, FxIndexSet};
use rustc_data_structures::sync::{HashMapExt, Lock, Lrc, OnceCell};
use rustc_data_structures::thin_vec::ThinVec;
use rustc_errors::Diagnostic;
use rustc_hir::def_id::{CrateNum, DefId, DefIndex, LocalDefId, LOCAL_CRATE};
use rustc_hir::definitions::DefPathHash;
use rustc_index::vec::{Idx, IndexVec};
use rustc_serialize::{
    opaque, Decodable, Decoder, Encodable, Encoder, SpecializedDecoder, SpecializedEncoder,
    UseSpecializedDecodable, UseSpecializedEncodable,
};
use rustc_session::{CrateDisambiguator, Session};
use rustc_span::hygiene::{
    ExpnDataDecodeMode, ExpnDataEncodeMode, ExpnId, HygieneDecodeContext, HygieneEncodeContext,
    SyntaxContext, SyntaxContextData,
};
use rustc_span::source_map::{SourceMap, StableSourceFileId};
use rustc_span::symbol::Ident;
use rustc_span::CachingSourceMapView;
use rustc_span::{BytePos, ExpnData, SourceFile, Span, DUMMY_SP};
use std::mem;

const TAG_FILE_FOOTER: u128 = 0xC0FFEE_C0FFEE_C0FFEE_C0FFEE_C0FFEE;

const TAG_VALID_SPAN: u8 = 0;
const TAG_INVALID_SPAN: u8 = 1;

const TAG_SYNTAX_CONTEXT: u8 = 0;
const TAG_EXPN_DATA: u8 = 1;

/// Provides an interface to incremental compilation data cached from the
/// previous compilation session. This data will eventually include the results
/// of a few selected queries (like `typeck` and `mir_optimized`) and
/// any diagnostics that have been emitted during a query.
pub struct OnDiskCache<'sess> {
    // The complete cache data in serialized form.
    serialized_data: Vec<u8>,

    // Collects all `Diagnostic`s emitted during the current compilation
    // session.
    current_diagnostics: Lock<FxHashMap<DepNodeIndex, Vec<Diagnostic>>>,

    prev_cnums: Vec<(u32, String, CrateDisambiguator)>,
    cnum_map: OnceCell<IndexVec<CrateNum, Option<CrateNum>>>,

    source_map: &'sess SourceMap,
    file_index_to_stable_id: FxHashMap<SourceFileIndex, StableSourceFileId>,

    // Caches that are populated lazily during decoding.
    file_index_to_file: Lock<FxHashMap<SourceFileIndex, Lrc<SourceFile>>>,

    // A map from dep-node to the position of the cached query result in
    // `serialized_data`.
    query_result_index: FxHashMap<SerializedDepNodeIndex, AbsoluteBytePos>,

    // A map from dep-node to the position of any associated diagnostics in
    // `serialized_data`.
    prev_diagnostics_index: FxHashMap<SerializedDepNodeIndex, AbsoluteBytePos>,

    alloc_decoding_state: AllocDecodingState,

    // A map from syntax context ids to the position of their associated
    // `SyntaxContextData`. We use a `u32` instead of a `SyntaxContext`
    // to represent the fact that we are storing *encoded* ids. When we decode
    // a `SyntaxContext`, a new id will be allocated from the global `HygieneData`,
    // which will almost certainly be different than the serialized id.
    syntax_contexts: FxHashMap<u32, AbsoluteBytePos>,
    // A map from the `DefPathHash` of an `ExpnId` to the position
    // of their associated `ExpnData`. Ideally, we would store a `DefId`,
    // but we need to decode this before we've constructed a `TyCtxt` (which
    // makes it difficult to decode a `DefId`).

    // Note that these `DefPathHashes` correspond to both local and foreign
    // `ExpnData` (e.g `ExpnData.krate` may not be `LOCAL_CRATE`). Alternatively,
    // we could look up the `ExpnData` from the metadata of foreign crates,
    // but it seemed easier to have `OnDiskCache` be independent of the `CStore`.
    expn_data: FxHashMap<u32, AbsoluteBytePos>,
    // Additional information used when decoding hygiene data.
    hygiene_context: HygieneDecodeContext,
}

// This type is used only for serialization and deserialization.
#[derive(RustcEncodable, RustcDecodable)]
struct Footer {
    file_index_to_stable_id: FxHashMap<SourceFileIndex, StableSourceFileId>,
    prev_cnums: Vec<(u32, String, CrateDisambiguator)>,
    query_result_index: EncodedQueryResultIndex,
    diagnostics_index: EncodedQueryResultIndex,
    // The location of all allocations.
    interpret_alloc_index: Vec<u32>,
    // See `OnDiskCache.syntax_contexts`
    syntax_contexts: FxHashMap<u32, AbsoluteBytePos>,
    // See `OnDiskCache.expn_data`
    expn_data: FxHashMap<u32, AbsoluteBytePos>,
}

type EncodedQueryResultIndex = Vec<(SerializedDepNodeIndex, AbsoluteBytePos)>;
type EncodedDiagnosticsIndex = Vec<(SerializedDepNodeIndex, AbsoluteBytePos)>;
type EncodedDiagnostics = Vec<Diagnostic>;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
struct SourceFileIndex(u32);

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, RustcEncodable, RustcDecodable)]
struct AbsoluteBytePos(u32);

impl AbsoluteBytePos {
    fn new(pos: usize) -> AbsoluteBytePos {
        debug_assert!(pos <= u32::MAX as usize);
        AbsoluteBytePos(pos as u32)
    }

    fn to_usize(self) -> usize {
        self.0 as usize
    }
}

impl<'sess> OnDiskCache<'sess> {
    /// Creates a new `OnDiskCache` instance from the serialized data in `data`.
    pub fn new(sess: &'sess Session, data: Vec<u8>, start_pos: usize) -> Self {
        debug_assert!(sess.opts.incremental.is_some());

        // Wrap in a scope so we can borrow `data`.
        let footer: Footer = {
            let mut decoder = opaque::Decoder::new(&data[..], start_pos);

            // Decode the *position* of the footer, which can be found in the
            // last 8 bytes of the file.
            decoder.set_position(data.len() - IntEncodedWithFixedSize::ENCODED_SIZE);
            let footer_pos = IntEncodedWithFixedSize::decode(&mut decoder)
                .expect("error while trying to decode footer position")
                .0 as usize;

            // Decode the file footer, which contains all the lookup tables, etc.
            decoder.set_position(footer_pos);

            decode_tagged(&mut decoder, TAG_FILE_FOOTER)
                .expect("error while trying to decode footer position")
        };

        Self {
            serialized_data: data,
            file_index_to_stable_id: footer.file_index_to_stable_id,
            file_index_to_file: Default::default(),
            prev_cnums: footer.prev_cnums,
            cnum_map: OnceCell::new(),
            source_map: sess.source_map(),
            current_diagnostics: Default::default(),
            query_result_index: footer.query_result_index.into_iter().collect(),
            prev_diagnostics_index: footer.diagnostics_index.into_iter().collect(),
            alloc_decoding_state: AllocDecodingState::new(footer.interpret_alloc_index),
            syntax_contexts: footer.syntax_contexts,
            expn_data: footer.expn_data,
            hygiene_context: Default::default(),
        }
    }

    pub fn new_empty(source_map: &'sess SourceMap) -> Self {
        Self {
            serialized_data: Vec::new(),
            file_index_to_stable_id: Default::default(),
            file_index_to_file: Default::default(),
            prev_cnums: vec![],
            cnum_map: OnceCell::new(),
            source_map,
            current_diagnostics: Default::default(),
            query_result_index: Default::default(),
            prev_diagnostics_index: Default::default(),
            alloc_decoding_state: AllocDecodingState::new(Vec::new()),
            syntax_contexts: FxHashMap::default(),
            expn_data: FxHashMap::default(),
            hygiene_context: Default::default(),
        }
    }

    pub fn serialize<'tcx, E>(&self, tcx: TyCtxt<'tcx>, encoder: &mut E) -> Result<(), E::Error>
    where
        E: TyEncoder,
    {
        // Serializing the `DepGraph` should not modify it.
        tcx.dep_graph.with_ignore(|| {
            // Allocate `SourceFileIndex`es.
            let (file_to_file_index, file_index_to_stable_id) = {
                let files = tcx.sess.source_map().files();
                let mut file_to_file_index =
                    FxHashMap::with_capacity_and_hasher(files.len(), Default::default());
                let mut file_index_to_stable_id =
                    FxHashMap::with_capacity_and_hasher(files.len(), Default::default());

                for (index, file) in files.iter().enumerate() {
                    let index = SourceFileIndex(index as u32);
                    let file_ptr: *const SourceFile = &**file as *const _;
                    file_to_file_index.insert(file_ptr, index);
                    file_index_to_stable_id.insert(index, StableSourceFileId::new(&file));
                }

                (file_to_file_index, file_index_to_stable_id)
            };

            let hygiene_encode_context = HygieneEncodeContext::default();

            let mut encoder = CacheEncoder {
                tcx,
                encoder,
                type_shorthands: Default::default(),
                predicate_shorthands: Default::default(),
                interpret_allocs: Default::default(),
                source_map: CachingSourceMapView::new(tcx.sess.source_map()),
                file_to_file_index,
                hygiene_context: &hygiene_encode_context,
            };

            // Load everything into memory so we can write it out to the on-disk
            // cache. The vast majority of cacheable query results should already
            // be in memory, so this should be a cheap operation.
            tcx.dep_graph.exec_cache_promotions(tcx);

            // Encode query results.
            let mut query_result_index = EncodedQueryResultIndex::new();

            tcx.sess.time("encode_query_results", || {
                let enc = &mut encoder;
                let qri = &mut query_result_index;

                macro_rules! encode_queries {
                    ($($query:ident,)*) => {
                        $(
                            encode_query_results::<ty::query::queries::$query<'_>, _>(
                                tcx,
                                enc,
                                qri
                            )?;
                        )*
                    }
                }

                rustc_cached_queries!(encode_queries!);

                Ok(())
            })?;

            // Encode diagnostics.
            let diagnostics_index: EncodedDiagnosticsIndex = self
                .current_diagnostics
                .borrow()
                .iter()
                .map(|(dep_node_index, diagnostics)| {
                    let pos = AbsoluteBytePos::new(encoder.position());
                    // Let's make sure we get the expected type here.
                    let diagnostics: &EncodedDiagnostics = diagnostics;
                    let dep_node_index = SerializedDepNodeIndex::new(dep_node_index.index());
                    encoder.encode_tagged(dep_node_index, diagnostics)?;

                    Ok((dep_node_index, pos))
                })
                .collect::<Result<_, _>>()?;

            let interpret_alloc_index = {
                let mut interpret_alloc_index = Vec::new();
                let mut n = 0;
                loop {
                    let new_n = encoder.interpret_allocs.len();
                    // If we have found new IDs, serialize those too.
                    if n == new_n {
                        // Otherwise, abort.
                        break;
                    }
                    interpret_alloc_index.reserve(new_n - n);
                    for idx in n..new_n {
                        let id = encoder.interpret_allocs[idx];
                        let pos = encoder.position() as u32;
                        interpret_alloc_index.push(pos);
                        interpret::specialized_encode_alloc_id(&mut encoder, tcx, id)?;
                    }
                    n = new_n;
                }
                interpret_alloc_index
            };

            let sorted_cnums = sorted_cnums_including_local_crate(tcx);
            let prev_cnums: Vec<_> = sorted_cnums
                .iter()
                .map(|&cnum| {
                    let crate_name = tcx.original_crate_name(cnum).to_string();
                    let crate_disambiguator = tcx.crate_disambiguator(cnum);
                    (cnum.as_u32(), crate_name, crate_disambiguator)
                })
                .collect();

            let mut syntax_contexts = FxHashMap::default();
            let mut expn_ids = FxHashMap::default();

            // Encode all hygiene data (`SyntaxContextData` and `ExpnData`) from the current
            // session.

            hygiene_encode_context.encode(
                &mut encoder,
                |encoder, index, ctxt_data| {
                    let pos = AbsoluteBytePos::new(encoder.position());
                    encoder.encode_tagged(TAG_SYNTAX_CONTEXT, ctxt_data)?;
                    syntax_contexts.insert(index, pos);
                    Ok(())
                },
                |encoder, index, expn_data| {
                    let pos = AbsoluteBytePos::new(encoder.position());
                    encoder.encode_tagged(TAG_EXPN_DATA, expn_data)?;
                    expn_ids.insert(index, pos);
                    Ok(())
                },
            )?;

            // `Encode the file footer.
            let footer_pos = encoder.position() as u64;
            encoder.encode_tagged(
                TAG_FILE_FOOTER,
                &Footer {
                    file_index_to_stable_id,
                    prev_cnums,
                    query_result_index,
                    diagnostics_index,
                    interpret_alloc_index,
                    syntax_contexts,
                    expn_data: expn_ids,
                },
            )?;

            // Encode the position of the footer as the last 8 bytes of the
            // file so we know where to look for it.
            IntEncodedWithFixedSize(footer_pos).encode(encoder.encoder)?;

            // DO NOT WRITE ANYTHING TO THE ENCODER AFTER THIS POINT! The address
            // of the footer must be the last thing in the data stream.

            return Ok(());

            fn sorted_cnums_including_local_crate(tcx: TyCtxt<'_>) -> Vec<CrateNum> {
                let mut cnums = vec![LOCAL_CRATE];
                cnums.extend_from_slice(&tcx.crates()[..]);
                cnums.sort_unstable();
                // Just to be sure...
                cnums.dedup();
                cnums
            }
        })
    }

    /// Loads a diagnostic emitted during the previous compilation session.
    pub fn load_diagnostics(
        &self,
        tcx: TyCtxt<'_>,
        dep_node_index: SerializedDepNodeIndex,
    ) -> Vec<Diagnostic> {
        let diagnostics: Option<EncodedDiagnostics> =
            self.load_indexed(tcx, dep_node_index, &self.prev_diagnostics_index, "diagnostics");

        diagnostics.unwrap_or_default()
    }

    /// Stores a diagnostic emitted during the current compilation session.
    /// Anything stored like this will be available via `load_diagnostics` in
    /// the next compilation session.
    #[inline(never)]
    #[cold]
    pub fn store_diagnostics(
        &self,
        dep_node_index: DepNodeIndex,
        diagnostics: ThinVec<Diagnostic>,
    ) {
        let mut current_diagnostics = self.current_diagnostics.borrow_mut();
        let prev = current_diagnostics.insert(dep_node_index, diagnostics.into());
        debug_assert!(prev.is_none());
    }

    /// Returns the cached query result if there is something in the cache for
    /// the given `SerializedDepNodeIndex`; otherwise returns `None`.
    pub fn try_load_query_result<T>(
        &self,
        tcx: TyCtxt<'_>,
        dep_node_index: SerializedDepNodeIndex,
    ) -> Option<T>
    where
        T: Decodable,
    {
        self.load_indexed(tcx, dep_node_index, &self.query_result_index, "query result")
    }

    /// Stores a diagnostic emitted during computation of an anonymous query.
    /// Since many anonymous queries can share the same `DepNode`, we aggregate
    /// them -- as opposed to regular queries where we assume that there is a
    /// 1:1 relationship between query-key and `DepNode`.
    #[inline(never)]
    #[cold]
    pub fn store_diagnostics_for_anon_node(
        &self,
        dep_node_index: DepNodeIndex,
        diagnostics: ThinVec<Diagnostic>,
    ) {
        let mut current_diagnostics = self.current_diagnostics.borrow_mut();

        let x = current_diagnostics.entry(dep_node_index).or_insert(Vec::new());

        x.extend(Into::<Vec<_>>::into(diagnostics));
    }

    fn load_indexed<'tcx, T>(
        &self,
        tcx: TyCtxt<'tcx>,
        dep_node_index: SerializedDepNodeIndex,
        index: &FxHashMap<SerializedDepNodeIndex, AbsoluteBytePos>,
        debug_tag: &'static str,
    ) -> Option<T>
    where
        T: Decodable,
    {
        let pos = index.get(&dep_node_index).cloned()?;

        self.with_decoder(tcx, pos, |decoder| match decode_tagged(decoder, dep_node_index) {
            Ok(v) => Some(v),
            Err(e) => bug!("could not decode cached {}: {}", debug_tag, e),
        })
    }

    fn with_decoder<'tcx, T, F: FnOnce(&mut CacheDecoder<'sess, 'tcx>) -> T>(
        &'sess self,
        tcx: TyCtxt<'tcx>,
        pos: AbsoluteBytePos,
        f: F,
    ) -> T
    where
        T: Decodable,
    {
        let cnum_map =
            self.cnum_map.get_or_init(|| Self::compute_cnum_map(tcx, &self.prev_cnums[..]));

        let mut decoder = CacheDecoder {
            tcx,
            opaque: opaque::Decoder::new(&self.serialized_data[..], pos.to_usize()),
            source_map: self.source_map,
            cnum_map,
            file_index_to_file: &self.file_index_to_file,
            file_index_to_stable_id: &self.file_index_to_stable_id,
            alloc_decoding_session: self.alloc_decoding_state.new_decoding_session(),
            syntax_contexts: &self.syntax_contexts,
            expn_data: &self.expn_data,
            hygiene_context: &self.hygiene_context,
        };
        f(&mut decoder)
    }

    // This function builds mapping from previous-session-`CrateNum` to
    // current-session-`CrateNum`. There might be `CrateNum`s from the previous
    // `Session` that don't occur in the current one. For these, the mapping
    // maps to None.
    fn compute_cnum_map(
        tcx: TyCtxt<'_>,
        prev_cnums: &[(u32, String, CrateDisambiguator)],
    ) -> IndexVec<CrateNum, Option<CrateNum>> {
        tcx.dep_graph.with_ignore(|| {
            let current_cnums = tcx
                .all_crate_nums(LOCAL_CRATE)
                .iter()
                .map(|&cnum| {
                    let crate_name = tcx.original_crate_name(cnum).to_string();
                    let crate_disambiguator = tcx.crate_disambiguator(cnum);
                    ((crate_name, crate_disambiguator), cnum)
                })
                .collect::<FxHashMap<_, _>>();

            let map_size = prev_cnums.iter().map(|&(cnum, ..)| cnum).max().unwrap_or(0) + 1;
            let mut map = IndexVec::from_elem_n(None, map_size as usize);

            for &(prev_cnum, ref crate_name, crate_disambiguator) in prev_cnums {
                let key = (crate_name.clone(), crate_disambiguator);
                map[CrateNum::from_u32(prev_cnum)] = current_cnums.get(&key).cloned();
            }

            map[LOCAL_CRATE] = Some(LOCAL_CRATE);
            map
        })
    }
}

//- DECODING -------------------------------------------------------------------

/// A decoder that can read from the incr. comp. cache. It is similar to the one
/// we use for crate metadata decoding in that it can rebase spans and eventually
/// will also handle things that contain `Ty` instances.
struct CacheDecoder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    opaque: opaque::Decoder<'a>,
    source_map: &'a SourceMap,
    cnum_map: &'a IndexVec<CrateNum, Option<CrateNum>>,
    file_index_to_file: &'a Lock<FxHashMap<SourceFileIndex, Lrc<SourceFile>>>,
    file_index_to_stable_id: &'a FxHashMap<SourceFileIndex, StableSourceFileId>,
    alloc_decoding_session: AllocDecodingSession<'a>,
    syntax_contexts: &'a FxHashMap<u32, AbsoluteBytePos>,
    expn_data: &'a FxHashMap<u32, AbsoluteBytePos>,
    hygiene_context: &'a HygieneDecodeContext,
}

impl<'a, 'tcx> CacheDecoder<'a, 'tcx> {
    fn file_index_to_file(&self, index: SourceFileIndex) -> Lrc<SourceFile> {
        let CacheDecoder {
            ref file_index_to_file,
            ref file_index_to_stable_id,
            ref source_map,
            ..
        } = *self;

        file_index_to_file
            .borrow_mut()
            .entry(index)
            .or_insert_with(|| {
                let stable_id = file_index_to_stable_id[&index];
                source_map
                    .source_file_by_stable_id(stable_id)
                    .expect("failed to lookup `SourceFile` in new context")
            })
            .clone()
    }
}

trait DecoderWithPosition: Decoder {
    fn position(&self) -> usize;
}

impl<'a> DecoderWithPosition for opaque::Decoder<'a> {
    fn position(&self) -> usize {
        self.position()
    }
}

impl<'a, 'tcx> DecoderWithPosition for CacheDecoder<'a, 'tcx> {
    fn position(&self) -> usize {
        self.opaque.position()
    }
}

// Decodes something that was encoded with `encode_tagged()` and verify that the
// tag matches and the correct amount of bytes was read.
fn decode_tagged<D, T, V>(decoder: &mut D, expected_tag: T) -> Result<V, D::Error>
where
    T: Decodable + Eq + ::std::fmt::Debug,
    V: Decodable,
    D: DecoderWithPosition,
{
    let start_pos = decoder.position();

    let actual_tag = T::decode(decoder)?;
    assert_eq!(actual_tag, expected_tag);
    let value = V::decode(decoder)?;
    let end_pos = decoder.position();

    let expected_len: u64 = Decodable::decode(decoder)?;
    assert_eq!((end_pos - start_pos) as u64, expected_len);

    Ok(value)
}

impl<'a, 'tcx> TyDecoder<'tcx> for CacheDecoder<'a, 'tcx> {
    #[inline]
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    #[inline]
    fn position(&self) -> usize {
        self.opaque.position()
    }

    #[inline]
    fn peek_byte(&self) -> u8 {
        self.opaque.data[self.opaque.position()]
    }

    fn cached_ty_for_shorthand<F>(
        &mut self,
        shorthand: usize,
        or_insert_with: F,
    ) -> Result<Ty<'tcx>, Self::Error>
    where
        F: FnOnce(&mut Self) -> Result<Ty<'tcx>, Self::Error>,
    {
        let tcx = self.tcx();

        let cache_key =
            ty::CReaderCacheKey { cnum: CrateNum::ReservedForIncrCompCache, pos: shorthand };

        if let Some(&ty) = tcx.ty_rcache.borrow().get(&cache_key) {
            return Ok(ty);
        }

        let ty = or_insert_with(self)?;
        // This may overwrite the entry, but it should overwrite with the same value.
        tcx.ty_rcache.borrow_mut().insert_same(cache_key, ty);
        Ok(ty)
    }

    fn cached_predicate_for_shorthand<F>(
        &mut self,
        shorthand: usize,
        or_insert_with: F,
    ) -> Result<ty::Predicate<'tcx>, Self::Error>
    where
        F: FnOnce(&mut Self) -> Result<ty::Predicate<'tcx>, Self::Error>,
    {
        let tcx = self.tcx();

        let cache_key =
            ty::CReaderCacheKey { cnum: CrateNum::ReservedForIncrCompCache, pos: shorthand };

        if let Some(&pred) = tcx.pred_rcache.borrow().get(&cache_key) {
            return Ok(pred);
        }

        let pred = or_insert_with(self)?;
        // This may overwrite the entry, but it should overwrite with the same value.
        tcx.pred_rcache.borrow_mut().insert_same(cache_key, pred);
        Ok(pred)
    }

    fn with_position<F, R>(&mut self, pos: usize, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        debug_assert!(pos < self.opaque.data.len());

        let new_opaque = opaque::Decoder::new(self.opaque.data, pos);
        let old_opaque = mem::replace(&mut self.opaque, new_opaque);
        let r = f(self);
        self.opaque = old_opaque;
        r
    }

    fn map_encoded_cnum_to_current(&self, cnum: CrateNum) -> CrateNum {
        self.cnum_map[cnum].unwrap_or_else(|| bug!("could not find new `CrateNum` for {:?}", cnum))
    }
}

implement_ty_decoder!(CacheDecoder<'a, 'tcx>);

impl<'a, 'tcx> SpecializedDecoder<SyntaxContext> for CacheDecoder<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<SyntaxContext, Self::Error> {
        let syntax_contexts = self.syntax_contexts;
        rustc_span::hygiene::decode_syntax_context(self, self.hygiene_context, |this, id| {
            // This closure is invoked if we haven't already decoded the data for the `SyntaxContext` we are deserializing.
            // We look up the position of the associated `SyntaxData` and decode it.
            let pos = syntax_contexts.get(&id).unwrap();
            this.with_position(pos.to_usize(), |decoder| {
                let data: SyntaxContextData = decode_tagged(decoder, TAG_SYNTAX_CONTEXT)?;
                Ok(data)
            })
        })
    }
}

impl<'a, 'tcx> SpecializedDecoder<ExpnId> for CacheDecoder<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<ExpnId, Self::Error> {
        let expn_data = self.expn_data;
        rustc_span::hygiene::decode_expn_id(
            self,
            ExpnDataDecodeMode::incr_comp(self.hygiene_context),
            |this, index| {
                // This closure is invoked if we haven't already decoded the data for the `ExpnId` we are deserializing.
                // We look up the position of the associated `ExpnData` and decode it.
                let pos = expn_data
                    .get(&index)
                    .unwrap_or_else(|| panic!("Bad index {:?} (map {:?})", index, expn_data));

                this.with_position(pos.to_usize(), |decoder| {
                    let data: ExpnData = decode_tagged(decoder, TAG_EXPN_DATA)?;
                    Ok(data)
                })
            },
        )
    }
}

impl<'a, 'tcx> SpecializedDecoder<interpret::AllocId> for CacheDecoder<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<interpret::AllocId, Self::Error> {
        let alloc_decoding_session = self.alloc_decoding_session;
        alloc_decoding_session.decode_alloc_id(self)
    }
}

impl<'a, 'tcx> SpecializedDecoder<Span> for CacheDecoder<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<Span, Self::Error> {
        let tag: u8 = Decodable::decode(self)?;

        if tag == TAG_INVALID_SPAN {
            return Ok(DUMMY_SP);
        } else {
            debug_assert_eq!(tag, TAG_VALID_SPAN);
        }

        let file_lo_index = SourceFileIndex::decode(self)?;
        let line_lo = usize::decode(self)?;
        let col_lo = BytePos::decode(self)?;
        let len = BytePos::decode(self)?;
        let ctxt = SyntaxContext::decode(self)?;

        let file_lo = self.file_index_to_file(file_lo_index);
        let lo = file_lo.lines[line_lo - 1] + col_lo;
        let hi = lo + len;

        Ok(Span::new(lo, hi, ctxt))
    }
}

impl<'a, 'tcx> SpecializedDecoder<Ident> for CacheDecoder<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<Ident, Self::Error> {
        // FIXME: Handle hygiene in incremental
        bug!("Trying to decode Ident for incremental");
    }
}

// This impl makes sure that we get a runtime error when we try decode a
// `DefIndex` that is not contained in a `DefId`. Such a case would be problematic
// because we would not know how to transform the `DefIndex` to the current
// context.
impl<'a, 'tcx> SpecializedDecoder<DefIndex> for CacheDecoder<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<DefIndex, Self::Error> {
        bug!("trying to decode `DefIndex` outside the context of a `DefId`")
    }
}

// Both the `CrateNum` and the `DefIndex` of a `DefId` can change in between two
// compilation sessions. We use the `DefPathHash`, which is stable across
// sessions, to map the old `DefId` to the new one.
impl<'a, 'tcx> SpecializedDecoder<DefId> for CacheDecoder<'a, 'tcx> {
    #[inline]
    fn specialized_decode(&mut self) -> Result<DefId, Self::Error> {
        // Load the `DefPathHash` which is was we encoded the `DefId` as.
        let def_path_hash = DefPathHash::decode(self)?;

        // Using the `DefPathHash`, we can lookup the new `DefId`.
        Ok(self.tcx().def_path_hash_to_def_id.as_ref().unwrap()[&def_path_hash])
    }
}

impl<'a, 'tcx> SpecializedDecoder<LocalDefId> for CacheDecoder<'a, 'tcx> {
    #[inline]
    fn specialized_decode(&mut self) -> Result<LocalDefId, Self::Error> {
        Ok(DefId::decode(self)?.expect_local())
    }
}

impl<'a, 'tcx> SpecializedDecoder<Fingerprint> for CacheDecoder<'a, 'tcx> {
    fn specialized_decode(&mut self) -> Result<Fingerprint, Self::Error> {
        Fingerprint::decode_opaque(&mut self.opaque)
    }
}

//- ENCODING -------------------------------------------------------------------

/// An encoder that can write the incr. comp. cache.
struct CacheEncoder<'a, 'tcx, E: ty_codec::TyEncoder> {
    tcx: TyCtxt<'tcx>,
    encoder: &'a mut E,
    type_shorthands: FxHashMap<Ty<'tcx>, usize>,
    predicate_shorthands: FxHashMap<ty::Predicate<'tcx>, usize>,
    interpret_allocs: FxIndexSet<interpret::AllocId>,
    source_map: CachingSourceMapView<'tcx>,
    file_to_file_index: FxHashMap<*const SourceFile, SourceFileIndex>,
    hygiene_context: &'a HygieneEncodeContext,
}

impl<'a, 'tcx, E> CacheEncoder<'a, 'tcx, E>
where
    E: 'a + TyEncoder,
{
    fn source_file_index(&mut self, source_file: Lrc<SourceFile>) -> SourceFileIndex {
        self.file_to_file_index[&(&*source_file as *const SourceFile)]
    }

    /// Encode something with additional information that allows to do some
    /// sanity checks when decoding the data again. This method will first
    /// encode the specified tag, then the given value, then the number of
    /// bytes taken up by tag and value. On decoding, we can then verify that
    /// we get the expected tag and read the expected number of bytes.
    fn encode_tagged<T: Encodable, V: Encodable>(
        &mut self,
        tag: T,
        value: &V,
    ) -> Result<(), E::Error> {
        let start_pos = self.position();

        tag.encode(self)?;
        value.encode(self)?;

        let end_pos = self.position();
        ((end_pos - start_pos) as u64).encode(self)
    }
}

impl<'a, 'tcx, E> SpecializedEncoder<interpret::AllocId> for CacheEncoder<'a, 'tcx, E>
where
    E: 'a + TyEncoder,
{
    fn specialized_encode(&mut self, alloc_id: &interpret::AllocId) -> Result<(), Self::Error> {
        let (index, _) = self.interpret_allocs.insert_full(*alloc_id);
        index.encode(self)
    }
}

impl<'a, 'tcx, E> SpecializedEncoder<SyntaxContext> for CacheEncoder<'a, 'tcx, E>
where
    E: 'a + TyEncoder,
{
    fn specialized_encode(&mut self, ctxt: &SyntaxContext) -> Result<(), Self::Error> {
        rustc_span::hygiene::raw_encode_syntax_context(*ctxt, self.hygiene_context, self)
    }
}

impl<'a, 'tcx, E> SpecializedEncoder<ExpnId> for CacheEncoder<'a, 'tcx, E>
where
    E: 'a + TyEncoder,
{
    fn specialized_encode(&mut self, expn: &ExpnId) -> Result<(), Self::Error> {
        rustc_span::hygiene::raw_encode_expn_id(
            *expn,
            self.hygiene_context,
            ExpnDataEncodeMode::IncrComp,
            self,
        )
    }
}

impl<'a, 'tcx, E> SpecializedEncoder<Span> for CacheEncoder<'a, 'tcx, E>
where
    E: 'a + TyEncoder,
{
    fn specialized_encode(&mut self, span: &Span) -> Result<(), Self::Error> {
        if *span == DUMMY_SP {
            return TAG_INVALID_SPAN.encode(self);
        }

        let span_data = span.data();
        let (file_lo, line_lo, col_lo) =
            match self.source_map.byte_pos_to_line_and_col(span_data.lo) {
                Some(pos) => pos,
                None => return TAG_INVALID_SPAN.encode(self),
            };

        if !file_lo.contains(span_data.hi) {
            return TAG_INVALID_SPAN.encode(self);
        }

        let len = span_data.hi - span_data.lo;

        let source_file_index = self.source_file_index(file_lo);

        TAG_VALID_SPAN.encode(self)?;
        source_file_index.encode(self)?;
        line_lo.encode(self)?;
        col_lo.encode(self)?;
        len.encode(self)?;
        span_data.ctxt.encode(self)?;
        Ok(())
    }
}

impl<'a, 'tcx, E> SpecializedEncoder<Ident> for CacheEncoder<'a, 'tcx, E>
where
    E: 'a + ty_codec::TyEncoder,
{
    fn specialized_encode(&mut self, _: &Ident) -> Result<(), Self::Error> {
        // We don't currently encode enough information to ensure hygiene works
        // with incremental, so panic rather than risk incremental bugs.

        // FIXME: handle hygiene in incremental.
        bug!("trying to encode `Ident` for incremental");
    }
}

impl<'a, 'tcx, E> ty_codec::TyEncoder for CacheEncoder<'a, 'tcx, E>
where
    E: 'a + TyEncoder,
{
    #[inline]
    fn position(&self) -> usize {
        self.encoder.position()
    }
}

impl<'a, 'tcx, E> SpecializedEncoder<CrateNum> for CacheEncoder<'a, 'tcx, E>
where
    E: 'a + TyEncoder,
{
    #[inline]
    fn specialized_encode(&mut self, cnum: &CrateNum) -> Result<(), Self::Error> {
        self.emit_u32(cnum.as_u32())
    }
}

impl<'a, 'b, 'c, 'tcx, E> SpecializedEncoder<&'b ty::TyS<'c>> for CacheEncoder<'a, 'tcx, E>
where
    E: 'a + TyEncoder,
    &'b ty::TyS<'c>: UseSpecializedEncodable,
{
    #[inline]
    fn specialized_encode(&mut self, ty: &&'b ty::TyS<'c>) -> Result<(), Self::Error> {
        debug_assert!(self.tcx.lift(ty).is_some());
        let ty = unsafe { std::mem::transmute::<&&'b ty::TyS<'c>, &&'tcx ty::TyS<'tcx>>(ty) };
        ty_codec::encode_with_shorthand(self, ty, |encoder| &mut encoder.type_shorthands)
    }
}

impl<'a, 'b, 'tcx, E> SpecializedEncoder<ty::Predicate<'b>> for CacheEncoder<'a, 'tcx, E>
where
    E: 'a + TyEncoder,
{
    #[inline]
    fn specialized_encode(&mut self, predicate: &ty::Predicate<'b>) -> Result<(), Self::Error> {
        debug_assert!(self.tcx.lift(predicate).is_some());
        let predicate =
            unsafe { std::mem::transmute::<&ty::Predicate<'b>, &ty::Predicate<'tcx>>(predicate) };
        ty_codec::encode_with_shorthand(self, predicate, |encoder| {
            &mut encoder.predicate_shorthands
        })
    }
}

impl<'a, 'tcx, E> SpecializedEncoder<DefId> for CacheEncoder<'a, 'tcx, E>
where
    E: 'a + TyEncoder,
{
    #[inline]
    fn specialized_encode(&mut self, id: &DefId) -> Result<(), Self::Error> {
        let def_path_hash = self.tcx.def_path_hash(*id);
        def_path_hash.encode(self)
    }
}

impl<'a, 'tcx, E> SpecializedEncoder<LocalDefId> for CacheEncoder<'a, 'tcx, E>
where
    E: 'a + TyEncoder,
{
    #[inline]
    fn specialized_encode(&mut self, id: &LocalDefId) -> Result<(), Self::Error> {
        id.to_def_id().encode(self)
    }
}

impl<'a, 'tcx, E> SpecializedEncoder<DefIndex> for CacheEncoder<'a, 'tcx, E>
where
    E: 'a + TyEncoder,
{
    fn specialized_encode(&mut self, _: &DefIndex) -> Result<(), Self::Error> {
        bug!("encoding `DefIndex` without context");
    }
}

impl<'a, 'tcx> SpecializedEncoder<Fingerprint> for CacheEncoder<'a, 'tcx, opaque::Encoder> {
    fn specialized_encode(&mut self, f: &Fingerprint) -> Result<(), Self::Error> {
        f.encode_opaque(&mut self.encoder)
    }
}

macro_rules! encoder_methods {
    ($($name:ident($ty:ty);)*) => {
        #[inline]
        $(fn $name(&mut self, value: $ty) -> Result<(), Self::Error> {
            self.encoder.$name(value)
        })*
    }
}

impl<'a, 'tcx, E> Encoder for CacheEncoder<'a, 'tcx, E>
where
    E: 'a + TyEncoder,
{
    type Error = E::Error;

    #[inline]
    fn emit_unit(&mut self) -> Result<(), Self::Error> {
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

// An integer that will always encode to 8 bytes.
struct IntEncodedWithFixedSize(u64);

impl IntEncodedWithFixedSize {
    pub const ENCODED_SIZE: usize = 8;
}

impl UseSpecializedEncodable for IntEncodedWithFixedSize {}
impl UseSpecializedDecodable for IntEncodedWithFixedSize {}

impl SpecializedEncoder<IntEncodedWithFixedSize> for opaque::Encoder {
    fn specialized_encode(&mut self, x: &IntEncodedWithFixedSize) -> Result<(), Self::Error> {
        let start_pos = self.position();
        for i in 0..IntEncodedWithFixedSize::ENCODED_SIZE {
            ((x.0 >> (i * 8)) as u8).encode(self)?;
        }
        let end_pos = self.position();
        assert_eq!((end_pos - start_pos), IntEncodedWithFixedSize::ENCODED_SIZE);
        Ok(())
    }
}

impl<'a> SpecializedDecoder<IntEncodedWithFixedSize> for opaque::Decoder<'a> {
    fn specialized_decode(&mut self) -> Result<IntEncodedWithFixedSize, Self::Error> {
        let mut value: u64 = 0;
        let start_pos = self.position();

        for i in 0..IntEncodedWithFixedSize::ENCODED_SIZE {
            let byte: u8 = Decodable::decode(self)?;
            value |= (byte as u64) << (i * 8);
        }

        let end_pos = self.position();
        assert_eq!((end_pos - start_pos), IntEncodedWithFixedSize::ENCODED_SIZE);

        Ok(IntEncodedWithFixedSize(value))
    }
}

fn encode_query_results<'a, 'tcx, Q, E>(
    tcx: TyCtxt<'tcx>,
    encoder: &mut CacheEncoder<'a, 'tcx, E>,
    query_result_index: &mut EncodedQueryResultIndex,
) -> Result<(), E::Error>
where
    Q: super::QueryDescription<TyCtxt<'tcx>> + super::QueryAccessors<TyCtxt<'tcx>>,
    Q::Value: Encodable,
    E: 'a + TyEncoder,
{
    let _timer = tcx
        .sess
        .prof
        .extra_verbose_generic_activity("encode_query_results_for", ::std::any::type_name::<Q>());

    let state = Q::query_state(tcx);
    assert!(state.all_inactive());

    state.iter_results(|results| {
        for (key, value, dep_node) in results {
            if Q::cache_on_disk(tcx, &key, Some(&value)) {
                let dep_node = SerializedDepNodeIndex::new(dep_node.index());

                // Record position of the cache entry.
                query_result_index.push((dep_node, AbsoluteBytePos::new(encoder.position())));

                // Encode the type check tables with the `SerializedDepNodeIndex`
                // as tag.
                encoder.encode_tagged(dep_node, &value)?;
            }
        }
        Ok(())
    })
}
