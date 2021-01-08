use crate::dep_graph::{DepNode, DepNodeIndex, SerializedDepNodeIndex};
use crate::mir::interpret::{AllocDecodingSession, AllocDecodingState};
use crate::mir::{self, interpret};
use crate::ty::codec::{OpaqueEncoder, RefDecodable, TyDecoder, TyEncoder};
use crate::ty::context::TyCtxt;
use crate::ty::{self, Ty};
use rustc_data_structures::fingerprint::{Fingerprint, FingerprintDecoder, FingerprintEncoder};
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_data_structures::sync::{HashMapExt, Lock, Lrc, OnceCell};
use rustc_data_structures::thin_vec::ThinVec;
use rustc_data_structures::unhash::UnhashMap;
use rustc_errors::Diagnostic;
use rustc_hir::def_id::{CrateNum, DefId, DefIndex, LocalDefId, LOCAL_CRATE};
use rustc_hir::definitions::DefPathHash;
use rustc_hir::definitions::Definitions;
use rustc_index::vec::{Idx, IndexVec};
use rustc_serialize::{opaque, Decodable, Decoder, Encodable, Encoder};
use rustc_session::{CrateDisambiguator, Session};
use rustc_span::hygiene::{
    ExpnDataDecodeMode, ExpnDataEncodeMode, ExpnId, HygieneDecodeContext, HygieneEncodeContext,
    SyntaxContext, SyntaxContextData,
};
use rustc_span::source_map::{SourceMap, StableSourceFileId};
use rustc_span::CachingSourceMapView;
use rustc_span::{BytePos, ExpnData, SourceFile, Span, DUMMY_SP};
use std::collections::hash_map::Entry;
use std::iter::FromIterator;
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
    // Maps `DefPathHash`es to their `RawDefId`s from the *previous*
    // compilation session. This is used as an initial 'guess' when
    // we try to map a `DefPathHash` to its `DefId` in the current compilation
    // session.
    foreign_def_path_hashes: UnhashMap<DefPathHash, RawDefId>,

    // The *next* compilation sessison's `foreign_def_path_hashes` - at
    // the end of our current compilation session, this will get written
    // out to the `foreign_def_path_hashes` field of the `Footer`, which
    // will become `foreign_def_path_hashes` of the next compilation session.
    // This stores any `DefPathHash` that we may need to map to a `DefId`
    // during the next compilation session.
    latest_foreign_def_path_hashes: Lock<UnhashMap<DefPathHash, RawDefId>>,

    // Maps `DefPathHashes` to their corresponding `LocalDefId`s for all
    // local items in the current compilation session. This is only populated
    // when we are in incremental mode and have loaded a pre-existing cache
    // from disk, since this map is only used when deserializing a `DefPathHash`
    // from the incremental cache.
    local_def_path_hash_to_def_id: UnhashMap<DefPathHash, LocalDefId>,
    // Caches all lookups of `DefPathHashes`, both for local and foreign
    // definitions. A definition from the previous compilation session
    // may no longer exist in the current compilation session, so
    // we use `Option<DefId>` so that we can cache a lookup failure.
    def_path_hash_to_def_id_cache: Lock<UnhashMap<DefPathHash, Option<DefId>>>,
}

// This type is used only for serialization and deserialization.
#[derive(Encodable, Decodable)]
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
    foreign_def_path_hashes: UnhashMap<DefPathHash, RawDefId>,
}

type EncodedQueryResultIndex = Vec<(SerializedDepNodeIndex, AbsoluteBytePos)>;
type EncodedDiagnosticsIndex = Vec<(SerializedDepNodeIndex, AbsoluteBytePos)>;
type EncodedDiagnostics = Vec<Diagnostic>;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Encodable, Decodable)]
struct SourceFileIndex(u32);

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Encodable, Decodable)]
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

/// Represents a potentially invalid `DefId`. This is used during incremental
/// compilation to represent a `DefId` from the *previous* compilation session,
/// which may no longer be valid. This is used to help map a `DefPathHash`
/// to a `DefId` in the current compilation session.
#[derive(Encodable, Decodable, Copy, Clone, Debug)]
crate struct RawDefId {
    // We deliberately do not use `CrateNum` and `DefIndex`
    // here, since a crate/index from the previous compilation
    // session may no longer exist.
    pub krate: u32,
    pub index: u32,
}

fn make_local_def_path_hash_map(definitions: &Definitions) -> UnhashMap<DefPathHash, LocalDefId> {
    UnhashMap::from_iter(
        definitions
            .def_path_table()
            .all_def_path_hashes_and_def_ids(LOCAL_CRATE)
            .map(|(hash, def_id)| (hash, def_id.as_local().unwrap())),
    )
}

impl<'sess> OnDiskCache<'sess> {
    /// Creates a new `OnDiskCache` instance from the serialized data in `data`.
    pub fn new(
        sess: &'sess Session,
        data: Vec<u8>,
        start_pos: usize,
        definitions: &Definitions,
    ) -> Self {
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
            foreign_def_path_hashes: footer.foreign_def_path_hashes,
            latest_foreign_def_path_hashes: Default::default(),
            local_def_path_hash_to_def_id: make_local_def_path_hash_map(definitions),
            def_path_hash_to_def_id_cache: Default::default(),
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
            foreign_def_path_hashes: Default::default(),
            latest_foreign_def_path_hashes: Default::default(),
            local_def_path_hash_to_def_id: Default::default(),
            def_path_hash_to_def_id_cache: Default::default(),
        }
    }

    pub fn serialize<'tcx, E>(&self, tcx: TyCtxt<'tcx>, encoder: &mut E) -> Result<(), E::Error>
    where
        E: OpaqueEncoder,
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

            // Register any dep nodes that we reused from the previous session,
            // but didn't `DepNode::construct` in this session. This ensures
            // that their `DefPathHash` to `RawDefId` mappings are registered
            // in 'latest_foreign_def_path_hashes' if necessary, since that
            // normally happens in `DepNode::construct`.
            tcx.dep_graph.register_reused_dep_nodes(tcx);

            // Load everything into memory so we can write it out to the on-disk
            // cache. The vast majority of cacheable query results should already
            // be in memory, so this should be a cheap operation.
            // Do this *before* we clone 'latest_foreign_def_path_hashes', since
            // loading existing queries may cause us to create new DepNodes, which
            // may in turn end up invoking `store_foreign_def_id_hash`
            tcx.dep_graph.exec_cache_promotions(tcx);

            let latest_foreign_def_path_hashes = self.latest_foreign_def_path_hashes.lock().clone();
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
                latest_foreign_def_path_hashes,
            };

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

            let foreign_def_path_hashes =
                std::mem::take(&mut encoder.latest_foreign_def_path_hashes);

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
                    foreign_def_path_hashes,
                },
            )?;

            // Encode the position of the footer as the last 8 bytes of the
            // file so we know where to look for it.
            IntEncodedWithFixedSize(footer_pos).encode(encoder.encoder.opaque())?;

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

    fn get_raw_def_id(&self, hash: &DefPathHash) -> Option<RawDefId> {
        self.foreign_def_path_hashes.get(hash).copied()
    }

    fn try_remap_cnum(&self, tcx: TyCtxt<'_>, cnum: u32) -> Option<CrateNum> {
        let cnum_map =
            self.cnum_map.get_or_init(|| Self::compute_cnum_map(tcx, &self.prev_cnums[..]));
        debug!("try_remap_cnum({}): cnum_map={:?}", cnum, cnum_map);

        cnum_map[CrateNum::from_u32(cnum)]
    }

    pub(crate) fn store_foreign_def_id_hash(&self, def_id: DefId, hash: DefPathHash) {
        // We may overwrite an existing entry, but it will have the same value,
        // so it's fine
        self.latest_foreign_def_path_hashes
            .lock()
            .insert(hash, RawDefId { krate: def_id.krate.as_u32(), index: def_id.index.as_u32() });
    }

    /// If the given `dep_node`'s hash still exists in the current compilation,
    /// and its current `DefId` is foreign, calls `store_foreign_def_id` with it.
    ///
    /// Normally, `store_foreign_def_id_hash` can be called directly by
    /// the dependency graph when we construct a `DepNode`. However,
    /// when we re-use a deserialized `DepNode` from the previous compilation
    /// session, we only have the `DefPathHash` available. This method is used
    /// to that any `DepNode` that we re-use has a `DefPathHash` -> `RawId` written
    /// out for usage in the next compilation session.
    pub fn register_reused_dep_node(&self, tcx: TyCtxt<'tcx>, dep_node: &DepNode) {
        // For reused dep nodes, we only need to store the mapping if the node
        // is one whose query key we can reconstruct from the hash. We use the
        // mapping to aid that reconstruction in the next session. While we also
        // use it to decode `DefId`s we encoded in the cache as `DefPathHashes`,
        // they're already registered during `DefId` encoding.
        if dep_node.kind.can_reconstruct_query_key() {
            let hash = DefPathHash(dep_node.hash.into());

            // We can't simply copy the `RawDefId` from `foreign_def_path_hashes` to
            // `latest_foreign_def_path_hashes`, since the `RawDefId` might have
            // changed in the current compilation session (e.g. we've added/removed crates,
            // or added/removed definitions before/after the target definition).
            if let Some(def_id) = self.def_path_hash_to_def_id(tcx, hash) {
                if !def_id.is_local() {
                    self.store_foreign_def_id_hash(def_id, hash);
                }
            }
        }
    }

    /// Returns the cached query result if there is something in the cache for
    /// the given `SerializedDepNodeIndex`; otherwise returns `None`.
    crate fn try_load_query_result<'tcx, T>(
        &self,
        tcx: TyCtxt<'tcx>,
        dep_node_index: SerializedDepNodeIndex,
    ) -> Option<T>
    where
        T: for<'a> Decodable<CacheDecoder<'a, 'tcx>>,
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
        T: for<'a> Decodable<CacheDecoder<'a, 'tcx>>,
    {
        let pos = index.get(&dep_node_index).cloned()?;

        self.with_decoder(tcx, pos, |decoder| match decode_tagged(decoder, dep_node_index) {
            Ok(v) => Some(v),
            Err(e) => bug!("could not decode cached {}: {}", debug_tag, e),
        })
    }

    fn with_decoder<'a, 'tcx, T, F: FnOnce(&mut CacheDecoder<'sess, 'tcx>) -> T>(
        &'sess self,
        tcx: TyCtxt<'tcx>,
        pos: AbsoluteBytePos,
        f: F,
    ) -> T
    where
        T: Decodable<CacheDecoder<'a, 'tcx>>,
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

    /// Converts a `DefPathHash` to its corresponding `DefId` in the current compilation
    /// session, if it still exists. This is used during incremental compilation to
    /// turn a deserialized `DefPathHash` into its current `DefId`.
    pub(crate) fn def_path_hash_to_def_id(
        &self,
        tcx: TyCtxt<'tcx>,
        hash: DefPathHash,
    ) -> Option<DefId> {
        let mut cache = self.def_path_hash_to_def_id_cache.lock();
        match cache.entry(hash) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                debug!("def_path_hash_to_def_id({:?})", hash);
                // Check if the `DefPathHash` corresponds to a definition in the current
                // crate
                if let Some(def_id) = self.local_def_path_hash_to_def_id.get(&hash).cloned() {
                    let def_id = def_id.to_def_id();
                    e.insert(Some(def_id));
                    return Some(def_id);
                }
                // This `raw_def_id` represents the `DefId` of this `DefPathHash` in
                // the *previous* compliation session. The `DefPathHash` includes the
                // owning crate, so if the corresponding definition still exists in the
                // current compilation session, the crate is guaranteed to be the same
                // (otherwise, we would compute a different `DefPathHash`).
                let raw_def_id = self.get_raw_def_id(&hash)?;
                debug!("def_path_hash_to_def_id({:?}): raw_def_id = {:?}", hash, raw_def_id);
                // If the owning crate no longer exists, the corresponding definition definitely
                // no longer exists.
                let krate = self.try_remap_cnum(tcx, raw_def_id.krate)?;
                debug!("def_path_hash_to_def_id({:?}): krate = {:?}", hash, krate);
                // If our `DefPathHash` corresponded to a definition in the local crate,
                // we should have either found it in `local_def_path_hash_to_def_id`, or
                // never attempted to load it in the first place. Any query result or `DepNode`
                // that references a local `DefId` should depend on some HIR-related `DepNode`.
                // If a local definition is removed/modified such that its old `DefPathHash`
                // no longer has a corresponding definition, that HIR-related `DepNode` should
                // end up red. This should prevent us from ever calling
                // `tcx.def_path_hash_to_def_id`, since we'll end up recomputing any
                // queries involved.
                debug_assert_ne!(krate, LOCAL_CRATE);
                // Try to find a definition in the current session, using the previous `DefIndex`
                // as an initial guess.
                let opt_def_id = tcx.cstore.def_path_hash_to_def_id(krate, raw_def_id.index, hash);
                debug!("def_path_to_def_id({:?}): opt_def_id = {:?}", hash, opt_def_id);
                e.insert(opt_def_id);
                opt_def_id
            }
        }
    }
}

//- DECODING -------------------------------------------------------------------

/// A decoder that can read from the incremental compilation cache. It is similar to the one
/// we use for crate metadata decoding in that it can rebase spans and eventually
/// will also handle things that contain `Ty` instances.
crate struct CacheDecoder<'a, 'tcx> {
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
    T: Decodable<D> + Eq + std::fmt::Debug,
    V: Decodable<D>,
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
    const CLEAR_CROSS_CRATE: bool = false;

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

    fn decode_alloc_id(&mut self) -> Result<interpret::AllocId, Self::Error> {
        let alloc_decoding_session = self.alloc_decoding_session;
        alloc_decoding_session.decode_alloc_id(self)
    }
}

crate::implement_ty_decoder!(CacheDecoder<'a, 'tcx>);

// This ensures that the `Decodable<opaque::Decoder>::decode` specialization for `Vec<u8>` is used
// when a `CacheDecoder` is passed to `Decodable::decode`. Unfortunately, we have to manually opt
// into specializations this way, given how `CacheDecoder` and the decoding traits currently work.
impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for Vec<u8> {
    fn decode(d: &mut CacheDecoder<'a, 'tcx>) -> Result<Self, String> {
        Decodable::decode(&mut d.opaque)
    }
}

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for SyntaxContext {
    fn decode(decoder: &mut CacheDecoder<'a, 'tcx>) -> Result<Self, String> {
        let syntax_contexts = decoder.syntax_contexts;
        rustc_span::hygiene::decode_syntax_context(decoder, decoder.hygiene_context, |this, id| {
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

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for ExpnId {
    fn decode(decoder: &mut CacheDecoder<'a, 'tcx>) -> Result<Self, String> {
        let expn_data = decoder.expn_data;
        rustc_span::hygiene::decode_expn_id(
            decoder,
            ExpnDataDecodeMode::incr_comp(decoder.hygiene_context),
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

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for Span {
    fn decode(decoder: &mut CacheDecoder<'a, 'tcx>) -> Result<Self, String> {
        let tag: u8 = Decodable::decode(decoder)?;

        if tag == TAG_INVALID_SPAN {
            return Ok(DUMMY_SP);
        } else {
            debug_assert_eq!(tag, TAG_VALID_SPAN);
        }

        let file_lo_index = SourceFileIndex::decode(decoder)?;
        let line_lo = usize::decode(decoder)?;
        let col_lo = BytePos::decode(decoder)?;
        let len = BytePos::decode(decoder)?;
        let ctxt = SyntaxContext::decode(decoder)?;

        let file_lo = decoder.file_index_to_file(file_lo_index);
        let lo = file_lo.lines[line_lo - 1] + col_lo;
        let hi = lo + len;

        Ok(Span::new(lo, hi, ctxt))
    }
}

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for CrateNum {
    fn decode(d: &mut CacheDecoder<'a, 'tcx>) -> Result<Self, String> {
        let cnum = CrateNum::from_u32(u32::decode(d)?);
        Ok(d.map_encoded_cnum_to_current(cnum))
    }
}

// This impl makes sure that we get a runtime error when we try decode a
// `DefIndex` that is not contained in a `DefId`. Such a case would be problematic
// because we would not know how to transform the `DefIndex` to the current
// context.
impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for DefIndex {
    fn decode(d: &mut CacheDecoder<'a, 'tcx>) -> Result<DefIndex, String> {
        Err(d.error("trying to decode `DefIndex` outside the context of a `DefId`"))
    }
}

// Both the `CrateNum` and the `DefIndex` of a `DefId` can change in between two
// compilation sessions. We use the `DefPathHash`, which is stable across
// sessions, to map the old `DefId` to the new one.
impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for DefId {
    fn decode(d: &mut CacheDecoder<'a, 'tcx>) -> Result<Self, String> {
        // Load the `DefPathHash` which is was we encoded the `DefId` as.
        let def_path_hash = DefPathHash::decode(d)?;

        // Using the `DefPathHash`, we can lookup the new `DefId`.
        // Subtle: We only encode a `DefId` as part of a query result.
        // If we get to this point, then all of the query inputs were green,
        // which means that the definition with this hash is guaranteed to
        // still exist in the current compilation session.
        Ok(d.tcx()
            .queries
            .on_disk_cache
            .as_ref()
            .unwrap()
            .def_path_hash_to_def_id(d.tcx(), def_path_hash)
            .unwrap())
    }
}

impl<'a, 'tcx> FingerprintDecoder for CacheDecoder<'a, 'tcx> {
    fn decode_fingerprint(&mut self) -> Result<Fingerprint, Self::Error> {
        Fingerprint::decode_opaque(&mut self.opaque)
    }
}

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for &'tcx FxHashSet<LocalDefId> {
    fn decode(d: &mut CacheDecoder<'a, 'tcx>) -> Result<Self, String> {
        RefDecodable::decode(d)
    }
}

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>>
    for &'tcx IndexVec<mir::Promoted, mir::Body<'tcx>>
{
    fn decode(d: &mut CacheDecoder<'a, 'tcx>) -> Result<Self, String> {
        RefDecodable::decode(d)
    }
}

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for &'tcx [mir::abstract_const::Node<'tcx>] {
    fn decode(d: &mut CacheDecoder<'a, 'tcx>) -> Result<Self, String> {
        RefDecodable::decode(d)
    }
}

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for &'tcx [(ty::Predicate<'tcx>, Span)] {
    fn decode(d: &mut CacheDecoder<'a, 'tcx>) -> Result<Self, String> {
        RefDecodable::decode(d)
    }
}

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for &'tcx [rustc_ast::InlineAsmTemplatePiece] {
    fn decode(d: &mut CacheDecoder<'a, 'tcx>) -> Result<Self, String> {
        RefDecodable::decode(d)
    }
}

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for &'tcx [Span] {
    fn decode(d: &mut CacheDecoder<'a, 'tcx>) -> Result<Self, String> {
        RefDecodable::decode(d)
    }
}

//- ENCODING -------------------------------------------------------------------

/// An encoder that can write to the incremental compilation cache.
struct CacheEncoder<'a, 'tcx, E: OpaqueEncoder> {
    tcx: TyCtxt<'tcx>,
    encoder: &'a mut E,
    type_shorthands: FxHashMap<Ty<'tcx>, usize>,
    predicate_shorthands: FxHashMap<ty::Predicate<'tcx>, usize>,
    interpret_allocs: FxIndexSet<interpret::AllocId>,
    source_map: CachingSourceMapView<'tcx>,
    file_to_file_index: FxHashMap<*const SourceFile, SourceFileIndex>,
    hygiene_context: &'a HygieneEncodeContext,
    latest_foreign_def_path_hashes: UnhashMap<DefPathHash, RawDefId>,
}

impl<'a, 'tcx, E> CacheEncoder<'a, 'tcx, E>
where
    E: 'a + OpaqueEncoder,
{
    fn source_file_index(&mut self, source_file: Lrc<SourceFile>) -> SourceFileIndex {
        self.file_to_file_index[&(&*source_file as *const SourceFile)]
    }

    /// Encode something with additional information that allows to do some
    /// sanity checks when decoding the data again. This method will first
    /// encode the specified tag, then the given value, then the number of
    /// bytes taken up by tag and value. On decoding, we can then verify that
    /// we get the expected tag and read the expected number of bytes.
    fn encode_tagged<T: Encodable<Self>, V: Encodable<Self>>(
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

impl<'a, 'tcx> FingerprintEncoder for CacheEncoder<'a, 'tcx, rustc_serialize::opaque::Encoder> {
    fn encode_fingerprint(&mut self, f: &Fingerprint) -> opaque::EncodeResult {
        f.encode_opaque(self.encoder)
    }
}

impl<'a, 'tcx, E> Encodable<CacheEncoder<'a, 'tcx, E>> for SyntaxContext
where
    E: 'a + OpaqueEncoder,
{
    fn encode(&self, s: &mut CacheEncoder<'a, 'tcx, E>) -> Result<(), E::Error> {
        rustc_span::hygiene::raw_encode_syntax_context(*self, s.hygiene_context, s)
    }
}

impl<'a, 'tcx, E> Encodable<CacheEncoder<'a, 'tcx, E>> for ExpnId
where
    E: 'a + OpaqueEncoder,
{
    fn encode(&self, s: &mut CacheEncoder<'a, 'tcx, E>) -> Result<(), E::Error> {
        rustc_span::hygiene::raw_encode_expn_id(
            *self,
            s.hygiene_context,
            ExpnDataEncodeMode::IncrComp,
            s,
        )
    }
}

impl<'a, 'tcx, E> Encodable<CacheEncoder<'a, 'tcx, E>> for Span
where
    E: 'a + OpaqueEncoder,
{
    fn encode(&self, s: &mut CacheEncoder<'a, 'tcx, E>) -> Result<(), E::Error> {
        if *self == DUMMY_SP {
            return TAG_INVALID_SPAN.encode(s);
        }

        let span_data = self.data();
        let (file_lo, line_lo, col_lo) = match s.source_map.byte_pos_to_line_and_col(span_data.lo) {
            Some(pos) => pos,
            None => return TAG_INVALID_SPAN.encode(s),
        };

        if !file_lo.contains(span_data.hi) {
            return TAG_INVALID_SPAN.encode(s);
        }

        let len = span_data.hi - span_data.lo;

        let source_file_index = s.source_file_index(file_lo);

        TAG_VALID_SPAN.encode(s)?;
        source_file_index.encode(s)?;
        line_lo.encode(s)?;
        col_lo.encode(s)?;
        len.encode(s)?;
        span_data.ctxt.encode(s)
    }
}

impl<'a, 'tcx, E> TyEncoder<'tcx> for CacheEncoder<'a, 'tcx, E>
where
    E: 'a + OpaqueEncoder,
{
    const CLEAR_CROSS_CRATE: bool = false;

    fn position(&self) -> usize {
        self.encoder.encoder_position()
    }
    fn type_shorthands(&mut self) -> &mut FxHashMap<Ty<'tcx>, usize> {
        &mut self.type_shorthands
    }
    fn predicate_shorthands(&mut self) -> &mut FxHashMap<ty::Predicate<'tcx>, usize> {
        &mut self.predicate_shorthands
    }
    fn encode_alloc_id(&mut self, alloc_id: &interpret::AllocId) -> Result<(), Self::Error> {
        let (index, _) = self.interpret_allocs.insert_full(*alloc_id);

        index.encode(self)
    }
}

impl<'a, 'tcx, E> Encodable<CacheEncoder<'a, 'tcx, E>> for DefId
where
    E: 'a + OpaqueEncoder,
{
    fn encode(&self, s: &mut CacheEncoder<'a, 'tcx, E>) -> Result<(), E::Error> {
        let def_path_hash = s.tcx.def_path_hash(*self);
        // Store additional information when we encode a foreign `DefId`,
        // so that we can map its `DefPathHash` back to a `DefId` in the next
        // compilation session.
        if !self.is_local() {
            s.latest_foreign_def_path_hashes.insert(
                def_path_hash,
                RawDefId { krate: self.krate.as_u32(), index: self.index.as_u32() },
            );
        }
        def_path_hash.encode(s)
    }
}

impl<'a, 'tcx, E> Encodable<CacheEncoder<'a, 'tcx, E>> for DefIndex
where
    E: 'a + OpaqueEncoder,
{
    fn encode(&self, _: &mut CacheEncoder<'a, 'tcx, E>) -> Result<(), E::Error> {
        bug!("encoding `DefIndex` without context");
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
    E: 'a + OpaqueEncoder,
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

// This ensures that the `Encodable<opaque::Encoder>::encode` specialization for byte slices
// is used when a `CacheEncoder` having an `opaque::Encoder` is passed to `Encodable::encode`.
// Unfortunately, we have to manually opt into specializations this way, given how `CacheEncoder`
// and the encoding traits currently work.
impl<'a, 'tcx> Encodable<CacheEncoder<'a, 'tcx, opaque::Encoder>> for [u8] {
    fn encode(&self, e: &mut CacheEncoder<'a, 'tcx, opaque::Encoder>) -> opaque::EncodeResult {
        self.encode(e.encoder)
    }
}

// An integer that will always encode to 8 bytes.
struct IntEncodedWithFixedSize(u64);

impl IntEncodedWithFixedSize {
    pub const ENCODED_SIZE: usize = 8;
}

impl Encodable<opaque::Encoder> for IntEncodedWithFixedSize {
    fn encode(&self, e: &mut opaque::Encoder) -> Result<(), !> {
        let start_pos = e.position();
        for i in 0..IntEncodedWithFixedSize::ENCODED_SIZE {
            ((self.0 >> (i * 8)) as u8).encode(e)?;
        }
        let end_pos = e.position();
        assert_eq!((end_pos - start_pos), IntEncodedWithFixedSize::ENCODED_SIZE);
        Ok(())
    }
}

impl<'a> Decodable<opaque::Decoder<'a>> for IntEncodedWithFixedSize {
    fn decode(decoder: &mut opaque::Decoder<'a>) -> Result<IntEncodedWithFixedSize, String> {
        let mut value: u64 = 0;
        let start_pos = decoder.position();

        for i in 0..IntEncodedWithFixedSize::ENCODED_SIZE {
            let byte: u8 = Decodable::decode(decoder)?;
            value |= (byte as u64) << (i * 8);
        }

        let end_pos = decoder.position();
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
    Q::Value: Encodable<CacheEncoder<'a, 'tcx, E>>,
    E: 'a + OpaqueEncoder,
{
    let _timer = tcx
        .sess
        .prof
        .extra_verbose_generic_activity("encode_query_results_for", std::any::type_name::<Q>());

    let state = Q::query_state(tcx);
    assert!(state.all_inactive());

    state.iter_results(|results| {
        for (key, value, dep_node) in results {
            if Q::cache_on_disk(tcx, &key, Some(value)) {
                let dep_node = SerializedDepNodeIndex::new(dep_node.index());

                // Record position of the cache entry.
                query_result_index
                    .push((dep_node, AbsoluteBytePos::new(encoder.encoder.opaque().position())));

                // Encode the type check tables with the `SerializedDepNodeIndex`
                // as tag.
                encoder.encode_tagged(dep_node, value)?;
            }
        }
        Ok(())
    })
}
