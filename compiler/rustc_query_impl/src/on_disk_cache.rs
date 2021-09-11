use crate::QueryCtxt;
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_data_structures::memmap::Mmap;
use rustc_data_structures::sync::{HashMapExt, Lock, Lrc, OnceCell, RwLock};
use rustc_data_structures::unhash::UnhashMap;
use rustc_hir::def_id::{CrateNum, DefId, DefIndex, LocalDefId, StableCrateId, LOCAL_CRATE};
use rustc_hir::definitions::DefPathHash;
use rustc_index::vec::{Idx, IndexVec};
use rustc_middle::dep_graph::{DepNode, DepNodeIndex, SerializedDepNodeIndex};
use rustc_middle::mir::interpret::{AllocDecodingSession, AllocDecodingState};
use rustc_middle::mir::{self, interpret};
use rustc_middle::ty::codec::{RefDecodable, TyDecoder, TyEncoder};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_query_system::dep_graph::DepContext;
use rustc_query_system::query::{QueryContext, QuerySideEffects};
use rustc_serialize::{
    opaque::{self, FileEncodeResult, FileEncoder, IntEncodedWithFixedSize},
    Decodable, Decoder, Encodable, Encoder,
};
use rustc_session::Session;
use rustc_span::hygiene::{
    ExpnId, HygieneDecodeContext, HygieneEncodeContext, SyntaxContext, SyntaxContextData,
};
use rustc_span::source_map::{SourceMap, StableSourceFileId};
use rustc_span::CachingSourceMapView;
use rustc_span::{BytePos, ExpnData, ExpnHash, Pos, SourceFile, Span};
use std::collections::hash_map::Entry;
use std::mem;

const TAG_FILE_FOOTER: u128 = 0xC0FFEE_C0FFEE_C0FFEE_C0FFEE_C0FFEE;

// A normal span encoded with both location information and a `SyntaxContext`
const TAG_FULL_SPAN: u8 = 0;
// A partial span with no location information, encoded only with a `SyntaxContext`
const TAG_PARTIAL_SPAN: u8 = 1;
const TAG_RELATIVE_SPAN: u8 = 2;

const TAG_SYNTAX_CONTEXT: u8 = 0;
const TAG_EXPN_DATA: u8 = 1;

/// Provides an interface to incremental compilation data cached from the
/// previous compilation session. This data will eventually include the results
/// of a few selected queries (like `typeck` and `mir_optimized`) and
/// any side effects that have been emitted during a query.
pub struct OnDiskCache<'sess> {
    // The complete cache data in serialized form.
    serialized_data: RwLock<Option<Mmap>>,

    // Collects all `QuerySideEffects` created during the current compilation
    // session.
    current_side_effects: Lock<FxHashMap<DepNodeIndex, QuerySideEffects>>,

    cnum_map: OnceCell<UnhashMap<StableCrateId, CrateNum>>,

    source_map: &'sess SourceMap,
    file_index_to_stable_id: FxHashMap<SourceFileIndex, EncodedSourceFileId>,

    // Caches that are populated lazily during decoding.
    file_index_to_file: Lock<FxHashMap<SourceFileIndex, Lrc<SourceFile>>>,

    // A map from dep-node to the position of the cached query result in
    // `serialized_data`.
    query_result_index: FxHashMap<SerializedDepNodeIndex, AbsoluteBytePos>,

    // A map from dep-node to the position of any associated `QuerySideEffects` in
    // `serialized_data`.
    prev_side_effects_index: FxHashMap<SerializedDepNodeIndex, AbsoluteBytePos>,

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
    expn_data: UnhashMap<ExpnHash, AbsoluteBytePos>,
    // Additional information used when decoding hygiene data.
    hygiene_context: HygieneDecodeContext,
    // Maps `DefPathHash`es to their `RawDefId`s from the *previous*
    // compilation session. This is used as an initial 'guess' when
    // we try to map a `DefPathHash` to its `DefId` in the current compilation
    // session.
    foreign_def_path_hashes: UnhashMap<DefPathHash, RawDefId>,
    // Likewise for ExpnId.
    foreign_expn_data: UnhashMap<ExpnHash, u32>,

    // The *next* compilation sessison's `foreign_def_path_hashes` - at
    // the end of our current compilation session, this will get written
    // out to the `foreign_def_path_hashes` field of the `Footer`, which
    // will become `foreign_def_path_hashes` of the next compilation session.
    // This stores any `DefPathHash` that we may need to map to a `DefId`
    // during the next compilation session.
    latest_foreign_def_path_hashes: Lock<UnhashMap<DefPathHash, RawDefId>>,

    // Caches all lookups of `DefPathHashes`, both for local and foreign
    // definitions. A definition from the previous compilation session
    // may no longer exist in the current compilation session, so
    // we use `Option<DefId>` so that we can cache a lookup failure.
    def_path_hash_to_def_id_cache: Lock<UnhashMap<DefPathHash, Option<DefId>>>,
}

// This type is used only for serialization and deserialization.
#[derive(Encodable, Decodable)]
struct Footer {
    file_index_to_stable_id: FxHashMap<SourceFileIndex, EncodedSourceFileId>,
    query_result_index: EncodedDepNodeIndex,
    side_effects_index: EncodedDepNodeIndex,
    // The location of all allocations.
    interpret_alloc_index: Vec<u32>,
    // See `OnDiskCache.syntax_contexts`
    syntax_contexts: FxHashMap<u32, AbsoluteBytePos>,
    // See `OnDiskCache.expn_data`
    expn_data: UnhashMap<ExpnHash, AbsoluteBytePos>,
    foreign_def_path_hashes: UnhashMap<DefPathHash, RawDefId>,
    foreign_expn_data: UnhashMap<ExpnHash, u32>,
}

pub type EncodedDepNodeIndex = Vec<(SerializedDepNodeIndex, AbsoluteBytePos)>;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Encodable, Decodable)]
struct SourceFileIndex(u32);

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Encodable, Decodable)]
pub struct AbsoluteBytePos(u32);

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

/// An `EncodedSourceFileId` is the same as a `StableSourceFileId` except that
/// the source crate is represented as a [StableCrateId] instead of as a
/// `CrateNum`. This way `EncodedSourceFileId` can be encoded and decoded
/// without any additional context, i.e. with a simple `opaque::Decoder` (which
/// is the only thing available when decoding the cache's [Footer].
#[derive(Encodable, Decodable, Clone, Debug)]
struct EncodedSourceFileId {
    file_name_hash: u64,
    stable_crate_id: StableCrateId,
}

impl EncodedSourceFileId {
    fn translate(&self, cnum_map: &UnhashMap<StableCrateId, CrateNum>) -> StableSourceFileId {
        let cnum = cnum_map[&self.stable_crate_id];
        StableSourceFileId { file_name_hash: self.file_name_hash, cnum }
    }

    fn new(tcx: TyCtxt<'_>, file: &SourceFile) -> EncodedSourceFileId {
        let source_file_id = StableSourceFileId::new(file);
        EncodedSourceFileId {
            file_name_hash: source_file_id.file_name_hash,
            stable_crate_id: tcx.stable_crate_id(source_file_id.cnum),
        }
    }
}

impl<'sess> rustc_middle::ty::OnDiskCache<'sess> for OnDiskCache<'sess> {
    /// Creates a new `OnDiskCache` instance from the serialized data in `data`.
    fn new(sess: &'sess Session, data: Mmap, start_pos: usize) -> Self {
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
            serialized_data: RwLock::new(Some(data)),
            file_index_to_stable_id: footer.file_index_to_stable_id,
            file_index_to_file: Default::default(),
            cnum_map: OnceCell::new(),
            source_map: sess.source_map(),
            current_side_effects: Default::default(),
            query_result_index: footer.query_result_index.into_iter().collect(),
            prev_side_effects_index: footer.side_effects_index.into_iter().collect(),
            alloc_decoding_state: AllocDecodingState::new(footer.interpret_alloc_index),
            syntax_contexts: footer.syntax_contexts,
            expn_data: footer.expn_data,
            foreign_expn_data: footer.foreign_expn_data,
            hygiene_context: Default::default(),
            foreign_def_path_hashes: footer.foreign_def_path_hashes,
            latest_foreign_def_path_hashes: Default::default(),
            def_path_hash_to_def_id_cache: Default::default(),
        }
    }

    fn new_empty(source_map: &'sess SourceMap) -> Self {
        Self {
            serialized_data: RwLock::new(None),
            file_index_to_stable_id: Default::default(),
            file_index_to_file: Default::default(),
            cnum_map: OnceCell::new(),
            source_map,
            current_side_effects: Default::default(),
            query_result_index: Default::default(),
            prev_side_effects_index: Default::default(),
            alloc_decoding_state: AllocDecodingState::new(Vec::new()),
            syntax_contexts: FxHashMap::default(),
            expn_data: UnhashMap::default(),
            foreign_expn_data: UnhashMap::default(),
            hygiene_context: Default::default(),
            foreign_def_path_hashes: Default::default(),
            latest_foreign_def_path_hashes: Default::default(),
            def_path_hash_to_def_id_cache: Default::default(),
        }
    }

    /// Execute all cache promotions and release the serialized backing Mmap.
    ///
    /// Cache promotions require invoking queries, which needs to read the serialized data.
    /// In order to serialize the new on-disk cache, the former on-disk cache file needs to be
    /// deleted, hence we won't be able to refer to its memmapped data.
    fn drop_serialized_data(&self, tcx: TyCtxt<'tcx>) {
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
        tcx.dep_graph.exec_cache_promotions(QueryCtxt::from_tcx(tcx));

        *self.serialized_data.write() = None;
    }

    fn serialize<'tcx>(&self, tcx: TyCtxt<'tcx>, encoder: &mut FileEncoder) -> FileEncodeResult {
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
                    let source_file_id = EncodedSourceFileId::new(tcx, &file);
                    file_index_to_stable_id.insert(index, source_file_id);
                }

                (file_to_file_index, file_index_to_stable_id)
            };

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
            let mut query_result_index = EncodedDepNodeIndex::new();

            tcx.sess.time("encode_query_results", || -> FileEncodeResult {
                let enc = &mut encoder;
                let qri = &mut query_result_index;
                QueryCtxt::from_tcx(tcx).encode_query_results(enc, qri)
            })?;

            // Encode side effects.
            let side_effects_index: EncodedDepNodeIndex = self
                .current_side_effects
                .borrow()
                .iter()
                .map(
                    |(dep_node_index, side_effects)| -> Result<_, <FileEncoder as Encoder>::Error> {
                        let pos = AbsoluteBytePos::new(encoder.position());
                        let dep_node_index = SerializedDepNodeIndex::new(dep_node_index.index());
                        encoder.encode_tagged(dep_node_index, side_effects)?;

                        Ok((dep_node_index, pos))
                    },
                )
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

            let mut syntax_contexts = FxHashMap::default();
            let mut expn_data = UnhashMap::default();
            let mut foreign_expn_data = UnhashMap::default();

            // Encode all hygiene data (`SyntaxContextData` and `ExpnData`) from the current
            // session.

            hygiene_encode_context.encode(
                &mut encoder,
                |encoder, index, ctxt_data| -> FileEncodeResult {
                    let pos = AbsoluteBytePos::new(encoder.position());
                    encoder.encode_tagged(TAG_SYNTAX_CONTEXT, ctxt_data)?;
                    syntax_contexts.insert(index, pos);
                    Ok(())
                },
                |encoder, expn_id, data, hash| -> FileEncodeResult {
                    if expn_id.krate == LOCAL_CRATE {
                        let pos = AbsoluteBytePos::new(encoder.position());
                        encoder.encode_tagged(TAG_EXPN_DATA, data)?;
                        expn_data.insert(hash, pos);
                    } else {
                        foreign_expn_data.insert(hash, expn_id.local_id.as_u32());
                    }
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
                    query_result_index,
                    side_effects_index,
                    interpret_alloc_index,
                    syntax_contexts,
                    expn_data,
                    foreign_expn_data,
                    foreign_def_path_hashes,
                },
            )?;

            // Encode the position of the footer as the last 8 bytes of the
            // file so we know where to look for it.
            IntEncodedWithFixedSize(footer_pos).encode(encoder.encoder)?;

            // DO NOT WRITE ANYTHING TO THE ENCODER AFTER THIS POINT! The address
            // of the footer must be the last thing in the data stream.

            Ok(())
        })
    }

    fn def_path_hash_to_def_id(&self, tcx: TyCtxt<'tcx>, hash: DefPathHash) -> Option<DefId> {
        let mut cache = self.def_path_hash_to_def_id_cache.lock();
        match cache.entry(hash) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                debug!("def_path_hash_to_def_id({:?})", hash);
                // Check if the `DefPathHash` corresponds to a definition in the current
                // crate
                if let Some(def_id) =
                    tcx.definitions_untracked().local_def_path_hash_to_def_id(hash)
                {
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
                let krate = self.try_remap_cnum(tcx, hash.stable_crate_id())?;
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
                let opt_def_id =
                    tcx.cstore_untracked().def_path_hash_to_def_id(krate, raw_def_id.index, hash);
                debug!("def_path_to_def_id({:?}): opt_def_id = {:?}", hash, opt_def_id);
                e.insert(opt_def_id);
                opt_def_id
            }
        }
    }

    fn register_reused_dep_node(&self, tcx: TyCtxt<'sess>, dep_node: &DepNode) {
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

    fn store_foreign_def_id_hash(&self, def_id: DefId, hash: DefPathHash) {
        // We may overwrite an existing entry, but it will have the same value,
        // so it's fine
        self.latest_foreign_def_path_hashes
            .lock()
            .insert(hash, RawDefId { krate: def_id.krate.as_u32(), index: def_id.index.as_u32() });
    }
}

impl<'sess> OnDiskCache<'sess> {
    pub fn as_dyn(&self) -> &dyn rustc_middle::ty::OnDiskCache<'sess> {
        self as _
    }

    /// Loads a `QuerySideEffects` created during the previous compilation session.
    pub fn load_side_effects(
        &self,
        tcx: TyCtxt<'_>,
        dep_node_index: SerializedDepNodeIndex,
    ) -> QuerySideEffects {
        let side_effects: Option<QuerySideEffects> =
            self.load_indexed(tcx, dep_node_index, &self.prev_side_effects_index, "side_effects");

        side_effects.unwrap_or_default()
    }

    /// Stores a `QuerySideEffects` emitted during the current compilation session.
    /// Anything stored like this will be available via `load_side_effects` in
    /// the next compilation session.
    #[inline(never)]
    #[cold]
    pub fn store_side_effects(&self, dep_node_index: DepNodeIndex, side_effects: QuerySideEffects) {
        let mut current_side_effects = self.current_side_effects.borrow_mut();
        let prev = current_side_effects.insert(dep_node_index, side_effects);
        debug_assert!(prev.is_none());
    }

    fn get_raw_def_id(&self, hash: &DefPathHash) -> Option<RawDefId> {
        self.foreign_def_path_hashes.get(hash).copied()
    }

    fn try_remap_cnum(&self, tcx: TyCtxt<'_>, stable_crate_id: StableCrateId) -> Option<CrateNum> {
        let cnum_map = self.cnum_map.get_or_init(|| Self::compute_cnum_map(tcx));
        debug!("try_remap_cnum({:?}): cnum_map={:?}", stable_crate_id, cnum_map);

        cnum_map.get(&stable_crate_id).copied()
    }

    /// Returns the cached query result if there is something in the cache for
    /// the given `SerializedDepNodeIndex`; otherwise returns `None`.
    pub fn try_load_query_result<'tcx, T>(
        &self,
        tcx: TyCtxt<'tcx>,
        dep_node_index: SerializedDepNodeIndex,
    ) -> Option<T>
    where
        T: for<'a> Decodable<CacheDecoder<'a, 'tcx>>,
    {
        self.load_indexed(tcx, dep_node_index, &self.query_result_index, "query result")
    }

    /// Stores side effect emitted during computation of an anonymous query.
    /// Since many anonymous queries can share the same `DepNode`, we aggregate
    /// them -- as opposed to regular queries where we assume that there is a
    /// 1:1 relationship between query-key and `DepNode`.
    #[inline(never)]
    #[cold]
    pub fn store_side_effects_for_anon_node(
        &self,
        dep_node_index: DepNodeIndex,
        side_effects: QuerySideEffects,
    ) {
        let mut current_side_effects = self.current_side_effects.borrow_mut();

        let x = current_side_effects.entry(dep_node_index).or_default();
        x.append(side_effects);
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

    fn with_decoder<'a, 'tcx, T, F: for<'s> FnOnce(&mut CacheDecoder<'s, 'tcx>) -> T>(
        &'sess self,
        tcx: TyCtxt<'tcx>,
        pos: AbsoluteBytePos,
        f: F,
    ) -> T
    where
        T: Decodable<CacheDecoder<'a, 'tcx>>,
    {
        let cnum_map = self.cnum_map.get_or_init(|| Self::compute_cnum_map(tcx));

        let serialized_data = self.serialized_data.read();
        let mut decoder = CacheDecoder {
            tcx,
            opaque: opaque::Decoder::new(serialized_data.as_deref().unwrap_or(&[]), pos.to_usize()),
            source_map: self.source_map,
            cnum_map,
            file_index_to_file: &self.file_index_to_file,
            file_index_to_stable_id: &self.file_index_to_stable_id,
            alloc_decoding_session: self.alloc_decoding_state.new_decoding_session(),
            syntax_contexts: &self.syntax_contexts,
            expn_data: &self.expn_data,
            foreign_expn_data: &self.foreign_expn_data,
            hygiene_context: &self.hygiene_context,
        };
        f(&mut decoder)
    }

    // This function builds mapping from previous-session-`CrateNum` to
    // current-session-`CrateNum`. There might be `CrateNum`s from the previous
    // `Session` that don't occur in the current one. For these, the mapping
    // maps to None.
    fn compute_cnum_map(tcx: TyCtxt<'_>) -> UnhashMap<StableCrateId, CrateNum> {
        tcx.dep_graph.with_ignore(|| {
            tcx.crates(())
                .iter()
                .chain(std::iter::once(&LOCAL_CRATE))
                .map(|&cnum| {
                    let hash = tcx.def_path_hash(cnum.as_def_id()).stable_crate_id();
                    (hash, cnum)
                })
                .collect()
        })
    }
}

//- DECODING -------------------------------------------------------------------

/// A decoder that can read from the incremental compilation cache. It is similar to the one
/// we use for crate metadata decoding in that it can rebase spans and eventually
/// will also handle things that contain `Ty` instances.
pub struct CacheDecoder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    opaque: opaque::Decoder<'a>,
    source_map: &'a SourceMap,
    cnum_map: &'a UnhashMap<StableCrateId, CrateNum>,
    file_index_to_file: &'a Lock<FxHashMap<SourceFileIndex, Lrc<SourceFile>>>,
    file_index_to_stable_id: &'a FxHashMap<SourceFileIndex, EncodedSourceFileId>,
    alloc_decoding_session: AllocDecodingSession<'a>,
    syntax_contexts: &'a FxHashMap<u32, AbsoluteBytePos>,
    expn_data: &'a UnhashMap<ExpnHash, AbsoluteBytePos>,
    foreign_expn_data: &'a UnhashMap<ExpnHash, u32>,
    hygiene_context: &'a HygieneDecodeContext,
}

impl<'a, 'tcx> CacheDecoder<'a, 'tcx> {
    fn file_index_to_file(&self, index: SourceFileIndex) -> Lrc<SourceFile> {
        let CacheDecoder {
            ref file_index_to_file,
            ref file_index_to_stable_id,
            ref source_map,
            ref cnum_map,
            ..
        } = *self;

        file_index_to_file
            .borrow_mut()
            .entry(index)
            .or_insert_with(|| {
                let stable_id = file_index_to_stable_id[&index].translate(cnum_map);
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

        let cache_key = ty::CReaderCacheKey { cnum: None, pos: shorthand };

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

    fn decode_alloc_id(&mut self) -> Result<interpret::AllocId, Self::Error> {
        let alloc_decoding_session = self.alloc_decoding_session;
        alloc_decoding_session.decode_alloc_id(self)
    }
}

rustc_middle::implement_ty_decoder!(CacheDecoder<'a, 'tcx>);

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
        let hash = ExpnHash::decode(decoder)?;
        if hash.is_root() {
            return Ok(ExpnId::root());
        }

        if let Some(expn_id) = ExpnId::from_hash(hash) {
            return Ok(expn_id);
        }

        let krate = decoder.cnum_map[&hash.stable_crate_id()];

        let expn_id = if krate == LOCAL_CRATE {
            // We look up the position of the associated `ExpnData` and decode it.
            let pos = decoder
                .expn_data
                .get(&hash)
                .unwrap_or_else(|| panic!("Bad hash {:?} (map {:?})", hash, decoder.expn_data));

            let data: ExpnData = decoder
                .with_position(pos.to_usize(), |decoder| decode_tagged(decoder, TAG_EXPN_DATA))?;
            rustc_span::hygiene::register_local_expn_id(data, hash)
        } else {
            let index_guess = decoder.foreign_expn_data[&hash];
            decoder.tcx.cstore_untracked().expn_hash_to_expn_id(krate, index_guess, hash)
        };

        #[cfg(debug_assertions)]
        {
            use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
            let mut hcx = decoder.tcx.create_stable_hashing_context();
            let mut hasher = StableHasher::new();
            hcx.while_hashing_spans(true, |hcx| expn_id.expn_data().hash_stable(hcx, &mut hasher));
            let local_hash: u64 = hasher.finish();
            debug_assert_eq!(hash.local_hash(), local_hash);
        }

        Ok(expn_id)
    }
}

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for Span {
    fn decode(decoder: &mut CacheDecoder<'a, 'tcx>) -> Result<Self, String> {
        let ctxt = SyntaxContext::decode(decoder)?;
        let parent = Option::<LocalDefId>::decode(decoder)?;
        let tag: u8 = Decodable::decode(decoder)?;

        if tag == TAG_PARTIAL_SPAN {
            return Ok(Span::new(BytePos(0), BytePos(0), ctxt, parent));
        } else if tag == TAG_RELATIVE_SPAN {
            let dlo = u32::decode(decoder)?;
            let dto = u32::decode(decoder)?;

            let enclosing =
                decoder.tcx.definitions_untracked().def_span(parent.unwrap()).data_untracked();
            let span = Span::new(
                enclosing.lo + BytePos::from_u32(dlo),
                enclosing.lo + BytePos::from_u32(dto),
                ctxt,
                parent,
            );

            return Ok(span);
        } else {
            debug_assert_eq!(tag, TAG_FULL_SPAN);
        }

        let file_lo_index = SourceFileIndex::decode(decoder)?;
        let line_lo = usize::decode(decoder)?;
        let col_lo = BytePos::decode(decoder)?;
        let len = BytePos::decode(decoder)?;

        let file_lo = decoder.file_index_to_file(file_lo_index);
        let lo = file_lo.lines[line_lo - 1] + col_lo;
        let hi = lo + len;

        Ok(Span::new(lo, hi, ctxt, parent))
    }
}

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for CrateNum {
    fn decode(d: &mut CacheDecoder<'a, 'tcx>) -> Result<Self, String> {
        let stable_id = StableCrateId::decode(d)?;
        let cnum = d.cnum_map[&stable_id];
        Ok(cnum)
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
            .on_disk_cache
            .as_ref()
            .unwrap()
            .def_path_hash_to_def_id(d.tcx(), def_path_hash)
            .unwrap())
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

pub trait OpaqueEncoder: Encoder {
    fn position(&self) -> usize;
}

impl OpaqueEncoder for FileEncoder {
    #[inline]
    fn position(&self) -> usize {
        FileEncoder::position(self)
    }
}

/// An encoder that can write to the incremental compilation cache.
pub struct CacheEncoder<'a, 'tcx, E: OpaqueEncoder> {
    tcx: TyCtxt<'tcx>,
    encoder: &'a mut E,
    type_shorthands: FxHashMap<Ty<'tcx>, usize>,
    predicate_shorthands: FxHashMap<ty::PredicateKind<'tcx>, usize>,
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
        s.hygiene_context.schedule_expn_data_for_encoding(*self);
        self.expn_hash().encode(s)
    }
}

impl<'a, 'tcx, E> Encodable<CacheEncoder<'a, 'tcx, E>> for Span
where
    E: 'a + OpaqueEncoder,
{
    fn encode(&self, s: &mut CacheEncoder<'a, 'tcx, E>) -> Result<(), E::Error> {
        let span_data = self.data_untracked();
        span_data.ctxt.encode(s)?;
        span_data.parent.encode(s)?;

        if span_data.is_dummy() {
            return TAG_PARTIAL_SPAN.encode(s);
        }

        if let Some(parent) = span_data.parent {
            let enclosing = s.tcx.definitions_untracked().def_span(parent).data_untracked();
            if enclosing.contains(span_data) {
                TAG_RELATIVE_SPAN.encode(s)?;
                (span_data.lo - enclosing.lo).to_u32().encode(s)?;
                (span_data.hi - enclosing.lo).to_u32().encode(s)?;
                return Ok(());
            }
        }

        let pos = s.source_map.byte_pos_to_line_and_col(span_data.lo);
        let partial_span = match &pos {
            Some((file_lo, _, _)) => !file_lo.contains(span_data.hi),
            None => true,
        };

        if partial_span {
            return TAG_PARTIAL_SPAN.encode(s);
        }

        let (file_lo, line_lo, col_lo) = pos.unwrap();

        let len = span_data.hi - span_data.lo;

        let source_file_index = s.source_file_index(file_lo);

        TAG_FULL_SPAN.encode(s)?;
        source_file_index.encode(s)?;
        line_lo.encode(s)?;
        col_lo.encode(s)?;
        len.encode(s)
    }
}

impl<'a, 'tcx, E> TyEncoder<'tcx> for CacheEncoder<'a, 'tcx, E>
where
    E: 'a + OpaqueEncoder,
{
    const CLEAR_CROSS_CRATE: bool = false;

    fn position(&self) -> usize {
        self.encoder.position()
    }
    fn type_shorthands(&mut self) -> &mut FxHashMap<Ty<'tcx>, usize> {
        &mut self.type_shorthands
    }
    fn predicate_shorthands(&mut self) -> &mut FxHashMap<ty::PredicateKind<'tcx>, usize> {
        &mut self.predicate_shorthands
    }
    fn encode_alloc_id(&mut self, alloc_id: &interpret::AllocId) -> Result<(), Self::Error> {
        let (index, _) = self.interpret_allocs.insert_full(*alloc_id);

        index.encode(self)
    }
}

impl<'a, 'tcx, E> Encodable<CacheEncoder<'a, 'tcx, E>> for CrateNum
where
    E: 'a + OpaqueEncoder,
{
    fn encode(&self, s: &mut CacheEncoder<'a, 'tcx, E>) -> Result<(), E::Error> {
        s.tcx.stable_crate_id(*self).encode(s)
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
        emit_raw_bytes(&[u8]);
    }
}

// This ensures that the `Encodable<opaque::FileEncoder>::encode` specialization for byte slices
// is used when a `CacheEncoder` having an `opaque::FileEncoder` is passed to `Encodable::encode`.
// Unfortunately, we have to manually opt into specializations this way, given how `CacheEncoder`
// and the encoding traits currently work.
impl<'a, 'tcx> Encodable<CacheEncoder<'a, 'tcx, FileEncoder>> for [u8] {
    fn encode(&self, e: &mut CacheEncoder<'a, 'tcx, FileEncoder>) -> FileEncodeResult {
        self.encode(e.encoder)
    }
}

pub fn encode_query_results<'a, 'tcx, CTX, Q>(
    tcx: CTX,
    encoder: &mut CacheEncoder<'a, 'tcx, FileEncoder>,
    query_result_index: &mut EncodedDepNodeIndex,
) -> FileEncodeResult
where
    CTX: QueryContext + 'tcx,
    Q: super::QueryDescription<CTX> + super::QueryAccessors<CTX>,
    Q::Value: Encodable<CacheEncoder<'a, 'tcx, FileEncoder>>,
{
    let _timer = tcx
        .dep_context()
        .profiler()
        .extra_verbose_generic_activity("encode_query_results_for", std::any::type_name::<Q>());

    assert!(Q::query_state(tcx).all_inactive());
    let cache = Q::query_cache(tcx);
    let mut res = Ok(());
    cache.iter_results(&mut |key, value, dep_node| {
        if res.is_err() {
            return;
        }
        if Q::cache_on_disk(tcx, &key, Some(value)) {
            let dep_node = SerializedDepNodeIndex::new(dep_node.index());

            // Record position of the cache entry.
            query_result_index.push((dep_node, AbsoluteBytePos::new(encoder.encoder.position())));

            // Encode the type check tables with the `SerializedDepNodeIndex`
            // as tag.
            match encoder.encode_tagged(dep_node, value) {
                Ok(()) => {}
                Err(e) => {
                    res = Err(e);
                }
            }
        }
    });

    res
}
