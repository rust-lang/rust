use rustc_data_structures::fx::{FxHashMap, FxIndexSet};
use rustc_data_structures::memmap::Mmap;
use rustc_data_structures::stable_hasher::Hash64;
use rustc_data_structures::sync::{HashMapExt, Lock, Lrc, RwLock};
use rustc_data_structures::unhash::UnhashMap;
use rustc_data_structures::unord::UnordSet;
use rustc_hir::def_id::{CrateNum, DefId, DefIndex, LocalDefId, StableCrateId, LOCAL_CRATE};
use rustc_hir::definitions::DefPathHash;
use rustc_index::{Idx, IndexVec};
use rustc_middle::dep_graph::{DepNodeIndex, SerializedDepNodeIndex};
use rustc_middle::mir::interpret::{AllocDecodingSession, AllocDecodingState};
use rustc_middle::mir::{self, interpret};
use rustc_middle::ty::codec::{RefDecodable, TyDecoder, TyEncoder};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_query_system::query::QuerySideEffects;
use rustc_serialize::{
    opaque::{FileEncodeResult, FileEncoder, IntEncodedWithFixedSize, MemDecoder},
    Decodable, Decoder, Encodable, Encoder,
};
use rustc_session::Session;
use rustc_span::hygiene::{
    ExpnId, HygieneDecodeContext, HygieneEncodeContext, SyntaxContext, SyntaxContextData,
};
use rustc_span::source_map::{SourceMap, StableSourceFileId};
use rustc_span::{BytePos, ExpnData, ExpnHash, Pos, SourceFile, Span};
use rustc_span::{CachingSourceMapView, Symbol};
use std::collections::hash_map::Entry;
use std::io;
use std::mem;

const TAG_FILE_FOOTER: u128 = 0xC0FFEE_C0FFEE_C0FFEE_C0FFEE_C0FFEE;

// A normal span encoded with both location information and a `SyntaxContext`
const TAG_FULL_SPAN: u8 = 0;
// A partial span with no location information, encoded only with a `SyntaxContext`
const TAG_PARTIAL_SPAN: u8 = 1;
const TAG_RELATIVE_SPAN: u8 = 2;

const TAG_SYNTAX_CONTEXT: u8 = 0;
const TAG_EXPN_DATA: u8 = 1;

// Tags for encoding Symbol's
const SYMBOL_STR: u8 = 0;
const SYMBOL_OFFSET: u8 = 1;
const SYMBOL_PREINTERNED: u8 = 2;

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
    // Maps `ExpnHash`es to their raw value from the *previous*
    // compilation session. This is used as an initial 'guess' when
    // we try to map an `ExpnHash` to its value in the current
    // compilation session.
    foreign_expn_data: UnhashMap<ExpnHash, u32>,
}

// This type is used only for serialization and deserialization.
#[derive(Encodable, Decodable)]
struct Footer {
    file_index_to_stable_id: FxHashMap<SourceFileIndex, EncodedSourceFileId>,
    query_result_index: EncodedDepNodeIndex,
    side_effects_index: EncodedDepNodeIndex,
    // The location of all allocations.
    // Most uses only need values up to u32::MAX, but benchmarking indicates that we can use a u64
    // without measurable overhead. This permits larger const allocations without ICEing.
    interpret_alloc_index: Vec<u64>,
    // See `OnDiskCache.syntax_contexts`
    syntax_contexts: FxHashMap<u32, AbsoluteBytePos>,
    // See `OnDiskCache.expn_data`
    expn_data: UnhashMap<ExpnHash, AbsoluteBytePos>,
    foreign_expn_data: UnhashMap<ExpnHash, u32>,
}

pub type EncodedDepNodeIndex = Vec<(SerializedDepNodeIndex, AbsoluteBytePos)>;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Encodable, Decodable)]
struct SourceFileIndex(u32);

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Encodable, Decodable)]
pub struct AbsoluteBytePos(u64);

impl AbsoluteBytePos {
    #[inline]
    pub fn new(pos: usize) -> AbsoluteBytePos {
        AbsoluteBytePos(pos.try_into().expect("Incremental cache file size overflowed u64."))
    }

    #[inline]
    fn to_usize(self) -> usize {
        self.0 as usize
    }
}

/// An `EncodedSourceFileId` is the same as a `StableSourceFileId` except that
/// the source crate is represented as a [StableCrateId] instead of as a
/// `CrateNum`. This way `EncodedSourceFileId` can be encoded and decoded
/// without any additional context, i.e. with a simple `opaque::Decoder` (which
/// is the only thing available when decoding the cache's [Footer].
#[derive(Encodable, Decodable, Clone, Debug)]
struct EncodedSourceFileId {
    file_name_hash: Hash64,
    stable_crate_id: StableCrateId,
}

impl EncodedSourceFileId {
    #[inline]
    fn translate(&self, tcx: TyCtxt<'_>) -> StableSourceFileId {
        let cnum = tcx.stable_crate_id_to_crate_num(self.stable_crate_id);
        StableSourceFileId { file_name_hash: self.file_name_hash, cnum }
    }

    #[inline]
    fn new(tcx: TyCtxt<'_>, file: &SourceFile) -> EncodedSourceFileId {
        let source_file_id = StableSourceFileId::new(file);
        EncodedSourceFileId {
            file_name_hash: source_file_id.file_name_hash,
            stable_crate_id: tcx.stable_crate_id(source_file_id.cnum),
        }
    }
}

impl<'sess> OnDiskCache<'sess> {
    /// Creates a new `OnDiskCache` instance from the serialized data in `data`.
    pub fn new(sess: &'sess Session, data: Mmap, start_pos: usize) -> Self {
        debug_assert!(sess.opts.incremental.is_some());

        // Wrap in a scope so we can borrow `data`.
        let footer: Footer = {
            let mut decoder = MemDecoder::new(&data, start_pos);

            // Decode the *position* of the footer, which can be found in the
            // last 8 bytes of the file.
            let footer_pos = decoder
                .with_position(decoder.len() - IntEncodedWithFixedSize::ENCODED_SIZE, |decoder| {
                    IntEncodedWithFixedSize::decode(decoder).0 as usize
                });
            // Decode the file footer, which contains all the lookup tables, etc.
            decoder.with_position(footer_pos, |decoder| decode_tagged(decoder, TAG_FILE_FOOTER))
        };

        Self {
            serialized_data: RwLock::new(Some(data)),
            file_index_to_stable_id: footer.file_index_to_stable_id,
            file_index_to_file: Default::default(),
            source_map: sess.source_map(),
            current_side_effects: Default::default(),
            query_result_index: footer.query_result_index.into_iter().collect(),
            prev_side_effects_index: footer.side_effects_index.into_iter().collect(),
            alloc_decoding_state: AllocDecodingState::new(footer.interpret_alloc_index),
            syntax_contexts: footer.syntax_contexts,
            expn_data: footer.expn_data,
            foreign_expn_data: footer.foreign_expn_data,
            hygiene_context: Default::default(),
        }
    }

    pub fn new_empty(source_map: &'sess SourceMap) -> Self {
        Self {
            serialized_data: RwLock::new(None),
            file_index_to_stable_id: Default::default(),
            file_index_to_file: Default::default(),
            source_map,
            current_side_effects: Default::default(),
            query_result_index: Default::default(),
            prev_side_effects_index: Default::default(),
            alloc_decoding_state: AllocDecodingState::new(Vec::new()),
            syntax_contexts: FxHashMap::default(),
            expn_data: UnhashMap::default(),
            foreign_expn_data: UnhashMap::default(),
            hygiene_context: Default::default(),
        }
    }

    /// Execute all cache promotions and release the serialized backing Mmap.
    ///
    /// Cache promotions require invoking queries, which needs to read the serialized data.
    /// In order to serialize the new on-disk cache, the former on-disk cache file needs to be
    /// deleted, hence we won't be able to refer to its memmapped data.
    pub fn drop_serialized_data(&self, tcx: TyCtxt<'_>) {
        // Load everything into memory so we can write it out to the on-disk
        // cache. The vast majority of cacheable query results should already
        // be in memory, so this should be a cheap operation.
        // Do this *before* we clone 'latest_foreign_def_path_hashes', since
        // loading existing queries may cause us to create new DepNodes, which
        // may in turn end up invoking `store_foreign_def_id_hash`
        tcx.dep_graph.exec_cache_promotions(tcx);

        *self.serialized_data.write() = None;
    }

    pub fn serialize(&self, tcx: TyCtxt<'_>, encoder: FileEncoder) -> FileEncodeResult {
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
                symbol_table: Default::default(),
            };

            // Encode query results.
            let mut query_result_index = EncodedDepNodeIndex::new();

            tcx.sess.time("encode_query_results", || {
                let enc = &mut encoder;
                let qri = &mut query_result_index;
                (tcx.query_system.fns.encode_query_results)(tcx, enc, qri);
            });

            // Encode side effects.
            let side_effects_index: EncodedDepNodeIndex = self
                .current_side_effects
                .borrow()
                .iter()
                .map(|(dep_node_index, side_effects)| {
                    let pos = AbsoluteBytePos::new(encoder.position());
                    let dep_node_index = SerializedDepNodeIndex::new(dep_node_index.index());
                    encoder.encode_tagged(dep_node_index, side_effects);

                    (dep_node_index, pos)
                })
                .collect();

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
                        let pos: u64 = encoder.position().try_into().unwrap();
                        interpret_alloc_index.push(pos);
                        interpret::specialized_encode_alloc_id(&mut encoder, tcx, id);
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
                |encoder, index, ctxt_data| {
                    let pos = AbsoluteBytePos::new(encoder.position());
                    encoder.encode_tagged(TAG_SYNTAX_CONTEXT, ctxt_data);
                    syntax_contexts.insert(index, pos);
                },
                |encoder, expn_id, data, hash| {
                    if expn_id.krate == LOCAL_CRATE {
                        let pos = AbsoluteBytePos::new(encoder.position());
                        encoder.encode_tagged(TAG_EXPN_DATA, data);
                        expn_data.insert(hash, pos);
                    } else {
                        foreign_expn_data.insert(hash, expn_id.local_id.as_u32());
                    }
                },
            );

            // Encode the file footer.
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
                },
            );

            // Encode the position of the footer as the last 8 bytes of the
            // file so we know where to look for it.
            IntEncodedWithFixedSize(footer_pos).encode(&mut encoder.encoder);

            // DO NOT WRITE ANYTHING TO THE ENCODER AFTER THIS POINT! The address
            // of the footer must be the last thing in the data stream.

            encoder.finish()
        })
    }

    /// Loads a `QuerySideEffects` created during the previous compilation session.
    pub fn load_side_effects(
        &self,
        tcx: TyCtxt<'_>,
        dep_node_index: SerializedDepNodeIndex,
    ) -> QuerySideEffects {
        let side_effects: Option<QuerySideEffects> =
            self.load_indexed(tcx, dep_node_index, &self.prev_side_effects_index);

        side_effects.unwrap_or_default()
    }

    /// Stores a `QuerySideEffects` emitted during the current compilation session.
    /// Anything stored like this will be available via `load_side_effects` in
    /// the next compilation session.
    pub fn store_side_effects(&self, dep_node_index: DepNodeIndex, side_effects: QuerySideEffects) {
        let mut current_side_effects = self.current_side_effects.borrow_mut();
        let prev = current_side_effects.insert(dep_node_index, side_effects);
        debug_assert!(prev.is_none());
    }

    /// Return whether the cached query result can be decoded.
    #[inline]
    pub fn loadable_from_disk(&self, dep_node_index: SerializedDepNodeIndex) -> bool {
        self.query_result_index.contains_key(&dep_node_index)
        // with_decoder is infallible, so we can stop here
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
        let opt_value = self.load_indexed(tcx, dep_node_index, &self.query_result_index);
        debug_assert_eq!(opt_value.is_some(), self.loadable_from_disk(dep_node_index));
        opt_value
    }

    /// Stores side effect emitted during computation of an anonymous query.
    /// Since many anonymous queries can share the same `DepNode`, we aggregate
    /// them -- as opposed to regular queries where we assume that there is a
    /// 1:1 relationship between query-key and `DepNode`.
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
    ) -> Option<T>
    where
        T: for<'a> Decodable<CacheDecoder<'a, 'tcx>>,
    {
        let pos = index.get(&dep_node_index).cloned()?;
        let value = self.with_decoder(tcx, pos, |decoder| decode_tagged(decoder, dep_node_index));
        Some(value)
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
        let serialized_data = self.serialized_data.read();
        let mut decoder = CacheDecoder {
            tcx,
            opaque: MemDecoder::new(serialized_data.as_deref().unwrap_or(&[]), pos.to_usize()),
            source_map: self.source_map,
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
}

//- DECODING -------------------------------------------------------------------

/// A decoder that can read from the incremental compilation cache. It is similar to the one
/// we use for crate metadata decoding in that it can rebase spans and eventually
/// will also handle things that contain `Ty` instances.
pub struct CacheDecoder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    opaque: MemDecoder<'a>,
    source_map: &'a SourceMap,
    file_index_to_file: &'a Lock<FxHashMap<SourceFileIndex, Lrc<SourceFile>>>,
    file_index_to_stable_id: &'a FxHashMap<SourceFileIndex, EncodedSourceFileId>,
    alloc_decoding_session: AllocDecodingSession<'a>,
    syntax_contexts: &'a FxHashMap<u32, AbsoluteBytePos>,
    expn_data: &'a UnhashMap<ExpnHash, AbsoluteBytePos>,
    foreign_expn_data: &'a UnhashMap<ExpnHash, u32>,
    hygiene_context: &'a HygieneDecodeContext,
}

impl<'a, 'tcx> CacheDecoder<'a, 'tcx> {
    #[inline]
    fn file_index_to_file(&self, index: SourceFileIndex) -> Lrc<SourceFile> {
        let CacheDecoder {
            tcx,
            ref file_index_to_file,
            ref file_index_to_stable_id,
            ref source_map,
            ..
        } = *self;

        file_index_to_file
            .borrow_mut()
            .entry(index)
            .or_insert_with(|| {
                let stable_id = file_index_to_stable_id[&index].translate(tcx);

                // If this `SourceFile` is from a foreign crate, then make sure
                // that we've imported all of the source files from that crate.
                // This has usually already been done during macro invocation.
                // However, when encoding query results like `TypeckResults`,
                // we might encode an `AdtDef` for a foreign type (because it
                // was referenced in the body of the function). There is no guarantee
                // that we will load the source files from that crate during macro
                // expansion, so we use `import_source_files` to ensure that the foreign
                // source files are actually imported before we call `source_file_by_stable_id`.
                if stable_id.cnum != LOCAL_CRATE {
                    self.tcx.cstore_untracked().import_source_files(self.tcx.sess, stable_id.cnum);
                }

                source_map
                    .source_file_by_stable_id(stable_id)
                    .expect("failed to lookup `SourceFile` in new context")
            })
            .clone()
    }
}

// Decodes something that was encoded with `encode_tagged()` and verify that the
// tag matches and the correct amount of bytes was read.
fn decode_tagged<D, T, V>(decoder: &mut D, expected_tag: T) -> V
where
    T: Decodable<D> + Eq + std::fmt::Debug,
    V: Decodable<D>,
    D: Decoder,
{
    let start_pos = decoder.position();

    let actual_tag = T::decode(decoder);
    assert_eq!(actual_tag, expected_tag);
    let value = V::decode(decoder);
    let end_pos = decoder.position();

    let expected_len: u64 = Decodable::decode(decoder);
    assert_eq!((end_pos - start_pos) as u64, expected_len);

    value
}

impl<'a, 'tcx> TyDecoder for CacheDecoder<'a, 'tcx> {
    type I = TyCtxt<'tcx>;
    const CLEAR_CROSS_CRATE: bool = false;

    #[inline]
    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn cached_ty_for_shorthand<F>(&mut self, shorthand: usize, or_insert_with: F) -> Ty<'tcx>
    where
        F: FnOnce(&mut Self) -> Ty<'tcx>,
    {
        let tcx = self.tcx;

        let cache_key = ty::CReaderCacheKey { cnum: None, pos: shorthand };

        if let Some(&ty) = tcx.ty_rcache.borrow().get(&cache_key) {
            return ty;
        }

        let ty = or_insert_with(self);
        // This may overwrite the entry, but it should overwrite with the same value.
        tcx.ty_rcache.borrow_mut().insert_same(cache_key, ty);
        ty
    }

    fn with_position<F, R>(&mut self, pos: usize, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        debug_assert!(pos < self.opaque.len());

        let new_opaque = MemDecoder::new(self.opaque.data(), pos);
        let old_opaque = mem::replace(&mut self.opaque, new_opaque);
        let r = f(self);
        self.opaque = old_opaque;
        r
    }

    fn decode_alloc_id(&mut self) -> interpret::AllocId {
        let alloc_decoding_session = self.alloc_decoding_session;
        alloc_decoding_session.decode_alloc_id(self)
    }
}

rustc_middle::implement_ty_decoder!(CacheDecoder<'a, 'tcx>);

// This ensures that the `Decodable<opaque::Decoder>::decode` specialization for `Vec<u8>` is used
// when a `CacheDecoder` is passed to `Decodable::decode`. Unfortunately, we have to manually opt
// into specializations this way, given how `CacheDecoder` and the decoding traits currently work.
impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for Vec<u8> {
    fn decode(d: &mut CacheDecoder<'a, 'tcx>) -> Self {
        Decodable::decode(&mut d.opaque)
    }
}

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for SyntaxContext {
    fn decode(decoder: &mut CacheDecoder<'a, 'tcx>) -> Self {
        let syntax_contexts = decoder.syntax_contexts;
        rustc_span::hygiene::decode_syntax_context(decoder, decoder.hygiene_context, |this, id| {
            // This closure is invoked if we haven't already decoded the data for the `SyntaxContext` we are deserializing.
            // We look up the position of the associated `SyntaxData` and decode it.
            let pos = syntax_contexts.get(&id).unwrap();
            this.with_position(pos.to_usize(), |decoder| {
                let data: SyntaxContextData = decode_tagged(decoder, TAG_SYNTAX_CONTEXT);
                data
            })
        })
    }
}

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for ExpnId {
    fn decode(decoder: &mut CacheDecoder<'a, 'tcx>) -> Self {
        let hash = ExpnHash::decode(decoder);
        if hash.is_root() {
            return ExpnId::root();
        }

        if let Some(expn_id) = ExpnId::from_hash(hash) {
            return expn_id;
        }

        let krate = decoder.tcx.stable_crate_id_to_crate_num(hash.stable_crate_id());

        let expn_id = if krate == LOCAL_CRATE {
            // We look up the position of the associated `ExpnData` and decode it.
            let pos = decoder
                .expn_data
                .get(&hash)
                .unwrap_or_else(|| panic!("Bad hash {:?} (map {:?})", hash, decoder.expn_data));

            let data: ExpnData = decoder
                .with_position(pos.to_usize(), |decoder| decode_tagged(decoder, TAG_EXPN_DATA));
            let expn_id = rustc_span::hygiene::register_local_expn_id(data, hash);

            #[cfg(debug_assertions)]
            {
                use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
                let local_hash = decoder.tcx.with_stable_hashing_context(|mut hcx| {
                    let mut hasher = StableHasher::new();
                    expn_id.expn_data().hash_stable(&mut hcx, &mut hasher);
                    hasher.finish()
                });
                debug_assert_eq!(hash.local_hash(), local_hash);
            }

            expn_id
        } else {
            let index_guess = decoder.foreign_expn_data[&hash];
            decoder.tcx.cstore_untracked().expn_hash_to_expn_id(
                decoder.tcx.sess,
                krate,
                index_guess,
                hash,
            )
        };

        debug_assert_eq!(expn_id.krate, krate);
        expn_id
    }
}

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for Span {
    fn decode(decoder: &mut CacheDecoder<'a, 'tcx>) -> Self {
        let ctxt = SyntaxContext::decode(decoder);
        let parent = Option::<LocalDefId>::decode(decoder);
        let tag: u8 = Decodable::decode(decoder);

        if tag == TAG_PARTIAL_SPAN {
            return Span::new(BytePos(0), BytePos(0), ctxt, parent);
        } else if tag == TAG_RELATIVE_SPAN {
            let dlo = u32::decode(decoder);
            let dto = u32::decode(decoder);

            let enclosing = decoder.tcx.source_span_untracked(parent.unwrap()).data_untracked();
            let span = Span::new(
                enclosing.lo + BytePos::from_u32(dlo),
                enclosing.lo + BytePos::from_u32(dto),
                ctxt,
                parent,
            );

            return span;
        } else {
            debug_assert_eq!(tag, TAG_FULL_SPAN);
        }

        let file_lo_index = SourceFileIndex::decode(decoder);
        let line_lo = usize::decode(decoder);
        let col_lo = BytePos::decode(decoder);
        let len = BytePos::decode(decoder);

        let file_lo = decoder.file_index_to_file(file_lo_index);
        let lo = file_lo.lines(|lines| lines[line_lo - 1] + col_lo);
        let hi = lo + len;

        Span::new(lo, hi, ctxt, parent)
    }
}

// copy&paste impl from rustc_metadata
impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for Symbol {
    #[inline]
    fn decode(d: &mut CacheDecoder<'a, 'tcx>) -> Self {
        let tag = d.read_u8();

        match tag {
            SYMBOL_STR => {
                let s = d.read_str();
                Symbol::intern(s)
            }
            SYMBOL_OFFSET => {
                // read str offset
                let pos = d.read_usize();

                // move to str offset and read
                d.opaque.with_position(pos, |d| {
                    let s = d.read_str();
                    Symbol::intern(s)
                })
            }
            SYMBOL_PREINTERNED => {
                let symbol_index = d.read_u32();
                Symbol::new_from_decoded(symbol_index)
            }
            _ => unreachable!(),
        }
    }
}

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for CrateNum {
    #[inline]
    fn decode(d: &mut CacheDecoder<'a, 'tcx>) -> Self {
        let stable_id = StableCrateId::decode(d);
        let cnum = d.tcx.stable_crate_id_to_crate_num(stable_id);
        cnum
    }
}

// This impl makes sure that we get a runtime error when we try decode a
// `DefIndex` that is not contained in a `DefId`. Such a case would be problematic
// because we would not know how to transform the `DefIndex` to the current
// context.
impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for DefIndex {
    fn decode(_d: &mut CacheDecoder<'a, 'tcx>) -> DefIndex {
        panic!("trying to decode `DefIndex` outside the context of a `DefId`")
    }
}

// Both the `CrateNum` and the `DefIndex` of a `DefId` can change in between two
// compilation sessions. We use the `DefPathHash`, which is stable across
// sessions, to map the old `DefId` to the new one.
impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for DefId {
    #[inline]
    fn decode(d: &mut CacheDecoder<'a, 'tcx>) -> Self {
        // Load the `DefPathHash` which is was we encoded the `DefId` as.
        let def_path_hash = DefPathHash::decode(d);

        // Using the `DefPathHash`, we can lookup the new `DefId`.
        // Subtle: We only encode a `DefId` as part of a query result.
        // If we get to this point, then all of the query inputs were green,
        // which means that the definition with this hash is guaranteed to
        // still exist in the current compilation session.
        d.tcx.def_path_hash_to_def_id(def_path_hash, &mut || {
            panic!("Failed to convert DefPathHash {def_path_hash:?}")
        })
    }
}

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for &'tcx UnordSet<LocalDefId> {
    #[inline]
    fn decode(d: &mut CacheDecoder<'a, 'tcx>) -> Self {
        RefDecodable::decode(d)
    }
}

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>>
    for &'tcx FxHashMap<DefId, ty::EarlyBinder<Ty<'tcx>>>
{
    #[inline]
    fn decode(d: &mut CacheDecoder<'a, 'tcx>) -> Self {
        RefDecodable::decode(d)
    }
}

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>>
    for &'tcx IndexVec<mir::Promoted, mir::Body<'tcx>>
{
    #[inline]
    fn decode(d: &mut CacheDecoder<'a, 'tcx>) -> Self {
        RefDecodable::decode(d)
    }
}

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for &'tcx [(ty::Clause<'tcx>, Span)] {
    #[inline]
    fn decode(d: &mut CacheDecoder<'a, 'tcx>) -> Self {
        RefDecodable::decode(d)
    }
}

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for &'tcx [rustc_ast::InlineAsmTemplatePiece] {
    #[inline]
    fn decode(d: &mut CacheDecoder<'a, 'tcx>) -> Self {
        RefDecodable::decode(d)
    }
}

macro_rules! impl_ref_decoder {
    (<$tcx:tt> $($ty:ty,)*) => {
        $(impl<'a, $tcx> Decodable<CacheDecoder<'a, $tcx>> for &$tcx [$ty] {
            #[inline]
            fn decode(d: &mut CacheDecoder<'a, $tcx>) -> Self {
                RefDecodable::decode(d)
            }
        })*
    };
}

impl_ref_decoder! {<'tcx>
    Span,
    rustc_ast::Attribute,
    rustc_span::symbol::Ident,
    ty::Variance,
    rustc_span::def_id::DefId,
    rustc_span::def_id::LocalDefId,
    (rustc_middle::middle::exported_symbols::ExportedSymbol<'tcx>, rustc_middle::middle::exported_symbols::SymbolExportInfo),
    ty::DeducedParamAttrs,
}

//- ENCODING -------------------------------------------------------------------

/// An encoder that can write to the incremental compilation cache.
pub struct CacheEncoder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    encoder: FileEncoder,
    type_shorthands: FxHashMap<Ty<'tcx>, usize>,
    predicate_shorthands: FxHashMap<ty::PredicateKind<'tcx>, usize>,
    interpret_allocs: FxIndexSet<interpret::AllocId>,
    source_map: CachingSourceMapView<'tcx>,
    file_to_file_index: FxHashMap<*const SourceFile, SourceFileIndex>,
    hygiene_context: &'a HygieneEncodeContext,
    symbol_table: FxHashMap<Symbol, usize>,
}

impl<'a, 'tcx> CacheEncoder<'a, 'tcx> {
    #[inline]
    fn source_file_index(&mut self, source_file: Lrc<SourceFile>) -> SourceFileIndex {
        self.file_to_file_index[&(&*source_file as *const SourceFile)]
    }

    /// Encode something with additional information that allows to do some
    /// sanity checks when decoding the data again. This method will first
    /// encode the specified tag, then the given value, then the number of
    /// bytes taken up by tag and value. On decoding, we can then verify that
    /// we get the expected tag and read the expected number of bytes.
    pub fn encode_tagged<T: Encodable<Self>, V: Encodable<Self>>(&mut self, tag: T, value: &V) {
        let start_pos = self.position();

        tag.encode(self);
        value.encode(self);

        let end_pos = self.position();
        ((end_pos - start_pos) as u64).encode(self);
    }

    #[inline]
    fn finish(self) -> Result<usize, io::Error> {
        self.encoder.finish()
    }
}

impl<'a, 'tcx> Encodable<CacheEncoder<'a, 'tcx>> for SyntaxContext {
    fn encode(&self, s: &mut CacheEncoder<'a, 'tcx>) {
        rustc_span::hygiene::raw_encode_syntax_context(*self, s.hygiene_context, s);
    }
}

impl<'a, 'tcx> Encodable<CacheEncoder<'a, 'tcx>> for ExpnId {
    fn encode(&self, s: &mut CacheEncoder<'a, 'tcx>) {
        s.hygiene_context.schedule_expn_data_for_encoding(*self);
        self.expn_hash().encode(s);
    }
}

impl<'a, 'tcx> Encodable<CacheEncoder<'a, 'tcx>> for Span {
    fn encode(&self, s: &mut CacheEncoder<'a, 'tcx>) {
        let span_data = self.data_untracked();
        span_data.ctxt.encode(s);
        span_data.parent.encode(s);

        if span_data.is_dummy() {
            return TAG_PARTIAL_SPAN.encode(s);
        }

        if let Some(parent) = span_data.parent {
            let enclosing = s.tcx.source_span(parent).data_untracked();
            if enclosing.contains(span_data) {
                TAG_RELATIVE_SPAN.encode(s);
                (span_data.lo - enclosing.lo).to_u32().encode(s);
                (span_data.hi - enclosing.lo).to_u32().encode(s);
                return;
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

        TAG_FULL_SPAN.encode(s);
        source_file_index.encode(s);
        line_lo.encode(s);
        col_lo.encode(s);
        len.encode(s);
    }
}

// copy&paste impl from rustc_metadata
impl<'a, 'tcx> Encodable<CacheEncoder<'a, 'tcx>> for Symbol {
    fn encode(&self, s: &mut CacheEncoder<'a, 'tcx>) {
        // if symbol preinterned, emit tag and symbol index
        if self.is_preinterned() {
            s.encoder.emit_u8(SYMBOL_PREINTERNED);
            s.encoder.emit_u32(self.as_u32());
        } else {
            // otherwise write it as string or as offset to it
            match s.symbol_table.entry(*self) {
                Entry::Vacant(o) => {
                    s.encoder.emit_u8(SYMBOL_STR);
                    let pos = s.encoder.position();
                    o.insert(pos);
                    s.emit_str(self.as_str());
                }
                Entry::Occupied(o) => {
                    let x = *o.get();
                    s.emit_u8(SYMBOL_OFFSET);
                    s.emit_usize(x);
                }
            }
        }
    }
}

impl<'a, 'tcx> TyEncoder for CacheEncoder<'a, 'tcx> {
    type I = TyCtxt<'tcx>;
    const CLEAR_CROSS_CRATE: bool = false;

    #[inline]
    fn position(&self) -> usize {
        self.encoder.position()
    }
    #[inline]
    fn type_shorthands(&mut self) -> &mut FxHashMap<Ty<'tcx>, usize> {
        &mut self.type_shorthands
    }
    #[inline]
    fn predicate_shorthands(&mut self) -> &mut FxHashMap<ty::PredicateKind<'tcx>, usize> {
        &mut self.predicate_shorthands
    }
    #[inline]
    fn encode_alloc_id(&mut self, alloc_id: &interpret::AllocId) {
        let (index, _) = self.interpret_allocs.insert_full(*alloc_id);

        index.encode(self);
    }
}

impl<'a, 'tcx> Encodable<CacheEncoder<'a, 'tcx>> for CrateNum {
    #[inline]
    fn encode(&self, s: &mut CacheEncoder<'a, 'tcx>) {
        s.tcx.stable_crate_id(*self).encode(s);
    }
}

impl<'a, 'tcx> Encodable<CacheEncoder<'a, 'tcx>> for DefId {
    #[inline]
    fn encode(&self, s: &mut CacheEncoder<'a, 'tcx>) {
        s.tcx.def_path_hash(*self).encode(s);
    }
}

impl<'a, 'tcx> Encodable<CacheEncoder<'a, 'tcx>> for DefIndex {
    fn encode(&self, _: &mut CacheEncoder<'a, 'tcx>) {
        bug!("encoding `DefIndex` without context");
    }
}

macro_rules! encoder_methods {
    ($($name:ident($ty:ty);)*) => {
        #[inline]
        $(fn $name(&mut self, value: $ty) {
            self.encoder.$name(value)
        })*
    }
}

impl<'a, 'tcx> Encoder for CacheEncoder<'a, 'tcx> {
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

        emit_raw_bytes(&[u8]);
    }
}

// This ensures that the `Encodable<opaque::FileEncoder>::encode` specialization for byte slices
// is used when a `CacheEncoder` having an `opaque::FileEncoder` is passed to `Encodable::encode`.
// Unfortunately, we have to manually opt into specializations this way, given how `CacheEncoder`
// and the encoding traits currently work.
impl<'a, 'tcx> Encodable<CacheEncoder<'a, 'tcx>> for [u8] {
    fn encode(&self, e: &mut CacheEncoder<'a, 'tcx>) {
        self.encode(&mut e.encoder);
    }
}
