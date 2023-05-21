use rustc_data_structures::fx::{FxHashMap, FxIndexSet};
use rustc_data_structures::memmap::Mmap;
use rustc_data_structures::stable_hasher::Hash64;
use rustc_data_structures::sync::{Lock, Lrc, RwLock};
use rustc_data_structures::unhash::UnhashMap;
use rustc_hir::def_id::{CrateNum, DefId, DefIndex, StableCrateId, LOCAL_CRATE};
use rustc_index::Idx;
use rustc_middle::dep_graph::{DepNodeIndex, SerializedDepNodeIndex};
use rustc_middle::mir::interpret;
use rustc_middle::mir::interpret::AllocDecodingState;
use rustc_middle::ty::codec::TyEncoder;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_query_system::query::QuerySideEffects;
use rustc_serialize::{
    opaque::{FileEncodeResult, FileEncoder, IntEncodedWithFixedSize, MemDecoder},
    Decodable, Decoder, Encodable, Encoder,
};
use rustc_session::Session;
use rustc_span::hygiene::{ExpnId, HygieneDecodeContext, HygieneEncodeContext, SyntaxContext};
use rustc_span::source_map::{SourceMap, StableSourceFileId};
use rustc_span::{CachingSourceMapView, Symbol};
use rustc_span::{ExpnHash, Pos, SourceFile, Span};
use std::collections::hash_map::Entry;
use std::io;

const TAG_FILE_FOOTER: u128 = 0xC0FFEE_C0FFEE_C0FFEE_C0FFEE_C0FFEE;

// A normal span encoded with both location information and a `SyntaxContext`
pub const TAG_FULL_SPAN: u8 = 0;
// A partial span with no location information, encoded only with a `SyntaxContext`
pub const TAG_PARTIAL_SPAN: u8 = 1;
pub const TAG_RELATIVE_SPAN: u8 = 2;

pub const TAG_SYNTAX_CONTEXT: u8 = 0;
pub const TAG_EXPN_DATA: u8 = 1;

// Tags for encoding Symbol's
pub const SYMBOL_STR: u8 = 0;
pub const SYMBOL_OFFSET: u8 = 1;
pub const SYMBOL_PREINTERNED: u8 = 2;

/// Provides an interface to incremental compilation data cached from the
/// previous compilation session. This data will eventually include the results
/// of a few selected queries (like `typeck` and `mir_optimized`) and
/// any side effects that have been emitted during a query.
pub struct OnDiskCache<'sess> {
    // The complete cache data in serialized form.
    pub serialized_data: RwLock<Option<Mmap>>,

    // Collects all `QuerySideEffects` created during the current compilation
    // session.
    current_side_effects: Lock<FxHashMap<DepNodeIndex, QuerySideEffects>>,

    pub source_map: &'sess SourceMap,
    pub file_index_to_stable_id: FxHashMap<SourceFileIndex, EncodedSourceFileId>,

    // Caches that are populated lazily during decoding.
    pub file_index_to_file: Lock<FxHashMap<SourceFileIndex, Lrc<SourceFile>>>,

    // A map from dep-node to the position of the cached query result in
    // `serialized_data`.
    pub query_result_index: FxHashMap<SerializedDepNodeIndex, AbsoluteBytePos>,

    // A map from dep-node to the position of any associated `QuerySideEffects` in
    // `serialized_data`.
    pub prev_side_effects_index: FxHashMap<SerializedDepNodeIndex, AbsoluteBytePos>,

    pub alloc_decoding_state: AllocDecodingState,

    // A map from syntax context ids to the position of their associated
    // `SyntaxContextData`. We use a `u32` instead of a `SyntaxContext`
    // to represent the fact that we are storing *encoded* ids. When we decode
    // a `SyntaxContext`, a new id will be allocated from the global `HygieneData`,
    // which will almost certainly be different than the serialized id.
    pub syntax_contexts: FxHashMap<u32, AbsoluteBytePos>,
    // A map from the `DefPathHash` of an `ExpnId` to the position
    // of their associated `ExpnData`. Ideally, we would store a `DefId`,
    // but we need to decode this before we've constructed a `TyCtxt` (which
    // makes it difficult to decode a `DefId`).

    // Note that these `DefPathHashes` correspond to both local and foreign
    // `ExpnData` (e.g `ExpnData.krate` may not be `LOCAL_CRATE`). Alternatively,
    // we could look up the `ExpnData` from the metadata of foreign crates,
    // but it seemed easier to have `OnDiskCache` be independent of the `CStore`.
    pub expn_data: UnhashMap<ExpnHash, AbsoluteBytePos>,
    // Additional information used when decoding hygiene data.
    pub hygiene_context: HygieneDecodeContext,
    // Maps `ExpnHash`es to their raw value from the *previous*
    // compilation session. This is used as an initial 'guess' when
    // we try to map an `ExpnHash` to its value in the current
    // compilation session.
    pub foreign_expn_data: UnhashMap<ExpnHash, u32>,
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
    foreign_expn_data: UnhashMap<ExpnHash, u32>,
}

pub type EncodedDepNodeIndex = Vec<(SerializedDepNodeIndex, AbsoluteBytePos)>;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Encodable, Decodable)]
pub struct SourceFileIndex(u32);

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Encodable, Decodable)]
pub struct AbsoluteBytePos(u64);

impl AbsoluteBytePos {
    #[inline]
    pub fn new(pos: usize) -> AbsoluteBytePos {
        AbsoluteBytePos(pos.try_into().expect("Incremental cache file size overflowed u64."))
    }

    #[inline]
    pub fn to_usize(self) -> usize {
        self.0 as usize
    }
}

/// An `EncodedSourceFileId` is the same as a `StableSourceFileId` except that
/// the source crate is represented as a [StableCrateId] instead of as a
/// `CrateNum`. This way `EncodedSourceFileId` can be encoded and decoded
/// without any additional context, i.e. with a simple `opaque::Decoder` (which
/// is the only thing available when decoding the cache's [Footer].
#[derive(Encodable, Decodable, Clone, Debug)]
pub struct EncodedSourceFileId {
    file_name_hash: Hash64,
    stable_crate_id: StableCrateId,
}

impl EncodedSourceFileId {
    #[inline]
    pub fn translate(&self, tcx: TyCtxt<'_>) -> StableSourceFileId {
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
                        let pos: u32 = encoder.position().try_into().unwrap();
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
}

// Decodes something that was encoded with `encode_tagged()` and verify that the
// tag matches and the correct amount of bytes was read.
pub fn decode_tagged<D, T, V>(decoder: &mut D, expected_tag: T) -> V
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
