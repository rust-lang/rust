// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use dep_graph::{DepNodeIndex, SerializedDepNodeIndex};
use errors::Diagnostic;
use hir;
use hir::def_id::{CrateNum, DefIndex, DefId, LocalDefId,
                  RESERVED_FOR_INCR_COMP_CACHE, LOCAL_CRATE};
use hir::map::definitions::DefPathHash;
use middle::cstore::CrateStore;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::{IndexVec, Idx};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder, opaque,
                      SpecializedDecoder, SpecializedEncoder,
                      UseSpecializedDecodable, UseSpecializedEncodable};
use session::{CrateDisambiguator, Session};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::mem;
use syntax::ast::NodeId;
use syntax::codemap::{CodeMap, StableFilemapId};
use syntax_pos::{BytePos, Span, NO_EXPANSION, DUMMY_SP};
use ty;
use ty::codec::{self as ty_codec, TyDecoder, TyEncoder};
use ty::context::TyCtxt;

// Some magic values used for verifying that encoding and decoding. These are
// basically random numbers.
const PREV_DIAGNOSTICS_TAG: u64 = 0x1234_5678_A1A1_A1A1;
const QUERY_RESULT_INDEX_TAG: u64 = 0x1234_5678_C3C3_C3C3;

/// `OnDiskCache` provides an interface to incr. comp. data cached from the
/// previous compilation session. This data will eventually include the results
/// of a few selected queries (like `typeck_tables_of` and `mir_optimized`) and
/// any diagnostics that have been emitted during a query.
pub struct OnDiskCache<'sess> {

    // The complete cache data in serialized form.
    serialized_data: Vec<u8>,

    // The diagnostics emitted during the previous compilation session.
    prev_diagnostics: FxHashMap<SerializedDepNodeIndex, Vec<Diagnostic>>,

    // This field collects all Diagnostics emitted during the current
    // compilation session.
    current_diagnostics: RefCell<FxHashMap<DepNodeIndex, Vec<Diagnostic>>>,

    prev_cnums: Vec<(u32, String, CrateDisambiguator)>,
    cnum_map: RefCell<Option<IndexVec<CrateNum, Option<CrateNum>>>>,

    prev_filemap_starts: BTreeMap<BytePos, StableFilemapId>,
    codemap: &'sess CodeMap,

    // A map from dep-node to the position of the cached query result in
    // `serialized_data`.
    query_result_index: FxHashMap<SerializedDepNodeIndex, usize>,
}

// This type is used only for (de-)serialization.
#[derive(RustcEncodable, RustcDecodable)]
struct Header {
    prev_filemap_starts: BTreeMap<BytePos, StableFilemapId>,
    prev_cnums: Vec<(u32, String, CrateDisambiguator)>,
}

type EncodedPrevDiagnostics = Vec<(SerializedDepNodeIndex, Vec<Diagnostic>)>;
type EncodedQueryResultIndex = Vec<(SerializedDepNodeIndex, usize)>;

impl<'sess> OnDiskCache<'sess> {
    /// Create a new OnDiskCache instance from the serialized data in `data`.
    pub fn new(sess: &'sess Session, data: Vec<u8>, start_pos: usize) -> OnDiskCache<'sess> {
        debug_assert!(sess.opts.incremental.is_some());

        // Decode the header
        let (header, post_header_pos) = {
            let mut decoder = opaque::Decoder::new(&data[..], start_pos);
            let header = Header::decode(&mut decoder)
                .expect("Error while trying to decode incr. comp. cache header.");
            (header, decoder.position())
        };

        let (prev_diagnostics, query_result_index) = {
            let mut decoder = CacheDecoder {
                tcx: None,
                opaque: opaque::Decoder::new(&data[..], post_header_pos),
                codemap: sess.codemap(),
                prev_filemap_starts: &header.prev_filemap_starts,
                cnum_map: &IndexVec::new(),
            };

            // Decode Diagnostics
            let prev_diagnostics: FxHashMap<_, _> = {
                let diagnostics: EncodedPrevDiagnostics =
                    decode_tagged(&mut decoder, PREV_DIAGNOSTICS_TAG)
                        .expect("Error while trying to decode previous session \
                                 diagnostics from incr. comp. cache.");
                diagnostics.into_iter().collect()
            };

            // Decode the *position* of the query result index
            let query_result_index_pos = {
                let pos_pos = data.len() - IntEncodedWithFixedSize::ENCODED_SIZE;
                decoder.with_position(pos_pos, |decoder| {
                    IntEncodedWithFixedSize::decode(decoder)
                }).expect("Error while trying to decode query result index position.")
                .0 as usize
            };

            // Decode the query result index itself
            let query_result_index: EncodedQueryResultIndex =
                decoder.with_position(query_result_index_pos, |decoder| {
                    decode_tagged(decoder, QUERY_RESULT_INDEX_TAG)
                }).expect("Error while trying to decode query result index.");

            (prev_diagnostics, query_result_index)
        };

        OnDiskCache {
            serialized_data: data,
            prev_diagnostics,
            prev_filemap_starts: header.prev_filemap_starts,
            prev_cnums: header.prev_cnums,
            cnum_map: RefCell::new(None),
            codemap: sess.codemap(),
            current_diagnostics: RefCell::new(FxHashMap()),
            query_result_index: query_result_index.into_iter().collect(),
        }
    }

    pub fn new_empty(codemap: &'sess CodeMap) -> OnDiskCache<'sess> {
        OnDiskCache {
            serialized_data: Vec::new(),
            prev_diagnostics: FxHashMap(),
            prev_filemap_starts: BTreeMap::new(),
            prev_cnums: vec![],
            cnum_map: RefCell::new(None),
            codemap,
            current_diagnostics: RefCell::new(FxHashMap()),
            query_result_index: FxHashMap(),
        }
    }

    pub fn serialize<'a, 'tcx, E>(&self,
                                  tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                  cstore: &CrateStore,
                                  encoder: &mut E)
                                  -> Result<(), E::Error>
        where E: ty_codec::TyEncoder
     {
        // Serializing the DepGraph should not modify it:
        let _in_ignore = tcx.dep_graph.in_ignore();

        let mut encoder = CacheEncoder {
            tcx,
            encoder,
            type_shorthands: FxHashMap(),
            predicate_shorthands: FxHashMap(),
        };


        // Encode the file header
        let prev_filemap_starts: BTreeMap<_, _> = self
            .codemap
            .files()
            .iter()
            .map(|fm| (fm.start_pos, StableFilemapId::new(fm)))
            .collect();

        let sorted_cnums = sorted_cnums_including_local_crate(cstore);

        let prev_cnums: Vec<_> = sorted_cnums.iter().map(|&cnum| {
            let crate_name = tcx.original_crate_name(cnum).as_str().to_string();
            let crate_disambiguator = tcx.crate_disambiguator(cnum);
            (cnum.as_u32(), crate_name, crate_disambiguator)
        }).collect();

        Header {
            prev_filemap_starts,
            prev_cnums,
        }.encode(&mut encoder)?;


        // Encode Diagnostics
        let diagnostics: EncodedPrevDiagnostics =
            self.current_diagnostics
                .borrow()
                .iter()
                .map(|(k, v)| (SerializedDepNodeIndex::new(k.index()), v.clone()))
                .collect();

        encoder.encode_tagged(PREV_DIAGNOSTICS_TAG, &diagnostics)?;


        // Encode query results
        let mut query_result_index = EncodedQueryResultIndex::new();

        {
            use ty::maps::queries::*;
            let enc = &mut encoder;
            let qri = &mut query_result_index;

            // Encode TypeckTables
            encode_query_results::<typeck_tables_of, _>(tcx, enc, qri)?;
        }

        // Encode query result index
        let query_result_index_pos = encoder.position() as u64;
        encoder.encode_tagged(QUERY_RESULT_INDEX_TAG, &query_result_index)?;

        // Encode the position of the query result index as the last 8 bytes of
        // file so we know where to look for it.
        IntEncodedWithFixedSize(query_result_index_pos).encode(&mut encoder)?;

        return Ok(());

        fn sorted_cnums_including_local_crate(cstore: &CrateStore) -> Vec<CrateNum> {
            let mut cnums = vec![LOCAL_CRATE];
            cnums.extend_from_slice(&cstore.crates_untracked()[..]);
            cnums.sort_unstable();
            // Just to be sure...
            cnums.dedup();
            cnums
        }
    }

    /// Load a diagnostic emitted during the previous compilation session.
    pub fn load_diagnostics(&self,
                            dep_node_index: SerializedDepNodeIndex)
                            -> Vec<Diagnostic> {
        self.prev_diagnostics.get(&dep_node_index).cloned().unwrap_or(vec![])
    }

    /// Store a diagnostic emitted during the current compilation session.
    /// Anything stored like this will be available via `load_diagnostics` in
    /// the next compilation session.
    pub fn store_diagnostics(&self,
                             dep_node_index: DepNodeIndex,
                             diagnostics: Vec<Diagnostic>) {
        let mut current_diagnostics = self.current_diagnostics.borrow_mut();
        let prev = current_diagnostics.insert(dep_node_index, diagnostics);
        debug_assert!(prev.is_none());
    }

    pub fn load_query_result<'a, 'tcx, T>(&self,
                                          tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                          dep_node_index: SerializedDepNodeIndex)
                                          -> T
        where T: Decodable
    {
        let pos = self.query_result_index[&dep_node_index];

        let mut cnum_map = self.cnum_map.borrow_mut();
        if cnum_map.is_none() {
            *cnum_map = Some(Self::compute_cnum_map(tcx, &self.prev_cnums[..]));
        }

        let mut decoder = CacheDecoder {
            tcx: Some(tcx),
            opaque: opaque::Decoder::new(&self.serialized_data[..], pos),
            codemap: self.codemap,
            prev_filemap_starts: &self.prev_filemap_starts,
            cnum_map: cnum_map.as_ref().unwrap(),
        };

        match decode_tagged(&mut decoder, dep_node_index) {
            Ok(value) => {
                value
            }
            Err(e) => {
                bug!("Could not decode cached query result: {}", e)
            }
        }
    }

    /// Store a diagnostic emitted during computation of an anonymous query.
    /// Since many anonymous queries can share the same `DepNode`, we aggregate
    /// them -- as opposed to regular queries where we assume that there is a
    /// 1:1 relationship between query-key and `DepNode`.
    pub fn store_diagnostics_for_anon_node(&self,
                                           dep_node_index: DepNodeIndex,
                                           mut diagnostics: Vec<Diagnostic>) {
        let mut current_diagnostics = self.current_diagnostics.borrow_mut();

        let x = current_diagnostics.entry(dep_node_index).or_insert_with(|| {
            mem::replace(&mut diagnostics, Vec::new())
        });

        x.extend(diagnostics.into_iter());
    }

    // This function builds mapping from previous-session-CrateNum to
    // current-session-CrateNum. There might be CrateNums from the previous
    // Session that don't occur in the current one. For these, the mapping
    // maps to None.
    fn compute_cnum_map(tcx: TyCtxt,
                        prev_cnums: &[(u32, String, CrateDisambiguator)])
                        -> IndexVec<CrateNum, Option<CrateNum>>
    {
        let _in_ignore = tcx.dep_graph.in_ignore();

        let current_cnums = tcx.all_crate_nums(LOCAL_CRATE).iter().map(|&cnum| {
            let crate_name = tcx.original_crate_name(cnum)
                                .as_str()
                                .to_string();
            let crate_disambiguator = tcx.crate_disambiguator(cnum);
            ((crate_name, crate_disambiguator), cnum)
        }).collect::<FxHashMap<_,_>>();

        let map_size = prev_cnums.iter()
                                 .map(|&(cnum, ..)| cnum)
                                 .max()
                                 .unwrap_or(0) + 1;
        let mut map = IndexVec::new();
        map.resize(map_size as usize, None);

        for &(prev_cnum, ref crate_name, crate_disambiguator) in prev_cnums {
            let key = (crate_name.clone(), crate_disambiguator);
            map[CrateNum::from_u32(prev_cnum)] = current_cnums.get(&key).cloned();
        }

        map[LOCAL_CRATE] = Some(LOCAL_CRATE);
        map
    }
}


//- DECODING -------------------------------------------------------------------

/// A decoder that can read the incr. comp. cache. It is similar to the one
/// we use for crate metadata decoding in that it can rebase spans and
/// eventually will also handle things that contain `Ty` instances.
struct CacheDecoder<'a, 'tcx: 'a, 'x> {
    tcx: Option<TyCtxt<'a, 'tcx, 'tcx>>,
    opaque: opaque::Decoder<'x>,
    codemap: &'x CodeMap,
    prev_filemap_starts: &'x BTreeMap<BytePos, StableFilemapId>,
    cnum_map: &'x IndexVec<CrateNum, Option<CrateNum>>,
}

impl<'a, 'tcx, 'x> CacheDecoder<'a, 'tcx, 'x> {
    fn find_filemap_prev_bytepos(&self,
                                 prev_bytepos: BytePos)
                                 -> Option<(BytePos, StableFilemapId)> {
        for (start, id) in self.prev_filemap_starts.range(BytePos(0) ..= prev_bytepos).rev() {
            return Some((*start, *id))
        }

        None
    }
}

// Decode something that was encoded with encode_tagged() and verify that the
// tag matches and the correct amount of bytes was read.
fn decode_tagged<'a, 'tcx, D, T, V>(decoder: &mut D,
                                    expected_tag: T)
                                    -> Result<V, D::Error>
    where T: Decodable + Eq + ::std::fmt::Debug,
          V: Decodable,
          D: Decoder + ty_codec::TyDecoder<'a, 'tcx>,
          'tcx: 'a,
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


impl<'a, 'tcx: 'a, 'x> ty_codec::TyDecoder<'a, 'tcx> for CacheDecoder<'a, 'tcx, 'x> {

    #[inline]
    fn tcx(&self) -> TyCtxt<'a, 'tcx, 'tcx> {
        self.tcx.expect("missing TyCtxt in CacheDecoder")
    }

    #[inline]
    fn position(&self) -> usize {
        self.opaque.position()
    }

    #[inline]
    fn peek_byte(&self) -> u8 {
        self.opaque.data[self.opaque.position()]
    }

    fn cached_ty_for_shorthand<F>(&mut self,
                                  shorthand: usize,
                                  or_insert_with: F)
                                  -> Result<ty::Ty<'tcx>, Self::Error>
        where F: FnOnce(&mut Self) -> Result<ty::Ty<'tcx>, Self::Error>
    {
        let tcx = self.tcx();

        let cache_key = ty::CReaderCacheKey {
            cnum: RESERVED_FOR_INCR_COMP_CACHE,
            pos: shorthand,
        };

        if let Some(&ty) = tcx.rcache.borrow().get(&cache_key) {
            return Ok(ty);
        }

        let ty = or_insert_with(self)?;
        tcx.rcache.borrow_mut().insert(cache_key, ty);
        Ok(ty)
    }

    fn with_position<F, R>(&mut self, pos: usize, f: F) -> R
        where F: FnOnce(&mut Self) -> R
    {
        debug_assert!(pos < self.opaque.data.len());

        let new_opaque = opaque::Decoder::new(self.opaque.data, pos);
        let old_opaque = mem::replace(&mut self.opaque, new_opaque);
        let r = f(self);
        self.opaque = old_opaque;
        r
    }

    fn map_encoded_cnum_to_current(&self, cnum: CrateNum) -> CrateNum {
        self.cnum_map[cnum].unwrap_or_else(|| {
            bug!("Could not find new CrateNum for {:?}", cnum)
        })
    }
}

implement_ty_decoder!( CacheDecoder<'a, 'tcx, 'x> );

impl<'a, 'tcx, 'x> SpecializedDecoder<Span> for CacheDecoder<'a, 'tcx, 'x> {
    fn specialized_decode(&mut self) -> Result<Span, Self::Error> {
        let lo = BytePos::decode(self)?;
        let hi = BytePos::decode(self)?;

        if let Some((prev_filemap_start, filemap_id)) = self.find_filemap_prev_bytepos(lo) {
            if let Some(current_filemap) = self.codemap.filemap_by_stable_id(filemap_id) {
                let lo = (lo + current_filemap.start_pos) - prev_filemap_start;
                let hi = (hi + current_filemap.start_pos) - prev_filemap_start;
                return Ok(Span::new(lo, hi, NO_EXPANSION));
            }
        }

        Ok(DUMMY_SP)
    }
}

// This impl makes sure that we get a runtime error when we try decode a
// DefIndex that is not contained in a DefId. Such a case would be problematic
// because we would not know how to transform the DefIndex to the current
// context.
impl<'a, 'tcx, 'x> SpecializedDecoder<DefIndex> for CacheDecoder<'a, 'tcx, 'x> {
    fn specialized_decode(&mut self) -> Result<DefIndex, Self::Error> {
        bug!("Trying to decode DefIndex outside the context of a DefId")
    }
}

// Both the CrateNum and the DefIndex of a DefId can change in between two
// compilation sessions. We use the DefPathHash, which is stable across
// sessions, to map the old DefId to the new one.
impl<'a, 'tcx, 'x> SpecializedDecoder<DefId> for CacheDecoder<'a, 'tcx, 'x> {
    fn specialized_decode(&mut self) -> Result<DefId, Self::Error> {
        // Load the DefPathHash which is was we encoded the DefId as.
        let def_path_hash = DefPathHash::decode(self)?;

        // Using the DefPathHash, we can lookup the new DefId
        Ok(self.tcx().def_path_hash_to_def_id.as_ref().unwrap()[&def_path_hash])
    }
}

impl<'a, 'tcx, 'x> SpecializedDecoder<LocalDefId> for CacheDecoder<'a, 'tcx, 'x> {
    fn specialized_decode(&mut self) -> Result<LocalDefId, Self::Error> {
        Ok(LocalDefId::from_def_id(DefId::decode(self)?))
    }
}

impl<'a, 'tcx, 'x> SpecializedDecoder<hir::HirId> for CacheDecoder<'a, 'tcx, 'x> {
    fn specialized_decode(&mut self) -> Result<hir::HirId, Self::Error> {
        // Load the DefPathHash which is was we encoded the DefIndex as.
        let def_path_hash = DefPathHash::decode(self)?;

        // Use the DefPathHash to map to the current DefId.
        let def_id = self.tcx()
                         .def_path_hash_to_def_id
                         .as_ref()
                         .unwrap()[&def_path_hash];

        debug_assert!(def_id.is_local());

        // The ItemLocalId needs no remapping.
        let local_id = hir::ItemLocalId::decode(self)?;

        // Reconstruct the HirId and look up the corresponding NodeId in the
        // context of the current session.
        Ok(hir::HirId {
            owner: def_id.index,
            local_id
        })
    }
}

// NodeIds are not stable across compilation sessions, so we store them in their
// HirId representation. This allows use to map them to the current NodeId.
impl<'a, 'tcx, 'x> SpecializedDecoder<NodeId> for CacheDecoder<'a, 'tcx, 'x> {
    fn specialized_decode(&mut self) -> Result<NodeId, Self::Error> {
        let hir_id = hir::HirId::decode(self)?;
        Ok(self.tcx().hir.hir_to_node_id(hir_id))
    }
}

//- ENCODING -------------------------------------------------------------------

struct CacheEncoder<'enc, 'a, 'tcx, E>
    where E: 'enc + ty_codec::TyEncoder,
          'tcx: 'a,
{
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    encoder: &'enc mut E,
    type_shorthands: FxHashMap<ty::Ty<'tcx>, usize>,
    predicate_shorthands: FxHashMap<ty::Predicate<'tcx>, usize>,
}

impl<'enc, 'a, 'tcx, E> CacheEncoder<'enc, 'a, 'tcx, E>
    where E: 'enc + ty_codec::TyEncoder
{
    /// Encode something with additional information that allows to do some
    /// sanity checks when decoding the data again. This method will first
    /// encode the specified tag, then the given value, then the number of
    /// bytes taken up by tag and value. On decoding, we can then verify that
    /// we get the expected tag and read the expected number of bytes.
    fn encode_tagged<T: Encodable, V: Encodable>(&mut self,
                                                 tag: T,
                                                 value: &V)
                                                 -> Result<(), E::Error>
    {
        use ty::codec::TyEncoder;
        let start_pos = self.position();

        tag.encode(self)?;
        value.encode(self)?;

        let end_pos = self.position();
        ((end_pos - start_pos) as u64).encode(self)
    }
}

impl<'enc, 'a, 'tcx, E> ty_codec::TyEncoder for CacheEncoder<'enc, 'a, 'tcx, E>
    where E: 'enc + ty_codec::TyEncoder
{
    #[inline]
    fn position(&self) -> usize {
        self.encoder.position()
    }
}

impl<'enc, 'a, 'tcx, E> SpecializedEncoder<CrateNum> for CacheEncoder<'enc, 'a, 'tcx, E>
    where E: 'enc + ty_codec::TyEncoder
{
    #[inline]
    fn specialized_encode(&mut self, cnum: &CrateNum) -> Result<(), Self::Error> {
        self.emit_u32(cnum.as_u32())
    }
}

impl<'enc, 'a, 'tcx, E> SpecializedEncoder<ty::Ty<'tcx>> for CacheEncoder<'enc, 'a, 'tcx, E>
    where E: 'enc + ty_codec::TyEncoder
{
    #[inline]
    fn specialized_encode(&mut self, ty: &ty::Ty<'tcx>) -> Result<(), Self::Error> {
        ty_codec::encode_with_shorthand(self, ty,
            |encoder| &mut encoder.type_shorthands)
    }
}

impl<'enc, 'a, 'tcx, E> SpecializedEncoder<ty::GenericPredicates<'tcx>>
    for CacheEncoder<'enc, 'a, 'tcx, E>
    where E: 'enc + ty_codec::TyEncoder
{
    #[inline]
    fn specialized_encode(&mut self,
                          predicates: &ty::GenericPredicates<'tcx>)
                          -> Result<(), Self::Error> {
        ty_codec::encode_predicates(self, predicates,
            |encoder| &mut encoder.predicate_shorthands)
    }
}

impl<'enc, 'a, 'tcx, E> SpecializedEncoder<hir::HirId> for CacheEncoder<'enc, 'a, 'tcx, E>
    where E: 'enc + ty_codec::TyEncoder
{
    #[inline]
    fn specialized_encode(&mut self, id: &hir::HirId) -> Result<(), Self::Error> {
        let hir::HirId {
            owner,
            local_id,
        } = *id;

        let def_path_hash = self.tcx.hir.definitions().def_path_hash(owner);

        def_path_hash.encode(self)?;
        local_id.encode(self)
    }
}


impl<'enc, 'a, 'tcx, E> SpecializedEncoder<DefId> for CacheEncoder<'enc, 'a, 'tcx, E>
    where E: 'enc + ty_codec::TyEncoder
{
    #[inline]
    fn specialized_encode(&mut self, id: &DefId) -> Result<(), Self::Error> {
        let def_path_hash = self.tcx.def_path_hash(*id);
        def_path_hash.encode(self)
    }
}

impl<'enc, 'a, 'tcx, E> SpecializedEncoder<LocalDefId> for CacheEncoder<'enc, 'a, 'tcx, E>
    where E: 'enc + ty_codec::TyEncoder
{
    #[inline]
    fn specialized_encode(&mut self, id: &LocalDefId) -> Result<(), Self::Error> {
        id.to_def_id().encode(self)
    }
}

impl<'enc, 'a, 'tcx, E> SpecializedEncoder<DefIndex> for CacheEncoder<'enc, 'a, 'tcx, E>
    where E: 'enc + ty_codec::TyEncoder
{
    fn specialized_encode(&mut self, _: &DefIndex) -> Result<(), Self::Error> {
        bug!("Encoding DefIndex without context.")
    }
}

// NodeIds are not stable across compilation sessions, so we store them in their
// HirId representation. This allows use to map them to the current NodeId.
impl<'enc, 'a, 'tcx, E> SpecializedEncoder<NodeId> for CacheEncoder<'enc, 'a, 'tcx, E>
    where E: 'enc + ty_codec::TyEncoder
{
    #[inline]
    fn specialized_encode(&mut self, node_id: &NodeId) -> Result<(), Self::Error> {
        let hir_id = self.tcx.hir.node_to_hir_id(*node_id);
        hir_id.encode(self)
    }
}

macro_rules! encoder_methods {
    ($($name:ident($ty:ty);)*) => {
        $(fn $name(&mut self, value: $ty) -> Result<(), Self::Error> {
            self.encoder.$name(value)
        })*
    }
}

impl<'enc, 'a, 'tcx, E> Encoder for CacheEncoder<'enc, 'a, 'tcx, E>
    where E: 'enc + ty_codec::TyEncoder
{
    type Error = E::Error;

    fn emit_nil(&mut self) -> Result<(), Self::Error> {
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

impl<'enc, 'a, 'tcx, E> SpecializedEncoder<IntEncodedWithFixedSize>
for CacheEncoder<'enc, 'a, 'tcx, E>
    where E: 'enc + ty_codec::TyEncoder
{
    fn specialized_encode(&mut self, x: &IntEncodedWithFixedSize) -> Result<(), Self::Error> {
        let start_pos = self.position();
        for i in 0 .. IntEncodedWithFixedSize::ENCODED_SIZE {
            ((x.0 >> i * 8) as u8).encode(self)?;
        }
        let end_pos = self.position();
        assert_eq!((end_pos - start_pos), IntEncodedWithFixedSize::ENCODED_SIZE);
        Ok(())
    }
}

impl<'a, 'tcx, 'x> SpecializedDecoder<IntEncodedWithFixedSize>
for CacheDecoder<'a, 'tcx, 'x> {
    fn specialized_decode(&mut self) -> Result<IntEncodedWithFixedSize, Self::Error> {
        let mut value: u64 = 0;
        let start_pos = self.position();

        for i in 0 .. IntEncodedWithFixedSize::ENCODED_SIZE {
            let byte: u8 = Decodable::decode(self)?;
            value |= (byte as u64) << (i * 8);
        }

        let end_pos = self.position();
        assert_eq!((end_pos - start_pos), IntEncodedWithFixedSize::ENCODED_SIZE);

        Ok(IntEncodedWithFixedSize(value))
    }
}

fn encode_query_results<'enc, 'a, 'tcx, Q, E>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                              encoder: &mut CacheEncoder<'enc, 'a, 'tcx, E>,
                                              query_result_index: &mut EncodedQueryResultIndex)
                                              -> Result<(), E::Error>
    where Q: super::plumbing::GetCacheInternal<'tcx>,
          E: 'enc + TyEncoder,
          Q::Value: Encodable,
{
    for (key, entry) in Q::get_cache_internal(tcx).map.iter() {
        if Q::cache_on_disk(key.clone()) {
            let dep_node = SerializedDepNodeIndex::new(entry.index.index());

            // Record position of the cache entry
            query_result_index.push((dep_node, encoder.position()));

            // Encode the type check tables with the SerializedDepNodeIndex
            // as tag.
            encoder.encode_tagged(dep_node, &entry.value)?;
        }
    }

    Ok(())
}
