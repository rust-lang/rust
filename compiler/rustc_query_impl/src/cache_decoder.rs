use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::sync::{HashMapExt, Lock, Lrc};
use rustc_data_structures::unhash::UnhashMap;
use rustc_data_structures::unord::UnordSet;
use rustc_hir::def_id::{CrateNum, DefId, DefIndex, LocalDefId, StableCrateId, LOCAL_CRATE};
use rustc_hir::definitions::DefPathHash;
use rustc_index::IndexVec;
use rustc_middle::dep_graph::SerializedDepNodeIndex;
use rustc_middle::mir::interpret::AllocDecodingSession;
use rustc_middle::mir::{self, interpret};
use rustc_middle::query::on_disk_cache::{
    decode_tagged, AbsoluteBytePos, EncodedSourceFileId, OnDiskCache, SourceFileIndex,
    SYMBOL_OFFSET, SYMBOL_PREINTERNED, SYMBOL_STR, TAG_EXPN_DATA, TAG_FULL_SPAN, TAG_PARTIAL_SPAN,
    TAG_RELATIVE_SPAN, TAG_SYNTAX_CONTEXT,
};
use rustc_middle::ty::codec::{RefDecodable, TyDecoder};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_query_system::query::QuerySideEffects;
use rustc_serialize::{opaque::MemDecoder, Decodable, Decoder};
use rustc_span::hygiene::{ExpnId, HygieneDecodeContext, SyntaxContext, SyntaxContextData};
use rustc_span::source_map::SourceMap;
use rustc_span::Symbol;
use rustc_span::{BytePos, ExpnData, ExpnHash, Pos, SourceFile, Span};
use std::mem;

/// Loads a `QuerySideEffects` created during the previous compilation session.
pub fn load_side_effects(
    on_disk_cache: &OnDiskCache<'_>,
    tcx: TyCtxt<'_>,
    dep_node_index: SerializedDepNodeIndex,
) -> QuerySideEffects {
    let side_effects: Option<QuerySideEffects> =
        load_indexed(on_disk_cache, tcx, dep_node_index, &on_disk_cache.prev_side_effects_index);

    side_effects.unwrap_or_default()
}

/// Returns the cached query result if there is something in the cache for
/// the given `SerializedDepNodeIndex`; otherwise returns `None`.
pub fn try_load_query_result<'tcx, T>(
    on_disk_cache: &OnDiskCache<'_>,
    tcx: TyCtxt<'tcx>,
    dep_node_index: SerializedDepNodeIndex,
) -> Option<T>
where
    T: for<'a> Decodable<CacheDecoder<'a, 'tcx>>,
{
    let opt_value =
        load_indexed(on_disk_cache, tcx, dep_node_index, &on_disk_cache.query_result_index);
    debug_assert_eq!(opt_value.is_some(), on_disk_cache.loadable_from_disk(dep_node_index));
    opt_value
}

fn load_indexed<'tcx, T>(
    on_disk_cache: &OnDiskCache<'_>,
    tcx: TyCtxt<'tcx>,
    dep_node_index: SerializedDepNodeIndex,
    index: &FxHashMap<SerializedDepNodeIndex, AbsoluteBytePos>,
) -> Option<T>
where
    T: for<'a> Decodable<CacheDecoder<'a, 'tcx>>,
{
    let pos = index.get(&dep_node_index).cloned()?;
    let value =
        with_decoder(on_disk_cache, tcx, pos, |decoder| decode_tagged(decoder, dep_node_index));
    Some(value)
}

fn with_decoder<'a, 'tcx, T, F: for<'s> FnOnce(&mut CacheDecoder<'s, 'tcx>) -> T>(
    on_disk_cache: &OnDiskCache<'_>,
    tcx: TyCtxt<'tcx>,
    pos: AbsoluteBytePos,
    f: F,
) -> T
where
    T: Decodable<CacheDecoder<'a, 'tcx>>,
{
    let serialized_data = on_disk_cache.serialized_data.read();
    let mut decoder = CacheDecoder {
        tcx,
        opaque: MemDecoder::new(serialized_data.as_deref().unwrap_or(&[]), pos.to_usize()),
        source_map: on_disk_cache.source_map,
        file_index_to_file: &on_disk_cache.file_index_to_file,
        file_index_to_stable_id: &on_disk_cache.file_index_to_stable_id,
        alloc_decoding_session: on_disk_cache.alloc_decoding_state.new_decoding_session(),
        syntax_contexts: &on_disk_cache.syntax_contexts,
        expn_data: &on_disk_cache.expn_data,
        foreign_expn_data: &on_disk_cache.foreign_expn_data,
        hygiene_context: &on_disk_cache.hygiene_context,
    };
    f(&mut decoder)
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

impl<'a, 'tcx> Decodable<CacheDecoder<'a, 'tcx>> for &'tcx [(ty::Predicate<'tcx>, Span)] {
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
