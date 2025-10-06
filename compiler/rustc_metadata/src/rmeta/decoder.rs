// Decoding metadata from a single crate's metadata

use std::iter::TrustedLen;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::{io, mem};

pub(super) use cstore_impl::provide;
use rustc_ast as ast;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::owned_slice::OwnedSlice;
use rustc_data_structures::sync::Lock;
use rustc_data_structures::unhash::UnhashMap;
use rustc_expand::base::{SyntaxExtension, SyntaxExtensionKind};
use rustc_expand::proc_macro::{AttrProcMacro, BangProcMacro, DeriveProcMacro};
use rustc_hir::Safety;
use rustc_hir::def::Res;
use rustc_hir::def_id::{CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc_hir::definitions::{DefPath, DefPathData};
use rustc_hir::diagnostic_items::DiagnosticItems;
use rustc_index::Idx;
use rustc_middle::middle::lib_features::LibFeatures;
use rustc_middle::mir::interpret::{AllocDecodingSession, AllocDecodingState};
use rustc_middle::ty::Visibility;
use rustc_middle::ty::codec::TyDecoder;
use rustc_middle::{bug, implement_ty_decoder};
use rustc_proc_macro::bridge::client::ProcMacro;
use rustc_serialize::opaque::MemDecoder;
use rustc_serialize::{Decodable, Decoder};
use rustc_session::Session;
use rustc_session::config::TargetModifier;
use rustc_session::cstore::{CrateSource, ExternCrate};
use rustc_span::hygiene::HygieneDecodeContext;
use rustc_span::{
    BytePos, ByteSymbol, DUMMY_SP, Pos, SpanData, SpanDecoder, Symbol, SyntaxContext, kw,
};
use tracing::debug;

use crate::creader::CStore;
use crate::rmeta::table::IsDefault;
use crate::rmeta::*;

mod cstore_impl;

/// A reference to the raw binary version of crate metadata.
/// This struct applies [`MemDecoder`]'s validation when constructed
/// so that later constructions are guaranteed to succeed.
pub(crate) struct MetadataBlob(OwnedSlice);

impl std::ops::Deref for MetadataBlob {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        &self.0[..]
    }
}

impl MetadataBlob {
    /// Runs the [`MemDecoder`] validation and if it passes, constructs a new [`MetadataBlob`].
    pub(crate) fn new(slice: OwnedSlice) -> Result<Self, ()> {
        if MemDecoder::new(&slice, 0).is_ok() { Ok(Self(slice)) } else { Err(()) }
    }

    /// Since this has passed the validation of [`MetadataBlob::new`], this returns bytes which are
    /// known to pass the [`MemDecoder`] validation.
    pub(crate) fn bytes(&self) -> &OwnedSlice {
        &self.0
    }
}

/// A map from external crate numbers (as decoded from some crate file) to
/// local crate numbers (as generated during this session). Each external
/// crate may refer to types in other external crates, and each has their
/// own crate numbers.
pub(crate) type CrateNumMap = IndexVec<CrateNum, CrateNum>;

/// Target modifiers - abi or exploit mitigations flags
pub(crate) type TargetModifiers = Vec<TargetModifier>;

pub(crate) struct CrateMetadata {
    /// The primary crate data - binary metadata blob.
    blob: MetadataBlob,

    // --- Some data pre-decoded from the metadata blob, usually for performance ---
    /// Data about the top-level items in a crate, as well as various crate-level metadata.
    root: CrateRoot,
    /// Trait impl data.
    /// FIXME: Used only from queries and can use query cache,
    /// so pre-decoding can probably be avoided.
    trait_impls: FxIndexMap<(u32, DefIndex), LazyArray<(DefIndex, Option<SimplifiedType>)>>,
    /// Inherent impls which do not follow the normal coherence rules.
    ///
    /// These can be introduced using either `#![rustc_coherence_is_core]`
    /// or `#[rustc_allow_incoherent_impl]`.
    incoherent_impls: FxIndexMap<SimplifiedType, LazyArray<DefIndex>>,
    /// Proc macro descriptions for this crate, if it's a proc macro crate.
    raw_proc_macros: Option<&'static [ProcMacro]>,
    /// Source maps for code from the crate.
    source_map_import_info: Lock<Vec<Option<ImportedSourceFile>>>,
    /// For every definition in this crate, maps its `DefPathHash` to its `DefIndex`.
    def_path_hash_map: DefPathHashMapRef<'static>,
    /// Likewise for ExpnHash.
    expn_hash_map: OnceLock<UnhashMap<ExpnHash, ExpnIndex>>,
    /// Used for decoding interpret::AllocIds in a cached & thread-safe manner.
    alloc_decoding_state: AllocDecodingState,
    /// Caches decoded `DefKey`s.
    def_key_cache: Lock<FxHashMap<DefIndex, DefKey>>,

    // --- Other significant crate properties ---
    /// ID of this crate, from the current compilation session's point of view.
    cnum: CrateNum,
    /// Maps crate IDs as they are were seen from this crate's compilation sessions into
    /// IDs as they are seen from the current compilation session.
    cnum_map: CrateNumMap,
    /// Same ID set as `cnum_map` plus maybe some injected crates like panic runtime.
    dependencies: Vec<CrateNum>,
    /// How to link (or not link) this crate to the currently compiled crate.
    dep_kind: CrateDepKind,
    /// Filesystem location of this crate.
    source: Arc<CrateSource>,
    /// Whether or not this crate should be consider a private dependency.
    /// Used by the 'exported_private_dependencies' lint, and for determining
    /// whether to emit suggestions that reference this crate.
    private_dep: bool,
    /// The hash for the host proc macro. Used to support `-Z dual-proc-macro`.
    host_hash: Option<Svh>,
    /// The crate was used non-speculatively.
    used: bool,

    /// Additional data used for decoding `HygieneData` (e.g. `SyntaxContext`
    /// and `ExpnId`).
    /// Note that we store a `HygieneDecodeContext` for each `CrateMetadata`. This is
    /// because `SyntaxContext` ids are not globally unique, so we need
    /// to track which ids we've decoded on a per-crate basis.
    hygiene_context: HygieneDecodeContext,

    // --- Data used only for improving diagnostics ---
    /// Information about the `extern crate` item or path that caused this crate to be loaded.
    /// If this is `None`, then the crate was injected (e.g., by the allocator).
    extern_crate: Option<ExternCrate>,
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
    translated_source_file: Arc<rustc_span::SourceFile>,
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
            // FIXME: This unwrap should never panic because we check that it won't when creating
            // `MetadataBlob`. Ideally we'd just have a `MetadataDecoder` and hand out subslices of
            // it as we do elsewhere in the compiler using `MetadataDecoder::split_at`. But we own
            // the data for the decoder so holding onto the `MemDecoder` too would make us a
            // self-referential struct which is downright goofy because `MetadataBlob` is already
            // self-referential. Probably `MemDecoder` should contain an `OwnedSlice`, but that
            // demands a significant refactoring due to our crate graph.
            opaque: MemDecoder::new(self.blob(), pos).unwrap(),
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
    #[inline]
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
    #[inline]
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
        let Some(tcx) = self.tcx else {
            bug!(
                "No TyCtxt found for decoding. \
                You need to explicitly pass `(crate_metadata_ref, tcx)` to `decode` instead of just `crate_metadata_ref`."
            );
        };
        tcx
    }

    #[inline]
    pub(crate) fn blob(&self) -> &'a MetadataBlob {
        self.blob
    }

    #[inline]
    fn cdata(&self) -> CrateMetadataRef<'a> {
        debug_assert!(self.cdata.is_some(), "missing CrateMetadata in DecodeContext");
        self.cdata.unwrap()
    }

    #[inline]
    fn map_encoded_cnum_to_current(&self, cnum: CrateNum) -> CrateNum {
        self.cdata().map_encoded_cnum_to_current(cnum)
    }

    #[inline]
    fn read_lazy_offset_then<T>(&mut self, f: impl Fn(NonZero<usize>) -> T) -> T {
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
        let position = NonZero::new(position).unwrap();
        self.lazy_state = LazyState::Previous(position);
        f(position)
    }

    fn read_lazy<T>(&mut self) -> LazyValue<T> {
        self.read_lazy_offset_then(|pos| LazyValue::from_position(pos))
    }

    fn read_lazy_array<T>(&mut self, len: usize) -> LazyArray<T> {
        self.read_lazy_offset_then(|pos| LazyArray::from_position_and_num_elems(pos, len))
    }

    fn read_lazy_table<I, T>(&mut self, width: usize, len: usize) -> LazyTable<I, T> {
        self.read_lazy_offset_then(|pos| LazyTable::from_position_and_encoded_size(pos, width, len))
    }

    #[inline]
    fn read_raw_bytes(&mut self, len: usize) -> &[u8] {
        self.opaque.read_raw_bytes(len)
    }

    fn decode_symbol_or_byte_symbol<S>(
        &mut self,
        new_from_index: impl Fn(u32) -> S,
        read_and_intern_str_or_byte_str_this: impl Fn(&mut Self) -> S,
        read_and_intern_str_or_byte_str_opaque: impl Fn(&mut MemDecoder<'a>) -> S,
    ) -> S {
        let tag = self.read_u8();

        match tag {
            SYMBOL_STR => read_and_intern_str_or_byte_str_this(self),
            SYMBOL_OFFSET => {
                // read str offset
                let pos = self.read_usize();

                // move to str offset and read
                self.opaque.with_position(pos, |d| read_and_intern_str_or_byte_str_opaque(d))
            }
            SYMBOL_PREDEFINED => new_from_index(self.read_u32()),
            _ => unreachable!(),
        }
    }
}

impl<'a, 'tcx> TyDecoder<'tcx> for DecodeContext<'a, 'tcx> {
    const CLEAR_CROSS_CRATE: bool = true;

    #[inline]
    fn interner(&self) -> TyCtxt<'tcx> {
        self.tcx()
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
        let new_opaque = self.opaque.split_at(pos);
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

impl<'a, 'tcx> Decodable<DecodeContext<'a, 'tcx>> for ExpnIndex {
    #[inline]
    fn decode(d: &mut DecodeContext<'a, 'tcx>) -> ExpnIndex {
        ExpnIndex::from_u32(d.read_u32())
    }
}

impl<'a, 'tcx> SpanDecoder for DecodeContext<'a, 'tcx> {
    fn decode_attr_id(&mut self) -> rustc_span::AttrId {
        let sess = self.sess.expect("can't decode AttrId without Session");
        sess.psess.attr_id_generator.mk_attr_id()
    }

    fn decode_crate_num(&mut self) -> CrateNum {
        let cnum = CrateNum::from_u32(self.read_u32());
        self.map_encoded_cnum_to_current(cnum)
    }

    fn decode_def_index(&mut self) -> DefIndex {
        DefIndex::from_u32(self.read_u32())
    }

    fn decode_def_id(&mut self) -> DefId {
        DefId { krate: Decodable::decode(self), index: Decodable::decode(self) }
    }

    fn decode_syntax_context(&mut self) -> SyntaxContext {
        let cdata = self.cdata();

        let Some(sess) = self.sess else {
            bug!(
                "Cannot decode SyntaxContext without Session.\
                You need to explicitly pass `(crate_metadata_ref, tcx)` to `decode` instead of just `crate_metadata_ref`."
            );
        };

        let cname = cdata.root.name();
        rustc_span::hygiene::decode_syntax_context(self, &cdata.hygiene_context, |_, id| {
            debug!("SpecializedDecoder<SyntaxContext>: decoding {}", id);
            cdata
                .root
                .syntax_contexts
                .get(cdata, id)
                .unwrap_or_else(|| panic!("Missing SyntaxContext {id:?} for crate {cname:?}"))
                .decode((cdata, sess))
        })
    }

    fn decode_expn_id(&mut self) -> ExpnId {
        let local_cdata = self.cdata();

        let Some(sess) = self.sess else {
            bug!(
                "Cannot decode ExpnId without Session. \
                You need to explicitly pass `(crate_metadata_ref, tcx)` to `decode` instead of just `crate_metadata_ref`."
            );
        };

        let cnum = CrateNum::decode(self);
        let index = u32::decode(self);

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

    fn decode_span(&mut self) -> Span {
        let start = self.position();
        let tag = SpanTag(self.peek_byte());
        let data = if tag.kind() == SpanKind::Indirect {
            // Skip past the tag we just peek'd.
            self.read_u8();
            // indirect tag lengths are safe to access, since they're (0, 8)
            let bytes_needed = tag.length().unwrap().0 as usize;
            let mut total = [0u8; usize::BITS as usize / 8];
            total[..bytes_needed].copy_from_slice(self.read_raw_bytes(bytes_needed));
            let offset_or_position = usize::from_le_bytes(total);
            let position = if tag.is_relative_offset() {
                start - offset_or_position
            } else {
                offset_or_position
            };
            self.with_position(position, SpanData::decode)
        } else {
            SpanData::decode(self)
        };
        data.span()
    }

    fn decode_symbol(&mut self) -> Symbol {
        self.decode_symbol_or_byte_symbol(
            Symbol::new,
            |this| Symbol::intern(this.read_str()),
            |opaque| Symbol::intern(opaque.read_str()),
        )
    }

    fn decode_byte_symbol(&mut self) -> ByteSymbol {
        self.decode_symbol_or_byte_symbol(
            ByteSymbol::new,
            |this| ByteSymbol::intern(this.read_byte_str()),
            |opaque| ByteSymbol::intern(opaque.read_byte_str()),
        )
    }
}

impl<'a, 'tcx> Decodable<DecodeContext<'a, 'tcx>> for SpanData {
    fn decode(decoder: &mut DecodeContext<'a, 'tcx>) -> SpanData {
        let tag = SpanTag::decode(decoder);
        let ctxt = tag.context().unwrap_or_else(|| SyntaxContext::decode(decoder));

        if tag.kind() == SpanKind::Partial {
            return DUMMY_SP.with_ctxt(ctxt).data();
        }

        debug_assert!(tag.kind() == SpanKind::Local || tag.kind() == SpanKind::Foreign);

        let lo = BytePos::decode(decoder);
        let len = tag.length().unwrap_or_else(|| BytePos::decode(decoder));
        let hi = lo + len;

        let Some(sess) = decoder.sess else {
            bug!(
                "Cannot decode Span without Session. \
                You need to explicitly pass `(crate_metadata_ref, tcx)` to `decode` instead of just `crate_metadata_ref`."
            )
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
        let source_file = if tag.kind() == SpanKind::Local {
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
                    decoder.cdata().root.header.name,
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

        // Do not try to decode parent for foreign spans (it wasn't encoded in the first place).
        SpanData { lo, hi, ctxt, parent: None }
    }
}

impl<'a, 'tcx> Decodable<DecodeContext<'a, 'tcx>> for &'tcx [(ty::Clause<'tcx>, Span)] {
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
    #[inline]
    fn decode(decoder: &mut DecodeContext<'a, 'tcx>) -> Self {
        let len = decoder.read_usize();
        if len == 0 { LazyArray::default() } else { decoder.read_lazy_array(len) }
    }
}

impl<'a, 'tcx, I: Idx, T> Decodable<DecodeContext<'a, 'tcx>> for LazyTable<I, T> {
    fn decode(decoder: &mut DecodeContext<'a, 'tcx>) -> Self {
        let width = decoder.read_usize();
        let len = decoder.read_usize();
        decoder.read_lazy_table(width, len)
    }
}

implement_ty_decoder!(DecodeContext<'a, 'tcx>);

impl MetadataBlob {
    pub(crate) fn check_compatibility(
        &self,
        cfg_version: &'static str,
    ) -> Result<(), Option<String>> {
        if !self.blob().starts_with(METADATA_HEADER) {
            if self.blob().starts_with(b"rust") {
                return Err(Some("<unknown rustc version>".to_owned()));
            }
            return Err(None);
        }

        let found_version =
            LazyValue::<String>::from_position(NonZero::new(METADATA_HEADER.len() + 8).unwrap())
                .decode(self);
        if rustc_version(cfg_version) != found_version {
            return Err(Some(found_version));
        }

        Ok(())
    }

    fn root_pos(&self) -> NonZero<usize> {
        let offset = METADATA_HEADER.len();
        let pos_bytes = self.blob()[offset..][..8].try_into().unwrap();
        let pos = u64::from_le_bytes(pos_bytes);
        NonZero::new(pos as usize).unwrap()
    }

    pub(crate) fn get_header(&self) -> CrateHeader {
        let pos = self.root_pos();
        LazyValue::<CrateHeader>::from_position(pos).decode(self)
    }

    pub(crate) fn get_root(&self) -> CrateRoot {
        let pos = self.root_pos();
        LazyValue::<CrateRoot>::from_position(pos).decode(self)
    }

    pub(crate) fn list_crate_metadata(
        &self,
        out: &mut dyn io::Write,
        ls_kinds: &[String],
    ) -> io::Result<()> {
        let root = self.get_root();

        let all_ls_kinds = vec![
            "root".to_owned(),
            "lang_items".to_owned(),
            "features".to_owned(),
            "items".to_owned(),
        ];
        let ls_kinds = if ls_kinds.contains(&"all".to_owned()) { &all_ls_kinds } else { ls_kinds };

        for kind in ls_kinds {
            match &**kind {
                "root" => {
                    writeln!(out, "Crate info:")?;
                    writeln!(out, "name {}{}", root.name(), root.extra_filename)?;
                    writeln!(
                        out,
                        "hash {} stable_crate_id {:?}",
                        root.hash(),
                        root.stable_crate_id
                    )?;
                    writeln!(out, "proc_macro {:?}", root.proc_macro_data.is_some())?;
                    writeln!(out, "triple {}", root.header.triple.tuple())?;
                    writeln!(out, "edition {}", root.edition)?;
                    writeln!(out, "symbol_mangling_version {:?}", root.symbol_mangling_version)?;
                    writeln!(
                        out,
                        "required_panic_strategy {:?} panic_in_drop_strategy {:?}",
                        root.required_panic_strategy, root.panic_in_drop_strategy
                    )?;
                    writeln!(
                        out,
                        "has_global_allocator {} has_alloc_error_handler {} has_panic_handler {} has_default_lib_allocator {}",
                        root.has_global_allocator,
                        root.has_alloc_error_handler,
                        root.has_panic_handler,
                        root.has_default_lib_allocator
                    )?;
                    writeln!(
                        out,
                        "compiler_builtins {} needs_allocator {} needs_panic_runtime {} no_builtins {} panic_runtime {} profiler_runtime {}",
                        root.compiler_builtins,
                        root.needs_allocator,
                        root.needs_panic_runtime,
                        root.no_builtins,
                        root.panic_runtime,
                        root.profiler_runtime
                    )?;

                    writeln!(out, "=External Dependencies=")?;
                    let dylib_dependency_formats =
                        root.dylib_dependency_formats.decode(self).collect::<Vec<_>>();
                    for (i, dep) in root.crate_deps.decode(self).enumerate() {
                        let CrateDep { name, extra_filename, hash, host_hash, kind, is_private } =
                            dep;
                        let number = i + 1;

                        writeln!(
                            out,
                            "{number} {name}{extra_filename} hash {hash} host_hash {host_hash:?} kind {kind:?} {privacy}{linkage}",
                            privacy = if is_private { "private" } else { "public" },
                            linkage = if dylib_dependency_formats.is_empty() {
                                String::new()
                            } else {
                                format!(" linkage {:?}", dylib_dependency_formats[i])
                            }
                        )?;
                    }
                    write!(out, "\n")?;
                }

                "lang_items" => {
                    writeln!(out, "=Lang items=")?;
                    for (id, lang_item) in root.lang_items.decode(self) {
                        writeln!(
                            out,
                            "{} = crate{}",
                            lang_item.name(),
                            DefPath::make(LOCAL_CRATE, id, |parent| root
                                .tables
                                .def_keys
                                .get(self, parent)
                                .unwrap()
                                .decode(self))
                            .to_string_no_crate_verbose()
                        )?;
                    }
                    for lang_item in root.lang_items_missing.decode(self) {
                        writeln!(out, "{} = <missing>", lang_item.name())?;
                    }
                    write!(out, "\n")?;
                }

                "features" => {
                    writeln!(out, "=Lib features=")?;
                    for (feature, since) in root.lib_features.decode(self) {
                        writeln!(
                            out,
                            "{}{}",
                            feature,
                            if let FeatureStability::AcceptedSince(since) = since {
                                format!(" since {since}")
                            } else {
                                String::new()
                            }
                        )?;
                    }
                    write!(out, "\n")?;
                }

                "items" => {
                    writeln!(out, "=Items=")?;

                    fn print_item(
                        blob: &MetadataBlob,
                        out: &mut dyn io::Write,
                        item: DefIndex,
                        indent: usize,
                    ) -> io::Result<()> {
                        let root = blob.get_root();

                        let def_kind = root.tables.def_kind.get(blob, item).unwrap();
                        let def_key = root.tables.def_keys.get(blob, item).unwrap().decode(blob);
                        #[allow(rustc::symbol_intern_string_literal)]
                        let def_name = if item == CRATE_DEF_INDEX {
                            kw::Crate
                        } else {
                            def_key
                                .disambiguated_data
                                .data
                                .get_opt_name()
                                .unwrap_or_else(|| Symbol::intern("???"))
                        };
                        let visibility =
                            root.tables.visibility.get(blob, item).unwrap().decode(blob).map_id(
                                |index| {
                                    format!(
                                        "crate{}",
                                        DefPath::make(LOCAL_CRATE, index, |parent| root
                                            .tables
                                            .def_keys
                                            .get(blob, parent)
                                            .unwrap()
                                            .decode(blob))
                                        .to_string_no_crate_verbose()
                                    )
                                },
                            );
                        write!(
                            out,
                            "{nil: <indent$}{:?} {:?} {} {{",
                            visibility,
                            def_kind,
                            def_name,
                            nil = "",
                        )?;

                        if let Some(children) =
                            root.tables.module_children_non_reexports.get(blob, item)
                        {
                            write!(out, "\n")?;
                            for child in children.decode(blob) {
                                print_item(blob, out, child, indent + 4)?;
                            }
                            writeln!(out, "{nil: <indent$}}}", nil = "")?;
                        } else {
                            writeln!(out, "}}")?;
                        }

                        Ok(())
                    }

                    print_item(self, out, CRATE_DEF_INDEX, 0)?;

                    write!(out, "\n")?;
                }

                _ => {
                    writeln!(
                        out,
                        "unknown -Zls kind. allowed values are: all, root, lang_items, features, items"
                    )?;
                }
            }
        }

        Ok(())
    }
}

impl CrateRoot {
    pub(crate) fn is_proc_macro_crate(&self) -> bool {
        self.proc_macro_data.is_some()
    }

    pub(crate) fn name(&self) -> Symbol {
        self.header.name
    }

    pub(crate) fn hash(&self) -> Svh {
        self.header.hash
    }

    pub(crate) fn stable_crate_id(&self) -> StableCrateId {
        self.stable_crate_id
    }

    pub(crate) fn decode_crate_deps<'a>(
        &self,
        metadata: &'a MetadataBlob,
    ) -> impl ExactSizeIterator<Item = CrateDep> {
        self.crate_deps.decode(metadata)
    }

    pub(crate) fn decode_target_modifiers<'a>(
        &self,
        metadata: &'a MetadataBlob,
    ) -> impl ExactSizeIterator<Item = TargetModifier> {
        self.target_modifiers.decode(metadata)
    }
}

impl<'a> CrateMetadataRef<'a> {
    fn missing(self, descr: &str, id: DefIndex) -> ! {
        bug!("missing `{descr}` for {:?}", self.local_def_id(id))
    }

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
        let span = self
            .root
            .tables
            .def_ident_span
            .get(self, item_index)
            .unwrap_or_else(|| self.missing("def_ident_span", item_index))
            .decode((self, sess));
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
        self.root
            .tables
            .def_kind
            .get(self, item_id)
            .unwrap_or_else(|| self.missing("def_kind", item_id))
    }

    fn get_span(self, index: DefIndex, sess: &Session) -> Span {
        self.root
            .tables
            .def_span
            .get(self, index)
            .unwrap_or_else(|| self.missing("def_span", index))
            .decode((self, sess))
    }

    fn load_proc_macro<'tcx>(self, id: DefIndex, tcx: TyCtxt<'tcx>) -> SyntaxExtension {
        let (name, kind, helper_attrs) = match *self.raw_proc_macro(id) {
            ProcMacro::CustomDerive { trait_name, attributes, client } => {
                let helper_attrs =
                    attributes.iter().cloned().map(Symbol::intern).collect::<Vec<_>>();
                (
                    trait_name,
                    SyntaxExtensionKind::Derive(Arc::new(DeriveProcMacro { client })),
                    helper_attrs,
                )
            }
            ProcMacro::Attr { name, client } => {
                (name, SyntaxExtensionKind::Attr(Arc::new(AttrProcMacro { client })), Vec::new())
            }
            ProcMacro::Bang { name, client } => {
                (name, SyntaxExtensionKind::Bang(Arc::new(BangProcMacro { client })), Vec::new())
            }
        };

        let sess = tcx.sess;
        let attrs: Vec<_> = self.get_item_attrs(id, sess).collect();
        SyntaxExtension::new(
            sess,
            kind,
            self.get_span(id, sess),
            helper_attrs,
            self.root.edition,
            Symbol::intern(name),
            &attrs,
            false,
        )
    }

    fn get_variant(
        self,
        kind: DefKind,
        index: DefIndex,
        parent_did: DefId,
    ) -> (VariantIdx, ty::VariantDef) {
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

        (
            data.idx,
            ty::VariantDef::new(
                self.item_name(index),
                variant_did,
                ctor,
                data.discr,
                self.get_associated_item_or_field_def_ids(index)
                    .map(|did| ty::FieldDef {
                        did,
                        name: self.item_name(did.index),
                        vis: self.get_visibility(did.index),
                        safety: self.get_safety(did.index),
                        value: self.get_default_field(did.index),
                    })
                    .collect(),
                parent_did,
                None,
                data.is_non_exhaustive,
            ),
        )
    }

    fn get_adt_def<'tcx>(self, item_id: DefIndex, tcx: TyCtxt<'tcx>) -> ty::AdtDef<'tcx> {
        let kind = self.def_kind(item_id);
        let did = self.local_def_id(item_id);

        let adt_kind = match kind {
            DefKind::Enum => ty::AdtKind::Enum,
            DefKind::Struct => ty::AdtKind::Struct,
            DefKind::Union => ty::AdtKind::Union,
            _ => bug!("get_adt_def called on a non-ADT {:?}", did),
        };
        let repr = self.root.tables.repr_options.get(self, item_id).unwrap().decode(self);

        let mut variants: Vec<_> = if let ty::AdtKind::Enum = adt_kind {
            self.root
                .tables
                .module_children_non_reexports
                .get(self, item_id)
                .expect("variants are not encoded for an enum")
                .decode(self)
                .filter_map(|index| {
                    let kind = self.def_kind(index);
                    match kind {
                        DefKind::Ctor(..) => None,
                        _ => Some(self.get_variant(kind, index, did)),
                    }
                })
                .collect()
        } else {
            std::iter::once(self.get_variant(kind, item_id, did)).collect()
        };

        variants.sort_by_key(|(idx, _)| *idx);

        tcx.mk_adt_def(
            did,
            adt_kind,
            variants.into_iter().map(|(_, variant)| variant).collect(),
            repr,
        )
    }

    fn get_visibility(self, id: DefIndex) -> Visibility<DefId> {
        self.root
            .tables
            .visibility
            .get(self, id)
            .unwrap_or_else(|| self.missing("visibility", id))
            .decode(self)
            .map_id(|index| self.local_def_id(index))
    }

    fn get_safety(self, id: DefIndex) -> Safety {
        self.root.tables.safety.get(self, id).unwrap_or_else(|| self.missing("safety", id))
    }

    fn get_default_field(self, id: DefIndex) -> Option<DefId> {
        self.root.tables.default_fields.get(self, id).map(|d| d.decode(self))
    }

    fn get_expn_that_defined(self, id: DefIndex, sess: &Session) -> ExpnId {
        self.root
            .tables
            .expn_that_defined
            .get(self, id)
            .unwrap_or_else(|| self.missing("expn_that_defined", id))
            .decode((self, sess))
    }

    fn get_debugger_visualizers(self) -> Vec<DebuggerVisualizerFile> {
        self.root.debugger_visualizers.decode(self).collect::<Vec<_>>()
    }

    /// Iterates over all the stability attributes in the given crate.
    fn get_lib_features(self) -> LibFeatures {
        LibFeatures {
            stability: self
                .root
                .lib_features
                .decode(self)
                .map(|(sym, stab)| (sym, (stab, DUMMY_SP)))
                .collect(),
        }
    }

    /// Iterates over the stability implications in the given crate (when a `#[unstable]` attribute
    /// has an `implied_by` meta item, then the mapping from the implied feature to the actual
    /// feature is a stability implication).
    fn get_stability_implications<'tcx>(self, tcx: TyCtxt<'tcx>) -> &'tcx [(Symbol, Symbol)] {
        tcx.arena.alloc_from_iter(self.root.stability_implications.decode(self))
    }

    /// Iterates over the lang items in the given crate.
    fn get_lang_items<'tcx>(self, tcx: TyCtxt<'tcx>) -> &'tcx [(DefId, LangItem)] {
        tcx.arena.alloc_from_iter(
            self.root
                .lang_items
                .decode(self)
                .map(move |(def_index, index)| (self.local_def_id(def_index), index)),
        )
    }

    fn get_stripped_cfg_items<'tcx>(
        self,
        cnum: CrateNum,
        tcx: TyCtxt<'tcx>,
    ) -> &'tcx [StrippedCfgItem] {
        let item_names = self
            .root
            .stripped_cfg_items
            .decode((self, tcx))
            .map(|item| item.map_mod_id(|index| DefId { krate: cnum, index }));
        tcx.arena.alloc_from_iter(item_names)
    }

    /// Iterates over the diagnostic items in the given crate.
    fn get_diagnostic_items(self) -> DiagnosticItems {
        let mut id_to_name = DefIdMap::default();
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
        let res = Res::Def(self.def_kind(id), self.local_def_id(id));
        let vis = self.get_visibility(id);

        ModChild { ident, res, vis, reexport_chain: Default::default() }
    }

    /// Iterates over all named children of the given module,
    /// including both proper items and reexports.
    /// Module here is understood in name resolution sense - it can be a `mod` item,
    /// or a crate root, or an enum, or a trait.
    fn get_module_children(
        self,
        id: DefIndex,
        sess: &'a Session,
    ) -> impl Iterator<Item = ModChild> {
        gen move {
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
                let non_reexports = self.root.tables.module_children_non_reexports.get(self, id);
                for child_index in non_reexports.unwrap().decode(self) {
                    yield self.get_mod_child(child_index, sess);
                }

                let reexports = self.root.tables.module_children_reexports.get(self, id);
                if !reexports.is_default() {
                    for reexport in reexports.decode((self, sess)) {
                        yield reexport;
                    }
                }
            }
        }
    }

    fn is_ctfe_mir_available(self, id: DefIndex) -> bool {
        self.root.tables.mir_for_ctfe.get(self, id).is_some()
    }

    fn is_item_mir_available(self, id: DefIndex) -> bool {
        self.root.tables.optimized_mir.get(self, id).is_some()
    }

    fn get_fn_has_self_parameter(self, id: DefIndex, sess: &'a Session) -> bool {
        self.root
            .tables
            .fn_arg_idents
            .get(self, id)
            .expect("argument names not encoded for a function")
            .decode((self, sess))
            .nth(0)
            .is_some_and(|ident| matches!(ident, Some(Ident { name: kw::SelfLower, .. })))
    }

    fn get_associated_item_or_field_def_ids(self, id: DefIndex) -> impl Iterator<Item = DefId> {
        self.root
            .tables
            .associated_item_or_field_def_ids
            .get(self, id)
            .unwrap_or_else(|| self.missing("associated_item_or_field_def_ids", id))
            .decode(self)
            .map(move |child_index| self.local_def_id(child_index))
    }

    fn get_associated_item(self, id: DefIndex, sess: &'a Session) -> ty::AssocItem {
        let kind = match self.def_kind(id) {
            DefKind::AssocConst => ty::AssocKind::Const { name: self.item_name(id) },
            DefKind::AssocFn => ty::AssocKind::Fn {
                name: self.item_name(id),
                has_self: self.get_fn_has_self_parameter(id, sess),
            },
            DefKind::AssocTy => {
                let data = if let Some(rpitit_info) = self.root.tables.opt_rpitit_info.get(self, id)
                {
                    ty::AssocTypeData::Rpitit(rpitit_info.decode(self))
                } else {
                    ty::AssocTypeData::Normal(self.item_name(id))
                };
                ty::AssocKind::Type { data }
            }
            _ => bug!("cannot get associated-item of `{:?}`", self.def_key(id)),
        };
        let container = self.root.tables.assoc_container.get(self, id).unwrap().decode(self);

        ty::AssocItem { kind, def_id: self.local_def_id(id), container }
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
    ) -> impl Iterator<Item = hir::Attribute> {
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

    fn get_inherent_implementations_for_type<'tcx>(
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

    /// Decodes all traits in the crate (for rustdoc and rustc diagnostics).
    fn get_traits(self) -> impl Iterator<Item = DefId> {
        self.root.traits.decode(self).map(move |index| self.local_def_id(index))
    }

    /// Decodes all trait impls in the crate (for rustdoc).
    fn get_trait_impls(self) -> impl Iterator<Item = DefId> {
        self.cdata.trait_impls.values().flat_map(move |impls| {
            impls.decode(self).map(move |(impl_index, _)| self.local_def_id(impl_index))
        })
    }

    fn get_incoherent_impls<'tcx>(self, tcx: TyCtxt<'tcx>, simp: SimplifiedType) -> &'tcx [DefId] {
        if let Some(impls) = self.cdata.incoherent_impls.get(&simp) {
            tcx.arena.alloc_from_iter(impls.decode(self).map(|idx| self.local_def_id(idx)))
        } else {
            &[]
        }
    }

    fn get_implementations_of_trait<'tcx>(
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

    fn get_native_libraries(self, sess: &'a Session) -> impl Iterator<Item = NativeLib> {
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

    fn get_foreign_modules(self, sess: &'a Session) -> impl Iterator<Item = ForeignModule> {
        self.root.foreign_modules.decode((self, sess))
    }

    fn get_dylib_dependency_formats<'tcx>(
        self,
        tcx: TyCtxt<'tcx>,
    ) -> &'tcx [(CrateNum, LinkagePreference)] {
        tcx.arena.alloc_from_iter(
            self.root.dylib_dependency_formats.decode(self).enumerate().flat_map(|(i, link)| {
                let cnum = CrateNum::new(i + 1); // We skipped LOCAL_CRATE when encoding
                link.map(|link| (self.cnum_map[cnum], link))
            }),
        )
    }

    fn get_missing_lang_items<'tcx>(self, tcx: TyCtxt<'tcx>) -> &'tcx [LangItem] {
        tcx.arena.alloc_from_iter(self.root.lang_items_missing.decode(self))
    }

    fn get_exportable_items(self) -> impl Iterator<Item = DefId> {
        self.root.exportable_items.decode(self).map(move |index| self.local_def_id(index))
    }

    fn get_stable_order_of_exportable_impls(self) -> impl Iterator<Item = (DefId, usize)> {
        self.root
            .stable_order_of_exportable_impls
            .decode(self)
            .map(move |v| (self.local_def_id(v.0), v.1))
    }

    fn exported_non_generic_symbols<'tcx>(
        self,
        tcx: TyCtxt<'tcx>,
    ) -> &'tcx [(ExportedSymbol<'tcx>, SymbolExportInfo)] {
        tcx.arena.alloc_from_iter(self.root.exported_non_generic_symbols.decode((self, tcx)))
    }

    fn exported_generic_symbols<'tcx>(
        self,
        tcx: TyCtxt<'tcx>,
    ) -> &'tcx [(ExportedSymbol<'tcx>, SymbolExportInfo)] {
        tcx.arena.alloc_from_iter(self.root.exported_generic_symbols.decode((self, tcx)))
    }

    fn get_macro(self, id: DefIndex, sess: &Session) -> ast::MacroDef {
        match self.def_kind(id) {
            DefKind::Macro(_) => {
                let macro_rules = self.root.tables.is_macro_rules.get(self, id);
                let body =
                    self.root.tables.macro_definition.get(self, id).unwrap().decode((self, sess));
                ast::MacroDef { macro_rules, body: Box::new(body) }
            }
            _ => bug!(),
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

    #[inline]
    fn def_path_hash(self, index: DefIndex) -> DefPathHash {
        // This is a hack to workaround the fact that we can't easily encode/decode a Hash64
        // into the FixedSizeEncoding, as Hash64 lacks a Default impl. A future refactor to
        // relax the Default restriction will likely fix this.
        let fingerprint = Fingerprint::new(
            self.root.stable_crate_id.as_u64(),
            self.root.tables.def_path_hashes.get(self, index),
        );
        DefPathHash::new(self.root.stable_crate_id, fingerprint.split().1)
    }

    #[inline]
    fn def_path_hash_to_def_index(self, hash: DefPathHash) -> Option<DefIndex> {
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
        fn filter<'a>(
            sess: &Session,
            real_source_base_dir: &Option<PathBuf>,
            path: Option<&'a Path>,
        ) -> Option<&'a Path> {
            path.filter(|_| {
                // Only spend time on further checks if we have what to translate *to*.
                real_source_base_dir.is_some()
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

        let try_to_translate_virtual_to_real =
            |virtual_source_base_dir: Option<&str>,
             real_source_base_dir: &Option<PathBuf>,
             name: &mut rustc_span::FileName| {
                let virtual_source_base_dir = [
                    filter(sess, real_source_base_dir, virtual_source_base_dir.map(Path::new)),
                    filter(
                        sess,
                        real_source_base_dir,
                        sess.opts.unstable_opts.simulate_remapped_rust_src_base.as_deref(),
                    ),
                ];

                debug!(
                    "try_to_translate_virtual_to_real(name={:?}): \
                     virtual_source_base_dir={:?}, real_source_base_dir={:?}",
                    name, virtual_source_base_dir, real_source_base_dir,
                );

                for virtual_dir in virtual_source_base_dir.iter().flatten() {
                    if let Some(real_dir) = &real_source_base_dir
                        && let rustc_span::FileName::Real(old_name) = name
                        && let rustc_span::RealFileName::Remapped { local_path: _, virtual_name } =
                            old_name
                        && let Ok(rest) = virtual_name.strip_prefix(virtual_dir)
                    {
                        let new_path = real_dir.join(rest);

                        debug!(
                            "try_to_translate_virtual_to_real: `{}` -> `{}`",
                            virtual_name.display(),
                            new_path.display(),
                        );

                        // Check if the translated real path is affected by any user-requested
                        // remaps via --remap-path-prefix. Apply them if so.
                        // Note that this is a special case for imported rust-src paths specified by
                        // https://rust-lang.github.io/rfcs/3127-trim-paths.html#handling-sysroot-paths.
                        // Other imported paths are not currently remapped (see #66251).
                        let (user_remapped, applied) =
                            sess.source_map().path_mapping().map_prefix(&new_path);
                        let new_name = if applied {
                            rustc_span::RealFileName::Remapped {
                                local_path: Some(new_path.clone()),
                                virtual_name: user_remapped.to_path_buf(),
                            }
                        } else {
                            rustc_span::RealFileName::LocalPath(new_path)
                        };
                        *old_name = new_name;
                    }
                }
            };

        let try_to_translate_real_to_virtual =
            |virtual_source_base_dir: Option<&str>,
             real_source_base_dir: &Option<PathBuf>,
             subdir: &str,
             name: &mut rustc_span::FileName| {
                if let Some(virtual_dir) = &sess.opts.unstable_opts.simulate_remapped_rust_src_base
                    && let Some(real_dir) = real_source_base_dir
                    && let rustc_span::FileName::Real(old_name) = name
                {
                    let relative_path = match old_name {
                        rustc_span::RealFileName::LocalPath(local) => {
                            local.strip_prefix(real_dir).ok()
                        }
                        rustc_span::RealFileName::Remapped { virtual_name, .. } => {
                            virtual_source_base_dir
                                .and_then(|virtual_dir| virtual_name.strip_prefix(virtual_dir).ok())
                        }
                    };
                    debug!(
                        ?relative_path,
                        ?virtual_dir,
                        ?subdir,
                        "simulate_remapped_rust_src_base"
                    );
                    if let Some(rest) = relative_path.and_then(|p| p.strip_prefix(subdir).ok()) {
                        *old_name = rustc_span::RealFileName::Remapped {
                            local_path: None,
                            virtual_name: virtual_dir.join(subdir).join(rest),
                        };
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
                let original_end_pos = source_file_to_import.end_position();
                let rustc_span::SourceFile {
                    mut name,
                    src_hash,
                    checksum_hash,
                    start_pos: original_start_pos,
                    source_len,
                    lines,
                    multibyte_chars,
                    normalized_pos,
                    stable_id,
                    ..
                } = source_file_to_import;

                // If this file is under $sysroot/lib/rustlib/src/
                // and the user wish to simulate remapping with -Z simulate-remapped-rust-src-base,
                // then we change `name` to a similar state as if the rust was bootstrapped
                // with `remap-debuginfo = true`.
                // This is useful for testing so that tests about the effects of
                // `try_to_translate_virtual_to_real` don't have to worry about how the
                // compiler is bootstrapped.
                try_to_translate_real_to_virtual(
                    option_env!("CFG_VIRTUAL_RUST_SOURCE_BASE_DIR"),
                    &sess.opts.real_rust_source_base_dir,
                    "library",
                    &mut name,
                );

                // If this file is under $sysroot/lib/rustlib/rustc-src/
                // and the user wish to simulate remapping with -Z simulate-remapped-rust-src-base,
                // then we change `name` to a similar state as if the rust was bootstrapped
                // with `remap-debuginfo = true`.
                try_to_translate_real_to_virtual(
                    option_env!("CFG_VIRTUAL_RUSTC_DEV_SOURCE_BASE_DIR"),
                    &sess.opts.real_rustc_dev_source_base_dir,
                    "compiler",
                    &mut name,
                );

                // If this file's path has been remapped to `/rustc/$hash`,
                // we might be able to reverse that.
                //
                // NOTE: if you update this, you might need to also update bootstrap's code for generating
                // the `rust-src` component in `Src::run` in `src/bootstrap/dist.rs`.
                try_to_translate_virtual_to_real(
                    option_env!("CFG_VIRTUAL_RUST_SOURCE_BASE_DIR"),
                    &sess.opts.real_rust_source_base_dir,
                    &mut name,
                );

                // If this file's path has been remapped to `/rustc-dev/$hash`,
                // we might be able to reverse that.
                //
                // NOTE: if you update this, you might need to also update bootstrap's code for generating
                // the `rustc-dev` component in `Src::run` in `src/bootstrap/dist.rs`.
                try_to_translate_virtual_to_real(
                    option_env!("CFG_VIRTUAL_RUSTC_DEV_SOURCE_BASE_DIR"),
                    &sess.opts.real_rustc_dev_source_base_dir,
                    &mut name,
                );

                let local_version = sess.source_map().new_imported_source_file(
                    name,
                    src_hash,
                    checksum_hash,
                    stable_id,
                    source_len.to_u32(),
                    self.cnum,
                    lines,
                    multibyte_chars,
                    normalized_pos,
                    source_file_index,
                );
                debug!(
                    "CrateMetaData::imported_source_files alloc \
                         source_file {:?} original (start_pos {:?} source_len {:?}) \
                         translated (start_pos {:?} source_len {:?})",
                    local_version.name,
                    original_start_pos,
                    source_len,
                    local_version.start_pos,
                    local_version.source_len
                );

                ImportedSourceFile {
                    original_start_pos,
                    original_end_pos,
                    translated_source_file: local_version,
                }
            })
            .clone()
    }

    fn get_attr_flags(self, index: DefIndex) -> AttrFlags {
        self.root.tables.attr_flags.get(self, index)
    }

    fn get_intrinsic(self, index: DefIndex) -> Option<ty::IntrinsicDef> {
        self.root.tables.intrinsic.get(self, index).map(|d| d.decode(self))
    }

    fn get_doc_link_resolutions(self, index: DefIndex) -> DocLinkResMap {
        self.root
            .tables
            .doc_link_resolutions
            .get(self, index)
            .expect("no resolutions for a doc link")
            .decode(self)
    }

    fn get_doc_link_traits_in_scope(self, index: DefIndex) -> impl Iterator<Item = DefId> {
        self.root
            .tables
            .doc_link_traits_in_scope
            .get(self, index)
            .expect("no traits in scope for a doc link")
            .decode(self)
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
        let dependencies = cnum_map.iter().copied().collect();

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
            dep_kind,
            source: Arc::new(source),
            private_dep,
            host_hash,
            used: false,
            extern_crate: None,
            hygiene_context: Default::default(),
            def_key_cache: Default::default(),
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

    pub(crate) fn dependencies(&self) -> impl Iterator<Item = CrateNum> {
        self.dependencies.iter().copied()
    }

    pub(crate) fn target_modifiers(&self) -> TargetModifiers {
        self.root.decode_target_modifiers(&self.blob).collect()
    }

    /// Keep `new_extern_crate` if it looks better in diagnostics
    pub(crate) fn update_extern_crate_diagnostics(
        &mut self,
        new_extern_crate: ExternCrate,
    ) -> bool {
        let update =
            self.extern_crate.as_ref().is_none_or(|old| old.rank() < new_extern_crate.rank());
        if update {
            self.extern_crate = Some(new_extern_crate);
        }
        update
    }

    pub(crate) fn source(&self) -> &CrateSource {
        &*self.source
    }

    pub(crate) fn dep_kind(&self) -> CrateDepKind {
        self.dep_kind
    }

    pub(crate) fn set_dep_kind(&mut self, dep_kind: CrateDepKind) {
        self.dep_kind = dep_kind;
    }

    pub(crate) fn update_and_private_dep(&mut self, private_dep: bool) {
        self.private_dep &= private_dep;
    }

    pub(crate) fn used(&self) -> bool {
        self.used
    }

    pub(crate) fn required_panic_strategy(&self) -> Option<PanicStrategy> {
        self.root.required_panic_strategy
    }

    pub(crate) fn needs_panic_runtime(&self) -> bool {
        self.root.needs_panic_runtime
    }

    pub(crate) fn is_private_dep(&self) -> bool {
        self.private_dep
    }

    pub(crate) fn is_panic_runtime(&self) -> bool {
        self.root.panic_runtime
    }

    pub(crate) fn is_profiler_runtime(&self) -> bool {
        self.root.profiler_runtime
    }

    pub(crate) fn is_compiler_builtins(&self) -> bool {
        self.root.compiler_builtins
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

    pub(crate) fn proc_macros_for_crate(
        &self,
        krate: CrateNum,
        cstore: &CStore,
    ) -> impl Iterator<Item = DefId> {
        gen move {
            for def_id in self.root.proc_macro_data.as_ref().into_iter().flat_map(move |data| {
                data.macros
                    .decode(CrateMetadataRef { cdata: self, cstore })
                    .map(move |index| DefId { index, krate })
            }) {
                yield def_id;
            }
        }
    }

    pub(crate) fn name(&self) -> Symbol {
        self.root.header.name
    }

    pub(crate) fn hash(&self) -> Svh {
        self.root.header.hash
    }

    pub(crate) fn has_async_drops(&self) -> bool {
        self.root.tables.adt_async_destructor.len > 0
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
