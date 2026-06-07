use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::hash_map::Entry;
use std::fs::File;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::memmap::{Mmap, MmapMut};
use rustc_data_structures::sync::{par_for_each_in, par_join};
use rustc_data_structures::temp_dir::MaybeTempDir;
use rustc_data_structures::thousands::usize_with_underscores;
use rustc_hir as hir;
use rustc_hir::attrs::{AttributeKind, EncodeCrossCrate};
use rustc_hir::def_id::{CRATE_DEF_ID, LOCAL_CRATE, LocalDefId, LocalDefIdSet};
use rustc_hir::definitions::DefPathData;
use rustc_hir::find_attr;
use rustc_hir_pretty::id_to_string;
use rustc_middle::dep_graph::WorkProductId;
use rustc_middle::middle::dependency_format::Linkage;
use rustc_middle::mir::interpret;
use rustc_middle::query::Providers;
use rustc_middle::traits::specialization_graph;
use rustc_middle::ty::codec::TyEncoder;
use rustc_middle::ty::fast_reject::{self, TreatParams};
use rustc_middle::ty::{AssocContainer, Visibility};
use rustc_middle::{bug, span_bug};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder, opaque};
use rustc_session::config::mitigation_coverage::DeniedPartialMitigation;
use rustc_session::config::{CrateType, OptLevel, TargetModifier};
use rustc_span::hygiene::HygieneEncodeContext;
use rustc_span::{
    ByteSymbol, ExternalSource, FileName, SourceFile, SpanData, SpanEncoder, StableSourceFileId,
    Symbol, SyntaxContext, sym,
};
use tracing::{debug, instrument, trace};

use self::public_api_hasher::{
    HashableCrateHeader, HashableCrateRoot, Hashed, NoneIfHashed, PublicApiHasher,
    PublicApiHashingContext,
};
use crate::eii::EiiMapEncodedKeyValue;
use crate::errors::{FailCreateFileEncoder, FailWriteFile};
use crate::rmeta::encoder::public_api_hasher::PublicApiHashState;
use crate::rmeta::*;

pub(super) mod public_api_hasher;

pub(super) struct EncodeContext<'a, 'tcx> {
    opaque: opaque::FileEncoder,
    tcx: TyCtxt<'tcx>,
    feat: &'tcx rustc_feature::Features,
    tables: TableBuilders,

    lazy_state: LazyState,
    span_shorthands: FxHashMap<Span, usize>,
    type_shorthands: FxHashMap<Ty<'tcx>, usize>,
    predicate_shorthands: FxHashMap<ty::PredicateKind<'tcx>, usize>,

    interpret_allocs: FxIndexSet<interpret::AllocId>,

    // This is used to speed up Span encoding.
    // The `usize` is an index into the `MonotonicVec`
    // that stores the `SourceFile`
    source_file_cache: (Arc<SourceFile>, usize),
    // The indices (into the `SourceMap`'s `MonotonicVec`)
    // of all of the `SourceFiles` that we need to serialize.
    // When we serialize a `Span`, we insert the index of its
    // `SourceFile` into the `FxIndexSet`.
    // The order inside the `FxIndexSet` is used as on-disk
    // order of `SourceFiles`, and encoded inside `Span`s.
    required_source_files: Option<FxIndexSet<usize>>,
    is_proc_macro: bool,
    hygiene_ctxt: &'a HygieneEncodeContext,
    // Used for both `Symbol`s and `ByteSymbol`s.
    symbol_index_table: FxHashMap<u32, usize>,
}

/// If the current crate is a proc-macro, returns early with `LazyArray::default()`.
/// This is useful for skipping the encoding of things that aren't needed
/// for proc-macro crates.
macro_rules! empty_proc_macro {
    ($self:ident) => {
        if $self.is_proc_macro {
            return Default::default();
        }
    };
}

macro_rules! encoder_methods {
    ($($name:ident($ty:ty);)*) => {
        $(fn $name(&mut self, value: $ty) {
            self.opaque.$name(value)
        })*
    }
}

impl<'a, 'tcx> Encoder for EncodeContext<'a, 'tcx> {
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

impl<'a, 'tcx, T> Encodable<EncodeContext<'a, 'tcx>> for LazyValue<T> {
    fn encode(&self, e: &mut EncodeContext<'a, 'tcx>) {
        e.emit_lazy_distance(self.position);
    }
}

impl<'a, 'tcx, T> Encodable<EncodeContext<'a, 'tcx>> for LazyArray<T> {
    fn encode(&self, e: &mut EncodeContext<'a, 'tcx>) {
        e.emit_usize(self.num_elems);
        if self.num_elems > 0 {
            e.emit_lazy_distance(self.position)
        }
    }
}

impl<'a, 'tcx, I, T> Encodable<EncodeContext<'a, 'tcx>> for LazyTable<I, T> {
    fn encode(&self, e: &mut EncodeContext<'a, 'tcx>) {
        e.emit_usize(self.width);
        e.emit_usize(self.len);
        e.emit_lazy_distance(self.position);
    }
}

impl<'a, 'tcx> Encodable<EncodeContext<'a, 'tcx>> for ExpnIndex {
    fn encode(&self, s: &mut EncodeContext<'a, 'tcx>) {
        s.emit_u32(self.as_u32());
    }
}

impl<'a, 'tcx> SpanEncoder for EncodeContext<'a, 'tcx> {
    fn encode_crate_num(&mut self, crate_num: CrateNum) {
        if crate_num != LOCAL_CRATE && self.is_proc_macro {
            panic!("Attempted to encode non-local CrateNum {crate_num:?} for proc-macro crate");
        }
        self.emit_u32(crate_num.as_u32());
    }

    fn encode_def_index(&mut self, def_index: DefIndex) {
        self.emit_u32(def_index.as_u32());
    }

    fn encode_def_id(&mut self, def_id: DefId) {
        def_id.krate.encode(self);
        def_id.index.encode(self);
    }

    fn encode_syntax_context(&mut self, syntax_context: SyntaxContext) {
        rustc_span::hygiene::raw_encode_syntax_context(syntax_context, self.hygiene_ctxt, self);
    }

    fn encode_expn_id(&mut self, expn_id: ExpnId) {
        if expn_id.krate == LOCAL_CRATE {
            // We will only write details for local expansions. Non-local expansions will fetch
            // data from the corresponding crate's metadata.
            // FIXME(#43047) FIXME(#74731) We may eventually want to avoid relying on external
            // metadata from proc-macro crates.
            self.hygiene_ctxt.schedule_expn_data_for_encoding(expn_id);
        }
        expn_id.krate.encode(self);
        expn_id.local_id.encode(self);
    }

    fn encode_span(&mut self, span: Span) {
        match self.span_shorthands.entry(span) {
            Entry::Occupied(o) => {
                // If an offset is smaller than the absolute position, we encode with the offset.
                // This saves space since smaller numbers encode in less bits.
                let last_location = *o.get();
                // This cannot underflow. Metadata is written with increasing position(), so any
                // previously saved offset must be smaller than the current position.
                let offset = self.opaque.position() - last_location;
                if offset < last_location {
                    let needed = bytes_needed(offset);
                    SpanTag::indirect(true, needed as u8).encode(self);
                    self.opaque.write_with(|dest| {
                        *dest = offset.to_le_bytes();
                        needed
                    });
                } else {
                    let needed = bytes_needed(last_location);
                    SpanTag::indirect(false, needed as u8).encode(self);
                    self.opaque.write_with(|dest| {
                        *dest = last_location.to_le_bytes();
                        needed
                    });
                }
            }
            Entry::Vacant(v) => {
                let position = self.opaque.position();
                v.insert(position);
                // Data is encoded with a SpanTag prefix (see below).
                span.data().encode(self);
            }
        }
    }

    fn encode_symbol(&mut self, sym: Symbol) {
        self.encode_symbol_or_byte_symbol(sym.as_u32(), |this| this.emit_str(sym.as_str()));
    }

    fn encode_byte_symbol(&mut self, byte_sym: ByteSymbol) {
        self.encode_symbol_or_byte_symbol(byte_sym.as_u32(), |this| {
            this.emit_byte_str(byte_sym.as_byte_str())
        });
    }
}

fn bytes_needed(n: usize) -> usize {
    (usize::BITS - n.leading_zeros()).div_ceil(u8::BITS) as usize
}

impl<'a, 'tcx> Encodable<EncodeContext<'a, 'tcx>> for SpanData {
    fn encode(&self, s: &mut EncodeContext<'a, 'tcx>) {
        // Don't serialize any `SyntaxContext`s from a proc-macro crate,
        // since we don't load proc-macro dependencies during serialization.
        // This means that any hygiene information from macros used *within*
        // a proc-macro crate (e.g. invoking a macro that expands to a proc-macro
        // definition) will be lost.
        //
        // This can show up in two ways:
        //
        // 1. Any hygiene information associated with identifier of
        // a proc macro (e.g. `#[proc_macro] pub fn $name`) will be lost.
        // Since proc-macros can only be invoked from a different crate,
        // real code should never need to care about this.
        //
        // 2. Using `Span::def_site` or `Span::mixed_site` will not
        // include any hygiene information associated with the definition
        // site. This means that a proc-macro cannot emit a `$crate`
        // identifier which resolves to one of its dependencies,
        // which also should never come up in practice.
        //
        // Additionally, this affects `Span::parent`, and any other
        // span inspection APIs that would otherwise allow traversing
        // the `SyntaxContexts` associated with a span.
        //
        // None of these user-visible effects should result in any
        // cross-crate inconsistencies (getting one behavior in the same
        // crate, and a different behavior in another crate) due to the
        // limited surface that proc-macros can expose.
        //
        // IMPORTANT: If this is ever changed, be sure to update
        // `rustc_span::hygiene::raw_encode_expn_id` to handle
        // encoding `ExpnData` for proc-macro crates.
        let ctxt = if s.is_proc_macro { SyntaxContext::root() } else { self.ctxt };

        if self.is_dummy() {
            let tag = SpanTag::new(SpanKind::Partial, ctxt, 0);
            tag.encode(s);
            if tag.context().is_none() {
                ctxt.encode(s);
            }
            return;
        }

        // The Span infrastructure should make sure that this invariant holds:
        debug_assert!(self.lo <= self.hi);

        if !s.source_file_cache.0.contains(self.lo) {
            let source_map = s.tcx.sess.source_map();
            let source_file_index = source_map.lookup_source_file_idx(self.lo);
            s.source_file_cache =
                (Arc::clone(&source_map.files()[source_file_index]), source_file_index);
        }
        let (ref source_file, source_file_index) = s.source_file_cache;
        debug_assert!(source_file.contains(self.lo));

        if !source_file.contains(self.hi) {
            // Unfortunately, macro expansion still sometimes generates Spans
            // that malformed in this way.
            let tag = SpanTag::new(SpanKind::Partial, ctxt, 0);
            tag.encode(s);
            if tag.context().is_none() {
                ctxt.encode(s);
            }
            return;
        }

        // There are two possible cases here:
        // 1. This span comes from a 'foreign' crate - e.g. some crate upstream of the
        // crate we are writing metadata for. When the metadata for *this* crate gets
        // deserialized, the deserializer will need to know which crate it originally came
        // from. We use `TAG_VALID_SPAN_FOREIGN` to indicate that a `CrateNum` should
        // be deserialized after the rest of the span data, which tells the deserializer
        // which crate contains the source map information.
        // 2. This span comes from our own crate. No special handling is needed - we just
        // write `TAG_VALID_SPAN_LOCAL` to let the deserializer know that it should use
        // our own source map information.
        //
        // If we're a proc-macro crate, we always treat this as a local `Span`.
        // In `encode_source_map`, we serialize foreign `SourceFile`s into our metadata
        // if we're a proc-macro crate.
        // This allows us to avoid loading the dependencies of proc-macro crates: all of
        // the information we need to decode `Span`s is stored in the proc-macro crate.
        let (kind, metadata_index) = if source_file.is_imported() && !s.is_proc_macro {
            // To simplify deserialization, we 'rebase' this span onto the crate it originally came
            // from (the crate that 'owns' the file it references. These rebased 'lo' and 'hi'
            // values are relative to the source map information for the 'foreign' crate whose
            // CrateNum we write into the metadata. This allows `imported_source_files` to binary
            // search through the 'foreign' crate's source map information, using the
            // deserialized 'lo' and 'hi' values directly.
            //
            // All of this logic ensures that the final result of deserialization is a 'normal'
            // Span that can be used without any additional trouble.
            let metadata_index = {
                // Introduce a new scope so that we drop the 'read()' temporary
                match &*source_file.external_src.read() {
                    ExternalSource::Foreign { metadata_index, .. } => *metadata_index,
                    src => panic!("Unexpected external source {src:?}"),
                }
            };

            (SpanKind::Foreign, metadata_index)
        } else {
            // Record the fact that we need to encode the data for this `SourceFile`
            let source_files =
                s.required_source_files.as_mut().expect("Already encoded SourceMap!");
            let (metadata_index, _) = source_files.insert_full(source_file_index);
            let metadata_index: u32 =
                metadata_index.try_into().expect("cannot export more than U32_MAX files");

            (SpanKind::Local, metadata_index)
        };

        // Encode the start position relative to the file start, so we profit more from the
        // variable-length integer encoding.
        // IMPORTANT: if this is ever changed, the public api span hashing must be updated. It
        // currently uses the `hash_spans_as_parentless` option to make sure spans are hashed not
        // relative to their parent, but relative to their file.
        let lo = self.lo - source_file.start_pos;

        // Encode length which is usually less than span.hi and profits more
        // from the variable-length integer encoding that we use.
        let len = self.hi - self.lo;

        let tag = SpanTag::new(kind, ctxt, len.0 as usize);
        tag.encode(s);
        if tag.context().is_none() {
            ctxt.encode(s);
        }
        lo.encode(s);
        if tag.length().is_none() {
            len.encode(s);
        }

        // Encode the index of the `SourceFile` for the span, in order to make decoding faster.
        metadata_index.encode(s);

        if kind == SpanKind::Foreign {
            // This needs to be two lines to avoid holding the `s.source_file_cache`
            // while calling `cnum.encode(s)`
            let cnum = s.source_file_cache.0.cnum;
            cnum.encode(s);
        }
    }
}

impl<'a, 'tcx> Encodable<EncodeContext<'a, 'tcx>> for [u8] {
    fn encode(&self, e: &mut EncodeContext<'a, 'tcx>) {
        Encoder::emit_usize(e, self.len());
        e.emit_raw_bytes(self);
    }
}

impl<'a, 'tcx> TyEncoder<'tcx> for EncodeContext<'a, 'tcx> {
    const CLEAR_CROSS_CRATE: bool = true;

    fn position(&self) -> usize {
        self.opaque.position()
    }

    fn type_shorthands(&mut self) -> &mut FxHashMap<Ty<'tcx>, usize> {
        &mut self.type_shorthands
    }

    fn predicate_shorthands(&mut self) -> &mut FxHashMap<ty::PredicateKind<'tcx>, usize> {
        &mut self.predicate_shorthands
    }

    fn encode_alloc_id(&mut self, alloc_id: &rustc_middle::mir::interpret::AllocId) {
        let (index, _) = self.interpret_allocs.insert_full(*alloc_id);

        index.encode(self);
    }
}

// Shorthand for `$self.$tables.$table.set_some($def_id.index, $self.lazy($value))`, which would
// normally need extra variables to avoid errors about multiple mutable borrows.
macro_rules! record {
    ($self:ident.$tables:ident.$table:ident[$def_id:expr] <- $value:expr, $hcx:ident) => {{
        {
            let value = $value;
            record!($self.$tables.$table[$def_id] <- value, $hcx, value)
        }
    }};
    ($self:ident.$tables:ident.$table:ident[$def_id:expr] <- $value:expr, $hcx:ident, $hashed_value:expr) => {{
        {
            let lazy = $self.lazy($value);
            $self.$tables.$table.set_some_hashed(
                $def_id.index,
                lazy,
                ($def_id, $hashed_value),
                $hcx,
            );
        }
    }};
}

// Shorthand for `$self.$tables.$table.set_some($def_id.index, $self.lazy_array($value))`, which would
// normally need extra variables to avoid errors about multiple mutable borrows.
macro_rules! record_array {
    ($self:ident.$tables:ident.$table:ident[$def_id:expr] <- $value:expr, $hcx:ident) => {
        record_array!($self.$tables.$table[$def_id] <- $value, $hcx, |v| v)
    };
    ($self:ident.$tables:ident.$table:ident[$def_id:expr] <- $value:expr, $hcx:ident, $encode_map:expr) => {{
        {
            let value = $value;
            let mut hasher = $self.$tables.$table.iter_hasher();
            let lazy = $self.lazy_array(hasher.inspect_digest(value, $hcx).map($encode_map));
            $self.$tables.$table.set_some_hashed(
                $def_id.index,
                lazy,
                ($def_id, hasher.finish()),
                $hcx,
            );
        }
    }};
}

macro_rules! record_defaulted_array {
    ($self:ident.$tables:ident.$table:ident[$def_id:expr] <- $value:expr, $hcx:ident) => {
        record_defaulted_array!($self.$tables.$table[$def_id] <- $value, $hcx, |v| v)
    };
    ($self:ident.$tables:ident.$table:ident[$def_id:expr] <- $value:expr, $hcx:ident, $encode_map:expr) => {{
        {
            let value = $value;
            let mut hasher = $self.$tables.$table.iter_hasher();
            let lazy = $self.lazy_array(hasher.inspect_digest(value, $hcx).map($encode_map));
            $self.$tables.$table.set_hashed(
                $def_id.index,
                lazy,
                ($def_id, hasher.finish()),
                $hcx,
            );
        }
    }};
}

/// Stable hashes an iterator while encoding it as a LazyArray.
///
/// The two forms it accepts are
/// ```text
/// hashed_lazy_array!(self, hashed_iterator, hcx)
/// ```
/// and
/// ```text
/// hashed_lazy_array!(self, hashed_iterator, hcx, map)
/// ```
/// `map` maps from hashed value returned from hashed_iterator to the encoded value. This is
/// mostly used to map `LocalDefId`-s to `DefIndex` in the encoded values.
macro_rules! hashed_lazy_array {
    ($self:ident, $values:expr, $hcx:ident, $encode_map:expr) => {{
        {
            let mut hasher = PublicApiHasher::default();
            let array = $self.lazy_array($values.into_iter().map(|v| {
                hasher.digest(&v, $hcx);
                $encode_map(v)
            }));
            Hashed { value: array, hash: hasher.finish($hcx) }
        }
    }};
    ($self:ident, $values:expr, $hcx:ident) => {
        hashed_lazy_array!($self, $values, $hcx, |v| v)
    };
    ($self:ident, $values:expr, $hcx:ident,) => {
        hashed_lazy_array!($self, $values, $hcx, |v| v)
    };
}

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
    fn emit_lazy_distance(&mut self, position: NonZero<usize>) {
        let pos = position.get();
        let distance = match self.lazy_state {
            LazyState::NoNode => bug!("emit_lazy_distance: outside of a metadata node"),
            LazyState::NodeStart(start) => {
                let start = start.get();
                assert!(pos <= start);
                start - pos
            }
            LazyState::Previous(last_pos) => {
                assert!(
                    last_pos <= position,
                    "make sure that the calls to `lazy*` \
                     are in the same order as the metadata fields",
                );
                position.get() - last_pos.get()
            }
        };
        self.lazy_state = LazyState::Previous(NonZero::new(pos).unwrap());
        self.emit_usize(distance);
    }

    fn lazy<T: ParameterizedOverTcx, B: Borrow<T::Value<'tcx>>>(&mut self, value: B) -> LazyValue<T>
    where
        T::Value<'tcx>: Encodable<EncodeContext<'a, 'tcx>>,
    {
        let pos = NonZero::new(self.position()).unwrap();

        assert_eq!(self.lazy_state, LazyState::NoNode);
        self.lazy_state = LazyState::NodeStart(pos);
        value.borrow().encode(self);
        self.lazy_state = LazyState::NoNode;

        assert!(pos.get() <= self.position());

        LazyValue::from_position(pos)
    }

    fn lazy_array<T: ParameterizedOverTcx, I: IntoIterator<Item = B>, B: Borrow<T::Value<'tcx>>>(
        &mut self,
        values: I,
    ) -> LazyArray<T>
    where
        T::Value<'tcx>: Encodable<EncodeContext<'a, 'tcx>>,
    {
        let pos = NonZero::new(self.position()).unwrap();

        assert_eq!(self.lazy_state, LazyState::NoNode);
        self.lazy_state = LazyState::NodeStart(pos);
        let len = values.into_iter().map(|value| value.borrow().encode(self)).count();
        self.lazy_state = LazyState::NoNode;

        assert!(pos.get() <= self.position());

        LazyArray::from_position_and_num_elems(pos, len)
    }

    fn encode_symbol_or_byte_symbol(
        &mut self,
        index: u32,
        emit_str_or_byte_str: impl Fn(&mut Self),
    ) {
        // if symbol/byte symbol is predefined, emit tag and symbol index
        if Symbol::is_predefined(index) {
            self.opaque.emit_u8(SYMBOL_PREDEFINED);
            self.opaque.emit_u32(index);
        } else {
            // otherwise write it as string or as offset to it
            match self.symbol_index_table.entry(index) {
                Entry::Vacant(o) => {
                    self.opaque.emit_u8(SYMBOL_STR);
                    let pos = self.opaque.position();
                    o.insert(pos);
                    emit_str_or_byte_str(self);
                }
                Entry::Occupied(o) => {
                    let x = *o.get();
                    self.emit_u8(SYMBOL_OFFSET);
                    self.emit_usize(x);
                }
            }
        }
    }

    fn encode_def_path_table(&mut self) {
        let defs = self.tcx.definitions();
        if self.is_proc_macro {
            for def_id in std::iter::once(CRATE_DEF_ID)
                .chain(self.tcx.resolutions(()).proc_macros.iter().copied())
            {
                let def_key = self.lazy(defs.def_key(def_id));
                let def_path_hash = defs.def_path_hash(def_id);
                self.tables.def_keys.set_some_unhashed(def_id.local_def_index, def_key);
                self.tables
                    .def_path_hashes
                    .set_unhashed(def_id.local_def_index, def_path_hash.local_hash().as_u64());
            }
        } else {
            for (def_index, def_key, def_path_hash) in defs.enumerated_keys_and_path_hashes() {
                let def_key = self.lazy(def_key);
                self.tables.def_keys.set_some_unhashed(def_index, def_key);
                self.tables
                    .def_path_hashes
                    .set_unhashed(def_index, def_path_hash.local_hash().as_u64());
            }
        }
    }

    fn encode_def_path_hash_map<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyValue<DefPathHashMapRef<'static>>> {
        let value = self
            .lazy(DefPathHashMapRef::BorrowedFromTcx(self.tcx.def_path_hash_to_def_index_map()));
        // an ordered hash of all local defids encapsulates all information contained in a reverse
        // mapping as well.
        let mut hasher = PublicApiHasher::default();
        if hcx.enabled() {
            hasher.digest_iter(self.tcx.iter_local_def_id(), hcx);
        }
        Hashed { hash: hasher.finish(hcx), value }
    }

    fn encode_source_map<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyTable<u32, Option<LazyValue<rustc_span::SourceFile>>>> {
        let source_map = self.tcx.sess.source_map();
        let all_source_files = source_map.files();

        // By replacing the `Option` with `None`, we ensure that we can't
        // accidentally serialize any more `Span`s after the source map encoding
        // is done.
        let required_source_files = self.required_source_files.take().unwrap();

        let mut adapted = TableBuilder::<RDRHashAll<_>, _, _>::default();

        let local_crate_stable_id = self.tcx.stable_crate_id(LOCAL_CRATE);

        // Only serialize `SourceFile`s that were used during the encoding of a `Span`.
        //
        // The order in which we encode source files is important here: the on-disk format for
        // `Span` contains the index of the corresponding `SourceFile`.
        for (on_disk_index, &source_file_index) in required_source_files.iter().enumerate() {
            let source_file = &all_source_files[source_file_index];
            // Don't serialize imported `SourceFile`s, unless we're in a proc-macro crate.
            assert!(!source_file.is_imported() || self.is_proc_macro);

            // At export time we expand all source file paths to absolute paths because
            // downstream compilation sessions can have a different compiler working
            // directory, so relative paths from this or any other upstream crate
            // won't be valid anymore.
            //
            // At this point we also erase the actual on-disk path and only keep
            // the remapped version -- as is necessary for reproducible builds.
            let mut adapted_source_file = (**source_file).clone();

            match source_file.name {
                FileName::Real(ref original_file_name) => {
                    let mut adapted_file_name = original_file_name.clone();
                    adapted_file_name.update_for_crate_metadata();
                    adapted_source_file.name = FileName::Real(adapted_file_name);
                }
                _ => {
                    // expanded code, not from a file
                }
            };

            // We're serializing this `SourceFile` into our crate metadata,
            // so mark it as coming from this crate.
            // This also ensures that we don't try to deserialize the
            // `CrateNum` for a proc-macro dependency - since proc macro
            // dependencies aren't loaded when we deserialize a proc-macro,
            // trying to remap the `CrateNum` would fail.
            if self.is_proc_macro {
                adapted_source_file.cnum = LOCAL_CRATE;
            }

            // Update the `StableSourceFileId` to make sure it incorporates the
            // id of the current crate. This way it will be unique within the
            // crate graph during downstream compilation sessions.
            adapted_source_file.stable_id = StableSourceFileId::from_filename_for_export(
                &adapted_source_file.name,
                local_crate_stable_id,
            );

            let on_disk_index: u32 =
                on_disk_index.try_into().expect("cannot export more than U32_MAX files");
            adapted.set_some_hashed(
                on_disk_index,
                self.lazy(&adapted_source_file),
                {
                    let SourceFile {
                        name,
                        src,
                        src_hash,
                        checksum_hash,
                        external_src,
                        start_pos,
                        normalized_source_len,
                        unnormalized_source_len,
                        lines,
                        multibyte_chars,
                        normalized_pos,
                        stable_id,
                        cnum,
                    } = &adapted_source_file;
                    // not encoded
                    let _ = (src, external_src, start_pos);
                    // hashed as adapted_source_file.lines()
                    let _ = lines;
                    // hashed with stable_id
                    let _ = name;
                    (
                        (src_hash, checksum_hash, normalized_source_len, unnormalized_source_len),
                        (adapted_source_file.lines(), multibyte_chars, stable_id, normalized_pos),
                        cnum,
                    )
                },
                hcx,
            );
        }

        adapted.encode(&mut self.opaque, hcx)
    }

    fn encode_crate_root<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> (LazyValue<CrateRoot>, CrateHashes) {
        let tcx = self.tcx;
        let mut stats: Vec<(&'static str, usize)> = Vec::with_capacity(32);

        macro_rules! stat {
            ($label:literal, $f:expr) => {{
                let orig_pos = self.position();
                let res = $f();
                stats.push(($label, self.position() - orig_pos));
                res
            }};
        }

        // We have already encoded some things. Get their combined size from the current position.
        stats.push(("preamble", self.position()));

        let externally_implementable_items = stat!("externally-implementable-items", || self
            .encode_externally_implementable_items(hcx));

        let (crate_deps, dylib_dependency_formats) = stat!("dep", || (
            self.encode_crate_deps(hcx),
            self.encode_dylib_dependency_formats(hcx)
        ));

        let lib_features = stat!("lib-features", || self.encode_lib_features(hcx));

        let stability_implications =
            stat!("stability-implications", || self.encode_stability_implications(hcx));

        let (lang_items, lang_items_missing) = stat!("lang-items", || {
            (self.encode_lang_items(hcx), self.encode_lang_items_missing(hcx))
        });

        let stripped_cfg_items =
            stat!("stripped-cfg-items", || self.encode_stripped_cfg_items(hcx));

        let diagnostic_items = stat!("diagnostic-items", || self.encode_diagnostic_items(hcx));

        let native_libraries = stat!("native-libs", || self.encode_native_libraries(hcx));

        let foreign_modules = stat!("foreign-modules", || self.encode_foreign_modules(hcx));

        _ = stat!("def-path-table", || self.encode_def_path_table());

        // Encode the def IDs of traits, for rustdoc and diagnostics.
        let traits = stat!("traits", || self.encode_traits(hcx));

        // Encode the def IDs of impls, for coherence checking.
        let impls = stat!("impls", || self.encode_impls(hcx));

        let incoherent_impls = stat!("incoherent-impls", || self.encode_incoherent_impls(hcx));

        _ = stat!("mir", || self.encode_mir(hcx));

        _ = stat!("def-ids", || self.encode_def_ids(hcx));

        let interpret_alloc_index = stat!("interpret-alloc-index", || {
            let mut interpret_alloc_index = Vec::new();
            let mut n = 0;
            let mut hasher = PublicApiHasher::default();
            trace!("beginning to encode alloc ids");
            loop {
                let new_n = self.interpret_allocs.len();
                // if we have found new ids, serialize those, too
                if n == new_n {
                    // otherwise, abort
                    break;
                }
                trace!("encoding {} further alloc ids", new_n - n);
                for idx in n..new_n {
                    let id = self.interpret_allocs[idx];
                    let pos = self.position() as u64;
                    interpret_alloc_index.push(pos);
                    if hcx.enabled() {
                        hasher.digest(tcx.global_alloc(id), hcx);
                    }
                    interpret::specialized_encode_alloc_id(self, tcx, id);
                }
                n = new_n;
            }
            Hashed { value: self.lazy_array(interpret_alloc_index), hash: hasher.finish(hcx) }
        });

        // Encode the proc macro data. This affects `tables`, so we need to do this before we
        // encode the tables. This overwrites def_keys, so it must happen after
        // encode_def_path_table.
        let proc_macro_data = stat!("proc-macro-data", || self.encode_proc_macros(hcx));

        let tables = stat!("tables", || self.tables.encode(&mut self.opaque, hcx));

        let debugger_visualizers =
            stat!("debugger-visualizers", || self.encode_debugger_visualizers(hcx));

        let exportable_items = stat!("exportable-items", || self.encode_exportable_items(hcx));

        let stable_order_of_exportable_impls =
            stat!("exportable-items", || self.encode_stable_order_of_exportable_impls(hcx));

        // Encode exported symbols info. This is prefetched in `encode_metadata`.
        let (exported_non_generic_symbols, exported_generic_symbols) =
            stat!("exported-symbols", || {
                (
                    self.encode_exported_symbols(
                        tcx.exported_non_generic_symbols(LOCAL_CRATE),
                        hcx,
                    ),
                    self.encode_exported_symbols(tcx.exported_generic_symbols(LOCAL_CRATE), hcx),
                )
            });

        // Encode the hygiene data.
        // IMPORTANT: this *must* be the last thing that we encode (other than `SourceMap`). The
        // process of encoding other items (e.g. `optimized_mir`) may cause us to load data from
        // the incremental cache. If this causes us to deserialize a `Span`, then we may load
        // additional `SyntaxContext`s into the global `HygieneData`. Therefore, we need to encode
        // the hygiene data last to ensure that we encode any `SyntaxContext`s that might be used.
        let (syntax_contexts, expn_data, expn_hashes) =
            stat!("hygiene", || self.encode_hygiene(hcx));

        let def_path_hash_map = stat!("def-path-hash-map", || self.encode_def_path_hash_map(hcx));

        // Encode source_map. This needs to be done last, because encoding `Span`s tells us which
        // `SourceFiles` we actually need to encode.
        let source_map = stat!("source-map", || self.encode_source_map(hcx));
        let target_modifiers = stat!("target-modifiers", || self.encode_target_modifiers(hcx));
        let denied_partial_mitigations = stat!("denied-partial-mitigations", || self
            .encode_enabled_denied_partial_mitigations(hcx));

        let attrs = tcx.hir_krate_attrs();
        let crate_root = HashableCrateRoot {
            header: HashableCrateHeader {
                name: tcx.crate_name(LOCAL_CRATE),
                triple: tcx.sess.opts.target_triple.clone(),
                is_proc_macro_crate: proc_macro_data.is_some(),
                is_stub: false,
            },
            extra_filename: tcx.sess.opts.cg.extra_filename.clone(),
            stable_crate_id: tcx.def_path_hash(LOCAL_CRATE.as_def_id()).stable_crate_id(),
            required_panic_strategy: tcx.required_panic_strategy(LOCAL_CRATE),
            panic_in_drop_strategy: tcx.sess.opts.unstable_opts.panic_in_drop,
            edition: tcx.sess.edition(),
            has_global_allocator: tcx.has_global_allocator(LOCAL_CRATE),
            has_alloc_error_handler: tcx.has_alloc_error_handler(LOCAL_CRATE),
            has_panic_handler: tcx.has_panic_handler(LOCAL_CRATE),
            has_default_lib_allocator: find_attr!(attrs, DefaultLibAllocator),
            externally_implementable_items,
            proc_macro_data: NoneIfHashed { value: proc_macro_data },
            debugger_visualizers,
            compiler_builtins: find_attr!(attrs, CompilerBuiltins),
            needs_allocator: find_attr!(attrs, NeedsAllocator),
            needs_panic_runtime: find_attr!(attrs, NeedsPanicRuntime),
            no_builtins: find_attr!(attrs, NoBuiltins),
            panic_runtime: find_attr!(attrs, PanicRuntime),
            profiler_runtime: find_attr!(attrs, ProfilerRuntime),
            symbol_mangling_version: tcx.sess.opts.get_symbol_mangling_version(),

            crate_deps,
            dylib_dependency_formats,
            lib_features,
            stability_implications,
            lang_items,
            diagnostic_items,
            lang_items_missing,
            stripped_cfg_items,
            native_libraries,
            foreign_modules,
            source_map,
            target_modifiers,
            denied_partial_mitigations,
            traits,
            impls,
            incoherent_impls,
            exportable_items,
            stable_order_of_exportable_impls,
            exported_non_generic_symbols,
            exported_generic_symbols,
            interpret_alloc_index,
            tables,
            syntax_contexts,
            expn_data,
            expn_hashes,
            def_path_hash_map,
            specialization_enabled_in: tcx.specialization_enabled_in(LOCAL_CRATE),
        };
        let crate_root = crate_root.into_crate_root(self.tcx, hcx);
        let hashes = crate_root.header.hashes;

        let root = stat!("final", || { self.lazy(crate_root) });

        let total_bytes = self.position();

        let computed_total_bytes: usize = stats.iter().map(|(_, size)| size).sum();
        assert_eq!(total_bytes, computed_total_bytes);

        if tcx.sess.opts.unstable_opts.meta_stats {
            use std::fmt::Write;

            self.opaque.flush();

            // Rewind and re-read all the metadata to count the zero bytes we wrote.
            let pos_before_rewind = self.opaque.file().stream_position().unwrap();
            let mut zero_bytes = 0;
            self.opaque.file().rewind().unwrap();
            let file = std::io::BufReader::new(self.opaque.file());
            for e in file.bytes() {
                if e.unwrap() == 0 {
                    zero_bytes += 1;
                }
            }
            assert_eq!(self.opaque.file().stream_position().unwrap(), pos_before_rewind);

            stats.sort_by_key(|&(_, usize)| usize);
            stats.reverse(); // bigger items first

            let prefix = "meta-stats";
            let perc = |bytes| (bytes * 100) as f64 / total_bytes as f64;

            let section_w = 23;
            let size_w = 10;
            let banner_w = 64;

            // We write all the text into a string and print it with a single
            // `eprint!`. This is an attempt to minimize interleaved text if multiple
            // rustc processes are printing macro-stats at the same time (e.g. with
            // `RUSTFLAGS='-Zmeta-stats' cargo build`). It still doesn't guarantee
            // non-interleaving, though.
            let mut s = String::new();
            _ = writeln!(s, "{prefix} {}", "=".repeat(banner_w));
            _ = writeln!(s, "{prefix} METADATA STATS: {}", tcx.crate_name(LOCAL_CRATE));
            _ = writeln!(s, "{prefix} {:<section_w$}{:>size_w$}", "Section", "Size");
            _ = writeln!(s, "{prefix} {}", "-".repeat(banner_w));
            for (label, size) in stats {
                _ = writeln!(
                    s,
                    "{prefix} {:<section_w$}{:>size_w$} ({:4.1}%)",
                    label,
                    usize_with_underscores(size),
                    perc(size)
                );
            }
            _ = writeln!(s, "{prefix} {}", "-".repeat(banner_w));
            _ = writeln!(
                s,
                "{prefix} {:<section_w$}{:>size_w$} (of which {:.1}% are zero bytes)",
                "Total",
                usize_with_underscores(total_bytes),
                perc(zero_bytes)
            );
            _ = writeln!(s, "{prefix} {}", "=".repeat(banner_w));
            eprint!("{s}");
        }

        (root, hashes)
    }
}

struct AnalyzeAttrState {
    is_exported: bool,
    is_doc_hidden: bool,
}

/// Returns whether an attribute needs to be recorded in metadata, that is, if it's usable and
/// useful in downstream crates. Local-only attributes are an obvious example, but some
/// rustdoc-specific attributes can equally be of use while documenting the current crate only.
///
/// Removing these superfluous attributes speeds up compilation by making the metadata smaller.
///
/// Note: the `is_exported` parameter is used to cache whether the given `DefId` has a public
/// visibility: this is a piece of data that can be computed once per defid, and not once per
/// attribute. Some attributes would only be usable downstream if they are public.
#[inline]
fn analyze_attr(attr: &hir::Attribute, state: &mut AnalyzeAttrState) -> bool {
    let mut should_encode = false;
    if let hir::Attribute::Parsed(p) = attr
        && p.encode_cross_crate() == EncodeCrossCrate::No
    {
        // Attributes not marked encode-cross-crate don't need to be encoded for downstream crates.
    } else if let Some(name) = attr.name()
        && [sym::warn, sym::allow, sym::expect, sym::forbid, sym::deny].contains(&name)
    {
        // Lint attributes don't need to be encoded for downstream crates.
        // FIXME remove this when #152369 is re-merged
    } else if let hir::Attribute::Parsed(AttributeKind::DocComment { .. }) = attr {
        // We keep all doc comments reachable to rustdoc because they might be "imported" into
        // downstream crates if they use `#[doc(inline)]` to copy an item's documentation into
        // their own.
        if state.is_exported {
            should_encode = true;
        }
    } else if let hir::Attribute::Parsed(AttributeKind::Doc(d)) = attr {
        should_encode = true;
        if d.hidden.is_some() {
            state.is_doc_hidden = true;
        }
    } else {
        should_encode = true;
    }
    should_encode
}

fn should_encode_span(def_kind: DefKind) -> bool {
    match def_kind {
        DefKind::Mod
        | DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Variant
        | DefKind::Trait
        | DefKind::TyAlias
        | DefKind::ForeignTy
        | DefKind::TraitAlias
        | DefKind::AssocTy
        | DefKind::TyParam
        | DefKind::ConstParam
        | DefKind::LifetimeParam
        | DefKind::Fn
        | DefKind::Const { .. }
        | DefKind::Static { .. }
        | DefKind::Ctor(..)
        | DefKind::AssocFn
        | DefKind::AssocConst { .. }
        | DefKind::Macro(_)
        | DefKind::ExternCrate
        | DefKind::Use
        | DefKind::AnonConst
        | DefKind::InlineConst
        | DefKind::OpaqueTy
        | DefKind::Field
        | DefKind::Impl { .. }
        | DefKind::Closure
        | DefKind::SyntheticCoroutineBody => true,
        DefKind::ForeignMod | DefKind::GlobalAsm => false,
    }
}

fn should_encode_attrs(def_kind: DefKind) -> bool {
    match def_kind {
        DefKind::Mod
        | DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Variant
        | DefKind::Trait
        | DefKind::TyAlias
        | DefKind::ForeignTy
        | DefKind::TraitAlias
        | DefKind::AssocTy
        | DefKind::Fn
        | DefKind::Const { .. }
        | DefKind::Static { nested: false, .. }
        | DefKind::AssocFn
        | DefKind::AssocConst { .. }
        | DefKind::Macro(_)
        | DefKind::Field
        | DefKind::Impl { .. } => true,
        // Encoding attrs for `Use` items allows `#[doc(hidden)]` on re-exports
        // to be read cross-crate, which is needed for diagnostic path selection
        // in `visible_parent_map`. See #153477.
        DefKind::Use => true,
        // Tools may want to be able to detect their tool lints on
        // closures from upstream crates, too. This is used by
        // https://github.com/model-checking/kani and is not a performance
        // or maintenance issue for us.
        DefKind::Closure => true,
        DefKind::SyntheticCoroutineBody => false,
        DefKind::TyParam
        | DefKind::ConstParam
        | DefKind::Ctor(..)
        | DefKind::ExternCrate
        | DefKind::ForeignMod
        | DefKind::AnonConst
        | DefKind::InlineConst
        | DefKind::OpaqueTy
        | DefKind::LifetimeParam
        | DefKind::Static { nested: true, .. }
        | DefKind::GlobalAsm => false,
    }
}

fn should_encode_expn_that_defined(def_kind: DefKind) -> bool {
    match def_kind {
        DefKind::Mod
        | DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Variant
        | DefKind::Trait
        | DefKind::Impl { .. } => true,
        DefKind::TyAlias
        | DefKind::ForeignTy
        | DefKind::TraitAlias
        | DefKind::AssocTy
        | DefKind::TyParam
        | DefKind::Fn
        | DefKind::Const { .. }
        | DefKind::ConstParam
        | DefKind::Static { .. }
        | DefKind::Ctor(..)
        | DefKind::AssocFn
        | DefKind::AssocConst { .. }
        | DefKind::Macro(_)
        | DefKind::ExternCrate
        | DefKind::Use
        | DefKind::ForeignMod
        | DefKind::AnonConst
        | DefKind::InlineConst
        | DefKind::OpaqueTy
        | DefKind::Field
        | DefKind::LifetimeParam
        | DefKind::GlobalAsm
        | DefKind::Closure
        | DefKind::SyntheticCoroutineBody => false,
    }
}

fn should_encode_visibility(def_kind: DefKind) -> bool {
    match def_kind {
        DefKind::Mod
        | DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Variant
        | DefKind::Trait
        | DefKind::TyAlias
        | DefKind::ForeignTy
        | DefKind::TraitAlias
        | DefKind::AssocTy
        | DefKind::Fn
        | DefKind::Const { .. }
        | DefKind::Static { nested: false, .. }
        | DefKind::Ctor(..)
        | DefKind::AssocFn
        | DefKind::AssocConst { .. }
        | DefKind::Macro(..)
        | DefKind::Field => true,
        DefKind::Use
        | DefKind::ForeignMod
        | DefKind::TyParam
        | DefKind::ConstParam
        | DefKind::LifetimeParam
        | DefKind::AnonConst
        | DefKind::InlineConst
        | DefKind::Static { nested: true, .. }
        | DefKind::OpaqueTy
        | DefKind::GlobalAsm
        | DefKind::Impl { .. }
        | DefKind::Closure
        | DefKind::ExternCrate
        | DefKind::SyntheticCoroutineBody => false,
    }
}

fn should_encode_stability(def_kind: DefKind) -> bool {
    match def_kind {
        DefKind::Mod
        | DefKind::Ctor(..)
        | DefKind::Variant
        | DefKind::Field
        | DefKind::Struct
        | DefKind::AssocTy
        | DefKind::AssocFn
        | DefKind::AssocConst { .. }
        | DefKind::TyParam
        | DefKind::ConstParam
        | DefKind::Static { .. }
        | DefKind::Const { .. }
        | DefKind::Fn
        | DefKind::ForeignMod
        | DefKind::TyAlias
        | DefKind::OpaqueTy
        | DefKind::Enum
        | DefKind::Union
        | DefKind::Impl { .. }
        | DefKind::Trait
        | DefKind::TraitAlias
        | DefKind::Macro(..)
        | DefKind::ForeignTy => true,
        DefKind::Use
        | DefKind::LifetimeParam
        | DefKind::AnonConst
        | DefKind::InlineConst
        | DefKind::GlobalAsm
        | DefKind::Closure
        | DefKind::ExternCrate
        | DefKind::SyntheticCoroutineBody => false,
    }
}

/// Whether we should encode MIR. Return a pair, resp. for CTFE and for LLVM.
///
/// Computing, optimizing and encoding the MIR is a relatively expensive operation.
/// We want to avoid this work when not required. Therefore:
/// - we only compute `mir_for_ctfe` on items with const-eval semantics;
/// - we skip `optimized_mir` for check runs.
/// - we only encode `optimized_mir` that could be generated in other crates, that is, a code that
///   is either generic or has inline hint, and is reachable from the other crates (contained
///   in reachable set).
///
/// Note: Reachable set describes definitions that might be generated or referenced from other
/// crates and it can be used to limit optimized MIR that needs to be encoded. On the other hand,
/// the reachable set doesn't have much to say about which definitions might be evaluated at compile
/// time in other crates, so it cannot be used to omit CTFE MIR. For example, `f` below is
/// unreachable and yet it can be evaluated in other crates:
///
/// ```
/// const fn f() -> usize { 0 }
/// pub struct S { pub a: [usize; f()] }
/// ```
fn should_encode_mir(
    tcx: TyCtxt<'_>,
    reachable_set: &LocalDefIdSet,
    def_id: LocalDefId,
) -> (bool, bool) {
    match tcx.def_kind(def_id) {
        // instance_mir uses mir_for_ctfe rather than optimized_mir for constructors
        DefKind::Ctor(_, _) => (true, false),
        // Constants
        DefKind::AnonConst { .. }
        | DefKind::InlineConst
        | DefKind::AssocConst { .. }
        | DefKind::Const { .. } => (true, false),
        // Coroutines require optimized MIR to compute layout.
        DefKind::Closure if tcx.is_coroutine(def_id.to_def_id()) => (false, true),
        DefKind::SyntheticCoroutineBody => (false, true),
        // Full-fledged functions + closures
        DefKind::AssocFn | DefKind::Fn | DefKind::Closure => {
            let opt = tcx.sess.opts.unstable_opts.always_encode_mir
                || (tcx.sess.opts.output_types.should_codegen()
                    && reachable_set.contains(&def_id)
                    && (tcx.generics_of(def_id).requires_monomorphization(tcx)
                        || tcx.cross_crate_inlinable(def_id)));
            // The function has a `const` modifier or is in a `const trait`.
            let is_const_fn = tcx.is_const_fn(def_id.to_def_id());
            (is_const_fn, opt)
        }
        // The others don't have MIR.
        _ => (false, false),
    }
}

fn should_encode_variances<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId, def_kind: DefKind) -> bool {
    match def_kind {
        DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::OpaqueTy
        | DefKind::Fn
        | DefKind::Ctor(..)
        | DefKind::AssocFn => true,
        DefKind::AssocTy => {
            // Only encode variances for RPITITs (for traits)
            matches!(tcx.opt_rpitit_info(def_id), Some(ty::ImplTraitInTraitData::Trait { .. }))
        }
        DefKind::Mod
        | DefKind::Variant
        | DefKind::Field
        | DefKind::AssocConst { .. }
        | DefKind::TyParam
        | DefKind::ConstParam
        | DefKind::Static { .. }
        | DefKind::Const { .. }
        | DefKind::ForeignMod
        | DefKind::Impl { .. }
        | DefKind::Trait
        | DefKind::TraitAlias
        | DefKind::Macro(..)
        | DefKind::ForeignTy
        | DefKind::Use
        | DefKind::LifetimeParam
        | DefKind::AnonConst
        | DefKind::InlineConst
        | DefKind::GlobalAsm
        | DefKind::Closure
        | DefKind::ExternCrate
        | DefKind::SyntheticCoroutineBody => false,
        DefKind::TyAlias => tcx.type_alias_is_lazy(def_id),
    }
}

fn should_encode_generics(def_kind: DefKind) -> bool {
    match def_kind {
        DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Variant
        | DefKind::Trait
        | DefKind::TyAlias
        | DefKind::ForeignTy
        | DefKind::TraitAlias
        | DefKind::AssocTy
        | DefKind::Fn
        | DefKind::Const { .. }
        | DefKind::Static { .. }
        | DefKind::Ctor(..)
        | DefKind::AssocFn
        | DefKind::AssocConst { .. }
        | DefKind::AnonConst
        | DefKind::InlineConst
        | DefKind::OpaqueTy
        | DefKind::Impl { .. }
        | DefKind::Field
        | DefKind::TyParam
        | DefKind::Closure
        | DefKind::SyntheticCoroutineBody => true,
        DefKind::Mod
        | DefKind::ForeignMod
        | DefKind::ConstParam
        | DefKind::Macro(..)
        | DefKind::Use
        | DefKind::LifetimeParam
        | DefKind::GlobalAsm
        | DefKind::ExternCrate => false,
    }
}

fn should_encode_type(tcx: TyCtxt<'_>, def_id: LocalDefId, def_kind: DefKind) -> bool {
    match def_kind {
        DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Variant
        | DefKind::Ctor(..)
        | DefKind::Field
        | DefKind::Fn
        | DefKind::Const { .. }
        | DefKind::Static { nested: false, .. }
        | DefKind::TyAlias
        | DefKind::ForeignTy
        | DefKind::Impl { .. }
        | DefKind::AssocFn
        | DefKind::AssocConst { .. }
        | DefKind::Closure
        | DefKind::ConstParam
        | DefKind::AnonConst
        | DefKind::InlineConst
        | DefKind::SyntheticCoroutineBody => true,

        DefKind::OpaqueTy => {
            let origin = tcx.local_opaque_ty_origin(def_id);
            if let hir::OpaqueTyOrigin::FnReturn { parent, .. }
            | hir::OpaqueTyOrigin::AsyncFn { parent, .. } = origin
                && let hir::Node::TraitItem(trait_item) = tcx.hir_node_by_def_id(parent)
                && let (_, hir::TraitFn::Required(..)) = trait_item.expect_fn()
            {
                false
            } else {
                true
            }
        }

        DefKind::AssocTy => {
            let assoc_item = tcx.associated_item(def_id);
            match assoc_item.container {
                ty::AssocContainer::InherentImpl | ty::AssocContainer::TraitImpl(_) => true,
                ty::AssocContainer::Trait => assoc_item.defaultness(tcx).has_value(),
            }
        }
        DefKind::TyParam => {
            let hir::Node::GenericParam(param) = tcx.hir_node_by_def_id(def_id) else { bug!() };
            let hir::GenericParamKind::Type { default, .. } = param.kind else { bug!() };
            default.is_some()
        }

        DefKind::Trait
        | DefKind::TraitAlias
        | DefKind::Mod
        | DefKind::ForeignMod
        | DefKind::Macro(..)
        | DefKind::Static { nested: true, .. }
        | DefKind::Use
        | DefKind::LifetimeParam
        | DefKind::GlobalAsm
        | DefKind::ExternCrate => false,
    }
}

fn should_encode_fn_sig(def_kind: DefKind) -> bool {
    match def_kind {
        DefKind::Fn | DefKind::AssocFn | DefKind::Ctor(_, CtorKind::Fn) => true,

        DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Variant
        | DefKind::Field
        | DefKind::Const { .. }
        | DefKind::Static { .. }
        | DefKind::Ctor(..)
        | DefKind::TyAlias
        | DefKind::OpaqueTy
        | DefKind::ForeignTy
        | DefKind::Impl { .. }
        | DefKind::AssocConst { .. }
        | DefKind::Closure
        | DefKind::ConstParam
        | DefKind::AnonConst
        | DefKind::InlineConst
        | DefKind::AssocTy
        | DefKind::TyParam
        | DefKind::Trait
        | DefKind::TraitAlias
        | DefKind::Mod
        | DefKind::ForeignMod
        | DefKind::Macro(..)
        | DefKind::Use
        | DefKind::LifetimeParam
        | DefKind::GlobalAsm
        | DefKind::ExternCrate
        | DefKind::SyntheticCoroutineBody => false,
    }
}

fn should_encode_constness(def_kind: DefKind) -> bool {
    match def_kind {
        DefKind::Fn
        | DefKind::AssocFn
        | DefKind::Closure
        | DefKind::Ctor(_, CtorKind::Fn)
        | DefKind::Impl { of_trait: false } => true,

        DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Field
        | DefKind::Const { .. }
        | DefKind::AssocConst { .. }
        | DefKind::AnonConst
        | DefKind::Static { .. }
        | DefKind::TyAlias
        | DefKind::OpaqueTy
        | DefKind::Impl { .. }
        | DefKind::ForeignTy
        | DefKind::ConstParam
        | DefKind::InlineConst
        | DefKind::AssocTy
        | DefKind::TyParam
        | DefKind::Trait
        | DefKind::TraitAlias
        | DefKind::Mod
        | DefKind::ForeignMod
        | DefKind::Macro(..)
        | DefKind::Use
        | DefKind::LifetimeParam
        | DefKind::GlobalAsm
        | DefKind::ExternCrate
        | DefKind::Ctor(_, CtorKind::Const)
        | DefKind::Variant
        | DefKind::SyntheticCoroutineBody => false,
    }
}

fn should_encode_const(def_kind: DefKind) -> bool {
    match def_kind {
        // FIXME(mgca): should we remove Const and AssocConst here?
        DefKind::Const { .. }
        | DefKind::AssocConst { .. }
        | DefKind::AnonConst
        | DefKind::InlineConst => true,

        DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Variant
        | DefKind::Ctor(..)
        | DefKind::Field
        | DefKind::Fn
        | DefKind::Static { .. }
        | DefKind::TyAlias
        | DefKind::OpaqueTy
        | DefKind::ForeignTy
        | DefKind::Impl { .. }
        | DefKind::AssocFn
        | DefKind::Closure
        | DefKind::ConstParam
        | DefKind::AssocTy
        | DefKind::TyParam
        | DefKind::Trait
        | DefKind::TraitAlias
        | DefKind::Mod
        | DefKind::ForeignMod
        | DefKind::Macro(..)
        | DefKind::Use
        | DefKind::LifetimeParam
        | DefKind::GlobalAsm
        | DefKind::ExternCrate
        | DefKind::SyntheticCoroutineBody => false,
    }
}

fn should_encode_const_of_item<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId, def_kind: DefKind) -> bool {
    // AssocConst ==> assoc item has value
    tcx.is_type_const(def_id)
        && (!matches!(def_kind, DefKind::AssocConst { .. }) || assoc_item_has_value(tcx, def_id))
}

fn assoc_item_has_value<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> bool {
    let assoc_item = tcx.associated_item(def_id);
    match assoc_item.container {
        ty::AssocContainer::InherentImpl | ty::AssocContainer::TraitImpl(_) => true,
        ty::AssocContainer::Trait => assoc_item.defaultness(tcx).has_value(),
    }
}

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
    fn encode_attrs<'h>(&mut self, def_id: LocalDefId, hcx: &mut impl PublicApiHashState<'h>) {
        let tcx = self.tcx;
        let mut state = AnalyzeAttrState {
            is_exported: tcx.effective_visibilities(()).is_exported(def_id),
            is_doc_hidden: false,
        };
        let attr_iter = tcx
            .hir_attrs(tcx.local_def_id_to_hir_id(def_id))
            .iter()
            .filter(|attr| analyze_attr(*attr, &mut state));

        record_array!(self.tables.attributes[def_id.to_def_id()] <- attr_iter, hcx);

        let mut attr_flags = AttrFlags::empty();
        if state.is_doc_hidden {
            attr_flags |= AttrFlags::IS_DOC_HIDDEN;
        }
        self.tables.attr_flags.set_hashed(
            def_id.local_def_index,
            attr_flags,
            (def_id, attr_flags.bits()),
            hcx,
        );
    }

    fn encode_def_ids<'h>(&mut self, hcx: &mut impl PublicApiHashState<'h>) {
        self.encode_info_for_mod(CRATE_DEF_ID, hcx);

        // Proc-macro crates only export proc-macro items, which are looked
        // up using `proc_macro_data`
        if self.is_proc_macro {
            return;
        }

        let tcx = self.tcx;

        for local_id in tcx.iter_local_def_id() {
            let def_id = local_id.to_def_id();
            let def_kind = tcx.def_kind(local_id);
            self.tables.def_kind.set_some_local_hashed(local_id, def_kind, hcx);

            // The `DefCollector` will sometimes create unnecessary `DefId`s
            // for trivial const arguments which are directly lowered to
            // `ConstArgKind::Path`. We never actually access this `DefId`
            // anywhere so we don't need to encode it for other crates.
            if def_kind == DefKind::AnonConst
                && match tcx.hir_node_by_def_id(local_id) {
                    hir::Node::ConstArg(hir::ConstArg { kind, .. }) => match kind {
                        // Skip encoding defs for these as they should not have had a `DefId` created
                        hir::ConstArgKind::Error(..)
                        | hir::ConstArgKind::Struct(..)
                        | hir::ConstArgKind::Array(..)
                        | hir::ConstArgKind::TupleCall(..)
                        | hir::ConstArgKind::Tup(..)
                        | hir::ConstArgKind::Path(..)
                        | hir::ConstArgKind::Literal { .. }
                        | hir::ConstArgKind::Infer(..) => true,
                        hir::ConstArgKind::Anon(..) => false,
                    },
                    _ => false,
                }
            {
                // MGCA doesn't have unnecessary DefIds
                if !tcx.features().min_generic_const_args() {
                    continue;
                }
            }

            if def_kind == DefKind::Field
                && let hir::Node::Field(field) = tcx.hir_node_by_def_id(local_id)
                && let Some(anon) = field.default
            {
                record!(self.tables.default_fields[def_id] <- anon.def_id.to_def_id(), hcx);
            }

            if should_encode_span(def_kind) {
                let def_span = tcx.def_span(local_id);
                record!(self.tables.def_span[def_id] <- def_span, hcx);
            }
            if should_encode_attrs(def_kind) {
                self.encode_attrs(local_id, hcx);
            }
            if should_encode_expn_that_defined(def_kind) {
                record!(self.tables.expn_that_defined[def_id] <- self.tcx.expn_that_defined(def_id), hcx);
            }
            if should_encode_span(def_kind)
                && let Some(ident_span) = tcx.def_ident_span(def_id)
            {
                record!(self.tables.def_ident_span[def_id] <- ident_span, hcx);
            }
            if def_kind.has_codegen_attrs() {
                record!(self.tables.codegen_fn_attrs[def_id] <- self.tcx.codegen_fn_attrs(def_id), hcx);
            }
            if should_encode_visibility(def_kind) {
                let vis = tcx.local_visibility(local_id);
                record!(self.tables.visibility[def_id] <- vis.map_id(|def_id| def_id.local_def_index), hcx, vis);
            }
            if should_encode_stability(def_kind) {
                self.encode_stability(def_id, hcx);
                self.encode_const_stability(def_id, hcx);
                self.encode_default_body_stability(def_id, hcx);
                self.encode_deprecation(def_id, hcx);
            }
            if should_encode_variances(tcx, def_id, def_kind) {
                let v = self.tcx.variances_of(def_id);
                record_array!(self.tables.variances_of[def_id] <- v, hcx);
            }
            if should_encode_fn_sig(def_kind) {
                record!(self.tables.fn_sig[def_id] <- tcx.fn_sig(def_id), hcx);
            }
            if should_encode_generics(def_kind) {
                let g = tcx.generics_of(def_id);
                record!(self.tables.generics_of[def_id] <- g, hcx);
                record!(self.tables.explicit_predicates_of[def_id] <- self.tcx.explicit_predicates_of(def_id), hcx);
                let inferred_outlives = self.tcx.inferred_outlives_of(def_id);
                record_defaulted_array!(self.tables.inferred_outlives_of[def_id] <- inferred_outlives, hcx);

                for param in &g.own_params {
                    if let ty::GenericParamDefKind::Const { has_default: true, .. } = param.kind {
                        let default = self.tcx.const_param_default(param.def_id);
                        record!(self.tables.const_param_default[param.def_id] <- default, hcx);
                    }
                }
            }
            if tcx.is_conditionally_const(def_id) {
                record!(self.tables.const_conditions[def_id] <- self.tcx.const_conditions(def_id), hcx);
            }
            if should_encode_type(tcx, local_id, def_kind) {
                record!(self.tables.type_of[def_id] <- self.tcx.type_of(def_id), hcx);
            }
            if should_encode_constness(def_kind) {
                let constness = self.tcx.constness(def_id);
                self.tables.constness.set_local_hashed(def_id.expect_local(), constness, hcx);
            }
            if let DefKind::Fn | DefKind::AssocFn = def_kind {
                let asyncness = tcx.asyncness(def_id);
                self.tables.asyncness.set_local_hashed(def_id.expect_local(), asyncness, hcx);
                record_array!(self.tables.fn_arg_idents[def_id] <- tcx.fn_arg_idents(def_id), hcx);
            }
            if let Some(name) = tcx.intrinsic(def_id) {
                record!(self.tables.intrinsic[def_id] <- name, hcx);
            }
            if let DefKind::TyParam = def_kind {
                let default = self.tcx.object_lifetime_default(def_id);
                record!(self.tables.object_lifetime_default[def_id] <- default, hcx);
            }
            if let DefKind::Trait = def_kind {
                record!(self.tables.trait_def[def_id] <- self.tcx.trait_def(def_id), hcx);
                record_defaulted_array!(self.tables.explicit_super_predicates_of[def_id] <-
                    self.tcx.explicit_super_predicates_of(def_id).skip_binder(), hcx);
                record_defaulted_array!(self.tables.explicit_implied_predicates_of[def_id] <-
                    self.tcx.explicit_implied_predicates_of(def_id).skip_binder(), hcx);
                let module_children = self.tcx.module_children_local(local_id);
                record_array!(self.tables.module_children_non_reexports[def_id] <-
                    module_children.iter().map(|child| child.res.def_id()), hcx,
                    |def_id| def_id.index);
                if self.tcx.is_const_trait(def_id) {
                    record_defaulted_array!(self.tables.explicit_implied_const_bounds[def_id]
                        <- self.tcx.explicit_implied_const_bounds(def_id).skip_binder(), hcx);
                }
            }
            if let DefKind::TraitAlias = def_kind {
                record!(self.tables.trait_def[def_id] <- self.tcx.trait_def(def_id), hcx);
                record_defaulted_array!(self.tables.explicit_super_predicates_of[def_id] <-
                    self.tcx.explicit_super_predicates_of(def_id).skip_binder(), hcx);
                record_defaulted_array!(self.tables.explicit_implied_predicates_of[def_id] <-
                    self.tcx.explicit_implied_predicates_of(def_id).skip_binder(), hcx);
            }
            if let DefKind::Trait | DefKind::Impl { .. } = def_kind {
                let associated_item_def_ids = self.tcx.associated_item_def_ids(def_id);
                record_array!(self.tables.associated_item_or_field_def_ids[def_id] <-
                    associated_item_def_ids.iter(), hcx, |&def_id| {
                        assert!(def_id.is_local());
                        def_id.index
                    }
                );
                for &def_id in associated_item_def_ids {
                    self.encode_info_for_assoc_item(def_id, hcx);
                }
            }
            if let DefKind::Closure | DefKind::SyntheticCoroutineBody = def_kind
                && let Some(coroutine_kind) = self.tcx.coroutine_kind(def_id)
            {
                self.tables.coroutine_kind.set_local_hashed(
                    def_id.expect_local(),
                    Some(coroutine_kind),
                    hcx,
                )
            }
            if def_kind == DefKind::Closure
                && tcx.type_of(def_id).skip_binder().is_coroutine_closure()
            {
                let coroutine_for_closure = self.tcx.coroutine_for_closure(def_id);
                self.tables.coroutine_for_closure.set_hashed(
                    def_id.index,
                    Some(coroutine_for_closure.into()),
                    (def_id, coroutine_for_closure),
                    hcx,
                );

                // If this async closure has a by-move body, record it too.
                if tcx.needs_coroutine_by_move_body_def_id(coroutine_for_closure) {
                    self.tables.coroutine_by_move_body_def_id.set_hashed(
                        coroutine_for_closure.index,
                        Some(self.tcx.coroutine_by_move_body_def_id(coroutine_for_closure).into()),
                        (
                            coroutine_for_closure,
                            self.tcx.coroutine_by_move_body_def_id(coroutine_for_closure),
                        ),
                        hcx,
                    );
                }
            }
            if let DefKind::Static { .. } = def_kind {
                if !self.tcx.is_foreign_item(def_id) {
                    let data = self.tcx.eval_static_initializer(def_id).unwrap();
                    record!(self.tables.eval_static_initializer[def_id] <- data, hcx);
                }
            }
            if let DefKind::Enum | DefKind::Struct | DefKind::Union = def_kind {
                self.encode_info_for_adt(local_id, hcx);
            }
            if let DefKind::Mod = def_kind {
                self.encode_info_for_mod(local_id, hcx);
            }
            if let DefKind::Macro(_) = def_kind {
                self.encode_info_for_macro(local_id, hcx);
            }
            if let DefKind::TyAlias = def_kind {
                self.tables.type_alias_is_lazy.set_local_hashed(
                    def_id.expect_local(),
                    self.tcx.type_alias_is_lazy(def_id),
                    hcx,
                );
            }
            if let DefKind::OpaqueTy = def_kind {
                self.encode_explicit_item_bounds(def_id, hcx);
                self.encode_explicit_item_self_bounds(def_id, hcx);
                record!(self.tables.opaque_ty_origin[def_id] <- self.tcx.opaque_ty_origin(def_id), hcx);
                self.encode_precise_capturing_args(def_id, hcx);
                if tcx.is_conditionally_const(def_id) {
                    record_defaulted_array!(self.tables.explicit_implied_const_bounds[def_id]
                        <- tcx.explicit_implied_const_bounds(def_id).skip_binder(), hcx);
                }
            }
            if let DefKind::AnonConst = def_kind {
                record!(self.tables.anon_const_kind[def_id] <- self.tcx.anon_const_kind(def_id), hcx);
            }
            if should_encode_const_of_item(self.tcx, def_id, def_kind) {
                record!(self.tables.const_of_item[def_id] <- self.tcx.const_of_item(def_id), hcx);
            }
            if tcx.impl_method_has_trait_impl_trait_tys(def_id)
                && let Ok(table) = self.tcx.collect_return_position_impl_trait_in_trait_tys(def_id)
            {
                record!(self.tables.collect_return_position_impl_trait_in_trait_tys[def_id] <- table, hcx);
            }
            if let DefKind::Impl { .. } | DefKind::Trait = def_kind {
                let table = tcx.associated_types_for_impl_traits_in_trait_or_impl(def_id);
                record!(self.tables.associated_types_for_impl_traits_in_trait_or_impl[def_id] <- table, hcx);
            }
        }

        for (def_id, impls) in &tcx.crate_inherent_impls(()).0.inherent_impls {
            record_defaulted_array!(self.tables.inherent_impls[def_id.to_def_id()] <- impls.iter(), hcx, |def_id| {
                assert!(def_id.is_local());
                def_id.index
            });
        }

        for (def_id, res_map) in &tcx.resolutions(()).doc_link_resolutions {
            record!(self.tables.doc_link_resolutions[def_id.to_def_id()] <- res_map, hcx);
        }

        for (def_id, traits) in &tcx.resolutions(()).doc_link_traits_in_scope {
            record_array!(self.tables.doc_link_traits_in_scope[def_id.to_def_id()] <- traits, hcx);
        }
    }

    fn encode_externally_implementable_items<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyArray<EiiMapEncodedKeyValue>> {
        empty_proc_macro!(self);
        let externally_implementable_items = self.tcx.externally_implementable_items(LOCAL_CRATE);

        hashed_lazy_array!(
            self,
            externally_implementable_items.iter().map(|(foreign_item, (decl, impls))| {
                (
                    *foreign_item,
                    (decl.clone(), impls.iter().map(|(impl_did, i)| (*impl_did, *i)).collect()),
                )
            },),
            hcx
        )
    }

    #[instrument(level = "trace", skip(self, hcx))]
    fn encode_info_for_adt<'h>(
        &mut self,
        local_def_id: LocalDefId,
        hcx: &mut impl PublicApiHashState<'h>,
    ) {
        let def_id = local_def_id.to_def_id();
        let tcx = self.tcx;
        let adt_def = tcx.adt_def(def_id);
        record!(self.tables.repr_options[def_id] <- adt_def.repr(), hcx);

        let params_in_repr = self.tcx.params_in_repr(def_id);
        record!(self.tables.params_in_repr[def_id] <- params_in_repr, hcx);

        if adt_def.is_enum() {
            let module_children = tcx.module_children_local(local_def_id);
            record_array!(self.tables.module_children_non_reexports[def_id] <-
                module_children.iter().map(|child| child.res.def_id()), hcx, |def_id| def_id.index);
        } else {
            // For non-enum, there is only one variant, and its def_id is the adt's.
            debug_assert_eq!(adt_def.variants().len(), 1);
            debug_assert_eq!(adt_def.non_enum_variant().def_id, def_id);
            // Therefore, the loop over variants will encode its fields as the adt's children.
        }

        for (idx, variant) in adt_def.variants().iter_enumerated() {
            let data = VariantData {
                discr: variant.discr,
                idx,
                ctor: variant.ctor.map(|(kind, def_id)| (kind, def_id.index)),
                is_non_exhaustive: variant.is_field_list_non_exhaustive(),
            };
            record!(
                self.tables.variant_data[variant.def_id] <- data,
                hcx,
                (idx, variant.discr, variant.ctor, variant.is_field_list_non_exhaustive())
            );

            record_array!(self.tables.associated_item_or_field_def_ids[variant.def_id] <- variant.fields.iter().map(|f| {
                assert!(f.did.is_local());
                f.did
            }), hcx, |def_id| def_id.index);

            for field in &variant.fields {
                self.tables.safety.set_local_hashed(field.did.expect_local(), field.safety, hcx);
            }

            if let Some((CtorKind::Fn, ctor_def_id)) = variant.ctor {
                let fn_sig = tcx.fn_sig(ctor_def_id);
                // FIXME only encode signature for ctor_def_id
                record!(self.tables.fn_sig[variant.def_id] <- fn_sig, hcx);
            }
        }

        if let Some(destructor) = tcx.adt_destructor(local_def_id) {
            record!(self.tables.adt_destructor[def_id] <- destructor, hcx);
        }

        if let Some(destructor) = tcx.adt_async_destructor(local_def_id) {
            record!(self.tables.adt_async_destructor[def_id] <- destructor, hcx);
        }
    }

    #[instrument(level = "debug", skip(self, hcx))]
    fn encode_info_for_mod<'h>(
        &mut self,
        local_def_id: LocalDefId,
        hcx: &mut impl PublicApiHashState<'h>,
    ) {
        let tcx = self.tcx;
        let def_id = local_def_id.to_def_id();

        // If we are encoding a proc-macro crates, `encode_info_for_mod` will
        // only ever get called for the crate root. We still want to encode
        // the crate root for consistency with other crates (some of the resolver
        // code uses it). However, we skip encoding anything relating to child
        // items - we encode information about proc-macros later on.
        if self.is_proc_macro {
            // Encode this here because we don't do it in encode_def_ids.
            record!(self.tables.expn_that_defined[def_id] <- tcx.expn_that_defined(local_def_id), hcx);
        } else {
            let module_children = tcx.module_children_local(local_def_id);

            record_array!(self.tables.module_children_non_reexports[def_id] <-
                module_children.iter().filter(|child| child.reexport_chain.is_empty())
                    .map(|child| child.res.def_id()), hcx, |def_id| def_id.index);

            record_defaulted_array!(self.tables.module_children_reexports[def_id] <-
                module_children.iter().filter(|child| !child.reexport_chain.is_empty()), hcx);

            let ambig_module_children = tcx
                .resolutions(())
                .ambig_module_children
                .get(&local_def_id)
                .map_or_default(|v| &v[..]);
            record_defaulted_array!(self.tables.ambig_module_children[def_id] <-
                ambig_module_children, hcx);
        }
    }

    fn encode_explicit_item_bounds<'h>(
        &mut self,
        def_id: DefId,
        hcx: &mut impl PublicApiHashState<'h>,
    ) {
        debug!("EncodeContext::encode_explicit_item_bounds({:?})", def_id);
        let bounds = self.tcx.explicit_item_bounds(def_id).skip_binder();
        record_defaulted_array!(self.tables.explicit_item_bounds[def_id] <- bounds, hcx);
    }

    fn encode_explicit_item_self_bounds<'h>(
        &mut self,
        def_id: DefId,
        hcx: &mut impl PublicApiHashState<'h>,
    ) {
        debug!("EncodeContext::encode_explicit_item_self_bounds({:?})", def_id);
        let bounds = self.tcx.explicit_item_self_bounds(def_id).skip_binder();
        record_defaulted_array!(self.tables.explicit_item_self_bounds[def_id] <- bounds, hcx);
    }

    #[instrument(level = "debug", skip(self, hcx))]
    fn encode_info_for_assoc_item<'h>(
        &mut self,
        def_id: DefId,
        hcx: &mut impl PublicApiHashState<'h>,
    ) {
        let tcx = self.tcx;
        let item = tcx.associated_item(def_id);

        if matches!(item.container, AssocContainer::Trait | AssocContainer::TraitImpl(_)) {
            self.tables.defaultness.set_local_hashed(
                def_id.expect_local(),
                item.defaultness(tcx),
                hcx,
            );
        }

        record!(self.tables.assoc_container[def_id] <- item.container, hcx);

        if let AssocContainer::Trait = item.container
            && item.is_type()
        {
            self.encode_explicit_item_bounds(def_id, hcx);
            self.encode_explicit_item_self_bounds(def_id, hcx);
            if tcx.is_conditionally_const(def_id) {
                record_defaulted_array!(self.tables.explicit_implied_const_bounds[def_id]
                    <- self.tcx.explicit_implied_const_bounds(def_id).skip_binder(), hcx);
            }
        }
        if let ty::AssocKind::Type { data: ty::AssocTypeData::Rpitit(rpitit_info) } = item.kind {
            record!(self.tables.opt_rpitit_info[def_id] <- rpitit_info, hcx);
            if matches!(rpitit_info, ty::ImplTraitInTraitData::Trait { .. }) {
                record_array!(
                    self.tables.assumed_wf_types_for_rpitit[def_id]
                        <- self.tcx.assumed_wf_types_for_rpitit(def_id), hcx
                );
                self.encode_precise_capturing_args(def_id, hcx);
            }
        }
    }

    fn encode_precise_capturing_args<'h>(
        &mut self,
        def_id: DefId,
        hcx: &mut impl PublicApiHashState<'h>,
    ) {
        let Some(precise_capturing_args) = self.tcx.rendered_precise_capturing_args(def_id) else {
            return;
        };

        record_array!(self.tables.rendered_precise_capturing_args[def_id] <- precise_capturing_args, hcx);
    }

    fn encode_mir<'h>(&mut self, hcx: &mut impl PublicApiHashState<'h>) {
        if self.is_proc_macro {
            return;
        }

        let tcx = self.tcx;
        let reachable_set = tcx.reachable_set(());

        let keys_and_jobs = tcx.mir_keys(()).iter().filter_map(|&def_id| {
            let (encode_const, encode_opt) = should_encode_mir(tcx, reachable_set, def_id);
            if encode_const || encode_opt { Some((def_id, encode_const, encode_opt)) } else { None }
        });
        for (def_id, encode_const, encode_opt) in keys_and_jobs {
            debug_assert!(encode_const || encode_opt);

            debug!("EntryBuilder::encode_mir({:?})", def_id);
            if encode_opt {
                record!(self.tables.optimized_mir[def_id.to_def_id()] <- tcx.optimized_mir(def_id), hcx);
                self.tables.cross_crate_inlinable.set_local_hashed(
                    def_id,
                    self.tcx.cross_crate_inlinable(def_id),
                    hcx,
                );
                record!(self.tables.closure_saved_names_of_captured_variables[def_id.to_def_id()]
                    <- tcx.closure_saved_names_of_captured_variables(def_id), hcx);

                if self.tcx.is_coroutine(def_id.to_def_id())
                    && let Some(witnesses) = tcx.mir_coroutine_witnesses(def_id)
                {
                    record!(self.tables.mir_coroutine_witnesses[def_id.to_def_id()] <- witnesses, hcx);
                }
            }
            let mut is_trivial = false;
            if encode_const {
                if let Some((val, ty)) = tcx.trivial_const(def_id) {
                    is_trivial = true;
                    record!(self.tables.trivial_const[def_id.to_def_id()] <- (val, ty), hcx);
                } else {
                    is_trivial = false;
                    record!(self.tables.mir_for_ctfe[def_id.to_def_id()] <- tcx.mir_for_ctfe(def_id), hcx);
                }

                // FIXME(generic_const_exprs): this feels wrong to have in `encode_mir`
                let abstract_const = tcx.thir_abstract_const(def_id);
                if let Ok(Some(abstract_const)) = abstract_const {
                    record!(self.tables.thir_abstract_const[def_id.to_def_id()] <- abstract_const, hcx);
                }

                if should_encode_const(tcx.def_kind(def_id)) {
                    let qualifs = tcx.mir_const_qualif(def_id);
                    record!(self.tables.mir_const_qualif[def_id.to_def_id()] <- qualifs, hcx);
                    let body = tcx.hir_maybe_body_owned_by(def_id);
                    if let Some(body) = body {
                        let const_data = rendered_const(self.tcx, &body, def_id);
                        record!(self.tables.rendered_const[def_id.to_def_id()] <- &const_data, hcx);
                    }
                }
            }
            if !is_trivial {
                record!(self.tables.promoted_mir[def_id.to_def_id()] <- tcx.promoted_mir(def_id), hcx);
            }

            if self.tcx.is_coroutine(def_id.to_def_id())
                && let Some(witnesses) = tcx.mir_coroutine_witnesses(def_id)
            {
                record!(self.tables.mir_coroutine_witnesses[def_id.to_def_id()] <- witnesses, hcx);
            }
        }

        // Encode all the deduced parameter attributes for everything that has MIR, even for items
        // that can't be inlined. But don't if we aren't optimizing in non-incremental mode, to
        // save the query traffic.
        if tcx.sess.opts.output_types.should_codegen()
            && tcx.sess.opts.optimize != OptLevel::No
            && tcx.sess.opts.incremental.is_none()
        {
            for &local_def_id in tcx.mir_keys(()) {
                if let DefKind::AssocFn | DefKind::Fn = tcx.def_kind(local_def_id) {
                    record_array!(self.tables.deduced_param_attrs[local_def_id.to_def_id()] <-
                        self.tcx.deduced_param_attrs(local_def_id.to_def_id()), hcx);
                }
            }
        }
    }

    #[instrument(level = "debug", skip(self, hcx))]
    fn encode_stability<'h>(&mut self, def_id: DefId, hcx: &mut impl PublicApiHashState<'h>) {
        // The query lookup can take a measurable amount of time in crates with many items. Check if
        // the stability attributes are even enabled before using their queries.
        if self.feat.staged_api() || self.tcx.sess.opts.unstable_opts.force_unstable_if_unmarked {
            if let Some(stab) = self.tcx.lookup_stability(def_id) {
                record!(self.tables.lookup_stability[def_id] <- stab, hcx)
            }
        }
    }

    #[instrument(level = "debug", skip(self, hcx))]
    fn encode_const_stability<'h>(&mut self, def_id: DefId, hcx: &mut impl PublicApiHashState<'h>) {
        // The query lookup can take a measurable amount of time in crates with many items. Check if
        // the stability attributes are even enabled before using their queries.
        if self.feat.staged_api() || self.tcx.sess.opts.unstable_opts.force_unstable_if_unmarked {
            if let Some(stab) = self.tcx.lookup_const_stability(def_id) {
                record!(self.tables.lookup_const_stability[def_id] <- stab, hcx)
            }
        }
    }

    #[instrument(level = "debug", skip(self, hcx))]
    fn encode_default_body_stability<'h>(
        &mut self,
        def_id: DefId,
        hcx: &mut impl PublicApiHashState<'h>,
    ) {
        // The query lookup can take a measurable amount of time in crates with many items. Check if
        // the stability attributes are even enabled before using their queries.
        if self.feat.staged_api() || self.tcx.sess.opts.unstable_opts.force_unstable_if_unmarked {
            if let Some(stab) = self.tcx.lookup_default_body_stability(def_id) {
                record!(self.tables.lookup_default_body_stability[def_id] <- stab, hcx)
            }
        }
    }

    #[instrument(level = "debug", skip(self, hcx))]
    fn encode_deprecation<'h>(&mut self, def_id: DefId, hcx: &mut impl PublicApiHashState<'h>) {
        if let Some(depr) = self.tcx.lookup_deprecation(def_id) {
            record!(self.tables.lookup_deprecation_entry[def_id] <- depr, hcx);
        }
    }

    #[instrument(level = "debug", skip(self, hcx))]
    fn encode_info_for_macro<'h>(
        &mut self,
        def_id: LocalDefId,
        hcx: &mut impl PublicApiHashState<'h>,
    ) {
        let tcx = self.tcx;

        let (_, macro_def, _) = tcx.hir_expect_item(def_id).expect_macro();
        self.tables.is_macro_rules.set_local_hashed(def_id, macro_def.macro_rules, hcx);
        record!(self.tables.macro_definition[def_id.to_def_id()] <- &*macro_def.body, hcx);
    }

    fn encode_native_libraries<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyArray<NativeLib>> {
        empty_proc_macro!(self);
        let used_libraries = self.tcx.native_libraries(LOCAL_CRATE);
        hashed_lazy_array!(self, used_libraries, hcx)
    }

    fn encode_foreign_modules<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyArray<ForeignModule>> {
        empty_proc_macro!(self);
        let foreign_modules = self.tcx.foreign_modules(LOCAL_CRATE);
        hashed_lazy_array!(self, foreign_modules.iter().map(|(_, m)| m), hcx)
    }

    fn encode_hygiene<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> (Hashed<SyntaxContextTable>, Hashed<ExpnDataTable>, Hashed<ExpnHashTable>) {
        let mut syntax_contexts: TableBuilder<RDRHashAll<_>, _, _> = Default::default();
        let mut expn_data_table: TableBuilder<RDRHashAll<_>, _, _> = Default::default();
        let mut expn_hash_table: TableBuilder<RDRHashNone<_>, _, _> = Default::default();
        let hcx = RefCell::new(hcx);

        self.hygiene_ctxt.encode(
            &mut (&mut *self, &mut syntax_contexts, &mut expn_data_table, &mut expn_hash_table),
            |(this, syntax_contexts, _, _), index, ctxt_data| {
                syntax_contexts.set_some_hashed(
                    index,
                    this.lazy(ctxt_data),
                    ctxt_data,
                    *hcx.borrow_mut(),
                );
            },
            |(this, _, expn_data_table, expn_hash_table), index, expn_data, hash| {
                if let Some(index) = index.as_local() {
                    expn_data_table.set_some_hashed(
                        index.as_raw(),
                        this.lazy(expn_data),
                        index,
                        *hcx.borrow_mut(),
                    );
                    // don't need to hash it since it is already included with `expn_data_table`
                    expn_hash_table.set_some_unhashed(index.as_raw(), this.lazy(hash));
                }
            },
        );
        let hcx = hcx.into_inner();

        (
            syntax_contexts.encode(&mut self.opaque, hcx),
            expn_data_table.encode(&mut self.opaque, hcx),
            expn_hash_table.encode(&mut self.opaque, hcx),
        )
    }

    fn encode_proc_macros<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Option<ProcMacroData> {
        let is_proc_macro = self.tcx.crate_types().contains(&CrateType::ProcMacro);
        if is_proc_macro {
            let tcx = self.tcx;
            let proc_macro_decls_static = tcx.proc_macro_decls_static(()).unwrap().local_def_index;
            let stability = tcx.lookup_stability(CRATE_DEF_ID);
            let macros =
                self.lazy_array(tcx.resolutions(()).proc_macros.iter().map(|p| p.local_def_index));
            for (i, span) in self.tcx.sess.proc_macro_quoted_spans() {
                let encoded_span = self.lazy(span);
                self.tables.proc_macro_quoted_spans.set_some_hashed(i, encoded_span, span, hcx);
            }

            self.tables.def_kind.set_some_local_hashed(CRATE_DEF_ID, DefKind::Mod, hcx);
            record!(self.tables.def_span[LOCAL_CRATE.as_def_id()] <- tcx.def_span(LOCAL_CRATE.as_def_id()), hcx);
            self.encode_attrs(LOCAL_CRATE.as_def_id().expect_local(), hcx);
            let vis = tcx.local_visibility(CRATE_DEF_ID);
            record!(self.tables.visibility[LOCAL_CRATE.as_def_id()] <- vis.map_id(|def_id| def_id.local_def_index), hcx, vis);
            if let Some(stability) = stability {
                record!(self.tables.lookup_stability[LOCAL_CRATE.as_def_id()] <- stability, hcx);
            }
            self.encode_deprecation(LOCAL_CRATE.as_def_id(), hcx);
            if let Some(res_map) = tcx.resolutions(()).doc_link_resolutions.get(&CRATE_DEF_ID) {
                record!(self.tables.doc_link_resolutions[LOCAL_CRATE.as_def_id()] <- res_map, hcx);
            }
            if let Some(traits) = tcx.resolutions(()).doc_link_traits_in_scope.get(&CRATE_DEF_ID) {
                record_array!(self.tables.doc_link_traits_in_scope[LOCAL_CRATE.as_def_id()] <- traits, hcx);
            }

            // Normally, this information is encoded when we walk the items
            // defined in this crate. However, we skip doing that for proc-macro crates,
            // so we manually encode just the information that we need
            for &proc_macro in &tcx.resolutions(()).proc_macros {
                let id = proc_macro;
                let proc_macro = tcx.local_def_id_to_hir_id(proc_macro);
                let mut name = tcx.hir_name(proc_macro);
                let span = tcx.hir_span(proc_macro);
                // Proc-macros may have attributes like `#[allow_internal_unstable]`,
                // so downstream crates need access to them.
                let attrs = tcx.hir_attrs(proc_macro);
                let macro_kind = if find_attr!(attrs, ProcMacro) {
                    MacroKind::Bang
                } else if find_attr!(attrs, ProcMacroAttribute) {
                    MacroKind::Attr
                } else if let Some(trait_name) =
                    find_attr!(attrs, ProcMacroDerive { trait_name, ..} => trait_name)
                {
                    name = *trait_name;
                    MacroKind::Derive
                } else {
                    bug!("Unknown proc-macro type for item {:?}", id);
                };

                let mut def_key = self.tcx.hir_def_key(id);
                def_key.disambiguated_data.data = DefPathData::MacroNs(name);

                let def_id = id.to_def_id();
                self.tables.def_kind.set_some_local_hashed(
                    def_id.expect_local(),
                    DefKind::Macro(macro_kind.into()),
                    hcx,
                );
                self.encode_attrs(id, hcx);
                record!(self.tables.def_keys[def_id] <- def_key, hcx, def_id);
                record!(self.tables.def_ident_span[def_id] <- span, hcx);
                record!(self.tables.def_span[def_id] <- span, hcx);
                record!(self.tables.visibility[def_id] <- Visibility::Public, hcx, Visibility::<DefId>::Public);
                if let Some(stability) = stability {
                    record!(self.tables.lookup_stability[def_id] <- stability, hcx);
                }
            }

            Some(ProcMacroData { proc_macro_decls_static, stability, macros })
        } else {
            None
        }
    }

    fn encode_debugger_visualizers<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyArray<DebuggerVisualizerFile>> {
        empty_proc_macro!(self);

        hashed_lazy_array!(
            self,
            self.tcx
                .debugger_visualizers(LOCAL_CRATE)
                .iter()
                // Erase the path since it may contain privacy sensitive data
                // that we don't want to end up in crate metadata.
                // The path is only needed for the local crate because of
                // `--emit dep-info`.
                .map(DebuggerVisualizerFile::path_erased),
            hcx,
        )
    }

    fn encode_crate_deps<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyArray<CrateDep>> {
        empty_proc_macro!(self);

        let deps = crate_deps(self.tcx).collect::<Vec<_>>();

        {
            // Sanity-check the crate numbers
            let mut expected_cnum = 1;
            for &(n, _) in &deps {
                assert_eq!(n, CrateNum::new(expected_cnum));
                expected_cnum += 1;
            }
        }

        // We're just going to write a list of crate 'name-hash-version's, with
        // the assumption that they are numbered 1 to n.
        // FIXME (#2166): This is not nearly enough to support correct versioning
        // but is enough to get transitive crate dependencies working.
        hashed_lazy_array!(self, deps.iter().map(|(_, dep)| dep), hcx)
    }

    fn encode_target_modifiers<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyArray<TargetModifier>> {
        empty_proc_macro!(self);
        let tcx = self.tcx;
        hashed_lazy_array!(self, tcx.sess.opts.gather_target_modifiers(), hcx)
    }

    fn encode_enabled_denied_partial_mitigations<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyArray<DeniedPartialMitigation>> {
        empty_proc_macro!(self);
        let tcx = self.tcx;
        hashed_lazy_array!(self, tcx.sess.gather_enabled_denied_partial_mitigations(), hcx)
    }

    fn encode_lib_features<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyArray<(Symbol, FeatureStability)>> {
        empty_proc_macro!(self);
        let tcx = self.tcx;
        let lib_features = tcx.lib_features(LOCAL_CRATE);
        hashed_lazy_array!(self, lib_features.to_sorted_vec(), hcx)
    }

    fn encode_stability_implications<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyArray<(Symbol, Symbol)>> {
        empty_proc_macro!(self);
        let tcx = self.tcx;
        let implications = tcx.stability_implications(LOCAL_CRATE);
        let sorted = implications.to_sorted_stable_ord();
        hashed_lazy_array!(self, sorted.into_iter().map(|(k, v)| (*k, *v)), hcx)
    }

    fn encode_diagnostic_items<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyArray<(Symbol, DefIndex)>> {
        empty_proc_macro!(self);
        let tcx = self.tcx;
        let diagnostic_items = &tcx.diagnostic_items(LOCAL_CRATE).name_to_id;
        hashed_lazy_array!(
            self,
            diagnostic_items.iter().map(|(name, id)| (*name, *id)),
            hcx,
            |(name, def_id): (Symbol, DefId)| (name, def_id.index)
        )
    }

    fn encode_lang_items<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyArray<(DefIndex, LangItem)>> {
        empty_proc_macro!(self);
        let lang_items = self.tcx.lang_items().iter();
        hashed_lazy_array!(
            self,
            lang_items.filter(|(_lang_item, def_id)| { def_id.is_local() }),
            hcx,
            |(lang_item, id): (LangItem, DefId)| (id.index, lang_item)
        )
    }

    fn encode_lang_items_missing<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyArray<LangItem>> {
        empty_proc_macro!(self);
        let tcx = self.tcx;
        hashed_lazy_array!(self, &tcx.lang_items().missing, hcx)
    }

    fn encode_stripped_cfg_items<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyArray<StrippedCfgItem<DefIndex>>> {
        hashed_lazy_array!(
            self,
            self.tcx.stripped_cfg_items(LOCAL_CRATE),
            hcx,
            |item: &StrippedCfgItem| item.clone().map_scope_id(|def_id| def_id.index)
        )
    }

    fn encode_traits<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyArray<DefIndex>> {
        empty_proc_macro!(self);
        hashed_lazy_array!(
            self,
            self.tcx.traits(LOCAL_CRATE).iter().copied(),
            hcx,
            |def_id: DefId| def_id.index
        )
    }

    /// Encodes an index, mapping each trait to its (local) implementations.
    #[instrument(level = "debug", skip(self, hcx))]
    fn encode_impls<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyArray<TraitImpls>> {
        empty_proc_macro!(self);
        let tcx = self.tcx;
        let mut trait_impls_map: FxIndexMap<DefId, Vec<(LocalDefId, Option<SimplifiedType>)>> =
            FxIndexMap::default();

        for id in tcx.hir_free_items() {
            let DefKind::Impl { of_trait } = tcx.def_kind(id.owner_id) else {
                continue;
            };
            let def_id = id.owner_id.to_def_id();

            if of_trait {
                let header = tcx.impl_trait_header(def_id);
                record!(self.tables.impl_trait_header[def_id] <- header, hcx);

                self.tables.defaultness.set_local_hashed(
                    def_id.expect_local(),
                    tcx.defaultness(def_id),
                    hcx,
                );

                let trait_ref = header.trait_ref.instantiate_identity().skip_norm_wip();
                let simplified_self_ty = fast_reject::simplify_type(
                    self.tcx,
                    trait_ref.self_ty(),
                    TreatParams::InstantiateWithInfer,
                );
                trait_impls_map
                    .entry(trait_ref.def_id)
                    .or_default()
                    .push((id.owner_id.def_id, simplified_self_ty));

                let trait_def = tcx.trait_def(trait_ref.def_id);
                if let Ok(mut an) = trait_def.ancestors(tcx, def_id)
                    && let Some(specialization_graph::Node::Impl(parent)) = an.nth(1)
                {
                    self.tables.impl_parent.set_hashed(
                        def_id.index,
                        Some(parent.into()),
                        (def_id, parent),
                        hcx,
                    );
                }

                // if this is an impl of `CoerceUnsized`, create its
                // "unsized info", else just store None
                if tcx.is_lang_item(trait_ref.def_id, LangItem::CoerceUnsized) {
                    let coerce_unsized_info = tcx.coerce_unsized_info(def_id).unwrap();
                    record!(self.tables.coerce_unsized_info[def_id] <- coerce_unsized_info, hcx);
                }
            }
        }

        let mut hasher = PublicApiHasher::default();
        let trait_impls: Vec<_> = trait_impls_map
            .iter()
            .map(|(trait_def_id, impls)| {
                hasher.digest(trait_def_id, hcx);
                hasher.digest(impls, hcx);
                TraitImpls {
                    trait_id: (trait_def_id.krate.as_u32(), trait_def_id.index),
                    impls: self.lazy_array(impls.iter().map(|(id, ty)| (id.local_def_index, *ty))),
                }
            })
            .collect();

        Hashed { value: self.lazy_array(trait_impls), hash: hasher.finish(hcx) }
    }

    #[instrument(level = "debug", skip(self, hcx))]
    fn encode_incoherent_impls<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyArray<IncoherentImpls>> {
        empty_proc_macro!(self);
        let tcx = self.tcx;

        let mut hasher = PublicApiHasher::default();
        let all_impls: Vec<_> = tcx
            .crate_inherent_impls(())
            .0
            .incoherent_impls
            .iter()
            .map(|(&simp, impls)| IncoherentImpls {
                self_ty: self.lazy({
                    hasher.digest(simp, hcx);
                    simp
                }),
                impls: self.lazy_array({
                    hasher.digest(impls, hcx);
                    impls.iter().map(|def_id| def_id.local_def_index)
                }),
            })
            .collect();

        Hashed { value: self.lazy_array(&all_impls), hash: hasher.finish(hcx) }
    }

    fn encode_exportable_items<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyArray<DefIndex>> {
        empty_proc_macro!(self);
        hashed_lazy_array!(
            self,
            self.tcx.exportable_items(LOCAL_CRATE).iter().copied(),
            hcx,
            |def_id: DefId| { def_id.index }
        )
    }

    fn encode_stable_order_of_exportable_impls<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyArray<(DefIndex, usize)>> {
        empty_proc_macro!(self);
        let stable_order_of_exportable_impls =
            self.tcx.stable_order_of_exportable_impls(LOCAL_CRATE);
        hashed_lazy_array!(
            self,
            stable_order_of_exportable_impls.iter().map(|(id, idx)| (*id, *idx)),
            hcx,
            |(def_id, idx): (DefId, usize)| (def_id.index, idx)
        )
    }

    // Encodes all symbols exported from this crate into the metadata.
    //
    // This pass is seeded off the reachability list calculated in the
    // middle::reachable module but filters out items that either don't have a
    // symbol associated with them (they weren't translated) or if they're an FFI
    // definition (as that's not defined in this crate).
    fn encode_exported_symbols<'h>(
        &mut self,
        exported_symbols: &[(ExportedSymbol<'tcx>, SymbolExportInfo)],
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyArray<(ExportedSymbol<'static>, SymbolExportInfo)>> {
        empty_proc_macro!(self);

        hashed_lazy_array!(self, exported_symbols, hcx)
    }

    fn encode_dylib_dependency_formats<'h>(
        &mut self,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> Hashed<LazyArray<Option<LinkagePreference>>> {
        empty_proc_macro!(self);
        let arr = dylib_dependency_formats(self.tcx).into_iter().flatten();
        hashed_lazy_array!(self, arr.map(|(_id, slot)| slot), hcx)
    }
}

/// Used to prefetch queries which will be needed later by metadata encoding.
/// Only a subset of the queries are actually prefetched to keep this code smaller.
fn prefetch_mir(tcx: TyCtxt<'_>) {
    if !tcx.sess.opts.output_types.should_codegen() {
        // We won't emit MIR, so don't prefetch it.
        return;
    }

    let reachable_set = tcx.reachable_set(());
    par_for_each_in(tcx.mir_keys(()), |&&def_id| {
        if tcx.is_trivial_const(def_id) {
            return;
        }
        let (encode_const, encode_opt) = should_encode_mir(tcx, reachable_set, def_id);

        if encode_const {
            tcx.ensure_done().mir_for_ctfe(def_id);
        }
        if encode_opt {
            tcx.ensure_done().optimized_mir(def_id);
        }
        if encode_opt || encode_const {
            tcx.ensure_done().promoted_mir(def_id);
        }
    })
}

// NOTE(eddyb) The following comment was preserved for posterity, even
// though it's no longer relevant as EBML (which uses nested & tagged
// "documents") was replaced with a scheme that can't go out of bounds.
//
// And here we run into yet another obscure archive bug: in which metadata
// loaded from archives may have trailing garbage bytes. Awhile back one of
// our tests was failing sporadically on the macOS 64-bit builders (both nopt
// and opt) by having ebml generate an out-of-bounds panic when looking at
// metadata.
//
// Upon investigation it turned out that the metadata file inside of an rlib
// (and ar archive) was being corrupted. Some compilations would generate a
// metadata file which would end in a few extra bytes, while other
// compilations would not have these extra bytes appended to the end. These
// extra bytes were interpreted by ebml as an extra tag, so they ended up
// being interpreted causing the out-of-bounds.
//
// The root cause of why these extra bytes were appearing was never
// discovered, and in the meantime the solution we're employing is to insert
// the length of the metadata to the start of the metadata. Later on this
// will allow us to slice the metadata to the precise length that we just
// generated regardless of trailing bytes that end up in it.

pub struct EncodedMetadata {
    // The declaration order matters because `full_metadata` should be dropped
    // before `_temp_dir`.
    full_metadata: Option<Mmap>,
    // This is an optional stub metadata containing only the crate header.
    // The header should be very small, so we load it directly into memory.
    stub_metadata: Option<Vec<u8>>,
    // The path containing the metadata, to record as work product.
    path: Option<Box<Path>>,
    // We need to carry MaybeTempDir to avoid deleting the temporary
    // directory while accessing the Mmap.
    _temp_dir: Option<MaybeTempDir>,
}

impl EncodedMetadata {
    #[inline]
    pub fn from_path(
        path: PathBuf,
        stub_path: Option<PathBuf>,
        temp_dir: Option<MaybeTempDir>,
    ) -> std::io::Result<Self> {
        let file = std::fs::File::open(&path)?;
        let file_metadata = file.metadata()?;
        if file_metadata.len() == 0 {
            return Ok(Self {
                full_metadata: None,
                stub_metadata: None,
                path: None,
                _temp_dir: None,
            });
        }
        let full_mmap = unsafe { Some(Mmap::map(file)?) };

        let stub =
            if let Some(stub_path) = stub_path { Some(std::fs::read(stub_path)?) } else { None };

        Ok(Self {
            full_metadata: full_mmap,
            stub_metadata: stub,
            path: Some(path.into()),
            _temp_dir: temp_dir,
        })
    }

    #[inline]
    pub fn full(&self) -> &[u8] {
        &self.full_metadata.as_deref().unwrap_or_default()
    }

    #[inline]
    pub fn stub_or_full(&self) -> &[u8] {
        self.stub_metadata.as_deref().unwrap_or(self.full())
    }

    #[inline]
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }
}

impl<S: Encoder> Encodable<S> for EncodedMetadata {
    fn encode(&self, s: &mut S) {
        self.stub_metadata.encode(s);

        let slice = self.full();
        slice.encode(s)
    }
}

impl<D: Decoder> Decodable<D> for EncodedMetadata {
    fn decode(d: &mut D) -> Self {
        let stub = <Option<Vec<u8>>>::decode(d);

        let len = d.read_usize();
        let full_metadata = if len > 0 {
            let mut mmap = MmapMut::map_anon(len).unwrap();
            mmap.copy_from_slice(d.read_raw_bytes(len));
            Some(mmap.make_read_only().unwrap())
        } else {
            None
        };

        Self { full_metadata, stub_metadata: stub, path: None, _temp_dir: None }
    }
}

#[instrument(level = "trace", skip(tcx))]
pub fn encode_metadata(tcx: TyCtxt<'_>, path: &Path, ref_path: Option<&Path>) {
    // Since encoding metadata is not in a query, and nothing is cached,
    // there's no need to do dep-graph tracking for any of it.
    tcx.dep_graph.assert_ignored();

    let _prof_timer = tcx.prof.verbose_generic_activity("generate_crate_metadata");

    let dep_node = tcx.metadata_dep_node();

    // If the metadata dep-node is green, try to reuse the saved work product.
    if tcx.dep_graph.is_fully_enabled()
        && let work_product_id = WorkProductId::from_cgu_name("metadata")
        && let Some(work_product) = tcx.dep_graph.previous_work_product(&work_product_id)
        && tcx.dep_graph.try_mark_green(tcx, &dep_node).is_some()
    {
        let saved_path = &work_product.saved_files["rmeta"];
        let incr_comp_session_dir = tcx.sess.incr_comp_session_dir_opt().unwrap();
        let source_file = rustc_incremental::in_incr_comp_dir(&incr_comp_session_dir, saved_path);
        debug!("copying preexisting metadata from {source_file:?} to {path:?}");
        match rustc_fs_util::link_or_copy(&source_file, path) {
            Ok(_) => {}
            Err(err) => tcx.dcx().emit_fatal(FailCreateFileEncoder { err }),
        };
        return;
    };

    if tcx.sess.threads().is_some() {
        // Prefetch some queries used by metadata encoding.
        // This is not necessary for correctness, but is only done for performance reasons.
        // It can be removed if it turns out to cause trouble or be detrimental to performance.
        par_join(
            || prefetch_mir(tcx),
            || {
                let _ = tcx.exported_non_generic_symbols(LOCAL_CRATE);
                let _ = tcx.exported_generic_symbols(LOCAL_CRATE);
            },
        );
    }

    let mut hashes = CrateHashes {
        private_hash: tcx.crate_hash(LOCAL_CRATE),
        public_hash: tcx.crate_hash(LOCAL_CRATE),
    };

    // Perform metadata encoding inside a task, so the dep-graph can check if any encoded
    // information changes, and maybe reuse the work product.
    tcx.dep_graph.with_task(
        dep_node,
        tcx,
        || {
            tcx.with_stable_hashing_context(|mut hcx| {
                hcx.set_hash_spans_as_parentless(true);
                let is_proc_macro = tcx.crate_types().contains(&CrateType::ProcMacro);
                let hash_public_api = tcx.sess.opts.unstable_opts.public_api_hash
                    & !is_proc_macro
                    & tcx.sess.opts.incremental.is_some();
                with_encode_metadata_header(tcx, path, |ecx| {
                    // Encode all the entries and extra information in the crate,
                    // culminating in the `CrateRoot` which points to all of it.
                    let (root, crate_hashes) = if hash_public_api {
                        let mut hcx = PublicApiHashingContext::<true>::new(hcx);
                        ecx.encode_crate_root(&mut hcx)
                    } else {
                        let mut hcx = PublicApiHashingContext::<false>::new(hcx);
                        ecx.encode_crate_root(&mut hcx)
                    };
                    hashes = crate_hashes;

                    // Flush buffer to ensure backing file has the correct size.
                    ecx.opaque.flush();
                    // Record metadata size for self-profiling
                    tcx.prof.artifact_size(
                        "crate_metadata",
                        "crate_metadata",
                        ecx.opaque.file().metadata().unwrap().len(),
                    );

                    root.position.get()
                });
            })
        },
        None,
    );

    // Generate the metadata stub manually, as that is a small file compared to full metadata.
    if let Some(ref_path) = ref_path {
        let _prof_timer = tcx.prof.verbose_generic_activity("generate_crate_metadata_stub");

        with_encode_metadata_header(tcx, ref_path, |ecx| {
            let header: LazyValue<CrateHeader> = ecx.lazy(CrateHeader {
                name: tcx.crate_name(LOCAL_CRATE),
                triple: tcx.sess.opts.target_triple.clone(),
                hashes,
                is_proc_macro_crate: false,
                is_stub: true,
            });
            header.position.get()
        })
    }
}

fn with_encode_metadata_header(
    tcx: TyCtxt<'_>,
    path: &Path,
    f: impl FnOnce(&mut EncodeContext<'_, '_>) -> usize,
) {
    let mut encoder = opaque::FileEncoder::new(path)
        .unwrap_or_else(|err| tcx.dcx().emit_fatal(FailCreateFileEncoder { err }));
    encoder.emit_raw_bytes(METADATA_HEADER);

    // Will be filled with the root position after encoding everything.
    encoder.emit_raw_bytes(&0u64.to_le_bytes());

    let source_map_files = tcx.sess.source_map().files();
    let source_file_cache = (Arc::clone(&source_map_files[0]), 0);
    let required_source_files = Some(FxIndexSet::default());
    drop(source_map_files);

    let hygiene_ctxt = HygieneEncodeContext::default();

    let mut ecx = EncodeContext {
        opaque: encoder,
        tcx,
        feat: tcx.features(),
        tables: Default::default(),
        lazy_state: LazyState::NoNode,
        span_shorthands: Default::default(),
        type_shorthands: Default::default(),
        predicate_shorthands: Default::default(),
        source_file_cache,
        interpret_allocs: Default::default(),
        required_source_files,
        is_proc_macro: tcx.crate_types().contains(&CrateType::ProcMacro),
        hygiene_ctxt: &hygiene_ctxt,
        symbol_index_table: Default::default(),
    };

    // Encode the rustc version string in a predictable location.
    rustc_version(tcx.sess.cfg_version).encode(&mut ecx);

    let root_position = f(&mut ecx);

    // Make sure we report any errors from writing to the file.
    // If we forget this, compilation can succeed with an incomplete rmeta file,
    // causing an ICE when the rmeta file is read by another compilation.
    if let Err((path, err)) = ecx.opaque.finish() {
        tcx.dcx().emit_fatal(FailWriteFile { path: &path, err });
    }

    let file = ecx.opaque.file();
    if let Err(err) = encode_root_position(file, root_position) {
        tcx.dcx().emit_fatal(FailWriteFile { path: ecx.opaque.path(), err });
    }
}

fn encode_root_position(mut file: &File, pos: usize) -> Result<(), std::io::Error> {
    // We will return to this position after writing the root position.
    let pos_before_seek = file.stream_position().unwrap();

    // Encode the root position.
    let header = METADATA_HEADER.len();
    file.seek(std::io::SeekFrom::Start(header as u64))?;
    file.write_all(&pos.to_le_bytes())?;

    // Return to the position where we are before writing the root position.
    file.seek(std::io::SeekFrom::Start(pos_before_seek))?;
    Ok(())
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers {
        doc_link_resolutions: |tcx, def_id| {
            tcx.resolutions(())
                .doc_link_resolutions
                .get(&def_id)
                .unwrap_or_else(|| span_bug!(tcx.def_span(def_id), "no resolutions for a doc link"))
        },
        doc_link_traits_in_scope: |tcx, def_id| {
            tcx.resolutions(()).doc_link_traits_in_scope.get(&def_id).unwrap_or_else(|| {
                span_bug!(tcx.def_span(def_id), "no traits in scope for a doc link")
            })
        },

        ..*providers
    }
}

/// Build a textual representation of an unevaluated constant expression.
///
/// If the const expression is too complex, an underscore `_` is returned.
/// For const arguments, it's `{ _ }` to be precise.
/// This means that the output is not necessarily valid Rust code.
///
/// Currently, only
///
/// * literals (optionally with a leading `-`)
/// * unit `()`
/// * blocks (`{ … }`) around simple expressions and
/// * paths without arguments
///
/// are considered simple enough. Simple blocks are included since they are
/// necessary to disambiguate unit from the unit type.
/// This list might get extended in the future.
///
/// Without this censoring, in a lot of cases the output would get too large
/// and verbose. Consider `match` expressions, blocks and deeply nested ADTs.
/// Further, private and `doc(hidden)` fields of structs would get leaked
/// since HIR datatypes like the `body` parameter do not contain enough
/// semantic information for this function to be able to hide them –
/// at least not without significant performance overhead.
///
/// Whenever possible, prefer to evaluate the constant first and try to
/// use a different method for pretty-printing. Ideally this function
/// should only ever be used as a fallback.
pub fn rendered_const<'tcx>(tcx: TyCtxt<'tcx>, body: &hir::Body<'_>, def_id: LocalDefId) -> String {
    let value = body.value;

    #[derive(PartialEq, Eq)]
    enum Classification {
        Literal,
        Simple,
        Complex,
    }

    use Classification::*;

    fn classify(expr: &hir::Expr<'_>) -> Classification {
        match &expr.kind {
            hir::ExprKind::Unary(hir::UnOp::Neg, expr) => {
                if matches!(expr.kind, hir::ExprKind::Lit(_)) { Literal } else { Complex }
            }
            hir::ExprKind::Lit(_) => Literal,
            hir::ExprKind::Tup([]) => Simple,
            hir::ExprKind::Block(hir::Block { stmts: [], expr: Some(expr), .. }, _) => {
                if classify(expr) == Complex { Complex } else { Simple }
            }
            // Paths with a self-type or arguments are too “complex” following our measure since
            // they may leak private fields of structs (with feature `adt_const_params`).
            // Consider: `<Self as Trait<{ Struct { private: () } }>>::CONSTANT`.
            // Paths without arguments are definitely harmless though.
            hir::ExprKind::Path(hir::QPath::Resolved(_, hir::Path { segments, .. })) => {
                if segments.iter().all(|segment| segment.args.is_none()) { Simple } else { Complex }
            }
            // FIXME: Claiming that those kinds of QPaths are simple is probably not true if the Ty
            //        contains const arguments. Is there a *concise* way to check for this?
            hir::ExprKind::Path(hir::QPath::TypeRelative(..)) => Simple,
            _ => Complex,
        }
    }

    match classify(value) {
        // For non-macro literals, we avoid invoking the pretty-printer and use the source snippet
        // instead to preserve certain stylistic choices the user likely made for the sake of
        // legibility, like:
        //
        // * hexadecimal notation
        // * underscores
        // * character escapes
        //
        // FIXME: This passes through `-/*spacer*/0` verbatim.
        Literal
            if !value.span.from_expansion()
                && let Ok(snippet) = tcx.sess.source_map().span_to_snippet(value.span) =>
        {
            snippet
        }

        // Otherwise we prefer pretty-printing to get rid of extraneous whitespace, comments and
        // other formatting artifacts.
        Literal | Simple => id_to_string(&tcx, body.id().hir_id),

        // FIXME: Omit the curly braces if the enclosing expression is an array literal
        //        with a repeated element (an `ExprKind::Repeat`) as in such case it
        //        would not actually need any disambiguation.
        Complex => {
            if tcx.def_kind(def_id) == DefKind::AnonConst {
                "{ _ }".to_owned()
            } else {
                "_".to_owned()
            }
        }
    }
}

fn crate_deps(tcx: TyCtxt<'_>) -> impl Iterator<Item = (CrateNum, CrateDep)> + '_ {
    tcx.crates(()).iter().map(move |&cnum| {
        let dep = CrateDep {
            name: tcx.crate_name(cnum),
            hash: tcx.public_api_hash(cnum),
            host_hash: tcx.crate_host_hash(cnum),
            kind: tcx.crate_dep_kind(cnum),
            extra_filename: tcx.extra_filename(cnum).clone(),
            is_private: tcx.is_private_dep(cnum),
        };
        (cnum, dep)
    })
}

fn dylib_dependency_formats(
    tcx: TyCtxt<'_>,
) -> Option<impl Iterator<Item = (CrateNum, Option<LinkagePreference>)>> {
    let formats = tcx.dependency_formats(());
    formats.get(&CrateType::Dylib).map(|arr| {
        arr.iter().enumerate().skip(1 /* skip LOCAL_CRATE */).map(|(i, slot)| {
            (
                CrateNum::new(i),
                match *slot {
                    Linkage::NotLinked | Linkage::IncludedFromDylib => None,

                    Linkage::Dynamic => Some(LinkagePreference::RequireDynamic),
                    Linkage::Static => Some(LinkagePreference::RequireStatic),
                },
            )
        })
    })
}
