use std::borrow::Borrow;
use std::collections::hash_map::Entry;
use std::fs::File;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rustc_ast::attr::AttributeExt;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::memmap::{Mmap, MmapMut};
use rustc_data_structures::sync::{join, par_for_each_in};
use rustc_data_structures::temp_dir::MaybeTempDir;
use rustc_data_structures::thousands::usize_with_underscores;
use rustc_feature::Features;
use rustc_hir as hir;
use rustc_hir::def_id::{CRATE_DEF_ID, CRATE_DEF_INDEX, LOCAL_CRATE, LocalDefId, LocalDefIdSet};
use rustc_hir::definitions::DefPathData;
use rustc_hir_pretty::id_to_string;
use rustc_middle::middle::dependency_format::Linkage;
use rustc_middle::middle::exported_symbols::metadata_symbol_name;
use rustc_middle::mir::interpret;
use rustc_middle::query::Providers;
use rustc_middle::traits::specialization_graph;
use rustc_middle::ty::codec::TyEncoder;
use rustc_middle::ty::fast_reject::{self, TreatParams};
use rustc_middle::ty::{AssocItemContainer, SymbolName};
use rustc_middle::{bug, span_bug};
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder, opaque};
use rustc_session::config::{CrateType, OptLevel, TargetModifier};
use rustc_span::hygiene::HygieneEncodeContext;
use rustc_span::{
    ExternalSource, FileName, SourceFile, SpanData, SpanEncoder, StableSourceFileId, SyntaxContext,
    sym,
};
use tracing::{debug, instrument, trace};

use crate::errors::{FailCreateFileEncoder, FailWriteFile};
use crate::rmeta::*;

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
    symbol_table: FxHashMap<Symbol, usize>,
}

/// If the current crate is a proc-macro, returns early with `LazyArray::default()`.
/// This is useful for skipping the encoding of things that aren't needed
/// for proc-macro crates.
macro_rules! empty_proc_macro {
    ($self:ident) => {
        if $self.is_proc_macro {
            return LazyArray::default();
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

    fn encode_symbol(&mut self, symbol: Symbol) {
        // if symbol predefined, emit tag and symbol index
        if symbol.is_predefined() {
            self.opaque.emit_u8(SYMBOL_PREDEFINED);
            self.opaque.emit_u32(symbol.as_u32());
        } else {
            // otherwise write it as string or as offset to it
            match self.symbol_table.entry(symbol) {
                Entry::Vacant(o) => {
                    self.opaque.emit_u8(SYMBOL_STR);
                    let pos = self.opaque.position();
                    o.insert(pos);
                    self.emit_str(symbol.as_str());
                }
                Entry::Occupied(o) => {
                    let x = *o.get();
                    self.emit_u8(SYMBOL_OFFSET);
                    self.emit_usize(x);
                }
            }
        }
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
    ($self:ident.$tables:ident.$table:ident[$def_id:expr] <- $value:expr) => {{
        {
            let value = $value;
            let lazy = $self.lazy(value);
            $self.$tables.$table.set_some($def_id.index, lazy);
        }
    }};
}

// Shorthand for `$self.$tables.$table.set_some($def_id.index, $self.lazy_array($value))`, which would
// normally need extra variables to avoid errors about multiple mutable borrows.
macro_rules! record_array {
    ($self:ident.$tables:ident.$table:ident[$def_id:expr] <- $value:expr) => {{
        {
            let value = $value;
            let lazy = $self.lazy_array(value);
            $self.$tables.$table.set_some($def_id.index, lazy);
        }
    }};
}

macro_rules! record_defaulted_array {
    ($self:ident.$tables:ident.$table:ident[$def_id:expr] <- $value:expr) => {{
        {
            let value = $value;
            let lazy = $self.lazy_array(value);
            $self.$tables.$table.set($def_id.index, lazy);
        }
    }};
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

    fn encode_def_path_table(&mut self) {
        let table = self.tcx.def_path_table();
        if self.is_proc_macro {
            for def_index in std::iter::once(CRATE_DEF_INDEX)
                .chain(self.tcx.resolutions(()).proc_macros.iter().map(|p| p.local_def_index))
            {
                let def_key = self.lazy(table.def_key(def_index));
                let def_path_hash = table.def_path_hash(def_index);
                self.tables.def_keys.set_some(def_index, def_key);
                self.tables.def_path_hashes.set(def_index, def_path_hash.local_hash().as_u64());
            }
        } else {
            for (def_index, def_key, def_path_hash) in table.enumerated_keys_and_path_hashes() {
                let def_key = self.lazy(def_key);
                self.tables.def_keys.set_some(def_index, def_key);
                self.tables.def_path_hashes.set(def_index, def_path_hash.local_hash().as_u64());
            }
        }
    }

    fn encode_def_path_hash_map(&mut self) -> LazyValue<DefPathHashMapRef<'static>> {
        self.lazy(DefPathHashMapRef::BorrowedFromTcx(self.tcx.def_path_hash_to_def_index_map()))
    }

    fn encode_source_map(&mut self) -> LazyTable<u32, Option<LazyValue<rustc_span::SourceFile>>> {
        let source_map = self.tcx.sess.source_map();
        let all_source_files = source_map.files();

        // By replacing the `Option` with `None`, we ensure that we can't
        // accidentally serialize any more `Span`s after the source map encoding
        // is done.
        let required_source_files = self.required_source_files.take().unwrap();

        let working_directory = &self.tcx.sess.opts.working_dir;

        let mut adapted = TableBuilder::default();

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
                    let adapted_file_name = source_map
                        .path_mapping()
                        .to_embeddable_absolute_path(original_file_name.clone(), working_directory);

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
            adapted.set_some(on_disk_index, self.lazy(adapted_source_file));
        }

        adapted.encode(&mut self.opaque)
    }

    fn encode_crate_root(&mut self) -> LazyValue<CrateRoot> {
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

        let (crate_deps, dylib_dependency_formats) =
            stat!("dep", || (self.encode_crate_deps(), self.encode_dylib_dependency_formats()));

        let lib_features = stat!("lib-features", || self.encode_lib_features());

        let stability_implications =
            stat!("stability-implications", || self.encode_stability_implications());

        let (lang_items, lang_items_missing) = stat!("lang-items", || {
            (self.encode_lang_items(), self.encode_lang_items_missing())
        });

        let stripped_cfg_items = stat!("stripped-cfg-items", || self.encode_stripped_cfg_items());

        let diagnostic_items = stat!("diagnostic-items", || self.encode_diagnostic_items());

        let native_libraries = stat!("native-libs", || self.encode_native_libraries());

        let foreign_modules = stat!("foreign-modules", || self.encode_foreign_modules());

        _ = stat!("def-path-table", || self.encode_def_path_table());

        // Encode the def IDs of traits, for rustdoc and diagnostics.
        let traits = stat!("traits", || self.encode_traits());

        // Encode the def IDs of impls, for coherence checking.
        let impls = stat!("impls", || self.encode_impls());

        let incoherent_impls = stat!("incoherent-impls", || self.encode_incoherent_impls());

        _ = stat!("mir", || self.encode_mir());

        _ = stat!("def-ids", || self.encode_def_ids());

        let interpret_alloc_index = stat!("interpret-alloc-index", || {
            let mut interpret_alloc_index = Vec::new();
            let mut n = 0;
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
                    interpret::specialized_encode_alloc_id(self, tcx, id);
                }
                n = new_n;
            }
            self.lazy_array(interpret_alloc_index)
        });

        // Encode the proc macro data. This affects `tables`, so we need to do this before we
        // encode the tables. This overwrites def_keys, so it must happen after
        // encode_def_path_table.
        let proc_macro_data = stat!("proc-macro-data", || self.encode_proc_macros());

        let tables = stat!("tables", || self.tables.encode(&mut self.opaque));

        let debugger_visualizers =
            stat!("debugger-visualizers", || self.encode_debugger_visualizers());

        let exportable_items = stat!("exportable-items", || self.encode_exportable_items());

        let stable_order_of_exportable_impls =
            stat!("exportable-items", || self.encode_stable_order_of_exportable_impls());

        // Encode exported symbols info. This is prefetched in `encode_metadata`.
        let exported_symbols = stat!("exported-symbols", || {
            self.encode_exported_symbols(tcx.exported_symbols(LOCAL_CRATE))
        });

        // Encode the hygiene data.
        // IMPORTANT: this *must* be the last thing that we encode (other than `SourceMap`). The
        // process of encoding other items (e.g. `optimized_mir`) may cause us to load data from
        // the incremental cache. If this causes us to deserialize a `Span`, then we may load
        // additional `SyntaxContext`s into the global `HygieneData`. Therefore, we need to encode
        // the hygiene data last to ensure that we encode any `SyntaxContext`s that might be used.
        let (syntax_contexts, expn_data, expn_hashes) = stat!("hygiene", || self.encode_hygiene());

        let def_path_hash_map = stat!("def-path-hash-map", || self.encode_def_path_hash_map());

        // Encode source_map. This needs to be done last, because encoding `Span`s tells us which
        // `SourceFiles` we actually need to encode.
        let source_map = stat!("source-map", || self.encode_source_map());
        let target_modifiers = stat!("target-modifiers", || self.encode_target_modifiers());

        let root = stat!("final", || {
            let attrs = tcx.hir_krate_attrs();
            self.lazy(CrateRoot {
                header: CrateHeader {
                    name: tcx.crate_name(LOCAL_CRATE),
                    triple: tcx.sess.opts.target_triple.clone(),
                    hash: tcx.crate_hash(LOCAL_CRATE),
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
                has_default_lib_allocator: ast::attr::contains_name(
                    attrs,
                    sym::default_lib_allocator,
                ),
                proc_macro_data,
                debugger_visualizers,
                compiler_builtins: ast::attr::contains_name(attrs, sym::compiler_builtins),
                needs_allocator: ast::attr::contains_name(attrs, sym::needs_allocator),
                needs_panic_runtime: ast::attr::contains_name(attrs, sym::needs_panic_runtime),
                no_builtins: ast::attr::contains_name(attrs, sym::no_builtins),
                panic_runtime: ast::attr::contains_name(attrs, sym::panic_runtime),
                profiler_runtime: ast::attr::contains_name(attrs, sym::profiler_runtime),
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
                traits,
                impls,
                incoherent_impls,
                exportable_items,
                stable_order_of_exportable_impls,
                exported_symbols,
                interpret_alloc_index,
                tables,
                syntax_contexts,
                expn_data,
                expn_hashes,
                def_path_hash_map,
                specialization_enabled_in: tcx.specialization_enabled_in(LOCAL_CRATE),
            })
        });

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

        root
    }
}

struct AnalyzeAttrState<'a> {
    is_exported: bool,
    is_doc_hidden: bool,
    features: &'a Features,
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
fn analyze_attr(attr: &impl AttributeExt, state: &mut AnalyzeAttrState<'_>) -> bool {
    let mut should_encode = false;
    if let Some(name) = attr.name()
        && !rustc_feature::encode_cross_crate(name)
    {
        // Attributes not marked encode-cross-crate don't need to be encoded for downstream crates.
    } else if attr.doc_str().is_some() {
        // We keep all doc comments reachable to rustdoc because they might be "imported" into
        // downstream crates if they use `#[doc(inline)]` to copy an item's documentation into
        // their own.
        if state.is_exported {
            should_encode = true;
        }
    } else if attr.has_name(sym::doc) {
        // If this is a `doc` attribute that doesn't have anything except maybe `inline` (as in
        // `#[doc(inline)]`), then we can remove it. It won't be inlinable in downstream crates.
        if let Some(item_list) = attr.meta_item_list() {
            for item in item_list {
                if !item.has_name(sym::inline) {
                    should_encode = true;
                    if item.has_name(sym::hidden) {
                        state.is_doc_hidden = true;
                        break;
                    }
                }
            }
        }
    } else if let &[sym::diagnostic, seg] = &*attr.path() {
        should_encode = rustc_feature::is_stable_diagnostic_attribute(seg, state.features);
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
        | DefKind::Const
        | DefKind::Static { .. }
        | DefKind::Ctor(..)
        | DefKind::AssocFn
        | DefKind::AssocConst
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
        | DefKind::Const
        | DefKind::Static { nested: false, .. }
        | DefKind::AssocFn
        | DefKind::AssocConst
        | DefKind::Macro(_)
        | DefKind::Field
        | DefKind::Impl { .. } => true,
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
        | DefKind::Use
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
        | DefKind::Const
        | DefKind::ConstParam
        | DefKind::Static { .. }
        | DefKind::Ctor(..)
        | DefKind::AssocFn
        | DefKind::AssocConst
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
        | DefKind::Const
        | DefKind::Static { nested: false, .. }
        | DefKind::Ctor(..)
        | DefKind::AssocFn
        | DefKind::AssocConst
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
        | DefKind::AssocConst
        | DefKind::TyParam
        | DefKind::ConstParam
        | DefKind::Static { .. }
        | DefKind::Const
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
        // Constructors
        DefKind::Ctor(_, _) => {
            let mir_opt_base = tcx.sess.opts.output_types.should_codegen()
                || tcx.sess.opts.unstable_opts.always_encode_mir;
            (true, mir_opt_base)
        }
        // Constants
        DefKind::AnonConst | DefKind::InlineConst | DefKind::AssocConst | DefKind::Const => {
            (true, false)
        }
        // Coroutines require optimized MIR to compute layout.
        DefKind::Closure if tcx.is_coroutine(def_id.to_def_id()) => (false, true),
        DefKind::SyntheticCoroutineBody => (false, true),
        // Full-fledged functions + closures
        DefKind::AssocFn | DefKind::Fn | DefKind::Closure => {
            let generics = tcx.generics_of(def_id);
            let opt = tcx.sess.opts.unstable_opts.always_encode_mir
                || (tcx.sess.opts.output_types.should_codegen()
                    && reachable_set.contains(&def_id)
                    && (generics.requires_monomorphization(tcx)
                        || tcx.cross_crate_inlinable(def_id)));
            // The function has a `const` modifier or is in a `#[const_trait]`.
            let is_const_fn = tcx.is_const_fn(def_id.to_def_id())
                || tcx.is_const_default_method(def_id.to_def_id());
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
        | DefKind::AssocConst
        | DefKind::TyParam
        | DefKind::ConstParam
        | DefKind::Static { .. }
        | DefKind::Const
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
        | DefKind::Const
        | DefKind::Static { .. }
        | DefKind::Ctor(..)
        | DefKind::AssocFn
        | DefKind::AssocConst
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
        | DefKind::Const
        | DefKind::Static { nested: false, .. }
        | DefKind::TyAlias
        | DefKind::ForeignTy
        | DefKind::Impl { .. }
        | DefKind::AssocFn
        | DefKind::AssocConst
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
                ty::AssocItemContainer::Impl => true,
                ty::AssocItemContainer::Trait => assoc_item.defaultness(tcx).has_value(),
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
        | DefKind::Const
        | DefKind::Static { .. }
        | DefKind::Ctor(..)
        | DefKind::TyAlias
        | DefKind::OpaqueTy
        | DefKind::ForeignTy
        | DefKind::Impl { .. }
        | DefKind::AssocConst
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
        DefKind::Fn | DefKind::AssocFn | DefKind::Closure | DefKind::Ctor(_, CtorKind::Fn) => true,

        DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Field
        | DefKind::Const
        | DefKind::AssocConst
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
        DefKind::Const | DefKind::AssocConst | DefKind::AnonConst | DefKind::InlineConst => true,

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

fn should_encode_fn_impl_trait_in_trait<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> bool {
    if let Some(assoc_item) = tcx.opt_associated_item(def_id)
        && assoc_item.container == ty::AssocItemContainer::Trait
        && assoc_item.is_fn()
    {
        true
    } else {
        false
    }
}

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
    fn encode_attrs(&mut self, def_id: LocalDefId) {
        let tcx = self.tcx;
        let mut state = AnalyzeAttrState {
            is_exported: tcx.effective_visibilities(()).is_exported(def_id),
            is_doc_hidden: false,
            features: &tcx.features(),
        };
        let attr_iter = tcx
            .hir_attrs(tcx.local_def_id_to_hir_id(def_id))
            .iter()
            .filter(|attr| analyze_attr(*attr, &mut state));

        record_array!(self.tables.attributes[def_id.to_def_id()] <- attr_iter);

        let mut attr_flags = AttrFlags::empty();
        if state.is_doc_hidden {
            attr_flags |= AttrFlags::IS_DOC_HIDDEN;
        }
        self.tables.attr_flags.set(def_id.local_def_index, attr_flags);
    }

    fn encode_def_ids(&mut self) {
        self.encode_info_for_mod(CRATE_DEF_ID);

        // Proc-macro crates only export proc-macro items, which are looked
        // up using `proc_macro_data`
        if self.is_proc_macro {
            return;
        }

        let tcx = self.tcx;

        for local_id in tcx.iter_local_def_id() {
            let def_id = local_id.to_def_id();
            let def_kind = tcx.def_kind(local_id);
            self.tables.def_kind.set_some(def_id.index, def_kind);

            // The `DefCollector` will sometimes create unnecessary `DefId`s
            // for trivial const arguments which are directly lowered to
            // `ConstArgKind::Path`. We never actually access this `DefId`
            // anywhere so we don't need to encode it for other crates.
            if def_kind == DefKind::AnonConst
                && match tcx.hir_node_by_def_id(local_id) {
                    hir::Node::ConstArg(hir::ConstArg { kind, .. }) => match kind {
                        // Skip encoding defs for these as they should not have had a `DefId` created
                        hir::ConstArgKind::Path(..) | hir::ConstArgKind::Infer(..) => true,
                        hir::ConstArgKind::Anon(..) => false,
                    },
                    _ => false,
                }
            {
                continue;
            }

            if def_kind == DefKind::Field
                && let hir::Node::Field(field) = tcx.hir_node_by_def_id(local_id)
                && let Some(anon) = field.default
            {
                record!(self.tables.default_fields[def_id] <- anon.def_id.to_def_id());
            }

            if should_encode_span(def_kind) {
                let def_span = tcx.def_span(local_id);
                record!(self.tables.def_span[def_id] <- def_span);
            }
            if should_encode_attrs(def_kind) {
                self.encode_attrs(local_id);
            }
            if should_encode_expn_that_defined(def_kind) {
                record!(self.tables.expn_that_defined[def_id] <- self.tcx.expn_that_defined(def_id));
            }
            if should_encode_span(def_kind)
                && let Some(ident_span) = tcx.def_ident_span(def_id)
            {
                record!(self.tables.def_ident_span[def_id] <- ident_span);
            }
            if def_kind.has_codegen_attrs() {
                record!(self.tables.codegen_fn_attrs[def_id] <- self.tcx.codegen_fn_attrs(def_id));
            }
            if should_encode_visibility(def_kind) {
                let vis =
                    self.tcx.local_visibility(local_id).map_id(|def_id| def_id.local_def_index);
                record!(self.tables.visibility[def_id] <- vis);
            }
            if should_encode_stability(def_kind) {
                self.encode_stability(def_id);
                self.encode_const_stability(def_id);
                self.encode_default_body_stability(def_id);
                self.encode_deprecation(def_id);
            }
            if should_encode_variances(tcx, def_id, def_kind) {
                let v = self.tcx.variances_of(def_id);
                record_array!(self.tables.variances_of[def_id] <- v);
            }
            if should_encode_fn_sig(def_kind) {
                record!(self.tables.fn_sig[def_id] <- tcx.fn_sig(def_id));
            }
            if should_encode_generics(def_kind) {
                let g = tcx.generics_of(def_id);
                record!(self.tables.generics_of[def_id] <- g);
                record!(self.tables.explicit_predicates_of[def_id] <- self.tcx.explicit_predicates_of(def_id));
                let inferred_outlives = self.tcx.inferred_outlives_of(def_id);
                record_defaulted_array!(self.tables.inferred_outlives_of[def_id] <- inferred_outlives);

                for param in &g.own_params {
                    if let ty::GenericParamDefKind::Const { has_default: true, .. } = param.kind {
                        let default = self.tcx.const_param_default(param.def_id);
                        record!(self.tables.const_param_default[param.def_id] <- default);
                    }
                }
            }
            if tcx.is_conditionally_const(def_id) {
                record!(self.tables.const_conditions[def_id] <- self.tcx.const_conditions(def_id));
            }
            if should_encode_type(tcx, local_id, def_kind) {
                record!(self.tables.type_of[def_id] <- self.tcx.type_of(def_id));
            }
            if should_encode_constness(def_kind) {
                self.tables.constness.set_some(def_id.index, self.tcx.constness(def_id));
            }
            if let DefKind::Fn | DefKind::AssocFn = def_kind {
                self.tables.asyncness.set_some(def_id.index, tcx.asyncness(def_id));
                record_array!(self.tables.fn_arg_idents[def_id] <- tcx.fn_arg_idents(def_id));
            }
            if let Some(name) = tcx.intrinsic(def_id) {
                record!(self.tables.intrinsic[def_id] <- name);
            }
            if let DefKind::TyParam = def_kind {
                let default = self.tcx.object_lifetime_default(def_id);
                record!(self.tables.object_lifetime_default[def_id] <- default);
            }
            if let DefKind::Trait = def_kind {
                record!(self.tables.trait_def[def_id] <- self.tcx.trait_def(def_id));
                record_defaulted_array!(self.tables.explicit_super_predicates_of[def_id] <-
                    self.tcx.explicit_super_predicates_of(def_id).skip_binder());
                record_defaulted_array!(self.tables.explicit_implied_predicates_of[def_id] <-
                    self.tcx.explicit_implied_predicates_of(def_id).skip_binder());
                let module_children = self.tcx.module_children_local(local_id);
                record_array!(self.tables.module_children_non_reexports[def_id] <-
                    module_children.iter().map(|child| child.res.def_id().index));
                if self.tcx.is_const_trait(def_id) {
                    record_defaulted_array!(self.tables.explicit_implied_const_bounds[def_id]
                        <- self.tcx.explicit_implied_const_bounds(def_id).skip_binder());
                }
            }
            if let DefKind::TraitAlias = def_kind {
                record!(self.tables.trait_def[def_id] <- self.tcx.trait_def(def_id));
                record_defaulted_array!(self.tables.explicit_super_predicates_of[def_id] <-
                    self.tcx.explicit_super_predicates_of(def_id).skip_binder());
                record_defaulted_array!(self.tables.explicit_implied_predicates_of[def_id] <-
                    self.tcx.explicit_implied_predicates_of(def_id).skip_binder());
            }
            if let DefKind::Trait | DefKind::Impl { .. } = def_kind {
                let associated_item_def_ids = self.tcx.associated_item_def_ids(def_id);
                record_array!(self.tables.associated_item_or_field_def_ids[def_id] <-
                    associated_item_def_ids.iter().map(|&def_id| {
                        assert!(def_id.is_local());
                        def_id.index
                    })
                );
                for &def_id in associated_item_def_ids {
                    self.encode_info_for_assoc_item(def_id);
                }
            }
            if let DefKind::Closure | DefKind::SyntheticCoroutineBody = def_kind
                && let Some(coroutine_kind) = self.tcx.coroutine_kind(def_id)
            {
                self.tables.coroutine_kind.set(def_id.index, Some(coroutine_kind))
            }
            if def_kind == DefKind::Closure
                && tcx.type_of(def_id).skip_binder().is_coroutine_closure()
            {
                let coroutine_for_closure = self.tcx.coroutine_for_closure(def_id);
                self.tables
                    .coroutine_for_closure
                    .set_some(def_id.index, coroutine_for_closure.into());

                // If this async closure has a by-move body, record it too.
                if tcx.needs_coroutine_by_move_body_def_id(coroutine_for_closure) {
                    self.tables.coroutine_by_move_body_def_id.set_some(
                        coroutine_for_closure.index,
                        self.tcx.coroutine_by_move_body_def_id(coroutine_for_closure).into(),
                    );
                }
            }
            if let DefKind::Static { .. } = def_kind {
                if !self.tcx.is_foreign_item(def_id) {
                    let data = self.tcx.eval_static_initializer(def_id).unwrap();
                    record!(self.tables.eval_static_initializer[def_id] <- data);
                }
            }
            if let DefKind::Enum | DefKind::Struct | DefKind::Union = def_kind {
                self.encode_info_for_adt(local_id);
            }
            if let DefKind::Mod = def_kind {
                self.encode_info_for_mod(local_id);
            }
            if let DefKind::Macro(_) = def_kind {
                self.encode_info_for_macro(local_id);
            }
            if let DefKind::TyAlias = def_kind {
                self.tables
                    .type_alias_is_lazy
                    .set(def_id.index, self.tcx.type_alias_is_lazy(def_id));
            }
            if let DefKind::OpaqueTy = def_kind {
                self.encode_explicit_item_bounds(def_id);
                self.encode_explicit_item_self_bounds(def_id);
                record!(self.tables.opaque_ty_origin[def_id] <- self.tcx.opaque_ty_origin(def_id));
                self.encode_precise_capturing_args(def_id);
                if tcx.is_conditionally_const(def_id) {
                    record_defaulted_array!(self.tables.explicit_implied_const_bounds[def_id]
                        <- tcx.explicit_implied_const_bounds(def_id).skip_binder());
                }
            }
            if let DefKind::AnonConst = def_kind {
                record!(self.tables.anon_const_kind[def_id] <- self.tcx.anon_const_kind(def_id));
            }
            if tcx.impl_method_has_trait_impl_trait_tys(def_id)
                && let Ok(table) = self.tcx.collect_return_position_impl_trait_in_trait_tys(def_id)
            {
                record!(self.tables.trait_impl_trait_tys[def_id] <- table);
            }
            if should_encode_fn_impl_trait_in_trait(tcx, def_id) {
                let table = tcx.associated_types_for_impl_traits_in_associated_fn(def_id);
                record_defaulted_array!(self.tables.associated_types_for_impl_traits_in_associated_fn[def_id] <- table);
            }
        }

        for (def_id, impls) in &tcx.crate_inherent_impls(()).0.inherent_impls {
            record_defaulted_array!(self.tables.inherent_impls[def_id.to_def_id()] <- impls.iter().map(|def_id| {
                assert!(def_id.is_local());
                def_id.index
            }));
        }

        for (def_id, res_map) in &tcx.resolutions(()).doc_link_resolutions {
            record!(self.tables.doc_link_resolutions[def_id.to_def_id()] <- res_map);
        }

        for (def_id, traits) in &tcx.resolutions(()).doc_link_traits_in_scope {
            record_array!(self.tables.doc_link_traits_in_scope[def_id.to_def_id()] <- traits);
        }
    }

    #[instrument(level = "trace", skip(self))]
    fn encode_info_for_adt(&mut self, local_def_id: LocalDefId) {
        let def_id = local_def_id.to_def_id();
        let tcx = self.tcx;
        let adt_def = tcx.adt_def(def_id);
        record!(self.tables.repr_options[def_id] <- adt_def.repr());

        let params_in_repr = self.tcx.params_in_repr(def_id);
        record!(self.tables.params_in_repr[def_id] <- params_in_repr);

        if adt_def.is_enum() {
            let module_children = tcx.module_children_local(local_def_id);
            record_array!(self.tables.module_children_non_reexports[def_id] <-
                module_children.iter().map(|child| child.res.def_id().index));
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
            record!(self.tables.variant_data[variant.def_id] <- data);

            record_array!(self.tables.associated_item_or_field_def_ids[variant.def_id] <- variant.fields.iter().map(|f| {
                assert!(f.did.is_local());
                f.did.index
            }));

            for field in &variant.fields {
                self.tables.safety.set_some(field.did.index, field.safety);
            }

            if let Some((CtorKind::Fn, ctor_def_id)) = variant.ctor {
                let fn_sig = tcx.fn_sig(ctor_def_id);
                // FIXME only encode signature for ctor_def_id
                record!(self.tables.fn_sig[variant.def_id] <- fn_sig);
            }
        }

        if let Some(destructor) = tcx.adt_destructor(local_def_id) {
            record!(self.tables.adt_destructor[def_id] <- destructor);
        }

        if let Some(destructor) = tcx.adt_async_destructor(local_def_id) {
            record!(self.tables.adt_async_destructor[def_id] <- destructor);
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn encode_info_for_mod(&mut self, local_def_id: LocalDefId) {
        let tcx = self.tcx;
        let def_id = local_def_id.to_def_id();

        // If we are encoding a proc-macro crates, `encode_info_for_mod` will
        // only ever get called for the crate root. We still want to encode
        // the crate root for consistency with other crates (some of the resolver
        // code uses it). However, we skip encoding anything relating to child
        // items - we encode information about proc-macros later on.
        if self.is_proc_macro {
            // Encode this here because we don't do it in encode_def_ids.
            record!(self.tables.expn_that_defined[def_id] <- tcx.expn_that_defined(local_def_id));
        } else {
            let module_children = tcx.module_children_local(local_def_id);

            record_array!(self.tables.module_children_non_reexports[def_id] <-
                module_children.iter().filter(|child| child.reexport_chain.is_empty())
                    .map(|child| child.res.def_id().index));

            record_defaulted_array!(self.tables.module_children_reexports[def_id] <-
                module_children.iter().filter(|child| !child.reexport_chain.is_empty()));
        }
    }

    fn encode_explicit_item_bounds(&mut self, def_id: DefId) {
        debug!("EncodeContext::encode_explicit_item_bounds({:?})", def_id);
        let bounds = self.tcx.explicit_item_bounds(def_id).skip_binder();
        record_defaulted_array!(self.tables.explicit_item_bounds[def_id] <- bounds);
    }

    fn encode_explicit_item_self_bounds(&mut self, def_id: DefId) {
        debug!("EncodeContext::encode_explicit_item_self_bounds({:?})", def_id);
        let bounds = self.tcx.explicit_item_self_bounds(def_id).skip_binder();
        record_defaulted_array!(self.tables.explicit_item_self_bounds[def_id] <- bounds);
    }

    #[instrument(level = "debug", skip(self))]
    fn encode_info_for_assoc_item(&mut self, def_id: DefId) {
        let tcx = self.tcx;
        let item = tcx.associated_item(def_id);

        self.tables.defaultness.set_some(def_id.index, item.defaultness(tcx));
        self.tables.assoc_container.set_some(def_id.index, item.container);

        match item.container {
            AssocItemContainer::Trait => {
                if item.is_type() {
                    self.encode_explicit_item_bounds(def_id);
                    self.encode_explicit_item_self_bounds(def_id);
                    if tcx.is_conditionally_const(def_id) {
                        record_defaulted_array!(self.tables.explicit_implied_const_bounds[def_id]
                            <- self.tcx.explicit_implied_const_bounds(def_id).skip_binder());
                    }
                }
            }
            AssocItemContainer::Impl => {
                if let Some(trait_item_def_id) = item.trait_item_def_id {
                    self.tables.trait_item_def_id.set_some(def_id.index, trait_item_def_id.into());
                }
            }
        }
        if let ty::AssocKind::Type { data: ty::AssocTypeData::Rpitit(rpitit_info) } = item.kind {
            record!(self.tables.opt_rpitit_info[def_id] <- rpitit_info);
            if matches!(rpitit_info, ty::ImplTraitInTraitData::Trait { .. }) {
                record_array!(
                    self.tables.assumed_wf_types_for_rpitit[def_id]
                        <- self.tcx.assumed_wf_types_for_rpitit(def_id)
                );
                self.encode_precise_capturing_args(def_id);
            }
        }
    }

    fn encode_precise_capturing_args(&mut self, def_id: DefId) {
        let Some(precise_capturing_args) = self.tcx.rendered_precise_capturing_args(def_id) else {
            return;
        };

        record_array!(self.tables.rendered_precise_capturing_args[def_id] <- precise_capturing_args);
    }

    fn encode_mir(&mut self) {
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
                record!(self.tables.optimized_mir[def_id.to_def_id()] <- tcx.optimized_mir(def_id));
                self.tables
                    .cross_crate_inlinable
                    .set(def_id.to_def_id().index, self.tcx.cross_crate_inlinable(def_id));
                record!(self.tables.closure_saved_names_of_captured_variables[def_id.to_def_id()]
                    <- tcx.closure_saved_names_of_captured_variables(def_id));

                if self.tcx.is_coroutine(def_id.to_def_id())
                    && let Some(witnesses) = tcx.mir_coroutine_witnesses(def_id)
                {
                    record!(self.tables.mir_coroutine_witnesses[def_id.to_def_id()] <- witnesses);
                }
            }
            if encode_const {
                record!(self.tables.mir_for_ctfe[def_id.to_def_id()] <- tcx.mir_for_ctfe(def_id));

                // FIXME(generic_const_exprs): this feels wrong to have in `encode_mir`
                let abstract_const = tcx.thir_abstract_const(def_id);
                if let Ok(Some(abstract_const)) = abstract_const {
                    record!(self.tables.thir_abstract_const[def_id.to_def_id()] <- abstract_const);
                }

                if should_encode_const(tcx.def_kind(def_id)) {
                    let qualifs = tcx.mir_const_qualif(def_id);
                    record!(self.tables.mir_const_qualif[def_id.to_def_id()] <- qualifs);
                    let body = tcx.hir_maybe_body_owned_by(def_id);
                    if let Some(body) = body {
                        let const_data = rendered_const(self.tcx, &body, def_id);
                        record!(self.tables.rendered_const[def_id.to_def_id()] <- const_data);
                    }
                }
            }
            record!(self.tables.promoted_mir[def_id.to_def_id()] <- tcx.promoted_mir(def_id));

            if self.tcx.is_coroutine(def_id.to_def_id())
                && let Some(witnesses) = tcx.mir_coroutine_witnesses(def_id)
            {
                record!(self.tables.mir_coroutine_witnesses[def_id.to_def_id()] <- witnesses);
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
                        self.tcx.deduced_param_attrs(local_def_id.to_def_id()));
                }
            }
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn encode_stability(&mut self, def_id: DefId) {
        // The query lookup can take a measurable amount of time in crates with many items. Check if
        // the stability attributes are even enabled before using their queries.
        if self.feat.staged_api() || self.tcx.sess.opts.unstable_opts.force_unstable_if_unmarked {
            if let Some(stab) = self.tcx.lookup_stability(def_id) {
                record!(self.tables.lookup_stability[def_id] <- stab)
            }
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn encode_const_stability(&mut self, def_id: DefId) {
        // The query lookup can take a measurable amount of time in crates with many items. Check if
        // the stability attributes are even enabled before using their queries.
        if self.feat.staged_api() || self.tcx.sess.opts.unstable_opts.force_unstable_if_unmarked {
            if let Some(stab) = self.tcx.lookup_const_stability(def_id) {
                record!(self.tables.lookup_const_stability[def_id] <- stab)
            }
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn encode_default_body_stability(&mut self, def_id: DefId) {
        // The query lookup can take a measurable amount of time in crates with many items. Check if
        // the stability attributes are even enabled before using their queries.
        if self.feat.staged_api() || self.tcx.sess.opts.unstable_opts.force_unstable_if_unmarked {
            if let Some(stab) = self.tcx.lookup_default_body_stability(def_id) {
                record!(self.tables.lookup_default_body_stability[def_id] <- stab)
            }
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn encode_deprecation(&mut self, def_id: DefId) {
        if let Some(depr) = self.tcx.lookup_deprecation(def_id) {
            record!(self.tables.lookup_deprecation_entry[def_id] <- depr);
        }
    }

    #[instrument(level = "debug", skip(self))]
    fn encode_info_for_macro(&mut self, def_id: LocalDefId) {
        let tcx = self.tcx;

        let (_, macro_def, _) = tcx.hir_expect_item(def_id).expect_macro();
        self.tables.is_macro_rules.set(def_id.local_def_index, macro_def.macro_rules);
        record!(self.tables.macro_definition[def_id.to_def_id()] <- &*macro_def.body);
    }

    fn encode_native_libraries(&mut self) -> LazyArray<NativeLib> {
        empty_proc_macro!(self);
        let used_libraries = self.tcx.native_libraries(LOCAL_CRATE);
        self.lazy_array(used_libraries.iter())
    }

    fn encode_foreign_modules(&mut self) -> LazyArray<ForeignModule> {
        empty_proc_macro!(self);
        let foreign_modules = self.tcx.foreign_modules(LOCAL_CRATE);
        self.lazy_array(foreign_modules.iter().map(|(_, m)| m).cloned())
    }

    fn encode_hygiene(&mut self) -> (SyntaxContextTable, ExpnDataTable, ExpnHashTable) {
        let mut syntax_contexts: TableBuilder<_, _> = Default::default();
        let mut expn_data_table: TableBuilder<_, _> = Default::default();
        let mut expn_hash_table: TableBuilder<_, _> = Default::default();

        self.hygiene_ctxt.encode(
            &mut (&mut *self, &mut syntax_contexts, &mut expn_data_table, &mut expn_hash_table),
            |(this, syntax_contexts, _, _), index, ctxt_data| {
                syntax_contexts.set_some(index, this.lazy(ctxt_data));
            },
            |(this, _, expn_data_table, expn_hash_table), index, expn_data, hash| {
                if let Some(index) = index.as_local() {
                    expn_data_table.set_some(index.as_raw(), this.lazy(expn_data));
                    expn_hash_table.set_some(index.as_raw(), this.lazy(hash));
                }
            },
        );

        (
            syntax_contexts.encode(&mut self.opaque),
            expn_data_table.encode(&mut self.opaque),
            expn_hash_table.encode(&mut self.opaque),
        )
    }

    fn encode_proc_macros(&mut self) -> Option<ProcMacroData> {
        let is_proc_macro = self.tcx.crate_types().contains(&CrateType::ProcMacro);
        if is_proc_macro {
            let tcx = self.tcx;
            let proc_macro_decls_static = tcx.proc_macro_decls_static(()).unwrap().local_def_index;
            let stability = tcx.lookup_stability(CRATE_DEF_ID);
            let macros =
                self.lazy_array(tcx.resolutions(()).proc_macros.iter().map(|p| p.local_def_index));
            for (i, span) in self.tcx.sess.psess.proc_macro_quoted_spans() {
                let span = self.lazy(span);
                self.tables.proc_macro_quoted_spans.set_some(i, span);
            }

            self.tables.def_kind.set_some(LOCAL_CRATE.as_def_id().index, DefKind::Mod);
            record!(self.tables.def_span[LOCAL_CRATE.as_def_id()] <- tcx.def_span(LOCAL_CRATE.as_def_id()));
            self.encode_attrs(LOCAL_CRATE.as_def_id().expect_local());
            let vis = tcx.local_visibility(CRATE_DEF_ID).map_id(|def_id| def_id.local_def_index);
            record!(self.tables.visibility[LOCAL_CRATE.as_def_id()] <- vis);
            if let Some(stability) = stability {
                record!(self.tables.lookup_stability[LOCAL_CRATE.as_def_id()] <- stability);
            }
            self.encode_deprecation(LOCAL_CRATE.as_def_id());
            if let Some(res_map) = tcx.resolutions(()).doc_link_resolutions.get(&CRATE_DEF_ID) {
                record!(self.tables.doc_link_resolutions[LOCAL_CRATE.as_def_id()] <- res_map);
            }
            if let Some(traits) = tcx.resolutions(()).doc_link_traits_in_scope.get(&CRATE_DEF_ID) {
                record_array!(self.tables.doc_link_traits_in_scope[LOCAL_CRATE.as_def_id()] <- traits);
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
                let macro_kind = if ast::attr::contains_name(attrs, sym::proc_macro) {
                    MacroKind::Bang
                } else if ast::attr::contains_name(attrs, sym::proc_macro_attribute) {
                    MacroKind::Attr
                } else if let Some(attr) = ast::attr::find_by_name(attrs, sym::proc_macro_derive) {
                    // This unwrap chain should have been checked by the proc-macro harness.
                    name = attr.meta_item_list().unwrap()[0]
                        .meta_item()
                        .unwrap()
                        .ident()
                        .unwrap()
                        .name;
                    MacroKind::Derive
                } else {
                    bug!("Unknown proc-macro type for item {:?}", id);
                };

                let mut def_key = self.tcx.hir_def_key(id);
                def_key.disambiguated_data.data = DefPathData::MacroNs(name);

                let def_id = id.to_def_id();
                self.tables.def_kind.set_some(def_id.index, DefKind::Macro(macro_kind));
                self.tables.proc_macro.set_some(def_id.index, macro_kind);
                self.encode_attrs(id);
                record!(self.tables.def_keys[def_id] <- def_key);
                record!(self.tables.def_ident_span[def_id] <- span);
                record!(self.tables.def_span[def_id] <- span);
                record!(self.tables.visibility[def_id] <- ty::Visibility::Public);
                if let Some(stability) = stability {
                    record!(self.tables.lookup_stability[def_id] <- stability);
                }
            }

            Some(ProcMacroData { proc_macro_decls_static, stability, macros })
        } else {
            None
        }
    }

    fn encode_debugger_visualizers(&mut self) -> LazyArray<DebuggerVisualizerFile> {
        empty_proc_macro!(self);
        self.lazy_array(
            self.tcx
                .debugger_visualizers(LOCAL_CRATE)
                .iter()
                // Erase the path since it may contain privacy sensitive data
                // that we don't want to end up in crate metadata.
                // The path is only needed for the local crate because of
                // `--emit dep-info`.
                .map(DebuggerVisualizerFile::path_erased),
        )
    }

    fn encode_crate_deps(&mut self) -> LazyArray<CrateDep> {
        empty_proc_macro!(self);

        let deps = self
            .tcx
            .crates(())
            .iter()
            .map(|&cnum| {
                let dep = CrateDep {
                    name: self.tcx.crate_name(cnum),
                    hash: self.tcx.crate_hash(cnum),
                    host_hash: self.tcx.crate_host_hash(cnum),
                    kind: self.tcx.dep_kind(cnum),
                    extra_filename: self.tcx.extra_filename(cnum).clone(),
                    is_private: self.tcx.is_private_dep(cnum),
                };
                (cnum, dep)
            })
            .collect::<Vec<_>>();

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
        self.lazy_array(deps.iter().map(|(_, dep)| dep))
    }

    fn encode_target_modifiers(&mut self) -> LazyArray<TargetModifier> {
        empty_proc_macro!(self);
        let tcx = self.tcx;
        self.lazy_array(tcx.sess.opts.gather_target_modifiers())
    }

    fn encode_lib_features(&mut self) -> LazyArray<(Symbol, FeatureStability)> {
        empty_proc_macro!(self);
        let tcx = self.tcx;
        let lib_features = tcx.lib_features(LOCAL_CRATE);
        self.lazy_array(lib_features.to_sorted_vec())
    }

    fn encode_stability_implications(&mut self) -> LazyArray<(Symbol, Symbol)> {
        empty_proc_macro!(self);
        let tcx = self.tcx;
        let implications = tcx.stability_implications(LOCAL_CRATE);
        let sorted = implications.to_sorted_stable_ord();
        self.lazy_array(sorted.into_iter().map(|(k, v)| (*k, *v)))
    }

    fn encode_diagnostic_items(&mut self) -> LazyArray<(Symbol, DefIndex)> {
        empty_proc_macro!(self);
        let tcx = self.tcx;
        let diagnostic_items = &tcx.diagnostic_items(LOCAL_CRATE).name_to_id;
        self.lazy_array(diagnostic_items.iter().map(|(&name, def_id)| (name, def_id.index)))
    }

    fn encode_lang_items(&mut self) -> LazyArray<(DefIndex, LangItem)> {
        empty_proc_macro!(self);
        let lang_items = self.tcx.lang_items().iter();
        self.lazy_array(lang_items.filter_map(|(lang_item, def_id)| {
            def_id.as_local().map(|id| (id.local_def_index, lang_item))
        }))
    }

    fn encode_lang_items_missing(&mut self) -> LazyArray<LangItem> {
        empty_proc_macro!(self);
        let tcx = self.tcx;
        self.lazy_array(&tcx.lang_items().missing)
    }

    fn encode_stripped_cfg_items(&mut self) -> LazyArray<StrippedCfgItem<DefIndex>> {
        self.lazy_array(
            self.tcx
                .stripped_cfg_items(LOCAL_CRATE)
                .into_iter()
                .map(|item| item.clone().map_mod_id(|def_id| def_id.index)),
        )
    }

    fn encode_traits(&mut self) -> LazyArray<DefIndex> {
        empty_proc_macro!(self);
        self.lazy_array(self.tcx.traits(LOCAL_CRATE).iter().map(|def_id| def_id.index))
    }

    /// Encodes an index, mapping each trait to its (local) implementations.
    #[instrument(level = "debug", skip(self))]
    fn encode_impls(&mut self) -> LazyArray<TraitImpls> {
        empty_proc_macro!(self);
        let tcx = self.tcx;
        let mut trait_impls: FxIndexMap<DefId, Vec<(DefIndex, Option<SimplifiedType>)>> =
            FxIndexMap::default();

        for id in tcx.hir_free_items() {
            let DefKind::Impl { of_trait } = tcx.def_kind(id.owner_id) else {
                continue;
            };
            let def_id = id.owner_id.to_def_id();

            self.tables.defaultness.set_some(def_id.index, tcx.defaultness(def_id));

            if of_trait && let Some(header) = tcx.impl_trait_header(def_id) {
                record!(self.tables.impl_trait_header[def_id] <- header);

                let trait_ref = header.trait_ref.instantiate_identity();
                let simplified_self_ty = fast_reject::simplify_type(
                    self.tcx,
                    trait_ref.self_ty(),
                    TreatParams::InstantiateWithInfer,
                );
                trait_impls
                    .entry(trait_ref.def_id)
                    .or_default()
                    .push((id.owner_id.def_id.local_def_index, simplified_self_ty));

                let trait_def = tcx.trait_def(trait_ref.def_id);
                if let Ok(mut an) = trait_def.ancestors(tcx, def_id) {
                    if let Some(specialization_graph::Node::Impl(parent)) = an.nth(1) {
                        self.tables.impl_parent.set_some(def_id.index, parent.into());
                    }
                }

                // if this is an impl of `CoerceUnsized`, create its
                // "unsized info", else just store None
                if tcx.is_lang_item(trait_ref.def_id, LangItem::CoerceUnsized) {
                    let coerce_unsized_info = tcx.coerce_unsized_info(def_id).unwrap();
                    record!(self.tables.coerce_unsized_info[def_id] <- coerce_unsized_info);
                }
            }
        }

        let trait_impls: Vec<_> = trait_impls
            .into_iter()
            .map(|(trait_def_id, impls)| TraitImpls {
                trait_id: (trait_def_id.krate.as_u32(), trait_def_id.index),
                impls: self.lazy_array(&impls),
            })
            .collect();

        self.lazy_array(&trait_impls)
    }

    #[instrument(level = "debug", skip(self))]
    fn encode_incoherent_impls(&mut self) -> LazyArray<IncoherentImpls> {
        empty_proc_macro!(self);
        let tcx = self.tcx;

        let all_impls: Vec<_> = tcx
            .crate_inherent_impls(())
            .0
            .incoherent_impls
            .iter()
            .map(|(&simp, impls)| IncoherentImpls {
                self_ty: simp,
                impls: self.lazy_array(impls.iter().map(|def_id| def_id.local_def_index)),
            })
            .collect();

        self.lazy_array(&all_impls)
    }

    fn encode_exportable_items(&mut self) -> LazyArray<DefIndex> {
        empty_proc_macro!(self);
        self.lazy_array(self.tcx.exportable_items(LOCAL_CRATE).iter().map(|def_id| def_id.index))
    }

    fn encode_stable_order_of_exportable_impls(&mut self) -> LazyArray<(DefIndex, usize)> {
        empty_proc_macro!(self);
        let stable_order_of_exportable_impls =
            self.tcx.stable_order_of_exportable_impls(LOCAL_CRATE);
        self.lazy_array(
            stable_order_of_exportable_impls.iter().map(|(def_id, idx)| (def_id.index, *idx)),
        )
    }

    // Encodes all symbols exported from this crate into the metadata.
    //
    // This pass is seeded off the reachability list calculated in the
    // middle::reachable module but filters out items that either don't have a
    // symbol associated with them (they weren't translated) or if they're an FFI
    // definition (as that's not defined in this crate).
    fn encode_exported_symbols(
        &mut self,
        exported_symbols: &[(ExportedSymbol<'tcx>, SymbolExportInfo)],
    ) -> LazyArray<(ExportedSymbol<'static>, SymbolExportInfo)> {
        empty_proc_macro!(self);
        // The metadata symbol name is special. It should not show up in
        // downstream crates.
        let metadata_symbol_name = SymbolName::new(self.tcx, &metadata_symbol_name(self.tcx));

        self.lazy_array(
            exported_symbols
                .iter()
                .filter(|&(exported_symbol, _)| match *exported_symbol {
                    ExportedSymbol::NoDefId(symbol_name) => symbol_name != metadata_symbol_name,
                    _ => true,
                })
                .cloned(),
        )
    }

    fn encode_dylib_dependency_formats(&mut self) -> LazyArray<Option<LinkagePreference>> {
        empty_proc_macro!(self);
        let formats = self.tcx.dependency_formats(());
        if let Some(arr) = formats.get(&CrateType::Dylib) {
            return self.lazy_array(arr.iter().skip(1 /* skip LOCAL_CRATE */).map(
                |slot| match *slot {
                    Linkage::NotLinked | Linkage::IncludedFromDylib => None,

                    Linkage::Dynamic => Some(LinkagePreference::RequireDynamic),
                    Linkage::Static => Some(LinkagePreference::RequireStatic),
                },
            ));
        }
        LazyArray::default()
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
            return Ok(Self { full_metadata: None, stub_metadata: None, _temp_dir: None });
        }
        let full_mmap = unsafe { Some(Mmap::map(file)?) };

        let stub =
            if let Some(stub_path) = stub_path { Some(std::fs::read(stub_path)?) } else { None };

        Ok(Self { full_metadata: full_mmap, stub_metadata: stub, _temp_dir: temp_dir })
    }

    #[inline]
    pub fn full(&self) -> &[u8] {
        &self.full_metadata.as_deref().unwrap_or_default()
    }

    #[inline]
    pub fn stub_or_full(&self) -> &[u8] {
        self.stub_metadata.as_deref().unwrap_or(self.full())
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

        Self { full_metadata, stub_metadata: stub, _temp_dir: None }
    }
}

pub fn encode_metadata(tcx: TyCtxt<'_>, path: &Path, ref_path: Option<&Path>) {
    let _prof_timer = tcx.prof.verbose_generic_activity("generate_crate_metadata");

    // Since encoding metadata is not in a query, and nothing is cached,
    // there's no need to do dep-graph tracking for any of it.
    tcx.dep_graph.assert_ignored();

    if tcx.sess.threads() != 1 {
        // Prefetch some queries used by metadata encoding.
        // This is not necessary for correctness, but is only done for performance reasons.
        // It can be removed if it turns out to cause trouble or be detrimental to performance.
        join(|| prefetch_mir(tcx), || tcx.exported_symbols(LOCAL_CRATE));
    }

    with_encode_metadata_header(tcx, path, |ecx| {
        // Encode all the entries and extra information in the crate,
        // culminating in the `CrateRoot` which points to all of it.
        let root = ecx.encode_crate_root();

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

    if let Some(ref_path) = ref_path {
        with_encode_metadata_header(tcx, ref_path, |ecx| {
            let header: LazyValue<CrateHeader> = ecx.lazy(CrateHeader {
                name: tcx.crate_name(LOCAL_CRATE),
                triple: tcx.sess.opts.target_triple.clone(),
                hash: tcx.crate_hash(LOCAL_CRATE),
                is_proc_macro_crate: false,
                is_stub: true,
            });
            header.position.get()
        });
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
        symbol_table: Default::default(),
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
            // FIXME: Can they contain const arguments and thus leak private struct fields?
            hir::ExprKind::Path(hir::QPath::LangItem(..)) => Simple,
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
