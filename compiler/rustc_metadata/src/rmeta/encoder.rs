use crate::rmeta::table::{FixedSizeEncoding, TableBuilder};
use crate::rmeta::*;

use rustc_ast as ast;
use rustc_data_structures::fingerprint::{Fingerprint, FingerprintEncoder};
use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_data_structures::stable_hasher::StableHasher;
use rustc_data_structures::sync::{join, Lrc};
use rustc_hir as hir;
use rustc_hir::def::CtorKind;
use rustc_hir::def_id::{CrateNum, DefId, DefIndex, LocalDefId, CRATE_DEF_INDEX, LOCAL_CRATE};
use rustc_hir::definitions::DefPathData;
use rustc_hir::intravisit::{self, NestedVisitorMap, Visitor};
use rustc_hir::itemlikevisit::{ItemLikeVisitor, ParItemLikeVisitor};
use rustc_hir::lang_items;
use rustc_hir::{AnonConst, GenericParamKind};
use rustc_index::bit_set::GrowableBitSet;
use rustc_index::vec::Idx;
use rustc_middle::hir::map::Map;
use rustc_middle::middle::cstore::{EncodedMetadata, ForeignModule, LinkagePreference, NativeLib};
use rustc_middle::middle::dependency_format::Linkage;
use rustc_middle::middle::exported_symbols::{
    metadata_symbol_name, ExportedSymbol, SymbolExportLevel,
};
use rustc_middle::mir::interpret;
use rustc_middle::traits::specialization_graph;
use rustc_middle::ty::codec::TyEncoder;
use rustc_middle::ty::{self, SymbolName, Ty, TyCtxt};
use rustc_serialize::{opaque, Encodable, Encoder};
use rustc_session::config::CrateType;
use rustc_span::hygiene::{ExpnDataEncodeMode, HygieneEncodeContext, MacroKind};
use rustc_span::symbol::{sym, Ident, Symbol};
use rustc_span::{self, ExternalSource, FileName, SourceFile, Span, SyntaxContext};
use rustc_target::abi::VariantIdx;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::path::Path;
use tracing::{debug, trace};

pub(super) struct EncodeContext<'a, 'tcx> {
    opaque: opaque::Encoder,
    tcx: TyCtxt<'tcx>,
    feat: &'tcx rustc_feature::Features,

    tables: TableBuilders<'tcx>,

    lazy_state: LazyState,
    type_shorthands: FxHashMap<Ty<'tcx>, usize>,
    predicate_shorthands: FxHashMap<ty::Predicate<'tcx>, usize>,

    interpret_allocs: FxIndexSet<interpret::AllocId>,

    // This is used to speed up Span encoding.
    // The `usize` is an index into the `MonotonicVec`
    // that stores the `SourceFile`
    source_file_cache: (Lrc<SourceFile>, usize),
    // The indices (into the `SourceMap`'s `MonotonicVec`)
    // of all of the `SourceFiles` that we need to serialize.
    // When we serialize a `Span`, we insert the index of its
    // `SourceFile` into the `GrowableBitSet`.
    //
    // This needs to be a `GrowableBitSet` and not a
    // regular `BitSet` because we may actually import new `SourceFiles`
    // during metadata encoding, due to executing a query
    // with a result containing a foreign `Span`.
    required_source_files: Option<GrowableBitSet<usize>>,
    is_proc_macro: bool,
    hygiene_ctxt: &'a HygieneEncodeContext,
}

/// If the current crate is a proc-macro, returns early with `Lazy:empty()`.
/// This is useful for skipping the encoding of things that aren't needed
/// for proc-macro crates.
macro_rules! empty_proc_macro {
    ($self:ident) => {
        if $self.is_proc_macro {
            return Lazy::empty();
        }
    };
}

macro_rules! encoder_methods {
    ($($name:ident($ty:ty);)*) => {
        $(fn $name(&mut self, value: $ty) -> Result<(), Self::Error> {
            self.opaque.$name(value)
        })*
    }
}

impl<'a, 'tcx> Encoder for EncodeContext<'a, 'tcx> {
    type Error = <opaque::Encoder as Encoder>::Error;

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

impl<'a, 'tcx, T: Encodable<EncodeContext<'a, 'tcx>>> Encodable<EncodeContext<'a, 'tcx>>
    for Lazy<T>
{
    fn encode(&self, e: &mut EncodeContext<'a, 'tcx>) -> opaque::EncodeResult {
        e.emit_lazy_distance(*self)
    }
}

impl<'a, 'tcx, T: Encodable<EncodeContext<'a, 'tcx>>> Encodable<EncodeContext<'a, 'tcx>>
    for Lazy<[T]>
{
    fn encode(&self, e: &mut EncodeContext<'a, 'tcx>) -> opaque::EncodeResult {
        e.emit_usize(self.meta)?;
        if self.meta == 0 {
            return Ok(());
        }
        e.emit_lazy_distance(*self)
    }
}

impl<'a, 'tcx, I: Idx, T: Encodable<EncodeContext<'a, 'tcx>>> Encodable<EncodeContext<'a, 'tcx>>
    for Lazy<Table<I, T>>
where
    Option<T>: FixedSizeEncoding,
{
    fn encode(&self, e: &mut EncodeContext<'a, 'tcx>) -> opaque::EncodeResult {
        e.emit_usize(self.meta)?;
        e.emit_lazy_distance(*self)
    }
}

impl<'a, 'tcx> Encodable<EncodeContext<'a, 'tcx>> for CrateNum {
    fn encode(&self, s: &mut EncodeContext<'a, 'tcx>) -> opaque::EncodeResult {
        if *self != LOCAL_CRATE && s.is_proc_macro {
            panic!("Attempted to encode non-local CrateNum {:?} for proc-macro crate", self);
        }
        s.emit_u32(self.as_u32())
    }
}

impl<'a, 'tcx> Encodable<EncodeContext<'a, 'tcx>> for DefIndex {
    fn encode(&self, s: &mut EncodeContext<'a, 'tcx>) -> opaque::EncodeResult {
        s.emit_u32(self.as_u32())
    }
}

impl<'a, 'tcx> Encodable<EncodeContext<'a, 'tcx>> for SyntaxContext {
    fn encode(&self, s: &mut EncodeContext<'a, 'tcx>) -> opaque::EncodeResult {
        rustc_span::hygiene::raw_encode_syntax_context(*self, &s.hygiene_ctxt, s)
    }
}

impl<'a, 'tcx> Encodable<EncodeContext<'a, 'tcx>> for ExpnId {
    fn encode(&self, s: &mut EncodeContext<'a, 'tcx>) -> opaque::EncodeResult {
        rustc_span::hygiene::raw_encode_expn_id(
            *self,
            &s.hygiene_ctxt,
            ExpnDataEncodeMode::Metadata,
            s,
        )
    }
}

impl<'a, 'tcx> Encodable<EncodeContext<'a, 'tcx>> for Span {
    fn encode(&self, s: &mut EncodeContext<'a, 'tcx>) -> opaque::EncodeResult {
        if *self == rustc_span::DUMMY_SP {
            return TAG_INVALID_SPAN.encode(s);
        }

        let span = self.data();

        // The Span infrastructure should make sure that this invariant holds:
        debug_assert!(span.lo <= span.hi);

        if !s.source_file_cache.0.contains(span.lo) {
            let source_map = s.tcx.sess.source_map();
            let source_file_index = source_map.lookup_source_file_idx(span.lo);
            s.source_file_cache =
                (source_map.files()[source_file_index].clone(), source_file_index);
        }

        if !s.source_file_cache.0.contains(span.hi) {
            // Unfortunately, macro expansion still sometimes generates Spans
            // that malformed in this way.
            return TAG_INVALID_SPAN.encode(s);
        }

        let source_files = s.required_source_files.as_mut().expect("Already encoded SourceMap!");
        // Record the fact that we need to encode the data for this `SourceFile`
        source_files.insert(s.source_file_cache.1);

        // There are two possible cases here:
        // 1. This span comes from a 'foreign' crate - e.g. some crate upstream of the
        // crate we are writing metadata for. When the metadata for *this* crate gets
        // deserialized, the deserializer will need to know which crate it originally came
        // from. We use `TAG_VALID_SPAN_FOREIGN` to indicate that a `CrateNum` should
        // be deserialized after the rest of the span data, which tells the deserializer
        // which crate contains the source map information.
        // 2. This span comes from our own crate. No special hamdling is needed - we just
        // write `TAG_VALID_SPAN_LOCAL` to let the deserializer know that it should use
        // our own source map information.
        //
        // If we're a proc-macro crate, we always treat this as a local `Span`.
        // In `encode_source_map`, we serialize foreign `SourceFile`s into our metadata
        // if we're a proc-macro crate.
        // This allows us to avoid loading the dependencies of proc-macro crates: all of
        // the information we need to decode `Span`s is stored in the proc-macro crate.
        let (tag, lo, hi) = if s.source_file_cache.0.is_imported() && !s.is_proc_macro {
            // To simplify deserialization, we 'rebase' this span onto the crate it originally came from
            // (the crate that 'owns' the file it references. These rebased 'lo' and 'hi' values
            // are relative to the source map information for the 'foreign' crate whose CrateNum
            // we write into the metadata. This allows `imported_source_files` to binary
            // search through the 'foreign' crate's source map information, using the
            // deserialized 'lo' and 'hi' values directly.
            //
            // All of this logic ensures that the final result of deserialization is a 'normal'
            // Span that can be used without any additional trouble.
            let external_start_pos = {
                // Introduce a new scope so that we drop the 'lock()' temporary
                match &*s.source_file_cache.0.external_src.lock() {
                    ExternalSource::Foreign { original_start_pos, .. } => *original_start_pos,
                    src => panic!("Unexpected external source {:?}", src),
                }
            };
            let lo = (span.lo - s.source_file_cache.0.start_pos) + external_start_pos;
            let hi = (span.hi - s.source_file_cache.0.start_pos) + external_start_pos;

            (TAG_VALID_SPAN_FOREIGN, lo, hi)
        } else {
            (TAG_VALID_SPAN_LOCAL, span.lo, span.hi)
        };

        tag.encode(s)?;
        lo.encode(s)?;

        // Encode length which is usually less than span.hi and profits more
        // from the variable-length integer encoding that we use.
        let len = hi - lo;
        len.encode(s)?;

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
        if s.is_proc_macro {
            SyntaxContext::root().encode(s)?;
        } else {
            span.ctxt.encode(s)?;
        }

        if tag == TAG_VALID_SPAN_FOREIGN {
            // This needs to be two lines to avoid holding the `s.source_file_cache`
            // while calling `cnum.encode(s)`
            let cnum = s.source_file_cache.0.cnum;
            cnum.encode(s)?;
        }

        Ok(())
    }
}

impl<'a, 'tcx> FingerprintEncoder for EncodeContext<'a, 'tcx> {
    fn encode_fingerprint(&mut self, f: &Fingerprint) -> Result<(), Self::Error> {
        f.encode_opaque(&mut self.opaque)
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

    fn predicate_shorthands(&mut self) -> &mut FxHashMap<rustc_middle::ty::Predicate<'tcx>, usize> {
        &mut self.predicate_shorthands
    }

    fn encode_alloc_id(
        &mut self,
        alloc_id: &rustc_middle::mir::interpret::AllocId,
    ) -> Result<(), Self::Error> {
        let (index, _) = self.interpret_allocs.insert_full(*alloc_id);

        index.encode(self)
    }
}

impl<'a, 'tcx> Encodable<EncodeContext<'a, 'tcx>> for &'tcx [mir::abstract_const::Node<'tcx>] {
    fn encode(&self, s: &mut EncodeContext<'a, 'tcx>) -> opaque::EncodeResult {
        (**self).encode(s)
    }
}

impl<'a, 'tcx> Encodable<EncodeContext<'a, 'tcx>> for &'tcx [(ty::Predicate<'tcx>, Span)] {
    fn encode(&self, s: &mut EncodeContext<'a, 'tcx>) -> opaque::EncodeResult {
        (**self).encode(s)
    }
}

/// Helper trait to allow overloading `EncodeContext::lazy` for iterators.
trait EncodeContentsForLazy<'a, 'tcx, T: ?Sized + LazyMeta> {
    fn encode_contents_for_lazy(self, ecx: &mut EncodeContext<'a, 'tcx>) -> T::Meta;
}

impl<'a, 'tcx, T: Encodable<EncodeContext<'a, 'tcx>>> EncodeContentsForLazy<'a, 'tcx, T> for &T {
    fn encode_contents_for_lazy(self, ecx: &mut EncodeContext<'a, 'tcx>) {
        self.encode(ecx).unwrap()
    }
}

impl<'a, 'tcx, T: Encodable<EncodeContext<'a, 'tcx>>> EncodeContentsForLazy<'a, 'tcx, T> for T {
    fn encode_contents_for_lazy(self, ecx: &mut EncodeContext<'a, 'tcx>) {
        self.encode(ecx).unwrap()
    }
}

impl<'a, 'tcx, I, T: Encodable<EncodeContext<'a, 'tcx>>> EncodeContentsForLazy<'a, 'tcx, [T]> for I
where
    I: IntoIterator,
    I::Item: EncodeContentsForLazy<'a, 'tcx, T>,
{
    fn encode_contents_for_lazy(self, ecx: &mut EncodeContext<'a, 'tcx>) -> usize {
        self.into_iter().map(|value| value.encode_contents_for_lazy(ecx)).count()
    }
}

// Shorthand for `$self.$tables.$table.set($def_id.index, $self.lazy($value))`, which would
// normally need extra variables to avoid errors about multiple mutable borrows.
macro_rules! record {
    ($self:ident.$tables:ident.$table:ident[$def_id:expr] <- $value:expr) => {{
        {
            let value = $value;
            let lazy = $self.lazy(value);
            $self.$tables.$table.set($def_id.index, lazy);
        }
    }};
}

impl<'a, 'tcx> EncodeContext<'a, 'tcx> {
    fn emit_lazy_distance<T: ?Sized + LazyMeta>(
        &mut self,
        lazy: Lazy<T>,
    ) -> Result<(), <Self as Encoder>::Error> {
        let min_end = lazy.position.get() + T::min_size(lazy.meta);
        let distance = match self.lazy_state {
            LazyState::NoNode => bug!("emit_lazy_distance: outside of a metadata node"),
            LazyState::NodeStart(start) => {
                let start = start.get();
                assert!(min_end <= start);
                start - min_end
            }
            LazyState::Previous(last_min_end) => {
                assert!(
                    last_min_end <= lazy.position,
                    "make sure that the calls to `lazy*` \
                     are in the same order as the metadata fields",
                );
                lazy.position.get() - last_min_end.get()
            }
        };
        self.lazy_state = LazyState::Previous(NonZeroUsize::new(min_end).unwrap());
        self.emit_usize(distance)
    }

    fn lazy<T: ?Sized + LazyMeta>(
        &mut self,
        value: impl EncodeContentsForLazy<'a, 'tcx, T>,
    ) -> Lazy<T> {
        let pos = NonZeroUsize::new(self.position()).unwrap();

        assert_eq!(self.lazy_state, LazyState::NoNode);
        self.lazy_state = LazyState::NodeStart(pos);
        let meta = value.encode_contents_for_lazy(self);
        self.lazy_state = LazyState::NoNode;

        assert!(pos.get() + <T>::min_size(meta) <= self.position());

        Lazy::from_position_and_meta(pos, meta)
    }

    fn encode_info_for_items(&mut self) {
        let krate = self.tcx.hir().krate();
        self.encode_info_for_mod(hir::CRATE_HIR_ID, &krate.item.module, &krate.item.attrs);

        // Proc-macro crates only export proc-macro items, which are looked
        // up using `proc_macro_data`
        if self.is_proc_macro {
            return;
        }

        krate.visit_all_item_likes(&mut self.as_deep_visitor());
        for macro_def in krate.exported_macros {
            self.visit_macro_def(macro_def);
        }
    }

    fn encode_def_path_table(&mut self) {
        let table = self.tcx.hir().definitions().def_path_table();
        if self.is_proc_macro {
            for def_index in std::iter::once(CRATE_DEF_INDEX)
                .chain(self.tcx.hir().krate().proc_macros.iter().map(|p| p.owner.local_def_index))
            {
                let def_key = self.lazy(table.def_key(def_index));
                let def_path_hash = self.lazy(table.def_path_hash(def_index));
                self.tables.def_keys.set(def_index, def_key);
                self.tables.def_path_hashes.set(def_index, def_path_hash);
            }
        } else {
            for (def_index, def_key, def_path_hash) in table.enumerated_keys_and_path_hashes() {
                let def_key = self.lazy(def_key);
                let def_path_hash = self.lazy(def_path_hash);
                self.tables.def_keys.set(def_index, def_key);
                self.tables.def_path_hashes.set(def_index, def_path_hash);
            }
        }
    }

    fn encode_source_map(&mut self) -> Lazy<[rustc_span::SourceFile]> {
        let source_map = self.tcx.sess.source_map();
        let all_source_files = source_map.files();

        let (working_dir, _cwd_remapped) = self.tcx.sess.working_dir.clone();
        // By replacing the `Option` with `None`, we ensure that we can't
        // accidentally serialize any more `Span`s after the source map encoding
        // is done.
        let required_source_files = self.required_source_files.take().unwrap();

        let adapted = all_source_files
            .iter()
            .enumerate()
            .filter(|(idx, source_file)| {
                // Only serialize `SourceFile`s that were used
                // during the encoding of a `Span`
                required_source_files.contains(*idx) &&
                // Don't serialize imported `SourceFile`s, unless
                // we're in a proc-macro crate.
                (!source_file.is_imported() || self.is_proc_macro)
            })
            .map(|(_, source_file)| {
                let mut adapted = match source_file.name {
                    // This path of this SourceFile has been modified by
                    // path-remapping, so we use it verbatim (and avoid
                    // cloning the whole map in the process).
                    _ if source_file.name_was_remapped => source_file.clone(),

                    // Otherwise expand all paths to absolute paths because
                    // any relative paths are potentially relative to a
                    // wrong directory.
                    FileName::Real(ref name) => {
                        let name = name.stable_name();
                        let mut adapted = (**source_file).clone();
                        adapted.name = Path::new(&working_dir).join(name).into();
                        adapted.name_hash = {
                            let mut hasher: StableHasher = StableHasher::new();
                            adapted.name.hash(&mut hasher);
                            hasher.finish::<u128>()
                        };
                        Lrc::new(adapted)
                    }

                    // expanded code, not from a file
                    _ => source_file.clone(),
                };

                // We're serializing this `SourceFile` into our crate metadata,
                // so mark it as coming from this crate.
                // This also ensures that we don't try to deserialize the
                // `CrateNum` for a proc-macro dependency - since proc macro
                // dependencies aren't loaded when we deserialize a proc-macro,
                // trying to remap the `CrateNum` would fail.
                if self.is_proc_macro {
                    Lrc::make_mut(&mut adapted).cnum = LOCAL_CRATE;
                }
                adapted
            })
            .collect::<Vec<_>>();

        self.lazy(adapted.iter().map(|rc| &**rc))
    }

    fn encode_crate_root(&mut self) -> Lazy<CrateRoot<'tcx>> {
        let mut i = self.position();

        // Encode the crate deps
        let crate_deps = self.encode_crate_deps();
        let dylib_dependency_formats = self.encode_dylib_dependency_formats();
        let dep_bytes = self.position() - i;

        // Encode the lib features.
        i = self.position();
        let lib_features = self.encode_lib_features();
        let lib_feature_bytes = self.position() - i;

        // Encode the language items.
        i = self.position();
        let lang_items = self.encode_lang_items();
        let lang_items_missing = self.encode_lang_items_missing();
        let lang_item_bytes = self.position() - i;

        // Encode the diagnostic items.
        i = self.position();
        let diagnostic_items = self.encode_diagnostic_items();
        let diagnostic_item_bytes = self.position() - i;

        // Encode the native libraries used
        i = self.position();
        let native_libraries = self.encode_native_libraries();
        let native_lib_bytes = self.position() - i;

        let foreign_modules = self.encode_foreign_modules();

        // Encode DefPathTable
        i = self.position();
        self.encode_def_path_table();
        let def_path_table_bytes = self.position() - i;

        // Encode the def IDs of impls, for coherence checking.
        i = self.position();
        let impls = self.encode_impls();
        let impl_bytes = self.position() - i;

        let tcx = self.tcx;

        // Encode the items.
        i = self.position();
        self.encode_info_for_items();
        let item_bytes = self.position() - i;

        // Encode the allocation index
        let interpret_alloc_index = {
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
                    let pos = self.position() as u32;
                    interpret_alloc_index.push(pos);
                    interpret::specialized_encode_alloc_id(self, tcx, id).unwrap();
                }
                n = new_n;
            }
            self.lazy(interpret_alloc_index)
        };

        // Encode the proc macro data. This affects 'tables',
        // so we need to do this before we encode the tables
        i = self.position();
        let proc_macro_data = self.encode_proc_macros();
        let proc_macro_data_bytes = self.position() - i;

        i = self.position();
        let tables = self.tables.encode(&mut self.opaque);
        let tables_bytes = self.position() - i;

        // Encode exported symbols info. This is prefetched in `encode_metadata` so we encode
        // this as late as possible to give the prefetching as much time as possible to complete.
        i = self.position();
        let exported_symbols = tcx.exported_symbols(LOCAL_CRATE);
        let exported_symbols = self.encode_exported_symbols(&exported_symbols);
        let exported_symbols_bytes = self.position() - i;

        // Encode the hygiene data,
        // IMPORTANT: this *must* be the last thing that we encode (other than `SourceMap`). The process
        // of encoding other items (e.g. `optimized_mir`) may cause us to load
        // data from the incremental cache. If this causes us to deserialize a `Span`,
        // then we may load additional `SyntaxContext`s into the global `HygieneData`.
        // Therefore, we need to encode the hygiene data last to ensure that we encode
        // any `SyntaxContext`s that might be used.
        i = self.position();
        let (syntax_contexts, expn_data) = self.encode_hygiene();
        let hygiene_bytes = self.position() - i;

        // Encode source_map. This needs to be done last,
        // since encoding `Span`s tells us which `SourceFiles` we actually
        // need to encode.
        i = self.position();
        let source_map = self.encode_source_map();
        let source_map_bytes = self.position() - i;

        let attrs = tcx.hir().krate_attrs();
        let has_default_lib_allocator = tcx.sess.contains_name(&attrs, sym::default_lib_allocator);

        let root = self.lazy(CrateRoot {
            name: tcx.crate_name(LOCAL_CRATE),
            extra_filename: tcx.sess.opts.cg.extra_filename.clone(),
            triple: tcx.sess.opts.target_triple.clone(),
            hash: tcx.crate_hash(LOCAL_CRATE),
            disambiguator: tcx.sess.local_crate_disambiguator(),
            panic_strategy: tcx.sess.panic_strategy(),
            edition: tcx.sess.edition(),
            has_global_allocator: tcx.has_global_allocator(LOCAL_CRATE),
            has_panic_handler: tcx.has_panic_handler(LOCAL_CRATE),
            has_default_lib_allocator,
            plugin_registrar_fn: tcx.plugin_registrar_fn(LOCAL_CRATE).map(|id| id.index),
            proc_macro_data,
            compiler_builtins: tcx.sess.contains_name(&attrs, sym::compiler_builtins),
            needs_allocator: tcx.sess.contains_name(&attrs, sym::needs_allocator),
            needs_panic_runtime: tcx.sess.contains_name(&attrs, sym::needs_panic_runtime),
            no_builtins: tcx.sess.contains_name(&attrs, sym::no_builtins),
            panic_runtime: tcx.sess.contains_name(&attrs, sym::panic_runtime),
            profiler_runtime: tcx.sess.contains_name(&attrs, sym::profiler_runtime),
            symbol_mangling_version: tcx.sess.opts.debugging_opts.get_symbol_mangling_version(),

            crate_deps,
            dylib_dependency_formats,
            lib_features,
            lang_items,
            diagnostic_items,
            lang_items_missing,
            native_libraries,
            foreign_modules,
            source_map,
            impls,
            exported_symbols,
            interpret_alloc_index,
            tables,
            syntax_contexts,
            expn_data,
        });

        let total_bytes = self.position();

        if tcx.sess.meta_stats() {
            let mut zero_bytes = 0;
            for e in self.opaque.data.iter() {
                if *e == 0 {
                    zero_bytes += 1;
                }
            }

            println!("metadata stats:");
            println!("             dep bytes: {}", dep_bytes);
            println!("     lib feature bytes: {}", lib_feature_bytes);
            println!("       lang item bytes: {}", lang_item_bytes);
            println!(" diagnostic item bytes: {}", diagnostic_item_bytes);
            println!("          native bytes: {}", native_lib_bytes);
            println!("         source_map bytes: {}", source_map_bytes);
            println!("            impl bytes: {}", impl_bytes);
            println!("    exp. symbols bytes: {}", exported_symbols_bytes);
            println!("  def-path table bytes: {}", def_path_table_bytes);
            println!(" proc-macro-data-bytes: {}", proc_macro_data_bytes);
            println!("            item bytes: {}", item_bytes);
            println!("           table bytes: {}", tables_bytes);
            println!("         hygiene bytes: {}", hygiene_bytes);
            println!("            zero bytes: {}", zero_bytes);
            println!("           total bytes: {}", total_bytes);
        }

        root
    }
}

impl EncodeContext<'a, 'tcx> {
    fn encode_variances_of(&mut self, def_id: DefId) {
        debug!("EncodeContext::encode_variances_of({:?})", def_id);
        record!(self.tables.variances[def_id] <- &self.tcx.variances_of(def_id)[..]);
    }

    fn encode_item_type(&mut self, def_id: DefId) {
        debug!("EncodeContext::encode_item_type({:?})", def_id);
        record!(self.tables.ty[def_id] <- self.tcx.type_of(def_id));
    }

    fn encode_enum_variant_info(&mut self, def: &ty::AdtDef, index: VariantIdx) {
        let tcx = self.tcx;
        let variant = &def.variants[index];
        let def_id = variant.def_id;
        debug!("EncodeContext::encode_enum_variant_info({:?})", def_id);

        let data = VariantData {
            ctor_kind: variant.ctor_kind,
            discr: variant.discr,
            ctor: variant.ctor_def_id.map(|did| did.index),
            is_non_exhaustive: variant.is_field_list_non_exhaustive(),
        };

        record!(self.tables.kind[def_id] <- EntryKind::Variant(self.lazy(data)));
        record!(self.tables.visibility[def_id] <- self.tcx.visibility(def_id));
        record!(self.tables.span[def_id] <- self.tcx.def_span(def_id));
        record!(self.tables.attributes[def_id] <- &self.tcx.get_attrs(def_id)[..]);
        record!(self.tables.expn_that_defined[def_id] <- self.tcx.expansion_that_defined(def_id));
        record!(self.tables.children[def_id] <- variant.fields.iter().map(|f| {
            assert!(f.did.is_local());
            f.did.index
        }));
        self.encode_ident_span(def_id, variant.ident);
        self.encode_stability(def_id);
        self.encode_deprecation(def_id);
        self.encode_item_type(def_id);
        if variant.ctor_kind == CtorKind::Fn {
            // FIXME(eddyb) encode signature only in `encode_enum_variant_ctor`.
            if let Some(ctor_def_id) = variant.ctor_def_id {
                record!(self.tables.fn_sig[def_id] <- tcx.fn_sig(ctor_def_id));
            }
            // FIXME(eddyb) is this ever used?
            self.encode_variances_of(def_id);
        }
        self.encode_generics(def_id);
        self.encode_explicit_predicates(def_id);
        self.encode_inferred_outlives(def_id);
        self.encode_optimized_mir(def_id.expect_local());
        self.encode_promoted_mir(def_id.expect_local());
    }

    fn encode_enum_variant_ctor(&mut self, def: &ty::AdtDef, index: VariantIdx) {
        let tcx = self.tcx;
        let variant = &def.variants[index];
        let def_id = variant.ctor_def_id.unwrap();
        debug!("EncodeContext::encode_enum_variant_ctor({:?})", def_id);

        // FIXME(eddyb) encode only the `CtorKind` for constructors.
        let data = VariantData {
            ctor_kind: variant.ctor_kind,
            discr: variant.discr,
            ctor: Some(def_id.index),
            is_non_exhaustive: variant.is_field_list_non_exhaustive(),
        };

        record!(self.tables.kind[def_id] <- EntryKind::Variant(self.lazy(data)));
        record!(self.tables.visibility[def_id] <- self.tcx.visibility(def_id));
        record!(self.tables.span[def_id] <- self.tcx.def_span(def_id));
        self.encode_stability(def_id);
        self.encode_deprecation(def_id);
        self.encode_item_type(def_id);
        if variant.ctor_kind == CtorKind::Fn {
            record!(self.tables.fn_sig[def_id] <- tcx.fn_sig(def_id));
            self.encode_variances_of(def_id);
        }
        self.encode_generics(def_id);
        self.encode_explicit_predicates(def_id);
        self.encode_inferred_outlives(def_id);
        self.encode_optimized_mir(def_id.expect_local());
        self.encode_promoted_mir(def_id.expect_local());
    }

    fn encode_info_for_mod(&mut self, id: hir::HirId, md: &hir::Mod<'_>, attrs: &[ast::Attribute]) {
        let tcx = self.tcx;
        let local_def_id = tcx.hir().local_def_id(id);
        let def_id = local_def_id.to_def_id();
        debug!("EncodeContext::encode_info_for_mod({:?})", def_id);

        // If we are encoding a proc-macro crates, `encode_info_for_mod` will
        // only ever get called for the crate root. We still want to encode
        // the crate root for consistency with other crates (some of the resolver
        // code uses it). However, we skip encoding anything relating to child
        // items - we encode information about proc-macros later on.
        let reexports = if !self.is_proc_macro {
            match tcx.module_exports(local_def_id) {
                Some(exports) => {
                    let hir = self.tcx.hir();
                    self.lazy(
                        exports
                            .iter()
                            .map(|export| export.map_id(|id| hir.local_def_id_to_hir_id(id))),
                    )
                }
                _ => Lazy::empty(),
            }
        } else {
            Lazy::empty()
        };

        let data = ModData {
            reexports,
            expansion: tcx.hir().definitions().expansion_that_defined(local_def_id),
        };

        record!(self.tables.kind[def_id] <- EntryKind::Mod(self.lazy(data)));
        record!(self.tables.visibility[def_id] <- self.tcx.visibility(def_id));
        record!(self.tables.span[def_id] <- self.tcx.def_span(def_id));
        record!(self.tables.attributes[def_id] <- attrs);
        if self.is_proc_macro {
            record!(self.tables.children[def_id] <- &[]);
        } else {
            record!(self.tables.children[def_id] <- md.item_ids.iter().map(|item_id| {
                tcx.hir().local_def_id(item_id.id).local_def_index
            }));
        }
        self.encode_stability(def_id);
        self.encode_deprecation(def_id);
    }

    fn encode_field(
        &mut self,
        adt_def: &ty::AdtDef,
        variant_index: VariantIdx,
        field_index: usize,
    ) {
        let tcx = self.tcx;
        let variant = &adt_def.variants[variant_index];
        let field = &variant.fields[field_index];

        let def_id = field.did;
        debug!("EncodeContext::encode_field({:?})", def_id);

        let variant_id = tcx.hir().local_def_id_to_hir_id(variant.def_id.expect_local());
        let variant_data = tcx.hir().expect_variant_data(variant_id);

        record!(self.tables.kind[def_id] <- EntryKind::Field);
        record!(self.tables.visibility[def_id] <- self.tcx.visibility(def_id));
        record!(self.tables.span[def_id] <- self.tcx.def_span(def_id));
        record!(self.tables.attributes[def_id] <- variant_data.fields()[field_index].attrs);
        record!(self.tables.expn_that_defined[def_id] <- self.tcx.expansion_that_defined(def_id));
        self.encode_ident_span(def_id, field.ident);
        self.encode_stability(def_id);
        self.encode_deprecation(def_id);
        self.encode_item_type(def_id);
        self.encode_generics(def_id);
        self.encode_explicit_predicates(def_id);
        self.encode_inferred_outlives(def_id);
    }

    fn encode_struct_ctor(&mut self, adt_def: &ty::AdtDef, def_id: DefId) {
        debug!("EncodeContext::encode_struct_ctor({:?})", def_id);
        let tcx = self.tcx;
        let variant = adt_def.non_enum_variant();

        let data = VariantData {
            ctor_kind: variant.ctor_kind,
            discr: variant.discr,
            ctor: Some(def_id.index),
            is_non_exhaustive: variant.is_field_list_non_exhaustive(),
        };

        record!(self.tables.kind[def_id] <- EntryKind::Struct(self.lazy(data), adt_def.repr));
        record!(self.tables.visibility[def_id] <- self.tcx.visibility(def_id));
        record!(self.tables.span[def_id] <- self.tcx.def_span(def_id));
        record!(self.tables.expn_that_defined[def_id] <- self.tcx.expansion_that_defined(def_id));
        self.encode_stability(def_id);
        self.encode_deprecation(def_id);
        self.encode_item_type(def_id);
        if variant.ctor_kind == CtorKind::Fn {
            record!(self.tables.fn_sig[def_id] <- tcx.fn_sig(def_id));
            self.encode_variances_of(def_id);
        }
        self.encode_generics(def_id);
        self.encode_explicit_predicates(def_id);
        self.encode_inferred_outlives(def_id);
        self.encode_optimized_mir(def_id.expect_local());
        self.encode_promoted_mir(def_id.expect_local());
    }

    fn encode_generics(&mut self, def_id: DefId) {
        debug!("EncodeContext::encode_generics({:?})", def_id);
        record!(self.tables.generics[def_id] <- self.tcx.generics_of(def_id));
    }

    fn encode_explicit_predicates(&mut self, def_id: DefId) {
        debug!("EncodeContext::encode_explicit_predicates({:?})", def_id);
        record!(self.tables.explicit_predicates[def_id] <-
            self.tcx.explicit_predicates_of(def_id));
    }

    fn encode_inferred_outlives(&mut self, def_id: DefId) {
        debug!("EncodeContext::encode_inferred_outlives({:?})", def_id);
        let inferred_outlives = self.tcx.inferred_outlives_of(def_id);
        if !inferred_outlives.is_empty() {
            record!(self.tables.inferred_outlives[def_id] <- inferred_outlives);
        }
    }

    fn encode_super_predicates(&mut self, def_id: DefId) {
        debug!("EncodeContext::encode_super_predicates({:?})", def_id);
        record!(self.tables.super_predicates[def_id] <- self.tcx.super_predicates_of(def_id));
    }

    fn encode_explicit_item_bounds(&mut self, def_id: DefId) {
        debug!("EncodeContext::encode_explicit_item_bounds({:?})", def_id);
        let bounds = self.tcx.explicit_item_bounds(def_id);
        if !bounds.is_empty() {
            record!(self.tables.explicit_item_bounds[def_id] <- bounds);
        }
    }

    fn encode_info_for_trait_item(&mut self, def_id: DefId) {
        debug!("EncodeContext::encode_info_for_trait_item({:?})", def_id);
        let tcx = self.tcx;

        let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
        let ast_item = tcx.hir().expect_trait_item(hir_id);
        let trait_item = tcx.associated_item(def_id);

        let container = match trait_item.defaultness {
            hir::Defaultness::Default { has_value: true } => AssocContainer::TraitWithDefault,
            hir::Defaultness::Default { has_value: false } => AssocContainer::TraitRequired,
            hir::Defaultness::Final => span_bug!(ast_item.span, "traits cannot have final items"),
        };

        record!(self.tables.kind[def_id] <- match trait_item.kind {
            ty::AssocKind::Const => {
                let rendered = rustc_hir_pretty::to_string(
                    &(&self.tcx.hir() as &dyn intravisit::Map<'_>),
                    |s| s.print_trait_item(ast_item)
                );
                let rendered_const = self.lazy(RenderedConst(rendered));

                EntryKind::AssocConst(
                    container,
                    Default::default(),
                    rendered_const,
                )
            }
            ty::AssocKind::Fn => {
                let fn_data = if let hir::TraitItemKind::Fn(m_sig, m) = &ast_item.kind {
                    let param_names = match *m {
                        hir::TraitFn::Required(ref names) => {
                            self.encode_fn_param_names(names)
                        }
                        hir::TraitFn::Provided(body) => {
                            self.encode_fn_param_names_for_body(body)
                        }
                    };
                    FnData {
                        asyncness: m_sig.header.asyncness,
                        constness: hir::Constness::NotConst,
                        param_names,
                    }
                } else {
                    bug!()
                };
                EntryKind::AssocFn(self.lazy(AssocFnData {
                    fn_data,
                    container,
                    has_self: trait_item.fn_has_self_parameter,
                }))
            }
            ty::AssocKind::Type => {
                self.encode_explicit_item_bounds(def_id);
                EntryKind::AssocType(container)
            }
        });
        record!(self.tables.visibility[def_id] <- self.tcx.visibility(def_id));
        record!(self.tables.span[def_id] <- ast_item.span);
        record!(self.tables.attributes[def_id] <- ast_item.attrs);
        self.encode_ident_span(def_id, ast_item.ident);
        self.encode_stability(def_id);
        self.encode_const_stability(def_id);
        self.encode_deprecation(def_id);
        match trait_item.kind {
            ty::AssocKind::Const | ty::AssocKind::Fn => {
                self.encode_item_type(def_id);
            }
            ty::AssocKind::Type => {
                if trait_item.defaultness.has_value() {
                    self.encode_item_type(def_id);
                }
            }
        }
        if trait_item.kind == ty::AssocKind::Fn {
            record!(self.tables.fn_sig[def_id] <- tcx.fn_sig(def_id));
            self.encode_variances_of(def_id);
        }
        self.encode_generics(def_id);
        self.encode_explicit_predicates(def_id);
        self.encode_inferred_outlives(def_id);

        // This should be kept in sync with `PrefetchVisitor.visit_trait_item`.
        self.encode_optimized_mir(def_id.expect_local());
        self.encode_promoted_mir(def_id.expect_local());
    }

    fn metadata_output_only(&self) -> bool {
        // MIR optimisation can be skipped when we're just interested in the metadata.
        !self.tcx.sess.opts.output_types.should_codegen()
    }

    fn encode_info_for_impl_item(&mut self, def_id: DefId) {
        debug!("EncodeContext::encode_info_for_impl_item({:?})", def_id);
        let tcx = self.tcx;

        let hir_id = self.tcx.hir().local_def_id_to_hir_id(def_id.expect_local());
        let ast_item = self.tcx.hir().expect_impl_item(hir_id);
        let impl_item = self.tcx.associated_item(def_id);

        let container = match impl_item.defaultness {
            hir::Defaultness::Default { has_value: true } => AssocContainer::ImplDefault,
            hir::Defaultness::Final => AssocContainer::ImplFinal,
            hir::Defaultness::Default { has_value: false } => {
                span_bug!(ast_item.span, "impl items always have values (currently)")
            }
        };

        record!(self.tables.kind[def_id] <- match impl_item.kind {
            ty::AssocKind::Const => {
                if let hir::ImplItemKind::Const(_, body_id) = ast_item.kind {
                    let qualifs = self.tcx.at(ast_item.span).mir_const_qualif(def_id);

                    EntryKind::AssocConst(
                        container,
                        qualifs,
                        self.encode_rendered_const_for_body(body_id))
                } else {
                    bug!()
                }
            }
            ty::AssocKind::Fn => {
                let fn_data = if let hir::ImplItemKind::Fn(ref sig, body) = ast_item.kind {
                    FnData {
                        asyncness: sig.header.asyncness,
                        constness: sig.header.constness,
                        param_names: self.encode_fn_param_names_for_body(body),
                    }
                } else {
                    bug!()
                };
                EntryKind::AssocFn(self.lazy(AssocFnData {
                    fn_data,
                    container,
                    has_self: impl_item.fn_has_self_parameter,
                }))
            }
            ty::AssocKind::Type => EntryKind::AssocType(container)
        });
        record!(self.tables.visibility[def_id] <- self.tcx.visibility(def_id));
        record!(self.tables.span[def_id] <- ast_item.span);
        record!(self.tables.attributes[def_id] <- ast_item.attrs);
        self.encode_ident_span(def_id, impl_item.ident);
        self.encode_stability(def_id);
        self.encode_const_stability(def_id);
        self.encode_deprecation(def_id);
        self.encode_item_type(def_id);
        if impl_item.kind == ty::AssocKind::Fn {
            record!(self.tables.fn_sig[def_id] <- tcx.fn_sig(def_id));
            self.encode_variances_of(def_id);
        }
        self.encode_generics(def_id);
        self.encode_explicit_predicates(def_id);
        self.encode_inferred_outlives(def_id);

        // The following part should be kept in sync with `PrefetchVisitor.visit_impl_item`.

        let mir = match ast_item.kind {
            hir::ImplItemKind::Const(..) => true,
            hir::ImplItemKind::Fn(ref sig, _) => {
                let generics = self.tcx.generics_of(def_id);
                let needs_inline = (generics.requires_monomorphization(self.tcx)
                    || tcx.codegen_fn_attrs(def_id).requests_inline())
                    && !self.metadata_output_only();
                let is_const_fn = sig.header.constness == hir::Constness::Const;
                let always_encode_mir = self.tcx.sess.opts.debugging_opts.always_encode_mir;
                needs_inline || is_const_fn || always_encode_mir
            }
            hir::ImplItemKind::TyAlias(..) => false,
        };
        if mir {
            self.encode_optimized_mir(def_id.expect_local());
            self.encode_promoted_mir(def_id.expect_local());
        }
    }

    fn encode_fn_param_names_for_body(&mut self, body_id: hir::BodyId) -> Lazy<[Ident]> {
        self.lazy(self.tcx.hir().body_param_names(body_id))
    }

    fn encode_fn_param_names(&mut self, param_names: &[Ident]) -> Lazy<[Ident]> {
        self.lazy(param_names.iter())
    }

    fn encode_optimized_mir(&mut self, def_id: LocalDefId) {
        debug!("EntryBuilder::encode_mir({:?})", def_id);
        if self.tcx.mir_keys(LOCAL_CRATE).contains(&def_id) {
            record!(self.tables.mir[def_id.to_def_id()] <- self.tcx.optimized_mir(def_id));

            let unused = self.tcx.unused_generic_params(def_id);
            if !unused.is_empty() {
                record!(self.tables.unused_generic_params[def_id.to_def_id()] <- unused);
            }

            let abstract_const = self.tcx.mir_abstract_const(def_id);
            if let Ok(Some(abstract_const)) = abstract_const {
                record!(self.tables.mir_abstract_consts[def_id.to_def_id()] <- abstract_const);
            }
        }
    }

    fn encode_promoted_mir(&mut self, def_id: LocalDefId) {
        debug!("EncodeContext::encode_promoted_mir({:?})", def_id);
        if self.tcx.mir_keys(LOCAL_CRATE).contains(&def_id) {
            record!(self.tables.promoted_mir[def_id.to_def_id()] <- self.tcx.promoted_mir(def_id));
        }
    }

    // Encodes the inherent implementations of a structure, enumeration, or trait.
    fn encode_inherent_implementations(&mut self, def_id: DefId) {
        debug!("EncodeContext::encode_inherent_implementations({:?})", def_id);
        let implementations = self.tcx.inherent_impls(def_id);
        if !implementations.is_empty() {
            record!(self.tables.inherent_impls[def_id] <- implementations.iter().map(|&def_id| {
                assert!(def_id.is_local());
                def_id.index
            }));
        }
    }

    fn encode_stability(&mut self, def_id: DefId) {
        debug!("EncodeContext::encode_stability({:?})", def_id);

        // The query lookup can take a measurable amount of time in crates with many items. Check if
        // the stability attributes are even enabled before using their queries.
        if self.feat.staged_api || self.tcx.sess.opts.debugging_opts.force_unstable_if_unmarked {
            if let Some(stab) = self.tcx.lookup_stability(def_id) {
                record!(self.tables.stability[def_id] <- stab)
            }
        }
    }

    fn encode_const_stability(&mut self, def_id: DefId) {
        debug!("EncodeContext::encode_const_stability({:?})", def_id);

        // The query lookup can take a measurable amount of time in crates with many items. Check if
        // the stability attributes are even enabled before using their queries.
        if self.feat.staged_api || self.tcx.sess.opts.debugging_opts.force_unstable_if_unmarked {
            if let Some(stab) = self.tcx.lookup_const_stability(def_id) {
                record!(self.tables.const_stability[def_id] <- stab)
            }
        }
    }

    fn encode_deprecation(&mut self, def_id: DefId) {
        debug!("EncodeContext::encode_deprecation({:?})", def_id);
        if let Some(depr) = self.tcx.lookup_deprecation(def_id) {
            record!(self.tables.deprecation[def_id] <- depr);
        }
    }

    fn encode_rendered_const_for_body(&mut self, body_id: hir::BodyId) -> Lazy<RenderedConst> {
        let hir = self.tcx.hir();
        let body = hir.body(body_id);
        let rendered = rustc_hir_pretty::to_string(&(&hir as &dyn intravisit::Map<'_>), |s| {
            s.print_expr(&body.value)
        });
        let rendered_const = &RenderedConst(rendered);
        self.lazy(rendered_const)
    }

    fn encode_info_for_item(&mut self, def_id: DefId, item: &'tcx hir::Item<'tcx>) {
        let tcx = self.tcx;

        debug!("EncodeContext::encode_info_for_item({:?})", def_id);

        self.encode_ident_span(def_id, item.ident);

        record!(self.tables.kind[def_id] <- match item.kind {
            hir::ItemKind::Static(_, hir::Mutability::Mut, _) => EntryKind::MutStatic,
            hir::ItemKind::Static(_, hir::Mutability::Not, _) => EntryKind::ImmStatic,
            hir::ItemKind::Const(_, body_id) => {
                let qualifs = self.tcx.at(item.span).mir_const_qualif(def_id);
                EntryKind::Const(
                    qualifs,
                    self.encode_rendered_const_for_body(body_id)
                )
            }
            hir::ItemKind::Fn(ref sig, .., body) => {
                let data = FnData {
                    asyncness: sig.header.asyncness,
                    constness: sig.header.constness,
                    param_names: self.encode_fn_param_names_for_body(body),
                };

                EntryKind::Fn(self.lazy(data))
            }
            hir::ItemKind::Mod(ref m) => {
                return self.encode_info_for_mod(item.hir_id, m, &item.attrs);
            }
            hir::ItemKind::ForeignMod { .. } => EntryKind::ForeignMod,
            hir::ItemKind::GlobalAsm(..) => EntryKind::GlobalAsm,
            hir::ItemKind::TyAlias(..) => EntryKind::Type,
            hir::ItemKind::OpaqueTy(..) => {
                self.encode_explicit_item_bounds(def_id);
                EntryKind::OpaqueTy
            }
            hir::ItemKind::Enum(..) => EntryKind::Enum(self.tcx.adt_def(def_id).repr),
            hir::ItemKind::Struct(ref struct_def, _) => {
                let adt_def = self.tcx.adt_def(def_id);
                let variant = adt_def.non_enum_variant();

                // Encode def_ids for each field and method
                // for methods, write all the stuff get_trait_method
                // needs to know
                let ctor = struct_def.ctor_hir_id().map(|ctor_hir_id| {
                    self.tcx.hir().local_def_id(ctor_hir_id).local_def_index
                });

                EntryKind::Struct(self.lazy(VariantData {
                    ctor_kind: variant.ctor_kind,
                    discr: variant.discr,
                    ctor,
                    is_non_exhaustive: variant.is_field_list_non_exhaustive(),
                }), adt_def.repr)
            }
            hir::ItemKind::Union(..) => {
                let adt_def = self.tcx.adt_def(def_id);
                let variant = adt_def.non_enum_variant();

                EntryKind::Union(self.lazy(VariantData {
                    ctor_kind: variant.ctor_kind,
                    discr: variant.discr,
                    ctor: None,
                    is_non_exhaustive: variant.is_field_list_non_exhaustive(),
                }), adt_def.repr)
            }
            hir::ItemKind::Impl { defaultness, .. } => {
                let trait_ref = self.tcx.impl_trait_ref(def_id);
                let polarity = self.tcx.impl_polarity(def_id);
                let parent = if let Some(trait_ref) = trait_ref {
                    let trait_def = self.tcx.trait_def(trait_ref.def_id);
                    trait_def.ancestors(self.tcx, def_id).ok()
                        .and_then(|mut an| an.nth(1).and_then(|node| {
                            match node {
                                specialization_graph::Node::Impl(parent) => Some(parent),
                                _ => None,
                            }
                        }))
                } else {
                    None
                };

                // if this is an impl of `CoerceUnsized`, create its
                // "unsized info", else just store None
                let coerce_unsized_info =
                    trait_ref.and_then(|t| {
                        if Some(t.def_id) == self.tcx.lang_items().coerce_unsized_trait() {
                            Some(self.tcx.at(item.span).coerce_unsized_info(def_id))
                        } else {
                            None
                        }
                    });

                let data = ImplData {
                    polarity,
                    defaultness,
                    parent_impl: parent,
                    coerce_unsized_info,
                };

                EntryKind::Impl(self.lazy(data))
            }
            hir::ItemKind::Trait(..) => {
                let trait_def = self.tcx.trait_def(def_id);
                let data = TraitData {
                    unsafety: trait_def.unsafety,
                    paren_sugar: trait_def.paren_sugar,
                    has_auto_impl: self.tcx.trait_is_auto(def_id),
                    is_marker: trait_def.is_marker,
                    specialization_kind: trait_def.specialization_kind,
                };

                EntryKind::Trait(self.lazy(data))
            }
            hir::ItemKind::TraitAlias(..) => EntryKind::TraitAlias,
            hir::ItemKind::ExternCrate(_) |
            hir::ItemKind::Use(..) => bug!("cannot encode info for item {:?}", item),
        });
        record!(self.tables.visibility[def_id] <- self.tcx.visibility(def_id));
        record!(self.tables.span[def_id] <- self.tcx.def_span(def_id));
        record!(self.tables.attributes[def_id] <- item.attrs);
        record!(self.tables.expn_that_defined[def_id] <- self.tcx.expansion_that_defined(def_id));
        // FIXME(eddyb) there should be a nicer way to do this.
        match item.kind {
            hir::ItemKind::ForeignMod { items, .. } => record!(self.tables.children[def_id] <-
                items
                    .iter()
                    .map(|foreign_item| tcx.hir().local_def_id(
                        foreign_item.id.hir_id).local_def_index)
            ),
            hir::ItemKind::Enum(..) => record!(self.tables.children[def_id] <-
                self.tcx.adt_def(def_id).variants.iter().map(|v| {
                    assert!(v.def_id.is_local());
                    v.def_id.index
                })
            ),
            hir::ItemKind::Struct(..) | hir::ItemKind::Union(..) => {
                record!(self.tables.children[def_id] <-
                    self.tcx.adt_def(def_id).non_enum_variant().fields.iter().map(|f| {
                        assert!(f.did.is_local());
                        f.did.index
                    })
                )
            }
            hir::ItemKind::Impl { .. } | hir::ItemKind::Trait(..) => {
                let associated_item_def_ids = self.tcx.associated_item_def_ids(def_id);
                record!(self.tables.children[def_id] <-
                    associated_item_def_ids.iter().map(|&def_id| {
                        assert!(def_id.is_local());
                        def_id.index
                    })
                );
            }
            _ => {}
        }
        self.encode_stability(def_id);
        self.encode_const_stability(def_id);
        self.encode_deprecation(def_id);
        match item.kind {
            hir::ItemKind::Static(..)
            | hir::ItemKind::Const(..)
            | hir::ItemKind::Fn(..)
            | hir::ItemKind::TyAlias(..)
            | hir::ItemKind::OpaqueTy(..)
            | hir::ItemKind::Enum(..)
            | hir::ItemKind::Struct(..)
            | hir::ItemKind::Union(..)
            | hir::ItemKind::Impl { .. } => self.encode_item_type(def_id),
            _ => {}
        }
        if let hir::ItemKind::Fn(..) = item.kind {
            record!(self.tables.fn_sig[def_id] <- tcx.fn_sig(def_id));
        }
        if let hir::ItemKind::Impl { .. } = item.kind {
            if let Some(trait_ref) = self.tcx.impl_trait_ref(def_id) {
                record!(self.tables.impl_trait_ref[def_id] <- trait_ref);
            }
        }
        self.encode_inherent_implementations(def_id);
        match item.kind {
            hir::ItemKind::Enum(..)
            | hir::ItemKind::Struct(..)
            | hir::ItemKind::Union(..)
            | hir::ItemKind::Fn(..) => self.encode_variances_of(def_id),
            _ => {}
        }
        match item.kind {
            hir::ItemKind::Static(..)
            | hir::ItemKind::Const(..)
            | hir::ItemKind::Fn(..)
            | hir::ItemKind::TyAlias(..)
            | hir::ItemKind::Enum(..)
            | hir::ItemKind::Struct(..)
            | hir::ItemKind::Union(..)
            | hir::ItemKind::Impl { .. }
            | hir::ItemKind::OpaqueTy(..)
            | hir::ItemKind::Trait(..)
            | hir::ItemKind::TraitAlias(..) => {
                self.encode_generics(def_id);
                self.encode_explicit_predicates(def_id);
                self.encode_inferred_outlives(def_id);
            }
            _ => {}
        }
        match item.kind {
            hir::ItemKind::Trait(..) | hir::ItemKind::TraitAlias(..) => {
                self.encode_super_predicates(def_id);
            }
            _ => {}
        }

        // The following part should be kept in sync with `PrefetchVisitor.visit_item`.

        let mir = match item.kind {
            hir::ItemKind::Static(..) | hir::ItemKind::Const(..) => true,
            hir::ItemKind::Fn(ref sig, ..) => {
                let generics = tcx.generics_of(def_id);
                let needs_inline = (generics.requires_monomorphization(tcx)
                    || tcx.codegen_fn_attrs(def_id).requests_inline())
                    && !self.metadata_output_only();
                let always_encode_mir = self.tcx.sess.opts.debugging_opts.always_encode_mir;
                needs_inline || sig.header.constness == hir::Constness::Const || always_encode_mir
            }
            _ => false,
        };
        if mir {
            self.encode_optimized_mir(def_id.expect_local());
            self.encode_promoted_mir(def_id.expect_local());
        }
    }

    /// Serialize the text of exported macros
    fn encode_info_for_macro_def(&mut self, macro_def: &hir::MacroDef<'_>) {
        let def_id = self.tcx.hir().local_def_id(macro_def.hir_id).to_def_id();
        record!(self.tables.kind[def_id] <- EntryKind::MacroDef(self.lazy(macro_def.ast.clone())));
        record!(self.tables.visibility[def_id] <- self.tcx.visibility(def_id));
        record!(self.tables.span[def_id] <- macro_def.span);
        record!(self.tables.attributes[def_id] <- macro_def.attrs);
        self.encode_ident_span(def_id, macro_def.ident);
        self.encode_stability(def_id);
        self.encode_deprecation(def_id);
    }

    fn encode_info_for_generic_param(&mut self, def_id: DefId, kind: EntryKind, encode_type: bool) {
        record!(self.tables.kind[def_id] <- kind);
        record!(self.tables.span[def_id] <- self.tcx.def_span(def_id));
        if encode_type {
            self.encode_item_type(def_id);
        }
    }

    fn encode_info_for_closure(&mut self, def_id: LocalDefId) {
        debug!("EncodeContext::encode_info_for_closure({:?})", def_id);

        // NOTE(eddyb) `tcx.type_of(def_id)` isn't used because it's fully generic,
        // including on the signature, which is inferred in `typeck.
        let hir_id = self.tcx.hir().local_def_id_to_hir_id(def_id);
        let ty = self.tcx.typeck(def_id).node_type(hir_id);

        record!(self.tables.kind[def_id.to_def_id()] <- match ty.kind() {
            ty::Generator(..) => {
                let data = self.tcx.generator_kind(def_id).unwrap();
                EntryKind::Generator(data)
            }

            ty::Closure(..) => EntryKind::Closure,

            _ => bug!("closure that is neither generator nor closure"),
        });
        record!(self.tables.span[def_id.to_def_id()] <- self.tcx.def_span(def_id));
        record!(self.tables.attributes[def_id.to_def_id()] <- &self.tcx.get_attrs(def_id.to_def_id())[..]);
        self.encode_item_type(def_id.to_def_id());
        if let ty::Closure(def_id, substs) = *ty.kind() {
            record!(self.tables.fn_sig[def_id] <- substs.as_closure().sig());
        }
        self.encode_generics(def_id.to_def_id());
        self.encode_optimized_mir(def_id);
        self.encode_promoted_mir(def_id);
    }

    fn encode_info_for_anon_const(&mut self, def_id: LocalDefId) {
        debug!("EncodeContext::encode_info_for_anon_const({:?})", def_id);
        let id = self.tcx.hir().local_def_id_to_hir_id(def_id);
        let body_id = self.tcx.hir().body_owned_by(id);
        let const_data = self.encode_rendered_const_for_body(body_id);
        let qualifs = self.tcx.mir_const_qualif(def_id);

        record!(self.tables.kind[def_id.to_def_id()] <- EntryKind::AnonConst(qualifs, const_data));
        record!(self.tables.span[def_id.to_def_id()] <- self.tcx.def_span(def_id));
        self.encode_item_type(def_id.to_def_id());
        self.encode_generics(def_id.to_def_id());
        self.encode_explicit_predicates(def_id.to_def_id());
        self.encode_inferred_outlives(def_id.to_def_id());
        self.encode_optimized_mir(def_id);
        self.encode_promoted_mir(def_id);
    }

    fn encode_native_libraries(&mut self) -> Lazy<[NativeLib]> {
        empty_proc_macro!(self);
        let used_libraries = self.tcx.native_libraries(LOCAL_CRATE);
        self.lazy(used_libraries.iter().cloned())
    }

    fn encode_foreign_modules(&mut self) -> Lazy<[ForeignModule]> {
        empty_proc_macro!(self);
        let foreign_modules = self.tcx.foreign_modules(LOCAL_CRATE);
        self.lazy(foreign_modules.iter().map(|(_, m)| m).cloned())
    }

    fn encode_hygiene(&mut self) -> (SyntaxContextTable, ExpnDataTable) {
        let mut syntax_contexts: TableBuilder<_, _> = Default::default();
        let mut expn_data_table: TableBuilder<_, _> = Default::default();

        let _: Result<(), !> = self.hygiene_ctxt.encode(
            &mut (&mut *self, &mut syntax_contexts, &mut expn_data_table),
            |(this, syntax_contexts, _), index, ctxt_data| {
                syntax_contexts.set(index, this.lazy(ctxt_data));
                Ok(())
            },
            |(this, _, expn_data_table), index, expn_data| {
                expn_data_table.set(index, this.lazy(expn_data));
                Ok(())
            },
        );

        (syntax_contexts.encode(&mut self.opaque), expn_data_table.encode(&mut self.opaque))
    }

    fn encode_proc_macros(&mut self) -> Option<ProcMacroData> {
        let is_proc_macro = self.tcx.sess.crate_types().contains(&CrateType::ProcMacro);
        if is_proc_macro {
            let tcx = self.tcx;
            let hir = tcx.hir();

            let proc_macro_decls_static = tcx.proc_macro_decls_static(LOCAL_CRATE).unwrap().index;
            let stability = tcx.lookup_stability(DefId::local(CRATE_DEF_INDEX)).copied();
            let macros = self.lazy(hir.krate().proc_macros.iter().map(|p| p.owner.local_def_index));

            // Normally, this information is encoded when we walk the items
            // defined in this crate. However, we skip doing that for proc-macro crates,
            // so we manually encode just the information that we need
            for proc_macro in &hir.krate().proc_macros {
                let id = proc_macro.owner.local_def_index;
                let mut name = hir.name(*proc_macro);
                let span = hir.span(*proc_macro);
                // Proc-macros may have attributes like `#[allow_internal_unstable]`,
                // so downstream crates need access to them.
                let attrs = hir.attrs(*proc_macro);
                let macro_kind = if tcx.sess.contains_name(attrs, sym::proc_macro) {
                    MacroKind::Bang
                } else if tcx.sess.contains_name(attrs, sym::proc_macro_attribute) {
                    MacroKind::Attr
                } else if let Some(attr) = tcx.sess.find_by_name(attrs, sym::proc_macro_derive) {
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

                let mut def_key = self.tcx.hir().def_key(proc_macro.owner);
                def_key.disambiguated_data.data = DefPathData::MacroNs(name);

                let def_id = DefId::local(id);
                record!(self.tables.kind[def_id] <- EntryKind::ProcMacro(macro_kind));
                record!(self.tables.attributes[def_id] <- attrs);
                record!(self.tables.def_keys[def_id] <- def_key);
                record!(self.tables.ident_span[def_id] <- span);
                record!(self.tables.span[def_id] <- span);
                record!(self.tables.visibility[def_id] <- ty::Visibility::Public);
                if let Some(stability) = stability {
                    record!(self.tables.stability[def_id] <- stability);
                }
            }

            Some(ProcMacroData { proc_macro_decls_static, stability, macros })
        } else {
            None
        }
    }

    fn encode_crate_deps(&mut self) -> Lazy<[CrateDep]> {
        empty_proc_macro!(self);
        let crates = self.tcx.crates();

        let mut deps = crates
            .iter()
            .map(|&cnum| {
                let dep = CrateDep {
                    name: self.tcx.original_crate_name(cnum),
                    hash: self.tcx.crate_hash(cnum),
                    host_hash: self.tcx.crate_host_hash(cnum),
                    kind: self.tcx.dep_kind(cnum),
                    extra_filename: self.tcx.extra_filename(cnum),
                };
                (cnum, dep)
            })
            .collect::<Vec<_>>();

        deps.sort_by_key(|&(cnum, _)| cnum);

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
        self.lazy(deps.iter().map(|&(_, ref dep)| dep))
    }

    fn encode_lib_features(&mut self) -> Lazy<[(Symbol, Option<Symbol>)]> {
        empty_proc_macro!(self);
        let tcx = self.tcx;
        let lib_features = tcx.lib_features();
        self.lazy(lib_features.to_vec())
    }

    fn encode_diagnostic_items(&mut self) -> Lazy<[(Symbol, DefIndex)]> {
        empty_proc_macro!(self);
        let tcx = self.tcx;
        let diagnostic_items = tcx.diagnostic_items(LOCAL_CRATE);
        self.lazy(diagnostic_items.iter().map(|(&name, def_id)| (name, def_id.index)))
    }

    fn encode_lang_items(&mut self) -> Lazy<[(DefIndex, usize)]> {
        empty_proc_macro!(self);
        let tcx = self.tcx;
        let lang_items = tcx.lang_items();
        let lang_items = lang_items.items().iter();
        self.lazy(lang_items.enumerate().filter_map(|(i, &opt_def_id)| {
            if let Some(def_id) = opt_def_id {
                if def_id.is_local() {
                    return Some((def_id.index, i));
                }
            }
            None
        }))
    }

    fn encode_lang_items_missing(&mut self) -> Lazy<[lang_items::LangItem]> {
        empty_proc_macro!(self);
        let tcx = self.tcx;
        self.lazy(&tcx.lang_items().missing)
    }

    /// Encodes an index, mapping each trait to its (local) implementations.
    fn encode_impls(&mut self) -> Lazy<[TraitImpls]> {
        empty_proc_macro!(self);
        debug!("EncodeContext::encode_impls()");
        let tcx = self.tcx;
        let mut visitor = ImplVisitor { tcx, impls: FxHashMap::default() };
        tcx.hir().krate().visit_all_item_likes(&mut visitor);

        let mut all_impls: Vec<_> = visitor.impls.into_iter().collect();

        // Bring everything into deterministic order for hashing
        all_impls.sort_by_cached_key(|&(trait_def_id, _)| tcx.def_path_hash(trait_def_id));

        let all_impls: Vec<_> = all_impls
            .into_iter()
            .map(|(trait_def_id, mut impls)| {
                // Bring everything into deterministic order for hashing
                impls.sort_by_cached_key(|&(index, _)| {
                    tcx.hir().definitions().def_path_hash(LocalDefId { local_def_index: index })
                });

                TraitImpls {
                    trait_id: (trait_def_id.krate.as_u32(), trait_def_id.index),
                    impls: self.lazy(&impls),
                }
            })
            .collect();

        self.lazy(&all_impls)
    }

    // Encodes all symbols exported from this crate into the metadata.
    //
    // This pass is seeded off the reachability list calculated in the
    // middle::reachable module but filters out items that either don't have a
    // symbol associated with them (they weren't translated) or if they're an FFI
    // definition (as that's not defined in this crate).
    fn encode_exported_symbols(
        &mut self,
        exported_symbols: &[(ExportedSymbol<'tcx>, SymbolExportLevel)],
    ) -> Lazy<[(ExportedSymbol<'tcx>, SymbolExportLevel)]> {
        empty_proc_macro!(self);
        // The metadata symbol name is special. It should not show up in
        // downstream crates.
        let metadata_symbol_name = SymbolName::new(self.tcx, &metadata_symbol_name(self.tcx));

        self.lazy(
            exported_symbols
                .iter()
                .filter(|&&(ref exported_symbol, _)| match *exported_symbol {
                    ExportedSymbol::NoDefId(symbol_name) => symbol_name != metadata_symbol_name,
                    _ => true,
                })
                .cloned(),
        )
    }

    fn encode_dylib_dependency_formats(&mut self) -> Lazy<[Option<LinkagePreference>]> {
        empty_proc_macro!(self);
        let formats = self.tcx.dependency_formats(LOCAL_CRATE);
        for (ty, arr) in formats.iter() {
            if *ty != CrateType::Dylib {
                continue;
            }
            return self.lazy(arr.iter().map(|slot| match *slot {
                Linkage::NotLinked | Linkage::IncludedFromDylib => None,

                Linkage::Dynamic => Some(LinkagePreference::RequireDynamic),
                Linkage::Static => Some(LinkagePreference::RequireStatic),
            }));
        }
        Lazy::empty()
    }

    fn encode_info_for_foreign_item(&mut self, def_id: DefId, nitem: &hir::ForeignItem<'_>) {
        let tcx = self.tcx;

        debug!("EncodeContext::encode_info_for_foreign_item({:?})", def_id);

        record!(self.tables.kind[def_id] <- match nitem.kind {
            hir::ForeignItemKind::Fn(_, ref names, _) => {
                let data = FnData {
                    asyncness: hir::IsAsync::NotAsync,
                    constness: if self.tcx.is_const_fn_raw(def_id) {
                        hir::Constness::Const
                    } else {
                        hir::Constness::NotConst
                    },
                    param_names: self.encode_fn_param_names(names),
                };
                EntryKind::ForeignFn(self.lazy(data))
            }
            hir::ForeignItemKind::Static(_, hir::Mutability::Mut) => EntryKind::ForeignMutStatic,
            hir::ForeignItemKind::Static(_, hir::Mutability::Not) => EntryKind::ForeignImmStatic,
            hir::ForeignItemKind::Type => EntryKind::ForeignType,
        });
        record!(self.tables.visibility[def_id] <- self.tcx.visibility(def_id));
        record!(self.tables.span[def_id] <- nitem.span);
        record!(self.tables.attributes[def_id] <- nitem.attrs);
        self.encode_ident_span(def_id, nitem.ident);
        self.encode_stability(def_id);
        self.encode_const_stability(def_id);
        self.encode_deprecation(def_id);
        self.encode_item_type(def_id);
        self.encode_inherent_implementations(def_id);
        if let hir::ForeignItemKind::Fn(..) = nitem.kind {
            record!(self.tables.fn_sig[def_id] <- tcx.fn_sig(def_id));
            self.encode_variances_of(def_id);
        }
        self.encode_generics(def_id);
        self.encode_explicit_predicates(def_id);
        self.encode_inferred_outlives(def_id);
    }
}

// FIXME(eddyb) make metadata encoding walk over all definitions, instead of HIR.
impl Visitor<'tcx> for EncodeContext<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::OnlyBodies(self.tcx.hir())
    }
    fn visit_expr(&mut self, ex: &'tcx hir::Expr<'tcx>) {
        intravisit::walk_expr(self, ex);
        self.encode_info_for_expr(ex);
    }
    fn visit_anon_const(&mut self, c: &'tcx AnonConst) {
        intravisit::walk_anon_const(self, c);
        let def_id = self.tcx.hir().local_def_id(c.hir_id);
        self.encode_info_for_anon_const(def_id);
    }
    fn visit_item(&mut self, item: &'tcx hir::Item<'tcx>) {
        intravisit::walk_item(self, item);
        let def_id = self.tcx.hir().local_def_id(item.hir_id);
        match item.kind {
            hir::ItemKind::ExternCrate(_) | hir::ItemKind::Use(..) => {} // ignore these
            _ => self.encode_info_for_item(def_id.to_def_id(), item),
        }
        self.encode_addl_info_for_item(item);
    }
    fn visit_foreign_item(&mut self, ni: &'tcx hir::ForeignItem<'tcx>) {
        intravisit::walk_foreign_item(self, ni);
        let def_id = self.tcx.hir().local_def_id(ni.hir_id);
        self.encode_info_for_foreign_item(def_id.to_def_id(), ni);
    }
    fn visit_generics(&mut self, generics: &'tcx hir::Generics<'tcx>) {
        intravisit::walk_generics(self, generics);
        self.encode_info_for_generics(generics);
    }
    fn visit_macro_def(&mut self, macro_def: &'tcx hir::MacroDef<'tcx>) {
        self.encode_info_for_macro_def(macro_def);
    }
}

impl EncodeContext<'a, 'tcx> {
    fn encode_fields(&mut self, adt_def: &ty::AdtDef) {
        for (variant_index, variant) in adt_def.variants.iter_enumerated() {
            for (field_index, _field) in variant.fields.iter().enumerate() {
                self.encode_field(adt_def, variant_index, field_index);
            }
        }
    }

    fn encode_info_for_generics(&mut self, generics: &hir::Generics<'tcx>) {
        for param in generics.params {
            let def_id = self.tcx.hir().local_def_id(param.hir_id);
            match param.kind {
                GenericParamKind::Lifetime { .. } => continue,
                GenericParamKind::Type { ref default, .. } => {
                    self.encode_info_for_generic_param(
                        def_id.to_def_id(),
                        EntryKind::TypeParam,
                        default.is_some(),
                    );
                    if default.is_some() {
                        self.encode_stability(def_id.to_def_id());
                    }
                }
                GenericParamKind::Const { .. } => {
                    self.encode_info_for_generic_param(
                        def_id.to_def_id(),
                        EntryKind::ConstParam,
                        true,
                    );
                    // FIXME(const_generics_defaults)
                }
            }
        }
    }

    fn encode_info_for_expr(&mut self, expr: &hir::Expr<'_>) {
        if let hir::ExprKind::Closure(..) = expr.kind {
            let def_id = self.tcx.hir().local_def_id(expr.hir_id);
            self.encode_info_for_closure(def_id);
        }
    }

    fn encode_ident_span(&mut self, def_id: DefId, ident: Ident) {
        record!(self.tables.ident_span[def_id] <- ident.span);
    }

    /// In some cases, along with the item itself, we also
    /// encode some sub-items. Usually we want some info from the item
    /// so it's easier to do that here then to wait until we would encounter
    /// normally in the visitor walk.
    fn encode_addl_info_for_item(&mut self, item: &hir::Item<'_>) {
        let def_id = self.tcx.hir().local_def_id(item.hir_id);
        match item.kind {
            hir::ItemKind::Static(..)
            | hir::ItemKind::Const(..)
            | hir::ItemKind::Fn(..)
            | hir::ItemKind::Mod(..)
            | hir::ItemKind::ForeignMod { .. }
            | hir::ItemKind::GlobalAsm(..)
            | hir::ItemKind::ExternCrate(..)
            | hir::ItemKind::Use(..)
            | hir::ItemKind::TyAlias(..)
            | hir::ItemKind::OpaqueTy(..)
            | hir::ItemKind::TraitAlias(..) => {
                // no sub-item recording needed in these cases
            }
            hir::ItemKind::Enum(..) => {
                let def = self.tcx.adt_def(def_id.to_def_id());
                self.encode_fields(def);

                for (i, variant) in def.variants.iter_enumerated() {
                    self.encode_enum_variant_info(def, i);

                    if let Some(_ctor_def_id) = variant.ctor_def_id {
                        self.encode_enum_variant_ctor(def, i);
                    }
                }
            }
            hir::ItemKind::Struct(ref struct_def, _) => {
                let def = self.tcx.adt_def(def_id.to_def_id());
                self.encode_fields(def);

                // If the struct has a constructor, encode it.
                if let Some(ctor_hir_id) = struct_def.ctor_hir_id() {
                    let ctor_def_id = self.tcx.hir().local_def_id(ctor_hir_id);
                    self.encode_struct_ctor(def, ctor_def_id.to_def_id());
                }
            }
            hir::ItemKind::Union(..) => {
                let def = self.tcx.adt_def(def_id.to_def_id());
                self.encode_fields(def);
            }
            hir::ItemKind::Impl { .. } => {
                for &trait_item_def_id in
                    self.tcx.associated_item_def_ids(def_id.to_def_id()).iter()
                {
                    self.encode_info_for_impl_item(trait_item_def_id);
                }
            }
            hir::ItemKind::Trait(..) => {
                for &item_def_id in self.tcx.associated_item_def_ids(def_id.to_def_id()).iter() {
                    self.encode_info_for_trait_item(item_def_id);
                }
            }
        }
    }
}

struct ImplVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    impls: FxHashMap<DefId, Vec<(DefIndex, Option<ty::fast_reject::SimplifiedType>)>>,
}

impl<'tcx, 'v> ItemLikeVisitor<'v> for ImplVisitor<'tcx> {
    fn visit_item(&mut self, item: &hir::Item<'_>) {
        if let hir::ItemKind::Impl { .. } = item.kind {
            let impl_id = self.tcx.hir().local_def_id(item.hir_id);
            if let Some(trait_ref) = self.tcx.impl_trait_ref(impl_id.to_def_id()) {
                let simplified_self_ty =
                    ty::fast_reject::simplify_type(self.tcx, trait_ref.self_ty(), false);

                self.impls
                    .entry(trait_ref.def_id)
                    .or_default()
                    .push((impl_id.local_def_index, simplified_self_ty));
            }
        }
    }

    fn visit_trait_item(&mut self, _trait_item: &'v hir::TraitItem<'v>) {}

    fn visit_impl_item(&mut self, _impl_item: &'v hir::ImplItem<'v>) {
        // handled in `visit_item` above
    }

    fn visit_foreign_item(&mut self, _foreign_item: &'v hir::ForeignItem<'v>) {}
}

/// Used to prefetch queries which will be needed later by metadata encoding.
/// Only a subset of the queries are actually prefetched to keep this code smaller.
struct PrefetchVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    mir_keys: &'tcx FxHashSet<LocalDefId>,
}

impl<'tcx> PrefetchVisitor<'tcx> {
    fn prefetch_mir(&self, def_id: LocalDefId) {
        if self.mir_keys.contains(&def_id) {
            self.tcx.ensure().optimized_mir(def_id);
            self.tcx.ensure().promoted_mir(def_id);
        }
    }
}

impl<'tcx, 'v> ParItemLikeVisitor<'v> for PrefetchVisitor<'tcx> {
    fn visit_item(&self, item: &hir::Item<'_>) {
        // This should be kept in sync with `encode_info_for_item`.
        let tcx = self.tcx;
        match item.kind {
            hir::ItemKind::Static(..) | hir::ItemKind::Const(..) => {
                self.prefetch_mir(tcx.hir().local_def_id(item.hir_id))
            }
            hir::ItemKind::Fn(ref sig, ..) => {
                let def_id = tcx.hir().local_def_id(item.hir_id);
                let generics = tcx.generics_of(def_id.to_def_id());
                let needs_inline = generics.requires_monomorphization(tcx)
                    || tcx.codegen_fn_attrs(def_id.to_def_id()).requests_inline();
                if needs_inline || sig.header.constness == hir::Constness::Const {
                    self.prefetch_mir(def_id)
                }
            }
            _ => (),
        }
    }

    fn visit_trait_item(&self, trait_item: &'v hir::TraitItem<'v>) {
        // This should be kept in sync with `encode_info_for_trait_item`.
        self.prefetch_mir(self.tcx.hir().local_def_id(trait_item.hir_id));
    }

    fn visit_impl_item(&self, impl_item: &'v hir::ImplItem<'v>) {
        // This should be kept in sync with `encode_info_for_impl_item`.
        let tcx = self.tcx;
        match impl_item.kind {
            hir::ImplItemKind::Const(..) => {
                self.prefetch_mir(tcx.hir().local_def_id(impl_item.hir_id))
            }
            hir::ImplItemKind::Fn(ref sig, _) => {
                let def_id = tcx.hir().local_def_id(impl_item.hir_id);
                let generics = tcx.generics_of(def_id.to_def_id());
                let needs_inline = generics.requires_monomorphization(tcx)
                    || tcx.codegen_fn_attrs(def_id.to_def_id()).requests_inline();
                let is_const_fn = sig.header.constness == hir::Constness::Const;
                if needs_inline || is_const_fn {
                    self.prefetch_mir(def_id)
                }
            }
            hir::ImplItemKind::TyAlias(..) => (),
        }
    }

    fn visit_foreign_item(&self, _foreign_item: &'v hir::ForeignItem<'v>) {
        // This should be kept in sync with `encode_info_for_foreign_item`.
        // Foreign items contain no MIR.
    }
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

pub(super) fn encode_metadata(tcx: TyCtxt<'_>) -> EncodedMetadata {
    // Since encoding metadata is not in a query, and nothing is cached,
    // there's no need to do dep-graph tracking for any of it.
    tcx.dep_graph.assert_ignored();

    join(
        || encode_metadata_impl(tcx),
        || {
            if tcx.sess.threads() == 1 {
                return;
            }
            // Prefetch some queries used by metadata encoding.
            // This is not necessary for correctness, but is only done for performance reasons.
            // It can be removed if it turns out to cause trouble or be detrimental to performance.
            join(
                || {
                    if !tcx.sess.opts.output_types.should_codegen() {
                        // We won't emit MIR, so don't prefetch it.
                        return;
                    }
                    tcx.hir().krate().par_visit_all_item_likes(&PrefetchVisitor {
                        tcx,
                        mir_keys: tcx.mir_keys(LOCAL_CRATE),
                    });
                },
                || tcx.exported_symbols(LOCAL_CRATE),
            );
        },
    )
    .0
}

fn encode_metadata_impl(tcx: TyCtxt<'_>) -> EncodedMetadata {
    let mut encoder = opaque::Encoder::new(vec![]);
    encoder.emit_raw_bytes(METADATA_HEADER);

    // Will be filled with the root position after encoding everything.
    encoder.emit_raw_bytes(&[0, 0, 0, 0]);

    let source_map_files = tcx.sess.source_map().files();
    let source_file_cache = (source_map_files[0].clone(), 0);
    let required_source_files = Some(GrowableBitSet::with_capacity(source_map_files.len()));
    drop(source_map_files);

    let hygiene_ctxt = HygieneEncodeContext::default();

    let mut ecx = EncodeContext {
        opaque: encoder,
        tcx,
        feat: tcx.features(),
        tables: Default::default(),
        lazy_state: LazyState::NoNode,
        type_shorthands: Default::default(),
        predicate_shorthands: Default::default(),
        source_file_cache,
        interpret_allocs: Default::default(),
        required_source_files,
        is_proc_macro: tcx.sess.crate_types().contains(&CrateType::ProcMacro),
        hygiene_ctxt: &hygiene_ctxt,
    };

    // Encode the rustc version string in a predictable location.
    rustc_version().encode(&mut ecx).unwrap();

    // Encode all the entries and extra information in the crate,
    // culminating in the `CrateRoot` which points to all of it.
    let root = ecx.encode_crate_root();

    let mut result = ecx.opaque.into_inner();

    // Encode the root position.
    let header = METADATA_HEADER.len();
    let pos = root.position.get();
    result[header + 0] = (pos >> 24) as u8;
    result[header + 1] = (pos >> 16) as u8;
    result[header + 2] = (pos >> 8) as u8;
    result[header + 3] = (pos >> 0) as u8;

    EncodedMetadata { raw_data: result }
}
