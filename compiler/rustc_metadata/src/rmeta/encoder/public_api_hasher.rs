use core::cell::RefCell;
use core::iter::Iterator;
use std::fmt;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::graph::scc::{Annotation, Annotations, Sccs};
use rustc_data_structures::graph::{DirectedGraph, Successors};
use rustc_data_structures::indexmap::map::Entry;
use rustc_data_structures::stable_hasher::{StableHash, StableHashCtxt, StableHasher};
use rustc_data_structures::svh::Svh;
use rustc_data_structures::unord::UnordMap;
use rustc_hir::LangItem;
use rustc_hir::attrs::StrippedCfgItem;
use rustc_hir::def_id::DefIndex;
use rustc_index::{Idx, IndexVec};
use rustc_macros::{LazyDecodable, MetadataEncodable, StableHash};
use rustc_middle::ich::StableHashingContext;
use rustc_middle::middle::debugger_visualizer::DebuggerVisualizerFile;
use rustc_middle::middle::exported_symbols::{ExportedSymbol, SymbolExportInfo};
use rustc_middle::middle::lib_features::FeatureStability;
use rustc_middle::ty::data_structures::IndexMap;
use rustc_middle::ty::{PredicateKind, Ty, TyCtxt};
use rustc_session::config::mitigation_coverage::DeniedPartialMitigation;
use rustc_session::config::{SymbolManglingVersion, TargetModifier};
use rustc_session::cstore::{ForeignModule, LinkagePreference, NativeLib};
use rustc_span::def_id::{CRATE_DEF_ID, DefId, LOCAL_CRATE, LocalDefId, StableCrateId};
use rustc_span::edition::Edition;
use rustc_span::hygiene::ExpnIndex;
use rustc_span::{ExpnId, LocalExpnId, Span, Symbol, SyntaxContext};
use rustc_target::spec::PanicStrategy;
use tracing::debug;

use super::{CrateDep, CrateHeader, CrateRoot, TargetTuple};
use crate::rmeta::{
    CrateHashes, DefPathHashMapRef, EiiMapEncodedKeyValue, EncodeContext, ExpnDataTable,
    ExpnHashTable, IncoherentImpls, LazyArray, LazyDecoder, LazyTable, LazyTables, LazyValue,
    ProcMacroData, SyntaxContextTable, TableBuilder, TraitImpls,
};

#[derive(Default)]
pub(crate) struct PublicApiHasher(StableHasher);

impl PublicApiHasher {
    pub(crate) fn finish(self, hcx: &mut PublicApiHashingContext<'_>) -> Option<Fingerprint> {
        hcx.hash_public_api.then(|| self.0.finish())
    }

    pub(crate) fn digest<'a, T: StableHash>(
        &mut self,
        value: T,
        hcx: &mut PublicApiHashingContext<'a>,
    ) {
        if hcx.hash_public_api {
            value.stable_hash(&mut hcx.hcx, &mut self.0);
        }
    }
}

pub(crate) trait TablePublicApiHasher<I: Idx>: Default {
    type IterHasher;

    fn digest<V>(
        &mut self,
        index: impl TableIndex<Encoded = I>,
        value: V,
        hcx: &mut PublicApiHashingContext<'_>,
    ) where
        V: StableHash;
    fn finish(&self, hcx: &mut PublicApiHashingContext<'_>) -> Option<Fingerprint>;

    fn iter_hasher(&self) -> Self::IterHasher;
}

pub(crate) struct RDRHashAll<I: Idx> {
    hash: Fingerprint,
    _marker: std::marker::PhantomData<I>,
}

impl<I: Idx> Default for RDRHashAll<I> {
    fn default() -> Self {
        Self { hash: Fingerprint::ZERO, _marker: Default::default() }
    }
}

pub(crate) struct PublicApiHashingContext<'a> {
    pub(crate) hcx: StableHashingContext<'a>,
    hash_public_api: bool,
    def_id_hashes: IndexGraphHashes,
}

impl<'a> PublicApiHashingContext<'a> {
    pub(crate) fn enabled(&self) -> bool {
        self.hash_public_api
    }
    pub(crate) fn new(hash_public_api: bool, hcx: StableHashingContext<'a>) -> Self {
        Self { hash_public_api, hcx, def_id_hashes: Default::default() }
    }
}

impl<I: Idx> TablePublicApiHasher<I> for RDRHashAll<I> {
    type IterHasher = OrderedIterHasher;
    fn digest<V>(
        &mut self,
        index: impl TableIndex<Encoded = I>,
        value: V,
        hcx: &mut PublicApiHashingContext<'_>,
    ) where
        V: StableHash,
    {
        if !hcx.hash_public_api {
            return;
        }
        let mut hasher = StableHasher::default();
        value.stable_hash(&mut hcx.hcx, &mut hasher);
        let hash: Fingerprint = hasher.finish();
        let idx_hash = hcx.def_id_hashes.get_mut(index);
        *idx_hash = idx_hash.combine_commutative(hash);
        // remove this later, not needed anymore
        self.hash.combine_commutative(hash);
    }

    fn finish(&self, hcx: &mut PublicApiHashingContext<'_>) -> Option<Fingerprint> {
        hcx.hash_public_api.then_some(self.hash)
    }

    fn iter_hasher(&self) -> Self::IterHasher {
        OrderedIterHasher::default()
    }
}

#[derive(Default)]
pub(crate) struct OrderedIterHasher(StableHasher);

impl OrderedIterHasher {
    pub(crate) fn inspect_digest<'a: 'b, 'b, I>(
        &'b mut self,
        iter: I,
        hcx: &'b mut PublicApiHashingContext<'a>,
    ) -> impl Iterator<Item = I::Item> + 'b
    where
        I: IntoIterator + 'b,
        I::Item: StableHash,
    {
        iter.into_iter().inspect(move |item: &I::Item| {
            if hcx.hash_public_api {
                item.stable_hash(&mut hcx.hcx, &mut self.0)
            }
        })
    }

    pub(crate) fn finish(self) -> Fingerprint {
        self.0.finish()
    }
}

pub(crate) struct RDRHashNone<I>(std::marker::PhantomData<I>);

impl<I> Default for RDRHashNone<I> {
    fn default() -> Self {
        RDRHashNone(Default::default())
    }
}

impl<I: Idx> TablePublicApiHasher<I> for RDRHashNone<I> {
    type IterHasher = RDRHashNone<()>;
    fn digest<V>(
        &mut self,
        _index: impl TableIndex<Encoded = I>,
        _value: V,
        _hcx: &mut PublicApiHashingContext<'_>,
    ) where
        V: StableHash,
    {
    }

    fn iter_hasher(&self) -> Self::IterHasher {
        Default::default()
    }

    fn finish(&self, hcx: &mut PublicApiHashingContext<'_>) -> Option<Fingerprint> {
        hcx.hash_public_api.then_some(Fingerprint::ZERO)
    }
}

#[derive(Default)]
pub(crate) struct Hashed<T> {
    pub(crate) hash: Option<Fingerprint>,
    pub(crate) value: T,
}

impl<T> fmt::Debug for Hashed<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.hash.as_ref().unwrap().fmt(f)
    }
}

impl<T> StableHash for Hashed<T> {
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        self.hash.expect("Hash must be present when stable hashing!").stable_hash(hcx, hasher);
    }
}

pub(crate) struct NoneIfHashed<T> {
    pub(super) value: Option<T>,
}

impl<T> fmt::Debug for NoneIfHashed<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        None::<()>.fmt(f)
    }
}

impl<T> StableHash for NoneIfHashed<T> {
    fn stable_hash<Hcx: StableHashCtxt>(&self, hcx: &mut Hcx, hasher: &mut StableHasher) {
        assert!(self.value.is_none());
        0u32.stable_hash(hcx, hasher);
    }
}

#[derive(StableHash, Debug)]
pub(crate) struct HashableCrateHeader {
    // FIXME do we need to hash this?
    pub(crate) triple: TargetTuple,
    // FIXME do we need to hash this?
    pub(crate) name: Symbol,
    // FIXME do we need to hash this?
    pub(crate) is_proc_macro_crate: bool,
    // FIXME do we need to hash this?
    pub(crate) is_stub: bool,
}

/// Stable hashable version of [`CrateRoot`] we use to calculate the public api hash. When adding
/// new fields to `CrateRoot`, it is important to include it in the public api hash. Not doing so
/// can cause silent miscompiles. This struct helps make sure the compiler does not allow forgetting
/// the hashing of new items added to rmeta and makes hashing them a local problem with a
/// straightforward solution.
///
/// When adding a new field to `CrateRoot` we have 3 cases: it is encoded directly, bools for
/// example, it is added as a `LazyValue` or `LazyArray`, or it is added as a new table to
/// `LazyTables`
/// 1. When encoded directly, the type likely implements `StableHash` or it can be derived for it.
///    Simply adding it as-is to `HashableCrateRoot` then moving it into `CrateRoot` in
///    `into_crate_root` should do the trick.
/// 2. When encoded as some kind of lazy value, if one tries to do the same as in 1, the compiler
///    will complain that it does not implement `StableHash`, so it cannot derive
///    `StableHash` for `HashableCrateRoot`. In this case one should wrap it in [`Hashed`] and hash
///    it where it was encoded. the `hash_lazy_array` macro can help with this for simple arrays.
/// 3. When added to `LazyTables` in the `define_tables!` macro, simply defining its type as
///    `Table<RDRHashAll, Idx, Value>` will take care of hashing it.
///
/// In all 3 cases a `// FIXME do we need to hash this?` comment should be included with the new
/// field to note that it wasn't reviewed from the public api hash point of view. It might need
/// stable sorting, or removing parts or all of it from the hash. However, removing anything from
/// the public api hash should be done with extreme scrutiny. In most cases one can likely improve
/// it by stable sorting before encoding, or removing unneeded items before encoding. If it does
/// not need to be in the public api, it likely does not need to be in the rmeta at all. This also
/// improves rmeta sizes.
#[derive(StableHash, Debug)]
pub(crate) struct HashableCrateRoot {
    // FIXME do we need to hash this?
    // everything_downstream
    pub(crate) header: HashableCrateHeader,

    // FIXME do we need to hash this?
    // everything downstream
    pub(crate) extra_filename: String,
    // FIXME do we need to hash this?
    // everything_downstream
    pub(crate) stable_crate_id: StableCrateId,
    // FIXME do we need to hash this?
    // everything_downstream, right?
    pub(crate) required_panic_strategy: Option<PanicStrategy>,
    // FIXME do we need to hash this?
    // everything_downstream, right?
    pub(crate) panic_in_drop_strategy: PanicStrategy,
    // FIXME do we need to hash this?
    // macro expansion in downstream crates uses it, only relevant for publicly reachable macros
    pub(crate) edition: Edition,
    // FIXME do we need to hash this?
    // everything_downstream, there can only be a single one globally
    pub(crate) has_global_allocator: bool,
    // FIXME do we need to hash this?
    // everything_downstream, there can only be a single one globally
    pub(crate) has_alloc_error_handler: bool,
    // FIXME do we need to hash this?
    // everything_downstream, there can only be a single one globally
    pub(crate) has_panic_handler: bool,
    // FIXME do we need to hash this?
    // everything_downstream, there can only be a single one globally
    pub(crate) has_default_lib_allocator: bool,
    // Changing externally implementable items should cause recompiles in all downstream crates.
    // FIXME EiiDecl and EiiImpl contain spans. Should changing the span of these items cause
    // recompiles?
    // FIXME eii-s are collected in `rustc_metadata::eii::collect`. We should probably stable sort
    // that there to make the query result more stable (but sorting might be useless if this
    // should be sensitive to span changes)
    // everything_downstream, there can only be a single one of each eii globally
    pub(crate) externally_implementable_items: Hashed<LazyArray<EiiMapEncodedKeyValue>>,

    // FIXME do we need to hash this?
    // we only need the everything_downstream hashes from here included in our everything_downstream
    pub(crate) crate_deps: Hashed<LazyArray<CrateDep>>,
    // FIXME do we need to hash this?
    // lets say everything_downstream, but not sure
    pub(crate) dylib_dependency_formats: Hashed<LazyArray<Option<LinkagePreference>>>,
    // FIXME do we need to hash this?
    // lets say everything_downstream, but not sure
    pub(crate) lib_features: Hashed<LazyArray<(Symbol, FeatureStability)>>,
    // FIXME do we need to hash this?
    // lets say everything_downstream, but not sure
    pub(crate) stability_implications: Hashed<LazyArray<(Symbol, Symbol)>>,
    // FIXME do we need to hash this?
    // everything_downstream, lang items impls are checked for uniqueness globally
    pub(crate) lang_items: Hashed<LazyArray<(DefIndex, LangItem)>>,
    // FIXME do we need to hash this?
    // everything_downstream, lang items impls are checked for uniqueness globally
    pub(crate) lang_items_missing: Hashed<LazyArray<LangItem>>,
    // FIXME do we need to hash this?
    // private_hash, only used when we are creating error diagnostics
    pub(crate) stripped_cfg_items: Hashed<LazyArray<StrippedCfgItem<DefIndex>>>,
    // FIXME do we need to hash this?
    // lets say everything_downstream for now, I think this is mostly used in the std
    pub(crate) diagnostic_items: Hashed<LazyArray<(Symbol, DefIndex)>>,
    // FIXME do we need to hash this?
    // let say everything_downstream
    pub(crate) native_libraries: Hashed<LazyArray<NativeLib>>,
    // FIXME do we need to hash this?
    // dont know what this is used for. doc says `extern` blocks
    pub(crate) foreign_modules: Hashed<LazyArray<ForeignModule>>,
    // FIXME do we need to hash this?
    // the traits defined in this crate. Definitely not needed in everything_downstream as is
    // maybe we can leak private traits through MIR?
    // According to the docs, and some personal digging, this is used by rustdoc and error reporting
    // rustdoc - problem for another day
    // error reporting: this should not be included in the public hash, as it is only read when the
    // compiler errors. Just like stripped_cfg_items
    pub(crate) traits: Hashed<LazyArray<DefIndex>>,
    // FIXME do we need to hash this?
    // the traits impls in this crate. Definitely not needed in everything_downstream as is
    // how is it needed?
    //
    // this is exposed as a full query with `trait_impls_of_crate`. Which is only used in
    // rustc_public. Otherwise the `implementations_of_trait` is the only other consumer of this
    // field. `implementations_of_trait` works as a map. You need to provide a trait DefId to get
    // its impls, so this should be intergrated into the IndexGraph.
    //
    // When a trait is local:
    //      if it isn't reachable, the impls can should be left out from the hash
    // When a trait is from another crate:
    //      even if this crate does not reexport that trait, a downstream dependency can import it
    //      from another crate, then invoke its methods. So all of those impls must be included, but
    //      only the ones that can be applied to publicly reachable types
    pub(crate) impls: Hashed<LazyArray<TraitImpls>>,
    // FIXME do we need to hash this?
    // what is this used for
    pub(crate) incoherent_impls: Hashed<LazyArray<IncoherentImpls>>,
    // FIXME do we need to hash this?
    // what is this used for, some kind of const eval?
    pub(crate) interpret_alloc_index: Hashed<LazyArray<u64>>,
    // FIXME do we need to hash this?
    // proc macro, ignored
    pub(crate) proc_macro_data: NoneIfHashed<ProcMacroData>,

    // FIXME do we need to hash this?
    // do it by table
    pub(crate) tables: Hashed<LazyTables>,
    // FIXME do we need to hash this?
    // likely don't only private. Do debuggers use this when debugging the linked binary? Then this
    // is private_hash
    pub(crate) debugger_visualizers: Hashed<LazyArray<DebuggerVisualizerFile>>,

    // FIXME do we need to hash this?
    // this is used to do symbol mangling in downstream crates. We should only include ones that
    // are somehow reachable.
    pub(crate) exportable_items: Hashed<LazyArray<DefIndex>>,
    // FIXME do we need to hash this?
    // what is this extactly, used for diagnostics
    pub(crate) stable_order_of_exportable_impls: Hashed<LazyArray<(DefIndex, usize)>>,
    // FIXME do we need to hash this?
    // exported_non_generic_symbols and exported_symbols are used by the linker, which likely means
    // that we should put them behind the private hash
    pub(crate) exported_non_generic_symbols:
        Hashed<LazyArray<(ExportedSymbol<'static>, SymbolExportInfo)>>,

    // FIXME do we need to hash this?
    // I think this is used to find upstream_monomorphizations or what
    // So this is tricky, and kind of makes this feature useless. For example, using a
    // Vec::<u32>::push in private code will add that monomorphized function to
    // exported_generic_symbols.
    // On one hand, removing this will slow down the codegen for the current crate, as it needs to
    // codegen these generics. On the other hand, keeping this forces a rebuild of all upstream
    // crates when any new monomorphized item (where the monomorphized generic is public or from
    // another crate).
    // Let's review this from an incremental recompilation perspecitve. The goal of
    // relink-dont-rebuild is mostly speeding that up.
    // Let's say we remove this, the first build was done
    // 1. recompile of a dependency that contains a monomorphization which could be used upstream.
    //    What changes? Nothing, we don't have to export the monomorphization for
    //    `upstream_monomorphizations`, but its is also used by the linker, so it has to be included
    //    anyways
    // 2. recompile of crate that could use an upstream monomorphization. Since it is disabled, the
    //    monomorphization is already in its local incremental cache. It will be pulled from there.
    // Given these two points, if the goal is fast incremental recompiles, then disabling upstream
    // monomorphizations, and moving this behind private_hash might be the best (maybe linking_hash)
    pub(crate) exported_generic_symbols:
        Hashed<LazyArray<(ExportedSymbol<'static>, SymbolExportInfo)>>,

    // FIXME do we need to hash this?
    // some kind of macro expansion stuff?
    pub(crate) syntax_contexts: Hashed<SyntaxContextTable>,
    // FIXME do we need to hash this?
    // some kind of macro expansion stuff?
    pub(crate) expn_data: Hashed<ExpnDataTable>,
    // FIXME do we need to hash this?
    // some kind of macro expansion stuff?
    pub(crate) expn_hashes: Hashed<ExpnHashTable>,

    // FIXME do we need to hash this?
    // we stbale hash all localdefids currently to get this. Likely only need the ones that are
    // somehow reachable
    pub(crate) def_path_hash_map: Hashed<LazyValue<DefPathHashMapRef<'static>>>,

    // FIXME do we need to hash this?
    // used for diagnostics and line info generation in codegen? To map span data to line/column
    // data
    // we need this while the spans are encoded like that.
    pub(crate) source_map: Hashed<LazyTable<u32, Option<LazyValue<rustc_span::SourceFile>>>>,
    // FIXME do we need to hash this?
    // everything_downstream, prbably not changing much
    pub(crate) target_modifiers: Hashed<LazyArray<TargetModifier>>,
    // FIXME do we need to hash this?
    // everything_downstream?
    pub(crate) denied_partial_mitigations: Hashed<LazyArray<DeniedPartialMitigation>>,

    // FIXME do we need to hash this?
    // everything_downstream?
    pub(crate) compiler_builtins: bool,
    // FIXME do we need to hash this?
    // everything_downstream?
    pub(crate) needs_allocator: bool,
    // FIXME do we need to hash this?
    // everything_downstream?
    pub(crate) needs_panic_runtime: bool,
    // FIXME do we need to hash this?
    // everything_downstream?
    pub(crate) no_builtins: bool,
    // FIXME do we need to hash this?
    // everything_downstream?
    pub(crate) panic_runtime: bool,
    // FIXME do we need to hash this?
    // everything_downstream?
    pub(crate) profiler_runtime: bool,
    // FIXME do we need to hash this?
    // everything_downstream?
    pub(crate) symbol_mangling_version: SymbolManglingVersion,

    // FIXME do we need to hash this?
    // everything_downstream?
    pub(crate) specialization_enabled_in: bool,
}

impl HashableCrateRoot {
    pub(super) fn into_crate_root(
        self,
        ecx: &mut EncodeContext<'_, '_>,
        hcx: &mut PublicApiHashingContext<'_>,
    ) -> CrateRoot {
        let tcx = ecx.tcx;
        let (rdr_hashes, hashes) = if hcx.hash_public_api {
            assert!(!self.header.is_proc_macro_crate);
            let graph = ecx.index_graph_builder.take().unwrap().build_graph(&mut hcx.hcx);
            debug!("Hash graph {graph:?}");
            let public_hashes =
                build_public_hashes(&graph, &hcx.def_id_hashes, ecx.tcx, &mut hcx.hcx);

            let mut hasher = StableHasher::default();
            let public_global_hash = stable_hash(&mut hcx.hcx, &self);
            public_global_hash.stable_hash(&mut hcx.hcx, &mut hasher);
            public_hashes.stable_hash(&mut hcx.hcx, &mut hasher);
            let rdr_hashes = public_hashes.value.encode(ecx, hcx, public_global_hash);
            let public_hash = Svh::new(hasher.finish());
            debug!("Hashed crate root: {self:#x?}");
            debug!("public api hash: {}", public_hash);
            (
                Some(rdr_hashes),
                CrateHashes { public_hash, private_hash: tcx.crate_hash(LOCAL_CRATE) },
            )
        } else {
            let hash = tcx.crate_hash(LOCAL_CRATE);
            (None, CrateHashes { public_hash: hash, private_hash: hash })
        };
        let header = self.header;
        let header = CrateHeader {
            triple: header.triple,
            hashes,
            name: header.name,
            is_proc_macro_crate: header.is_proc_macro_crate,
            is_stub: header.is_stub,
        };
        CrateRoot {
            header,

            extra_filename: self.extra_filename,
            stable_crate_id: self.stable_crate_id,
            required_panic_strategy: self.required_panic_strategy,
            panic_in_drop_strategy: self.panic_in_drop_strategy,
            edition: self.edition,
            has_global_allocator: self.has_global_allocator,
            has_alloc_error_handler: self.has_alloc_error_handler,
            has_panic_handler: self.has_panic_handler,
            has_default_lib_allocator: self.has_default_lib_allocator,
            externally_implementable_items: self.externally_implementable_items.value,

            crate_deps: self.crate_deps.value,
            dylib_dependency_formats: self.dylib_dependency_formats.value,
            lib_features: self.lib_features.value,
            stability_implications: self.stability_implications.value,
            lang_items: self.lang_items.value,
            lang_items_missing: self.lang_items_missing.value,
            stripped_cfg_items: self.stripped_cfg_items.value,
            diagnostic_items: self.diagnostic_items.value,
            native_libraries: self.native_libraries.value,
            foreign_modules: self.foreign_modules.value,
            traits: self.traits.value,
            impls: self.impls.value,
            incoherent_impls: self.incoherent_impls.value,
            interpret_alloc_index: self.interpret_alloc_index.value,
            proc_macro_data: self.proc_macro_data.value,

            tables: self.tables.value,
            debugger_visualizers: self.debugger_visualizers.value,

            exportable_items: self.exportable_items.value,
            stable_order_of_exportable_impls: self.stable_order_of_exportable_impls.value,
            exported_non_generic_symbols: self.exported_non_generic_symbols.value,
            exported_generic_symbols: self.exported_generic_symbols.value,

            syntax_contexts: self.syntax_contexts.value,
            expn_data: self.expn_data.value,
            expn_hashes: self.expn_hashes.value,

            def_path_hash_map: self.def_path_hash_map.value,

            source_map: self.source_map.value,
            target_modifiers: self.target_modifiers.value,
            denied_partial_mitigations: self.denied_partial_mitigations.value,

            compiler_builtins: self.compiler_builtins,
            needs_allocator: self.needs_allocator,
            needs_panic_runtime: self.needs_panic_runtime,
            no_builtins: self.no_builtins,
            panic_runtime: self.panic_runtime,
            profiler_runtime: self.profiler_runtime,
            symbol_mangling_version: self.symbol_mangling_version,

            specialization_enabled_in: self.specialization_enabled_in,
            rdr_hashes,
        }
    }
}

pub(super) enum RecordMode<'tcx> {
    From(LocalNode<'tcx>),
}

impl<'tcx> RecordMode<'tcx> {
    pub(super) fn from(from: LocalNode<'tcx>) -> Self {
        Self::From(from)
    }
}

struct IndexGraph<'tcx> {
    nodes: IndexMap<Node<'tcx>, Fingerprint>,
    edges: IndexVec<NodeIdx, Vec<NodeIdx>>,
    roots: Vec<NodeIdx>,
}

impl fmt::Debug for IndexGraph<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let node = |idx| self.nodes.get_index(idx).unwrap().0;
        let fingerprint = |idx| self.nodes.get_index(idx).unwrap().1;
        let reachable = self.reachable_set();
        writeln!(f, "roots:")?;
        for root in &self.roots {
            writeln!(f, "   {:?}", node(*root))?;
        }
        writeln!(f, "nodes:")?;
        for (from, edges) in self.edges.iter_enumerated() {
            writeln!(
                f,
                "{:?}: {:x?} reachable: {}",
                node(from),
                fingerprint(from),
                reachable[from]
            )?;
            for to in edges {
                writeln!(f, "   {:?}", node(*to))?;
            }
        }
        Ok(())
    }
}

impl<'tcx> IndexGraph<'tcx> {
    fn reachable_set(&self) -> IndexVec<NodeIdx, bool> {
        let mut reachable = IndexVec::from_elem_n(false, self.edges.len());
        let mut stack = self.roots.clone();

        while let Some(node) = stack.pop() {
            if reachable[node] {
                continue;
            }
            reachable[node] = true;
            stack.extend_from_slice(&self.edges[node]);
        }

        reachable
    }
}

pub(super) struct IndexGraphBuilder<'tcx> {
    pub(super) record_mode: Vec<RecordMode<'tcx>>,
    edges: FxHashMap<LocalNode<'tcx>, FxHashSet<Node<'tcx>>>,
    roots: FxHashSet<Node<'tcx>>,
}

impl Default for IndexGraphBuilder<'_> {
    fn default() -> Self {
        Self {
            record_mode: Default::default(),
            edges: Default::default(),
            roots: [Node::DefId(rustc_hir::def_id::CRATE_DEF_ID.into())].into_iter().collect(),
        }
    }
}

fn stable_hash<'a, T: StableHash + ?Sized>(
    hcx: &mut StableHashingContext<'a>,
    val: &T,
) -> Fingerprint {
    let mut hasher = StableHasher::new();
    val.stable_hash(hcx, &mut hasher);
    hasher.finish()
}

impl<'tcx> IndexGraphBuilder<'tcx> {
    fn build_graph(mut self, hcx: &mut StableHashingContext<'_>) -> IndexGraph<'tcx> {
        // ExpnId::root() contains CRATE_DEF_ID as its macro_def_id, but that isn't used for
        // anything.
        // TODO add newtype that ensures it really isn't used
        self.edges
            .get_mut(&LocalNode::ExpnId(LocalExpnId::ROOT))
            .unwrap()
            .remove(&Node::DefId(CRATE_DEF_ID.into()));
        let mut hashes = IndexMap::default();
        // iterating over FxHashSet and FxHashMap is fine here, as it is only used to build the
        // hashes map, which is never returned or iterated
        #[allow(rustc::potential_query_instability)]
        for node in self
            .edges
            .iter()
            .flat_map(|(node, edges)| {
                std::iter::once(node.into_node()).chain(edges.iter().copied())
            })
            .chain(self.roots.iter().copied())
        {
            match hashes.entry(node) {
                Entry::Vacant(v) => {
                    let hash = stable_hash(hcx, &node);
                    v.insert(hash);
                }
                Entry::Occupied(_o) => {}
            }
        }
        hashes.sort_by_key(|_, v| *v);

        // iterating here is fine, as we stable sort right after
        #[allow(rustc::potential_query_instability)]
        let mut roots: Vec<_> =
            self.roots.into_iter().map(|node| hashes.get_index_of(&node).unwrap()).collect();
        roots.sort();
        let mut edges = IndexVec::from_elem_n(Vec::default(), hashes.len());
        // iterating here is fine, as we stable when saving to `edges`
        #[allow(rustc::potential_query_instability)]
        for (node, node_edges) in self.edges {
            // iterating here is fine, as we stable sort right after
            #[allow(rustc::potential_query_instability)]
            let mut node_edges: Vec<_> =
                node_edges.into_iter().map(|node| hashes.get_index_of(&node).unwrap()).collect();
            node_edges.sort();
            edges[hashes.get_index_of(&node.into_node()).unwrap()] = node_edges;
        }

        IndexGraph { roots, edges, nodes: hashes }
    }

    pub(super) fn record(&mut self, to: Node<'tcx>) {
        match self.record_mode.last() {
            Some(RecordMode::From(from)) => {
                self.edges.entry(*from).or_insert(Default::default()).insert(to);
            }
            None => {
                self.roots.insert(to);
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct IndexGraphHashes {
    local: UnordMap<LocalDefId, Fingerprint>,
    expn: UnordMap<LocalExpnId, Fingerprint>,
    syntax: UnordMap<SyntaxContext, Fingerprint>,
}

impl IndexGraphHashes {
    fn get_mut<I: TableIndex>(&mut self, i: I) -> &mut Fingerprint {
        i.index_mut(self)
    }

    fn get_node(&self, node: &Node<'_>, tcx: TyCtxt<'_>) -> Option<Fingerprint> {
        match node {
            Node::DefId(id) => {
                if let Some(local) = id.as_local() {
                    self.local.get(&local).copied()
                } else {
                    Some(tcx.extern_def_public_hash(*id))
                }
            }
            Node::ExpnId(id) => {
                if let Some(local) = id.as_local() {
                    self.expn.get(&local).copied()
                } else {
                    Some(tcx.extern_expn_public_hash(*id))
                }
            }
            Node::SyntaxContext(id) => self.syntax.get(id).copied(),
            Node::Span(span) => tcx.with_stable_hashing_context(|hcx| {
                let hcx = RefCell::new(hcx);
                tcx.sess
                    .source_map()
                    .span_to_source(*span, |source, start, end| {
                        Ok(stable_hash(&mut hcx.borrow_mut(), &source[start..end]))
                    })
                    .ok()
            }),
            Node::Ty(_) => None,
            Node::Predicate(_) => None,
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Hash)]
pub(super) enum LocalNode<'tcx> {
    DefId(LocalDefId),
    ExpnId(LocalExpnId),
    SyntaxContext(SyntaxContext),
    Ty(Ty<'tcx>),
    Predicate(PredicateKind<'tcx>),
}

#[derive(PartialEq, Eq, Clone, Copy, Hash, StableHash, Debug)]
pub(super) enum Node<'tcx> {
    DefId(DefId),
    ExpnId(ExpnId),
    SyntaxContext(SyntaxContext),
    Span(Span),
    Ty(Ty<'tcx>),
    Predicate(PredicateKind<'tcx>),
}

impl<'tcx> LocalNode<'tcx> {
    fn into_node(self) -> Node<'tcx> {
        match self {
            LocalNode::DefId(id) => Node::DefId(id.into()),
            LocalNode::ExpnId(id) => Node::ExpnId(id.to_expn_id()),
            LocalNode::SyntaxContext(id) => Node::SyntaxContext(id),
            LocalNode::Ty(id) => Node::Ty(id),
            LocalNode::Predicate(id) => Node::Predicate(id),
        }
    }
}

impl From<LocalDefId> for Node<'_> {
    fn from(value: LocalDefId) -> Self {
        Self::DefId(value.into())
    }
}
impl From<DefId> for Node<'_> {
    fn from(value: DefId) -> Self {
        Self::DefId(value)
    }
}
impl From<ExpnId> for Node<'_> {
    fn from(value: ExpnId) -> Self {
        Self::ExpnId(value)
    }
}
impl From<SyntaxContext> for Node<'_> {
    fn from(value: SyntaxContext) -> Self {
        Self::SyntaxContext(value)
    }
}

pub(crate) trait TableIndex: Copy {
    type Encoded: Idx;
    fn index_mut(self, hashes: &mut IndexGraphHashes) -> &mut Fingerprint;
    fn into_encoded(self) -> Self::Encoded;
}

impl TableIndex for DefIndex {
    type Encoded = DefIndex;
    fn index_mut(self, hashes: &mut IndexGraphHashes) -> &mut Fingerprint {
        hashes.local.entry(LocalDefId { local_def_index: self }).or_insert(Fingerprint::ZERO)
    }
    fn into_encoded(self) -> Self::Encoded {
        self
    }
}

impl TableIndex for LocalDefId {
    type Encoded = DefIndex;
    fn index_mut(self, hashes: &mut IndexGraphHashes) -> &mut Fingerprint {
        hashes.local.entry(self).or_insert(Fingerprint::ZERO)
    }
    fn into_encoded(self) -> Self::Encoded {
        self.local_def_index
    }
}

impl TableIndex for LocalExpnId {
    type Encoded = ExpnIndex;
    fn index_mut(self, hashes: &mut IndexGraphHashes) -> &mut Fingerprint {
        hashes.expn.entry(self).or_insert(Fingerprint::ZERO)
    }
    fn into_encoded(self) -> Self::Encoded {
        self.as_raw()
    }
}

impl TableIndex for SyntaxContext {
    type Encoded = u32;
    fn index_mut(self, hashes: &mut IndexGraphHashes) -> &mut Fingerprint {
        hashes.syntax.entry(self).or_insert(Fingerprint::ZERO)
    }
    fn into_encoded(self) -> Self::Encoded {
        self.as_u32()
    }
}

impl DirectedGraph for IndexGraph<'_> {
    type Node = NodeIdx;

    fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

impl Successors for IndexGraph<'_> {
    fn successors(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> {
        self.edges[node].iter().copied()
    }
}

type NodeIdx = usize;
type SccIdx = u32;

#[derive(Debug, Copy, Clone)]
struct FingerprintAnnotation {
    fingerprint: Fingerprint,
}

impl Annotation for FingerprintAnnotation {
    fn update_scc(&mut self, other: &Self) {
        self.fingerprint = self.fingerprint.combine_commutative(other.fingerprint);
    }

    fn update_reachable(&mut self, other: &Self) {
        self.fingerprint = self.fingerprint.combine_commutative(other.fingerprint);
    }
}

struct FingerprintAnnotations<'a, 'tcx> {
    graph: &'a IndexGraph<'tcx>,
    hashes: &'a IndexGraphHashes,
    tcx: TyCtxt<'tcx>,
    scc_fingerprints: IndexVec<SccIdx, Fingerprint>,
}

impl<'a, 'tcx> FingerprintAnnotations<'a, 'tcx> {
    fn new(graph: &'a IndexGraph<'tcx>, hashes: &'a IndexGraphHashes, tcx: TyCtxt<'tcx>) -> Self {
        Self { graph, hashes, scc_fingerprints: IndexVec::with_capacity(graph.nodes.len()), tcx }
    }
}

impl<'a, 'tcx> Annotations<NodeIdx> for FingerprintAnnotations<'a, 'tcx> {
    type Ann = FingerprintAnnotation;
    type SccIdx = SccIdx;

    fn new(&self, node: NodeIdx) -> FingerprintAnnotation {
        FingerprintAnnotation {
            fingerprint: if let Some(encoded_data_hash) =
                self.hashes.get_node(self.graph.nodes.get_index(node).unwrap().0, self.tcx)
            {
                self.graph.nodes.get_index(node).unwrap().1.combine_commutative(encoded_data_hash)
            } else {
                *self.graph.nodes.get_index(node).unwrap().1
            },
        }
    }

    fn annotate_scc(&mut self, scc: SccIdx, annotation: FingerprintAnnotation) {
        assert_eq!(self.scc_fingerprints.len(), scc.index());
        self.scc_fingerprints.push(annotation.fingerprint);
    }
}

fn build_public_hashes<'tcx>(
    graph: &IndexGraph<'tcx>,
    hashes: &IndexGraphHashes,
    tcx: TyCtxt<'tcx>,
    hcx: &mut StableHashingContext<'_>,
) -> Hashed<ItemPublicHashesBuilder> {
    let mut annotations = FingerprintAnnotations::new(graph, hashes, tcx);
    let sccs = Sccs::<_, SccIdx>::new_with_annotation(graph, &mut annotations);
    let mut public_hashes = ItemPublicHashesBuilder::default();
    let mut hasher = StableHasher::new();
    for (node_index, reachable) in graph.reachable_set().iter_enumerated() {
        if !reachable {
            continue;
        }
        match graph.nodes.get_index(node_index).unwrap().0 {
            Node::DefId(id) => {
                if let Some(local) = id.as_local() {
                    let fingerprint = annotations.scc_fingerprints[sccs.scc(node_index)];
                    (local, fingerprint).stable_hash(hcx, &mut hasher);
                    public_hashes.local.set_some_unhashed(local.local_def_index, fingerprint);
                }
            }
            Node::ExpnId(id) => {
                if let Some(local) = id.as_local() {
                    let fingerprint = annotations.scc_fingerprints[sccs.scc(node_index)];
                    (local, fingerprint).stable_hash(hcx, &mut hasher);
                    public_hashes.expn.set_some_unhashed(local.as_raw(), fingerprint);
                }
            }
            Node::SyntaxContext(_) => (),
            Node::Ty(_) => (),
            Node::Predicate(_) => (),
            Node::Span(_) => (),
        }
    }
    Hashed { value: public_hashes, hash: Some(hasher.finish()) }
}

#[derive(Default)]
pub(crate) struct ItemPublicHashesBuilder {
    local: TableBuilder<RDRHashNone<DefIndex>, DefIndex, Option<Fingerprint>>,
    expn: TableBuilder<RDRHashNone<ExpnIndex>, ExpnIndex, Option<Fingerprint>>,
}

impl ItemPublicHashesBuilder {
    fn encode<'a, 'tcx>(
        &self,
        ecx: &mut EncodeContext<'a, 'tcx>,
        hcx: &mut PublicApiHashingContext<'_>,
        public_global_hash: Fingerprint,
    ) -> ItemPublicHashes {
        ItemPublicHashes {
            local: self.local.encode(&mut ecx.opaque, hcx).value,
            expn: self.expn.encode(&mut ecx.opaque, hcx).value,
            public_global_hash,
        }
    }
}

#[derive(LazyDecodable, MetadataEncodable)]
pub(crate) struct ItemPublicHashes {
    pub(crate) local: LazyTable<DefIndex, Option<Fingerprint>>,
    pub(crate) expn: LazyTable<ExpnIndex, Option<Fingerprint>>,
    pub(crate) public_global_hash: Fingerprint,
}
