use core::iter::Iterator;
use std::cell::RefCell;
use std::fmt;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::graph::scc::Sccs;
use rustc_data_structures::graph::{DirectedGraph, Successors};
use rustc_data_structures::indexmap::map::Entry;
use rustc_data_structures::stable_hash::{StableHash, StableHashCtxt, StableHasher};
use rustc_data_structures::svh::Svh;
use rustc_data_structures::unord::UnordMap;
use rustc_hir::LangItem;
use rustc_hir::attrs::StrippedCfgItem;
use rustc_hir::def_id::DefIndex;
use rustc_index::{Idx, IndexVec};
use rustc_macros::{LazyDecodable, MetadataEncodable, StableHash};
use rustc_middle::ich::StableHashState;
use rustc_middle::middle::debugger_visualizer::DebuggerVisualizerFile;
use rustc_middle::middle::exported_symbols::{ExportedSymbol, SymbolExportInfo};
use rustc_middle::middle::lib_features::FeatureStability;
use rustc_middle::mir::interpret::AllocId;
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
use crate::rmeta::encoder::{DefPathHashDefIdEncoder, MetadataEncoder};
use crate::rmeta::{
    CrateHashes, DefPathHashMapRef, EiiMapEncodedKeyValue, EncodeContext, EncodedTraitImpls,
    ExpnDataTable, ExpnHashTable, IncoherentImpls, LazyArray, LazyDecoder, LazyTable, LazyTables,
    LazyValue, ProcMacroData, SyntaxContextTable, TableBuilder,
};

#[derive(Default)]
pub(crate) struct PublicApiHasher(StableHasher);

impl PublicApiHasher {
    #[inline(always)]
    pub(crate) fn finish<'a>(self, hcx: &mut impl PublicApiHashState<'a>) -> Option<Fingerprint> {
        hcx.enabled().then(|| self.0.finish())
    }

    #[inline(always)]
    pub(crate) fn digest<'a, T: StableHash>(
        &mut self,
        value: T,
        hcx: &mut impl PublicApiHashState<'a>,
    ) {
        if hcx.enabled() {
            value.stable_hash(hcx.hcx_mut(), &mut self.0);
        }
    }
}

pub(crate) trait TablePublicApiHasher<I: Idx>: Default {
    type IterHasher;

    fn digest<'a, V>(
        &mut self,
        index: impl TableIndex<Encoded = I>,
        value: V,
        hcx: &mut impl PublicApiHashState<'a>,
    ) where
        V: StableHash;

    fn iter_hasher(&self) -> Self::IterHasher;
}

pub(crate) struct RDRHashAll<I: Idx> {
    _marker: std::marker::PhantomData<I>,
}

impl<I: Idx> Default for RDRHashAll<I> {
    fn default() -> Self {
        Self { _marker: Default::default() }
    }
}

pub(crate) trait PublicApiHashState<'a> {
    fn enabled(&self) -> bool;

    fn hcx_mut(&mut self) -> &mut StableHashState<'a, true>;

    fn add_node_hash(&mut self, node: &Node<'a>, hash: Fingerprint);

    fn hcx_with_def_id_hashes(
        &mut self,
    ) -> (&mut StableHashState<'a, true>, &ReachabilityGraphHashes);
}

impl<'a, const ENABLED: bool> PublicApiHashState<'a> for PublicApiHashingContext<'a, ENABLED> {
    #[inline(always)]
    fn enabled(&self) -> bool {
        ENABLED
    }

    #[inline(always)]
    fn hcx_mut(&mut self) -> &mut StableHashState<'a, true> {
        &mut self.hcx
    }

    #[inline(always)]
    fn add_node_hash(&mut self, node: &Node<'a>, hash: Fingerprint) {
        if self.enabled() {
            self.def_id_hashes.add_node_hash(node, hash);
        }
    }

    fn hcx_with_def_id_hashes(
        &mut self,
    ) -> (&mut StableHashState<'a, true>, &ReachabilityGraphHashes) {
        (&mut self.hcx, &self.def_id_hashes)
    }
}

pub(crate) struct PublicApiHashingContext<'a, const ENABLED: bool> {
    pub(crate) hcx: StableHashState<'a, true>,
    def_id_hashes: ReachabilityGraphHashes,
}

impl<'a, const ENABLED: bool> PublicApiHashingContext<'a, ENABLED> {
    pub(crate) fn new(hcx: StableHashState<'a, false>) -> Self {
        Self { hcx: hcx.hash_spans_as_parentless(), def_id_hashes: Default::default() }
    }
}

impl<I: Idx> TablePublicApiHasher<I> for RDRHashAll<I> {
    type IterHasher = OrderedIterHasher;
    #[inline(always)]
    fn digest<'a, V>(
        &mut self,
        index: impl TableIndex<Encoded = I>,
        value: V,
        hcx: &mut impl PublicApiHashState<'a>,
    ) where
        V: StableHash,
    {
        if !hcx.enabled() {
            return;
        }
        let mut hasher = StableHasher::default();
        value.stable_hash(hcx.hcx_mut(), &mut hasher);
        hcx.add_node_hash(&index.into(), hasher.finish());
    }

    #[inline(always)]
    fn iter_hasher(&self) -> Self::IterHasher {
        OrderedIterHasher::default()
    }
}

#[derive(Default)]
pub(crate) struct OrderedIterHasher(StableHasher);

impl OrderedIterHasher {
    #[inline(always)]
    pub(crate) fn inspect_digest<'a: 'b, 'b, I>(
        &'b mut self,
        iter: I,
        hcx: &'b mut impl PublicApiHashState<'a>,
    ) -> impl Iterator<Item = I::Item> + 'b
    where
        I: IntoIterator + 'b,
        I::Item: StableHash,
    {
        iter.into_iter().inspect(move |item: &I::Item| {
            if hcx.enabled() {
                item.stable_hash(hcx.hcx_mut(), &mut self.0)
            }
        })
    }

    #[inline(always)]
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
    #[inline(always)]
    fn digest<'a, V>(
        &mut self,
        _index: impl TableIndex<Encoded = I>,
        _value: V,
        _hcx: &mut impl PublicApiHashState<'a>,
    ) where
        V: StableHash,
    {
    }

    #[inline(always)]
    fn iter_hasher(&self) -> Self::IterHasher {
        Default::default()
    }
}

/// Map structs where the hashing took place for each item and stored into the [`ReachabilityGraphHashes`]
#[derive(Default)]
pub(crate) struct GraphHashed<T>(pub(crate) T);
impl<T> StableHash for GraphHashed<T> {
    fn stable_hash<Hcx: StableHashCtxt>(&self, _hcx: &mut Hcx, _hasher: &mut StableHasher) {}
}

impl<T> fmt::Debug for GraphHashed<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GraphHashed")
    }
}

#[derive(Default)]
pub(crate) struct Unhashed<T>(pub(crate) T);
impl<T> StableHash for Unhashed<T> {
    fn stable_hash<Hcx: StableHashCtxt>(&self, _hcx: &mut Hcx, _hasher: &mut StableHasher) {}
}

impl<T> fmt::Debug for Unhashed<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        "unhashed".fmt(f)
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
    pub(crate) triple: TargetTuple,
    pub(crate) name: Symbol,
    pub(crate) is_proc_macro_crate: bool,
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
    // =========== not reviewed ============
    // These are were not reviewed for public api hashing, so they are included in the global hash
    // for now. Any change in these fields will cause a recompile of all downstream dependencies.
    pub(crate) header: HashableCrateHeader,
    pub(crate) required_panic_strategy: Option<PanicStrategy>,
    pub(crate) panic_in_drop_strategy: PanicStrategy,
    pub(crate) dylib_dependency_formats: Hashed<LazyArray<Option<LinkagePreference>>>,
    pub(crate) lib_features: Hashed<LazyArray<(Symbol, FeatureStability)>>,
    pub(crate) stability_implications: Hashed<LazyArray<(Symbol, Symbol)>>,
    // I think this is mostly used in the std, so it doesn't matter much whether we include it
    pub(crate) diagnostic_items: Hashed<LazyArray<(Symbol, DefIndex)>>,
    pub(crate) native_libraries: Hashed<LazyArray<NativeLib>>,
    pub(crate) foreign_modules: Hashed<LazyArray<ForeignModule>>,
    pub(crate) incoherent_impls: Hashed<LazyArray<IncoherentImpls>>,
    pub(crate) debugger_visualizers: Hashed<LazyArray<DebuggerVisualizerFile>>,
    pub(crate) stable_order_of_exportable_impls: Hashed<LazyArray<(DefIndex, usize)>>,
    pub(crate) exported_non_generic_symbols:
        Hashed<LazyArray<(ExportedSymbol<'static>, SymbolExportInfo)>>,
    pub(crate) exported_generic_symbols:
        Hashed<LazyArray<(ExportedSymbol<'static>, SymbolExportInfo)>>,
    pub(crate) target_modifiers: Hashed<LazyArray<TargetModifier>>,
    pub(crate) denied_partial_mitigations: Hashed<LazyArray<DeniedPartialMitigation>>,
    pub(crate) compiler_builtins: bool,
    pub(crate) needs_allocator: bool,
    pub(crate) needs_panic_runtime: bool,
    pub(crate) no_builtins: bool,
    pub(crate) panic_runtime: bool,
    pub(crate) profiler_runtime: bool,
    pub(crate) symbol_mangling_version: SymbolManglingVersion,

    pub(crate) specialization_enabled_in: bool,
    pub(crate) public_api_hash_opt_enabled: bool,

    // =========== global hash  ============
    // Any change in these fields will cause a recompile of all downstream dependencies

    // For the next 4, there can only be a single one globally
    pub(crate) has_global_allocator: bool,
    pub(crate) has_alloc_error_handler: bool,
    pub(crate) has_panic_handler: bool,
    pub(crate) has_default_lib_allocator: bool,

    // extra_filenames and stable_crate_id must be in the global hash since these are needed
    // to differentiate between different version of the same crate. Different versions might have
    // the same public api, but we must keep them separate, as the user might be using different
    // versions exactly because their private implementations differ.
    pub(crate) extra_filename: String,
    pub(crate) stable_crate_id: StableCrateId,

    // macro expansion in downstream crates uses it, only relevant for publicly reachable macros,
    // FIXME: This is left here as it is unlikely to change much between compilations. But we
    // could move this to macro hashes and remove it from the global hash.
    pub(crate) edition: Edition,
    // Changing externally implementable items should cause recompiles in all downstream crates as
    // there can only be a single one of each eii globally.
    // FIXME EiiDecl and EiiImpl contain spans. Should changing the span of these items cause
    // recompiles?
    // FIXME eii-s are collected in `rustc_metadata::eii::collect`. We should probably stable sort
    // that there to make the query result more stable (but sorting might be useless if this
    // should be sensitive to span changes)
    pub(crate) externally_implementable_items: Hashed<LazyArray<EiiMapEncodedKeyValue>>,

    // lang items are checked for uniqueness globally
    pub(crate) lang_items: Hashed<LazyArray<(DefIndex, LangItem)>>,
    // FIXME: Should we exclude this? `lang_items` of this crate and the global hash of
    // dependencies should already cover it
    pub(crate) lang_items_missing: Hashed<LazyArray<LangItem>>,

    // We need the global hashes from the dependencies included in the global hash of this
    // crate
    pub(crate) crate_deps: Hashed<LazyArray<CrateDep>>,

    // =========== graph hashed ============

    // See the tables definition for what is hashed. Hashed tables are interpreted as
    // `LocalDefId -> data` maps and are included in the reachability graph
    pub(crate) tables: GraphHashed<LazyTables>,

    // The relevant parts of the source code are hashed when the spans are hashed as reachability
    // graph nodes.
    pub(crate) source_map: GraphHashed<LazyTable<u32, Option<LazyValue<rustc_span::SourceFile>>>>,

    // Macro expansion data
    pub(crate) syntax_contexts: GraphHashed<SyntaxContextTable>,
    pub(crate) expn_data: GraphHashed<ExpnDataTable>,
    pub(crate) expn_hashes: GraphHashed<ExpnHashTable>,

    // Mir interpreter allocations for const eval. Haven't looked too much into this, but the
    // allocations can include types (and other stuff?), so we include this in the reachability
    // graph to be sure.
    // FIXME Does stable hashing the AllocId already include the stable hash of everything we
    // need for the reachability graph? We might not need to include these explicitly as
    // reachability graph nodes.
    pub(crate) interpret_alloc_index: GraphHashed<LazyArray<u64>>,

    // this is exposed as a full query with `trait_impls_of_crate`. Which is only used in
    // rustc_public. Otherwise the `implementations_of_trait` is the only other consumer of this
    // field. `implementations_of_trait` works as a map. You need to provide a trait DefId to get
    // its impls, so this is integrated into the ReachabilityGraph.
    //
    // When a trait is local:
    //      if it isn't reachable, the impls should be left out from the hash
    // When a trait is from another crate:
    //      even if this crate does not reexport that trait, a downstream dependency can import it
    //      from another crate, then invoke its methods. So all of those impls must be included,
    //      but only the ones that can be applied to publicly reachable types
    pub(crate) impls: GraphHashed<EncodedTraitImpls>,

    // This is used to do symbol mangling in downstream crates. We should only include ones that
    // are somehow reachable.
    // While public api hashing is enabled, a table is used in the `is_exportable` field of
    // `tables` is used to provide this query. This field must be empty.
    pub(crate) exportable_items: GraphHashed<LazyArray<DefIndex>>,

    // =========== not needed in the public hash ==============
    // proc macro, ignored. We use the full crate hash as public hash for proc macros
    pub(crate) proc_macro_data: NoneIfHashed<ProcMacroData>,
    // The introduction of the option to store the DefIndex part of DefId-s as DefPathHash in the
    // metadata was done to remove this from the public hash.
    pub(crate) def_path_hash_map: Unhashed<LazyValue<DefPathHashMapRef<'static>>>,
    // Only used for diagnostics when the compilation session errors
    // FIXME: this is currently removed from the metadata when public api hashing is enabled. We
    // should add a safe way to access this when the compilation session errors.
    pub(crate) stripped_cfg_items: Unhashed<LazyArray<StrippedCfgItem<DefIndex>>>,

    // The traits defined in this crate.
    // According to the docs, and some personal digging, this is used by rustdoc and error reporting
    // error reporting: this should not be included in the public hash, as it is only read when the
    // compiler session finishes with an error. Just like stripped_cfg_items.
    // FIXME: this is currently removed from the metadata when public api hashing is enabled. We
    // should ensure that this is not misused before readding it.
    pub(crate) traits: Unhashed<LazyArray<DefIndex>>,
}

pub(super) fn crate_hashes<'tcx, 'h>(
    ecx: &mut EncodeContext<'_, 'tcx, DefPathHashDefIdEncoder<'tcx>>,
    hcx: &mut impl PublicApiHashState<'h>,
    root: &HashableCrateRoot,
) -> (CrateHashes, Option<ItemPublicHashes>) {
    assert!(!root.header.is_proc_macro_crate);
    let graph = std::mem::take(&mut ecx.spec_encoder_data.reachability_graph_builder)
        .build_graph(hcx.hcx_mut());
    debug!("Hash graph {graph:?}");
    let (hcx_mut, def_id_hashes) = hcx.hcx_with_def_id_hashes();
    let public_hashes = build_public_hashes(&graph, def_id_hashes, ecx.tcx, hcx_mut);

    let mut hasher = StableHasher::default();
    let public_global_hash = stable_hash(hcx.hcx_mut(), root);
    public_global_hash.stable_hash(hcx.hcx_mut(), &mut hasher);
    public_hashes.stable_hash(hcx.hcx_mut(), &mut hasher);
    let rdr_hashes = public_hashes.value.encode(ecx, public_global_hash);
    let public_hash = Svh::new(hasher.finish());
    debug!("Hashed crate root: {root:#x?}");
    debug!("public api hash: {}", public_hash);
    (CrateHashes { public_hash, private_hash: ecx.tcx.crate_hash(LOCAL_CRATE) }, Some(rdr_hashes))
}

impl HashableCrateRoot {
    pub(super) fn into_crate_root<'h, 'tcx, M: MetadataEncoder<'tcx>>(
        self,
        ecx: &mut EncodeContext<'_, 'tcx, M>,
        hcx: &mut impl PublicApiHashState<'h>,
    ) -> CrateRoot {
        let (hashes, rdr_hashes) = M::crate_hashes(ecx, hcx, &self);
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
            stripped_cfg_items: self.stripped_cfg_items.0,
            diagnostic_items: self.diagnostic_items.value,
            native_libraries: self.native_libraries.value,
            foreign_modules: self.foreign_modules.value,
            traits: self.traits.0,
            impls: self.impls.0,
            incoherent_impls: self.incoherent_impls.value,
            interpret_alloc_index: self.interpret_alloc_index.0,
            proc_macro_data: self.proc_macro_data.value,

            tables: self.tables.0,
            debugger_visualizers: self.debugger_visualizers.value,

            exportable_items: self.exportable_items.0,
            stable_order_of_exportable_impls: self.stable_order_of_exportable_impls.value,
            exported_non_generic_symbols: self.exported_non_generic_symbols.value,
            exported_generic_symbols: self.exported_generic_symbols.value,

            syntax_contexts: self.syntax_contexts.0,
            expn_data: self.expn_data.0,
            expn_hashes: self.expn_hashes.0,

            def_path_hash_map: self.def_path_hash_map.0,

            source_map: self.source_map.0,
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
            public_api_hash_opt_enabled: self.public_api_hash_opt_enabled,
        }
    }
}

pub(crate) enum RecordMode<'tcx> {
    From(Node<'tcx>),
    None,
}

struct ReachabilityGraph<'tcx> {
    nodes: IndexMap<Node<'tcx>, Fingerprint>,
    edges: IndexVec<NodeIdx, Vec<NodeIdx>>,
    roots: Vec<NodeIdx>,
}

impl fmt::Debug for ReachabilityGraph<'_> {
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

impl<'tcx> ReachabilityGraph<'tcx> {
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

pub(super) struct ReachabilityGraphBuilder<'tcx> {
    pub(super) record_mode: Vec<RecordMode<'tcx>>,
    edges: FxHashMap<Node<'tcx>, FxHashSet<Node<'tcx>>>,
    roots: FxHashSet<Node<'tcx>>,
}

impl Default for ReachabilityGraphBuilder<'_> {
    fn default() -> Self {
        Self {
            record_mode: Default::default(),
            edges: Default::default(),
            roots: [Node::DefId(rustc_hir::def_id::CRATE_DEF_ID.into())].into_iter().collect(),
        }
    }
}

fn stable_hash<'a, T: StableHash + ?Sized, const HASH_SPANS_AS_PARENTLESS: bool>(
    hcx: &mut StableHashState<'a, HASH_SPANS_AS_PARENTLESS>,
    val: &T,
) -> Fingerprint {
    let mut hasher = StableHasher::new();
    val.stable_hash(hcx, &mut hasher);
    hasher.finish()
}

impl<'tcx> ReachabilityGraphBuilder<'tcx> {
    fn build_graph(mut self, hcx: &mut StableHashState<'_, true>) -> ReachabilityGraph<'tcx> {
        // ExpnId::root() contains CRATE_DEF_ID as its macro_def_id, but that isn't used for
        // anything.
        // FIXME add newtype that ensures it really isn't used
        self.edges.get_mut(&LocalExpnId::ROOT.into()).unwrap().remove(&CRATE_DEF_ID.into());
        let mut hashes = IndexMap::default();
        // iterating over FxHashSet and FxHashMap is fine here, as it is only used to build the
        // hashes map, which is never returned or iterated
        #[allow(rustc::potential_query_instability)]
        for node in self
            .edges
            .iter()
            .flat_map(|(node, edges)| std::iter::once(*node).chain(edges.iter().copied()))
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
            let node_idx = hashes.get_index_of(&node).unwrap();
            // iterating here is fine, as we stable sort right after
            #[allow(rustc::potential_query_instability)]
            let mut node_edges: Vec<_> =
                node_edges.into_iter().map(|node| hashes.get_index_of(&node).unwrap()).collect();
            node_edges.sort();
            edges[node_idx] = node_edges;
        }

        ReachabilityGraph { roots, edges, nodes: hashes }
    }

    pub(super) fn record(&mut self, to: Node<'tcx>) {
        match self.record_mode.last() {
            Some(RecordMode::From(from)) => {
                self.edges.entry(*from).or_insert(Default::default()).insert(to);
            }
            Some(RecordMode::None) => (),
            None => {
                self.roots.insert(to);
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct ReachabilityGraphHashes {
    local: UnordMap<LocalDefId, Fingerprint>,
    non_local: UnordMap<DefId, Fingerprint>,
    expn: UnordMap<LocalExpnId, Fingerprint>,
    syntax: UnordMap<SyntaxContext, Fingerprint>,
}

impl ReachabilityGraphHashes {
    fn add_node_hash(&mut self, node: &Node<'_>, hash: Fingerprint) {
        let current_hash = match node {
            Node::DefId(id) => {
                if let Some(local) = id.as_local() {
                    self.local.entry(local).or_insert(Fingerprint::ZERO)
                } else {
                    self.non_local.entry(*id).or_insert(Fingerprint::ZERO)
                }
            }
            Node::ExpnId(id) => self.expn.entry(id.expect_local()).or_insert(Fingerprint::ZERO),
            Node::SyntaxContext(id) => self.syntax.entry(*id).or_insert(Fingerprint::ZERO),
            Node::Span(_) => unimplemented!(),
            Node::Ty(_) => unimplemented!(),
            Node::Predicate(_) => unimplemented!(),
            Node::AllocId(_) => unimplemented!(),
        };
        *current_hash = hash.combine_commutative(hash);
    }

    fn get_node(&self, node: &Node<'_>, tcx: TyCtxt<'_>) -> Option<Fingerprint> {
        match node {
            Node::DefId(id) => {
                if let Some(local) = id.as_local() {
                    self.local.get(&local).copied()
                } else {
                    let extern_hash = tcx.extern_def_public_hash(*id);
                    Some(if let Some(local_hash) = self.non_local.get(id) {
                        local_hash.combine_commutative(extern_hash)
                    } else {
                        extern_hash
                    })
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
            Node::AllocId(id) => Some(tcx.with_stable_hashing_context(|mut hcx| {
                stable_hash(&mut hcx, &tcx.global_alloc(*id))
            })),
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Hash, StableHash, Debug)]
pub(crate) enum Node<'tcx> {
    DefId(DefId),
    ExpnId(ExpnId),
    SyntaxContext(SyntaxContext),
    Span(Span),
    Ty(Ty<'tcx>),
    Predicate(PredicateKind<'tcx>),
    AllocId(AllocId),
}

impl From<AllocId> for Node<'_> {
    #[inline(always)]
    fn from(value: AllocId) -> Self {
        Self::AllocId(value)
    }
}
impl From<Span> for Node<'_> {
    #[inline(always)]
    fn from(value: Span) -> Self {
        Self::Span(value)
    }
}
impl<'tcx> From<Ty<'tcx>> for Node<'tcx> {
    #[inline(always)]
    fn from(value: Ty<'tcx>) -> Self {
        Self::Ty(value)
    }
}
impl<'tcx> From<PredicateKind<'tcx>> for Node<'tcx> {
    #[inline(always)]
    fn from(value: PredicateKind<'tcx>) -> Self {
        Self::Predicate(value)
    }
}
impl From<LocalDefId> for Node<'_> {
    #[inline(always)]
    fn from(value: LocalDefId) -> Self {
        Self::DefId(value.into())
    }
}
impl From<DefId> for Node<'_> {
    #[inline(always)]
    fn from(value: DefId) -> Self {
        Self::DefId(value)
    }
}
impl From<LocalExpnId> for Node<'_> {
    #[inline(always)]
    fn from(value: LocalExpnId) -> Self {
        Self::ExpnId(value.to_expn_id())
    }
}
impl From<DefIndex> for Node<'_> {
    #[inline(always)]
    fn from(local_def_index: DefIndex) -> Self {
        LocalDefId { local_def_index }.into()
    }
}
impl From<ExpnId> for Node<'_> {
    #[inline(always)]
    fn from(value: ExpnId) -> Self {
        Self::ExpnId(value)
    }
}
impl From<SyntaxContext> for Node<'_> {
    #[inline(always)]
    fn from(value: SyntaxContext) -> Self {
        Self::SyntaxContext(value)
    }
}

pub(crate) trait TableIndex: Copy + for<'a> Into<Node<'a>> {
    type Encoded: Idx;
    fn into_encoded(self) -> Self::Encoded;
}

impl TableIndex for DefIndex {
    type Encoded = DefIndex;
    fn into_encoded(self) -> Self::Encoded {
        self
    }
}

impl TableIndex for LocalDefId {
    type Encoded = DefIndex;
    fn into_encoded(self) -> Self::Encoded {
        self.local_def_index
    }
}

impl TableIndex for LocalExpnId {
    type Encoded = ExpnIndex;
    fn into_encoded(self) -> Self::Encoded {
        self.as_raw()
    }
}

impl TableIndex for SyntaxContext {
    type Encoded = u32;
    fn into_encoded(self) -> Self::Encoded {
        self.as_u32()
    }
}

impl DirectedGraph for ReachabilityGraph<'_> {
    type Node = NodeIdx;

    fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

impl Successors for ReachabilityGraph<'_> {
    fn successors(&self, node: Self::Node) -> impl Iterator<Item = Self::Node> {
        self.edges[node].iter().copied()
    }
}

type NodeIdx = usize;
type SccIdx = u32;

/// The fingerprint of a single node: its identity hash combined with the hash
/// of the data encoded for it (if any).
fn node_fingerprint(
    graph: &ReachabilityGraph<'_>,
    hashes: &ReachabilityGraphHashes,
    tcx: TyCtxt<'_>,
    node_index: NodeIdx,
) -> Fingerprint {
    let (node_key, base_hash) = graph.nodes.get_index(node_index).unwrap();
    match hashes.get_node(node_key, tcx) {
        Some(encoded_data_hash) => base_hash.combine_commutative(encoded_data_hash),
        None => *base_hash,
    }
}

/// Aggregate the per-node fingerprints into one fingerprint per SCC that
/// captures everything the SCC (transitively) reaches.
///
/// We deliberately do *not* use [`Sccs::new_with_annotation`] for this. That
/// machinery propagates annotations along the DFS as it discovers cycles, which
/// double-counts the cycle's entry node (and re-counts SCCs reached along
/// multiple paths). With an idempotent merge (`min`/`max`/set-union, as in
/// borrowck) that's invisible, but our merge is `Fingerprint::combine_commutative`
/// (128-bit wrapping addition), which is *not* idempotent, so the result would
/// depend on the DFS entry order — and therefore on unreachable nodes, which act
/// as extra DFS roots.
///
/// Instead we compute the fingerprints purely from the (DFS-order-independent)
/// condensation topology:
///   * `base(scc)` = sum of its member nodes' fingerprints, each counted once;
///   * `fingerprint(scc)` = `base(scc)` plus the fingerprint of every successor
///     SCC.
///
/// [`Sccs::all_sccs`] yields the condensation in dependency order (successors
/// before predecessors), so a single pass sees each successor's fingerprint
/// already finished.
fn scc_fingerprints(
    graph: &ReachabilityGraph<'_>,
    hashes: &ReachabilityGraphHashes,
    tcx: TyCtxt<'_>,
    sccs: &Sccs<NodeIdx, SccIdx>,
) -> IndexVec<SccIdx, Fingerprint> {
    let mut fingerprints: IndexVec<SccIdx, Fingerprint> =
        IndexVec::from_elem_n(Fingerprint::ZERO, sccs.num_sccs());
    for node_index in 0..graph.nodes.len() {
        let scc = sccs.scc(node_index);
        fingerprints[scc] =
            fingerprints[scc].combine_commutative(node_fingerprint(graph, hashes, tcx, node_index));
    }
    for scc in sccs.all_sccs() {
        for &succ in sccs.successors(scc) {
            fingerprints[scc] = fingerprints[scc].combine_commutative(fingerprints[succ]);
        }
    }
    fingerprints
}

fn build_public_hashes<'tcx>(
    graph: &ReachabilityGraph<'tcx>,
    hashes: &ReachabilityGraphHashes,
    tcx: TyCtxt<'tcx>,
    hcx: &mut StableHashState<'_, true>,
) -> Hashed<ItemPublicHashesBuilder> {
    let sccs = Sccs::<_, SccIdx>::new(graph);
    let scc_fingerprints = scc_fingerprints(graph, hashes, tcx, &sccs);
    let reachable = graph.reachable_set();

    let mut public_hashes = ItemPublicHashesBuilder::default();
    let mut hasher = StableHasher::new();
    for (node_index, reachable) in reachable.iter_enumerated() {
        if !reachable {
            continue;
        }
        match graph.nodes.get_index(node_index).unwrap().0 {
            Node::DefId(id) => {
                if let Some(local) = id.as_local() {
                    let fingerprint = scc_fingerprints[sccs.scc(node_index)];
                    public_hashes.local.set_some_unhashed(local.local_def_index, fingerprint);
                }
            }
            Node::ExpnId(id) => {
                if let Some(local) = id.as_local() {
                    let fingerprint = scc_fingerprints[sccs.scc(node_index)];
                    public_hashes.expn.set_some_unhashed(local.as_raw(), fingerprint);
                }
            }
            Node::SyntaxContext(_) => (),
            Node::Ty(_) => (),
            Node::Predicate(_) => (),
            Node::Span(_) => (),
            Node::AllocId(_) => (),
        }
    }
    for &node_index in &graph.roots {
        let fingerprint = scc_fingerprints[sccs.scc(node_index)];
        fingerprint.stable_hash(hcx, &mut hasher);
    }
    let roots_hash = hasher.finish::<Fingerprint>();
    debug!("build_public_hashes: roots fingerprint hash = {roots_hash:?}");
    Hashed { value: public_hashes, hash: Some(roots_hash) }
}

#[derive(Default)]
pub(crate) struct ItemPublicHashesBuilder {
    local: TableBuilder<RDRHashNone<DefIndex>, DefIndex, Option<Fingerprint>>,
    expn: TableBuilder<RDRHashNone<ExpnIndex>, ExpnIndex, Option<Fingerprint>>,
}

impl ItemPublicHashesBuilder {
    fn encode<'a, 'tcx, M: MetadataEncoder<'tcx>>(
        &self,
        ecx: &mut EncodeContext<'a, 'tcx, M>,
        public_global_hash: Fingerprint,
    ) -> ItemPublicHashes {
        ItemPublicHashes {
            local: self.local.encode(&mut ecx.opaque).0,
            expn: self.expn.encode(&mut ecx.opaque).0,
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
