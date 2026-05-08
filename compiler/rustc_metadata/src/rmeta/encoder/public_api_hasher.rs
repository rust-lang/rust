use std::fmt;

use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::stable_hash::{StableHash, StableHashCtxt, StableHasher};
use rustc_data_structures::svh::Svh;
use rustc_hir::LangItem;
use rustc_hir::attrs::StrippedCfgItem;
use rustc_hir::def_id::DefIndex;
use rustc_index::Idx;
use rustc_macros::StableHash;
use rustc_middle::ich::StableHashState;
use rustc_middle::middle::debugger_visualizer::DebuggerVisualizerFile;
use rustc_middle::middle::exported_symbols::{ExportedSymbol, SymbolExportInfo};
use rustc_middle::middle::lib_features::FeatureStability;
use rustc_middle::ty::TyCtxt;
use rustc_session::config::mitigation_coverage::DeniedPartialMitigation;
use rustc_session::config::{SymbolManglingVersion, TargetModifier};
use rustc_session::cstore::{ForeignModule, LinkagePreference, NativeLib};
use rustc_span::Symbol;
use rustc_span::def_id::{LOCAL_CRATE, StableCrateId};
use rustc_span::edition::Edition;
use rustc_target::spec::PanicStrategy;
use tracing::debug;

use super::{CrateDep, CrateHeader, CrateRoot, TargetTuple};
use crate::rmeta::{
    DefPathHashMapRef, EiiMapEncodedKeyValue, ExpnDataTable, ExpnHashTable, IncoherentImpls,
    LazyArray, LazyTable, LazyTables, LazyValue, ProcMacroData, RDRHashes, SyntaxContextTable,
    TraitImpls,
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
    pub(crate) fn digest_iter<'a, I>(&mut self, values: I, hcx: &mut PublicApiHashingContext<'a>)
    where
        I: IntoIterator,
        I::Item: StableHash,
    {
        if hcx.hash_public_api {
            for value in values {
                self.digest(value, hcx);
            }
        }
    }
}

pub(crate) trait TablePublicApiHasher<I: Idx>: Default {
    type IterHasher;

    fn digest<V>(&mut self, index: I, value: V, hcx: &mut PublicApiHashingContext<'_>)
    where
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
    hcx: StableHashState<'a>,
    hash_public_api: bool,
}

impl<'a> PublicApiHashingContext<'a> {
    pub(crate) fn new(hash_public_api: bool, hcx: StableHashState<'a>) -> Self {
        Self { hash_public_api, hcx }
    }
}

impl<I: Idx> TablePublicApiHasher<I> for RDRHashAll<I> {
    type IterHasher = OrderedIterHasher;
    fn digest<V>(&mut self, index: I, value: V, hcx: &mut PublicApiHashingContext<'_>)
    where
        V: StableHash,
    {
        if !hcx.hash_public_api {
            return;
        }
        let mut hasher = StableHasher::default();
        // add the non-stable hash of the index here to hash the order of items without storing them and iterating over it later
        (index.index(), value).stable_hash(&mut hcx.hcx, &mut hasher);
        let hash: Fingerprint = hasher.finish();
        self.hash = self.hash.combine_commutative(hash);
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
    fn digest<V>(&mut self, _index: I, _value: V, _hcx: &mut PublicApiHashingContext<'_>)
    where
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

#[derive(StableHash, Debug)]
pub(crate) struct HashableCrateRoot {
    // FIXME do we need to hash this?
    pub(crate) header: HashableCrateHeader,

    // FIXME do we need to hash this?
    pub(crate) extra_filename: String,
    // FIXME do we need to hash this?
    pub(crate) stable_crate_id: StableCrateId,
    // FIXME do we need to hash this?
    pub(crate) required_panic_strategy: Option<PanicStrategy>,
    // FIXME do we need to hash this?
    pub(crate) panic_in_drop_strategy: PanicStrategy,
    // FIXME do we need to hash this?
    pub(crate) edition: Edition,
    // FIXME do we need to hash this?
    pub(crate) has_global_allocator: bool,
    // FIXME do we need to hash this?
    pub(crate) has_alloc_error_handler: bool,
    // FIXME do we need to hash this?
    pub(crate) has_panic_handler: bool,
    // FIXME do we need to hash this?
    pub(crate) has_default_lib_allocator: bool,
    // FIXME do we need to hash this?
    pub(crate) externally_implementable_items: Hashed<LazyArray<EiiMapEncodedKeyValue>>,

    // FIXME do we need to hash this?
    pub(crate) crate_deps: Hashed<LazyArray<CrateDep>>,
    // FIXME do we need to hash this?
    pub(crate) dylib_dependency_formats: Hashed<LazyArray<Option<LinkagePreference>>>,
    // FIXME do we need to hash this?
    pub(crate) lib_features: Hashed<LazyArray<(Symbol, FeatureStability)>>,
    // FIXME do we need to hash this?
    pub(crate) stability_implications: Hashed<LazyArray<(Symbol, Symbol)>>,
    // FIXME do we need to hash this?
    pub(crate) lang_items: Hashed<LazyArray<(DefIndex, LangItem)>>,
    // FIXME do we need to hash this?
    pub(crate) lang_items_missing: Hashed<LazyArray<LangItem>>,
    // FIXME do we need to hash this?
    pub(crate) stripped_cfg_items: Hashed<LazyArray<StrippedCfgItem<DefIndex>>>,
    // FIXME do we need to hash this?
    pub(crate) diagnostic_items: Hashed<LazyArray<(Symbol, DefIndex)>>,
    // FIXME do we need to hash this?
    pub(crate) native_libraries: Hashed<LazyArray<NativeLib>>,
    // FIXME do we need to hash this?
    pub(crate) foreign_modules: Hashed<LazyArray<ForeignModule>>,
    // FIXME do we need to hash this?
    pub(crate) traits: Hashed<LazyArray<DefIndex>>,
    // FIXME do we need to hash this?
    pub(crate) impls: Hashed<LazyArray<TraitImpls>>,
    // FIXME do we need to hash this?
    pub(crate) incoherent_impls: Hashed<LazyArray<IncoherentImpls>>,
    // FIXME do we need to hash this?
    pub(crate) interpret_alloc_index: Hashed<LazyArray<u64>>,
    // FIXME do we need to hash this?
    pub(crate) proc_macro_data: NoneIfHashed<ProcMacroData>,

    // FIXME do we need to hash this?
    pub(crate) tables: Hashed<LazyTables>,
    // FIXME do we need to hash this?
    pub(crate) debugger_visualizers: Hashed<LazyArray<DebuggerVisualizerFile>>,

    // FIXME do we need to hash this?
    pub(crate) exportable_items: Hashed<LazyArray<DefIndex>>,
    // FIXME do we need to hash this?
    pub(crate) stable_order_of_exportable_impls: Hashed<LazyArray<(DefIndex, usize)>>,
    // FIXME do we need to hash this?
    pub(crate) exported_non_generic_symbols:
        Hashed<LazyArray<(ExportedSymbol<'static>, SymbolExportInfo)>>,
    // FIXME do we need to hash this?
    pub(crate) exported_generic_symbols:
        Hashed<LazyArray<(ExportedSymbol<'static>, SymbolExportInfo)>>,

    // FIXME do we need to hash this?
    pub(crate) syntax_contexts: Hashed<SyntaxContextTable>,
    // FIXME do we need to hash this?
    pub(crate) expn_data: Hashed<ExpnDataTable>,
    // FIXME do we need to hash this?
    pub(crate) expn_hashes: Hashed<ExpnHashTable>,

    // FIXME do we need to hash this?
    pub(crate) def_path_hash_map: Hashed<LazyValue<DefPathHashMapRef<'static>>>,

    // FIXME do we need to hash this?
    pub(crate) source_map: Hashed<LazyTable<u32, Option<LazyValue<rustc_span::SourceFile>>>>,
    // FIXME do we need to hash this?
    pub(crate) target_modifiers: Hashed<LazyArray<TargetModifier>>,
    // FIXME do we need to hash this?
    pub(crate) denied_partial_mitigations: Hashed<LazyArray<DeniedPartialMitigation>>,

    // FIXME do we need to hash this?
    pub(crate) compiler_builtins: bool,
    // FIXME do we need to hash this?
    pub(crate) needs_allocator: bool,
    // FIXME do we need to hash this?
    pub(crate) needs_panic_runtime: bool,
    // FIXME do we need to hash this?
    pub(crate) no_builtins: bool,
    // FIXME do we need to hash this?
    pub(crate) panic_runtime: bool,
    // FIXME do we need to hash this?
    pub(crate) profiler_runtime: bool,
    // FIXME do we need to hash this?
    pub(crate) symbol_mangling_version: SymbolManglingVersion,

    // FIXME do we need to hash this?
    pub(crate) specialization_enabled_in: bool,
}

impl HashableCrateRoot {
    pub(super) fn into_crate_root(
        self,
        tcx: TyCtxt<'_>,
        hcx: &mut PublicApiHashingContext<'_>,
    ) -> CrateRoot {
        let rdr_hashes = if hcx.hash_public_api {
            assert!(!self.header.is_proc_macro_crate);
            let mut hasher = StableHasher::default();
            self.stable_hash(&mut hcx.hcx, &mut hasher);
            let public_api_hash = Svh::new(hasher.finish());
            debug!("Hashed crate root: {self:#x?}");
            debug!("public api hash: {}", public_api_hash);
            RDRHashes { public_api_hash }
        } else {
            let hash = tcx.crate_hash(LOCAL_CRATE);
            RDRHashes { public_api_hash: hash }
        };
        let header = self.header;
        let header = CrateHeader {
            triple: header.triple,
            hash: rdr_hashes.public_api_hash,
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
