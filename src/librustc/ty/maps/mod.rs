// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use dep_graph::{DepConstructor, DepNode};
use errors::DiagnosticBuilder;
use hir::def_id::{CrateNum, DefId, DefIndex};
use hir::def::{Def, Export};
use hir::{self, TraitCandidate, ItemLocalId};
use hir::svh::Svh;
use lint;
use middle::borrowck::BorrowCheckResult;
use middle::const_val;
use middle::cstore::{ExternCrate, LinkagePreference, NativeLibrary,
                     ExternBodyNestedBodies};
use middle::cstore::{NativeLibraryKind, DepKind, CrateSource, ExternConstBody};
use middle::privacy::AccessLevels;
use middle::reachable::ReachableSet;
use middle::region;
use middle::resolve_lifetime::{ResolveLifetimes, Region, ObjectLifetimeDefault};
use middle::stability::{self, DeprecationEntry};
use middle::lang_items::{LanguageItems, LangItem};
use middle::exported_symbols::SymbolExportLevel;
use mir::mono::{CodegenUnit, Stats};
use mir;
use session::{CompileResult, CrateDisambiguator};
use session::config::OutputFilenames;
use traits::Vtable;
use traits::specialization_graph;
use ty::{self, CrateInherentImpls, Ty, TyCtxt};
use ty::steal::Steal;
use ty::subst::Substs;
use util::nodemap::{DefIdSet, DefIdMap, ItemLocalSet};
use util::common::{profq_msg, ErrorReported, ProfileQueriesMsg};

use rustc_data_structures::indexed_set::IdxSetBuf;
use rustc_back::PanicStrategy;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::stable_hasher::StableVec;

use std::ops::Deref;
use std::rc::Rc;
use std::sync::Arc;
use syntax_pos::{Span, DUMMY_SP};
use syntax_pos::symbol::InternedString;
use syntax::attr;
use syntax::ast;
use syntax::symbol::Symbol;

#[macro_use]
mod plumbing;
use self::plumbing::*;
pub use self::plumbing::force_from_dep_node;

mod keys;
pub use self::keys::Key;

mod values;
use self::values::Value;

mod config;
pub use self::config::QueryConfig;
use self::config::QueryDescription;

mod on_disk_cache;
pub use self::on_disk_cache::OnDiskCache;

// Each of these maps also corresponds to a method on a
// `Provider` trait for requesting a value of that type,
// and a method on `Maps` itself for doing that in a
// a way that memoizes and does dep-graph tracking,
// wrapping around the actual chain of providers that
// the driver creates (using several `rustc_*` crates).
define_maps! { <'tcx>
    /// Records the type of every item.
    [] fn type_of: TypeOfItem(DefId) -> Ty<'tcx>,

    /// Maps from the def-id of an item (trait/struct/enum/fn) to its
    /// associated generics and predicates.
    [] fn generics_of: GenericsOfItem(DefId) -> &'tcx ty::Generics,
    [] fn predicates_of: PredicatesOfItem(DefId) -> ty::GenericPredicates<'tcx>,

    /// Maps from the def-id of a trait to the list of
    /// super-predicates. This is a subset of the full list of
    /// predicates. We store these in a separate map because we must
    /// evaluate them even during type conversion, often before the
    /// full predicates are available (note that supertraits have
    /// additional acyclicity requirements).
    [] fn super_predicates_of: SuperPredicatesOfItem(DefId) -> ty::GenericPredicates<'tcx>,

    /// To avoid cycles within the predicates of a single item we compute
    /// per-type-parameter predicates for resolving `T::AssocTy`.
    [] fn type_param_predicates: type_param_predicates((DefId, DefId))
        -> ty::GenericPredicates<'tcx>,

    [] fn trait_def: TraitDefOfItem(DefId) -> &'tcx ty::TraitDef,
    [] fn adt_def: AdtDefOfItem(DefId) -> &'tcx ty::AdtDef,
    [] fn adt_destructor: AdtDestructor(DefId) -> Option<ty::Destructor>,
    [] fn adt_sized_constraint: SizedConstraint(DefId) -> &'tcx [Ty<'tcx>],
    [] fn adt_dtorck_constraint: DtorckConstraint(DefId) -> ty::DtorckConstraint<'tcx>,

    /// True if this is a const fn
    [] fn is_const_fn: IsConstFn(DefId) -> bool,

    /// True if this is a foreign item (i.e., linked via `extern { ... }`).
    [] fn is_foreign_item: IsForeignItem(DefId) -> bool,

    /// Get a map with the variance of every item; use `item_variance`
    /// instead.
    [] fn crate_variances: crate_variances(CrateNum) -> Rc<ty::CrateVariancesMap>,

    /// Maps from def-id of a type or region parameter to its
    /// (inferred) variance.
    [] fn variances_of: ItemVariances(DefId) -> Rc<Vec<ty::Variance>>,

    /// Maps from def-id of a type to its (inferred) outlives.
    [] fn inferred_outlives_of: InferredOutlivesOf(DefId) -> Vec<ty::Predicate<'tcx>>,

    /// Maps from an impl/trait def-id to a list of the def-ids of its items
    [] fn associated_item_def_ids: AssociatedItemDefIds(DefId) -> Rc<Vec<DefId>>,

    /// Maps from a trait item to the trait item "descriptor"
    [] fn associated_item: AssociatedItems(DefId) -> ty::AssociatedItem,

    [] fn impl_trait_ref: ImplTraitRef(DefId) -> Option<ty::TraitRef<'tcx>>,
    [] fn impl_polarity: ImplPolarity(DefId) -> hir::ImplPolarity,

    /// Maps a DefId of a type to a list of its inherent impls.
    /// Contains implementations of methods that are inherent to a type.
    /// Methods in these implementations don't need to be exported.
    [] fn inherent_impls: InherentImpls(DefId) -> Rc<Vec<DefId>>,

    /// Set of all the def-ids in this crate that have MIR associated with
    /// them. This includes all the body owners, but also things like struct
    /// constructors.
    [] fn mir_keys: mir_keys(CrateNum) -> Rc<DefIdSet>,

    /// Maps DefId's that have an associated Mir to the result
    /// of the MIR qualify_consts pass. The actual meaning of
    /// the value isn't known except to the pass itself.
    [] fn mir_const_qualif: MirConstQualif(DefId) -> (u8, Rc<IdxSetBuf<mir::Local>>),

    /// Fetch the MIR for a given def-id right after it's built - this includes
    /// unreachable code.
    [] fn mir_built: MirBuilt(DefId) -> &'tcx Steal<mir::Mir<'tcx>>,

    /// Fetch the MIR for a given def-id up till the point where it is
    /// ready for const evaluation.
    ///
    /// See the README for the `mir` module for details.
    [] fn mir_const: MirConst(DefId) -> &'tcx Steal<mir::Mir<'tcx>>,

    [] fn mir_validated: MirValidated(DefId) -> &'tcx Steal<mir::Mir<'tcx>>,

    /// MIR after our optimization passes have run. This is MIR that is ready
    /// for trans. This is also the only query that can fetch non-local MIR, at present.
    [] fn optimized_mir: MirOptimized(DefId) -> &'tcx mir::Mir<'tcx>,

    /// The result of unsafety-checking this def-id.
    [] fn unsafety_check_result: UnsafetyCheckResult(DefId) -> mir::UnsafetyCheckResult,

    /// HACK: when evaluated, this reports a "unsafe derive on repr(packed)" error
    [] fn unsafe_derive_on_repr_packed: UnsafeDeriveOnReprPacked(DefId) -> (),

    /// The signature of functions and closures.
    [] fn fn_sig: FnSignature(DefId) -> ty::PolyFnSig<'tcx>,

    /// Caches CoerceUnsized kinds for impls on custom types.
    [] fn coerce_unsized_info: CoerceUnsizedInfo(DefId)
        -> ty::adjustment::CoerceUnsizedInfo,

    [] fn typeck_item_bodies: typeck_item_bodies_dep_node(CrateNum) -> CompileResult,

    [] fn typeck_tables_of: TypeckTables(DefId) -> &'tcx ty::TypeckTables<'tcx>,

    [] fn used_trait_imports: UsedTraitImports(DefId) -> Rc<DefIdSet>,

    [] fn has_typeck_tables: HasTypeckTables(DefId) -> bool,

    [] fn coherent_trait: CoherenceCheckTrait(DefId) -> (),

    [] fn borrowck: BorrowCheck(DefId) -> Rc<BorrowCheckResult>,

    /// Borrow checks the function body. If this is a closure, returns
    /// additional requirements that the closure's creator must verify.
    [] fn mir_borrowck: MirBorrowCheck(DefId) -> Option<mir::ClosureRegionRequirements<'tcx>>,

    /// Gets a complete map from all types to their inherent impls.
    /// Not meant to be used directly outside of coherence.
    /// (Defined only for LOCAL_CRATE)
    [] fn crate_inherent_impls: crate_inherent_impls_dep_node(CrateNum) -> CrateInherentImpls,

    /// Checks all types in the krate for overlap in their inherent impls. Reports errors.
    /// Not meant to be used directly outside of coherence.
    /// (Defined only for LOCAL_CRATE)
    [] fn crate_inherent_impls_overlap_check: inherent_impls_overlap_check_dep_node(CrateNum) -> (),

    /// Results of evaluating const items or constants embedded in
    /// other items (such as enum variant explicit discriminants).
    [] fn const_eval: const_eval_dep_node(ty::ParamEnvAnd<'tcx, (DefId, &'tcx Substs<'tcx>)>)
        -> const_val::EvalResult<'tcx>,

    [] fn check_match: CheckMatch(DefId)
        -> Result<(), ErrorReported>,

    /// Performs the privacy check and computes "access levels".
    [] fn privacy_access_levels: PrivacyAccessLevels(CrateNum) -> Rc<AccessLevels>,

    [] fn reachable_set: reachability_dep_node(CrateNum) -> ReachableSet,

    /// Per-body `region::ScopeTree`. The `DefId` should be the owner-def-id for the body;
    /// in the case of closures, this will be redirected to the enclosing function.
    [] fn region_scope_tree: RegionScopeTree(DefId) -> Rc<region::ScopeTree>,

    [] fn mir_shims: mir_shim_dep_node(ty::InstanceDef<'tcx>) -> &'tcx mir::Mir<'tcx>,

    [] fn def_symbol_name: SymbolName(DefId) -> ty::SymbolName,
    [] fn symbol_name: symbol_name_dep_node(ty::Instance<'tcx>) -> ty::SymbolName,

    [] fn describe_def: DescribeDef(DefId) -> Option<Def>,
    [] fn def_span: DefSpan(DefId) -> Span,
    [] fn lookup_stability: LookupStability(DefId) -> Option<&'tcx attr::Stability>,
    [] fn lookup_deprecation_entry: LookupDeprecationEntry(DefId) -> Option<DeprecationEntry>,
    [] fn item_attrs: ItemAttrs(DefId) -> Rc<[ast::Attribute]>,
    [] fn fn_arg_names: FnArgNames(DefId) -> Vec<ast::Name>,
    [] fn impl_parent: ImplParent(DefId) -> Option<DefId>,
    [] fn trait_of_item: TraitOfItem(DefId) -> Option<DefId>,
    [] fn is_exported_symbol: IsExportedSymbol(DefId) -> bool,
    [] fn item_body_nested_bodies: ItemBodyNestedBodies(DefId) -> ExternBodyNestedBodies,
    [] fn const_is_rvalue_promotable_to_static: ConstIsRvaluePromotableToStatic(DefId) -> bool,
    [] fn rvalue_promotable_map: RvaluePromotableMap(DefId) -> Rc<ItemLocalSet>,
    [] fn is_mir_available: IsMirAvailable(DefId) -> bool,
    [] fn vtable_methods: vtable_methods_node(ty::PolyTraitRef<'tcx>)
                          -> Rc<Vec<Option<(DefId, &'tcx Substs<'tcx>)>>>,

    [] fn trans_fulfill_obligation: fulfill_obligation_dep_node(
        (ty::ParamEnv<'tcx>, ty::PolyTraitRef<'tcx>)) -> Vtable<'tcx, ()>,
    [] fn trait_impls_of: TraitImpls(DefId) -> Rc<ty::trait_def::TraitImpls>,
    [] fn specialization_graph_of: SpecializationGraph(DefId) -> Rc<specialization_graph::Graph>,
    [] fn is_object_safe: ObjectSafety(DefId) -> bool,

    // Get the ParameterEnvironment for a given item; this environment
    // will be in "user-facing" mode, meaning that it is suitabe for
    // type-checking etc, and it does not normalize specializable
    // associated types. This is almost always what you want,
    // unless you are doing MIR optimizations, in which case you
    // might want to use `reveal_all()` method to change modes.
    [] fn param_env: ParamEnv(DefId) -> ty::ParamEnv<'tcx>,

    // Trait selection queries. These are best used by invoking `ty.moves_by_default()`,
    // `ty.is_copy()`, etc, since that will prune the environment where possible.
    [] fn is_copy_raw: is_copy_dep_node(ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool,
    [] fn is_sized_raw: is_sized_dep_node(ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool,
    [] fn is_freeze_raw: is_freeze_dep_node(ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool,
    [] fn needs_drop_raw: needs_drop_dep_node(ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool,
    [] fn layout_raw: layout_dep_node(ty::ParamEnvAnd<'tcx, Ty<'tcx>>)
                                  -> Result<&'tcx ty::layout::LayoutDetails,
                                            ty::layout::LayoutError<'tcx>>,

    [] fn dylib_dependency_formats: DylibDepFormats(CrateNum)
                                    -> Rc<Vec<(CrateNum, LinkagePreference)>>,

    [] fn is_panic_runtime: IsPanicRuntime(CrateNum) -> bool,
    [] fn is_compiler_builtins: IsCompilerBuiltins(CrateNum) -> bool,
    [] fn has_global_allocator: HasGlobalAllocator(CrateNum) -> bool,
    [] fn is_sanitizer_runtime: IsSanitizerRuntime(CrateNum) -> bool,
    [] fn is_profiler_runtime: IsProfilerRuntime(CrateNum) -> bool,
    [] fn panic_strategy: GetPanicStrategy(CrateNum) -> PanicStrategy,
    [] fn is_no_builtins: IsNoBuiltins(CrateNum) -> bool,

    [] fn extern_crate: ExternCrate(DefId) -> Rc<Option<ExternCrate>>,

    [] fn specializes: specializes_node((DefId, DefId)) -> bool,
    [] fn in_scope_traits_map: InScopeTraits(DefIndex)
        -> Option<Rc<FxHashMap<ItemLocalId, Rc<StableVec<TraitCandidate>>>>>,
    [] fn module_exports: ModuleExports(DefId) -> Option<Rc<Vec<Export>>>,
    [] fn lint_levels: lint_levels_node(CrateNum) -> Rc<lint::LintLevelMap>,

    [] fn impl_defaultness: ImplDefaultness(DefId) -> hir::Defaultness,
    [] fn exported_symbol_ids: ExportedSymbolIds(CrateNum) -> Rc<DefIdSet>,
    [] fn native_libraries: NativeLibraries(CrateNum) -> Rc<Vec<NativeLibrary>>,
    [] fn plugin_registrar_fn: PluginRegistrarFn(CrateNum) -> Option<DefId>,
    [] fn derive_registrar_fn: DeriveRegistrarFn(CrateNum) -> Option<DefId>,
    [] fn crate_disambiguator: CrateDisambiguator(CrateNum) -> CrateDisambiguator,
    [] fn crate_hash: CrateHash(CrateNum) -> Svh,
    [] fn original_crate_name: OriginalCrateName(CrateNum) -> Symbol,

    [] fn implementations_of_trait: implementations_of_trait_node((CrateNum, DefId))
        -> Rc<Vec<DefId>>,
    [] fn all_trait_implementations: AllTraitImplementations(CrateNum)
        -> Rc<Vec<DefId>>,

    [] fn is_dllimport_foreign_item: IsDllimportForeignItem(DefId) -> bool,
    [] fn is_statically_included_foreign_item: IsStaticallyIncludedForeignItem(DefId) -> bool,
    [] fn native_library_kind: NativeLibraryKind(DefId)
        -> Option<NativeLibraryKind>,
    [] fn link_args: link_args_node(CrateNum) -> Rc<Vec<String>>,

    // Lifetime resolution. See `middle::resolve_lifetimes`.
    [] fn resolve_lifetimes: ResolveLifetimes(CrateNum) -> Rc<ResolveLifetimes>,
    [] fn named_region_map: NamedRegion(DefIndex) ->
        Option<Rc<FxHashMap<ItemLocalId, Region>>>,
    [] fn is_late_bound_map: IsLateBound(DefIndex) ->
        Option<Rc<FxHashSet<ItemLocalId>>>,
    [] fn object_lifetime_defaults_map: ObjectLifetimeDefaults(DefIndex)
        -> Option<Rc<FxHashMap<ItemLocalId, Rc<Vec<ObjectLifetimeDefault>>>>>,

    [] fn visibility: Visibility(DefId) -> ty::Visibility,
    [] fn dep_kind: DepKind(CrateNum) -> DepKind,
    [] fn crate_name: CrateName(CrateNum) -> Symbol,
    [] fn item_children: ItemChildren(DefId) -> Rc<Vec<Export>>,
    [] fn extern_mod_stmt_cnum: ExternModStmtCnum(DefId) -> Option<CrateNum>,

    [] fn get_lang_items: get_lang_items_node(CrateNum) -> Rc<LanguageItems>,
    [] fn defined_lang_items: DefinedLangItems(CrateNum) -> Rc<Vec<(DefId, usize)>>,
    [] fn missing_lang_items: MissingLangItems(CrateNum) -> Rc<Vec<LangItem>>,
    [] fn extern_const_body: ExternConstBody(DefId) -> ExternConstBody<'tcx>,
    [] fn visible_parent_map: visible_parent_map_node(CrateNum)
        -> Rc<DefIdMap<DefId>>,
    [] fn missing_extern_crate_item: MissingExternCrateItem(CrateNum) -> bool,
    [] fn used_crate_source: UsedCrateSource(CrateNum) -> Rc<CrateSource>,
    [] fn postorder_cnums: postorder_cnums_node(CrateNum) -> Rc<Vec<CrateNum>>,

    [] fn freevars: Freevars(DefId) -> Option<Rc<Vec<hir::Freevar>>>,
    [] fn maybe_unused_trait_import: MaybeUnusedTraitImport(DefId) -> bool,
    [] fn maybe_unused_extern_crates: maybe_unused_extern_crates_node(CrateNum)
        -> Rc<Vec<(DefId, Span)>>,

    [] fn stability_index: stability_index_node(CrateNum) -> Rc<stability::Index<'tcx>>,
    [] fn all_crate_nums: all_crate_nums_node(CrateNum) -> Rc<Vec<CrateNum>>,

    [] fn exported_symbols: ExportedSymbols(CrateNum)
        -> Arc<Vec<(String, Option<DefId>, SymbolExportLevel)>>,
    [] fn collect_and_partition_translation_items:
        collect_and_partition_translation_items_node(CrateNum)
        -> (Arc<DefIdSet>, Arc<Vec<Arc<CodegenUnit<'tcx>>>>),
    [] fn export_name: ExportName(DefId) -> Option<Symbol>,
    [] fn contains_extern_indicator: ContainsExternIndicator(DefId) -> bool,
    [] fn is_translated_function: IsTranslatedFunction(DefId) -> bool,
    [] fn codegen_unit: CodegenUnit(InternedString) -> Arc<CodegenUnit<'tcx>>,
    [] fn compile_codegen_unit: CompileCodegenUnit(InternedString) -> Stats,
    [] fn output_filenames: output_filenames_node(CrateNum)
        -> Arc<OutputFilenames>,

    [] fn has_copy_closures: HasCopyClosures(CrateNum) -> bool,
    [] fn has_clone_closures: HasCloneClosures(CrateNum) -> bool,

    // Erases regions from `ty` to yield a new type.
    // Normally you would just use `tcx.erase_regions(&value)`,
    // however, which uses this query as a kind of cache.
    [] fn erase_regions_ty: erase_regions_ty(Ty<'tcx>) -> Ty<'tcx>,
    [] fn fully_normalize_monormophic_ty: normalize_ty_node(Ty<'tcx>) -> Ty<'tcx>,

    [] fn substitute_normalize_and_test_predicates:
        substitute_normalize_and_test_predicates_node((DefId, &'tcx Substs<'tcx>)) -> bool,

    [] fn target_features_whitelist:
        target_features_whitelist_node(CrateNum) -> Rc<FxHashSet<String>>,
    [] fn target_features_enabled: TargetFeaturesEnabled(DefId) -> Rc<Vec<String>>,

}

//////////////////////////////////////////////////////////////////////
// These functions are little shims used to find the dep-node for a
// given query when there is not a *direct* mapping:

fn erase_regions_ty<'tcx>(ty: Ty<'tcx>) -> DepConstructor<'tcx> {
    DepConstructor::EraseRegionsTy { ty }
}

fn type_param_predicates<'tcx>((item_id, param_id): (DefId, DefId)) -> DepConstructor<'tcx> {
    DepConstructor::TypeParamPredicates {
        item_id,
        param_id
    }
}

fn fulfill_obligation_dep_node<'tcx>((param_env, trait_ref):
    (ty::ParamEnv<'tcx>, ty::PolyTraitRef<'tcx>)) -> DepConstructor<'tcx> {
    DepConstructor::FulfillObligation {
        param_env,
        trait_ref
    }
}

fn crate_inherent_impls_dep_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::Coherence
}

fn inherent_impls_overlap_check_dep_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::CoherenceInherentImplOverlapCheck
}

fn reachability_dep_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::Reachability
}

fn mir_shim_dep_node<'tcx>(instance_def: ty::InstanceDef<'tcx>) -> DepConstructor<'tcx> {
    DepConstructor::MirShim {
        instance_def
    }
}

fn symbol_name_dep_node<'tcx>(instance: ty::Instance<'tcx>) -> DepConstructor<'tcx> {
    DepConstructor::InstanceSymbolName { instance }
}

fn typeck_item_bodies_dep_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::TypeckBodiesKrate
}

fn const_eval_dep_node<'tcx>(param_env: ty::ParamEnvAnd<'tcx, (DefId, &'tcx Substs<'tcx>)>)
                             -> DepConstructor<'tcx> {
    DepConstructor::ConstEval { param_env }
}

fn mir_keys<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::MirKeys
}

fn crate_variances<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::CrateVariances
}

fn is_copy_dep_node<'tcx>(param_env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> DepConstructor<'tcx> {
    DepConstructor::IsCopy { param_env }
}

fn is_sized_dep_node<'tcx>(param_env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> DepConstructor<'tcx> {
    DepConstructor::IsSized { param_env }
}

fn is_freeze_dep_node<'tcx>(param_env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> DepConstructor<'tcx> {
    DepConstructor::IsFreeze { param_env }
}

fn needs_drop_dep_node<'tcx>(param_env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> DepConstructor<'tcx> {
    DepConstructor::NeedsDrop { param_env }
}

fn layout_dep_node<'tcx>(param_env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> DepConstructor<'tcx> {
    DepConstructor::Layout { param_env }
}

fn lint_levels_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::LintLevels
}

fn specializes_node<'tcx>((a, b): (DefId, DefId)) -> DepConstructor<'tcx> {
    DepConstructor::Specializes { impl1: a, impl2: b }
}

fn implementations_of_trait_node<'tcx>((krate, trait_id): (CrateNum, DefId))
    -> DepConstructor<'tcx>
{
    DepConstructor::ImplementationsOfTrait { krate, trait_id }
}

fn link_args_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::LinkArgs
}

fn get_lang_items_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::GetLangItems
}

fn visible_parent_map_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::VisibleParentMap
}

fn postorder_cnums_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::PostorderCnums
}

fn maybe_unused_extern_crates_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::MaybeUnusedExternCrates
}

fn stability_index_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::StabilityIndex
}

fn all_crate_nums_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::AllCrateNums
}

fn collect_and_partition_translation_items_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::CollectAndPartitionTranslationItems
}

fn output_filenames_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::OutputFilenames
}

fn vtable_methods_node<'tcx>(trait_ref: ty::PolyTraitRef<'tcx>) -> DepConstructor<'tcx> {
    DepConstructor::VtableMethods{ trait_ref }
}
fn normalize_ty_node<'tcx>(_: Ty<'tcx>) -> DepConstructor<'tcx> {
    DepConstructor::NormalizeTy
}

fn substitute_normalize_and_test_predicates_node<'tcx>(key: (DefId, &'tcx Substs<'tcx>))
                                            -> DepConstructor<'tcx> {
    DepConstructor::SubstituteNormalizeAndTestPredicates { key }
}

fn target_features_whitelist_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::TargetFeaturesWhitelist
}
