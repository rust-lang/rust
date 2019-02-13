use crate::dep_graph::{self, DepConstructor, DepNode};
use crate::hir::def_id::{CrateNum, DefId, DefIndex};
use crate::hir::def::{Def, Export};
use crate::hir::{self, TraitCandidate, ItemLocalId, CodegenFnAttrs};
use crate::infer::canonical::{self, Canonical};
use crate::lint;
use crate::middle::borrowck::BorrowCheckResult;
use crate::middle::cstore::{ExternCrate, LinkagePreference, NativeLibrary, ForeignModule};
use crate::middle::cstore::{NativeLibraryKind, DepKind, CrateSource};
use crate::middle::privacy::AccessLevels;
use crate::middle::reachable::ReachableSet;
use crate::middle::region;
use crate::middle::resolve_lifetime::{ResolveLifetimes, Region, ObjectLifetimeDefault};
use crate::middle::stability::{self, DeprecationEntry};
use crate::middle::lib_features::LibFeatures;
use crate::middle::lang_items::{LanguageItems, LangItem};
use crate::middle::exported_symbols::{SymbolExportLevel, ExportedSymbol};
use crate::mir::interpret::{ConstEvalRawResult, ConstEvalResult};
use crate::mir::mono::CodegenUnit;
use crate::mir;
use crate::mir::interpret::GlobalId;
use crate::session::{CompileResult, CrateDisambiguator};
use crate::session::config::{EntryFnType, OutputFilenames, OptLevel};
use crate::traits::{self, Vtable};
use crate::traits::query::{
    CanonicalPredicateGoal, CanonicalProjectionGoal,
    CanonicalTyGoal, CanonicalTypeOpAscribeUserTypeGoal,
    CanonicalTypeOpEqGoal, CanonicalTypeOpSubtypeGoal, CanonicalTypeOpProvePredicateGoal,
    CanonicalTypeOpNormalizeGoal, NoSolution,
};
use crate::traits::query::method_autoderef::MethodAutoderefStepsResult;
use crate::traits::query::dropck_outlives::{DtorckConstraint, DropckOutlivesResult};
use crate::traits::query::normalize::NormalizationResult;
use crate::traits::query::outlives_bounds::OutlivesBound;
use crate::traits::specialization_graph;
use crate::traits::Clauses;
use crate::ty::{self, CrateInherentImpls, ParamEnvAnd, Ty, TyCtxt};
use crate::ty::steal::Steal;
use crate::ty::subst::Substs;
use crate::util::nodemap::{DefIdSet, DefIdMap, ItemLocalSet};
use crate::util::common::{ErrorReported};
use crate::util::profiling::ProfileCategory::*;
use crate::session::Session;

use errors::DiagnosticBuilder;
use rustc_data_structures::svh::Svh;
use rustc_data_structures::bit_set::BitSet;
use rustc_data_structures::indexed_vec::IndexVec;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::stable_hasher::StableVec;
use rustc_data_structures::sync::Lrc;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_target::spec::PanicStrategy;

use std::borrow::Cow;
use std::ops::Deref;
use std::sync::Arc;
use std::intrinsics::type_name;
use syntax_pos::{Span, DUMMY_SP};
use syntax_pos::symbol::InternedString;
use syntax::attr;
use syntax::ast;
use syntax::feature_gate;
use syntax::symbol::Symbol;

#[macro_use]
mod plumbing;
use self::plumbing::*;
pub use self::plumbing::{force_from_dep_node, CycleError};

mod job;
pub use self::job::{QueryJob, QueryInfo};
#[cfg(parallel_compiler)]
pub use self::job::handle_deadlock;

mod keys;
use self::keys::Key;

mod values;
use self::values::Value;

mod config;
pub use self::config::QueryConfig;
use self::config::{QueryAccessors, QueryDescription};

mod on_disk_cache;
pub use self::on_disk_cache::OnDiskCache;

// Each of these quries corresponds to a function pointer field in the
// `Providers` struct for requesting a value of that type, and a method
// on `tcx: TyCtxt` (and `tcx.at(span)`) for doing that request in a way
// which memoizes and does dep-graph tracking, wrapping around the actual
// `Providers` that the driver creates (using several `rustc_*` crates).
//
// The result type of each query must implement `Clone`, and additionally
// `ty::query::values::Value`, which produces an appropriate placeholder
// (error) value if the query resulted in a query cycle.
// Queries marked with `fatal_cycle` do not need the latter implementation,
// as they will raise an fatal error on query cycles instead.
define_queries! { <'tcx>
    Other {
        /// Records the type of every item.
        [] fn type_of: TypeOfItem(DefId) -> Ty<'tcx>,

        /// Maps from the `DefId` of an item (trait/struct/enum/fn) to its
        /// associated generics.
        [] fn generics_of: GenericsOfItem(DefId) -> &'tcx ty::Generics,

        /// Maps from the `DefId` of an item (trait/struct/enum/fn) to the
        /// predicates (where-clauses) that must be proven true in order
        /// to reference it. This is almost always the "predicates query"
        /// that you want.
        ///
        /// `predicates_of` builds on `predicates_defined_on` -- in fact,
        /// it is almost always the same as that query, except for the
        /// case of traits. For traits, `predicates_of` contains
        /// an additional `Self: Trait<...>` predicate that users don't
        /// actually write. This reflects the fact that to invoke the
        /// trait (e.g., via `Default::default`) you must supply types
        /// that actually implement the trait. (However, this extra
        /// predicate gets in the way of some checks, which are intended
        /// to operate over only the actual where-clauses written by the
        /// user.)
        [] fn predicates_of: PredicatesOfItem(DefId) -> Lrc<ty::GenericPredicates<'tcx>>,

        /// Maps from the `DefId` of an item (trait/struct/enum/fn) to the
        /// predicates (where-clauses) directly defined on it. This is
        /// equal to the `explicit_predicates_of` predicates plus the
        /// `inferred_outlives_of` predicates.
        [] fn predicates_defined_on: PredicatesDefinedOnItem(DefId)
            -> Lrc<ty::GenericPredicates<'tcx>>,

        /// Returns the predicates written explicit by the user.
        [] fn explicit_predicates_of: ExplicitPredicatesOfItem(DefId)
            -> Lrc<ty::GenericPredicates<'tcx>>,

        /// Returns the inferred outlives predicates (e.g., for `struct
        /// Foo<'a, T> { x: &'a T }`, this would return `T: 'a`).
        [] fn inferred_outlives_of: InferredOutlivesOf(DefId) -> Lrc<Vec<ty::Predicate<'tcx>>>,

        /// Maps from the `DefId` of a trait to the list of
        /// super-predicates. This is a subset of the full list of
        /// predicates. We store these in a separate map because we must
        /// evaluate them even during type conversion, often before the
        /// full predicates are available (note that supertraits have
        /// additional acyclicity requirements).
        [] fn super_predicates_of: SuperPredicatesOfItem(DefId) -> Lrc<ty::GenericPredicates<'tcx>>,

        /// To avoid cycles within the predicates of a single item we compute
        /// per-type-parameter predicates for resolving `T::AssocTy`.
        [] fn type_param_predicates: type_param_predicates((DefId, DefId))
            -> Lrc<ty::GenericPredicates<'tcx>>,

        [] fn trait_def: TraitDefOfItem(DefId) -> &'tcx ty::TraitDef,
        [] fn adt_def: AdtDefOfItem(DefId) -> &'tcx ty::AdtDef,
        [] fn adt_destructor: AdtDestructor(DefId) -> Option<ty::Destructor>,
        [] fn adt_sized_constraint: SizedConstraint(DefId) -> &'tcx [Ty<'tcx>],
        [] fn adt_dtorck_constraint: DtorckConstraint(
            DefId
        ) -> Result<DtorckConstraint<'tcx>, NoSolution>,

        /// True if this is a const fn, use the `is_const_fn` to know whether your crate actually
        /// sees it as const fn (e.g., the const-fn-ness might be unstable and you might not have
        /// the feature gate active)
        ///
        /// **Do not call this function manually.** It is only meant to cache the base data for the
        /// `is_const_fn` function.
        [] fn is_const_fn_raw: IsConstFn(DefId) -> bool,


        /// Returns true if calls to the function may be promoted
        ///
        /// This is either because the function is e.g., a tuple-struct or tuple-variant
        /// constructor, or because it has the `#[rustc_promotable]` attribute. The attribute should
        /// be removed in the future in favour of some form of check which figures out whether the
        /// function does not inspect the bits of any of its arguments (so is essentially just a
        /// constructor function).
        [] fn is_promotable_const_fn: IsPromotableConstFn(DefId) -> bool,

        /// True if this is a foreign item (i.e., linked via `extern { ... }`).
        [] fn is_foreign_item: IsForeignItem(DefId) -> bool,

        /// Get a map with the variance of every item; use `item_variance`
        /// instead.
        [] fn crate_variances: crate_variances(CrateNum) -> Lrc<ty::CrateVariancesMap>,

        /// Maps from def-id of a type or region parameter to its
        /// (inferred) variance.
        [] fn variances_of: ItemVariances(DefId) -> Lrc<Vec<ty::Variance>>,
    },

    TypeChecking {
        /// Maps from def-id of a type to its (inferred) outlives.
        [] fn inferred_outlives_crate: InferredOutlivesCrate(CrateNum)
            -> Lrc<ty::CratePredicatesMap<'tcx>>,
    },

    Other {
        /// Maps from an impl/trait def-id to a list of the def-ids of its items
        [] fn associated_item_def_ids: AssociatedItemDefIds(DefId) -> Lrc<Vec<DefId>>,

        /// Maps from a trait item to the trait item "descriptor"
        [] fn associated_item: AssociatedItems(DefId) -> ty::AssociatedItem,

        [] fn impl_trait_ref: ImplTraitRef(DefId) -> Option<ty::TraitRef<'tcx>>,
        [] fn impl_polarity: ImplPolarity(DefId) -> hir::ImplPolarity,

        [] fn issue33140_self_ty: Issue33140SelfTy(DefId) -> Option<ty::Ty<'tcx>>,
    },

    TypeChecking {
        /// Maps a DefId of a type to a list of its inherent impls.
        /// Contains implementations of methods that are inherent to a type.
        /// Methods in these implementations don't need to be exported.
        [] fn inherent_impls: InherentImpls(DefId) -> Lrc<Vec<DefId>>,
    },

    Codegen {
        /// Set of all the `DefId`s in this crate that have MIR associated with
        /// them. This includes all the body owners, but also things like struct
        /// constructors.
        [] fn mir_keys: mir_keys(CrateNum) -> Lrc<DefIdSet>,

        /// Maps DefId's that have an associated Mir to the result
        /// of the MIR qualify_consts pass. The actual meaning of
        /// the value isn't known except to the pass itself.
        [] fn mir_const_qualif: MirConstQualif(DefId) -> (u8, Lrc<BitSet<mir::Local>>),

        /// Fetch the MIR for a given `DefId` right after it's built - this includes
        /// unreachable code.
        [] fn mir_built: MirBuilt(DefId) -> &'tcx Steal<mir::Mir<'tcx>>,

        /// Fetch the MIR for a given `DefId` up till the point where it is
        /// ready for const evaluation.
        ///
        /// See the README for the `mir` module for details.
        [no_hash] fn mir_const: MirConst(DefId) -> &'tcx Steal<mir::Mir<'tcx>>,

        [no_hash] fn mir_validated: MirValidated(DefId) -> &'tcx Steal<mir::Mir<'tcx>>,

        /// MIR after our optimization passes have run. This is MIR that is ready
        /// for codegen. This is also the only query that can fetch non-local MIR, at present.
        [] fn optimized_mir: MirOptimized(DefId) -> &'tcx mir::Mir<'tcx>,
    },

    TypeChecking {
        /// The result of unsafety-checking this `DefId`.
        [] fn unsafety_check_result: UnsafetyCheckResult(DefId) -> mir::UnsafetyCheckResult,

        /// HACK: when evaluated, this reports a "unsafe derive on repr(packed)" error
        [] fn unsafe_derive_on_repr_packed: UnsafeDeriveOnReprPacked(DefId) -> (),

        /// The signature of functions and closures.
        [] fn fn_sig: FnSignature(DefId) -> ty::PolyFnSig<'tcx>,
    },

    Other {
        /// Checks the attributes in the module
        [] fn check_mod_attrs: CheckModAttrs(DefId) -> (),

        [] fn check_mod_unstable_api_usage: CheckModUnstableApiUsage(DefId) -> (),

        /// Checks the loops in the module
        [] fn check_mod_loops: CheckModLoops(DefId) -> (),

        [] fn check_mod_item_types: CheckModItemTypes(DefId) -> (),

        [] fn check_mod_privacy: CheckModPrivacy(DefId) -> (),

        [] fn check_mod_intrinsics: CheckModIntrinsics(DefId) -> (),

        [] fn check_mod_liveness: CheckModLiveness(DefId) -> (),

        [] fn check_mod_impl_wf: CheckModImplWf(DefId) -> (),

        [] fn collect_mod_item_types: CollectModItemTypes(DefId) -> (),

        /// Caches CoerceUnsized kinds for impls on custom types.
        [] fn coerce_unsized_info: CoerceUnsizedInfo(DefId)
            -> ty::adjustment::CoerceUnsizedInfo,
    },

    TypeChecking {
        [] fn typeck_item_bodies: typeck_item_bodies_dep_node(CrateNum) -> CompileResult,

        [] fn typeck_tables_of: TypeckTables(DefId) -> &'tcx ty::TypeckTables<'tcx>,
    },

    Other {
        [] fn used_trait_imports: UsedTraitImports(DefId) -> Lrc<DefIdSet>,
    },

    TypeChecking {
        [] fn has_typeck_tables: HasTypeckTables(DefId) -> bool,

        [] fn coherent_trait: CoherenceCheckTrait(DefId) -> (),
    },

    BorrowChecking {
        [] fn borrowck: BorrowCheck(DefId) -> Lrc<BorrowCheckResult>,

        /// Borrow checks the function body. If this is a closure, returns
        /// additional requirements that the closure's creator must verify.
        [] fn mir_borrowck: MirBorrowCheck(DefId) -> mir::BorrowCheckResult<'tcx>,
    },

    TypeChecking {
        /// Gets a complete map from all types to their inherent impls.
        /// Not meant to be used directly outside of coherence.
        /// (Defined only for `LOCAL_CRATE`.)
        [] fn crate_inherent_impls: crate_inherent_impls_dep_node(CrateNum)
            -> Lrc<CrateInherentImpls>,

        /// Checks all types in the crate for overlap in their inherent impls. Reports errors.
        /// Not meant to be used directly outside of coherence.
        /// (Defined only for `LOCAL_CRATE`.)
        [] fn crate_inherent_impls_overlap_check: inherent_impls_overlap_check_dep_node(CrateNum)
            -> (),
    },

    Other {
        /// Evaluate a constant without running sanity checks
        ///
        /// **Do not use this** outside const eval. Const eval uses this to break query cycles
        /// during validation. Please add a comment to every use site explaining why using
        /// `const_eval` isn't sufficient
        [] fn const_eval_raw: const_eval_raw_dep_node(ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>)
            -> ConstEvalRawResult<'tcx>,

        /// Results of evaluating const items or constants embedded in
        /// other items (such as enum variant explicit discriminants).
        [] fn const_eval: const_eval_dep_node(ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>)
            -> ConstEvalResult<'tcx>,
    },

    TypeChecking {
        [] fn check_match: CheckMatch(DefId)
            -> Result<(), ErrorReported>,

        /// Performs the privacy check and computes "access levels".
        [] fn privacy_access_levels: PrivacyAccessLevels(CrateNum) -> Lrc<AccessLevels>,
    },

    Other {
        [] fn reachable_set: reachability_dep_node(CrateNum) -> ReachableSet,

        /// Per-body `region::ScopeTree`. The `DefId` should be the owner `DefId` for the body;
        /// in the case of closures, this will be redirected to the enclosing function.
        [] fn region_scope_tree: RegionScopeTree(DefId) -> Lrc<region::ScopeTree>,

        [] fn mir_shims: mir_shim_dep_node(ty::InstanceDef<'tcx>) -> &'tcx mir::Mir<'tcx>,

        [] fn def_symbol_name: SymbolName(DefId) -> ty::SymbolName,
        [] fn symbol_name: symbol_name_dep_node(ty::Instance<'tcx>) -> ty::SymbolName,

        [] fn describe_def: DescribeDef(DefId) -> Option<Def>,
        [] fn def_span: DefSpan(DefId) -> Span,
        [] fn lookup_stability: LookupStability(DefId) -> Option<&'tcx attr::Stability>,
        [] fn lookup_deprecation_entry: LookupDeprecationEntry(DefId) -> Option<DeprecationEntry>,
        [] fn item_attrs: ItemAttrs(DefId) -> Lrc<[ast::Attribute]>,
    },

    Codegen {
        [] fn codegen_fn_attrs: codegen_fn_attrs(DefId) -> CodegenFnAttrs,
    },

    Other {
        [] fn fn_arg_names: FnArgNames(DefId) -> Vec<ast::Name>,
        /// Gets the rendered value of the specified constant or associated constant.
        /// Used by rustdoc.
        [] fn rendered_const: RenderedConst(DefId) -> String,
        [] fn impl_parent: ImplParent(DefId) -> Option<DefId>,
    },

    TypeChecking {
        [] fn trait_of_item: TraitOfItem(DefId) -> Option<DefId>,
        [] fn const_is_rvalue_promotable_to_static: ConstIsRvaluePromotableToStatic(DefId) -> bool,
        [] fn rvalue_promotable_map: RvaluePromotableMap(DefId) -> Lrc<ItemLocalSet>,
    },

    Codegen {
        [] fn is_mir_available: IsMirAvailable(DefId) -> bool,
    },

    Other {
        [] fn vtable_methods: vtable_methods_node(ty::PolyTraitRef<'tcx>)
                            -> Lrc<Vec<Option<(DefId, &'tcx Substs<'tcx>)>>>,
    },

    Codegen {
        [] fn codegen_fulfill_obligation: fulfill_obligation_dep_node(
            (ty::ParamEnv<'tcx>, ty::PolyTraitRef<'tcx>)) -> Vtable<'tcx, ()>,
    },

    TypeChecking {
        [] fn trait_impls_of: TraitImpls(DefId) -> Lrc<ty::trait_def::TraitImpls>,
        [] fn specialization_graph_of: SpecializationGraph(DefId)
            -> Lrc<specialization_graph::Graph>,
        [] fn is_object_safe: ObjectSafety(DefId) -> bool,

        /// Gets the ParameterEnvironment for a given item; this environment
        /// will be in "user-facing" mode, meaning that it is suitabe for
        /// type-checking etc, and it does not normalize specializable
        /// associated types. This is almost always what you want,
        /// unless you are doing MIR optimizations, in which case you
        /// might want to use `reveal_all()` method to change modes.
        [] fn param_env: ParamEnv(DefId) -> ty::ParamEnv<'tcx>,

        /// Trait selection queries. These are best used by invoking `ty.is_copy_modulo_regions()`,
        /// `ty.is_copy()`, etc, since that will prune the environment where possible.
        [] fn is_copy_raw: is_copy_dep_node(ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool,
        [] fn is_sized_raw: is_sized_dep_node(ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool,
        [] fn is_freeze_raw: is_freeze_dep_node(ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool,
        [] fn needs_drop_raw: needs_drop_dep_node(ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool,
        [] fn layout_raw: layout_dep_node(ty::ParamEnvAnd<'tcx, Ty<'tcx>>)
                                    -> Result<&'tcx ty::layout::LayoutDetails,
                                                ty::layout::LayoutError<'tcx>>,
    },

    Other {
        [] fn dylib_dependency_formats: DylibDepFormats(CrateNum)
                                        -> Lrc<Vec<(CrateNum, LinkagePreference)>>,
    },

    Codegen {
        [fatal_cycle] fn is_panic_runtime: IsPanicRuntime(CrateNum) -> bool,
        [fatal_cycle] fn is_compiler_builtins: IsCompilerBuiltins(CrateNum) -> bool,
        [fatal_cycle] fn has_global_allocator: HasGlobalAllocator(CrateNum) -> bool,
        [fatal_cycle] fn has_panic_handler: HasPanicHandler(CrateNum) -> bool,
        [fatal_cycle] fn is_sanitizer_runtime: IsSanitizerRuntime(CrateNum) -> bool,
        [fatal_cycle] fn is_profiler_runtime: IsProfilerRuntime(CrateNum) -> bool,
        [fatal_cycle] fn panic_strategy: GetPanicStrategy(CrateNum) -> PanicStrategy,
        [fatal_cycle] fn is_no_builtins: IsNoBuiltins(CrateNum) -> bool,

        [] fn extern_crate: ExternCrate(DefId) -> Lrc<Option<ExternCrate>>,
    },

    TypeChecking {
        [] fn specializes: specializes_node((DefId, DefId)) -> bool,
        [] fn in_scope_traits_map: InScopeTraits(DefIndex)
            -> Option<Lrc<FxHashMap<ItemLocalId, Lrc<StableVec<TraitCandidate>>>>>,
    },

    Other {
        [] fn module_exports: ModuleExports(DefId) -> Option<Lrc<Vec<Export>>>,
        [] fn lint_levels: lint_levels_node(CrateNum) -> Lrc<lint::LintLevelMap>,
    },

    TypeChecking {
        [] fn impl_defaultness: ImplDefaultness(DefId) -> hir::Defaultness,

        [] fn check_item_well_formed: CheckItemWellFormed(DefId) -> (),
        [] fn check_trait_item_well_formed: CheckTraitItemWellFormed(DefId) -> (),
        [] fn check_impl_item_well_formed: CheckImplItemWellFormed(DefId) -> (),
    },

    Linking {
        // The DefIds of all non-generic functions and statics in the given crate
        // that can be reached from outside the crate.
        //
        // We expect this items to be available for being linked to.
        //
        // This query can also be called for LOCAL_CRATE. In this case it will
        // compute which items will be reachable to other crates, taking into account
        // the kind of crate that is currently compiled. Crates with only a
        // C interface have fewer reachable things.
        //
        // Does not include external symbols that don't have a corresponding DefId,
        // like the compiler-generated `main` function and so on.
        [] fn reachable_non_generics: ReachableNonGenerics(CrateNum)
            -> Lrc<DefIdMap<SymbolExportLevel>>,
        [] fn is_reachable_non_generic: IsReachableNonGeneric(DefId) -> bool,
        [] fn is_unreachable_local_definition: IsUnreachableLocalDefinition(DefId) -> bool,
    },

    Codegen {
        [] fn upstream_monomorphizations: UpstreamMonomorphizations(CrateNum)
            -> Lrc<DefIdMap<Lrc<FxHashMap<&'tcx Substs<'tcx>, CrateNum>>>>,
        [] fn upstream_monomorphizations_for: UpstreamMonomorphizationsFor(DefId)
            -> Option<Lrc<FxHashMap<&'tcx Substs<'tcx>, CrateNum>>>,
    },

    Other {
        [] fn native_libraries: NativeLibraries(CrateNum) -> Lrc<Vec<NativeLibrary>>,

        [] fn foreign_modules: ForeignModules(CrateNum) -> Lrc<Vec<ForeignModule>>,

        /// Identifies the entry-point (e.g., the `main` function) for a given
        /// crate, returning `None` if there is no entry point (such as for library crates).
        [] fn entry_fn: EntryFn(CrateNum) -> Option<(DefId, EntryFnType)>,
        [] fn plugin_registrar_fn: PluginRegistrarFn(CrateNum) -> Option<DefId>,
        [] fn proc_macro_decls_static: ProcMacroDeclsStatic(CrateNum) -> Option<DefId>,
        [] fn crate_disambiguator: CrateDisambiguator(CrateNum) -> CrateDisambiguator,
        [] fn crate_hash: CrateHash(CrateNum) -> Svh,
        [] fn original_crate_name: OriginalCrateName(CrateNum) -> Symbol,
        [] fn extra_filename: ExtraFileName(CrateNum) -> String,
    },

    TypeChecking {
        [] fn implementations_of_trait: implementations_of_trait_node((CrateNum, DefId))
            -> Lrc<Vec<DefId>>,
        [] fn all_trait_implementations: AllTraitImplementations(CrateNum)
            -> Lrc<Vec<DefId>>,
    },

    Other {
        [] fn dllimport_foreign_items: DllimportForeignItems(CrateNum)
            -> Lrc<FxHashSet<DefId>>,
        [] fn is_dllimport_foreign_item: IsDllimportForeignItem(DefId) -> bool,
        [] fn is_statically_included_foreign_item: IsStaticallyIncludedForeignItem(DefId) -> bool,
        [] fn native_library_kind: NativeLibraryKind(DefId)
            -> Option<NativeLibraryKind>,
    },

    Linking {
        [] fn link_args: link_args_node(CrateNum) -> Lrc<Vec<String>>,
    },

    BorrowChecking {
        // Lifetime resolution. See `middle::resolve_lifetimes`.
        [] fn resolve_lifetimes: ResolveLifetimes(CrateNum) -> Lrc<ResolveLifetimes>,
        [] fn named_region_map: NamedRegion(DefIndex) ->
            Option<Lrc<FxHashMap<ItemLocalId, Region>>>,
        [] fn is_late_bound_map: IsLateBound(DefIndex) ->
            Option<Lrc<FxHashSet<ItemLocalId>>>,
        [] fn object_lifetime_defaults_map: ObjectLifetimeDefaults(DefIndex)
            -> Option<Lrc<FxHashMap<ItemLocalId, Lrc<Vec<ObjectLifetimeDefault>>>>>,
    },

    TypeChecking {
        [] fn visibility: Visibility(DefId) -> ty::Visibility,
    },

    Other {
        [] fn dep_kind: DepKind(CrateNum) -> DepKind,
        [] fn crate_name: CrateName(CrateNum) -> Symbol,
        [] fn item_children: ItemChildren(DefId) -> Lrc<Vec<Export>>,
        [] fn extern_mod_stmt_cnum: ExternModStmtCnum(DefId) -> Option<CrateNum>,

        [] fn get_lib_features: get_lib_features_node(CrateNum) -> Lrc<LibFeatures>,
        [] fn defined_lib_features: DefinedLibFeatures(CrateNum)
            -> Lrc<Vec<(Symbol, Option<Symbol>)>>,
        [] fn get_lang_items: get_lang_items_node(CrateNum) -> Lrc<LanguageItems>,
        [] fn defined_lang_items: DefinedLangItems(CrateNum) -> Lrc<Vec<(DefId, usize)>>,
        [] fn missing_lang_items: MissingLangItems(CrateNum) -> Lrc<Vec<LangItem>>,
        [] fn visible_parent_map: visible_parent_map_node(CrateNum)
            -> Lrc<DefIdMap<DefId>>,
        [] fn missing_extern_crate_item: MissingExternCrateItem(CrateNum) -> bool,
        [] fn used_crate_source: UsedCrateSource(CrateNum) -> Lrc<CrateSource>,
        [] fn postorder_cnums: postorder_cnums_node(CrateNum) -> Lrc<Vec<CrateNum>>,

        [] fn freevars: Freevars(DefId) -> Option<Lrc<Vec<hir::Freevar>>>,
        [] fn maybe_unused_trait_import: MaybeUnusedTraitImport(DefId) -> bool,
        [] fn maybe_unused_extern_crates: maybe_unused_extern_crates_node(CrateNum)
            -> Lrc<Vec<(DefId, Span)>>,
        [] fn names_imported_by_glob_use: NamesImportedByGlobUse(DefId)
            -> Lrc<FxHashSet<ast::Name>>,

        [] fn stability_index: stability_index_node(CrateNum) -> Lrc<stability::Index<'tcx>>,
        [] fn all_crate_nums: all_crate_nums_node(CrateNum) -> Lrc<Vec<CrateNum>>,

        /// A vector of every trait accessible in the whole crate
        /// (i.e., including those from subcrates). This is used only for
        /// error reporting.
        [] fn all_traits: all_traits_node(CrateNum) -> Lrc<Vec<DefId>>,
    },

    Linking {
        [] fn exported_symbols: ExportedSymbols(CrateNum)
            -> Arc<Vec<(ExportedSymbol<'tcx>, SymbolExportLevel)>>,
    },

    Codegen {
        [] fn collect_and_partition_mono_items:
            collect_and_partition_mono_items_node(CrateNum)
            -> (Arc<DefIdSet>, Arc<Vec<Arc<CodegenUnit<'tcx>>>>),
        [] fn is_codegened_item: IsCodegenedItem(DefId) -> bool,
        [] fn codegen_unit: CodegenUnit(InternedString) -> Arc<CodegenUnit<'tcx>>,
        [] fn backend_optimization_level: BackendOptimizationLevel(CrateNum) -> OptLevel,
    },

    Other {
        [] fn output_filenames: output_filenames_node(CrateNum)
            -> Arc<OutputFilenames>,
    },

    TypeChecking {
        // Erases regions from `ty` to yield a new type.
        // Normally you would just use `tcx.erase_regions(&value)`,
        // however, which uses this query as a kind of cache.
        [] fn erase_regions_ty: erase_regions_ty(Ty<'tcx>) -> Ty<'tcx>,

        /// Do not call this query directly: invoke `normalize` instead.
        [] fn normalize_projection_ty: NormalizeProjectionTy(
            CanonicalProjectionGoal<'tcx>
        ) -> Result<
            Lrc<Canonical<'tcx, canonical::QueryResponse<'tcx, NormalizationResult<'tcx>>>>,
            NoSolution,
        >,

        /// Do not call this query directly: invoke `normalize_erasing_regions` instead.
        [] fn normalize_ty_after_erasing_regions: NormalizeTyAfterErasingRegions(
            ParamEnvAnd<'tcx, Ty<'tcx>>
        ) -> Ty<'tcx>,

        [] fn implied_outlives_bounds: ImpliedOutlivesBounds(
            CanonicalTyGoal<'tcx>
        ) -> Result<
            Lrc<Canonical<'tcx, canonical::QueryResponse<'tcx, Vec<OutlivesBound<'tcx>>>>>,
            NoSolution,
        >,

        /// Do not call this query directly: invoke `infcx.at().dropck_outlives()` instead.
        [] fn dropck_outlives: DropckOutlives(
            CanonicalTyGoal<'tcx>
        ) -> Result<
            Lrc<Canonical<'tcx, canonical::QueryResponse<'tcx, DropckOutlivesResult<'tcx>>>>,
            NoSolution,
        >,

        /// Do not call this query directly: invoke `infcx.predicate_may_hold()` or
        /// `infcx.predicate_must_hold()` instead.
        [] fn evaluate_obligation: EvaluateObligation(
            CanonicalPredicateGoal<'tcx>
        ) -> Result<traits::EvaluationResult, traits::OverflowError>,

        [] fn evaluate_goal: EvaluateGoal(
            traits::ChalkCanonicalGoal<'tcx>
        ) -> Result<
            Lrc<Canonical<'tcx, canonical::QueryResponse<'tcx, ()>>>,
            NoSolution
        >,

        /// Do not call this query directly: part of the `Eq` type-op
        [] fn type_op_ascribe_user_type: TypeOpAscribeUserType(
            CanonicalTypeOpAscribeUserTypeGoal<'tcx>
        ) -> Result<
            Lrc<Canonical<'tcx, canonical::QueryResponse<'tcx, ()>>>,
            NoSolution,
        >,

        /// Do not call this query directly: part of the `Eq` type-op
        [] fn type_op_eq: TypeOpEq(
            CanonicalTypeOpEqGoal<'tcx>
        ) -> Result<
            Lrc<Canonical<'tcx, canonical::QueryResponse<'tcx, ()>>>,
            NoSolution,
        >,

        /// Do not call this query directly: part of the `Subtype` type-op
        [] fn type_op_subtype: TypeOpSubtype(
            CanonicalTypeOpSubtypeGoal<'tcx>
        ) -> Result<
            Lrc<Canonical<'tcx, canonical::QueryResponse<'tcx, ()>>>,
            NoSolution,
        >,

        /// Do not call this query directly: part of the `ProvePredicate` type-op
        [] fn type_op_prove_predicate: TypeOpProvePredicate(
            CanonicalTypeOpProvePredicateGoal<'tcx>
        ) -> Result<
            Lrc<Canonical<'tcx, canonical::QueryResponse<'tcx, ()>>>,
            NoSolution,
        >,

        /// Do not call this query directly: part of the `Normalize` type-op
        [] fn type_op_normalize_ty: TypeOpNormalizeTy(
            CanonicalTypeOpNormalizeGoal<'tcx, Ty<'tcx>>
        ) -> Result<
            Lrc<Canonical<'tcx, canonical::QueryResponse<'tcx, Ty<'tcx>>>>,
            NoSolution,
        >,

        /// Do not call this query directly: part of the `Normalize` type-op
        [] fn type_op_normalize_predicate: TypeOpNormalizePredicate(
            CanonicalTypeOpNormalizeGoal<'tcx, ty::Predicate<'tcx>>
        ) -> Result<
            Lrc<Canonical<'tcx, canonical::QueryResponse<'tcx, ty::Predicate<'tcx>>>>,
            NoSolution,
        >,

        /// Do not call this query directly: part of the `Normalize` type-op
        [] fn type_op_normalize_poly_fn_sig: TypeOpNormalizePolyFnSig(
            CanonicalTypeOpNormalizeGoal<'tcx, ty::PolyFnSig<'tcx>>
        ) -> Result<
            Lrc<Canonical<'tcx, canonical::QueryResponse<'tcx, ty::PolyFnSig<'tcx>>>>,
            NoSolution,
        >,

        /// Do not call this query directly: part of the `Normalize` type-op
        [] fn type_op_normalize_fn_sig: TypeOpNormalizeFnSig(
            CanonicalTypeOpNormalizeGoal<'tcx, ty::FnSig<'tcx>>
        ) -> Result<
            Lrc<Canonical<'tcx, canonical::QueryResponse<'tcx, ty::FnSig<'tcx>>>>,
            NoSolution,
        >,

        [] fn substitute_normalize_and_test_predicates:
            substitute_normalize_and_test_predicates_node((DefId, &'tcx Substs<'tcx>)) -> bool,

        [] fn method_autoderef_steps: MethodAutoderefSteps(
            CanonicalTyGoal<'tcx>
        ) -> MethodAutoderefStepsResult<'tcx>,
    },

    Other {
        [] fn target_features_whitelist:
            target_features_whitelist_node(CrateNum) -> Lrc<FxHashMap<String, Option<String>>>,

        // Get an estimate of the size of an InstanceDef based on its MIR for CGU partitioning.
        [] fn instance_def_size_estimate: instance_def_size_estimate_dep_node(ty::InstanceDef<'tcx>)
            -> usize,

        [] fn features_query: features_node(CrateNum) -> Lrc<feature_gate::Features>,
    },

    TypeChecking {
        [] fn program_clauses_for: ProgramClausesFor(DefId) -> Clauses<'tcx>,

        [] fn program_clauses_for_env: ProgramClausesForEnv(
            traits::Environment<'tcx>
        ) -> Clauses<'tcx>,

        // Get the chalk-style environment of the given item.
        [] fn environment: Environment(DefId) -> traits::Environment<'tcx>,
    },

    Linking {
        [] fn wasm_import_module_map: WasmImportModuleMap(CrateNum)
            -> Lrc<FxHashMap<DefId, String>>,
    },
}

// `try_get_query` can't be public because it uses the private query
// implementation traits, so we provide access to it selectively.
impl<'a, 'tcx, 'lcx> TyCtxt<'a, 'tcx, 'lcx> {
    pub fn try_adt_sized_constraint(
        self,
        span: Span,
        key: DefId,
    ) -> Result<&'tcx [Ty<'tcx>], Box<DiagnosticBuilder<'a>>> {
        self.try_get_query::<queries::adt_sized_constraint<'_>>(span, key)
    }
    pub fn try_needs_drop_raw(
        self,
        span: Span,
        key: ty::ParamEnvAnd<'tcx, Ty<'tcx>>,
    ) -> Result<bool, Box<DiagnosticBuilder<'a>>> {
        self.try_get_query::<queries::needs_drop_raw<'_>>(span, key)
    }
    pub fn try_optimized_mir(
        self,
        span: Span,
        key: DefId,
    ) -> Result<&'tcx mir::Mir<'tcx>, Box<DiagnosticBuilder<'a>>> {
        self.try_get_query::<queries::optimized_mir<'_>>(span, key)
    }
}

//////////////////////////////////////////////////////////////////////
// These functions are little shims used to find the dep-node for a
// given query when there is not a *direct* mapping:


fn features_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::Features
}

fn codegen_fn_attrs<'tcx>(id: DefId) -> DepConstructor<'tcx> {
    DepConstructor::CodegenFnAttrs { 0: id }
}

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

fn const_eval_dep_node<'tcx>(param_env: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>)
                             -> DepConstructor<'tcx> {
    DepConstructor::ConstEval { param_env }
}
fn const_eval_raw_dep_node<'tcx>(param_env: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>)
                             -> DepConstructor<'tcx> {
    DepConstructor::ConstEvalRaw { param_env }
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

fn get_lib_features_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::GetLibFeatures
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

fn all_traits_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::AllTraits
}

fn collect_and_partition_mono_items_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::CollectAndPartitionMonoItems
}

fn output_filenames_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::OutputFilenames
}

fn vtable_methods_node<'tcx>(trait_ref: ty::PolyTraitRef<'tcx>) -> DepConstructor<'tcx> {
    DepConstructor::VtableMethods{ trait_ref }
}

fn substitute_normalize_and_test_predicates_node<'tcx>(key: (DefId, &'tcx Substs<'tcx>))
                                            -> DepConstructor<'tcx> {
    DepConstructor::SubstituteNormalizeAndTestPredicates { key }
}

fn target_features_whitelist_node<'tcx>(_: CrateNum) -> DepConstructor<'tcx> {
    DepConstructor::TargetFeaturesWhitelist
}

fn instance_def_size_estimate_dep_node<'tcx>(instance_def: ty::InstanceDef<'tcx>)
                                              -> DepConstructor<'tcx> {
    DepConstructor::InstanceDefSizeEstimate {
        instance_def
    }
}
