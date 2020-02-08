use crate::dep_graph::{DepKind, DepNode, RecoverKey, SerializedDepNodeIndex};
use crate::mir;
use crate::mir::interpret::{GlobalId, LitToConstInput};
use crate::traits;
use crate::traits::query::{
    CanonicalPredicateGoal, CanonicalProjectionGoal, CanonicalTyGoal,
    CanonicalTypeOpAscribeUserTypeGoal, CanonicalTypeOpEqGoal, CanonicalTypeOpNormalizeGoal,
    CanonicalTypeOpProvePredicateGoal, CanonicalTypeOpSubtypeGoal,
};
use crate::ty::query::queries;
use crate::ty::query::QueryDescription;
use crate::ty::subst::SubstsRef;
use crate::ty::{self, ParamEnvAnd, Ty, TyCtxt};
use rustc_hir::def_id::{CrateNum, DefId, DefIndex};

use rustc_span::symbol::Symbol;
use std::borrow::Cow;

fn describe_as_module(def_id: DefId, tcx: TyCtxt<'_>) -> String {
    if def_id.is_top_level_module() {
        format!("top-level module")
    } else {
        format!("module `{}`", tcx.def_path_str(def_id))
    }
}

// Each of these queries corresponds to a function pointer field in the
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
rustc_queries! {
    Other {
        query trigger_delay_span_bug(key: DefId) -> () {
            desc { "trigger a delay span bug" }
        }
    }

    Other {
        // Represents crate as a whole (as distinct from the top-level crate module).
        // If you call `hir_crate` (e.g., indirectly by calling `tcx.hir().krate()`),
        // we will have to assume that any change means that you need to be recompiled.
        // This is because the `hir_crate` query gives you access to all other items.
        // To avoid this fate, do not call `tcx.hir().krate()`; instead,
        // prefer wrappers like `tcx.visit_all_items_in_krate()`.
        query hir_crate(key: CrateNum) -> &'tcx Crate<'tcx> {
            eval_always
            no_hash
            desc { "get the crate HIR" }
        }

        /// Records the type of every item.
        query type_of(key: DefId) -> Ty<'tcx> {
            cache_on_disk_if { key.is_local() }
        }

        /// Maps from the `DefId` of an item (trait/struct/enum/fn) to its
        /// associated generics.
        query generics_of(key: DefId) -> &'tcx ty::Generics {
            cache_on_disk_if { key.is_local() }
            load_cached(tcx, id) {
                let generics: Option<ty::Generics> = tcx.queries.on_disk_cache
                                                        .try_load_query_result(tcx, id);
                generics.map(|x| &*tcx.arena.alloc(x))
            }
        }

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
        query predicates_of(key: DefId) -> ty::GenericPredicates<'tcx> {
            cache_on_disk_if { key.is_local() }
        }

        query native_libraries(_: CrateNum) -> Lrc<Vec<NativeLibrary>> {
            desc { "looking up the native libraries of a linked crate" }
        }

        query lint_levels(_: CrateNum) -> &'tcx LintLevelMap {
            eval_always
            desc { "computing the lint levels for items in this crate" }
        }
    }

    Codegen {
        query is_panic_runtime(_: CrateNum) -> bool {
            fatal_cycle
            desc { "checking if the crate is_panic_runtime" }
        }
    }

    Codegen {
        /// Set of all the `DefId`s in this crate that have MIR associated with
        /// them. This includes all the body owners, but also things like struct
        /// constructors.
        query mir_keys(_: CrateNum) -> &'tcx DefIdSet {
            desc { "getting a list of all mir_keys" }
        }

        /// Maps DefId's that have an associated `mir::Body` to the result
        /// of the MIR const-checking pass. This is the set of qualifs in
        /// the final value of a `const`.
        query mir_const_qualif(key: DefId) -> mir::ConstQualifs {
            desc { |tcx| "const checking `{}`", tcx.def_path_str(key) }
            cache_on_disk_if { key.is_local() }
        }

        /// Fetch the MIR for a given `DefId` right after it's built - this includes
        /// unreachable code.
        query mir_built(_: DefId) -> &'tcx Steal<mir::BodyAndCache<'tcx>> {}

        /// Fetch the MIR for a given `DefId` up till the point where it is
        /// ready for const evaluation.
        ///
        /// See the README for the `mir` module for details.
        query mir_const(_: DefId) -> &'tcx Steal<mir::BodyAndCache<'tcx>> {
            no_hash
        }

        query mir_validated(_: DefId) ->
            (
                &'tcx Steal<mir::BodyAndCache<'tcx>>,
                &'tcx Steal<IndexVec<mir::Promoted, mir::BodyAndCache<'tcx>>>
            ) {
            no_hash
        }

        /// MIR after our optimization passes have run. This is MIR that is ready
        /// for codegen. This is also the only query that can fetch non-local MIR, at present.
        query optimized_mir(key: DefId) -> &'tcx mir::BodyAndCache<'tcx> {
            cache_on_disk_if { key.is_local() }
            load_cached(tcx, id) {
                let mir: Option<crate::mir::BodyAndCache<'tcx>>
                    = tcx.queries.on_disk_cache.try_load_query_result(tcx, id);
                mir.map(|x| {
                    let cache = tcx.arena.alloc(x);
                    cache.ensure_predecessors();
                    &*cache
                })
            }
        }

        query promoted_mir(key: DefId) -> &'tcx IndexVec<mir::Promoted, mir::BodyAndCache<'tcx>> {
            cache_on_disk_if { key.is_local() }
            load_cached(tcx, id) {
                let promoted: Option<
                    rustc_index::vec::IndexVec<
                        crate::mir::Promoted,
                        crate::mir::BodyAndCache<'tcx>
                    >> = tcx.queries.on_disk_cache.try_load_query_result(tcx, id);
                promoted.map(|p| {
                    let cache = tcx.arena.alloc(p);
                    for body in cache.iter_mut() {
                        body.ensure_predecessors();
                    }
                    &*cache
                })
            }
        }
    }

    TypeChecking {
        // Erases regions from `ty` to yield a new type.
        // Normally you would just use `tcx.erase_regions(&value)`,
        // however, which uses this query as a kind of cache.
        query erase_regions_ty(ty: Ty<'tcx>) -> Ty<'tcx> {
            // This query is not expected to have input -- as a result, it
            // is not a good candidates for "replay" because it is essentially a
            // pure function of its input (and hence the expectation is that
            // no caller would be green **apart** from just these
            // queries). Making it anonymous avoids hashing the result, which
            // may save a bit of time.
            anon
            no_force
            desc { "erasing regions from `{:?}`", ty }
        }

        query program_clauses_for(_: DefId) -> Clauses<'tcx> {
            desc { "generating chalk-style clauses" }
        }

        query program_clauses_for_env(_: traits::Environment<'tcx>) -> Clauses<'tcx> {
            no_force
            desc { "generating chalk-style clauses for environment" }
        }

        // Get the chalk-style environment of the given item.
        query environment(_: DefId) -> traits::Environment<'tcx> {
            desc { "return a chalk-style environment" }
        }
    }

    Linking {
        query wasm_import_module_map(_: CrateNum) -> &'tcx FxHashMap<DefId, String> {
            desc { "wasm import module map" }
        }
    }

    Other {
        /// Maps from the `DefId` of an item (trait/struct/enum/fn) to the
        /// predicates (where-clauses) directly defined on it. This is
        /// equal to the `explicit_predicates_of` predicates plus the
        /// `inferred_outlives_of` predicates.
        query predicates_defined_on(_: DefId) -> ty::GenericPredicates<'tcx> {}

        /// Returns the predicates written explicitly by the user.
        query explicit_predicates_of(_: DefId) -> ty::GenericPredicates<'tcx> {}

        /// Returns the inferred outlives predicates (e.g., for `struct
        /// Foo<'a, T> { x: &'a T }`, this would return `T: 'a`).
        query inferred_outlives_of(_: DefId) -> &'tcx [(ty::Predicate<'tcx>, Span)] {}

        /// Maps from the `DefId` of a trait to the list of
        /// super-predicates. This is a subset of the full list of
        /// predicates. We store these in a separate map because we must
        /// evaluate them even during type conversion, often before the
        /// full predicates are available (note that supertraits have
        /// additional acyclicity requirements).
        query super_predicates_of(key: DefId) -> ty::GenericPredicates<'tcx> {
            desc { |tcx| "computing the supertraits of `{}`", tcx.def_path_str(key) }
        }

        /// To avoid cycles within the predicates of a single item we compute
        /// per-type-parameter predicates for resolving `T::AssocTy`.
        query type_param_predicates(key: (DefId, DefId)) -> ty::GenericPredicates<'tcx> {
            no_force
            desc { |tcx| "computing the bounds for type parameter `{}`", {
                let id = tcx.hir().as_local_hir_id(key.1).unwrap();
                tcx.hir().ty_param_name(id)
            }}
        }

        query trait_def(_: DefId) -> &'tcx ty::TraitDef {}
        query adt_def(_: DefId) -> &'tcx ty::AdtDef {}
        query adt_destructor(_: DefId) -> Option<ty::Destructor> {}

        // The cycle error here should be reported as an error by `check_representable`.
        // We consider the type as Sized in the meanwhile to avoid
        // further errors (done in impl Value for AdtSizedConstraint).
        // Use `cycle_delay_bug` to delay the cycle error here to be emitted later
        // in case we accidentally otherwise don't emit an error.
        query adt_sized_constraint(
            _: DefId
        ) -> AdtSizedConstraint<'tcx> {
            cycle_delay_bug
        }

        query adt_dtorck_constraint(
            _: DefId
        ) -> Result<DtorckConstraint<'tcx>, NoSolution> {}

        /// Returns `true` if this is a const fn, use the `is_const_fn` to know whether your crate
        /// actually sees it as const fn (e.g., the const-fn-ness might be unstable and you might
        /// not have the feature gate active).
        ///
        /// **Do not call this function manually.** It is only meant to cache the base data for the
        /// `is_const_fn` function.
        query is_const_fn_raw(key: DefId) -> bool {
            desc { |tcx| "checking if item is const fn: `{}`", tcx.def_path_str(key) }
        }

        query asyncness(key: DefId) -> hir::IsAsync {
            desc { |tcx| "checking if the function is async: `{}`", tcx.def_path_str(key) }
        }

        /// Returns `true` if calls to the function may be promoted.
        ///
        /// This is either because the function is e.g., a tuple-struct or tuple-variant
        /// constructor, or because it has the `#[rustc_promotable]` attribute. The attribute should
        /// be removed in the future in favour of some form of check which figures out whether the
        /// function does not inspect the bits of any of its arguments (so is essentially just a
        /// constructor function).
        query is_promotable_const_fn(_: DefId) -> bool {}

        query const_fn_is_allowed_fn_ptr(_: DefId) -> bool {}

        /// Returns `true` if this is a foreign item (i.e., linked via `extern { ... }`).
        query is_foreign_item(_: DefId) -> bool {}

        /// Returns `Some(mutability)` if the node pointed to by `def_id` is a static item.
        query static_mutability(_: DefId) -> Option<hir::Mutability> {}

        /// Gets a map with the variance of every item; use `item_variance` instead.
        query crate_variances(_: CrateNum) -> &'tcx ty::CrateVariancesMap<'tcx> {
            desc { "computing the variances for items in this crate" }
        }

        /// Maps from the `DefId` of a type or region parameter to its (inferred) variance.
        query variances_of(_: DefId) -> &'tcx [ty::Variance] {}
    }

    TypeChecking {
        /// Maps from thee `DefId` of a type to its (inferred) outlives.
        query inferred_outlives_crate(_: CrateNum)
            -> &'tcx ty::CratePredicatesMap<'tcx> {
            desc { "computing the inferred outlives predicates for items in this crate" }
        }
    }

    Other {
        /// Maps from an impl/trait `DefId to a list of the `DefId`s of its items.
        query associated_item_def_ids(_: DefId) -> &'tcx [DefId] {}

        /// Maps from a trait item to the trait item "descriptor".
        query associated_item(_: DefId) -> ty::AssocItem {}

        /// Collects the associated items defined on a trait or impl.
        query associated_items(key: DefId) -> &'tcx [ty::AssocItem] {
            desc { |tcx| "collecting associated items of {}", tcx.def_path_str(key) }
        }

        query impl_trait_ref(_: DefId) -> Option<ty::TraitRef<'tcx>> {}
        query impl_polarity(_: DefId) -> ty::ImplPolarity {}

        query issue33140_self_ty(_: DefId) -> Option<ty::Ty<'tcx>> {}
    }

    TypeChecking {
        /// Maps a `DefId` of a type to a list of its inherent impls.
        /// Contains implementations of methods that are inherent to a type.
        /// Methods in these implementations don't need to be exported.
        query inherent_impls(_: DefId) -> &'tcx [DefId] {
            eval_always
        }
    }

    TypeChecking {
        /// The result of unsafety-checking this `DefId`.
        query unsafety_check_result(key: DefId) -> mir::UnsafetyCheckResult {
            cache_on_disk_if { key.is_local() }
        }

        /// HACK: when evaluated, this reports a "unsafe derive on repr(packed)" error
        query unsafe_derive_on_repr_packed(_: DefId) -> () {}

        /// The signature of functions and closures.
        query fn_sig(_: DefId) -> ty::PolyFnSig<'tcx> {}
    }

    Other {
        query lint_mod(key: DefId) -> () {
            desc { |tcx| "linting {}", describe_as_module(key, tcx) }
        }

        /// Checks the attributes in the module.
        query check_mod_attrs(key: DefId) -> () {
            desc { |tcx| "checking attributes in {}", describe_as_module(key, tcx) }
        }

        query check_mod_unstable_api_usage(key: DefId) -> () {
            desc { |tcx| "checking for unstable API usage in {}", describe_as_module(key, tcx) }
        }

        /// Checks the const bodies in the module for illegal operations (e.g. `if` or `loop`).
        query check_mod_const_bodies(key: DefId) -> () {
            desc { |tcx| "checking consts in {}", describe_as_module(key, tcx) }
        }

        /// Checks the loops in the module.
        query check_mod_loops(key: DefId) -> () {
            desc { |tcx| "checking loops in {}", describe_as_module(key, tcx) }
        }

        query check_mod_item_types(key: DefId) -> () {
            desc { |tcx| "checking item types in {}", describe_as_module(key, tcx) }
        }

        query check_mod_privacy(key: DefId) -> () {
            desc { |tcx| "checking privacy in {}", describe_as_module(key, tcx) }
        }

        query check_mod_intrinsics(key: DefId) -> () {
            desc { |tcx| "checking intrinsics in {}", describe_as_module(key, tcx) }
        }

        query check_mod_liveness(key: DefId) -> () {
            desc { |tcx| "checking liveness of variables in {}", describe_as_module(key, tcx) }
        }

        query check_mod_impl_wf(key: DefId) -> () {
            desc { |tcx| "checking that impls are well-formed in {}", describe_as_module(key, tcx) }
        }

        query collect_mod_item_types(key: DefId) -> () {
            desc { |tcx| "collecting item types in {}", describe_as_module(key, tcx) }
        }

        /// Caches `CoerceUnsized` kinds for impls on custom types.
        query coerce_unsized_info(_: DefId)
            -> ty::adjustment::CoerceUnsizedInfo {}
    }

    TypeChecking {
        query typeck_item_bodies(_: CrateNum) -> () {
            desc { "type-checking all item bodies" }
        }

        query typeck_tables_of(key: DefId) -> &'tcx ty::TypeckTables<'tcx> {
            cache_on_disk_if { key.is_local() }
            load_cached(tcx, id) {
                let typeck_tables: Option<ty::TypeckTables<'tcx>> = tcx
                    .queries.on_disk_cache
                    .try_load_query_result(tcx, id);

                typeck_tables.map(|tables| &*tcx.arena.alloc(tables))
            }
        }
        query diagnostic_only_typeck_tables_of(key: DefId) -> &'tcx ty::TypeckTables<'tcx> {
            cache_on_disk_if { key.is_local() }
            load_cached(tcx, id) {
                let typeck_tables: Option<ty::TypeckTables<'tcx>> = tcx
                    .queries.on_disk_cache
                    .try_load_query_result(tcx, id);

                typeck_tables.map(|tables| &*tcx.arena.alloc(tables))
            }
        }
    }

    Other {
        query used_trait_imports(key: DefId) -> &'tcx DefIdSet {
            cache_on_disk_if { key.is_local() }
        }
    }

    TypeChecking {
        query has_typeck_tables(_: DefId) -> bool {}

        query coherent_trait(def_id: DefId) -> () {
            desc { |tcx| "coherence checking all impls of trait `{}`", tcx.def_path_str(def_id) }
        }
    }

    BorrowChecking {
        /// Borrow-checks the function body. If this is a closure, returns
        /// additional requirements that the closure's creator must verify.
        query mir_borrowck(key: DefId) -> mir::BorrowCheckResult<'tcx> {
            cache_on_disk_if(tcx, _) { key.is_local() && tcx.is_closure(key) }
        }
    }

    TypeChecking {
        /// Gets a complete map from all types to their inherent impls.
        /// Not meant to be used directly outside of coherence.
        /// (Defined only for `LOCAL_CRATE`.)
        query crate_inherent_impls(k: CrateNum)
            -> &'tcx CrateInherentImpls {
            eval_always
            desc { "all inherent impls defined in crate `{:?}`", k }
        }

        /// Checks all types in the crate for overlap in their inherent impls. Reports errors.
        /// Not meant to be used directly outside of coherence.
        /// (Defined only for `LOCAL_CRATE`.)
        query crate_inherent_impls_overlap_check(_: CrateNum)
            -> () {
            eval_always
            desc { "check for overlap between inherent impls defined in this crate" }
        }
    }

    Other {
        /// Evaluates a constant without running sanity checks.
        ///
        /// **Do not use this** outside const eval. Const eval uses this to break query cycles
        /// during validation. Please add a comment to every use site explaining why using
        /// `const_eval_validated` isn't sufficient. The returned constant also isn't in a suitable
        /// form to be used outside of const eval.
        query const_eval_raw(key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>)
            -> ConstEvalRawResult<'tcx> {
            no_force
            desc { |tcx|
                "const-evaluating `{}`",
                tcx.def_path_str(key.value.instance.def.def_id())
            }
        }

        /// Results of evaluating const items or constants embedded in
        /// other items (such as enum variant explicit discriminants).
        ///
        /// In contrast to `const_eval_raw` this performs some validation on the constant, and
        /// returns a proper constant that is usable by the rest of the compiler.
        ///
        /// **Do not use this** directly, use one of the following wrappers: `tcx.const_eval_poly`,
        /// `tcx.const_eval_resolve`, `tcx.const_eval_instance`, or `tcx.const_eval_promoted`.
        query const_eval_validated(key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>)
            -> ConstEvalResult<'tcx> {
            no_force
            desc { |tcx|
                "const-evaluating + checking `{}`",
                tcx.def_path_str(key.value.instance.def.def_id())
            }
            cache_on_disk_if(_, opt_result) {
                // Only store results without errors
                opt_result.map_or(true, |r| r.is_ok())
            }
        }

        /// Extracts a field of a (variant of a) const.
        query const_field(
            key: ty::ParamEnvAnd<'tcx, (&'tcx ty::Const<'tcx>, mir::Field)>
        ) -> &'tcx ty::Const<'tcx> {
            no_force
            desc { "extract field of const" }
        }

        /// Destructure a constant ADT or array into its variant indent and its
        /// field values.
        query destructure_const(
            key: ty::ParamEnvAnd<'tcx, &'tcx ty::Const<'tcx>>
        ) -> mir::DestructuredConst<'tcx> {
            no_force
            desc { "destructure constant" }
        }

        query const_caller_location(key: (rustc_span::Symbol, u32, u32)) -> &'tcx ty::Const<'tcx> {
            no_force
            desc { "get a &core::panic::Location referring to a span" }
        }

        query lit_to_const(
            key: LitToConstInput<'tcx>
        ) -> Result<&'tcx ty::Const<'tcx>, LitToConstError> {
            no_force
            desc { "converting literal to const" }
        }
    }

    TypeChecking {
        query check_match(key: DefId) {
            cache_on_disk_if { key.is_local() }
        }

        /// Performs part of the privacy check and computes "access levels".
        query privacy_access_levels(_: CrateNum) -> &'tcx AccessLevels {
            eval_always
            desc { "privacy access levels" }
        }
        query check_private_in_public(_: CrateNum) -> () {
            eval_always
            desc { "checking for private elements in public interfaces" }
        }
    }

    Other {
        query reachable_set(_: CrateNum) -> Lrc<HirIdSet> {
            desc { "reachability" }
        }

        /// Per-body `region::ScopeTree`. The `DefId` should be the owner `DefId` for the body;
        /// in the case of closures, this will be redirected to the enclosing function.
        query region_scope_tree(_: DefId) -> &'tcx region::ScopeTree {}

        query mir_shims(key: ty::InstanceDef<'tcx>) -> &'tcx mir::BodyAndCache<'tcx> {
            no_force
            desc { |tcx| "generating MIR shim for `{}`", tcx.def_path_str(key.def_id()) }
        }

        /// The `symbol_name` query provides the symbol name for calling a
        /// given instance from the local crate. In particular, it will also
        /// look up the correct symbol name of instances from upstream crates.
        query symbol_name(key: ty::Instance<'tcx>) -> ty::SymbolName {
            no_force
            desc { "computing the symbol for `{}`", key }
            cache_on_disk_if { true }
        }

        query def_kind(_: DefId) -> Option<DefKind> {}
        query def_span(_: DefId) -> Span {
            // FIXME(mw): DefSpans are not really inputs since they are derived from
            // HIR. But at the moment HIR hashing still contains some hacks that allow
            // to make type debuginfo to be source location independent. Declaring
            // DefSpan an input makes sure that changes to these are always detected
            // regardless of HIR hashing.
            eval_always
        }
        query lookup_stability(_: DefId) -> Option<&'tcx attr::Stability> {}
        query lookup_const_stability(_: DefId) -> Option<&'tcx attr::ConstStability> {}
        query lookup_deprecation_entry(_: DefId) -> Option<DeprecationEntry> {}
        query item_attrs(_: DefId) -> Lrc<[ast::Attribute]> {}
    }

    Codegen {
        query codegen_fn_attrs(_: DefId) -> CodegenFnAttrs {
            cache_on_disk_if { true }
        }
    }

    Other {
        query fn_arg_names(_: DefId) -> Vec<ast::Name> {}
        /// Gets the rendered value of the specified constant or associated constant.
        /// Used by rustdoc.
        query rendered_const(_: DefId) -> String {}
        query impl_parent(_: DefId) -> Option<DefId> {}
    }

    TypeChecking {
        query trait_of_item(_: DefId) -> Option<DefId> {}
    }

    Codegen {
        query is_mir_available(key: DefId) -> bool {
            desc { |tcx| "checking if item has mir available: `{}`", tcx.def_path_str(key) }
        }
    }

    Other {
        query vtable_methods(key: ty::PolyTraitRef<'tcx>)
                            -> &'tcx [Option<(DefId, SubstsRef<'tcx>)>] {
            no_force
            desc { |tcx| "finding all methods for trait {}", tcx.def_path_str(key.def_id()) }
        }
    }

    Codegen {
        query codegen_fulfill_obligation(
            key: (ty::ParamEnv<'tcx>, ty::PolyTraitRef<'tcx>)
        ) -> Vtable<'tcx, ()> {
            no_force
            cache_on_disk_if { true }
            desc { |tcx|
                "checking if `{}` fulfills its obligations",
                tcx.def_path_str(key.1.def_id())
            }
        }
    }

    TypeChecking {
        query trait_impls_of(key: DefId) -> &'tcx ty::trait_def::TraitImpls {
            desc { |tcx| "trait impls of `{}`", tcx.def_path_str(key) }
        }
        query specialization_graph_of(_: DefId) -> &'tcx specialization_graph::Graph {
            cache_on_disk_if { true }
        }
        query is_object_safe(key: DefId) -> bool {
            desc { |tcx| "determine object safety of trait `{}`", tcx.def_path_str(key) }
        }

        /// Gets the ParameterEnvironment for a given item; this environment
        /// will be in "user-facing" mode, meaning that it is suitabe for
        /// type-checking etc, and it does not normalize specializable
        /// associated types. This is almost always what you want,
        /// unless you are doing MIR optimizations, in which case you
        /// might want to use `reveal_all()` method to change modes.
        query param_env(_: DefId) -> ty::ParamEnv<'tcx> {}

        /// Trait selection queries. These are best used by invoking `ty.is_copy_modulo_regions()`,
        /// `ty.is_copy()`, etc, since that will prune the environment where possible.
        query is_copy_raw(env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool {
            no_force
            desc { "computing whether `{}` is `Copy`", env.value }
        }
        query is_sized_raw(env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool {
            no_force
            desc { "computing whether `{}` is `Sized`", env.value }
        }
        query is_freeze_raw(env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool {
            no_force
            desc { "computing whether `{}` is freeze", env.value }
        }

        // The cycle error here should be reported as an error by `check_representable`.
        // We consider the type as not needing drop in the meanwhile to avoid
        // further errors (done in impl Value for NeedsDrop).
        // Use `cycle_delay_bug` to delay the cycle error here to be emitted later
        // in case we accidentally otherwise don't emit an error.
        query needs_drop_raw(env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> NeedsDrop {
            cycle_delay_bug
            no_force
            desc { "computing whether `{}` needs drop", env.value }
        }

        query layout_raw(
            env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>
        ) -> Result<&'tcx ty::layout::LayoutDetails, ty::layout::LayoutError<'tcx>> {
            no_force
            desc { "computing layout of `{}`", env.value }
        }
    }

    Other {
        query dylib_dependency_formats(_: CrateNum)
                                        -> &'tcx [(CrateNum, LinkagePreference)] {
            desc { "dylib dependency formats of crate" }
        }

        query dependency_formats(_: CrateNum)
            -> Lrc<crate::middle::dependency_format::Dependencies>
        {
            desc { "get the linkage format of all dependencies" }
        }
    }

    Codegen {
        query is_compiler_builtins(_: CrateNum) -> bool {
            fatal_cycle
            desc { "checking if the crate is_compiler_builtins" }
        }
        query has_global_allocator(_: CrateNum) -> bool {
            fatal_cycle
            desc { "checking if the crate has_global_allocator" }
        }
        query has_panic_handler(_: CrateNum) -> bool {
            fatal_cycle
            desc { "checking if the crate has_panic_handler" }
        }
        query is_profiler_runtime(_: CrateNum) -> bool {
            fatal_cycle
            desc { "query a crate is `#![profiler_runtime]`" }
        }
        query panic_strategy(_: CrateNum) -> PanicStrategy {
            fatal_cycle
            desc { "query a crate's configured panic strategy" }
        }
        query is_no_builtins(_: CrateNum) -> bool {
            fatal_cycle
            desc { "test whether a crate has `#![no_builtins]`" }
        }
        query symbol_mangling_version(_: CrateNum) -> SymbolManglingVersion {
            fatal_cycle
            desc { "query a crate's symbol mangling version" }
        }

        query extern_crate(_: DefId) -> Option<&'tcx ExternCrate> {
            eval_always
            desc { "getting crate's ExternCrateData" }
        }
    }

    TypeChecking {
        query specializes(_: (DefId, DefId)) -> bool {
            no_force
            desc { "computing whether impls specialize one another" }
        }
        query in_scope_traits_map(_: DefIndex)
            -> Option<&'tcx FxHashMap<ItemLocalId, StableVec<TraitCandidate>>> {
            eval_always
            desc { "traits in scope at a block" }
        }
    }

    Other {
        query module_exports(_: DefId) -> Option<&'tcx [Export<hir::HirId>]> {
            eval_always
        }
    }

    TypeChecking {
        query impl_defaultness(_: DefId) -> hir::Defaultness {}

        query check_item_well_formed(_: DefId) -> () {}
        query check_trait_item_well_formed(_: DefId) -> () {}
        query check_impl_item_well_formed(_: DefId) -> () {}
    }

    Linking {
        // The `DefId`s of all non-generic functions and statics in the given crate
        // that can be reached from outside the crate.
        //
        // We expect this items to be available for being linked to.
        //
        // This query can also be called for `LOCAL_CRATE`. In this case it will
        // compute which items will be reachable to other crates, taking into account
        // the kind of crate that is currently compiled. Crates with only a
        // C interface have fewer reachable things.
        //
        // Does not include external symbols that don't have a corresponding DefId,
        // like the compiler-generated `main` function and so on.
        query reachable_non_generics(_: CrateNum)
            -> &'tcx DefIdMap<SymbolExportLevel> {
            desc { "looking up the exported symbols of a crate" }
        }
        query is_reachable_non_generic(_: DefId) -> bool {}
        query is_unreachable_local_definition(_: DefId) -> bool {}
    }

    Codegen {
        /// The entire set of monomorphizations the local crate can safely link
        /// to because they are exported from upstream crates. Do not depend on
        /// this directly, as its value changes anytime a monomorphization gets
        /// added or removed in any upstream crate. Instead use the narrower
        /// `upstream_monomorphizations_for`, `upstream_drop_glue_for`, or, even
        /// better, `Instance::upstream_monomorphization()`.
        query upstream_monomorphizations(
            k: CrateNum
        ) -> &'tcx DefIdMap<FxHashMap<SubstsRef<'tcx>, CrateNum>> {
            desc { "collecting available upstream monomorphizations `{:?}`", k }
        }

        /// Returns the set of upstream monomorphizations available for the
        /// generic function identified by the given `def_id`. The query makes
        /// sure to make a stable selection if the same monomorphization is
        /// available in multiple upstream crates.
        ///
        /// You likely want to call `Instance::upstream_monomorphization()`
        /// instead of invoking this query directly.
        query upstream_monomorphizations_for(_: DefId)
            -> Option<&'tcx FxHashMap<SubstsRef<'tcx>, CrateNum>> {}

        /// Returns the upstream crate that exports drop-glue for the given
        /// type (`substs` is expected to be a single-item list containing the
        /// type one wants drop-glue for).
        ///
        /// This is a subset of `upstream_monomorphizations_for` in order to
        /// increase dep-tracking granularity. Otherwise adding or removing any
        /// type with drop-glue in any upstream crate would invalidate all
        /// functions calling drop-glue of an upstream type.
        ///
        /// You likely want to call `Instance::upstream_monomorphization()`
        /// instead of invoking this query directly.
        ///
        /// NOTE: This query could easily be extended to also support other
        ///       common functions that have are large set of monomorphizations
        ///       (like `Clone::clone` for example).
        query upstream_drop_glue_for(substs: SubstsRef<'tcx>) -> Option<CrateNum> {
            desc { "available upstream drop-glue for `{:?}`", substs }
            no_force
        }
    }

    Other {
        query foreign_modules(_: CrateNum) -> &'tcx [ForeignModule] {
            desc { "looking up the foreign modules of a linked crate" }
        }

        /// Identifies the entry-point (e.g., the `main` function) for a given
        /// crate, returning `None` if there is no entry point (such as for library crates).
        query entry_fn(_: CrateNum) -> Option<(DefId, EntryFnType)> {
            desc { "looking up the entry function of a crate" }
        }
        query plugin_registrar_fn(_: CrateNum) -> Option<DefId> {
            desc { "looking up the plugin registrar for a crate" }
        }
        query proc_macro_decls_static(_: CrateNum) -> Option<DefId> {
            desc { "looking up the derive registrar for a crate" }
        }
        query crate_disambiguator(_: CrateNum) -> CrateDisambiguator {
            eval_always
            desc { "looking up the disambiguator a crate" }
        }
        query crate_hash(_: CrateNum) -> Svh {
            eval_always
            desc { "looking up the hash a crate" }
        }
        query crate_host_hash(_: CrateNum) -> Option<Svh> {
            eval_always
            desc { "looking up the hash of a host version of a crate" }
        }
        query original_crate_name(_: CrateNum) -> Symbol {
            eval_always
            desc { "looking up the original name a crate" }
        }
        query extra_filename(_: CrateNum) -> String {
            eval_always
            desc { "looking up the extra filename for a crate" }
        }
    }

    TypeChecking {
        query implementations_of_trait(_: (CrateNum, DefId))
            -> &'tcx [DefId] {
            no_force
            desc { "looking up implementations of a trait in a crate" }
        }
        query all_trait_implementations(_: CrateNum)
            -> &'tcx [DefId] {
            desc { "looking up all (?) trait implementations" }
        }
    }

    Other {
        query dllimport_foreign_items(_: CrateNum)
            -> &'tcx FxHashSet<DefId> {
            desc { "dllimport_foreign_items" }
        }
        query is_dllimport_foreign_item(_: DefId) -> bool {}
        query is_statically_included_foreign_item(_: DefId) -> bool {}
        query native_library_kind(_: DefId)
            -> Option<NativeLibraryKind> {}
    }

    Linking {
        query link_args(_: CrateNum) -> Lrc<Vec<String>> {
            eval_always
            desc { "looking up link arguments for a crate" }
        }
    }

    BorrowChecking {
        /// Lifetime resolution. See `middle::resolve_lifetimes`.
        query resolve_lifetimes(_: CrateNum) -> &'tcx ResolveLifetimes {
            desc { "resolving lifetimes" }
        }
        query named_region_map(_: DefIndex) ->
            Option<&'tcx FxHashMap<ItemLocalId, Region>> {
            desc { "looking up a named region" }
        }
        query is_late_bound_map(_: DefIndex) ->
            Option<&'tcx FxHashSet<ItemLocalId>> {
            desc { "testing if a region is late bound" }
        }
        query object_lifetime_defaults_map(_: DefIndex)
            -> Option<&'tcx FxHashMap<ItemLocalId, Vec<ObjectLifetimeDefault>>> {
            desc { "looking up lifetime defaults for a region" }
        }
    }

    TypeChecking {
        query visibility(_: DefId) -> ty::Visibility {}
    }

    Other {
        query dep_kind(_: CrateNum) -> DepKind {
            eval_always
            desc { "fetching what a dependency looks like" }
        }
        query crate_name(_: CrateNum) -> Symbol {
            eval_always
            desc { "fetching what a crate is named" }
        }
        query item_children(_: DefId) -> &'tcx [Export<hir::HirId>] {}
        query extern_mod_stmt_cnum(_: DefId) -> Option<CrateNum> {}

        query get_lib_features(_: CrateNum) -> &'tcx LibFeatures {
            eval_always
            desc { "calculating the lib features map" }
        }
        query defined_lib_features(_: CrateNum)
            -> &'tcx [(Symbol, Option<Symbol>)] {
            desc { "calculating the lib features defined in a crate" }
        }
        /// Returns the lang items defined in another crate by loading it from metadata.
        // FIXME: It is illegal to pass a `CrateNum` other than `LOCAL_CRATE` here, just get rid
        // of that argument?
        query get_lang_items(_: CrateNum) -> &'tcx LanguageItems {
            eval_always
            desc { "calculating the lang items map" }
        }

        /// Returns all diagnostic items defined in all crates.
        query all_diagnostic_items(_: CrateNum) -> &'tcx FxHashMap<Symbol, DefId> {
            eval_always
            desc { "calculating the diagnostic items map" }
        }

        /// Returns the lang items defined in another crate by loading it from metadata.
        query defined_lang_items(_: CrateNum) -> &'tcx [(DefId, usize)] {
            desc { "calculating the lang items defined in a crate" }
        }

        /// Returns the diagnostic items defined in a crate.
        query diagnostic_items(_: CrateNum) -> &'tcx FxHashMap<Symbol, DefId> {
            desc { "calculating the diagnostic items map in a crate" }
        }

        query missing_lang_items(_: CrateNum) -> &'tcx [LangItem] {
            desc { "calculating the missing lang items in a crate" }
        }
        query visible_parent_map(_: CrateNum)
            -> &'tcx DefIdMap<DefId> {
            desc { "calculating the visible parent map" }
        }
        query missing_extern_crate_item(_: CrateNum) -> bool {
            eval_always
            desc { "seeing if we're missing an `extern crate` item for this crate" }
        }
        query used_crate_source(_: CrateNum) -> Lrc<CrateSource> {
            eval_always
            desc { "looking at the source for a crate" }
        }
        query postorder_cnums(_: CrateNum) -> &'tcx [CrateNum] {
            eval_always
            desc { "generating a postorder list of CrateNums" }
        }

        query upvars(_: DefId) -> Option<&'tcx FxIndexMap<hir::HirId, hir::Upvar>> {
            eval_always
        }
        query maybe_unused_trait_import(_: DefId) -> bool {
            eval_always
        }
        query maybe_unused_extern_crates(_: CrateNum)
            -> &'tcx [(DefId, Span)] {
            eval_always
            desc { "looking up all possibly unused extern crates" }
        }
        query names_imported_by_glob_use(_: DefId)
            -> Lrc<FxHashSet<ast::Name>> {
            eval_always
        }

        query stability_index(_: CrateNum) -> &'tcx stability::Index<'tcx> {
            eval_always
            desc { "calculating the stability index for the local crate" }
        }
        query all_crate_nums(_: CrateNum) -> &'tcx [CrateNum] {
            eval_always
            desc { "fetching all foreign CrateNum instances" }
        }

        /// A vector of every trait accessible in the whole crate
        /// (i.e., including those from subcrates). This is used only for
        /// error reporting.
        query all_traits(_: CrateNum) -> &'tcx [DefId] {
            desc { "fetching all foreign and local traits" }
        }
    }

    Linking {
        /// The list of symbols exported from the given crate.
        ///
        /// - All names contained in `exported_symbols(cnum)` are guaranteed to
        ///   correspond to a publicly visible symbol in `cnum` machine code.
        /// - The `exported_symbols` sets of different crates do not intersect.
        query exported_symbols(_: CrateNum)
            -> Arc<Vec<(ExportedSymbol<'tcx>, SymbolExportLevel)>> {
            desc { "exported_symbols" }
        }
    }

    Codegen {
        query collect_and_partition_mono_items(_: CrateNum)
            -> (Arc<DefIdSet>, Arc<Vec<Arc<CodegenUnit<'tcx>>>>) {
            eval_always
            desc { "collect_and_partition_mono_items" }
        }
        query is_codegened_item(_: DefId) -> bool {}
        query codegen_unit(_: Symbol) -> Arc<CodegenUnit<'tcx>> {
            no_force
            desc { "codegen_unit" }
        }
        query backend_optimization_level(_: CrateNum) -> OptLevel {
            desc { "optimization level used by backend" }
        }
    }

    Other {
        query output_filenames(_: CrateNum) -> Arc<OutputFilenames> {
            eval_always
            desc { "output_filenames" }
        }
    }

    TypeChecking {
        /// Do not call this query directly: invoke `normalize` instead.
        query normalize_projection_ty(
            goal: CanonicalProjectionGoal<'tcx>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, NormalizationResult<'tcx>>>,
            NoSolution,
        > {
            no_force
            desc { "normalizing `{:?}`", goal }
        }

        /// Do not call this query directly: invoke `normalize_erasing_regions` instead.
        query normalize_ty_after_erasing_regions(
            goal: ParamEnvAnd<'tcx, Ty<'tcx>>
        ) -> Ty<'tcx> {
            no_force
            desc { "normalizing `{:?}`", goal }
        }

        query implied_outlives_bounds(
            goal: CanonicalTyGoal<'tcx>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, Vec<OutlivesBound<'tcx>>>>,
            NoSolution,
        > {
            no_force
            desc { "computing implied outlives bounds for `{:?}`", goal }
        }

        /// Do not call this query directly: invoke `infcx.at().dropck_outlives()` instead.
        query dropck_outlives(
            goal: CanonicalTyGoal<'tcx>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, DropckOutlivesResult<'tcx>>>,
            NoSolution,
        > {
            no_force
            desc { "computing dropck types for `{:?}`", goal }
        }

        /// Do not call this query directly: invoke `infcx.predicate_may_hold()` or
        /// `infcx.predicate_must_hold()` instead.
        query evaluate_obligation(
            goal: CanonicalPredicateGoal<'tcx>
        ) -> Result<traits::EvaluationResult, traits::OverflowError> {
            no_force
            desc { "evaluating trait selection obligation `{}`", goal.value.value }
        }

        query evaluate_goal(
            goal: traits::ChalkCanonicalGoal<'tcx>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, ()>>,
            NoSolution
        > {
            no_force
            desc { "evaluating trait selection obligation `{}`", goal.value.goal }
        }

        /// Do not call this query directly: part of the `Eq` type-op
        query type_op_ascribe_user_type(
            goal: CanonicalTypeOpAscribeUserTypeGoal<'tcx>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, ()>>,
            NoSolution,
        > {
            no_force
            desc { "evaluating `type_op_ascribe_user_type` `{:?}`", goal }
        }

        /// Do not call this query directly: part of the `Eq` type-op
        query type_op_eq(
            goal: CanonicalTypeOpEqGoal<'tcx>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, ()>>,
            NoSolution,
        > {
            no_force
            desc { "evaluating `type_op_eq` `{:?}`", goal }
        }

        /// Do not call this query directly: part of the `Subtype` type-op
        query type_op_subtype(
            goal: CanonicalTypeOpSubtypeGoal<'tcx>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, ()>>,
            NoSolution,
        > {
            no_force
            desc { "evaluating `type_op_subtype` `{:?}`", goal }
        }

        /// Do not call this query directly: part of the `ProvePredicate` type-op
        query type_op_prove_predicate(
            goal: CanonicalTypeOpProvePredicateGoal<'tcx>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, ()>>,
            NoSolution,
        > {
            no_force
            desc { "evaluating `type_op_prove_predicate` `{:?}`", goal }
        }

        /// Do not call this query directly: part of the `Normalize` type-op
        query type_op_normalize_ty(
            goal: CanonicalTypeOpNormalizeGoal<'tcx, Ty<'tcx>>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, Ty<'tcx>>>,
            NoSolution,
        > {
            no_force
            desc { "normalizing `{:?}`", goal }
        }

        /// Do not call this query directly: part of the `Normalize` type-op
        query type_op_normalize_predicate(
            goal: CanonicalTypeOpNormalizeGoal<'tcx, ty::Predicate<'tcx>>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, ty::Predicate<'tcx>>>,
            NoSolution,
        > {
            no_force
            desc { "normalizing `{:?}`", goal }
        }

        /// Do not call this query directly: part of the `Normalize` type-op
        query type_op_normalize_poly_fn_sig(
            goal: CanonicalTypeOpNormalizeGoal<'tcx, ty::PolyFnSig<'tcx>>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, ty::PolyFnSig<'tcx>>>,
            NoSolution,
        > {
            no_force
            desc { "normalizing `{:?}`", goal }
        }

        /// Do not call this query directly: part of the `Normalize` type-op
        query type_op_normalize_fn_sig(
            goal: CanonicalTypeOpNormalizeGoal<'tcx, ty::FnSig<'tcx>>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, ty::FnSig<'tcx>>>,
            NoSolution,
        > {
            no_force
            desc { "normalizing `{:?}`", goal }
        }

        query substitute_normalize_and_test_predicates(key: (DefId, SubstsRef<'tcx>)) -> bool {
            no_force
            desc { |tcx|
                "testing substituted normalized predicates:`{}`",
                tcx.def_path_str(key.0)
            }
        }

        query method_autoderef_steps(
            goal: CanonicalTyGoal<'tcx>
        ) -> MethodAutoderefStepsResult<'tcx> {
            no_force
            desc { "computing autoderef types for `{:?}`", goal }
        }
    }

    Other {
        query target_features_whitelist(_: CrateNum) -> &'tcx FxHashMap<String, Option<Symbol>> {
            eval_always
            desc { "looking up the whitelist of target features" }
        }

        // Get an estimate of the size of an InstanceDef based on its MIR for CGU partitioning.
        query instance_def_size_estimate(def: ty::InstanceDef<'tcx>)
            -> usize {
            no_force
            desc { |tcx| "estimating size for `{}`", tcx.def_path_str(def.def_id()) }
        }

        query features_query(_: CrateNum) -> &'tcx rustc_feature::Features {
            eval_always
            desc { "looking up enabled feature gates" }
        }
    }
}
