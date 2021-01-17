use crate::dep_graph::SerializedDepNodeIndex;
use crate::mir::interpret::{GlobalId, LitToConstInput};
use crate::traits;
use crate::traits::query::{
    CanonicalPredicateGoal, CanonicalProjectionGoal, CanonicalTyGoal,
    CanonicalTypeOpAscribeUserTypeGoal, CanonicalTypeOpEqGoal, CanonicalTypeOpNormalizeGoal,
    CanonicalTypeOpProvePredicateGoal, CanonicalTypeOpSubtypeGoal,
};
use crate::ty::query::queries;
use crate::ty::subst::{GenericArg, SubstsRef};
use crate::ty::{self, ParamEnvAnd, Ty, TyCtxt};
use rustc_hir::def_id::{CrateNum, DefId, LocalDefId};
use rustc_query_system::query::QueryDescription;

use rustc_span::symbol::Symbol;
use std::borrow::Cow;

fn describe_as_module(def_id: LocalDefId, tcx: TyCtxt<'_>) -> String {
    if def_id.is_top_level_module() {
        "top-level module".to_string()
    } else {
        format!("module `{}`", tcx.def_path_str(def_id.to_def_id()))
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
        /// Represents crate as a whole (as distinct from the top-level crate module).
        /// If you call `hir_crate` (e.g., indirectly by calling `tcx.hir().krate()`),
        /// we will have to assume that any change means that you need to be recompiled.
        /// This is because the `hir_crate` query gives you access to all other items.
        /// To avoid this fate, do not call `tcx.hir().krate()`; instead,
        /// prefer wrappers like `tcx.visit_all_items_in_krate()`.
        query hir_crate(key: CrateNum) -> &'tcx Crate<'tcx> {
            eval_always
            no_hash
            desc { "get the crate HIR" }
        }

        /// The indexed HIR. This can be conveniently accessed by `tcx.hir()`.
        /// Avoid calling this query directly.
        query index_hir(_: CrateNum) -> &'tcx map::IndexedHir<'tcx> {
            eval_always
            no_hash
            desc { "index HIR" }
        }

        /// The items in a module.
        ///
        /// This can be conveniently accessed by `tcx.hir().visit_item_likes_in_module`.
        /// Avoid calling this query directly.
        query hir_module_items(key: LocalDefId) -> &'tcx hir::ModuleItems {
            eval_always
            desc { |tcx| "HIR module items in `{}`", tcx.def_path_str(key.to_def_id()) }
        }

        /// Gives access to the HIR node for the HIR owner `key`.
        ///
        /// This can be conveniently accessed by methods on `tcx.hir()`.
        /// Avoid calling this query directly.
        query hir_owner(key: LocalDefId) -> Option<&'tcx crate::hir::Owner<'tcx>> {
            eval_always
            desc { |tcx| "HIR owner of `{}`", tcx.def_path_str(key.to_def_id()) }
        }

        /// Gives access to the HIR nodes and bodies inside the HIR owner `key`.
        ///
        /// This can be conveniently accessed by methods on `tcx.hir()`.
        /// Avoid calling this query directly.
        query hir_owner_nodes(key: LocalDefId) -> Option<&'tcx crate::hir::OwnerNodes<'tcx>> {
            eval_always
            desc { |tcx| "HIR owner items in `{}`", tcx.def_path_str(key.to_def_id()) }
        }

        /// Computes the `DefId` of the corresponding const parameter in case the `key` is a
        /// const argument and returns `None` otherwise.
        ///
        /// ```ignore (incomplete)
        /// let a = foo::<7>();
        /// //            ^ Calling `opt_const_param_of` for this argument,
        ///
        /// fn foo<const N: usize>()
        /// //           ^ returns this `DefId`.
        ///
        /// fn bar() {
        /// // ^ While calling `opt_const_param_of` for other bodies returns `None`.
        /// }
        /// ```
        // It looks like caching this query on disk actually slightly
        // worsened performance in #74376.
        //
        // Once const generics are more prevalently used, we might want to
        // consider only caching calls returning `Some`.
        query opt_const_param_of(key: LocalDefId) -> Option<DefId> {
            desc { |tcx| "computing the optional const parameter of `{}`", tcx.def_path_str(key.to_def_id()) }
        }

        /// Records the type of every item.
        query type_of(key: DefId) -> Ty<'tcx> {
            desc { |tcx| "computing type of `{}`", tcx.def_path_str(key) }
            cache_on_disk_if { key.is_local() }
        }

        query analysis(key: CrateNum) -> Result<(), ErrorReported> {
            eval_always
            desc { "running analysis passes on this crate" }
        }

        /// Maps from the `DefId` of an item (trait/struct/enum/fn) to its
        /// associated generics.
        query generics_of(key: DefId) -> ty::Generics {
            desc { |tcx| "computing generics of `{}`", tcx.def_path_str(key) }
            storage(ArenaCacheSelector<'tcx>)
            cache_on_disk_if { key.is_local() }
            load_cached(tcx, id) {
                let generics: Option<ty::Generics> = tcx.queries.on_disk_cache.as_ref()
                                                        .and_then(|c| c.try_load_query_result(tcx, id));
                generics
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
            desc { |tcx| "computing predicates of `{}`", tcx.def_path_str(key) }
            cache_on_disk_if { key.is_local() }
        }

        /// Returns the list of bounds that can be used for
        /// `SelectionCandidate::ProjectionCandidate(_)` and
        /// `ProjectionTyCandidate::TraitDef`.
        /// Specifically this is the bounds written on the trait's type
        /// definition, or those after the `impl` keyword
        ///
        /// ```ignore (incomplete)
        /// type X: Bound + 'lt
        /// //      ^^^^^^^^^^^
        /// impl Debug + Display
        /// //   ^^^^^^^^^^^^^^^
        /// ```
        ///
        /// `key` is the `DefId` of the associated type or opaque type.
        ///
        /// Bounds from the parent (e.g. with nested impl trait) are not included.
        query explicit_item_bounds(key: DefId) -> &'tcx [(ty::Predicate<'tcx>, Span)] {
            desc { |tcx| "finding item bounds for `{}`", tcx.def_path_str(key) }
        }

        /// Elaborated version of the predicates from `explicit_item_bounds`.
        ///
        /// For example:
        ///
        /// ```
        /// trait MyTrait {
        ///     type MyAType: Eq + ?Sized;
        /// }
        /// ```
        ///
        /// `explicit_item_bounds` returns `[<Self as MyTrait>::MyAType: Eq]`,
        /// and `item_bounds` returns
        /// ```text
        /// [
        ///     <Self as Trait>::MyAType: Eq,
        ///     <Self as Trait>::MyAType: PartialEq<<Self as Trait>::MyAType>
        /// ]
        /// ```
        ///
        /// Bounds from the parent (e.g. with nested impl trait) are not included.
        query item_bounds(key: DefId) -> &'tcx ty::List<ty::Predicate<'tcx>> {
            desc { |tcx| "elaborating item bounds for `{}`", tcx.def_path_str(key) }
        }

        query projection_ty_from_predicates(key: (DefId, DefId)) -> Option<ty::ProjectionTy<'tcx>> {
            desc { |tcx| "finding projection type inside predicates of `{}`", tcx.def_path_str(key.0) }
        }

        query native_libraries(_: CrateNum) -> Lrc<Vec<NativeLib>> {
            desc { "looking up the native libraries of a linked crate" }
        }

        query lint_levels(_: CrateNum) -> LintLevelMap {
            storage(ArenaCacheSelector<'tcx>)
            eval_always
            desc { "computing the lint levels for items in this crate" }
        }

        query parent_module_from_def_id(key: LocalDefId) -> LocalDefId {
            eval_always
            desc { |tcx| "parent module of `{}`", tcx.def_path_str(key.to_def_id()) }
        }

        /// Internal helper query. Use `tcx.expansion_that_defined` instead
        query expn_that_defined(key: DefId) -> rustc_span::ExpnId {
            desc { |tcx| "expansion that defined `{}`", tcx.def_path_str(key) }
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
        query mir_keys(_: CrateNum) -> FxHashSet<LocalDefId> {
            storage(ArenaCacheSelector<'tcx>)
            desc { "getting a list of all mir_keys" }
        }

        /// Maps DefId's that have an associated `mir::Body` to the result
        /// of the MIR const-checking pass. This is the set of qualifs in
        /// the final value of a `const`.
        query mir_const_qualif(key: DefId) -> mir::ConstQualifs {
            desc { |tcx| "const checking `{}`", tcx.def_path_str(key) }
            cache_on_disk_if { key.is_local() }
        }
        query mir_const_qualif_const_arg(
            key: (LocalDefId, DefId)
        ) -> mir::ConstQualifs {
            desc {
                |tcx| "const checking the const argument `{}`",
                tcx.def_path_str(key.0.to_def_id())
            }
        }

        /// Fetch the MIR for a given `DefId` right after it's built - this includes
        /// unreachable code.
        query mir_built(key: ty::WithOptConstParam<LocalDefId>) -> &'tcx Steal<mir::Body<'tcx>> {
            desc { |tcx| "building MIR for `{}`", tcx.def_path_str(key.did.to_def_id()) }
        }

        /// Fetch the MIR for a given `DefId` up till the point where it is
        /// ready for const qualification.
        ///
        /// See the README for the `mir` module for details.
        query mir_const(key: ty::WithOptConstParam<LocalDefId>) -> &'tcx Steal<mir::Body<'tcx>> {
            desc {
                |tcx| "processing MIR for {}`{}`",
                if key.const_param_did.is_some() { "the const argument " } else { "" },
                tcx.def_path_str(key.did.to_def_id()),
            }
            no_hash
        }

        /// Try to build an abstract representation of the given constant.
        query mir_abstract_const(
            key: DefId
        ) -> Result<Option<&'tcx [mir::abstract_const::Node<'tcx>]>, ErrorReported> {
            desc {
                |tcx| "building an abstract representation for {}", tcx.def_path_str(key),
            }
        }
        /// Try to build an abstract representation of the given constant.
        query mir_abstract_const_of_const_arg(
            key: (LocalDefId, DefId)
        ) -> Result<Option<&'tcx [mir::abstract_const::Node<'tcx>]>, ErrorReported> {
            desc {
                |tcx|
                "building an abstract representation for the const argument {}",
                tcx.def_path_str(key.0.to_def_id()),
            }
        }

        query try_unify_abstract_consts(key: (
            (ty::WithOptConstParam<DefId>, SubstsRef<'tcx>),
            (ty::WithOptConstParam<DefId>, SubstsRef<'tcx>)
        )) -> bool {
            desc {
                |tcx| "trying to unify the generic constants {} and {}",
                tcx.def_path_str(key.0.0.did), tcx.def_path_str(key.1.0.did)
            }
        }

        query mir_drops_elaborated_and_const_checked(
            key: ty::WithOptConstParam<LocalDefId>
        ) -> &'tcx Steal<mir::Body<'tcx>> {
            no_hash
            desc { |tcx| "elaborating drops for `{}`", tcx.def_path_str(key.did.to_def_id()) }
        }

        query mir_for_ctfe(
            key: DefId
        ) -> &'tcx mir::Body<'tcx> {
            desc { |tcx| "caching mir of `{}` for CTFE", tcx.def_path_str(key) }
            cache_on_disk_if { key.is_local() }
        }

        query mir_for_ctfe_of_const_arg(key: (LocalDefId, DefId)) -> &'tcx mir::Body<'tcx> {
            desc {
                |tcx| "MIR for CTFE of the const argument `{}`",
                tcx.def_path_str(key.0.to_def_id())
            }
        }

        query mir_promoted(key: ty::WithOptConstParam<LocalDefId>) ->
            (
                &'tcx Steal<mir::Body<'tcx>>,
                &'tcx Steal<IndexVec<mir::Promoted, mir::Body<'tcx>>>
            ) {
            no_hash
            desc {
                |tcx| "processing {}`{}`",
                if key.const_param_did.is_some() { "the const argument " } else { "" },
                tcx.def_path_str(key.did.to_def_id()),
            }
        }

        /// MIR after our optimization passes have run. This is MIR that is ready
        /// for codegen. This is also the only query that can fetch non-local MIR, at present.
        query optimized_mir(key: DefId) -> &'tcx mir::Body<'tcx> {
            desc { |tcx| "optimizing MIR for `{}`", tcx.def_path_str(key) }
            cache_on_disk_if { key.is_local() }
        }

        /// Returns coverage summary info for a function, after executing the `InstrumentCoverage`
        /// MIR pass (assuming the -Zinstrument-coverage option is enabled).
        query coverageinfo(key: DefId) -> mir::CoverageInfo {
            desc { |tcx| "retrieving coverage info from MIR for `{}`", tcx.def_path_str(key) }
            storage(ArenaCacheSelector<'tcx>)
            cache_on_disk_if { key.is_local() }
        }

        /// Returns the name of the file that contains the function body, if instrumented for coverage.
        query covered_file_name(key: DefId) -> Option<Symbol> {
            desc { |tcx| "retrieving the covered file name, if instrumented, for `{}`", tcx.def_path_str(key) }
            storage(ArenaCacheSelector<'tcx>)
            cache_on_disk_if { key.is_local() }
        }

        /// Returns the `CodeRegions` for a function that has instrumented coverage, in case the
        /// function was optimized out before codegen, and before being added to the Coverage Map.
        query covered_code_regions(key: DefId) -> Vec<&'tcx mir::coverage::CodeRegion> {
            desc { |tcx| "retrieving the covered `CodeRegion`s, if instrumented, for `{}`", tcx.def_path_str(key) }
            storage(ArenaCacheSelector<'tcx>)
            cache_on_disk_if { key.is_local() }
        }

        /// The `DefId` is the `DefId` of the containing MIR body. Promoteds do not have their own
        /// `DefId`. This function returns all promoteds in the specified body. The body references
        /// promoteds by the `DefId` and the `mir::Promoted` index. This is necessary, because
        /// after inlining a body may refer to promoteds from other bodies. In that case you still
        /// need to use the `DefId` of the original body.
        query promoted_mir(key: DefId) -> &'tcx IndexVec<mir::Promoted, mir::Body<'tcx>> {
            desc { |tcx| "optimizing promoted MIR for `{}`", tcx.def_path_str(key) }
            cache_on_disk_if { key.is_local() }
        }
        query promoted_mir_of_const_arg(
            key: (LocalDefId, DefId)
        ) -> &'tcx IndexVec<mir::Promoted, mir::Body<'tcx>> {
            desc {
                |tcx| "optimizing promoted MIR for the const argument `{}`",
                tcx.def_path_str(key.0.to_def_id()),
            }
        }
    }

    TypeChecking {
        /// Erases regions from `ty` to yield a new type.
        /// Normally you would just use `tcx.erase_regions(value)`,
        /// however, which uses this query as a kind of cache.
        query erase_regions_ty(ty: Ty<'tcx>) -> Ty<'tcx> {
            // This query is not expected to have input -- as a result, it
            // is not a good candidates for "replay" because it is essentially a
            // pure function of its input (and hence the expectation is that
            // no caller would be green **apart** from just these
            // queries). Making it anonymous avoids hashing the result, which
            // may save a bit of time.
            anon
            desc { "erasing regions from `{:?}`", ty }
        }
    }

    Linking {
        query wasm_import_module_map(_: CrateNum) -> FxHashMap<DefId, String> {
            storage(ArenaCacheSelector<'tcx>)
            desc { "wasm import module map" }
        }
    }

    Other {
        /// Maps from the `DefId` of an item (trait/struct/enum/fn) to the
        /// predicates (where-clauses) directly defined on it. This is
        /// equal to the `explicit_predicates_of` predicates plus the
        /// `inferred_outlives_of` predicates.
        query predicates_defined_on(key: DefId) -> ty::GenericPredicates<'tcx> {
            desc { |tcx| "computing predicates of `{}`", tcx.def_path_str(key) }
        }

        /// Returns everything that looks like a predicate written explicitly
        /// by the user on a trait item.
        ///
        /// Traits are unusual, because predicates on associated types are
        /// converted into bounds on that type for backwards compatibility:
        ///
        /// trait X where Self::U: Copy { type U; }
        ///
        /// becomes
        ///
        /// trait X { type U: Copy; }
        ///
        /// `explicit_predicates_of` and `explicit_item_bounds` will then take
        /// the appropriate subsets of the predicates here.
        query trait_explicit_predicates_and_bounds(key: LocalDefId) -> ty::GenericPredicates<'tcx> {
            desc { |tcx| "computing explicit predicates of trait `{}`", tcx.def_path_str(key.to_def_id()) }
        }

        /// Returns the predicates written explicitly by the user.
        query explicit_predicates_of(key: DefId) -> ty::GenericPredicates<'tcx> {
            desc { |tcx| "computing explicit predicates of `{}`", tcx.def_path_str(key) }
        }

        /// Returns the inferred outlives predicates (e.g., for `struct
        /// Foo<'a, T> { x: &'a T }`, this would return `T: 'a`).
        query inferred_outlives_of(key: DefId) -> &'tcx [(ty::Predicate<'tcx>, Span)] {
            desc { |tcx| "computing inferred outlives predicates of `{}`", tcx.def_path_str(key) }
        }

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
        query type_param_predicates(key: (DefId, LocalDefId)) -> ty::GenericPredicates<'tcx> {
            desc { |tcx| "computing the bounds for type parameter `{}`", {
                let id = tcx.hir().local_def_id_to_hir_id(key.1);
                tcx.hir().ty_param_name(id)
            }}
        }

        query trait_def(key: DefId) -> ty::TraitDef {
            desc { |tcx| "computing trait definition for `{}`", tcx.def_path_str(key) }
            storage(ArenaCacheSelector<'tcx>)
        }
        query adt_def(key: DefId) -> &'tcx ty::AdtDef {
            desc { |tcx| "computing ADT definition for `{}`", tcx.def_path_str(key) }
        }
        query adt_destructor(key: DefId) -> Option<ty::Destructor> {
            desc { |tcx| "computing `Drop` impl for `{}`", tcx.def_path_str(key) }
        }

        // The cycle error here should be reported as an error by `check_representable`.
        // We consider the type as Sized in the meanwhile to avoid
        // further errors (done in impl Value for AdtSizedConstraint).
        // Use `cycle_delay_bug` to delay the cycle error here to be emitted later
        // in case we accidentally otherwise don't emit an error.
        query adt_sized_constraint(
            key: DefId
        ) -> AdtSizedConstraint<'tcx> {
            desc { |tcx| "computing `Sized` constraints for `{}`", tcx.def_path_str(key) }
            cycle_delay_bug
        }

        query adt_dtorck_constraint(
            key: DefId
        ) -> Result<DtorckConstraint<'tcx>, NoSolution> {
            desc { |tcx| "computing drop-check constraints for `{}`", tcx.def_path_str(key) }
        }

        /// Returns `true` if this is a const fn, use the `is_const_fn` to know whether your crate
        /// actually sees it as const fn (e.g., the const-fn-ness might be unstable and you might
        /// not have the feature gate active).
        ///
        /// **Do not call this function manually.** It is only meant to cache the base data for the
        /// `is_const_fn` function.
        query is_const_fn_raw(key: DefId) -> bool {
            desc { |tcx| "checking if item is const fn: `{}`", tcx.def_path_str(key) }
        }

        /// Returns `true` if this is a const `impl`. **Do not call this function manually.**
        ///
        /// This query caches the base data for the `is_const_impl` helper function, which also
        /// takes into account stability attributes (e.g., `#[rustc_const_unstable]`).
        query is_const_impl_raw(key: DefId) -> bool {
            desc { |tcx| "checking if item is const impl: `{}`", tcx.def_path_str(key) }
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
        query is_promotable_const_fn(key: DefId) -> bool {
            desc { |tcx| "checking if item is promotable: `{}`", tcx.def_path_str(key) }
        }

        /// Returns `true` if this is a foreign item (i.e., linked via `extern { ... }`).
        query is_foreign_item(key: DefId) -> bool {
            desc { |tcx| "checking if `{}` is a foreign item", tcx.def_path_str(key) }
        }

        /// Returns `Some(mutability)` if the node pointed to by `def_id` is a static item.
        query static_mutability(def_id: DefId) -> Option<hir::Mutability> {
            desc { |tcx| "looking up static mutability of `{}`", tcx.def_path_str(def_id) }
        }

        /// Returns `Some(generator_kind)` if the node pointed to by `def_id` is a generator.
        query generator_kind(def_id: DefId) -> Option<hir::GeneratorKind> {
            desc { |tcx| "looking up generator kind of `{}`", tcx.def_path_str(def_id) }
        }

        /// Gets a map with the variance of every item; use `item_variance` instead.
        query crate_variances(_: CrateNum) -> ty::CrateVariancesMap<'tcx> {
            storage(ArenaCacheSelector<'tcx>)
            desc { "computing the variances for items in this crate" }
        }

        /// Maps from the `DefId` of a type or region parameter to its (inferred) variance.
        query variances_of(def_id: DefId) -> &'tcx [ty::Variance] {
            desc { |tcx| "computing the variances of `{}`", tcx.def_path_str(def_id) }
        }
    }

    TypeChecking {
        /// Maps from thee `DefId` of a type to its (inferred) outlives.
        query inferred_outlives_crate(_: CrateNum)
            -> ty::CratePredicatesMap<'tcx> {
            storage(ArenaCacheSelector<'tcx>)
            desc { "computing the inferred outlives predicates for items in this crate" }
        }
    }

    Other {
        /// Maps from an impl/trait `DefId to a list of the `DefId`s of its items.
        query associated_item_def_ids(key: DefId) -> &'tcx [DefId] {
            desc { |tcx| "collecting associated items of `{}`", tcx.def_path_str(key) }
        }

        /// Maps from a trait item to the trait item "descriptor".
        query associated_item(key: DefId) -> ty::AssocItem {
            desc { |tcx| "computing associated item data for `{}`", tcx.def_path_str(key) }
            storage(ArenaCacheSelector<'tcx>)
        }

        /// Collects the associated items defined on a trait or impl.
        query associated_items(key: DefId) -> ty::AssociatedItems<'tcx> {
            storage(ArenaCacheSelector<'tcx>)
            desc { |tcx| "collecting associated items of {}", tcx.def_path_str(key) }
        }

        /// Given an `impl_id`, return the trait it implements.
        /// Return `None` if this is an inherent impl.
        query impl_trait_ref(impl_id: DefId) -> Option<ty::TraitRef<'tcx>> {
            desc { |tcx| "computing trait implemented by `{}`", tcx.def_path_str(impl_id) }
        }
        query impl_polarity(impl_id: DefId) -> ty::ImplPolarity {
            desc { |tcx| "computing implementation polarity of `{}`", tcx.def_path_str(impl_id) }
        }

        query issue33140_self_ty(key: DefId) -> Option<ty::Ty<'tcx>> {
            desc { |tcx| "computing Self type wrt issue #33140 `{}`", tcx.def_path_str(key) }
        }
    }

    TypeChecking {
        /// Maps a `DefId` of a type to a list of its inherent impls.
        /// Contains implementations of methods that are inherent to a type.
        /// Methods in these implementations don't need to be exported.
        query inherent_impls(key: DefId) -> &'tcx [DefId] {
            desc { |tcx| "collecting inherent impls for `{}`", tcx.def_path_str(key) }
            eval_always
        }
    }

    TypeChecking {
        /// The result of unsafety-checking this `LocalDefId`.
        query unsafety_check_result(key: LocalDefId) -> &'tcx mir::UnsafetyCheckResult {
            desc { |tcx| "unsafety-checking `{}`", tcx.def_path_str(key.to_def_id()) }
            cache_on_disk_if { true }
        }
        query unsafety_check_result_for_const_arg(key: (LocalDefId, DefId)) -> &'tcx mir::UnsafetyCheckResult {
            desc {
                |tcx| "unsafety-checking the const argument `{}`",
                tcx.def_path_str(key.0.to_def_id())
            }
        }

        /// HACK: when evaluated, this reports a "unsafe derive on repr(packed)" error.
        ///
        /// Unsafety checking is executed for each method separately, but we only want
        /// to emit this error once per derive. As there are some impls with multiple
        /// methods, we use a query for deduplication.
        query unsafe_derive_on_repr_packed(key: LocalDefId) -> () {
            desc { |tcx| "processing `{}`", tcx.def_path_str(key.to_def_id()) }
        }

        /// The signature of functions.
        query fn_sig(key: DefId) -> ty::PolyFnSig<'tcx> {
            desc { |tcx| "computing function signature of `{}`", tcx.def_path_str(key) }
        }
    }

    Other {
        query lint_mod(key: LocalDefId) -> () {
            desc { |tcx| "linting {}", describe_as_module(key, tcx) }
        }

        /// Checks the attributes in the module.
        query check_mod_attrs(key: LocalDefId) -> () {
            desc { |tcx| "checking attributes in {}", describe_as_module(key, tcx) }
        }

        query check_mod_unstable_api_usage(key: LocalDefId) -> () {
            desc { |tcx| "checking for unstable API usage in {}", describe_as_module(key, tcx) }
        }

        /// Checks the const bodies in the module for illegal operations (e.g. `if` or `loop`).
        query check_mod_const_bodies(key: LocalDefId) -> () {
            desc { |tcx| "checking consts in {}", describe_as_module(key, tcx) }
        }

        /// Checks the loops in the module.
        query check_mod_loops(key: LocalDefId) -> () {
            desc { |tcx| "checking loops in {}", describe_as_module(key, tcx) }
        }

        query check_mod_naked_functions(key: LocalDefId) -> () {
            desc { |tcx| "checking naked functions in {}", describe_as_module(key, tcx) }
        }

        query check_mod_item_types(key: LocalDefId) -> () {
            desc { |tcx| "checking item types in {}", describe_as_module(key, tcx) }
        }

        query check_mod_privacy(key: LocalDefId) -> () {
            desc { |tcx| "checking privacy in {}", describe_as_module(key, tcx) }
        }

        query check_mod_intrinsics(key: LocalDefId) -> () {
            desc { |tcx| "checking intrinsics in {}", describe_as_module(key, tcx) }
        }

        query check_mod_liveness(key: LocalDefId) -> () {
            desc { |tcx| "checking liveness of variables in {}", describe_as_module(key, tcx) }
        }

        query check_mod_impl_wf(key: LocalDefId) -> () {
            desc { |tcx| "checking that impls are well-formed in {}", describe_as_module(key, tcx) }
        }

        query collect_mod_item_types(key: LocalDefId) -> () {
            desc { |tcx| "collecting item types in {}", describe_as_module(key, tcx) }
        }

        /// Caches `CoerceUnsized` kinds for impls on custom types.
        query coerce_unsized_info(key: DefId)
            -> ty::adjustment::CoerceUnsizedInfo {
                desc { |tcx| "computing CoerceUnsized info for `{}`", tcx.def_path_str(key) }
            }
    }

    TypeChecking {
        query typeck_item_bodies(_: CrateNum) -> () {
            desc { "type-checking all item bodies" }
        }

        query typeck(key: LocalDefId) -> &'tcx ty::TypeckResults<'tcx> {
            desc { |tcx| "type-checking `{}`", tcx.def_path_str(key.to_def_id()) }
            cache_on_disk_if { true }
        }
        query typeck_const_arg(
            key: (LocalDefId, DefId)
        ) -> &'tcx ty::TypeckResults<'tcx> {
            desc {
                |tcx| "type-checking the const argument `{}`",
                tcx.def_path_str(key.0.to_def_id()),
            }
        }
        query diagnostic_only_typeck(key: LocalDefId) -> &'tcx ty::TypeckResults<'tcx> {
            desc { |tcx| "type-checking `{}`", tcx.def_path_str(key.to_def_id()) }
            cache_on_disk_if { true }
            load_cached(tcx, id) {
                let typeck_results: Option<ty::TypeckResults<'tcx>> = tcx
                    .queries.on_disk_cache.as_ref()
                    .and_then(|c| c.try_load_query_result(tcx, id));

                typeck_results.map(|x| &*tcx.arena.alloc(x))
            }
        }
    }

    Other {
        query used_trait_imports(key: LocalDefId) -> &'tcx FxHashSet<LocalDefId> {
            desc { |tcx| "used_trait_imports `{}`", tcx.def_path_str(key.to_def_id()) }
            cache_on_disk_if { true }
        }
    }

    TypeChecking {
        query has_typeck_results(def_id: DefId) -> bool {
            desc { |tcx| "checking whether `{}` has a body", tcx.def_path_str(def_id) }
        }

        query coherent_trait(def_id: DefId) -> () {
            desc { |tcx| "coherence checking all impls of trait `{}`", tcx.def_path_str(def_id) }
        }
    }

    BorrowChecking {
        /// Borrow-checks the function body. If this is a closure, returns
        /// additional requirements that the closure's creator must verify.
        query mir_borrowck(key: LocalDefId) -> &'tcx mir::BorrowCheckResult<'tcx> {
            desc { |tcx| "borrow-checking `{}`", tcx.def_path_str(key.to_def_id()) }
            cache_on_disk_if(tcx, opt_result) {
                tcx.is_closure(key.to_def_id())
                    || opt_result.map_or(false, |r| !r.concrete_opaque_types.is_empty())
            }
        }
        query mir_borrowck_const_arg(key: (LocalDefId, DefId)) -> &'tcx mir::BorrowCheckResult<'tcx> {
            desc {
                |tcx| "borrow-checking the const argument`{}`",
                tcx.def_path_str(key.0.to_def_id())
            }
        }
    }

    TypeChecking {
        /// Gets a complete map from all types to their inherent impls.
        /// Not meant to be used directly outside of coherence.
        /// (Defined only for `LOCAL_CRATE`.)
        query crate_inherent_impls(k: CrateNum)
            -> CrateInherentImpls {
            storage(ArenaCacheSelector<'tcx>)
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
        /// Evaluates a constant and returns the computed allocation.
        ///
        /// **Do not use this** directly, use the `tcx.eval_static_initializer` wrapper.
        query eval_to_allocation_raw(key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>)
            -> EvalToAllocationRawResult<'tcx> {
            desc { |tcx|
                "const-evaluating + checking `{}`",
                key.value.display(tcx)
            }
            cache_on_disk_if { true }
        }

        /// Evaluates const items or anonymous constants
        /// (such as enum variant explicit discriminants or array lengths)
        /// into a representation suitable for the type system and const generics.
        ///
        /// **Do not use this** directly, use one of the following wrappers: `tcx.const_eval_poly`,
        /// `tcx.const_eval_resolve`, `tcx.const_eval_instance`, or `tcx.const_eval_global_id`.
        query eval_to_const_value_raw(key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>)
            -> EvalToConstValueResult<'tcx> {
            desc { |tcx|
                "simplifying constant for the type system `{}`",
                key.value.display(tcx)
            }
            cache_on_disk_if { true }
        }

        /// Destructure a constant ADT or array into its variant index and its
        /// field values.
        query destructure_const(
            key: ty::ParamEnvAnd<'tcx, &'tcx ty::Const<'tcx>>
        ) -> mir::DestructuredConst<'tcx> {
            desc { "destructure constant" }
        }

        /// Dereference a constant reference or raw pointer and turn the result into a constant
        /// again.
        query deref_const(
            key: ty::ParamEnvAnd<'tcx, &'tcx ty::Const<'tcx>>
        ) -> &'tcx ty::Const<'tcx> {
            desc { "deref constant" }
        }

        query const_caller_location(key: (rustc_span::Symbol, u32, u32)) -> ConstValue<'tcx> {
            desc { "get a &core::panic::Location referring to a span" }
        }

        query lit_to_const(
            key: LitToConstInput<'tcx>
        ) -> Result<&'tcx ty::Const<'tcx>, LitToConstError> {
            desc { "converting literal to const" }
        }
    }

    TypeChecking {
        query check_match(key: DefId) {
            desc { |tcx| "match-checking `{}`", tcx.def_path_str(key) }
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
        query reachable_set(_: CrateNum) -> FxHashSet<LocalDefId> {
            storage(ArenaCacheSelector<'tcx>)
            desc { "reachability" }
        }

        /// Per-body `region::ScopeTree`. The `DefId` should be the owner `DefId` for the body;
        /// in the case of closures, this will be redirected to the enclosing function.
        query region_scope_tree(def_id: DefId) -> &'tcx region::ScopeTree {
            desc { |tcx| "computing drop scopes for `{}`", tcx.def_path_str(def_id) }
        }

        query mir_shims(key: ty::InstanceDef<'tcx>) -> mir::Body<'tcx> {
            storage(ArenaCacheSelector<'tcx>)
            desc { |tcx| "generating MIR shim for `{}`", tcx.def_path_str(key.def_id()) }
        }

        /// The `symbol_name` query provides the symbol name for calling a
        /// given instance from the local crate. In particular, it will also
        /// look up the correct symbol name of instances from upstream crates.
        query symbol_name(key: ty::Instance<'tcx>) -> ty::SymbolName<'tcx> {
            desc { "computing the symbol for `{}`", key }
            cache_on_disk_if { true }
        }

        query def_kind(def_id: DefId) -> DefKind {
            desc { |tcx| "looking up definition kind of `{}`", tcx.def_path_str(def_id) }
        }

        query def_span(def_id: DefId) -> Span {
            desc { |tcx| "looking up span for `{}`", tcx.def_path_str(def_id) }
            // FIXME(mw): DefSpans are not really inputs since they are derived from
            // HIR. But at the moment HIR hashing still contains some hacks that allow
            // to make type debuginfo to be source location independent. Declaring
            // DefSpan an input makes sure that changes to these are always detected
            // regardless of HIR hashing.
            eval_always
        }

        query def_ident_span(def_id: DefId) -> Option<Span> {
            desc { |tcx| "looking up span for `{}`'s identifier", tcx.def_path_str(def_id) }
        }

        query lookup_stability(def_id: DefId) -> Option<&'tcx attr::Stability> {
            desc { |tcx| "looking up stability of `{}`", tcx.def_path_str(def_id) }
        }

        query lookup_const_stability(def_id: DefId) -> Option<&'tcx attr::ConstStability> {
            desc { |tcx| "looking up const stability of `{}`", tcx.def_path_str(def_id) }
        }

        query lookup_deprecation_entry(def_id: DefId) -> Option<DeprecationEntry> {
            desc { |tcx| "checking whether `{}` is deprecated", tcx.def_path_str(def_id) }
        }

        query item_attrs(def_id: DefId) -> &'tcx [ast::Attribute] {
            desc { |tcx| "collecting attributes of `{}`", tcx.def_path_str(def_id) }
        }
    }

    Codegen {
        query codegen_fn_attrs(def_id: DefId) -> CodegenFnAttrs {
            desc { |tcx| "computing codegen attributes of `{}`", tcx.def_path_str(def_id) }
            storage(ArenaCacheSelector<'tcx>)
            cache_on_disk_if { true }
        }
    }

    Other {
        query fn_arg_names(def_id: DefId) -> &'tcx [rustc_span::symbol::Ident] {
            desc { |tcx| "looking up function parameter names for `{}`", tcx.def_path_str(def_id) }
        }
        /// Gets the rendered value of the specified constant or associated constant.
        /// Used by rustdoc.
        query rendered_const(def_id: DefId) -> String {
            desc { |tcx| "rendering constant intializer of `{}`", tcx.def_path_str(def_id) }
        }
        query impl_parent(def_id: DefId) -> Option<DefId> {
            desc { |tcx| "computing specialization parent impl of `{}`", tcx.def_path_str(def_id) }
        }
    }

    TypeChecking {
        /// Given an `associated_item`, find the trait it belongs to.
        /// Return `None` if the `DefId` is not an associated item.
        query trait_of_item(associated_item: DefId) -> Option<DefId> {
            desc { |tcx| "finding trait defining `{}`", tcx.def_path_str(associated_item) }
        }
    }

    Codegen {
        query is_ctfe_mir_available(key: DefId) -> bool {
            desc { |tcx| "checking if item has ctfe mir available: `{}`", tcx.def_path_str(key) }
        }
        query is_mir_available(key: DefId) -> bool {
            desc { |tcx| "checking if item has mir available: `{}`", tcx.def_path_str(key) }
        }
    }

    Other {
        query vtable_methods(key: ty::PolyTraitRef<'tcx>)
                            -> &'tcx [Option<(DefId, SubstsRef<'tcx>)>] {
            desc { |tcx| "finding all methods for trait {}", tcx.def_path_str(key.def_id()) }
        }
    }

    Codegen {
        query codegen_fulfill_obligation(
            key: (ty::ParamEnv<'tcx>, ty::PolyTraitRef<'tcx>)
        ) -> Result<ImplSource<'tcx, ()>, ErrorReported> {
            cache_on_disk_if { true }
            desc { |tcx|
                "checking if `{}` fulfills its obligations",
                tcx.def_path_str(key.1.def_id())
            }
        }
    }

    TypeChecking {
        /// Return all `impl` blocks in the current crate.
        ///
        /// To allow caching this between crates, you must pass in [`LOCAL_CRATE`] as the crate number.
        /// Passing in any other crate will cause an ICE.
        ///
        /// [`LOCAL_CRATE`]: rustc_hir::def_id::LOCAL_CRATE
        query all_local_trait_impls(local_crate: CrateNum) -> &'tcx BTreeMap<DefId, Vec<hir::HirId>> {
            desc { "local trait impls" }
        }

        /// Given a trait `trait_id`, return all known `impl` blocks.
        query trait_impls_of(trait_id: DefId) -> ty::trait_def::TraitImpls {
            storage(ArenaCacheSelector<'tcx>)
            desc { |tcx| "trait impls of `{}`", tcx.def_path_str(trait_id) }
        }

        query specialization_graph_of(trait_id: DefId) -> specialization_graph::Graph {
            storage(ArenaCacheSelector<'tcx>)
            desc { |tcx| "building specialization graph of trait `{}`", tcx.def_path_str(trait_id) }
            cache_on_disk_if { true }
        }
        query object_safety_violations(trait_id: DefId) -> &'tcx [traits::ObjectSafetyViolation] {
            desc { |tcx| "determine object safety of trait `{}`", tcx.def_path_str(trait_id) }
        }

        /// Gets the ParameterEnvironment for a given item; this environment
        /// will be in "user-facing" mode, meaning that it is suitable for
        /// type-checking etc, and it does not normalize specializable
        /// associated types. This is almost always what you want,
        /// unless you are doing MIR optimizations, in which case you
        /// might want to use `reveal_all()` method to change modes.
        query param_env(def_id: DefId) -> ty::ParamEnv<'tcx> {
            desc { |tcx| "computing normalized predicates of `{}`", tcx.def_path_str(def_id) }
        }

        /// Like `param_env`, but returns the `ParamEnv in `Reveal::All` mode.
        /// Prefer this over `tcx.param_env(def_id).with_reveal_all_normalized(tcx)`,
        /// as this method is more efficient.
        query param_env_reveal_all_normalized(def_id: DefId) -> ty::ParamEnv<'tcx> {
            desc { |tcx| "computing revealed normalized predicates of `{}`", tcx.def_path_str(def_id) }
        }

        /// Trait selection queries. These are best used by invoking `ty.is_copy_modulo_regions()`,
        /// `ty.is_copy()`, etc, since that will prune the environment where possible.
        query is_copy_raw(env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool {
            desc { "computing whether `{}` is `Copy`", env.value }
        }
        /// Query backing `TyS::is_sized`.
        query is_sized_raw(env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool {
            desc { "computing whether `{}` is `Sized`", env.value }
        }
        /// Query backing `TyS::is_freeze`.
        query is_freeze_raw(env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool {
            desc { "computing whether `{}` is freeze", env.value }
        }
        /// Query backing `TyS::needs_drop`.
        query needs_drop_raw(env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>) -> bool {
            desc { "computing whether `{}` needs drop", env.value }
        }

        /// Query backing `TyS::is_structural_eq_shallow`.
        ///
        /// This is only correct for ADTs. Call `is_structural_eq_shallow` to handle all types
        /// correctly.
        query has_structural_eq_impls(ty: Ty<'tcx>) -> bool {
            desc {
                "computing whether `{:?}` implements `PartialStructuralEq` and `StructuralEq`",
                ty
            }
        }

        /// A list of types where the ADT requires drop if and only if any of
        /// those types require drop. If the ADT is known to always need drop
        /// then `Err(AlwaysRequiresDrop)` is returned.
        query adt_drop_tys(def_id: DefId) -> Result<&'tcx ty::List<Ty<'tcx>>, AlwaysRequiresDrop> {
            desc { |tcx| "computing when `{}` needs drop", tcx.def_path_str(def_id) }
            cache_on_disk_if { true }
        }

        query layout_raw(
            env: ty::ParamEnvAnd<'tcx, Ty<'tcx>>
        ) -> Result<&'tcx rustc_target::abi::Layout, ty::layout::LayoutError<'tcx>> {
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

        query extern_crate(def_id: DefId) -> Option<&'tcx ExternCrate> {
            eval_always
            desc { "getting crate's ExternCrateData" }
        }
    }

    TypeChecking {
        query specializes(_: (DefId, DefId)) -> bool {
            desc { "computing whether impls specialize one another" }
        }
        query in_scope_traits_map(_: LocalDefId)
            -> Option<&'tcx FxHashMap<ItemLocalId, StableVec<TraitCandidate>>> {
            eval_always
            desc { "traits in scope at a block" }
        }
    }

    Other {
        query module_exports(def_id: LocalDefId) -> Option<&'tcx [Export<LocalDefId>]> {
            desc { |tcx| "looking up items exported by `{}`", tcx.def_path_str(def_id.to_def_id()) }
            eval_always
        }
    }

    TypeChecking {
        query impl_defaultness(def_id: DefId) -> hir::Defaultness {
            desc { |tcx| "looking up whether `{}` is a default impl", tcx.def_path_str(def_id) }
        }

        query check_item_well_formed(key: LocalDefId) -> () {
            desc { |tcx| "checking that `{}` is well-formed", tcx.def_path_str(key.to_def_id()) }
        }
        query check_trait_item_well_formed(key: LocalDefId) -> () {
            desc { |tcx| "checking that `{}` is well-formed", tcx.def_path_str(key.to_def_id()) }
        }
        query check_impl_item_well_formed(key: LocalDefId) -> () {
            desc { |tcx| "checking that `{}` is well-formed", tcx.def_path_str(key.to_def_id()) }
        }
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
            -> DefIdMap<SymbolExportLevel> {
            storage(ArenaCacheSelector<'tcx>)
            desc { "looking up the exported symbols of a crate" }
        }
        query is_reachable_non_generic(def_id: DefId) -> bool {
            desc { |tcx| "checking whether `{}` is an exported symbol", tcx.def_path_str(def_id) }
        }
        query is_unreachable_local_definition(def_id: DefId) -> bool {
            desc { |tcx|
                "checking whether `{}` is reachable from outside the crate",
                tcx.def_path_str(def_id),
            }
        }
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
        ) -> DefIdMap<FxHashMap<SubstsRef<'tcx>, CrateNum>> {
            storage(ArenaCacheSelector<'tcx>)
            desc { "collecting available upstream monomorphizations `{:?}`", k }
        }

        /// Returns the set of upstream monomorphizations available for the
        /// generic function identified by the given `def_id`. The query makes
        /// sure to make a stable selection if the same monomorphization is
        /// available in multiple upstream crates.
        ///
        /// You likely want to call `Instance::upstream_monomorphization()`
        /// instead of invoking this query directly.
        query upstream_monomorphizations_for(def_id: DefId)
            -> Option<&'tcx FxHashMap<SubstsRef<'tcx>, CrateNum>> {
                desc { |tcx|
                    "collecting available upstream monomorphizations for `{}`",
                    tcx.def_path_str(def_id),
                }
            }

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
        }
    }

    Other {
        query foreign_modules(_: CrateNum) -> Lrc<FxHashMap<DefId, ForeignModule>> {
            desc { "looking up the foreign modules of a linked crate" }
        }

        /// Identifies the entry-point (e.g., the `main` function) for a given
        /// crate, returning `None` if there is no entry point (such as for library crates).
        query entry_fn(_: CrateNum) -> Option<(LocalDefId, EntryFnType)> {
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
        // The macro which defines `rustc_metadata::provide_extern` depends on this query's name.
        // Changing the name should cause a compiler error, but in case that changes, be aware.
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
        query crate_extern_paths(_: CrateNum) -> Vec<PathBuf> {
            eval_always
            desc { "looking up the paths for extern crates" }
        }
    }

    TypeChecking {
        /// Given a crate and a trait, look up all impls of that trait in the crate.
        /// Return `(impl_id, self_ty)`.
        query implementations_of_trait(_: (CrateNum, DefId))
            -> &'tcx [(DefId, Option<ty::fast_reject::SimplifiedType>)] {
            desc { "looking up implementations of a trait in a crate" }
        }

        /// Given a crate, look up all trait impls in that crate.
        /// Return `(impl_id, self_ty)`.
        query all_trait_implementations(_: CrateNum)
            -> &'tcx [(DefId, Option<ty::fast_reject::SimplifiedType>)] {
            desc { "looking up all (?) trait implementations" }
        }
    }

    Other {
        query is_dllimport_foreign_item(def_id: DefId) -> bool {
            desc { |tcx| "is_dllimport_foreign_item({})", tcx.def_path_str(def_id) }
        }
        query is_statically_included_foreign_item(def_id: DefId) -> bool {
            desc { |tcx| "is_statically_included_foreign_item({})", tcx.def_path_str(def_id) }
        }
        query native_library_kind(def_id: DefId)
            -> Option<NativeLibKind> {
            desc { |tcx| "native_library_kind({})", tcx.def_path_str(def_id) }
        }
    }

    Linking {
        query link_args(_: CrateNum) -> Lrc<Vec<String>> {
            eval_always
            desc { "looking up link arguments for a crate" }
        }
    }

    BorrowChecking {
        /// Lifetime resolution. See `middle::resolve_lifetimes`.
        query resolve_lifetimes(_: CrateNum) -> ResolveLifetimes {
            storage(ArenaCacheSelector<'tcx>)
            desc { "resolving lifetimes" }
        }
        query named_region_map(_: LocalDefId) ->
            Option<&'tcx FxHashMap<ItemLocalId, Region>> {
            desc { "looking up a named region" }
        }
        query is_late_bound_map(_: LocalDefId) ->
            Option<(LocalDefId, &'tcx FxHashSet<ItemLocalId>)> {
            desc { "testing if a region is late bound" }
        }
        query object_lifetime_defaults_map(_: LocalDefId)
            -> Option<&'tcx FxHashMap<ItemLocalId, Vec<ObjectLifetimeDefault>>> {
            desc { "looking up lifetime defaults for a region" }
        }
    }

    TypeChecking {
        query visibility(def_id: DefId) -> ty::Visibility {
            eval_always
            desc { |tcx| "computing visibility of `{}`", tcx.def_path_str(def_id) }
        }

        /// Computes the set of modules from which this type is visibly uninhabited.
        /// To check whether a type is uninhabited at all (not just from a given module), you could
        /// check whether the forest is empty.
        query type_uninhabited_from(
            key: ty::ParamEnvAnd<'tcx, Ty<'tcx>>
        ) -> ty::inhabitedness::DefIdForest {
            desc { "computing the inhabitedness of `{:?}`", key }
        }
    }

    Other {
        query dep_kind(_: CrateNum) -> CrateDepKind {
            eval_always
            desc { "fetching what a dependency looks like" }
        }
        query crate_name(_: CrateNum) -> Symbol {
            eval_always
            desc { "fetching what a crate is named" }
        }
        query item_children(def_id: DefId) -> &'tcx [Export<hir::HirId>] {
            desc { |tcx| "collecting child items of `{}`", tcx.def_path_str(def_id) }
        }
        query extern_mod_stmt_cnum(def_id: LocalDefId) -> Option<CrateNum> {
            desc { |tcx| "computing crate imported by `{}`", tcx.def_path_str(def_id.to_def_id()) }
        }

        query get_lib_features(_: CrateNum) -> LibFeatures {
            storage(ArenaCacheSelector<'tcx>)
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
        query get_lang_items(_: CrateNum) -> LanguageItems {
            storage(ArenaCacheSelector<'tcx>)
            eval_always
            desc { "calculating the lang items map" }
        }

        /// Returns all diagnostic items defined in all crates.
        query all_diagnostic_items(_: CrateNum) -> FxHashMap<Symbol, DefId> {
            storage(ArenaCacheSelector<'tcx>)
            eval_always
            desc { "calculating the diagnostic items map" }
        }

        /// Returns the lang items defined in another crate by loading it from metadata.
        query defined_lang_items(_: CrateNum) -> &'tcx [(DefId, usize)] {
            desc { "calculating the lang items defined in a crate" }
        }

        /// Returns the diagnostic items defined in a crate.
        query diagnostic_items(_: CrateNum) -> FxHashMap<Symbol, DefId> {
            storage(ArenaCacheSelector<'tcx>)
            desc { "calculating the diagnostic items map in a crate" }
        }

        query missing_lang_items(_: CrateNum) -> &'tcx [LangItem] {
            desc { "calculating the missing lang items in a crate" }
        }
        query visible_parent_map(_: CrateNum)
            -> DefIdMap<DefId> {
            storage(ArenaCacheSelector<'tcx>)
            desc { "calculating the visible parent map" }
        }
        query trimmed_def_paths(_: CrateNum)
            -> FxHashMap<DefId, Symbol> {
            storage(ArenaCacheSelector<'tcx>)
            desc { "calculating trimmed def paths" }
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

        query upvars_mentioned(def_id: DefId) -> Option<&'tcx FxIndexMap<hir::HirId, hir::Upvar>> {
            desc { |tcx| "collecting upvars mentioned in `{}`", tcx.def_path_str(def_id) }
            eval_always
        }
        query maybe_unused_trait_import(def_id: LocalDefId) -> bool {
            eval_always
            desc { |tcx| "maybe_unused_trait_import for `{}`", tcx.def_path_str(def_id.to_def_id()) }
        }
        query maybe_unused_extern_crates(_: CrateNum)
            -> &'tcx [(LocalDefId, Span)] {
            eval_always
            desc { "looking up all possibly unused extern crates" }
        }
        query names_imported_by_glob_use(def_id: LocalDefId)
            -> &'tcx FxHashSet<Symbol> {
            eval_always
            desc { |tcx| "names_imported_by_glob_use for `{}`", tcx.def_path_str(def_id.to_def_id()) }
        }

        query stability_index(_: CrateNum) -> stability::Index<'tcx> {
            storage(ArenaCacheSelector<'tcx>)
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
            -> &'tcx [(ExportedSymbol<'tcx>, SymbolExportLevel)] {
            desc { "exported_symbols" }
        }
    }

    Codegen {
        query collect_and_partition_mono_items(_: CrateNum)
            -> (&'tcx DefIdSet, &'tcx [CodegenUnit<'tcx>]) {
            eval_always
            desc { "collect_and_partition_mono_items" }
        }
        query is_codegened_item(def_id: DefId) -> bool {
            desc { |tcx| "determining whether `{}` needs codegen", tcx.def_path_str(def_id) }
        }
        query codegen_unit(_: Symbol) -> &'tcx CodegenUnit<'tcx> {
            desc { "codegen_unit" }
        }
        query unused_generic_params(key: DefId) -> FiniteBitSet<u32> {
            cache_on_disk_if { key.is_local() }
            desc {
                |tcx| "determining which generic parameters are unused by `{}`",
                    tcx.def_path_str(key)
            }
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
            desc { "normalizing `{:?}`", goal }
        }

        /// Do not call this query directly: invoke `normalize_erasing_regions` instead.
        query normalize_generic_arg_after_erasing_regions(
            goal: ParamEnvAnd<'tcx, GenericArg<'tcx>>
        ) -> GenericArg<'tcx> {
            desc { "normalizing `{}`", goal.value }
        }

        query implied_outlives_bounds(
            goal: CanonicalTyGoal<'tcx>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, Vec<OutlivesBound<'tcx>>>>,
            NoSolution,
        > {
            desc { "computing implied outlives bounds for `{:?}`", goal }
        }

        /// Do not call this query directly: invoke `infcx.at().dropck_outlives()` instead.
        query dropck_outlives(
            goal: CanonicalTyGoal<'tcx>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, DropckOutlivesResult<'tcx>>>,
            NoSolution,
        > {
            desc { "computing dropck types for `{:?}`", goal }
        }

        /// Do not call this query directly: invoke `infcx.predicate_may_hold()` or
        /// `infcx.predicate_must_hold()` instead.
        query evaluate_obligation(
            goal: CanonicalPredicateGoal<'tcx>
        ) -> Result<traits::EvaluationResult, traits::OverflowError> {
            desc { "evaluating trait selection obligation `{}`", goal.value.value }
        }

        query evaluate_goal(
            goal: traits::CanonicalChalkEnvironmentAndGoal<'tcx>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, ()>>,
            NoSolution
        > {
            desc { "evaluating trait selection obligation `{}`", goal.value }
        }

        query type_implements_trait(
            key: (DefId, Ty<'tcx>, SubstsRef<'tcx>, ty::ParamEnv<'tcx>, )
        ) -> bool {
            desc { "evaluating `type_implements_trait` `{:?}`", key }
        }

        /// Do not call this query directly: part of the `Eq` type-op
        query type_op_ascribe_user_type(
            goal: CanonicalTypeOpAscribeUserTypeGoal<'tcx>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, ()>>,
            NoSolution,
        > {
            desc { "evaluating `type_op_ascribe_user_type` `{:?}`", goal }
        }

        /// Do not call this query directly: part of the `Eq` type-op
        query type_op_eq(
            goal: CanonicalTypeOpEqGoal<'tcx>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, ()>>,
            NoSolution,
        > {
            desc { "evaluating `type_op_eq` `{:?}`", goal }
        }

        /// Do not call this query directly: part of the `Subtype` type-op
        query type_op_subtype(
            goal: CanonicalTypeOpSubtypeGoal<'tcx>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, ()>>,
            NoSolution,
        > {
            desc { "evaluating `type_op_subtype` `{:?}`", goal }
        }

        /// Do not call this query directly: part of the `ProvePredicate` type-op
        query type_op_prove_predicate(
            goal: CanonicalTypeOpProvePredicateGoal<'tcx>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, ()>>,
            NoSolution,
        > {
            desc { "evaluating `type_op_prove_predicate` `{:?}`", goal }
        }

        /// Do not call this query directly: part of the `Normalize` type-op
        query type_op_normalize_ty(
            goal: CanonicalTypeOpNormalizeGoal<'tcx, Ty<'tcx>>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, Ty<'tcx>>>,
            NoSolution,
        > {
            desc { "normalizing `{:?}`", goal }
        }

        /// Do not call this query directly: part of the `Normalize` type-op
        query type_op_normalize_predicate(
            goal: CanonicalTypeOpNormalizeGoal<'tcx, ty::Predicate<'tcx>>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, ty::Predicate<'tcx>>>,
            NoSolution,
        > {
            desc { "normalizing `{:?}`", goal }
        }

        /// Do not call this query directly: part of the `Normalize` type-op
        query type_op_normalize_poly_fn_sig(
            goal: CanonicalTypeOpNormalizeGoal<'tcx, ty::PolyFnSig<'tcx>>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, ty::PolyFnSig<'tcx>>>,
            NoSolution,
        > {
            desc { "normalizing `{:?}`", goal }
        }

        /// Do not call this query directly: part of the `Normalize` type-op
        query type_op_normalize_fn_sig(
            goal: CanonicalTypeOpNormalizeGoal<'tcx, ty::FnSig<'tcx>>
        ) -> Result<
            &'tcx Canonical<'tcx, canonical::QueryResponse<'tcx, ty::FnSig<'tcx>>>,
            NoSolution,
        > {
            desc { "normalizing `{:?}`", goal }
        }

        query subst_and_check_impossible_predicates(key: (DefId, SubstsRef<'tcx>)) -> bool {
            desc { |tcx|
                "impossible substituted predicates:`{}`",
                tcx.def_path_str(key.0)
            }
        }

        query method_autoderef_steps(
            goal: CanonicalTyGoal<'tcx>
        ) -> MethodAutoderefStepsResult<'tcx> {
            desc { "computing autoderef types for `{:?}`", goal }
        }
    }

    Other {
        query supported_target_features(_: CrateNum) -> FxHashMap<String, Option<Symbol>> {
            storage(ArenaCacheSelector<'tcx>)
            eval_always
            desc { "looking up supported target features" }
        }

        /// Get an estimate of the size of an InstanceDef based on its MIR for CGU partitioning.
        query instance_def_size_estimate(def: ty::InstanceDef<'tcx>)
            -> usize {
            desc { |tcx| "estimating size for `{}`", tcx.def_path_str(def.def_id()) }
        }

        query features_query(_: CrateNum) -> &'tcx rustc_feature::Features {
            eval_always
            desc { "looking up enabled feature gates" }
        }

        /// Attempt to resolve the given `DefId` to an `Instance`, for the
        /// given generics args (`SubstsRef`), returning one of:
        ///  * `Ok(Some(instance))` on success
        ///  * `Ok(None)` when the `SubstsRef` are still too generic,
        ///    and therefore don't allow finding the final `Instance`
        ///  * `Err(ErrorReported)` when the `Instance` resolution process
        ///    couldn't complete due to errors elsewhere - this is distinct
        ///    from `Ok(None)` to avoid misleading diagnostics when an error
        ///    has already been/will be emitted, for the original cause
        query resolve_instance(
            key: ty::ParamEnvAnd<'tcx, (DefId, SubstsRef<'tcx>)>
        ) -> Result<Option<ty::Instance<'tcx>>, ErrorReported> {
            desc { "resolving instance `{}`", ty::Instance::new(key.value.0, key.value.1) }
        }

        query resolve_instance_of_const_arg(
            key: ty::ParamEnvAnd<'tcx, (LocalDefId, DefId, SubstsRef<'tcx>)>
        ) -> Result<Option<ty::Instance<'tcx>>, ErrorReported> {
            desc {
                "resolving instance of the const argument `{}`",
                ty::Instance::new(key.value.0.to_def_id(), key.value.2),
            }
        }

        query normalize_opaque_types(key: &'tcx ty::List<ty::Predicate<'tcx>>) -> &'tcx ty::List<ty::Predicate<'tcx>> {
            desc { "normalizing opaque types in {:?}", key }
        }
    }
}
