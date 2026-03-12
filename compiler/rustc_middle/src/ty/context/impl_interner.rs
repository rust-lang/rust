//! Implementation of [`rustc_type_ir::Interner`] for [`TyCtxt`].

use std::ops::Range;
use std::{debug_assert_matches, fmt};

use rustc_abi::{ExternAbi, FieldIdx, VariantIdx};
use rustc_data_structures::intern::Interned;
use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, CtorOf, DefKind};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::lang_items::LangItem;
use rustc_span::{DUMMY_SP, Span, Symbol};
use rustc_type_ir::lang_items::{SolverAdtLangItem, SolverLangItem, SolverTraitLangItem};
use rustc_type_ir::{
    CollectAndApply, Interner, TypeFoldable, WithCachedTypeInfo, elaborate, search_graph,
};

use crate::dep_graph::{DepKind, DepNodeIndex};
use crate::infer::canonical::CanonicalVarKinds;
use crate::query::IntoQueryParam;
use crate::traits::ObligationCause;
use crate::traits::cache::WithDepNode;
use crate::traits::solve::{
    self, CanonicalInput, ExternalConstraints, ExternalConstraintsData, QueryResult, inspect,
};
use crate::ty::util::Discr;
use crate::ty::{
    self, Clause, Const, GenericArgs, List, ParamTy, Pattern, PolyExistentialPredicate, Predicate,
    Region, Ty, TyCtxt,
};

#[allow(rustc::usage_of_ty_tykind)]
impl<'tcx> Interner for TyCtxt<'tcx> {
    fn next_trait_solver_globally(self) -> bool {
        self.next_trait_solver_globally()
    }

    type DefId = DefId;
    type LocalDefId = LocalDefId;
    type TraitId = DefId;
    type ForeignId = DefId;
    type FunctionId = DefId;
    type ClosureId = DefId;
    type CoroutineClosureId = DefId;
    type CoroutineId = DefId;
    type AdtId = DefId;
    type GenericArgs = ty::GenericArgsRef<'tcx>;
    type VariantDef = &'tcx ty::VariantDef;
    type ImplId = DefId;
    type UnevaluatedConstId = DefId;
    type Span = Span;
    type InternedTyKindWithCachedInfo = Interned<'tcx, WithCachedTypeInfo<ty::TyKind<'tcx>>>;
    type VariantIdx = VariantIdx;
    type Discr = Discr<'tcx>;
    type ObligationCause = ObligationCause<'tcx>;
    type LangItem = LangItem;

    type GenericArgsSlice = &'tcx [ty::GenericArg<'tcx>];
    type GenericArg = ty::GenericArg<'tcx>;
    type Term = ty::Term<'tcx>;
    type BoundVarKinds = &'tcx List<ty::BoundVariableKind<'tcx>>;

    type PredefinedOpaques = solve::PredefinedOpaques<'tcx>;

    fn mk_predefined_opaques_in_body(
        self,
        data: &[(ty::OpaqueTypeKey<'tcx>, Ty<'tcx>)],
    ) -> Self::PredefinedOpaques {
        self.mk_predefined_opaques_in_body(data)
    }
    type LocalDefIds = &'tcx ty::List<LocalDefId>;
    type CanonicalVarKinds = CanonicalVarKinds<'tcx>;
    fn mk_canonical_var_kinds(
        self,
        kinds: &[ty::CanonicalVarKind<Self>],
    ) -> Self::CanonicalVarKinds {
        self.mk_canonical_var_kinds(kinds)
    }

    type ExternalConstraints = ExternalConstraints<'tcx>;
    fn mk_external_constraints(
        self,
        data: ExternalConstraintsData<Self>,
    ) -> ExternalConstraints<'tcx> {
        self.mk_external_constraints(data)
    }
    type DepNodeIndex = DepNodeIndex;
    fn with_cached_task<T>(self, task: impl FnOnce() -> T) -> (T, DepNodeIndex) {
        self.dep_graph.with_anon_task(self, DepKind::TraitSelect, task)
    }
    type Tys = &'tcx List<Ty<'tcx>>;

    type FnInputTys = &'tcx [Ty<'tcx>];
    type ParamTy = ParamTy;
    type Symbol = Symbol;

    type ErrorGuaranteed = ErrorGuaranteed;
    type BoundExistentialPredicates = &'tcx List<PolyExistentialPredicate<'tcx>>;

    type AllocId = crate::mir::interpret::AllocId;
    type Pat = Pattern<'tcx>;
    type PatList = &'tcx List<Pattern<'tcx>>;
    type Safety = hir::Safety;
    type Abi = ExternAbi;
    type Const = ty::Const<'tcx>;
    type Consts = &'tcx List<Self::Const>;

    type ParamConst = ty::ParamConst;
    type ValueConst = ty::Value<'tcx>;
    type ExprConst = ty::Expr<'tcx>;
    type ValTree = ty::ValTree<'tcx>;
    type ScalarInt = ty::ScalarInt;

    type Region = Region<'tcx>;
    type EarlyParamRegion = ty::EarlyParamRegion;
    type LateParamRegion = ty::LateParamRegion;

    type RegionAssumptions = &'tcx ty::List<ty::ArgOutlivesPredicate<'tcx>>;

    type ParamEnv = ty::ParamEnv<'tcx>;
    type Predicate = Predicate<'tcx>;

    type Clause = Clause<'tcx>;
    type Clauses = ty::Clauses<'tcx>;

    type Tracked<T: fmt::Debug + Clone> = WithDepNode<T>;
    fn mk_tracked<T: fmt::Debug + Clone>(
        self,
        data: T,
        dep_node: DepNodeIndex,
    ) -> Self::Tracked<T> {
        WithDepNode::new(dep_node, data)
    }
    fn get_tracked<T: fmt::Debug + Clone>(self, tracked: &Self::Tracked<T>) -> T {
        tracked.get(self)
    }

    fn with_global_cache<R>(self, f: impl FnOnce(&mut search_graph::GlobalCache<Self>) -> R) -> R {
        f(&mut *self.new_solver_evaluation_cache.lock())
    }

    fn canonical_param_env_cache_get_or_insert<R>(
        self,
        param_env: ty::ParamEnv<'tcx>,
        f: impl FnOnce() -> ty::CanonicalParamEnvCacheEntry<Self>,
        from_entry: impl FnOnce(&ty::CanonicalParamEnvCacheEntry<Self>) -> R,
    ) -> R {
        let mut cache = self.new_solver_canonical_param_env_cache.lock();
        let entry = cache.entry(param_env).or_insert_with(f);
        from_entry(entry)
    }

    fn assert_evaluation_is_concurrent(&self) {
        // Turns out, the assumption for this function isn't perfect.
        // See trait-system-refactor-initiative#234.
    }

    fn expand_abstract_consts<T: TypeFoldable<TyCtxt<'tcx>>>(self, t: T) -> T {
        self.expand_abstract_consts(t)
    }

    type GenericsOf = &'tcx ty::Generics;

    fn generics_of(self, def_id: DefId) -> &'tcx ty::Generics {
        self.generics_of(def_id)
    }

    type VariancesOf = &'tcx [ty::Variance];

    fn variances_of(self, def_id: DefId) -> Self::VariancesOf {
        self.variances_of(def_id)
    }

    fn opt_alias_variances(
        self,
        kind: impl Into<ty::AliasTermKind>,
        def_id: DefId,
    ) -> Option<&'tcx [ty::Variance]> {
        self.opt_alias_variances(kind, def_id)
    }

    fn type_of(self, def_id: DefId) -> ty::EarlyBinder<'tcx, Ty<'tcx>> {
        self.type_of(def_id)
    }
    fn type_of_opaque_hir_typeck(self, def_id: LocalDefId) -> ty::EarlyBinder<'tcx, Ty<'tcx>> {
        self.type_of_opaque_hir_typeck(def_id)
    }
    fn const_of_item(self, def_id: DefId) -> ty::EarlyBinder<'tcx, Const<'tcx>> {
        self.const_of_item(def_id)
    }
    fn anon_const_kind(self, def_id: DefId) -> ty::AnonConstKind {
        self.anon_const_kind(def_id)
    }

    type AdtDef = ty::AdtDef<'tcx>;
    fn adt_def(self, adt_def_id: DefId) -> Self::AdtDef {
        self.adt_def(adt_def_id)
    }

    fn alias_ty_kind(self, alias: ty::AliasTy<'tcx>) -> ty::AliasTyKind {
        match self.def_kind(alias.def_id) {
            DefKind::AssocTy => {
                if let DefKind::Impl { of_trait: false } = self.def_kind(self.parent(alias.def_id))
                {
                    ty::Inherent
                } else {
                    ty::Projection
                }
            }
            DefKind::OpaqueTy => ty::Opaque,
            DefKind::TyAlias => ty::Free,
            kind => bug!("unexpected DefKind in AliasTy: {kind:?}"),
        }
    }

    fn alias_term_kind(self, alias: ty::AliasTerm<'tcx>) -> ty::AliasTermKind {
        match self.def_kind(alias.def_id) {
            DefKind::AssocTy => {
                if let DefKind::Impl { of_trait: false } = self.def_kind(self.parent(alias.def_id))
                {
                    ty::AliasTermKind::InherentTy
                } else {
                    ty::AliasTermKind::ProjectionTy
                }
            }
            DefKind::AssocConst { .. } => {
                if let DefKind::Impl { of_trait: false } = self.def_kind(self.parent(alias.def_id))
                {
                    ty::AliasTermKind::InherentConst
                } else {
                    ty::AliasTermKind::ProjectionConst
                }
            }
            DefKind::OpaqueTy => ty::AliasTermKind::OpaqueTy,
            DefKind::TyAlias => ty::AliasTermKind::FreeTy,
            DefKind::Const { .. } => ty::AliasTermKind::FreeConst,
            DefKind::AnonConst | DefKind::Ctor(_, CtorKind::Const) => {
                ty::AliasTermKind::UnevaluatedConst
            }
            kind => bug!("unexpected DefKind in AliasTy: {kind:?}"),
        }
    }

    fn trait_ref_and_own_args_for_alias(
        self,
        def_id: DefId,
        args: ty::GenericArgsRef<'tcx>,
    ) -> (ty::TraitRef<'tcx>, &'tcx [ty::GenericArg<'tcx>]) {
        debug_assert_matches!(self.def_kind(def_id), DefKind::AssocTy | DefKind::AssocConst { .. });
        let trait_def_id = self.parent(def_id);
        debug_assert_matches!(self.def_kind(trait_def_id), DefKind::Trait);
        let trait_ref = ty::TraitRef::from_assoc(self, trait_def_id, args);
        (trait_ref, &args[trait_ref.args.len()..])
    }

    fn mk_args(self, args: &[Self::GenericArg]) -> ty::GenericArgsRef<'tcx> {
        self.mk_args(args)
    }

    fn mk_args_from_iter<I, T>(self, args: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<Self::GenericArg, ty::GenericArgsRef<'tcx>>,
    {
        self.mk_args_from_iter(args)
    }

    fn check_args_compatible(self, def_id: DefId, args: ty::GenericArgsRef<'tcx>) -> bool {
        self.check_args_compatible(def_id, args)
    }

    fn debug_assert_args_compatible(self, def_id: DefId, args: ty::GenericArgsRef<'tcx>) {
        self.debug_assert_args_compatible(def_id, args);
    }

    fn debug_assert_adt_def_compatible(self, def: ty::AdtDef<'tcx>) {
        if cfg!(debug_assertions) {
            match self.def_kind(def.did()) {
                DefKind::Struct | DefKind::Union | DefKind::Enum => {}
                DefKind::Mod
                | DefKind::Variant
                | DefKind::Trait
                | DefKind::TyAlias
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
                | DefKind::Macro(..)
                | DefKind::ExternCrate
                | DefKind::Use
                | DefKind::ForeignMod
                | DefKind::AnonConst
                | DefKind::InlineConst
                | DefKind::OpaqueTy
                | DefKind::Field
                | DefKind::LifetimeParam
                | DefKind::GlobalAsm
                | DefKind::Impl { .. }
                | DefKind::Closure
                | DefKind::SyntheticCoroutineBody => {
                    bug!("not an adt: {def:?} ({:?})", self.def_kind(def.did()))
                }
            }
        }
    }

    /// Given the `DefId`, returns the `DefId` of the innermost item that
    /// has its own type-checking context or "inference environment".
    ///
    /// For example, a closure has its own `DefId`, but it is type-checked
    /// with the containing item. Therefore, when we fetch the `typeck` of the closure,
    /// for example, we really wind up fetching the `typeck` of the enclosing fn item.
    fn typeck_root_def_id(self, def_id: DefId) -> DefId {
        let mut def_id = def_id;
        while self.is_typeck_child(def_id) {
            def_id = self.parent(def_id);
        }
        def_id
    }

    fn debug_assert_new_dynamic_compatible(
        self,
        obj: &'tcx List<ty::PolyExistentialPredicate<'tcx>>,
    ) {
        if cfg!(debug_assertions) {
            let projection_count = obj
                .projection_bounds()
                .filter(|item| !self.generics_require_sized_self(item.item_def_id()))
                .count();
            let expected_count: usize = obj
                .principal_def_id()
                .into_iter()
                .flat_map(|principal_def_id| {
                    // IMPORTANT: This has to agree with HIR ty lowering of dyn trait!
                    elaborate::supertraits(
                        self,
                        ty::Binder::dummy(ty::TraitRef::identity(self, principal_def_id)),
                    )
                    .map(|principal| {
                        self.associated_items(principal.def_id())
                            .in_definition_order()
                            .filter(|item| item.is_type() || item.is_const())
                            .filter(|item| !item.is_impl_trait_in_trait())
                            .filter(|item| !self.generics_require_sized_self(item.def_id))
                            .count()
                    })
                })
                .sum();
            assert_eq!(
                projection_count, expected_count,
                "expected {obj:?} to have {expected_count} projections, \
                but it has {projection_count}"
            );
        }
    }

    /// Assert that the args from an `ExistentialTraitRef` or `ExistentialProjection`
    /// are compatible with the `DefId`. Since we're missing a `Self` type, stick on
    /// a dummy self type and forward to `debug_assert_args_compatible`.
    fn debug_assert_existential_args_compatible(
        self,
        def_id: Self::DefId,
        args: Self::GenericArgs,
    ) {
        // FIXME: We could perhaps add a `skip: usize` to `debug_assert_args_compatible`
        // to avoid needing to reintern the set of args...
        if cfg!(debug_assertions) {
            self.debug_assert_args_compatible(
                def_id,
                self.mk_args_from_iter(
                    [self.types.trait_object_dummy_self.into()].into_iter().chain(args.iter()),
                ),
            );
        }
    }

    fn mk_type_list_from_iter<I, T>(self, args: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<Ty<'tcx>, &'tcx List<Ty<'tcx>>>,
    {
        self.mk_type_list_from_iter(args)
    }

    fn mk_ty_from_kind(self, kind: ty::TyKind<'tcx>) -> Ty<'tcx> {
        self.mk_ty_from_kind(kind)
    }

    fn mk_coroutine_witness_for_coroutine(
        self,
        def_id: DefId,
        args: Self::GenericArgs,
    ) -> Ty<'tcx> {
        Ty::new_coroutine_witness_for_coroutine(self, def_id, args)
    }

    fn ty_discriminant_ty(self, ty: Ty<'tcx>) -> Ty<'tcx> {
        ty.discriminant_ty(self)
    }

    fn parent(self, def_id: DefId) -> DefId {
        self.parent(def_id)
    }

    fn recursion_limit(self) -> usize {
        self.recursion_limit().0
    }

    type Features = &'tcx rustc_feature::Features;

    fn features(self) -> Self::Features {
        self.features()
    }

    fn coroutine_hidden_types(
        self,
        def_id: DefId,
    ) -> ty::EarlyBinder<'tcx, ty::Binder<'tcx, ty::CoroutineWitnessTypes<TyCtxt<'tcx>>>> {
        self.coroutine_hidden_types(def_id)
    }

    fn fn_sig(self, def_id: DefId) -> ty::EarlyBinder<'tcx, ty::PolyFnSig<'tcx>> {
        self.fn_sig(def_id)
    }

    fn coroutine_movability(self, def_id: DefId) -> rustc_ast::Movability {
        self.coroutine_movability(def_id)
    }

    fn coroutine_for_closure(self, def_id: DefId) -> DefId {
        self.coroutine_for_closure(def_id)
    }

    fn generics_require_sized_self(self, def_id: DefId) -> bool {
        self.generics_require_sized_self(def_id)
    }

    fn item_bounds(
        self,
        def_id: DefId,
    ) -> ty::EarlyBinder<'tcx, impl IntoIterator<Item = ty::Clause<'tcx>>> {
        self.item_bounds(def_id).map_bound(IntoIterator::into_iter)
    }

    fn item_self_bounds(
        self,
        def_id: DefId,
    ) -> ty::EarlyBinder<'tcx, impl IntoIterator<Item = ty::Clause<'tcx>>> {
        self.item_self_bounds(def_id).map_bound(IntoIterator::into_iter)
    }

    fn item_non_self_bounds(
        self,
        def_id: DefId,
    ) -> ty::EarlyBinder<'tcx, impl IntoIterator<Item = ty::Clause<'tcx>>> {
        self.item_non_self_bounds(def_id).map_bound(IntoIterator::into_iter)
    }

    fn predicates_of(
        self,
        def_id: DefId,
    ) -> ty::EarlyBinder<'tcx, impl IntoIterator<Item = ty::Clause<'tcx>>> {
        ty::EarlyBinder::bind(
            self.predicates_of(def_id).instantiate_identity(self).predicates.into_iter(),
        )
    }

    fn own_predicates_of(
        self,
        def_id: DefId,
    ) -> ty::EarlyBinder<'tcx, impl IntoIterator<Item = ty::Clause<'tcx>>> {
        ty::EarlyBinder::bind(
            self.predicates_of(def_id).instantiate_own_identity().map(|(clause, _)| clause),
        )
    }

    fn explicit_super_predicates_of(
        self,
        def_id: DefId,
    ) -> ty::EarlyBinder<'tcx, impl IntoIterator<Item = (ty::Clause<'tcx>, Span)>> {
        self.explicit_super_predicates_of(def_id).map_bound(|preds| preds.into_iter().copied())
    }

    fn explicit_implied_predicates_of(
        self,
        def_id: DefId,
    ) -> ty::EarlyBinder<'tcx, impl IntoIterator<Item = (ty::Clause<'tcx>, Span)>> {
        self.explicit_implied_predicates_of(def_id).map_bound(|preds| preds.into_iter().copied())
    }

    fn impl_super_outlives(
        self,
        impl_def_id: DefId,
    ) -> ty::EarlyBinder<'tcx, impl IntoIterator<Item = ty::Clause<'tcx>>> {
        self.impl_super_outlives(impl_def_id)
    }

    fn impl_is_const(self, def_id: DefId) -> bool {
        debug_assert_matches!(self.def_kind(def_id), DefKind::Impl { of_trait: true });
        self.is_conditionally_const(def_id)
    }

    fn fn_is_const(self, def_id: DefId) -> bool {
        debug_assert_matches!(
            self.def_kind(def_id),
            DefKind::Fn | DefKind::AssocFn | DefKind::Ctor(CtorOf::Struct, CtorKind::Fn)
        );
        self.is_conditionally_const(def_id)
    }

    fn alias_has_const_conditions(self, def_id: DefId) -> bool {
        debug_assert_matches!(self.def_kind(def_id), DefKind::AssocTy | DefKind::OpaqueTy);
        self.is_conditionally_const(def_id)
    }

    fn const_conditions(
        self,
        def_id: DefId,
    ) -> ty::EarlyBinder<'tcx, impl IntoIterator<Item = ty::Binder<'tcx, ty::TraitRef<'tcx>>>> {
        ty::EarlyBinder::bind(
            self.const_conditions(def_id).instantiate_identity(self).into_iter().map(|(c, _)| c),
        )
    }

    fn explicit_implied_const_bounds(
        self,
        def_id: DefId,
    ) -> ty::EarlyBinder<'tcx, impl IntoIterator<Item = ty::Binder<'tcx, ty::TraitRef<'tcx>>>> {
        ty::EarlyBinder::bind(
            self.explicit_implied_const_bounds(def_id).iter_identity_copied().map(|(c, _)| c),
        )
    }

    fn impl_self_is_guaranteed_unsized(self, impl_def_id: DefId) -> bool {
        self.impl_self_is_guaranteed_unsized(impl_def_id)
    }

    fn has_target_features(self, def_id: DefId) -> bool {
        !self.codegen_fn_attrs(def_id).target_features.is_empty()
    }

    fn require_lang_item(self, lang_item: SolverLangItem) -> DefId {
        self.require_lang_item(solver_lang_item_to_lang_item(lang_item), DUMMY_SP)
    }

    fn require_trait_lang_item(self, lang_item: SolverTraitLangItem) -> DefId {
        self.require_lang_item(solver_trait_lang_item_to_lang_item(lang_item), DUMMY_SP)
    }

    fn require_adt_lang_item(self, lang_item: SolverAdtLangItem) -> DefId {
        self.require_lang_item(solver_adt_lang_item_to_lang_item(lang_item), DUMMY_SP)
    }

    fn is_lang_item(self, def_id: DefId, lang_item: SolverLangItem) -> bool {
        self.is_lang_item(def_id, solver_lang_item_to_lang_item(lang_item))
    }

    fn is_c_void(self, adt: ty::AdtDef<'tcx>) -> bool {
        self.is_c_void(adt)
    }

    fn is_trait_lang_item(self, def_id: DefId, lang_item: SolverTraitLangItem) -> bool {
        self.is_lang_item(def_id, solver_trait_lang_item_to_lang_item(lang_item))
    }

    fn is_adt_lang_item(self, def_id: DefId, lang_item: SolverAdtLangItem) -> bool {
        self.is_lang_item(def_id, solver_adt_lang_item_to_lang_item(lang_item))
    }

    fn is_default_trait(self, def_id: DefId) -> bool {
        self.is_default_trait(def_id)
    }

    fn is_sizedness_trait(self, def_id: DefId) -> bool {
        self.is_sizedness_trait(def_id)
    }

    fn as_lang_item(self, def_id: DefId) -> Option<SolverLangItem> {
        lang_item_to_solver_lang_item(self.lang_items().from_def_id(def_id)?)
    }

    fn as_trait_lang_item(self, def_id: DefId) -> Option<SolverTraitLangItem> {
        lang_item_to_solver_trait_lang_item(self.lang_items().from_def_id(def_id)?)
    }

    fn as_adt_lang_item(self, def_id: DefId) -> Option<SolverAdtLangItem> {
        lang_item_to_solver_adt_lang_item(self.lang_items().from_def_id(def_id)?)
    }

    fn associated_type_def_ids(self, def_id: DefId) -> impl IntoIterator<Item = DefId> {
        self.associated_items(def_id)
            .in_definition_order()
            .filter(|assoc_item| assoc_item.is_type())
            .map(|assoc_item| assoc_item.def_id)
    }

    // This implementation is a bit different from `TyCtxt::for_each_relevant_impl`,
    // since we want to skip over blanket impls for non-rigid aliases, and also we
    // only want to consider types that *actually* unify with float/int vars.
    fn for_each_relevant_impl(
        self,
        trait_def_id: DefId,
        self_ty: Ty<'tcx>,
        mut f: impl FnMut(DefId),
    ) {
        let tcx = self;
        let trait_impls = tcx.trait_impls_of(trait_def_id);
        let mut consider_impls_for_simplified_type = |simp| {
            if let Some(impls_for_type) = trait_impls.non_blanket_impls().get(&simp) {
                for &impl_def_id in impls_for_type {
                    f(impl_def_id);
                }
            }
        };

        match self_ty.kind() {
            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Adt(_, _)
            | ty::Foreign(_)
            | ty::Str
            | ty::Array(_, _)
            | ty::Pat(_, _)
            | ty::Slice(_)
            | ty::RawPtr(_, _)
            | ty::Ref(_, _, _)
            | ty::FnDef(_, _)
            | ty::FnPtr(..)
            | ty::Dynamic(_, _)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(_, _)
            | ty::Never
            | ty::Tuple(_)
            | ty::UnsafeBinder(_) => {
                if let Some(simp) = ty::fast_reject::simplify_type(
                    tcx,
                    self_ty,
                    ty::fast_reject::TreatParams::AsRigid,
                ) {
                    consider_impls_for_simplified_type(simp);
                }
            }

            // HACK: For integer and float variables we have to manually look at all impls
            // which have some integer or float as a self type.
            ty::Infer(ty::IntVar(_)) => {
                use ty::IntTy::*;
                use ty::UintTy::*;
                // This causes a compiler error if any new integer kinds are added.
                let (I8 | I16 | I32 | I64 | I128 | Isize): ty::IntTy;
                let (U8 | U16 | U32 | U64 | U128 | Usize): ty::UintTy;
                let possible_integers = [
                    // signed integers
                    ty::SimplifiedType::Int(I8),
                    ty::SimplifiedType::Int(I16),
                    ty::SimplifiedType::Int(I32),
                    ty::SimplifiedType::Int(I64),
                    ty::SimplifiedType::Int(I128),
                    ty::SimplifiedType::Int(Isize),
                    // unsigned integers
                    ty::SimplifiedType::Uint(U8),
                    ty::SimplifiedType::Uint(U16),
                    ty::SimplifiedType::Uint(U32),
                    ty::SimplifiedType::Uint(U64),
                    ty::SimplifiedType::Uint(U128),
                    ty::SimplifiedType::Uint(Usize),
                ];
                for simp in possible_integers {
                    consider_impls_for_simplified_type(simp);
                }
            }

            ty::Infer(ty::FloatVar(_)) => {
                // This causes a compiler error if any new float kinds are added.
                let (ty::FloatTy::F16 | ty::FloatTy::F32 | ty::FloatTy::F64 | ty::FloatTy::F128);
                let possible_floats = [
                    ty::SimplifiedType::Float(ty::FloatTy::F16),
                    ty::SimplifiedType::Float(ty::FloatTy::F32),
                    ty::SimplifiedType::Float(ty::FloatTy::F64),
                    ty::SimplifiedType::Float(ty::FloatTy::F128),
                ];

                for simp in possible_floats {
                    consider_impls_for_simplified_type(simp);
                }
            }

            // The only traits applying to aliases and placeholders are blanket impls.
            //
            // Impls which apply to an alias after normalization are handled by
            // `assemble_candidates_after_normalizing_self_ty`.
            ty::Alias(_, _) | ty::Placeholder(..) | ty::Error(_) => (),

            // FIXME: These should ideally not exist as a self type. It would be nice for
            // the builtin auto trait impls of coroutines to instead directly recurse
            // into the witness.
            ty::CoroutineWitness(..) => (),

            // These variants should not exist as a self type.
            ty::Infer(ty::TyVar(_) | ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_))
            | ty::Param(_)
            | ty::Bound(_, _) => bug!("unexpected self type: {self_ty}"),
        }

        #[allow(rustc::usage_of_type_ir_traits)]
        self.for_each_blanket_impl(trait_def_id, f)
    }
    fn for_each_blanket_impl(self, trait_def_id: DefId, mut f: impl FnMut(DefId)) {
        let trait_impls = self.trait_impls_of(trait_def_id);
        for &impl_def_id in trait_impls.blanket_impls() {
            f(impl_def_id);
        }
    }

    fn has_item_definition(self, def_id: DefId) -> bool {
        self.defaultness(def_id).has_value()
    }

    fn impl_specializes(self, impl_def_id: Self::DefId, victim_def_id: Self::DefId) -> bool {
        self.specializes((impl_def_id, victim_def_id))
    }

    fn impl_is_default(self, impl_def_id: DefId) -> bool {
        self.defaultness(impl_def_id).is_default()
    }

    fn impl_trait_ref(self, impl_def_id: DefId) -> ty::EarlyBinder<'tcx, ty::TraitRef<'tcx>> {
        self.impl_trait_ref(impl_def_id)
    }

    fn impl_polarity(self, impl_def_id: DefId) -> ty::ImplPolarity {
        self.impl_polarity(impl_def_id)
    }

    fn trait_is_auto(self, trait_def_id: DefId) -> bool {
        self.trait_is_auto(trait_def_id)
    }

    fn trait_is_coinductive(self, trait_def_id: DefId) -> bool {
        self.trait_is_coinductive(trait_def_id)
    }

    fn trait_is_alias(self, trait_def_id: DefId) -> bool {
        self.trait_is_alias(trait_def_id)
    }

    fn trait_is_dyn_compatible(self, trait_def_id: DefId) -> bool {
        self.is_dyn_compatible(trait_def_id)
    }

    fn trait_is_fundamental(self, def_id: DefId) -> bool {
        self.trait_def(def_id).is_fundamental
    }

    fn trait_is_unsafe(self, trait_def_id: Self::DefId) -> bool {
        self.trait_def(trait_def_id).safety.is_unsafe()
    }

    fn is_impl_trait_in_trait(self, def_id: DefId) -> bool {
        self.is_impl_trait_in_trait(def_id)
    }

    fn delay_bug(self, msg: impl ToString) -> ErrorGuaranteed {
        self.dcx().span_delayed_bug(DUMMY_SP, msg.to_string())
    }

    fn is_general_coroutine(self, coroutine_def_id: DefId) -> bool {
        self.is_general_coroutine(coroutine_def_id)
    }

    fn coroutine_is_async(self, coroutine_def_id: DefId) -> bool {
        self.coroutine_is_async(coroutine_def_id)
    }

    fn coroutine_is_gen(self, coroutine_def_id: DefId) -> bool {
        self.coroutine_is_gen(coroutine_def_id)
    }

    fn coroutine_is_async_gen(self, coroutine_def_id: DefId) -> bool {
        self.coroutine_is_async_gen(coroutine_def_id)
    }

    type UnsizingParams = &'tcx rustc_index::bit_set::DenseBitSet<u32>;
    fn unsizing_params_for_adt(self, adt_def_id: DefId) -> Self::UnsizingParams {
        self.unsizing_params_for_adt(adt_def_id)
    }

    fn anonymize_bound_vars<T: TypeFoldable<TyCtxt<'tcx>>>(
        self,
        binder: ty::Binder<'tcx, T>,
    ) -> ty::Binder<'tcx, T> {
        self.anonymize_bound_vars(binder)
    }

    fn opaque_types_defined_by(self, defining_anchor: LocalDefId) -> Self::LocalDefIds {
        self.opaque_types_defined_by(defining_anchor)
    }

    fn opaque_types_and_coroutines_defined_by(
        self,
        defining_anchor: Self::LocalDefId,
    ) -> Self::LocalDefIds {
        let coroutines_defined_by = self
            .nested_bodies_within(defining_anchor)
            .iter()
            .filter(|def_id| self.is_coroutine(def_id.to_def_id()));
        self.mk_local_def_ids_from_iter(
            self.opaque_types_defined_by(defining_anchor).iter().chain(coroutines_defined_by),
        )
    }

    type Probe = &'tcx inspect::Probe<TyCtxt<'tcx>>;
    fn mk_probe(self, probe: inspect::Probe<Self>) -> &'tcx inspect::Probe<TyCtxt<'tcx>> {
        self.arena.alloc(probe)
    }
    fn evaluate_root_goal_for_proof_tree_raw(
        self,
        canonical_goal: CanonicalInput<'tcx>,
    ) -> (QueryResult<'tcx>, &'tcx inspect::Probe<TyCtxt<'tcx>>) {
        self.evaluate_root_goal_for_proof_tree_raw(canonical_goal)
    }

    fn item_name(self, id: DefId) -> Symbol {
        let id = id.into_query_param();
        self.opt_item_name(id).unwrap_or_else(|| {
            bug!("item_name: no name for {:?}", self.def_path(id));
        })
    }

    #[inline]
    fn i8_type(self) -> Ty<'tcx> {
        self.types.i8
    }

    #[inline]
    fn i16_type(self) -> Ty<'tcx> {
        self.types.i16
    }

    #[inline]
    fn i32_type(self) -> Ty<'tcx> {
        self.types.i32
    }

    #[inline]
    fn u8_type(self) -> Ty<'tcx> {
        self.types.u8
    }

    #[inline]
    fn usize_type(self) -> Ty<'tcx> {
        self.types.usize
    }

    #[inline]
    fn unit_type(self) -> Ty<'tcx> {
        self.types.unit
    }

    fn is_async_drop_in_place_coroutine(self, def_id: DefId) -> bool {
        self.is_async_drop_in_place_coroutine(def_id)
    }

    #[inline]
    fn coroutine_variant_range(
        self,
        def_id: DefId,
        coroutine_args: ty::CoroutineArgs<Self>,
    ) -> Range<VariantIdx> {
        self.coroutine_variant_range(def_id, coroutine_args)
    }

    fn struct_tail_raw(
        self,
        mut ty: Ty<'tcx>,
        cause: &ObligationCause<'tcx>,
        mut normalize: impl FnMut(Ty<'tcx>) -> Ty<'tcx>,
        mut f: impl FnMut() -> (),
    ) -> Ty<'tcx> {
        self.struct_tail_raw(ty, cause, normalize, f)
    }

    fn get_ty_var(self, id: usize) -> Option<Ty<'tcx>> {
        self.types.ty_vars.get(id).copied()
    }

    fn get_fresh_ty(self, id: usize) -> Option<Ty<'tcx>> {
        self.types.fresh_tys.get(id).copied()
    }

    fn get_fresh_ty_int(self, id: usize) -> Option<Ty<'tcx>> {
        self.types.fresh_int_tys.get(id).copied()
    }

    fn get_fresh_ty_float(self, id: usize) -> Option<Ty<'tcx>> {
        self.types.fresh_float_tys.get(id).copied()
    }

    fn get_anon_bound_ty(self, id: usize) -> Option<Vec<Ty<'tcx>>> {
        self.types.anon_bound_tys.get(id).copied()
    }

    fn get_anon_canonical_bound_ty(self, id: usize) -> Option<Ty<'tcx>> {
        self.types.anon_canonical_bound_tys.get(id.as_usize()).copied()
    }

    fn get_generic_args_for_item(
        self,
        def_id: DefId,
        coroutine_args: ty::GenericArgsRef<'tcx>,
    ) -> ty::GenericArgsRef<'tcx> {
        // HACK: Coroutine witness types are lifetime erased, so they
        // never reference any lifetime args from the coroutine. We erase
        // the regions here since we may get into situations where a
        // coroutine is recursively contained within itself, leading to
        // witness types that differ by region args. This means that
        // cycle detection in fulfillment will not kick in, which leads
        // to unnecessary overflows in async code. See the issue:
        // <https://github.com/rust-lang/rust/issues/145151>.
        ty::GenericArgs::for_item(self, self.typeck_root_def_id(def_id), |def, _| match def.kind {
            ty::GenericParamDefKind::Lifetime => self.lifetimes.re_erased.into(),
            ty::GenericParamDefKind::Type { .. } | ty::GenericParamDefKind::Const { .. } => {
                coroutine_args[def.index as usize]
            }
        })
    }

    fn get_generic_adt_args(
        self,
        wrapper_def_id: DefId,
        ty_param: Ty<'tcx>,
    ) -> ty::GenericArgsRef<'tcx> {
        GenericArgs::for_item(self, wrapper_def_id, |param, args| match param.kind {
            ty::GenericParamDefKind::Lifetime | ty::GenericParamDefKind::Const { .. } => bug!(),
            ty::GenericParamDefKind::Type { has_default, .. } => {
                if param.index == 0 {
                    ty_param.into()
                } else {
                    assert!(has_default);
                    self.type_of(param.def_id).instantiate(self, args).into()
                }
            }
        })
    }

    fn new_static_str(self) -> Ty<'tcx> {
        Ty::new_imm_ref(self, self.lifetimes.re_static, self.types.str_)
    }

    fn get_lang_item(self, item: LangItem) -> Option<DefId> {
        self.lang_items().get(item)
    }

    fn get_diagnostic_item(self, name: Symbol) -> Option<DefId> {
        self.get_diagnostic_item(name)
    }

    fn require_lang_item_owned_box(self) -> DefId {
        self.require_lang_item(LangItem::OwnedBox, DUMMY_SP)
    }

    fn require_lang_item_option(self) -> DefId {
        self.require_lang_item(LangItem::Option, DUMMY_SP)
    }

    fn require_lang_item_maybe_uninit(self) -> DefId {
        self.require_lang_item(LangItem::MaybeUninit, DUMMY_SP)
    }

    fn require_lang_item_pin(self) -> DefId {
        self.require_lang_item(LangItem::Pin, DUMMY_SP)
    }

    fn require_lang_item_context(self) -> DefId {
        self.require_lang_item(LangItem::Context, DUMMY_SP)
    }

    fn new_field_representing_type(
        self,
        base: Ty<'tcx>,
        variant: VariantIdx,
        field: FieldIdx,
    ) -> Ty<'tcx> {
        let Some(did) = self.lang_items().field_representing_type() else {
            bug!("could not locate the `FieldRepresentingType` lang item")
        };
        let def = self.adt_def(did);
        let args = self.mk_args(&[
            base.into(),
            Const::new_value(
                self,
                ty::ValTree::from_scalar_int(self, variant.as_u32().into()),
                self.types.u32,
            )
            .into(),
            Const::new_value(
                self,
                ty::ValTree::from_scalar_int(self, field.as_u32().into()),
                self.types.u32,
            )
            .into(),
        ]);
        Ty::new_adt(self, def, args)
    }

    fn get_re_erased_region(self) -> Region<'tcx> {
        self.lifetimes.re_erased
    }

    fn new_generic_adt(self, wrapper_def_id: DefId, ty_param: Ty<'tcx>) -> Ty<'tcx> {
        let adt_def = self.adt_def(wrapper_def_id);
        let args = GenericArgs::for_item(self, wrapper_def_id, |param, args| match param.kind {
            ty::GenericParamDefKind::Lifetime | ty::GenericParamDefKind::Const { .. } => bug!(),
            ty::GenericParamDefKind::Type { has_default, .. } => {
                if param.index == 0 {
                    ty_param.into()
                } else {
                    assert!(has_default);
                    self.type_of(param.def_id).instantiate(self, args).into()
                }
            }
        });
        Ty::new_adt(self, adt_def, args)
    }

    fn new_pinned_ref(self, r: Region<'tcx>, ty: Ty<'tcx>, mutbl: ty::Mutability) -> Ty<'tcx> {
        let pin = self.adt_def(self.require_lang_item(LangItem::Pin, DUMMY_SP));
        Ty::new_adt(self, pin, self.mk_args(&[Ty::new_ref(self, r, ty, mutbl).into()]))
    }

    fn new_task_context(self) -> Ty<'tcx> {
        let context_did = self.require_lang_item(LangItem::Context, DUMMY_SP);
        let context_adt_ref = self.adt_def(context_did);
        let context_args = self.mk_args(&[self.lifetimes.re_erased.into()]);
        let context_ty = Ty::new_adt(self, context_adt_ref, context_args);
        Ty::new_mut_ref(self, self.lifetimes.re_erased, context_ty)
    }
}

/// Defines trivial conversion functions between the main [`LangItem`] enum,
/// and some other lang-item enum that is a subset of it.
macro_rules! bidirectional_lang_item_map {
    (
        $solver_ty:ident, fn $to_solver:ident, fn $from_solver:ident;
        $($name:ident),+ $(,)?
    ) => {
        fn $from_solver(lang_item: $solver_ty) -> LangItem {
            match lang_item {
                $($solver_ty::$name => LangItem::$name,)+
            }
        }

        fn $to_solver(lang_item: LangItem) -> Option<$solver_ty> {
            Some(match lang_item {
                $(LangItem::$name => $solver_ty::$name,)+
                _ => return None,
            })
        }
    }
}

bidirectional_lang_item_map! {
    SolverLangItem, fn lang_item_to_solver_lang_item, fn solver_lang_item_to_lang_item;

// tidy-alphabetical-start
    AsyncFnKindUpvars,
    AsyncFnOnceOutput,
    CallOnceFuture,
    CallRefFuture,
    CoroutineReturn,
    CoroutineYield,
    DynMetadata,
    FieldBase,
    FieldType,
    FutureOutput,
    GlobalAlloc,
    Metadata,
// tidy-alphabetical-end
}

bidirectional_lang_item_map! {
    SolverAdtLangItem, fn lang_item_to_solver_adt_lang_item, fn solver_adt_lang_item_to_lang_item;

// tidy-alphabetical-start
    Option,
    Poll,
// tidy-alphabetical-end
}

bidirectional_lang_item_map! {
    SolverTraitLangItem, fn lang_item_to_solver_trait_lang_item, fn solver_trait_lang_item_to_lang_item;

// tidy-alphabetical-start
    AsyncFn,
    AsyncFnKindHelper,
    AsyncFnMut,
    AsyncFnOnce,
    AsyncFnOnceOutput,
    AsyncIterator,
    BikeshedGuaranteedNoDrop,
    Clone,
    Copy,
    Coroutine,
    Destruct,
    DiscriminantKind,
    Drop,
    Field,
    Fn,
    FnMut,
    FnOnce,
    FnPtrTrait,
    FusedIterator,
    Future,
    Iterator,
    MetaSized,
    PointeeSized,
    PointeeTrait,
    Sized,
    TransmuteTrait,
    TrivialClone,
    Tuple,
    Unpin,
    Unsize,
// tidy-alphabetical-end
}
