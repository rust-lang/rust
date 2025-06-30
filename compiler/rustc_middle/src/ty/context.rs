//! Type context book-keeping.

#![allow(rustc::usage_of_ty_tykind)]

pub mod tls;

use std::assert_matches::debug_assert_matches;
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::env::VarError;
use std::ffi::OsStr;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Bound, Deref};
use std::sync::{Arc, OnceLock};
use std::{fmt, iter, mem};

use rustc_abi::{ExternAbi, FieldIdx, Layout, LayoutData, TargetDataLayout, VariantIdx};
use rustc_ast as ast;
use rustc_data_structures::defer;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::intern::Interned;
use rustc_data_structures::jobserver::Proxy;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_data_structures::sharded::{IntoPointer, ShardedHashMap};
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::steal::Steal;
use rustc_data_structures::sync::{
    self, DynSend, DynSync, FreezeReadGuard, Lock, RwLock, WorkerLocal,
};
use rustc_errors::{
    Applicability, Diag, DiagCtxtHandle, ErrorGuaranteed, LintDiagnostic, LintEmitter, MultiSpan,
};
use rustc_hir::def::{CtorKind, DefKind};
use rustc_hir::def_id::{CrateNum, DefId, LOCAL_CRATE, LocalDefId};
use rustc_hir::definitions::{DefPathData, Definitions, DisambiguatorState};
use rustc_hir::intravisit::VisitorExt;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{self as hir, Attribute, HirId, Node, TraitCandidate};
use rustc_index::IndexVec;
use rustc_macros::{HashStable, TyDecodable, TyEncodable};
use rustc_query_system::cache::WithDepNode;
use rustc_query_system::dep_graph::DepNodeIndex;
use rustc_query_system::ich::StableHashingContext;
use rustc_serialize::PointeeSized;
use rustc_serialize::opaque::{FileEncodeResult, FileEncoder};
use rustc_session::config::CrateType;
use rustc_session::cstore::{CrateStoreDyn, Untracked};
use rustc_session::lint::Lint;
use rustc_session::{Limit, Session};
use rustc_span::def_id::{CRATE_DEF_ID, DefPathHash, StableCrateId};
use rustc_span::{DUMMY_SP, Ident, Span, Symbol, kw, sym};
use rustc_type_ir::TyKind::*;
use rustc_type_ir::lang_items::TraitSolverLangItem;
pub use rustc_type_ir::lift::Lift;
use rustc_type_ir::{
    CollectAndApply, Interner, TypeFlags, TypeFoldable, WithCachedTypeInfo, elaborate, search_graph,
};
use tracing::{debug, instrument};

use crate::arena::Arena;
use crate::dep_graph::{DepGraph, DepKindStruct};
use crate::infer::canonical::{CanonicalParamEnvCache, CanonicalVarKind, CanonicalVarKinds};
use crate::lint::lint_level;
use crate::metadata::ModChild;
use crate::middle::codegen_fn_attrs::{CodegenFnAttrs, TargetFeature};
use crate::middle::{resolve_bound_vars, stability};
use crate::mir::interpret::{self, Allocation, ConstAllocation};
use crate::mir::{Body, Local, Place, PlaceElem, ProjectionKind, Promoted};
use crate::query::plumbing::QuerySystem;
use crate::query::{IntoQueryParam, LocalCrate, Providers, TyCtxtAt};
use crate::thir::Thir;
use crate::traits;
use crate::traits::solve;
use crate::traits::solve::{
    ExternalConstraints, ExternalConstraintsData, PredefinedOpaques, PredefinedOpaquesData,
};
use crate::ty::predicate::ExistentialPredicateStableCmpExt as _;
use crate::ty::{
    self, AdtDef, AdtDefData, AdtKind, Binder, Clause, Clauses, Const, GenericArg, GenericArgs,
    GenericArgsRef, GenericParamDefKind, List, ListWithCachedTypeInfo, ParamConst, ParamTy,
    Pattern, PatternKind, PolyExistentialPredicate, PolyFnSig, Predicate, PredicateKind,
    PredicatePolarity, Region, RegionKind, ReprOptions, TraitObjectVisitor, Ty, TyKind, TyVid,
    ValTree, ValTreeKind, Visibility,
};

#[allow(rustc::usage_of_ty_tykind)]
impl<'tcx> Interner for TyCtxt<'tcx> {
    fn next_trait_solver_globally(self) -> bool {
        self.next_trait_solver_globally()
    }

    type DefId = DefId;
    type LocalDefId = LocalDefId;
    type Span = Span;

    type GenericArgs = ty::GenericArgsRef<'tcx>;

    type GenericArgsSlice = &'tcx [ty::GenericArg<'tcx>];
    type GenericArg = ty::GenericArg<'tcx>;
    type Term = ty::Term<'tcx>;
    type BoundVarKinds = &'tcx List<ty::BoundVariableKind>;

    type BoundVarKind = ty::BoundVariableKind;
    type PredefinedOpaques = solve::PredefinedOpaques<'tcx>;

    fn mk_predefined_opaques_in_body(
        self,
        data: PredefinedOpaquesData<Self>,
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
        self.dep_graph.with_anon_task(self, crate::dep_graph::dep_kinds::TraitSelect, task)
    }
    type Ty = Ty<'tcx>;
    type Tys = &'tcx List<Ty<'tcx>>;

    type FnInputTys = &'tcx [Ty<'tcx>];
    type ParamTy = ParamTy;
    type BoundTy = ty::BoundTy;

    type PlaceholderTy = ty::PlaceholderType;
    type ErrorGuaranteed = ErrorGuaranteed;
    type BoundExistentialPredicates = &'tcx List<PolyExistentialPredicate<'tcx>>;

    type AllocId = crate::mir::interpret::AllocId;
    type Pat = Pattern<'tcx>;
    type PatList = &'tcx List<Pattern<'tcx>>;
    type Safety = hir::Safety;
    type Abi = ExternAbi;
    type Const = ty::Const<'tcx>;
    type PlaceholderConst = ty::PlaceholderConst;

    type ParamConst = ty::ParamConst;
    type BoundConst = ty::BoundVar;
    type ValueConst = ty::Value<'tcx>;
    type ExprConst = ty::Expr<'tcx>;
    type ValTree = ty::ValTree<'tcx>;

    type Region = Region<'tcx>;
    type EarlyParamRegion = ty::EarlyParamRegion;
    type LateParamRegion = ty::LateParamRegion;
    type BoundRegion = ty::BoundRegion;
    type PlaceholderRegion = ty::PlaceholderRegion;

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

    fn evaluation_is_concurrent(&self) -> bool {
        self.sess.threads() > 1
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
            DefKind::AssocConst => {
                if let DefKind::Impl { of_trait: false } = self.def_kind(self.parent(alias.def_id))
                {
                    ty::AliasTermKind::InherentConst
                } else {
                    ty::AliasTermKind::ProjectionConst
                }
            }
            DefKind::OpaqueTy => ty::AliasTermKind::OpaqueTy,
            DefKind::TyAlias => ty::AliasTermKind::FreeTy,
            DefKind::Const => ty::AliasTermKind::FreeConst,
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
        debug_assert_matches!(self.def_kind(def_id), DefKind::AssocTy | DefKind::AssocConst);
        let trait_def_id = self.parent(def_id);
        debug_assert_matches!(self.def_kind(trait_def_id), DefKind::Trait);
        let trait_generics = self.generics_of(trait_def_id);
        (
            ty::TraitRef::new_from_args(self, trait_def_id, args.truncate_to(self, trait_generics)),
            &args[trait_generics.count()..],
        )
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
        debug_assert_matches!(self.def_kind(def_id), DefKind::Fn | DefKind::AssocFn);
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

    fn require_lang_item(self, lang_item: TraitSolverLangItem) -> DefId {
        self.require_lang_item(trait_lang_item_to_lang_item(lang_item), DUMMY_SP)
    }

    fn is_lang_item(self, def_id: DefId, lang_item: TraitSolverLangItem) -> bool {
        self.is_lang_item(def_id, trait_lang_item_to_lang_item(lang_item))
    }

    fn is_default_trait(self, def_id: DefId) -> bool {
        self.is_default_trait(def_id)
    }

    fn as_lang_item(self, def_id: DefId) -> Option<TraitSolverLangItem> {
        lang_item_to_trait_lang_item(self.lang_items().from_def_id(def_id)?)
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
            | ty::Dynamic(_, _, _)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(_, _)
            | ty::Never
            | ty::Tuple(_)
            | ty::UnsafeBinder(_) => {
                let simp = ty::fast_reject::simplify_type(
                    tcx,
                    self_ty,
                    ty::fast_reject::TreatParams::AsRigid,
                )
                .unwrap();
                consider_impls_for_simplified_type(simp);
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

        let trait_impls = tcx.trait_impls_of(trait_def_id);
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
        self.impl_trait_ref(impl_def_id).unwrap()
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

    fn trait_may_be_implemented_via_object(self, trait_def_id: DefId) -> bool {
        self.trait_def(trait_def_id).implement_via_object
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
        if self.next_trait_solver_globally() {
            let coroutines_defined_by = self
                .nested_bodies_within(defining_anchor)
                .iter()
                .filter(|def_id| self.is_coroutine(def_id.to_def_id()));
            self.mk_local_def_ids_from_iter(
                self.opaque_types_defined_by(defining_anchor).iter().chain(coroutines_defined_by),
            )
        } else {
            self.opaque_types_defined_by(defining_anchor)
        }
    }
}

macro_rules! bidirectional_lang_item_map {
    ($($name:ident),+ $(,)?) => {
        fn trait_lang_item_to_lang_item(lang_item: TraitSolverLangItem) -> LangItem {
            match lang_item {
                $(TraitSolverLangItem::$name => LangItem::$name,)+
            }
        }

        fn lang_item_to_trait_lang_item(lang_item: LangItem) -> Option<TraitSolverLangItem> {
            Some(match lang_item {
                $(LangItem::$name => TraitSolverLangItem::$name,)+
                _ => return None,
            })
        }
    }
}

bidirectional_lang_item_map! {
// tidy-alphabetical-start
    AsyncFn,
    AsyncFnKindHelper,
    AsyncFnKindUpvars,
    AsyncFnMut,
    AsyncFnOnce,
    AsyncFnOnceOutput,
    AsyncIterator,
    BikeshedGuaranteedNoDrop,
    CallOnceFuture,
    CallRefFuture,
    Clone,
    Copy,
    Coroutine,
    CoroutineReturn,
    CoroutineYield,
    Destruct,
    DiscriminantKind,
    Drop,
    DynMetadata,
    Fn,
    FnMut,
    FnOnce,
    FnPtrTrait,
    FusedIterator,
    Future,
    FutureOutput,
    Iterator,
    MetaSized,
    Metadata,
    Option,
    PointeeSized,
    PointeeTrait,
    Poll,
    Sized,
    TransmuteTrait,
    Tuple,
    Unpin,
    Unsize,
// tidy-alphabetical-end
}

impl<'tcx> rustc_type_ir::inherent::DefId<TyCtxt<'tcx>> for DefId {
    fn is_local(self) -> bool {
        self.is_local()
    }

    fn as_local(self) -> Option<LocalDefId> {
        self.as_local()
    }
}

impl<'tcx> rustc_type_ir::inherent::Abi<TyCtxt<'tcx>> for ExternAbi {
    fn rust() -> Self {
        ExternAbi::Rust
    }

    fn is_rust(self) -> bool {
        matches!(self, ExternAbi::Rust)
    }
}

impl<'tcx> rustc_type_ir::inherent::Safety<TyCtxt<'tcx>> for hir::Safety {
    fn safe() -> Self {
        hir::Safety::Safe
    }

    fn is_safe(self) -> bool {
        self.is_safe()
    }

    fn prefix_str(self) -> &'static str {
        self.prefix_str()
    }
}

impl<'tcx> rustc_type_ir::inherent::Features<TyCtxt<'tcx>> for &'tcx rustc_feature::Features {
    fn generic_const_exprs(self) -> bool {
        self.generic_const_exprs()
    }

    fn coroutine_clone(self) -> bool {
        self.coroutine_clone()
    }

    fn associated_const_equality(self) -> bool {
        self.associated_const_equality()
    }
}

impl<'tcx> rustc_type_ir::inherent::Span<TyCtxt<'tcx>> for Span {
    fn dummy() -> Self {
        DUMMY_SP
    }
}

type InternedSet<'tcx, T> = ShardedHashMap<InternedInSet<'tcx, T>, ()>;

pub struct CtxtInterners<'tcx> {
    /// The arena that types, regions, etc. are allocated from.
    arena: &'tcx WorkerLocal<Arena<'tcx>>,

    // Specifically use a speedy hash algorithm for these hash sets, since
    // they're accessed quite often.
    type_: InternedSet<'tcx, WithCachedTypeInfo<TyKind<'tcx>>>,
    const_lists: InternedSet<'tcx, List<ty::Const<'tcx>>>,
    args: InternedSet<'tcx, GenericArgs<'tcx>>,
    type_lists: InternedSet<'tcx, List<Ty<'tcx>>>,
    canonical_var_kinds: InternedSet<'tcx, List<CanonicalVarKind<'tcx>>>,
    region: InternedSet<'tcx, RegionKind<'tcx>>,
    poly_existential_predicates: InternedSet<'tcx, List<PolyExistentialPredicate<'tcx>>>,
    predicate: InternedSet<'tcx, WithCachedTypeInfo<ty::Binder<'tcx, PredicateKind<'tcx>>>>,
    clauses: InternedSet<'tcx, ListWithCachedTypeInfo<Clause<'tcx>>>,
    projs: InternedSet<'tcx, List<ProjectionKind>>,
    place_elems: InternedSet<'tcx, List<PlaceElem<'tcx>>>,
    const_: InternedSet<'tcx, WithCachedTypeInfo<ty::ConstKind<'tcx>>>,
    pat: InternedSet<'tcx, PatternKind<'tcx>>,
    const_allocation: InternedSet<'tcx, Allocation>,
    bound_variable_kinds: InternedSet<'tcx, List<ty::BoundVariableKind>>,
    layout: InternedSet<'tcx, LayoutData<FieldIdx, VariantIdx>>,
    adt_def: InternedSet<'tcx, AdtDefData>,
    external_constraints: InternedSet<'tcx, ExternalConstraintsData<TyCtxt<'tcx>>>,
    predefined_opaques_in_body: InternedSet<'tcx, PredefinedOpaquesData<TyCtxt<'tcx>>>,
    fields: InternedSet<'tcx, List<FieldIdx>>,
    local_def_ids: InternedSet<'tcx, List<LocalDefId>>,
    captures: InternedSet<'tcx, List<&'tcx ty::CapturedPlace<'tcx>>>,
    offset_of: InternedSet<'tcx, List<(VariantIdx, FieldIdx)>>,
    valtree: InternedSet<'tcx, ty::ValTreeKind<'tcx>>,
    patterns: InternedSet<'tcx, List<ty::Pattern<'tcx>>>,
}

impl<'tcx> CtxtInterners<'tcx> {
    fn new(arena: &'tcx WorkerLocal<Arena<'tcx>>) -> CtxtInterners<'tcx> {
        // Default interner size - this value has been chosen empirically, and may need to be adjusted
        // as the compiler evolves.
        const N: usize = 2048;
        CtxtInterners {
            arena,
            // The factors have been chosen by @FractalFir based on observed interner sizes, and local perf runs.
            // To get the interner sizes, insert `eprintln` printing the size of the interner in functions like `intern_ty`.
            // Bigger benchmarks tend to give more accurate ratios, so use something like `x perf eprintln --includes cargo`.
            type_: InternedSet::with_capacity(N * 16),
            const_lists: InternedSet::with_capacity(N * 4),
            args: InternedSet::with_capacity(N * 4),
            type_lists: InternedSet::with_capacity(N * 4),
            region: InternedSet::with_capacity(N * 4),
            poly_existential_predicates: InternedSet::with_capacity(N / 4),
            canonical_var_kinds: InternedSet::with_capacity(N / 2),
            predicate: InternedSet::with_capacity(N),
            clauses: InternedSet::with_capacity(N),
            projs: InternedSet::with_capacity(N * 4),
            place_elems: InternedSet::with_capacity(N * 2),
            const_: InternedSet::with_capacity(N * 2),
            pat: InternedSet::with_capacity(N),
            const_allocation: InternedSet::with_capacity(N),
            bound_variable_kinds: InternedSet::with_capacity(N * 2),
            layout: InternedSet::with_capacity(N),
            adt_def: InternedSet::with_capacity(N),
            external_constraints: InternedSet::with_capacity(N),
            predefined_opaques_in_body: InternedSet::with_capacity(N),
            fields: InternedSet::with_capacity(N * 4),
            local_def_ids: InternedSet::with_capacity(N),
            captures: InternedSet::with_capacity(N),
            offset_of: InternedSet::with_capacity(N),
            valtree: InternedSet::with_capacity(N),
            patterns: InternedSet::with_capacity(N),
        }
    }

    /// Interns a type. (Use `mk_*` functions instead, where possible.)
    #[allow(rustc::usage_of_ty_tykind)]
    #[inline(never)]
    fn intern_ty(&self, kind: TyKind<'tcx>, sess: &Session, untracked: &Untracked) -> Ty<'tcx> {
        Ty(Interned::new_unchecked(
            self.type_
                .intern(kind, |kind| {
                    let flags = ty::FlagComputation::<TyCtxt<'tcx>>::for_kind(&kind);
                    let stable_hash = self.stable_hash(&flags, sess, untracked, &kind);

                    InternedInSet(self.arena.alloc(WithCachedTypeInfo {
                        internee: kind,
                        stable_hash,
                        flags: flags.flags,
                        outer_exclusive_binder: flags.outer_exclusive_binder,
                    }))
                })
                .0,
        ))
    }

    /// Interns a const. (Use `mk_*` functions instead, where possible.)
    #[allow(rustc::usage_of_ty_tykind)]
    #[inline(never)]
    fn intern_const(
        &self,
        kind: ty::ConstKind<'tcx>,
        sess: &Session,
        untracked: &Untracked,
    ) -> Const<'tcx> {
        Const(Interned::new_unchecked(
            self.const_
                .intern(kind, |kind: ty::ConstKind<'_>| {
                    let flags = ty::FlagComputation::<TyCtxt<'tcx>>::for_const_kind(&kind);
                    let stable_hash = self.stable_hash(&flags, sess, untracked, &kind);

                    InternedInSet(self.arena.alloc(WithCachedTypeInfo {
                        internee: kind,
                        stable_hash,
                        flags: flags.flags,
                        outer_exclusive_binder: flags.outer_exclusive_binder,
                    }))
                })
                .0,
        ))
    }

    fn stable_hash<'a, T: HashStable<StableHashingContext<'a>>>(
        &self,
        flags: &ty::FlagComputation<TyCtxt<'tcx>>,
        sess: &'a Session,
        untracked: &'a Untracked,
        val: &T,
    ) -> Fingerprint {
        // It's impossible to hash inference variables (and will ICE), so we don't need to try to cache them.
        // Without incremental, we rarely stable-hash types, so let's not do it proactively.
        if flags.flags.intersects(TypeFlags::HAS_INFER) || sess.opts.incremental.is_none() {
            Fingerprint::ZERO
        } else {
            let mut hasher = StableHasher::new();
            let mut hcx = StableHashingContext::new(sess, untracked);
            val.hash_stable(&mut hcx, &mut hasher);
            hasher.finish()
        }
    }

    /// Interns a predicate. (Use `mk_predicate` instead, where possible.)
    #[inline(never)]
    fn intern_predicate(
        &self,
        kind: Binder<'tcx, PredicateKind<'tcx>>,
        sess: &Session,
        untracked: &Untracked,
    ) -> Predicate<'tcx> {
        Predicate(Interned::new_unchecked(
            self.predicate
                .intern(kind, |kind| {
                    let flags = ty::FlagComputation::<TyCtxt<'tcx>>::for_predicate(kind);

                    let stable_hash = self.stable_hash(&flags, sess, untracked, &kind);

                    InternedInSet(self.arena.alloc(WithCachedTypeInfo {
                        internee: kind,
                        stable_hash,
                        flags: flags.flags,
                        outer_exclusive_binder: flags.outer_exclusive_binder,
                    }))
                })
                .0,
        ))
    }

    fn intern_clauses(&self, clauses: &[Clause<'tcx>]) -> Clauses<'tcx> {
        if clauses.is_empty() {
            ListWithCachedTypeInfo::empty()
        } else {
            self.clauses
                .intern_ref(clauses, || {
                    let flags = ty::FlagComputation::<TyCtxt<'tcx>>::for_clauses(clauses);

                    InternedInSet(ListWithCachedTypeInfo::from_arena(
                        &*self.arena,
                        flags.into(),
                        clauses,
                    ))
                })
                .0
        }
    }
}

// For these preinterned values, an alternative would be to have
// variable-length vectors that grow as needed. But that turned out to be
// slightly more complex and no faster.

const NUM_PREINTERNED_TY_VARS: u32 = 100;
const NUM_PREINTERNED_FRESH_TYS: u32 = 20;
const NUM_PREINTERNED_FRESH_INT_TYS: u32 = 3;
const NUM_PREINTERNED_FRESH_FLOAT_TYS: u32 = 3;

// This number may seem high, but it is reached in all but the smallest crates.
const NUM_PREINTERNED_RE_VARS: u32 = 500;
const NUM_PREINTERNED_RE_LATE_BOUNDS_I: u32 = 2;
const NUM_PREINTERNED_RE_LATE_BOUNDS_V: u32 = 20;

pub struct CommonTypes<'tcx> {
    pub unit: Ty<'tcx>,
    pub bool: Ty<'tcx>,
    pub char: Ty<'tcx>,
    pub isize: Ty<'tcx>,
    pub i8: Ty<'tcx>,
    pub i16: Ty<'tcx>,
    pub i32: Ty<'tcx>,
    pub i64: Ty<'tcx>,
    pub i128: Ty<'tcx>,
    pub usize: Ty<'tcx>,
    pub u8: Ty<'tcx>,
    pub u16: Ty<'tcx>,
    pub u32: Ty<'tcx>,
    pub u64: Ty<'tcx>,
    pub u128: Ty<'tcx>,
    pub f16: Ty<'tcx>,
    pub f32: Ty<'tcx>,
    pub f64: Ty<'tcx>,
    pub f128: Ty<'tcx>,
    pub str_: Ty<'tcx>,
    pub never: Ty<'tcx>,
    pub self_param: Ty<'tcx>,

    /// Dummy type used for the `Self` of a `TraitRef` created for converting
    /// a trait object, and which gets removed in `ExistentialTraitRef`.
    /// This type must not appear anywhere in other converted types.
    /// `Infer(ty::FreshTy(0))` does the job.
    pub trait_object_dummy_self: Ty<'tcx>,

    /// Pre-interned `Infer(ty::TyVar(n))` for small values of `n`.
    pub ty_vars: Vec<Ty<'tcx>>,

    /// Pre-interned `Infer(ty::FreshTy(n))` for small values of `n`.
    pub fresh_tys: Vec<Ty<'tcx>>,

    /// Pre-interned `Infer(ty::FreshIntTy(n))` for small values of `n`.
    pub fresh_int_tys: Vec<Ty<'tcx>>,

    /// Pre-interned `Infer(ty::FreshFloatTy(n))` for small values of `n`.
    pub fresh_float_tys: Vec<Ty<'tcx>>,
}

pub struct CommonLifetimes<'tcx> {
    /// `ReStatic`
    pub re_static: Region<'tcx>,

    /// Erased region, used outside of type inference.
    pub re_erased: Region<'tcx>,

    /// Pre-interned `ReVar(ty::RegionVar(n))` for small values of `n`.
    pub re_vars: Vec<Region<'tcx>>,

    /// Pre-interned values of the form:
    /// `ReBound(DebruijnIndex(i), BoundRegion { var: v, kind: BrAnon })`
    /// for small values of `i` and `v`.
    pub re_late_bounds: Vec<Vec<Region<'tcx>>>,
}

pub struct CommonConsts<'tcx> {
    pub unit: Const<'tcx>,
    pub true_: Const<'tcx>,
    pub false_: Const<'tcx>,
    /// Use [`ty::ValTree::zst`] instead.
    pub(crate) valtree_zst: ValTree<'tcx>,
}

impl<'tcx> CommonTypes<'tcx> {
    fn new(
        interners: &CtxtInterners<'tcx>,
        sess: &Session,
        untracked: &Untracked,
    ) -> CommonTypes<'tcx> {
        let mk = |ty| interners.intern_ty(ty, sess, untracked);

        let ty_vars =
            (0..NUM_PREINTERNED_TY_VARS).map(|n| mk(Infer(ty::TyVar(TyVid::from(n))))).collect();
        let fresh_tys: Vec<_> =
            (0..NUM_PREINTERNED_FRESH_TYS).map(|n| mk(Infer(ty::FreshTy(n)))).collect();
        let fresh_int_tys: Vec<_> =
            (0..NUM_PREINTERNED_FRESH_INT_TYS).map(|n| mk(Infer(ty::FreshIntTy(n)))).collect();
        let fresh_float_tys: Vec<_> =
            (0..NUM_PREINTERNED_FRESH_FLOAT_TYS).map(|n| mk(Infer(ty::FreshFloatTy(n)))).collect();

        CommonTypes {
            unit: mk(Tuple(List::empty())),
            bool: mk(Bool),
            char: mk(Char),
            never: mk(Never),
            isize: mk(Int(ty::IntTy::Isize)),
            i8: mk(Int(ty::IntTy::I8)),
            i16: mk(Int(ty::IntTy::I16)),
            i32: mk(Int(ty::IntTy::I32)),
            i64: mk(Int(ty::IntTy::I64)),
            i128: mk(Int(ty::IntTy::I128)),
            usize: mk(Uint(ty::UintTy::Usize)),
            u8: mk(Uint(ty::UintTy::U8)),
            u16: mk(Uint(ty::UintTy::U16)),
            u32: mk(Uint(ty::UintTy::U32)),
            u64: mk(Uint(ty::UintTy::U64)),
            u128: mk(Uint(ty::UintTy::U128)),
            f16: mk(Float(ty::FloatTy::F16)),
            f32: mk(Float(ty::FloatTy::F32)),
            f64: mk(Float(ty::FloatTy::F64)),
            f128: mk(Float(ty::FloatTy::F128)),
            str_: mk(Str),
            self_param: mk(ty::Param(ty::ParamTy { index: 0, name: kw::SelfUpper })),

            trait_object_dummy_self: fresh_tys[0],

            ty_vars,
            fresh_tys,
            fresh_int_tys,
            fresh_float_tys,
        }
    }
}

impl<'tcx> CommonLifetimes<'tcx> {
    fn new(interners: &CtxtInterners<'tcx>) -> CommonLifetimes<'tcx> {
        let mk = |r| {
            Region(Interned::new_unchecked(
                interners.region.intern(r, |r| InternedInSet(interners.arena.alloc(r))).0,
            ))
        };

        let re_vars =
            (0..NUM_PREINTERNED_RE_VARS).map(|n| mk(ty::ReVar(ty::RegionVid::from(n)))).collect();

        let re_late_bounds = (0..NUM_PREINTERNED_RE_LATE_BOUNDS_I)
            .map(|i| {
                (0..NUM_PREINTERNED_RE_LATE_BOUNDS_V)
                    .map(|v| {
                        mk(ty::ReBound(
                            ty::DebruijnIndex::from(i),
                            ty::BoundRegion {
                                var: ty::BoundVar::from(v),
                                kind: ty::BoundRegionKind::Anon,
                            },
                        ))
                    })
                    .collect()
            })
            .collect();

        CommonLifetimes {
            re_static: mk(ty::ReStatic),
            re_erased: mk(ty::ReErased),
            re_vars,
            re_late_bounds,
        }
    }
}

impl<'tcx> CommonConsts<'tcx> {
    fn new(
        interners: &CtxtInterners<'tcx>,
        types: &CommonTypes<'tcx>,
        sess: &Session,
        untracked: &Untracked,
    ) -> CommonConsts<'tcx> {
        let mk_const = |c| {
            interners.intern_const(
                c, sess, // This is only used to create a stable hashing context.
                untracked,
            )
        };

        let mk_valtree = |v| {
            ty::ValTree(Interned::new_unchecked(
                interners.valtree.intern(v, |v| InternedInSet(interners.arena.alloc(v))).0,
            ))
        };

        let valtree_zst = mk_valtree(ty::ValTreeKind::Branch(Box::default()));
        let valtree_true = mk_valtree(ty::ValTreeKind::Leaf(ty::ScalarInt::TRUE));
        let valtree_false = mk_valtree(ty::ValTreeKind::Leaf(ty::ScalarInt::FALSE));

        CommonConsts {
            unit: mk_const(ty::ConstKind::Value(ty::Value {
                ty: types.unit,
                valtree: valtree_zst,
            })),
            true_: mk_const(ty::ConstKind::Value(ty::Value {
                ty: types.bool,
                valtree: valtree_true,
            })),
            false_: mk_const(ty::ConstKind::Value(ty::Value {
                ty: types.bool,
                valtree: valtree_false,
            })),
            valtree_zst,
        }
    }
}

/// This struct contains information regarding a free parameter region,
/// either a `ReEarlyParam` or `ReLateParam`.
#[derive(Debug)]
pub struct FreeRegionInfo {
    /// `LocalDefId` of the scope.
    pub scope: LocalDefId,
    /// the `DefId` of the free region.
    pub region_def_id: DefId,
    /// checks if bound region is in Impl Item
    pub is_impl_item: bool,
}

/// This struct should only be created by `create_def`.
#[derive(Copy, Clone)]
pub struct TyCtxtFeed<'tcx, KEY: Copy> {
    pub tcx: TyCtxt<'tcx>,
    // Do not allow direct access, as downstream code must not mutate this field.
    key: KEY,
}

/// Never return a `Feed` from a query. Only queries that create a `DefId` are
/// allowed to feed queries for that `DefId`.
impl<KEY: Copy, CTX> !HashStable<CTX> for TyCtxtFeed<'_, KEY> {}

/// The same as `TyCtxtFeed`, but does not contain a `TyCtxt`.
/// Use this to pass around when you have a `TyCtxt` elsewhere.
/// Just an optimization to save space and not store hundreds of
/// `TyCtxtFeed` in the resolver.
#[derive(Copy, Clone)]
pub struct Feed<'tcx, KEY: Copy> {
    _tcx: PhantomData<TyCtxt<'tcx>>,
    // Do not allow direct access, as downstream code must not mutate this field.
    key: KEY,
}

/// Never return a `Feed` from a query. Only queries that create a `DefId` are
/// allowed to feed queries for that `DefId`.
impl<KEY: Copy, CTX> !HashStable<CTX> for Feed<'_, KEY> {}

impl<T: fmt::Debug + Copy> fmt::Debug for Feed<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.key.fmt(f)
    }
}

/// Some workarounds to use cases that cannot use `create_def`.
/// Do not add new ways to create `TyCtxtFeed` without consulting
/// with T-compiler and making an analysis about why your addition
/// does not cause incremental compilation issues.
impl<'tcx> TyCtxt<'tcx> {
    /// Can only be fed before queries are run, and is thus exempt from any
    /// incremental issues. Do not use except for the initial query feeding.
    pub fn feed_unit_query(self) -> TyCtxtFeed<'tcx, ()> {
        self.dep_graph.assert_ignored();
        TyCtxtFeed { tcx: self, key: () }
    }

    /// Only used in the resolver to register the `CRATE_DEF_ID` `DefId` and feed
    /// some queries for it. It will panic if used twice.
    pub fn create_local_crate_def_id(self, span: Span) -> TyCtxtFeed<'tcx, LocalDefId> {
        let key = self.untracked().source_span.push(span);
        assert_eq!(key, CRATE_DEF_ID);
        TyCtxtFeed { tcx: self, key }
    }

    /// In order to break cycles involving `AnonConst`, we need to set the expected type by side
    /// effect. However, we do not want this as a general capability, so this interface restricts
    /// to the only allowed case.
    pub fn feed_anon_const_type(self, key: LocalDefId, value: ty::EarlyBinder<'tcx, Ty<'tcx>>) {
        debug_assert_eq!(self.def_kind(key), DefKind::AnonConst);
        TyCtxtFeed { tcx: self, key }.type_of(value)
    }
}

impl<'tcx, KEY: Copy> TyCtxtFeed<'tcx, KEY> {
    #[inline(always)]
    pub fn key(&self) -> KEY {
        self.key
    }

    #[inline(always)]
    pub fn downgrade(self) -> Feed<'tcx, KEY> {
        Feed { _tcx: PhantomData, key: self.key }
    }
}

impl<'tcx, KEY: Copy> Feed<'tcx, KEY> {
    #[inline(always)]
    pub fn key(&self) -> KEY {
        self.key
    }

    #[inline(always)]
    pub fn upgrade(self, tcx: TyCtxt<'tcx>) -> TyCtxtFeed<'tcx, KEY> {
        TyCtxtFeed { tcx, key: self.key }
    }
}

impl<'tcx> TyCtxtFeed<'tcx, LocalDefId> {
    #[inline(always)]
    pub fn def_id(&self) -> LocalDefId {
        self.key
    }

    // Caller must ensure that `self.key` ID is indeed an owner.
    pub fn feed_owner_id(&self) -> TyCtxtFeed<'tcx, hir::OwnerId> {
        TyCtxtFeed { tcx: self.tcx, key: hir::OwnerId { def_id: self.key } }
    }

    // Fills in all the important parts needed by HIR queries
    pub fn feed_hir(&self) {
        self.local_def_id_to_hir_id(HirId::make_owner(self.def_id()));

        let node = hir::OwnerNode::Synthetic;
        let bodies = Default::default();
        let attrs = hir::AttributeMap::EMPTY;

        let (opt_hash_including_bodies, _, _) =
            self.tcx.hash_owner_nodes(node, &bodies, &attrs.map, &[], attrs.define_opaque);
        let node = node.into();
        self.opt_hir_owner_nodes(Some(self.tcx.arena.alloc(hir::OwnerNodes {
            opt_hash_including_bodies,
            nodes: IndexVec::from_elem_n(
                hir::ParentedNode { parent: hir::ItemLocalId::INVALID, node },
                1,
            ),
            bodies,
        })));
        self.feed_owner_id().hir_attr_map(attrs);
    }
}

/// The central data structure of the compiler. It stores references
/// to the various **arenas** and also houses the results of the
/// various **compiler queries** that have been performed. See the
/// [rustc dev guide] for more details.
///
/// [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/ty.html
///
/// An implementation detail: `TyCtxt` is a wrapper type for [GlobalCtxt],
/// which is the struct that actually holds all the data. `TyCtxt` derefs to
/// `GlobalCtxt`, and in practice `TyCtxt` is passed around everywhere, and all
/// operations are done via `TyCtxt`. A `TyCtxt` is obtained for a `GlobalCtxt`
/// by calling `enter` with a closure `f`. That function creates both the
/// `TyCtxt`, and an `ImplicitCtxt` around it that is put into TLS. Within `f`:
/// - The `ImplicitCtxt` is available implicitly via TLS.
/// - The `TyCtxt` is available explicitly via the `tcx` parameter, and also
///   implicitly within the `ImplicitCtxt`. Explicit access is preferred when
///   possible.
#[derive(Copy, Clone)]
#[rustc_diagnostic_item = "TyCtxt"]
#[rustc_pass_by_value]
pub struct TyCtxt<'tcx> {
    gcx: &'tcx GlobalCtxt<'tcx>,
}

impl<'tcx> LintEmitter for TyCtxt<'tcx> {
    fn emit_node_span_lint(
        self,
        lint: &'static Lint,
        hir_id: HirId,
        span: impl Into<MultiSpan>,
        decorator: impl for<'a> LintDiagnostic<'a, ()>,
    ) {
        self.emit_node_span_lint(lint, hir_id, span, decorator);
    }
}

// Explicitly implement `DynSync` and `DynSend` for `TyCtxt` to short circuit trait resolution. Its
// field are asserted to implement these traits below, so this is trivially safe, and it greatly
// speeds-up compilation of this crate and its dependents.
unsafe impl DynSend for TyCtxt<'_> {}
unsafe impl DynSync for TyCtxt<'_> {}
fn _assert_tcx_fields() {
    sync::assert_dyn_sync::<&'_ GlobalCtxt<'_>>();
    sync::assert_dyn_send::<&'_ GlobalCtxt<'_>>();
}

impl<'tcx> Deref for TyCtxt<'tcx> {
    type Target = &'tcx GlobalCtxt<'tcx>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.gcx
    }
}

/// See [TyCtxt] for details about this type.
pub struct GlobalCtxt<'tcx> {
    pub arena: &'tcx WorkerLocal<Arena<'tcx>>,
    pub hir_arena: &'tcx WorkerLocal<hir::Arena<'tcx>>,

    interners: CtxtInterners<'tcx>,

    pub sess: &'tcx Session,
    crate_types: Vec<CrateType>,
    /// The `stable_crate_id` is constructed out of the crate name and all the
    /// `-C metadata` arguments passed to the compiler. Its value forms a unique
    /// global identifier for the crate. It is used to allow multiple crates
    /// with the same name to coexist. See the
    /// `rustc_symbol_mangling` crate for more information.
    stable_crate_id: StableCrateId,

    pub dep_graph: DepGraph,

    pub prof: SelfProfilerRef,

    /// Common types, pre-interned for your convenience.
    pub types: CommonTypes<'tcx>,

    /// Common lifetimes, pre-interned for your convenience.
    pub lifetimes: CommonLifetimes<'tcx>,

    /// Common consts, pre-interned for your convenience.
    pub consts: CommonConsts<'tcx>,

    /// Hooks to be able to register functions in other crates that can then still
    /// be called from rustc_middle.
    pub(crate) hooks: crate::hooks::Providers,

    untracked: Untracked,

    pub query_system: QuerySystem<'tcx>,
    pub(crate) query_kinds: &'tcx [DepKindStruct<'tcx>],

    // Internal caches for metadata decoding. No need to track deps on this.
    pub ty_rcache: Lock<FxHashMap<ty::CReaderCacheKey, Ty<'tcx>>>,

    /// Caches the results of trait selection. This cache is used
    /// for things that do not have to do with the parameters in scope.
    pub selection_cache: traits::SelectionCache<'tcx, ty::TypingEnv<'tcx>>,

    /// Caches the results of trait evaluation. This cache is used
    /// for things that do not have to do with the parameters in scope.
    /// Merge this with `selection_cache`?
    pub evaluation_cache: traits::EvaluationCache<'tcx, ty::TypingEnv<'tcx>>,

    /// Caches the results of goal evaluation in the new solver.
    pub new_solver_evaluation_cache: Lock<search_graph::GlobalCache<TyCtxt<'tcx>>>,
    pub new_solver_canonical_param_env_cache:
        Lock<FxHashMap<ty::ParamEnv<'tcx>, ty::CanonicalParamEnvCacheEntry<TyCtxt<'tcx>>>>,

    pub canonical_param_env_cache: CanonicalParamEnvCache<'tcx>,

    /// Caches the index of the highest bound var in clauses in a canonical binder.
    pub highest_var_in_clauses_cache: Lock<FxHashMap<ty::Clauses<'tcx>, usize>>,
    /// Caches the instantiation of a canonical binder given a set of args.
    pub clauses_cache:
        Lock<FxHashMap<(ty::Clauses<'tcx>, &'tcx [ty::GenericArg<'tcx>]), ty::Clauses<'tcx>>>,

    /// Data layout specification for the current target.
    pub data_layout: TargetDataLayout,

    /// Stores memory for globals (statics/consts).
    pub(crate) alloc_map: interpret::AllocMap<'tcx>,

    current_gcx: CurrentGcx,

    /// A jobserver reference used to release then acquire a token while waiting on a query.
    pub jobserver_proxy: Arc<Proxy>,
}

impl<'tcx> GlobalCtxt<'tcx> {
    /// Installs `self` in a `TyCtxt` and `ImplicitCtxt` for the duration of
    /// `f`.
    pub fn enter<F, R>(&'tcx self, f: F) -> R
    where
        F: FnOnce(TyCtxt<'tcx>) -> R,
    {
        let icx = tls::ImplicitCtxt::new(self);

        // Reset `current_gcx` to `None` when we exit.
        let _on_drop = defer(move || {
            *self.current_gcx.value.write() = None;
        });

        // Set this `GlobalCtxt` as the current one.
        {
            let mut guard = self.current_gcx.value.write();
            assert!(guard.is_none(), "no `GlobalCtxt` is currently set");
            *guard = Some(self as *const _ as *const ());
        }

        tls::enter_context(&icx, || f(icx.tcx))
    }
}

/// This is used to get a reference to a `GlobalCtxt` if one is available.
///
/// This is needed to allow the deadlock handler access to `GlobalCtxt` to look for query cycles.
/// It cannot use the `TLV` global because that's only guaranteed to be defined on the thread
/// creating the `GlobalCtxt`. Other threads have access to the `TLV` only inside Rayon jobs, but
/// the deadlock handler is not called inside such a job.
#[derive(Clone)]
pub struct CurrentGcx {
    /// This stores a pointer to a `GlobalCtxt`. This is set to `Some` inside `GlobalCtxt::enter`
    /// and reset to `None` when that function returns or unwinds.
    value: Arc<RwLock<Option<*const ()>>>,
}

unsafe impl DynSend for CurrentGcx {}
unsafe impl DynSync for CurrentGcx {}

impl CurrentGcx {
    pub fn new() -> Self {
        Self { value: Arc::new(RwLock::new(None)) }
    }

    pub fn access<R>(&self, f: impl for<'tcx> FnOnce(&'tcx GlobalCtxt<'tcx>) -> R) -> R {
        let read_guard = self.value.read();
        let gcx: *const GlobalCtxt<'_> = read_guard.unwrap() as *const _;
        // SAFETY: We hold the read lock for the `GlobalCtxt` pointer. That prevents
        // `GlobalCtxt::enter` from returning as it would first acquire the write lock.
        // This ensures the `GlobalCtxt` is live during `f`.
        f(unsafe { &*gcx })
    }
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn has_typeck_results(self, def_id: LocalDefId) -> bool {
        // Closures' typeck results come from their outermost function,
        // as they are part of the same "inference environment".
        let typeck_root_def_id = self.typeck_root_def_id(def_id.to_def_id());
        if typeck_root_def_id != def_id.to_def_id() {
            return self.has_typeck_results(typeck_root_def_id.expect_local());
        }

        self.hir_node_by_def_id(def_id).body_id().is_some()
    }

    /// Expects a body and returns its codegen attributes.
    ///
    /// Unlike `codegen_fn_attrs`, this returns `CodegenFnAttrs::EMPTY` for
    /// constants.
    pub fn body_codegen_attrs(self, def_id: DefId) -> &'tcx CodegenFnAttrs {
        let def_kind = self.def_kind(def_id);
        if def_kind.has_codegen_attrs() {
            self.codegen_fn_attrs(def_id)
        } else if matches!(
            def_kind,
            DefKind::AnonConst
                | DefKind::AssocConst
                | DefKind::Const
                | DefKind::InlineConst
                | DefKind::GlobalAsm
        ) {
            CodegenFnAttrs::EMPTY
        } else {
            bug!(
                "body_codegen_fn_attrs called on unexpected definition: {:?} {:?}",
                def_id,
                def_kind
            )
        }
    }

    pub fn alloc_steal_thir(self, thir: Thir<'tcx>) -> &'tcx Steal<Thir<'tcx>> {
        self.arena.alloc(Steal::new(thir))
    }

    pub fn alloc_steal_mir(self, mir: Body<'tcx>) -> &'tcx Steal<Body<'tcx>> {
        self.arena.alloc(Steal::new(mir))
    }

    pub fn alloc_steal_promoted(
        self,
        promoted: IndexVec<Promoted, Body<'tcx>>,
    ) -> &'tcx Steal<IndexVec<Promoted, Body<'tcx>>> {
        self.arena.alloc(Steal::new(promoted))
    }

    pub fn mk_adt_def(
        self,
        did: DefId,
        kind: AdtKind,
        variants: IndexVec<VariantIdx, ty::VariantDef>,
        repr: ReprOptions,
    ) -> ty::AdtDef<'tcx> {
        self.mk_adt_def_from_data(ty::AdtDefData::new(self, did, kind, variants, repr))
    }

    /// Allocates a read-only byte or string literal for `mir::interpret` with alignment 1.
    /// Returns the same `AllocId` if called again with the same bytes.
    pub fn allocate_bytes_dedup(self, bytes: &[u8], salt: usize) -> interpret::AllocId {
        // Create an allocation that just contains these bytes.
        let alloc = interpret::Allocation::from_bytes_byte_aligned_immutable(bytes, ());
        let alloc = self.mk_const_alloc(alloc);
        self.reserve_and_set_memory_dedup(alloc, salt)
    }

    /// Traits added on all bounds by default, excluding `Sized` which is treated separately.
    pub fn default_traits(self) -> &'static [rustc_hir::LangItem] {
        if self.sess.opts.unstable_opts.experimental_default_bounds {
            &[
                LangItem::DefaultTrait1,
                LangItem::DefaultTrait2,
                LangItem::DefaultTrait3,
                LangItem::DefaultTrait4,
            ]
        } else {
            &[]
        }
    }

    pub fn is_default_trait(self, def_id: DefId) -> bool {
        self.default_traits()
            .iter()
            .any(|&default_trait| self.lang_items().get(default_trait) == Some(def_id))
    }

    /// Returns a range of the start/end indices specified with the
    /// `rustc_layout_scalar_valid_range` attribute.
    // FIXME(eddyb) this is an awkward spot for this method, maybe move it?
    pub fn layout_scalar_valid_range(self, def_id: DefId) -> (Bound<u128>, Bound<u128>) {
        let get = |name| {
            let Some(attr) = self.get_attr(def_id, name) else {
                return Bound::Unbounded;
            };
            debug!("layout_scalar_valid_range: attr={:?}", attr);
            if let Some(
                &[
                    ast::MetaItemInner::Lit(ast::MetaItemLit {
                        kind: ast::LitKind::Int(a, _), ..
                    }),
                ],
            ) = attr.meta_item_list().as_deref()
            {
                Bound::Included(a.get())
            } else {
                self.dcx().span_delayed_bug(
                    attr.span(),
                    "invalid rustc_layout_scalar_valid_range attribute",
                );
                Bound::Unbounded
            }
        };
        (
            get(sym::rustc_layout_scalar_valid_range_start),
            get(sym::rustc_layout_scalar_valid_range_end),
        )
    }

    pub fn lift<T: Lift<TyCtxt<'tcx>>>(self, value: T) -> Option<T::Lifted> {
        value.lift_to_interner(self)
    }

    /// Creates a type context. To use the context call `fn enter` which
    /// provides a `TyCtxt`.
    ///
    /// By only providing the `TyCtxt` inside of the closure we enforce that the type
    /// context and any interned value (types, args, etc.) can only be used while `ty::tls`
    /// has a valid reference to the context, to allow formatting values that need it.
    pub fn create_global_ctxt<T>(
        gcx_cell: &'tcx OnceLock<GlobalCtxt<'tcx>>,
        s: &'tcx Session,
        crate_types: Vec<CrateType>,
        stable_crate_id: StableCrateId,
        arena: &'tcx WorkerLocal<Arena<'tcx>>,
        hir_arena: &'tcx WorkerLocal<hir::Arena<'tcx>>,
        untracked: Untracked,
        dep_graph: DepGraph,
        query_kinds: &'tcx [DepKindStruct<'tcx>],
        query_system: QuerySystem<'tcx>,
        hooks: crate::hooks::Providers,
        current_gcx: CurrentGcx,
        jobserver_proxy: Arc<Proxy>,
        f: impl FnOnce(TyCtxt<'tcx>) -> T,
    ) -> T {
        let data_layout = s.target.parse_data_layout().unwrap_or_else(|err| {
            s.dcx().emit_fatal(err);
        });
        let interners = CtxtInterners::new(arena);
        let common_types = CommonTypes::new(&interners, s, &untracked);
        let common_lifetimes = CommonLifetimes::new(&interners);
        let common_consts = CommonConsts::new(&interners, &common_types, s, &untracked);

        let gcx = gcx_cell.get_or_init(|| GlobalCtxt {
            sess: s,
            crate_types,
            stable_crate_id,
            arena,
            hir_arena,
            interners,
            dep_graph,
            hooks,
            prof: s.prof.clone(),
            types: common_types,
            lifetimes: common_lifetimes,
            consts: common_consts,
            untracked,
            query_system,
            query_kinds,
            ty_rcache: Default::default(),
            selection_cache: Default::default(),
            evaluation_cache: Default::default(),
            new_solver_evaluation_cache: Default::default(),
            new_solver_canonical_param_env_cache: Default::default(),
            canonical_param_env_cache: Default::default(),
            highest_var_in_clauses_cache: Default::default(),
            clauses_cache: Default::default(),
            data_layout,
            alloc_map: interpret::AllocMap::new(),
            current_gcx,
            jobserver_proxy,
        });

        // This is a separate function to work around a crash with parallel rustc (#135870)
        gcx.enter(f)
    }

    /// Obtain all lang items of this crate and all dependencies (recursively)
    pub fn lang_items(self) -> &'tcx rustc_hir::lang_items::LanguageItems {
        self.get_lang_items(())
    }

    /// Gets a `Ty` representing the [`LangItem::OrderingEnum`]
    #[track_caller]
    pub fn ty_ordering_enum(self, span: Span) -> Ty<'tcx> {
        let ordering_enum = self.require_lang_item(hir::LangItem::OrderingEnum, span);
        self.type_of(ordering_enum).no_bound_vars().unwrap()
    }

    /// Obtain the given diagnostic item's `DefId`. Use `is_diagnostic_item` if you just want to
    /// compare against another `DefId`, since `is_diagnostic_item` is cheaper.
    pub fn get_diagnostic_item(self, name: Symbol) -> Option<DefId> {
        self.all_diagnostic_items(()).name_to_id.get(&name).copied()
    }

    /// Obtain the diagnostic item's name
    pub fn get_diagnostic_name(self, id: DefId) -> Option<Symbol> {
        self.diagnostic_items(id.krate).id_to_name.get(&id).copied()
    }

    /// Check whether the diagnostic item with the given `name` has the given `DefId`.
    pub fn is_diagnostic_item(self, name: Symbol, did: DefId) -> bool {
        self.diagnostic_items(did.krate).name_to_id.get(&name) == Some(&did)
    }

    pub fn is_coroutine(self, def_id: DefId) -> bool {
        self.coroutine_kind(def_id).is_some()
    }

    pub fn is_async_drop_in_place_coroutine(self, def_id: DefId) -> bool {
        self.is_lang_item(self.parent(def_id), LangItem::AsyncDropInPlace)
    }

    /// Returns the movability of the coroutine of `def_id`, or panics
    /// if given a `def_id` that is not a coroutine.
    pub fn coroutine_movability(self, def_id: DefId) -> hir::Movability {
        self.coroutine_kind(def_id).expect("expected a coroutine").movability()
    }

    /// Returns `true` if the node pointed to by `def_id` is a coroutine for an async construct.
    pub fn coroutine_is_async(self, def_id: DefId) -> bool {
        matches!(
            self.coroutine_kind(def_id),
            Some(hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Async, _))
        )
    }

    // Whether the body owner is synthetic, which in this case means it does not correspond to
    // meaningful HIR. This is currently used to skip over MIR borrowck.
    pub fn is_synthetic_mir(self, def_id: impl Into<DefId>) -> bool {
        matches!(self.def_kind(def_id.into()), DefKind::SyntheticCoroutineBody)
    }

    /// Returns `true` if the node pointed to by `def_id` is a general coroutine that implements `Coroutine`.
    /// This means it is neither an `async` or `gen` construct.
    pub fn is_general_coroutine(self, def_id: DefId) -> bool {
        matches!(self.coroutine_kind(def_id), Some(hir::CoroutineKind::Coroutine(_)))
    }

    /// Returns `true` if the node pointed to by `def_id` is a coroutine for a `gen` construct.
    pub fn coroutine_is_gen(self, def_id: DefId) -> bool {
        matches!(
            self.coroutine_kind(def_id),
            Some(hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Gen, _))
        )
    }

    /// Returns `true` if the node pointed to by `def_id` is a coroutine for a `async gen` construct.
    pub fn coroutine_is_async_gen(self, def_id: DefId) -> bool {
        matches!(
            self.coroutine_kind(def_id),
            Some(hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::AsyncGen, _))
        )
    }

    pub fn stability(self) -> &'tcx stability::Index {
        self.stability_index(())
    }

    pub fn features(self) -> &'tcx rustc_feature::Features {
        self.features_query(())
    }

    pub fn def_key(self, id: impl IntoQueryParam<DefId>) -> rustc_hir::definitions::DefKey {
        let id = id.into_query_param();
        // Accessing the DefKey is ok, since it is part of DefPathHash.
        if let Some(id) = id.as_local() {
            self.definitions_untracked().def_key(id)
        } else {
            self.cstore_untracked().def_key(id)
        }
    }

    /// Converts a `DefId` into its fully expanded `DefPath` (every
    /// `DefId` is really just an interned `DefPath`).
    ///
    /// Note that if `id` is not local to this crate, the result will
    ///  be a non-local `DefPath`.
    pub fn def_path(self, id: DefId) -> rustc_hir::definitions::DefPath {
        // Accessing the DefPath is ok, since it is part of DefPathHash.
        if let Some(id) = id.as_local() {
            self.definitions_untracked().def_path(id)
        } else {
            self.cstore_untracked().def_path(id)
        }
    }

    #[inline]
    pub fn def_path_hash(self, def_id: DefId) -> rustc_hir::definitions::DefPathHash {
        // Accessing the DefPathHash is ok, it is incr. comp. stable.
        if let Some(def_id) = def_id.as_local() {
            self.definitions_untracked().def_path_hash(def_id)
        } else {
            self.cstore_untracked().def_path_hash(def_id)
        }
    }

    #[inline]
    pub fn crate_types(self) -> &'tcx [CrateType] {
        &self.crate_types
    }

    pub fn needs_metadata(self) -> bool {
        self.crate_types().iter().any(|ty| match *ty {
            CrateType::Executable
            | CrateType::Staticlib
            | CrateType::Cdylib
            | CrateType::Sdylib => false,
            CrateType::Rlib | CrateType::Dylib | CrateType::ProcMacro => true,
        })
    }

    pub fn needs_crate_hash(self) -> bool {
        // Why is the crate hash needed for these configurations?
        // - debug_assertions: for the "fingerprint the result" check in
        //   `rustc_query_system::query::plumbing::execute_job`.
        // - incremental: for query lookups.
        // - needs_metadata: for putting into crate metadata.
        // - instrument_coverage: for putting into coverage data (see
        //   `hash_mir_source`).
        // - metrics_dir: metrics use the strict version hash in the filenames
        //   for dumped metrics files to prevent overwriting distinct metrics
        //   for similar source builds (may change in the future, this is part
        //   of the proof of concept impl for the metrics initiative project goal)
        cfg!(debug_assertions)
            || self.sess.opts.incremental.is_some()
            || self.needs_metadata()
            || self.sess.instrument_coverage()
            || self.sess.opts.unstable_opts.metrics_dir.is_some()
    }

    #[inline]
    pub fn stable_crate_id(self, crate_num: CrateNum) -> StableCrateId {
        if crate_num == LOCAL_CRATE {
            self.stable_crate_id
        } else {
            self.cstore_untracked().stable_crate_id(crate_num)
        }
    }

    /// Maps a StableCrateId to the corresponding CrateNum. This method assumes
    /// that the crate in question has already been loaded by the CrateStore.
    #[inline]
    pub fn stable_crate_id_to_crate_num(self, stable_crate_id: StableCrateId) -> CrateNum {
        if stable_crate_id == self.stable_crate_id(LOCAL_CRATE) {
            LOCAL_CRATE
        } else {
            *self
                .untracked()
                .stable_crate_ids
                .read()
                .get(&stable_crate_id)
                .unwrap_or_else(|| bug!("uninterned StableCrateId: {stable_crate_id:?}"))
        }
    }

    /// Converts a `DefPathHash` to its corresponding `DefId` in the current compilation
    /// session, if it still exists. This is used during incremental compilation to
    /// turn a deserialized `DefPathHash` into its current `DefId`.
    pub fn def_path_hash_to_def_id(self, hash: DefPathHash) -> Option<DefId> {
        debug!("def_path_hash_to_def_id({:?})", hash);

        let stable_crate_id = hash.stable_crate_id();

        // If this is a DefPathHash from the local crate, we can look up the
        // DefId in the tcx's `Definitions`.
        if stable_crate_id == self.stable_crate_id(LOCAL_CRATE) {
            Some(self.untracked.definitions.read().local_def_path_hash_to_def_id(hash)?.to_def_id())
        } else {
            Some(self.def_path_hash_to_def_id_extern(hash, stable_crate_id))
        }
    }

    pub fn def_path_debug_str(self, def_id: DefId) -> String {
        // We are explicitly not going through queries here in order to get
        // crate name and stable crate id since this code is called from debug!()
        // statements within the query system and we'd run into endless
        // recursion otherwise.
        let (crate_name, stable_crate_id) = if def_id.is_local() {
            (self.crate_name(LOCAL_CRATE), self.stable_crate_id(LOCAL_CRATE))
        } else {
            let cstore = &*self.cstore_untracked();
            (cstore.crate_name(def_id.krate), cstore.stable_crate_id(def_id.krate))
        };

        format!(
            "{}[{:04x}]{}",
            crate_name,
            // Don't print the whole stable crate id. That's just
            // annoying in debug output.
            stable_crate_id.as_u64() >> (8 * 6),
            self.def_path(def_id).to_string_no_crate_verbose()
        )
    }

    pub fn dcx(self) -> DiagCtxtHandle<'tcx> {
        self.sess.dcx()
    }

    pub fn is_target_feature_call_safe(
        self,
        callee_features: &[TargetFeature],
        body_features: &[TargetFeature],
    ) -> bool {
        // If the called function has target features the calling function hasn't,
        // the call requires `unsafe`. Don't check this on wasm
        // targets, though. For more information on wasm see the
        // is_like_wasm check in hir_analysis/src/collect.rs
        self.sess.target.options.is_like_wasm
            || callee_features
                .iter()
                .all(|feature| body_features.iter().any(|f| f.name == feature.name))
    }

    /// Returns the safe version of the signature of the given function, if calling it
    /// would be safe in the context of the given caller.
    pub fn adjust_target_feature_sig(
        self,
        fun_def: DefId,
        fun_sig: ty::Binder<'tcx, ty::FnSig<'tcx>>,
        caller: DefId,
    ) -> Option<ty::Binder<'tcx, ty::FnSig<'tcx>>> {
        let fun_features = &self.codegen_fn_attrs(fun_def).target_features;
        let callee_features = &self.codegen_fn_attrs(caller).target_features;
        if self.is_target_feature_call_safe(&fun_features, &callee_features) {
            return Some(fun_sig.map_bound(|sig| ty::FnSig { safety: hir::Safety::Safe, ..sig }));
        }
        None
    }

    /// Helper to get a tracked environment variable via. [`TyCtxt::env_var_os`] and converting to
    /// UTF-8 like [`std::env::var`].
    pub fn env_var<K: ?Sized + AsRef<OsStr>>(self, key: &'tcx K) -> Result<&'tcx str, VarError> {
        match self.env_var_os(key.as_ref()) {
            Some(value) => value.to_str().ok_or_else(|| VarError::NotUnicode(value.to_os_string())),
            None => Err(VarError::NotPresent),
        }
    }
}

impl<'tcx> TyCtxtAt<'tcx> {
    /// Create a new definition within the incr. comp. engine.
    pub fn create_def(
        self,
        parent: LocalDefId,
        name: Option<Symbol>,
        def_kind: DefKind,
        override_def_path_data: Option<DefPathData>,
        disambiguator: &mut DisambiguatorState,
    ) -> TyCtxtFeed<'tcx, LocalDefId> {
        let feed =
            self.tcx.create_def(parent, name, def_kind, override_def_path_data, disambiguator);

        feed.def_span(self.span);
        feed
    }
}

impl<'tcx> TyCtxt<'tcx> {
    /// `tcx`-dependent operations performed for every created definition.
    pub fn create_def(
        self,
        parent: LocalDefId,
        name: Option<Symbol>,
        def_kind: DefKind,
        override_def_path_data: Option<DefPathData>,
        disambiguator: &mut DisambiguatorState,
    ) -> TyCtxtFeed<'tcx, LocalDefId> {
        let data = override_def_path_data.unwrap_or_else(|| def_kind.def_path_data(name));
        // The following call has the side effect of modifying the tables inside `definitions`.
        // These very tables are relied on by the incr. comp. engine to decode DepNodes and to
        // decode the on-disk cache.
        //
        // Any LocalDefId which is used within queries, either as key or result, either:
        // - has been created before the construction of the TyCtxt;
        // - has been created by this call to `create_def`.
        // As a consequence, this LocalDefId is always re-created before it is needed by the incr.
        // comp. engine itself.
        let def_id = self.untracked.definitions.write().create_def(parent, data, disambiguator);

        // This function modifies `self.definitions` using a side-effect.
        // We need to ensure that these side effects are re-run by the incr. comp. engine.
        // Depending on the forever-red node will tell the graph that the calling query
        // needs to be re-evaluated.
        self.dep_graph.read_index(DepNodeIndex::FOREVER_RED_NODE);

        let feed = TyCtxtFeed { tcx: self, key: def_id };
        feed.def_kind(def_kind);
        // Unique types created for closures participate in type privacy checking.
        // They have visibilities inherited from the module they are defined in.
        // Visibilities for opaque types are meaningless, but still provided
        // so that all items have visibilities.
        if matches!(def_kind, DefKind::Closure | DefKind::OpaqueTy) {
            let parent_mod = self.parent_module_from_def_id(def_id).to_def_id();
            feed.visibility(ty::Visibility::Restricted(parent_mod));
        }

        feed
    }

    pub fn create_crate_num(
        self,
        stable_crate_id: StableCrateId,
    ) -> Result<TyCtxtFeed<'tcx, CrateNum>, CrateNum> {
        if let Some(&existing) = self.untracked().stable_crate_ids.read().get(&stable_crate_id) {
            return Err(existing);
        }

        let num = CrateNum::new(self.untracked().stable_crate_ids.read().len());
        self.untracked().stable_crate_ids.write().insert(stable_crate_id, num);
        Ok(TyCtxtFeed { key: num, tcx: self })
    }

    pub fn iter_local_def_id(self) -> impl Iterator<Item = LocalDefId> {
        // Create a dependency to the red node to be sure we re-execute this when the amount of
        // definitions change.
        self.dep_graph.read_index(DepNodeIndex::FOREVER_RED_NODE);

        let definitions = &self.untracked.definitions;
        gen {
            let mut i = 0;

            // Recompute the number of definitions each time, because our caller may be creating
            // new ones.
            while i < { definitions.read().num_definitions() } {
                let local_def_index = rustc_span::def_id::DefIndex::from_usize(i);
                yield LocalDefId { local_def_index };
                i += 1;
            }

            // Freeze definitions once we finish iterating on them, to prevent adding new ones.
            definitions.freeze();
        }
    }

    pub fn def_path_table(self) -> &'tcx rustc_hir::definitions::DefPathTable {
        // Create a dependency to the crate to be sure we re-execute this when the amount of
        // definitions change.
        self.dep_graph.read_index(DepNodeIndex::FOREVER_RED_NODE);

        // Freeze definitions once we start iterating on them, to prevent adding new ones
        // while iterating. If some query needs to add definitions, it should be `ensure`d above.
        self.untracked.definitions.freeze().def_path_table()
    }

    pub fn def_path_hash_to_def_index_map(
        self,
    ) -> &'tcx rustc_hir::def_path_hash_map::DefPathHashMap {
        // Create a dependency to the crate to be sure we re-execute this when the amount of
        // definitions change.
        self.ensure_ok().hir_crate_items(());
        // Freeze definitions once we start iterating on them, to prevent adding new ones
        // while iterating. If some query needs to add definitions, it should be `ensure`d above.
        self.untracked.definitions.freeze().def_path_hash_to_def_index_map()
    }

    /// Note that this is *untracked* and should only be used within the query
    /// system if the result is otherwise tracked through queries
    #[inline]
    pub fn cstore_untracked(self) -> FreezeReadGuard<'tcx, CrateStoreDyn> {
        FreezeReadGuard::map(self.untracked.cstore.read(), |c| &**c)
    }

    /// Give out access to the untracked data without any sanity checks.
    pub fn untracked(self) -> &'tcx Untracked {
        &self.untracked
    }
    /// Note that this is *untracked* and should only be used within the query
    /// system if the result is otherwise tracked through queries
    #[inline]
    pub fn definitions_untracked(self) -> FreezeReadGuard<'tcx, Definitions> {
        self.untracked.definitions.read()
    }

    /// Note that this is *untracked* and should only be used within the query
    /// system if the result is otherwise tracked through queries
    #[inline]
    pub fn source_span_untracked(self, def_id: LocalDefId) -> Span {
        self.untracked.source_span.get(def_id).unwrap_or(DUMMY_SP)
    }

    #[inline(always)]
    pub fn with_stable_hashing_context<R>(
        self,
        f: impl FnOnce(StableHashingContext<'_>) -> R,
    ) -> R {
        f(StableHashingContext::new(self.sess, &self.untracked))
    }

    pub fn serialize_query_result_cache(self, encoder: FileEncoder) -> FileEncodeResult {
        self.query_system.on_disk_cache.as_ref().map_or(Ok(0), |c| c.serialize(self, encoder))
    }

    #[inline]
    pub fn local_crate_exports_generics(self) -> bool {
        self.crate_types().iter().any(|crate_type| {
            match crate_type {
                CrateType::Executable
                | CrateType::Staticlib
                | CrateType::ProcMacro
                | CrateType::Cdylib
                | CrateType::Sdylib => false,

                // FIXME rust-lang/rust#64319, rust-lang/rust#64872:
                // We want to block export of generics from dylibs,
                // but we must fix rust-lang/rust#65890 before we can
                // do that robustly.
                CrateType::Dylib => true,

                CrateType::Rlib => true,
            }
        })
    }

    /// Returns the `DefId` and the `BoundRegionKind` corresponding to the given region.
    pub fn is_suitable_region(
        self,
        generic_param_scope: LocalDefId,
        mut region: Region<'tcx>,
    ) -> Option<FreeRegionInfo> {
        let (suitable_region_binding_scope, region_def_id) = loop {
            let def_id =
                region.opt_param_def_id(self, generic_param_scope.to_def_id())?.as_local()?;
            let scope = self.local_parent(def_id);
            if self.def_kind(scope) == DefKind::OpaqueTy {
                // Lifetime params of opaque types are synthetic and thus irrelevant to
                // diagnostics. Map them back to their origin!
                region = self.map_opaque_lifetime_to_parent_lifetime(def_id);
                continue;
            }
            break (scope, def_id.into());
        };

        let is_impl_item = match self.hir_node_by_def_id(suitable_region_binding_scope) {
            Node::Item(..) | Node::TraitItem(..) => false,
            Node::ImplItem(..) => self.is_bound_region_in_impl_item(suitable_region_binding_scope),
            _ => false,
        };

        Some(FreeRegionInfo { scope: suitable_region_binding_scope, region_def_id, is_impl_item })
    }

    /// Given a `DefId` for an `fn`, return all the `dyn` and `impl` traits in its return type.
    pub fn return_type_impl_or_dyn_traits(
        self,
        scope_def_id: LocalDefId,
    ) -> Vec<&'tcx hir::Ty<'tcx>> {
        let hir_id = self.local_def_id_to_hir_id(scope_def_id);
        let Some(hir::FnDecl { output: hir::FnRetTy::Return(hir_output), .. }) =
            self.hir_fn_decl_by_hir_id(hir_id)
        else {
            return vec![];
        };

        let mut v = TraitObjectVisitor(vec![]);
        v.visit_ty_unambig(hir_output);
        v.0
    }

    /// Given a `DefId` for an `fn`, return all the `dyn` and `impl` traits in
    /// its return type, and the associated alias span when type alias is used,
    /// along with a span for lifetime suggestion (if there are existing generics).
    pub fn return_type_impl_or_dyn_traits_with_type_alias(
        self,
        scope_def_id: LocalDefId,
    ) -> Option<(Vec<&'tcx hir::Ty<'tcx>>, Span, Option<Span>)> {
        let hir_id = self.local_def_id_to_hir_id(scope_def_id);
        let mut v = TraitObjectVisitor(vec![]);
        // when the return type is a type alias
        if let Some(hir::FnDecl { output: hir::FnRetTy::Return(hir_output), .. }) = self.hir_fn_decl_by_hir_id(hir_id)
            && let hir::TyKind::Path(hir::QPath::Resolved(
                None,
                hir::Path { res: hir::def::Res::Def(DefKind::TyAlias, def_id), .. }, )) = hir_output.kind
            && let Some(local_id) = def_id.as_local()
            && let Some(alias_ty) = self.hir_node_by_def_id(local_id).alias_ty() // it is type alias
            && let Some(alias_generics) = self.hir_node_by_def_id(local_id).generics()
        {
            v.visit_ty_unambig(alias_ty);
            if !v.0.is_empty() {
                return Some((
                    v.0,
                    alias_generics.span,
                    alias_generics.span_for_lifetime_suggestion(),
                ));
            }
        }
        None
    }

    /// Checks if the bound region is in Impl Item.
    pub fn is_bound_region_in_impl_item(self, suitable_region_binding_scope: LocalDefId) -> bool {
        let container_id = self.parent(suitable_region_binding_scope.to_def_id());
        if self.impl_trait_ref(container_id).is_some() {
            // For now, we do not try to target impls of traits. This is
            // because this message is going to suggest that the user
            // change the fn signature, but they may not be free to do so,
            // since the signature must match the trait.
            //
            // FIXME(#42706) -- in some cases, we could do better here.
            return true;
        }
        false
    }

    /// Determines whether identifiers in the assembly have strict naming rules.
    /// Currently, only NVPTX* targets need it.
    pub fn has_strict_asm_symbol_naming(self) -> bool {
        self.sess.target.arch.contains("nvptx")
    }

    /// Returns `&'static core::panic::Location<'static>`.
    pub fn caller_location_ty(self) -> Ty<'tcx> {
        Ty::new_imm_ref(
            self,
            self.lifetimes.re_static,
            self.type_of(self.require_lang_item(LangItem::PanicLocation, DUMMY_SP))
                .instantiate(self, self.mk_args(&[self.lifetimes.re_static.into()])),
        )
    }

    /// Returns a displayable description and article for the given `def_id` (e.g. `("a", "struct")`).
    pub fn article_and_description(self, def_id: DefId) -> (&'static str, &'static str) {
        let kind = self.def_kind(def_id);
        (self.def_kind_descr_article(kind, def_id), self.def_kind_descr(kind, def_id))
    }

    pub fn type_length_limit(self) -> Limit {
        self.limits(()).type_length_limit
    }

    pub fn recursion_limit(self) -> Limit {
        self.limits(()).recursion_limit
    }

    pub fn move_size_limit(self) -> Limit {
        self.limits(()).move_size_limit
    }

    pub fn pattern_complexity_limit(self) -> Limit {
        self.limits(()).pattern_complexity_limit
    }

    /// All traits in the crate graph, including those not visible to the user.
    pub fn all_traits(self) -> impl Iterator<Item = DefId> {
        iter::once(LOCAL_CRATE)
            .chain(self.crates(()).iter().copied())
            .flat_map(move |cnum| self.traits(cnum).iter().copied())
    }

    /// All traits that are visible within the crate graph (i.e. excluding private dependencies).
    pub fn visible_traits(self) -> impl Iterator<Item = DefId> {
        let visible_crates =
            self.crates(()).iter().copied().filter(move |cnum| self.is_user_visible_dep(*cnum));

        iter::once(LOCAL_CRATE)
            .chain(visible_crates)
            .flat_map(move |cnum| self.traits(cnum).iter().copied())
    }

    #[inline]
    pub fn local_visibility(self, def_id: LocalDefId) -> Visibility {
        self.visibility(def_id).expect_local()
    }

    /// Returns the origin of the opaque type `def_id`.
    #[instrument(skip(self), level = "trace", ret)]
    pub fn local_opaque_ty_origin(self, def_id: LocalDefId) -> hir::OpaqueTyOrigin<LocalDefId> {
        self.hir_expect_opaque_ty(def_id).origin
    }

    pub fn finish(self) {
        // We assume that no queries are run past here. If there are new queries
        // after this point, they'll show up as "<unknown>" in self-profiling data.
        self.alloc_self_profile_query_strings();

        self.save_dep_graph();
        self.query_key_hash_verify_all();

        if let Err((path, error)) = self.dep_graph.finish_encoding() {
            self.sess.dcx().emit_fatal(crate::error::FailedWritingFile { path: &path, error });
        }
    }
}

macro_rules! nop_lift {
    ($set:ident; $ty:ty => $lifted:ty) => {
        impl<'a, 'tcx> Lift<TyCtxt<'tcx>> for $ty {
            type Lifted = $lifted;
            fn lift_to_interner(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
                // Assert that the set has the right type.
                // Given an argument that has an interned type, the return type has the type of
                // the corresponding interner set. This won't actually return anything, we're
                // just doing this to compute said type!
                fn _intern_set_ty_from_interned_ty<'tcx, Inner>(
                    _x: Interned<'tcx, Inner>,
                ) -> InternedSet<'tcx, Inner> {
                    unreachable!()
                }
                fn _type_eq<T>(_x: &T, _y: &T) {}
                fn _test<'tcx>(x: $lifted, tcx: TyCtxt<'tcx>) {
                    // If `x` is a newtype around an `Interned<T>`, then `interner` is an
                    // interner of appropriate type. (Ideally we'd also check that `x` is a
                    // newtype with just that one field. Not sure how to do that.)
                    let interner = _intern_set_ty_from_interned_ty(x.0);
                    // Now check that this is the same type as `interners.$set`.
                    _type_eq(&interner, &tcx.interners.$set);
                }

                tcx.interners
                    .$set
                    .contains_pointer_to(&InternedInSet(&*self.0.0))
                    // SAFETY: `self` is interned and therefore valid
                    // for the entire lifetime of the `TyCtxt`.
                    .then(|| unsafe { mem::transmute(self) })
            }
        }
    };
}

macro_rules! nop_list_lift {
    ($set:ident; $ty:ty => $lifted:ty) => {
        impl<'a, 'tcx> Lift<TyCtxt<'tcx>> for &'a List<$ty> {
            type Lifted = &'tcx List<$lifted>;
            fn lift_to_interner(self, tcx: TyCtxt<'tcx>) -> Option<Self::Lifted> {
                // Assert that the set has the right type.
                if false {
                    let _x: &InternedSet<'tcx, List<$lifted>> = &tcx.interners.$set;
                }

                if self.is_empty() {
                    return Some(List::empty());
                }
                tcx.interners
                    .$set
                    .contains_pointer_to(&InternedInSet(self))
                    .then(|| unsafe { mem::transmute(self) })
            }
        }
    };
}

nop_lift! { type_; Ty<'a> => Ty<'tcx> }
nop_lift! { region; Region<'a> => Region<'tcx> }
nop_lift! { const_; Const<'a> => Const<'tcx> }
nop_lift! { pat; Pattern<'a> => Pattern<'tcx> }
nop_lift! { const_allocation; ConstAllocation<'a> => ConstAllocation<'tcx> }
nop_lift! { predicate; Predicate<'a> => Predicate<'tcx> }
nop_lift! { predicate; Clause<'a> => Clause<'tcx> }
nop_lift! { layout; Layout<'a> => Layout<'tcx> }
nop_lift! { valtree; ValTree<'a> => ValTree<'tcx> }

nop_list_lift! { type_lists; Ty<'a> => Ty<'tcx> }
nop_list_lift! {
    poly_existential_predicates; PolyExistentialPredicate<'a> => PolyExistentialPredicate<'tcx>
}
nop_list_lift! { bound_variable_kinds; ty::BoundVariableKind => ty::BoundVariableKind }

// This is the impl for `&'a GenericArgs<'a>`.
nop_list_lift! { args; GenericArg<'a> => GenericArg<'tcx> }

macro_rules! sty_debug_print {
    ($fmt: expr, $ctxt: expr, $($variant: ident),*) => {{
        // Curious inner module to allow variant names to be used as
        // variable names.
        #[allow(non_snake_case)]
        mod inner {
            use crate::ty::{self, TyCtxt};
            use crate::ty::context::InternedInSet;

            #[derive(Copy, Clone)]
            struct DebugStat {
                total: usize,
                lt_infer: usize,
                ty_infer: usize,
                ct_infer: usize,
                all_infer: usize,
            }

            pub(crate) fn go(fmt: &mut std::fmt::Formatter<'_>, tcx: TyCtxt<'_>) -> std::fmt::Result {
                let mut total = DebugStat {
                    total: 0,
                    lt_infer: 0,
                    ty_infer: 0,
                    ct_infer: 0,
                    all_infer: 0,
                };
                $(let mut $variant = total;)*

                for shard in tcx.interners.type_.lock_shards() {
                    // It seems that ordering doesn't affect anything here.
                    #[allow(rustc::potential_query_instability)]
                    let types = shard.iter();
                    for &(InternedInSet(t), ()) in types {
                        let variant = match t.internee {
                            ty::Bool | ty::Char | ty::Int(..) | ty::Uint(..) |
                                ty::Float(..) | ty::Str | ty::Never => continue,
                            ty::Error(_) => /* unimportant */ continue,
                            $(ty::$variant(..) => &mut $variant,)*
                        };
                        let lt = t.flags.intersects(ty::TypeFlags::HAS_RE_INFER);
                        let ty = t.flags.intersects(ty::TypeFlags::HAS_TY_INFER);
                        let ct = t.flags.intersects(ty::TypeFlags::HAS_CT_INFER);

                        variant.total += 1;
                        total.total += 1;
                        if lt { total.lt_infer += 1; variant.lt_infer += 1 }
                        if ty { total.ty_infer += 1; variant.ty_infer += 1 }
                        if ct { total.ct_infer += 1; variant.ct_infer += 1 }
                        if lt && ty && ct { total.all_infer += 1; variant.all_infer += 1 }
                    }
                }
                writeln!(fmt, "Ty interner             total           ty lt ct all")?;
                $(writeln!(fmt, "    {:18}: {uses:6} {usespc:4.1}%, \
                            {ty:4.1}% {lt:5.1}% {ct:4.1}% {all:4.1}%",
                    stringify!($variant),
                    uses = $variant.total,
                    usespc = $variant.total as f64 * 100.0 / total.total as f64,
                    ty = $variant.ty_infer as f64 * 100.0  / total.total as f64,
                    lt = $variant.lt_infer as f64 * 100.0  / total.total as f64,
                    ct = $variant.ct_infer as f64 * 100.0  / total.total as f64,
                    all = $variant.all_infer as f64 * 100.0  / total.total as f64)?;
                )*
                writeln!(fmt, "                  total {uses:6}        \
                          {ty:4.1}% {lt:5.1}% {ct:4.1}% {all:4.1}%",
                    uses = total.total,
                    ty = total.ty_infer as f64 * 100.0  / total.total as f64,
                    lt = total.lt_infer as f64 * 100.0  / total.total as f64,
                    ct = total.ct_infer as f64 * 100.0  / total.total as f64,
                    all = total.all_infer as f64 * 100.0  / total.total as f64)
            }
        }

        inner::go($fmt, $ctxt)
    }}
}

impl<'tcx> TyCtxt<'tcx> {
    pub fn debug_stats(self) -> impl fmt::Debug {
        fmt::from_fn(move |fmt| {
            sty_debug_print!(
                fmt,
                self,
                Adt,
                Array,
                Slice,
                RawPtr,
                Ref,
                FnDef,
                FnPtr,
                UnsafeBinder,
                Placeholder,
                Coroutine,
                CoroutineWitness,
                Dynamic,
                Closure,
                CoroutineClosure,
                Tuple,
                Bound,
                Param,
                Infer,
                Alias,
                Pat,
                Foreign
            )?;

            writeln!(fmt, "GenericArgs interner: #{}", self.interners.args.len())?;
            writeln!(fmt, "Region interner: #{}", self.interners.region.len())?;
            writeln!(fmt, "Const Allocation interner: #{}", self.interners.const_allocation.len())?;
            writeln!(fmt, "Layout interner: #{}", self.interners.layout.len())?;

            Ok(())
        })
    }
}

// This type holds a `T` in the interner. The `T` is stored in the arena and
// this type just holds a pointer to it, but it still effectively owns it. It
// impls `Borrow` so that it can be looked up using the original
// (non-arena-memory-owning) types.
struct InternedInSet<'tcx, T: ?Sized + PointeeSized>(&'tcx T);

impl<'tcx, T: 'tcx + ?Sized + PointeeSized> Clone for InternedInSet<'tcx, T> {
    fn clone(&self) -> Self {
        InternedInSet(self.0)
    }
}

impl<'tcx, T: 'tcx + ?Sized + PointeeSized> Copy for InternedInSet<'tcx, T> {}

impl<'tcx, T: 'tcx + ?Sized + PointeeSized> IntoPointer for InternedInSet<'tcx, T> {
    fn into_pointer(&self) -> *const () {
        self.0 as *const _ as *const ()
    }
}

#[allow(rustc::usage_of_ty_tykind)]
impl<'tcx, T> Borrow<T> for InternedInSet<'tcx, WithCachedTypeInfo<T>> {
    fn borrow(&self) -> &T {
        &self.0.internee
    }
}

impl<'tcx, T: PartialEq> PartialEq for InternedInSet<'tcx, WithCachedTypeInfo<T>> {
    fn eq(&self, other: &InternedInSet<'tcx, WithCachedTypeInfo<T>>) -> bool {
        // The `Borrow` trait requires that `x.borrow() == y.borrow()` equals
        // `x == y`.
        self.0.internee == other.0.internee
    }
}

impl<'tcx, T: Eq> Eq for InternedInSet<'tcx, WithCachedTypeInfo<T>> {}

impl<'tcx, T: Hash> Hash for InternedInSet<'tcx, WithCachedTypeInfo<T>> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        // The `Borrow` trait requires that `x.borrow().hash(s) == x.hash(s)`.
        self.0.internee.hash(s)
    }
}

impl<'tcx, T> Borrow<[T]> for InternedInSet<'tcx, List<T>> {
    fn borrow(&self) -> &[T] {
        &self.0[..]
    }
}

impl<'tcx, T: PartialEq> PartialEq for InternedInSet<'tcx, List<T>> {
    fn eq(&self, other: &InternedInSet<'tcx, List<T>>) -> bool {
        // The `Borrow` trait requires that `x.borrow() == y.borrow()` equals
        // `x == y`.
        self.0[..] == other.0[..]
    }
}

impl<'tcx, T: Eq> Eq for InternedInSet<'tcx, List<T>> {}

impl<'tcx, T: Hash> Hash for InternedInSet<'tcx, List<T>> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        // The `Borrow` trait requires that `x.borrow().hash(s) == x.hash(s)`.
        self.0[..].hash(s)
    }
}

impl<'tcx, T> Borrow<[T]> for InternedInSet<'tcx, ListWithCachedTypeInfo<T>> {
    fn borrow(&self) -> &[T] {
        &self.0[..]
    }
}

impl<'tcx, T: PartialEq> PartialEq for InternedInSet<'tcx, ListWithCachedTypeInfo<T>> {
    fn eq(&self, other: &InternedInSet<'tcx, ListWithCachedTypeInfo<T>>) -> bool {
        // The `Borrow` trait requires that `x.borrow() == y.borrow()` equals
        // `x == y`.
        self.0[..] == other.0[..]
    }
}

impl<'tcx, T: Eq> Eq for InternedInSet<'tcx, ListWithCachedTypeInfo<T>> {}

impl<'tcx, T: Hash> Hash for InternedInSet<'tcx, ListWithCachedTypeInfo<T>> {
    fn hash<H: Hasher>(&self, s: &mut H) {
        // The `Borrow` trait requires that `x.borrow().hash(s) == x.hash(s)`.
        self.0[..].hash(s)
    }
}

macro_rules! direct_interners {
    ($($name:ident: $vis:vis $method:ident($ty:ty): $ret_ctor:ident -> $ret_ty:ty,)+) => {
        $(impl<'tcx> Borrow<$ty> for InternedInSet<'tcx, $ty> {
            fn borrow<'a>(&'a self) -> &'a $ty {
                &self.0
            }
        }

        impl<'tcx> PartialEq for InternedInSet<'tcx, $ty> {
            fn eq(&self, other: &Self) -> bool {
                // The `Borrow` trait requires that `x.borrow() == y.borrow()`
                // equals `x == y`.
                self.0 == other.0
            }
        }

        impl<'tcx> Eq for InternedInSet<'tcx, $ty> {}

        impl<'tcx> Hash for InternedInSet<'tcx, $ty> {
            fn hash<H: Hasher>(&self, s: &mut H) {
                // The `Borrow` trait requires that `x.borrow().hash(s) ==
                // x.hash(s)`.
                self.0.hash(s)
            }
        }

        impl<'tcx> TyCtxt<'tcx> {
            $vis fn $method(self, v: $ty) -> $ret_ty {
                $ret_ctor(Interned::new_unchecked(self.interners.$name.intern(v, |v| {
                    InternedInSet(self.interners.arena.alloc(v))
                }).0))
            }
        })+
    }
}

// Functions with a `mk_` prefix are intended for use outside this file and
// crate. Functions with an `intern_` prefix are intended for use within this
// crate only, and have a corresponding `mk_` function.
direct_interners! {
    region: pub(crate) intern_region(RegionKind<'tcx>): Region -> Region<'tcx>,
    valtree: pub(crate) intern_valtree(ValTreeKind<'tcx>): ValTree -> ValTree<'tcx>,
    pat: pub mk_pat(PatternKind<'tcx>): Pattern -> Pattern<'tcx>,
    const_allocation: pub mk_const_alloc(Allocation): ConstAllocation -> ConstAllocation<'tcx>,
    layout: pub mk_layout(LayoutData<FieldIdx, VariantIdx>): Layout -> Layout<'tcx>,
    adt_def: pub mk_adt_def_from_data(AdtDefData): AdtDef -> AdtDef<'tcx>,
    external_constraints: pub mk_external_constraints(ExternalConstraintsData<TyCtxt<'tcx>>):
        ExternalConstraints -> ExternalConstraints<'tcx>,
    predefined_opaques_in_body: pub mk_predefined_opaques_in_body(PredefinedOpaquesData<TyCtxt<'tcx>>):
        PredefinedOpaques -> PredefinedOpaques<'tcx>,
}

macro_rules! slice_interners {
    ($($field:ident: $vis:vis $method:ident($ty:ty)),+ $(,)?) => (
        impl<'tcx> TyCtxt<'tcx> {
            $($vis fn $method(self, v: &[$ty]) -> &'tcx List<$ty> {
                if v.is_empty() {
                    List::empty()
                } else {
                    self.interners.$field.intern_ref(v, || {
                        InternedInSet(List::from_arena(&*self.arena, (), v))
                    }).0
                }
            })+
        }
    );
}

// These functions intern slices. They all have a corresponding
// `mk_foo_from_iter` function that interns an iterator. The slice version
// should be used when possible, because it's faster.
slice_interners!(
    const_lists: pub mk_const_list(Const<'tcx>),
    args: pub mk_args(GenericArg<'tcx>),
    type_lists: pub mk_type_list(Ty<'tcx>),
    canonical_var_kinds: pub mk_canonical_var_kinds(CanonicalVarKind<'tcx>),
    poly_existential_predicates: intern_poly_existential_predicates(PolyExistentialPredicate<'tcx>),
    projs: pub mk_projs(ProjectionKind),
    place_elems: pub mk_place_elems(PlaceElem<'tcx>),
    bound_variable_kinds: pub mk_bound_variable_kinds(ty::BoundVariableKind),
    fields: pub mk_fields(FieldIdx),
    local_def_ids: intern_local_def_ids(LocalDefId),
    captures: intern_captures(&'tcx ty::CapturedPlace<'tcx>),
    offset_of: pub mk_offset_of((VariantIdx, FieldIdx)),
    patterns: pub mk_patterns(Pattern<'tcx>),
);

impl<'tcx> TyCtxt<'tcx> {
    /// Given a `fn` type, returns an equivalent `unsafe fn` type;
    /// that is, a `fn` type that is equivalent in every way for being
    /// unsafe.
    pub fn safe_to_unsafe_fn_ty(self, sig: PolyFnSig<'tcx>) -> Ty<'tcx> {
        assert!(sig.safety().is_safe());
        Ty::new_fn_ptr(self, sig.map_bound(|sig| ty::FnSig { safety: hir::Safety::Unsafe, ..sig }))
    }

    /// Given the def_id of a Trait `trait_def_id` and the name of an associated item `assoc_name`
    /// returns true if the `trait_def_id` defines an associated item of name `assoc_name`.
    pub fn trait_may_define_assoc_item(self, trait_def_id: DefId, assoc_name: Ident) -> bool {
        elaborate::supertrait_def_ids(self, trait_def_id).any(|trait_did| {
            self.associated_items(trait_did)
                .filter_by_name_unhygienic(assoc_name.name)
                .any(|item| self.hygienic_eq(assoc_name, item.ident(self), trait_did))
        })
    }

    /// Given a `ty`, return whether it's an `impl Future<...>`.
    pub fn ty_is_opaque_future(self, ty: Ty<'_>) -> bool {
        let ty::Alias(ty::Opaque, ty::AliasTy { def_id, .. }) = ty.kind() else { return false };
        let future_trait = self.require_lang_item(LangItem::Future, DUMMY_SP);

        self.explicit_item_self_bounds(def_id).skip_binder().iter().any(|&(predicate, _)| {
            let ty::ClauseKind::Trait(trait_predicate) = predicate.kind().skip_binder() else {
                return false;
            };
            trait_predicate.trait_ref.def_id == future_trait
                && trait_predicate.polarity == PredicatePolarity::Positive
        })
    }

    /// Given a closure signature, returns an equivalent fn signature. Detuples
    /// and so forth -- so e.g., if we have a sig with `Fn<(u32, i32)>` then
    /// you would get a `fn(u32, i32)`.
    /// `unsafety` determines the unsafety of the fn signature. If you pass
    /// `hir::Safety::Unsafe` in the previous example, then you would get
    /// an `unsafe fn (u32, i32)`.
    /// It cannot convert a closure that requires unsafe.
    pub fn signature_unclosure(self, sig: PolyFnSig<'tcx>, safety: hir::Safety) -> PolyFnSig<'tcx> {
        sig.map_bound(|s| {
            let params = match s.inputs()[0].kind() {
                ty::Tuple(params) => *params,
                _ => bug!(),
            };
            self.mk_fn_sig(params, s.output(), s.c_variadic, safety, ExternAbi::Rust)
        })
    }

    #[inline]
    pub fn mk_predicate(self, binder: Binder<'tcx, PredicateKind<'tcx>>) -> Predicate<'tcx> {
        self.interners.intern_predicate(
            binder,
            self.sess,
            // This is only used to create a stable hashing context.
            &self.untracked,
        )
    }

    #[inline]
    pub fn reuse_or_mk_predicate(
        self,
        pred: Predicate<'tcx>,
        binder: Binder<'tcx, PredicateKind<'tcx>>,
    ) -> Predicate<'tcx> {
        if pred.kind() != binder { self.mk_predicate(binder) } else { pred }
    }

    pub fn check_args_compatible(self, def_id: DefId, args: &'tcx [ty::GenericArg<'tcx>]) -> bool {
        self.check_args_compatible_inner(def_id, args, false)
    }

    fn check_args_compatible_inner(
        self,
        def_id: DefId,
        args: &'tcx [ty::GenericArg<'tcx>],
        nested: bool,
    ) -> bool {
        let generics = self.generics_of(def_id);

        // IATs themselves have a weird arg setup (self + own args), but nested items *in* IATs
        // (namely: opaques, i.e. ATPITs) do not.
        let own_args = if !nested
            && let DefKind::AssocTy = self.def_kind(def_id)
            && let DefKind::Impl { of_trait: false } = self.def_kind(self.parent(def_id))
        {
            if generics.own_params.len() + 1 != args.len() {
                return false;
            }

            if !matches!(args[0].kind(), ty::GenericArgKind::Type(_)) {
                return false;
            }

            &args[1..]
        } else {
            if generics.count() != args.len() {
                return false;
            }

            let (parent_args, own_args) = args.split_at(generics.parent_count);

            if let Some(parent) = generics.parent
                && !self.check_args_compatible_inner(parent, parent_args, true)
            {
                return false;
            }

            own_args
        };

        for (param, arg) in std::iter::zip(&generics.own_params, own_args) {
            match (&param.kind, arg.kind()) {
                (ty::GenericParamDefKind::Type { .. }, ty::GenericArgKind::Type(_))
                | (ty::GenericParamDefKind::Lifetime, ty::GenericArgKind::Lifetime(_))
                | (ty::GenericParamDefKind::Const { .. }, ty::GenericArgKind::Const(_)) => {}
                _ => return false,
            }
        }

        true
    }

    /// With `cfg(debug_assertions)`, assert that args are compatible with their generics,
    /// and print out the args if not.
    pub fn debug_assert_args_compatible(self, def_id: DefId, args: &'tcx [ty::GenericArg<'tcx>]) {
        if cfg!(debug_assertions) && !self.check_args_compatible(def_id, args) {
            if let DefKind::AssocTy = self.def_kind(def_id)
                && let DefKind::Impl { of_trait: false } = self.def_kind(self.parent(def_id))
            {
                bug!(
                    "args not compatible with generics for {}: args={:#?}, generics={:#?}",
                    self.def_path_str(def_id),
                    args,
                    // Make `[Self, GAT_ARGS...]` (this could be simplified)
                    self.mk_args_from_iter(
                        [self.types.self_param.into()].into_iter().chain(
                            self.generics_of(def_id)
                                .own_args(ty::GenericArgs::identity_for_item(self, def_id))
                                .iter()
                                .copied()
                        )
                    )
                );
            } else {
                bug!(
                    "args not compatible with generics for {}: args={:#?}, generics={:#?}",
                    self.def_path_str(def_id),
                    args,
                    ty::GenericArgs::identity_for_item(self, def_id)
                );
            }
        }
    }

    #[inline(always)]
    pub(crate) fn check_and_mk_args(
        self,
        def_id: DefId,
        args: impl IntoIterator<Item: Into<GenericArg<'tcx>>>,
    ) -> GenericArgsRef<'tcx> {
        let args = self.mk_args_from_iter(args.into_iter().map(Into::into));
        self.debug_assert_args_compatible(def_id, args);
        args
    }

    #[inline]
    pub fn mk_ct_from_kind(self, kind: ty::ConstKind<'tcx>) -> Const<'tcx> {
        self.interners.intern_const(
            kind,
            self.sess,
            // This is only used to create a stable hashing context.
            &self.untracked,
        )
    }

    // Avoid this in favour of more specific `Ty::new_*` methods, where possible.
    #[allow(rustc::usage_of_ty_tykind)]
    #[inline]
    pub fn mk_ty_from_kind(self, st: TyKind<'tcx>) -> Ty<'tcx> {
        self.interners.intern_ty(
            st,
            self.sess,
            // This is only used to create a stable hashing context.
            &self.untracked,
        )
    }

    pub fn mk_param_from_def(self, param: &ty::GenericParamDef) -> GenericArg<'tcx> {
        match param.kind {
            GenericParamDefKind::Lifetime => {
                ty::Region::new_early_param(self, param.to_early_bound_region_data()).into()
            }
            GenericParamDefKind::Type { .. } => Ty::new_param(self, param.index, param.name).into(),
            GenericParamDefKind::Const { .. } => {
                ty::Const::new_param(self, ParamConst { index: param.index, name: param.name })
                    .into()
            }
        }
    }

    pub fn mk_place_field(self, place: Place<'tcx>, f: FieldIdx, ty: Ty<'tcx>) -> Place<'tcx> {
        self.mk_place_elem(place, PlaceElem::Field(f, ty))
    }

    pub fn mk_place_deref(self, place: Place<'tcx>) -> Place<'tcx> {
        self.mk_place_elem(place, PlaceElem::Deref)
    }

    pub fn mk_place_downcast(
        self,
        place: Place<'tcx>,
        adt_def: AdtDef<'tcx>,
        variant_index: VariantIdx,
    ) -> Place<'tcx> {
        self.mk_place_elem(
            place,
            PlaceElem::Downcast(Some(adt_def.variant(variant_index).name), variant_index),
        )
    }

    pub fn mk_place_downcast_unnamed(
        self,
        place: Place<'tcx>,
        variant_index: VariantIdx,
    ) -> Place<'tcx> {
        self.mk_place_elem(place, PlaceElem::Downcast(None, variant_index))
    }

    pub fn mk_place_index(self, place: Place<'tcx>, index: Local) -> Place<'tcx> {
        self.mk_place_elem(place, PlaceElem::Index(index))
    }

    /// This method copies `Place`'s projection, add an element and reintern it. Should not be used
    /// to build a full `Place` it's just a convenient way to grab a projection and modify it in
    /// flight.
    pub fn mk_place_elem(self, place: Place<'tcx>, elem: PlaceElem<'tcx>) -> Place<'tcx> {
        let mut projection = place.projection.to_vec();
        projection.push(elem);

        Place { local: place.local, projection: self.mk_place_elems(&projection) }
    }

    pub fn mk_poly_existential_predicates(
        self,
        eps: &[PolyExistentialPredicate<'tcx>],
    ) -> &'tcx List<PolyExistentialPredicate<'tcx>> {
        assert!(!eps.is_empty());
        assert!(
            eps.array_windows()
                .all(|[a, b]| a.skip_binder().stable_cmp(self, &b.skip_binder())
                    != Ordering::Greater)
        );
        self.intern_poly_existential_predicates(eps)
    }

    pub fn mk_clauses(self, clauses: &[Clause<'tcx>]) -> Clauses<'tcx> {
        // FIXME consider asking the input slice to be sorted to avoid
        // re-interning permutations, in which case that would be asserted
        // here.
        self.interners.intern_clauses(clauses)
    }

    pub fn mk_local_def_ids(self, def_ids: &[LocalDefId]) -> &'tcx List<LocalDefId> {
        // FIXME consider asking the input slice to be sorted to avoid
        // re-interning permutations, in which case that would be asserted
        // here.
        self.intern_local_def_ids(def_ids)
    }

    pub fn mk_patterns_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<ty::Pattern<'tcx>, &'tcx List<ty::Pattern<'tcx>>>,
    {
        T::collect_and_apply(iter, |xs| self.mk_patterns(xs))
    }

    pub fn mk_local_def_ids_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<LocalDefId, &'tcx List<LocalDefId>>,
    {
        T::collect_and_apply(iter, |xs| self.mk_local_def_ids(xs))
    }

    pub fn mk_captures_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<
                &'tcx ty::CapturedPlace<'tcx>,
                &'tcx List<&'tcx ty::CapturedPlace<'tcx>>,
            >,
    {
        T::collect_and_apply(iter, |xs| self.intern_captures(xs))
    }

    pub fn mk_const_list_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<ty::Const<'tcx>, &'tcx List<ty::Const<'tcx>>>,
    {
        T::collect_and_apply(iter, |xs| self.mk_const_list(xs))
    }

    // Unlike various other `mk_*_from_iter` functions, this one uses `I:
    // IntoIterator` instead of `I: Iterator`, and it doesn't have a slice
    // variant, because of the need to combine `inputs` and `output`. This
    // explains the lack of `_from_iter` suffix.
    pub fn mk_fn_sig<I, T>(
        self,
        inputs: I,
        output: I::Item,
        c_variadic: bool,
        safety: hir::Safety,
        abi: ExternAbi,
    ) -> T::Output
    where
        I: IntoIterator<Item = T>,
        T: CollectAndApply<Ty<'tcx>, ty::FnSig<'tcx>>,
    {
        T::collect_and_apply(inputs.into_iter().chain(iter::once(output)), |xs| ty::FnSig {
            inputs_and_output: self.mk_type_list(xs),
            c_variadic,
            safety,
            abi,
        })
    }

    pub fn mk_poly_existential_predicates_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<
                PolyExistentialPredicate<'tcx>,
                &'tcx List<PolyExistentialPredicate<'tcx>>,
            >,
    {
        T::collect_and_apply(iter, |xs| self.mk_poly_existential_predicates(xs))
    }

    pub fn mk_clauses_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<Clause<'tcx>, Clauses<'tcx>>,
    {
        T::collect_and_apply(iter, |xs| self.mk_clauses(xs))
    }

    pub fn mk_type_list_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<Ty<'tcx>, &'tcx List<Ty<'tcx>>>,
    {
        T::collect_and_apply(iter, |xs| self.mk_type_list(xs))
    }

    pub fn mk_args_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<GenericArg<'tcx>, ty::GenericArgsRef<'tcx>>,
    {
        T::collect_and_apply(iter, |xs| self.mk_args(xs))
    }

    pub fn mk_canonical_var_infos_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<CanonicalVarKind<'tcx>, &'tcx List<CanonicalVarKind<'tcx>>>,
    {
        T::collect_and_apply(iter, |xs| self.mk_canonical_var_kinds(xs))
    }

    pub fn mk_place_elems_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<PlaceElem<'tcx>, &'tcx List<PlaceElem<'tcx>>>,
    {
        T::collect_and_apply(iter, |xs| self.mk_place_elems(xs))
    }

    pub fn mk_fields_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<FieldIdx, &'tcx List<FieldIdx>>,
    {
        T::collect_and_apply(iter, |xs| self.mk_fields(xs))
    }

    pub fn mk_offset_of_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<(VariantIdx, FieldIdx), &'tcx List<(VariantIdx, FieldIdx)>>,
    {
        T::collect_and_apply(iter, |xs| self.mk_offset_of(xs))
    }

    pub fn mk_args_trait(
        self,
        self_ty: Ty<'tcx>,
        rest: impl IntoIterator<Item = GenericArg<'tcx>>,
    ) -> GenericArgsRef<'tcx> {
        self.mk_args_from_iter(iter::once(self_ty.into()).chain(rest))
    }

    pub fn mk_bound_variable_kinds_from_iter<I, T>(self, iter: I) -> T::Output
    where
        I: Iterator<Item = T>,
        T: CollectAndApply<ty::BoundVariableKind, &'tcx List<ty::BoundVariableKind>>,
    {
        T::collect_and_apply(iter, |xs| self.mk_bound_variable_kinds(xs))
    }

    /// Emit a lint at `span` from a lint struct (some type that implements `LintDiagnostic`,
    /// typically generated by `#[derive(LintDiagnostic)]`).
    #[track_caller]
    pub fn emit_node_span_lint(
        self,
        lint: &'static Lint,
        hir_id: HirId,
        span: impl Into<MultiSpan>,
        decorator: impl for<'a> LintDiagnostic<'a, ()>,
    ) {
        let level = self.lint_level_at_node(lint, hir_id);
        lint_level(self.sess, lint, level, Some(span.into()), |lint| {
            decorator.decorate_lint(lint);
        })
    }

    /// Emit a lint at the appropriate level for a hir node, with an associated span.
    ///
    /// [`lint_level`]: rustc_middle::lint::lint_level#decorate-signature
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn node_span_lint(
        self,
        lint: &'static Lint,
        hir_id: HirId,
        span: impl Into<MultiSpan>,
        decorate: impl for<'a, 'b> FnOnce(&'b mut Diag<'a, ()>),
    ) {
        let level = self.lint_level_at_node(lint, hir_id);
        lint_level(self.sess, lint, level, Some(span.into()), decorate);
    }

    /// Find the appropriate span where `use` and outer attributes can be inserted at.
    pub fn crate_level_attribute_injection_span(self) -> Span {
        let node = self.hir_node(hir::CRATE_HIR_ID);
        let hir::Node::Crate(m) = node else { bug!() };
        m.spans.inject_use_span.shrink_to_lo()
    }

    pub fn disabled_nightly_features<E: rustc_errors::EmissionGuarantee>(
        self,
        diag: &mut Diag<'_, E>,
        features: impl IntoIterator<Item = (String, Symbol)>,
    ) {
        if !self.sess.is_nightly_build() {
            return;
        }

        let span = self.crate_level_attribute_injection_span();
        for (desc, feature) in features {
            // FIXME: make this string translatable
            let msg =
                format!("add `#![feature({feature})]` to the crate attributes to enable{desc}");
            diag.span_suggestion_verbose(
                span,
                msg,
                format!("#![feature({feature})]\n"),
                Applicability::MaybeIncorrect,
            );
        }
    }

    /// Emit a lint from a lint struct (some type that implements `LintDiagnostic`, typically
    /// generated by `#[derive(LintDiagnostic)]`).
    #[track_caller]
    pub fn emit_node_lint(
        self,
        lint: &'static Lint,
        id: HirId,
        decorator: impl for<'a> LintDiagnostic<'a, ()>,
    ) {
        self.node_lint(lint, id, |lint| {
            decorator.decorate_lint(lint);
        })
    }

    /// Emit a lint at the appropriate level for a hir node.
    ///
    /// [`lint_level`]: rustc_middle::lint::lint_level#decorate-signature
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn node_lint(
        self,
        lint: &'static Lint,
        id: HirId,
        decorate: impl for<'a, 'b> FnOnce(&'b mut Diag<'a, ()>),
    ) {
        let level = self.lint_level_at_node(lint, id);
        lint_level(self.sess, lint, level, None, decorate);
    }

    pub fn in_scope_traits(self, id: HirId) -> Option<&'tcx [TraitCandidate]> {
        let map = self.in_scope_traits_map(id.owner)?;
        let candidates = map.get(&id.local_id)?;
        Some(candidates)
    }

    pub fn named_bound_var(self, id: HirId) -> Option<resolve_bound_vars::ResolvedArg> {
        debug!(?id, "named_region");
        self.named_variable_map(id.owner).get(&id.local_id).cloned()
    }

    pub fn is_late_bound(self, id: HirId) -> bool {
        self.is_late_bound_map(id.owner).is_some_and(|set| set.contains(&id.local_id))
    }

    pub fn late_bound_vars(self, id: HirId) -> &'tcx List<ty::BoundVariableKind> {
        self.mk_bound_variable_kinds(
            &self
                .late_bound_vars_map(id.owner)
                .get(&id.local_id)
                .cloned()
                .unwrap_or_else(|| bug!("No bound vars found for {}", self.hir_id_to_string(id))),
        )
    }

    /// Given the def-id of an early-bound lifetime on an opaque corresponding to
    /// a duplicated captured lifetime, map it back to the early- or late-bound
    /// lifetime of the function from which it originally as captured. If it is
    /// a late-bound lifetime, this will represent the liberated (`ReLateParam`) lifetime
    /// of the signature.
    // FIXME(RPITIT): if we ever synthesize new lifetimes for RPITITs and not just
    // re-use the generics of the opaque, this function will need to be tweaked slightly.
    pub fn map_opaque_lifetime_to_parent_lifetime(
        self,
        mut opaque_lifetime_param_def_id: LocalDefId,
    ) -> ty::Region<'tcx> {
        debug_assert!(
            matches!(self.def_kind(opaque_lifetime_param_def_id), DefKind::LifetimeParam),
            "{opaque_lifetime_param_def_id:?} is a {}",
            self.def_descr(opaque_lifetime_param_def_id.to_def_id())
        );

        loop {
            let parent = self.local_parent(opaque_lifetime_param_def_id);
            let lifetime_mapping = self.opaque_captured_lifetimes(parent);

            let Some((lifetime, _)) = lifetime_mapping
                .iter()
                .find(|(_, duplicated_param)| *duplicated_param == opaque_lifetime_param_def_id)
            else {
                bug!("duplicated lifetime param should be present");
            };

            match *lifetime {
                resolve_bound_vars::ResolvedArg::EarlyBound(ebv) => {
                    let new_parent = self.local_parent(ebv);

                    // If we map to another opaque, then it should be a parent
                    // of the opaque we mapped from. Continue mapping.
                    if matches!(self.def_kind(new_parent), DefKind::OpaqueTy) {
                        debug_assert_eq!(self.local_parent(parent), new_parent);
                        opaque_lifetime_param_def_id = ebv;
                        continue;
                    }

                    let generics = self.generics_of(new_parent);
                    return ty::Region::new_early_param(
                        self,
                        ty::EarlyParamRegion {
                            index: generics
                                .param_def_id_to_index(self, ebv.to_def_id())
                                .expect("early-bound var should be present in fn generics"),
                            name: self.item_name(ebv.to_def_id()),
                        },
                    );
                }
                resolve_bound_vars::ResolvedArg::LateBound(_, _, lbv) => {
                    let new_parent = self.local_parent(lbv);
                    return ty::Region::new_late_param(
                        self,
                        new_parent.to_def_id(),
                        ty::LateParamRegionKind::Named(
                            lbv.to_def_id(),
                            self.item_name(lbv.to_def_id()),
                        ),
                    );
                }
                resolve_bound_vars::ResolvedArg::Error(guar) => {
                    return ty::Region::new_error(self, guar);
                }
                _ => {
                    return ty::Region::new_error_with_message(
                        self,
                        self.def_span(opaque_lifetime_param_def_id),
                        "cannot resolve lifetime",
                    );
                }
            }
        }
    }

    /// Whether `def_id` is a stable const fn (i.e., doesn't need any feature gates to be called).
    ///
    /// When this is `false`, the function may still be callable as a `const fn` due to features
    /// being enabled!
    pub fn is_stable_const_fn(self, def_id: DefId) -> bool {
        self.is_const_fn(def_id)
            && match self.lookup_const_stability(def_id) {
                None => true, // a fn in a non-staged_api crate
                Some(stability) if stability.is_const_stable() => true,
                _ => false,
            }
    }

    /// Whether the trait impl is marked const. This does not consider stability or feature gates.
    pub fn is_const_trait_impl(self, def_id: DefId) -> bool {
        self.def_kind(def_id) == DefKind::Impl { of_trait: true }
            && self.impl_trait_header(def_id).unwrap().constness == hir::Constness::Const
    }

    pub fn is_sdylib_interface_build(self) -> bool {
        self.sess.opts.unstable_opts.build_sdylib_interface
    }

    pub fn intrinsic(self, def_id: impl IntoQueryParam<DefId> + Copy) -> Option<ty::IntrinsicDef> {
        match self.def_kind(def_id) {
            DefKind::Fn | DefKind::AssocFn => {}
            _ => return None,
        }
        self.intrinsic_raw(def_id)
    }

    pub fn next_trait_solver_globally(self) -> bool {
        self.sess.opts.unstable_opts.next_solver.globally
    }

    pub fn next_trait_solver_in_coherence(self) -> bool {
        self.sess.opts.unstable_opts.next_solver.coherence
    }

    #[allow(rustc::bad_opt_access)]
    pub fn use_typing_mode_borrowck(self) -> bool {
        self.next_trait_solver_globally() || self.sess.opts.unstable_opts.typing_mode_borrowck
    }

    pub fn is_impl_trait_in_trait(self, def_id: DefId) -> bool {
        self.opt_rpitit_info(def_id).is_some()
    }

    /// Named module children from all kinds of items, including imports.
    /// In addition to regular items this list also includes struct and variant constructors, and
    /// items inside `extern {}` blocks because all of them introduce names into parent module.
    ///
    /// Module here is understood in name resolution sense - it can be a `mod` item,
    /// or a crate root, or an enum, or a trait.
    ///
    /// This is not a query, making it a query causes perf regressions
    /// (probably due to hashing spans in `ModChild`ren).
    pub fn module_children_local(self, def_id: LocalDefId) -> &'tcx [ModChild] {
        self.resolutions(()).module_children.get(&def_id).map_or(&[], |v| &v[..])
    }

    pub fn resolver_for_lowering(self) -> &'tcx Steal<(ty::ResolverAstLowering, Arc<ast::Crate>)> {
        self.resolver_for_lowering_raw(()).0
    }

    /// Given an `impl_id`, return the trait it implements.
    /// Return `None` if this is an inherent impl.
    pub fn impl_trait_ref(
        self,
        def_id: impl IntoQueryParam<DefId>,
    ) -> Option<ty::EarlyBinder<'tcx, ty::TraitRef<'tcx>>> {
        Some(self.impl_trait_header(def_id)?.trait_ref)
    }

    pub fn impl_polarity(self, def_id: impl IntoQueryParam<DefId>) -> ty::ImplPolarity {
        self.impl_trait_header(def_id).map_or(ty::ImplPolarity::Positive, |h| h.polarity)
    }

    pub fn needs_coroutine_by_move_body_def_id(self, def_id: DefId) -> bool {
        if let Some(hir::CoroutineKind::Desugared(_, hir::CoroutineSource::Closure)) =
            self.coroutine_kind(def_id)
            && let ty::Coroutine(_, args) = self.type_of(def_id).instantiate_identity().kind()
            && args.as_coroutine().kind_ty().to_opt_closure_kind() != Some(ty::ClosureKind::FnOnce)
        {
            true
        } else {
            false
        }
    }

    /// Whether this is a trait implementation that has `#[diagnostic::do_not_recommend]`
    pub fn do_not_recommend_impl(self, def_id: DefId) -> bool {
        self.get_diagnostic_attr(def_id, sym::do_not_recommend).is_some()
    }
}

/// Parameter attributes that can only be determined by examining the body of a function instead
/// of just its signature.
///
/// These can be useful for optimization purposes when a function is directly called. We compute
/// them and store them into the crate metadata so that downstream crates can make use of them.
///
/// Right now, we only have `read_only`, but `no_capture` and `no_alias` might be useful in the
/// future.
#[derive(Clone, Copy, PartialEq, Debug, Default, TyDecodable, TyEncodable, HashStable)]
pub struct DeducedParamAttrs {
    /// The parameter is marked immutable in the function and contains no `UnsafeCell` (i.e. its
    /// type is freeze).
    pub read_only: bool,
}

pub fn provide(providers: &mut Providers) {
    providers.maybe_unused_trait_imports =
        |tcx, ()| &tcx.resolutions(()).maybe_unused_trait_imports;
    providers.names_imported_by_glob_use = |tcx, id| {
        tcx.arena.alloc(tcx.resolutions(()).glob_map.get(&id).cloned().unwrap_or_default())
    };

    providers.extern_mod_stmt_cnum =
        |tcx, id| tcx.resolutions(()).extern_crate_map.get(&id).cloned();
    providers.is_panic_runtime =
        |tcx, LocalCrate| contains_name(tcx.hir_krate_attrs(), sym::panic_runtime);
    providers.is_compiler_builtins =
        |tcx, LocalCrate| contains_name(tcx.hir_krate_attrs(), sym::compiler_builtins);
    providers.has_panic_handler = |tcx, LocalCrate| {
        // We want to check if the panic handler was defined in this crate
        tcx.lang_items().panic_impl().is_some_and(|did| did.is_local())
    };
    providers.source_span = |tcx, def_id| tcx.untracked.source_span.get(def_id).unwrap_or(DUMMY_SP);
}

pub fn contains_name(attrs: &[Attribute], name: Symbol) -> bool {
    attrs.iter().any(|x| x.has_name(name))
}
