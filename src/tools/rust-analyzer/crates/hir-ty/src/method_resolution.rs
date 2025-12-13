//! This module is concerned with finding methods that a given type provides.
//! For details about how this works in rustc, see the method lookup page in the
//! [rustc guide](https://rust-lang.github.io/rustc-guide/method-lookup.html)
//! and the corresponding code mostly in rustc_hir_analysis/check/method/probe.rs.

mod confirm;
mod probe;

use either::Either;
use hir_expand::name::Name;
use span::Edition;
use tracing::{debug, instrument};

use base_db::Crate;
use hir_def::{
    AssocItemId, BlockIdLt, ConstId, FunctionId, GenericParamId, HasModule, ImplId,
    ItemContainerId, ModuleId, TraitId,
    attrs::AttrFlags,
    expr_store::path::GenericArgs as HirGenericArgs,
    hir::ExprId,
    nameres::{DefMap, block_def_map, crate_def_map},
    resolver::Resolver,
};
use intern::{Symbol, sym};
use rustc_hash::{FxHashMap, FxHashSet};
use rustc_type_ir::{
    TypeVisitableExt,
    fast_reject::{TreatParams, simplify_type},
    inherent::{BoundExistentialPredicates, IntoKind, SliceLike},
};
use stdx::impl_from;
use triomphe::Arc;

use crate::{
    all_super_traits,
    db::HirDatabase,
    infer::{InferenceContext, unify::InferenceTable},
    lower::GenericPredicates,
    next_solver::{
        Binder, ClauseKind, DbInterner, FnSig, GenericArgs, ParamEnv, PredicateKind,
        SimplifiedType, SolverDefId, TraitRef, Ty, TyKind, TypingMode,
        infer::{
            BoundRegionConversionTime, DbInternerInferExt, InferCtxt, InferOk,
            select::ImplSource,
            traits::{Obligation, ObligationCause, PredicateObligations},
        },
        obligation_ctxt::ObligationCtxt,
        util::clauses_as_obligations,
    },
    traits::ParamEnvAndCrate,
};

pub use self::probe::{
    Candidate, CandidateKind, CandidateStep, CandidateWithPrivate, Mode, Pick, PickKind,
};

#[derive(Debug, Clone)]
pub struct MethodResolutionUnstableFeatures {
    arbitrary_self_types: bool,
    arbitrary_self_types_pointers: bool,
    supertrait_item_shadowing: bool,
}

impl MethodResolutionUnstableFeatures {
    pub fn from_def_map(def_map: &DefMap) -> Self {
        Self {
            arbitrary_self_types: def_map.is_unstable_feature_enabled(&sym::arbitrary_self_types),
            arbitrary_self_types_pointers: def_map
                .is_unstable_feature_enabled(&sym::arbitrary_self_types_pointers),
            supertrait_item_shadowing: def_map
                .is_unstable_feature_enabled(&sym::supertrait_item_shadowing),
        }
    }
}

pub struct MethodResolutionContext<'a, 'db> {
    pub infcx: &'a InferCtxt<'db>,
    pub resolver: &'a Resolver<'db>,
    pub param_env: ParamEnv<'db>,
    pub traits_in_scope: &'a FxHashSet<TraitId>,
    pub edition: Edition,
    pub unstable_features: &'a MethodResolutionUnstableFeatures,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, salsa::Update)]
pub enum CandidateId {
    FunctionId(FunctionId),
    ConstId(ConstId),
}
impl_from!(FunctionId, ConstId for CandidateId);

impl CandidateId {
    fn container(self, db: &dyn HirDatabase) -> ItemContainerId {
        match self {
            CandidateId::FunctionId(id) => id.loc(db).container,
            CandidateId::ConstId(id) => id.loc(db).container,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MethodCallee<'db> {
    /// Impl method ID, for inherent methods, or trait method ID, otherwise.
    pub def_id: FunctionId,
    pub args: GenericArgs<'db>,

    /// Instantiated method signature, i.e., it has been
    /// instantiated, normalized, and has had late-bound
    /// lifetimes replaced with inference variables.
    pub sig: FnSig<'db>,
}

#[derive(Debug)]
pub enum MethodError<'db> {
    /// Did not find an applicable method.
    NoMatch,

    /// Multiple methods might apply.
    Ambiguity(Vec<CandidateSource>),

    /// Found an applicable method, but it is not visible.
    PrivateMatch(Pick<'db>),

    /// Found a `Self: Sized` bound where `Self` is a trait object.
    IllegalSizedBound { candidates: Vec<FunctionId>, needs_mut: bool },

    /// Error has already been emitted, no need to emit another one.
    ErrorReported,
}

// A pared down enum describing just the places from which a method
// candidate can arise. Used for error reporting only.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CandidateSource {
    Impl(ImplId),
    Trait(TraitId),
}

impl<'a, 'db> InferenceContext<'a, 'db> {
    /// Performs method lookup. If lookup is successful, it will return the callee
    /// and store an appropriate adjustment for the self-expr. In some cases it may
    /// report an error (e.g., invoking the `drop` method).
    #[instrument(level = "debug", skip(self))]
    pub(crate) fn lookup_method_including_private(
        &mut self,
        self_ty: Ty<'db>,
        name: Name,
        generic_args: Option<&HirGenericArgs>,
        receiver: ExprId,
        call_expr: ExprId,
    ) -> Result<(MethodCallee<'db>, bool), MethodError<'db>> {
        let (pick, is_visible) = match self.lookup_probe(name, self_ty) {
            Ok(it) => (it, true),
            Err(MethodError::PrivateMatch(it)) => {
                // FIXME: Report error.
                (it, false)
            }
            Err(err) => return Err(err),
        };

        let result = self.confirm_method(&pick, self_ty, call_expr, generic_args);
        debug!("result = {:?}", result);

        if result.illegal_sized_bound {
            // FIXME: Report an error.
        }

        self.write_expr_adj(receiver, result.adjustments);
        self.write_method_resolution(call_expr, result.callee.def_id, result.callee.args);

        Ok((result.callee, is_visible))
    }

    #[instrument(level = "debug", skip(self))]
    pub(crate) fn lookup_probe(
        &self,
        method_name: Name,
        self_ty: Ty<'db>,
    ) -> probe::PickResult<'db> {
        self.with_method_resolution(|ctx| {
            let pick = ctx.probe_for_name(probe::Mode::MethodCall, method_name, self_ty)?;
            Ok(pick)
        })
    }

    pub(crate) fn with_method_resolution<R>(
        &self,
        f: impl FnOnce(&MethodResolutionContext<'_, 'db>) -> R,
    ) -> R {
        let traits_in_scope = self.get_traits_in_scope();
        let traits_in_scope = match &traits_in_scope {
            Either::Left(it) => it,
            Either::Right(it) => *it,
        };
        let ctx = MethodResolutionContext {
            infcx: &self.table.infer_ctxt,
            resolver: &self.resolver,
            param_env: self.table.param_env,
            traits_in_scope,
            edition: self.edition,
            unstable_features: &self.unstable_features,
        };
        f(&ctx)
    }
}

/// Used by [FnCtxt::lookup_method_for_operator] with `-Znext-solver`.
///
/// With `AsRigid` we error on `impl Opaque: NotInItemBounds` while
/// `AsInfer` just treats it as ambiguous and succeeds. This is necessary
/// as we want [FnCtxt::check_expr_call] to treat not-yet-defined opaque
/// types as rigid to support `impl Deref<Target = impl FnOnce()>` and
/// `Box<impl FnOnce()>`.
///
/// We only want to treat opaque types as rigid if we need to eagerly choose
/// between multiple candidates. We otherwise treat them as ordinary inference
/// variable to avoid rejecting otherwise correct code.
#[derive(Debug)]
#[expect(dead_code)]
pub(super) enum TreatNotYetDefinedOpaques {
    AsInfer,
    AsRigid,
}

impl<'db> InferenceTable<'db> {
    /// `lookup_method_in_trait` is used for overloaded operators.
    /// It does a very narrow slice of what the normal probe/confirm path does.
    /// In particular, it doesn't really do any probing: it simply constructs
    /// an obligation for a particular trait with the given self type and checks
    /// whether that trait is implemented.
    #[instrument(level = "debug", skip(self))]
    pub(super) fn lookup_method_for_operator(
        &self,
        cause: ObligationCause,
        method_name: Symbol,
        trait_def_id: TraitId,
        self_ty: Ty<'db>,
        opt_rhs_ty: Option<Ty<'db>>,
        treat_opaques: TreatNotYetDefinedOpaques,
    ) -> Option<InferOk<'db, MethodCallee<'db>>> {
        // Construct a trait-reference `self_ty : Trait<input_tys>`
        let args = GenericArgs::for_item(
            self.interner(),
            trait_def_id.into(),
            |param_idx, param_id, _| match param_id {
                GenericParamId::LifetimeParamId(_) | GenericParamId::ConstParamId(_) => {
                    unreachable!("did not expect operator trait to have lifetime/const")
                }
                GenericParamId::TypeParamId(_) => {
                    if param_idx == 0 {
                        self_ty.into()
                    } else if let Some(rhs_ty) = opt_rhs_ty {
                        assert_eq!(param_idx, 1, "did not expect >1 param on operator trait");
                        rhs_ty.into()
                    } else {
                        // FIXME: We should stop passing `None` for the failure case
                        // when probing for call exprs. I.e. `opt_rhs_ty` should always
                        // be set when it needs to be.
                        self.next_var_for_param(param_id)
                    }
                }
            },
        );

        let obligation = Obligation::new(
            self.interner(),
            cause,
            self.param_env,
            TraitRef::new_from_args(self.interner(), trait_def_id.into(), args),
        );

        // Now we want to know if this can be matched
        let matches_trait = match treat_opaques {
            TreatNotYetDefinedOpaques::AsInfer => self.infer_ctxt.predicate_may_hold(&obligation),
            TreatNotYetDefinedOpaques::AsRigid => {
                self.infer_ctxt.predicate_may_hold_opaque_types_jank(&obligation)
            }
        };

        if !matches_trait {
            debug!("--> Cannot match obligation");
            // Cannot be matched, no such method resolution is possible.
            return None;
        }

        // Trait must have a method named `m_name` and it should not have
        // type parameters or early-bound regions.
        let interner = self.interner();
        // We use `Ident::with_dummy_span` since no built-in operator methods have
        // any macro-specific hygiene, so the span's context doesn't really matter.
        let Some(method_item) =
            trait_def_id.trait_items(self.db).method_by_name(&Name::new_symbol_root(method_name))
        else {
            panic!("expected associated item for operator trait")
        };

        let def_id = method_item;

        debug!("lookup_in_trait_adjusted: method_item={:?}", method_item);
        let mut obligations = PredicateObligations::new();

        // Instantiate late-bound regions and instantiate the trait
        // parameters into the method type to get the actual method type.
        //
        // N.B., instantiate late-bound regions before normalizing the
        // function signature so that normalization does not need to deal
        // with bound regions.
        let fn_sig =
            self.db.callable_item_signature(method_item.into()).instantiate(interner, args);
        let fn_sig = self
            .infer_ctxt
            .instantiate_binder_with_fresh_vars(BoundRegionConversionTime::FnCall, fn_sig);

        // Register obligations for the parameters. This will include the
        // `Self` parameter, which in turn has a bound of the main trait,
        // so this also effectively registers `obligation` as well. (We
        // used to register `obligation` explicitly, but that resulted in
        // double error messages being reported.)
        //
        // Note that as the method comes from a trait, it should not have
        // any late-bound regions appearing in its bounds.
        let bounds = GenericPredicates::query_all(self.db, method_item.into());
        let bounds = clauses_as_obligations(
            bounds.iter_instantiated_copied(interner, args.as_slice()),
            ObligationCause::new(),
            self.param_env,
        );

        obligations.extend(bounds);

        // Also add an obligation for the method type being well-formed.
        debug!(
            "lookup_method_in_trait: matched method fn_sig={:?} obligation={:?}",
            fn_sig, obligation
        );
        for ty in fn_sig.inputs_and_output {
            obligations.push(Obligation::new(
                interner,
                obligation.cause.clone(),
                self.param_env,
                Binder::dummy(PredicateKind::Clause(ClauseKind::WellFormed(ty.into()))),
            ));
        }

        let callee = MethodCallee { def_id, args, sig: fn_sig };
        debug!("callee = {:?}", callee);

        Some(InferOk { obligations, value: callee })
    }
}

pub fn lookup_impl_const<'db>(
    infcx: &InferCtxt<'db>,
    env: ParamEnv<'db>,
    const_id: ConstId,
    subs: GenericArgs<'db>,
) -> (ConstId, GenericArgs<'db>) {
    let interner = infcx.interner;
    let db = interner.db;

    let trait_id = match const_id.loc(db).container {
        ItemContainerId::TraitId(id) => id,
        _ => return (const_id, subs),
    };
    let trait_ref = TraitRef::new(interner, trait_id.into(), subs);

    let const_signature = db.const_signature(const_id);
    let name = match const_signature.name.as_ref() {
        Some(name) => name,
        None => return (const_id, subs),
    };

    lookup_impl_assoc_item_for_trait_ref(infcx, trait_ref, env, name)
        .and_then(
            |assoc| if let (AssocItemId::ConstId(id), s) = assoc { Some((id, s)) } else { None },
        )
        .unwrap_or((const_id, subs))
}

/// Checks if the self parameter of `Trait` method is the `dyn Trait` and we should
/// call the method using the vtable.
pub fn is_dyn_method<'db>(
    interner: DbInterner<'db>,
    _env: ParamEnv<'db>,
    func: FunctionId,
    fn_subst: GenericArgs<'db>,
) -> Option<usize> {
    let db = interner.db;

    let ItemContainerId::TraitId(trait_id) = func.loc(db).container else {
        return None;
    };
    let trait_params = db.generic_params(trait_id.into()).len();
    let fn_params = fn_subst.len() - trait_params;
    let trait_ref = TraitRef::new(
        interner,
        trait_id.into(),
        GenericArgs::new_from_iter(interner, fn_subst.iter().take(trait_params)),
    );
    let self_ty = trait_ref.self_ty();
    if let TyKind::Dynamic(d, _) = self_ty.kind() {
        // rustc doesn't accept `impl Foo<2> for dyn Foo<5>`, so if the trait id is equal, no matter
        // what the generics are, we are sure that the method is come from the vtable.
        let is_my_trait_in_bounds = d
            .principal_def_id()
            .is_some_and(|trait_| all_super_traits(db, trait_.0).contains(&trait_id));
        if is_my_trait_in_bounds {
            return Some(fn_params);
        }
    }
    None
}

/// Looks up the impl method that actually runs for the trait method `func`.
///
/// Returns `func` if it's not a method defined in a trait or the lookup failed.
pub(crate) fn lookup_impl_method_query<'db>(
    db: &'db dyn HirDatabase,
    env: ParamEnvAndCrate<'db>,
    func: FunctionId,
    fn_subst: GenericArgs<'db>,
) -> (FunctionId, GenericArgs<'db>) {
    let interner = DbInterner::new_with(db, env.krate);
    let infcx = interner.infer_ctxt().build(TypingMode::PostAnalysis);

    let ItemContainerId::TraitId(trait_id) = func.loc(db).container else {
        return (func, fn_subst);
    };
    let trait_params = db.generic_params(trait_id.into()).len();
    let trait_ref = TraitRef::new(
        interner,
        trait_id.into(),
        GenericArgs::new_from_iter(interner, fn_subst.iter().take(trait_params)),
    );

    let name = &db.function_signature(func).name;
    let Some((impl_fn, impl_subst)) = lookup_impl_assoc_item_for_trait_ref(
        &infcx,
        trait_ref,
        env.param_env,
        name,
    )
    .and_then(|assoc| {
        if let (AssocItemId::FunctionId(id), subst) = assoc { Some((id, subst)) } else { None }
    }) else {
        return (func, fn_subst);
    };

    (
        impl_fn,
        GenericArgs::new_from_iter(
            interner,
            impl_subst.iter().chain(fn_subst.iter().skip(trait_params)),
        ),
    )
}

fn lookup_impl_assoc_item_for_trait_ref<'db>(
    infcx: &InferCtxt<'db>,
    trait_ref: TraitRef<'db>,
    env: ParamEnv<'db>,
    name: &Name,
) -> Option<(AssocItemId, GenericArgs<'db>)> {
    let (impl_id, impl_subst) = find_matching_impl(infcx, env, trait_ref)?;
    let item =
        impl_id.impl_items(infcx.interner.db).items.iter().find_map(|(n, it)| match *it {
            AssocItemId::FunctionId(f) => (n == name).then_some(AssocItemId::FunctionId(f)),
            AssocItemId::ConstId(c) => (n == name).then_some(AssocItemId::ConstId(c)),
            AssocItemId::TypeAliasId(_) => None,
        })?;
    Some((item, impl_subst))
}

pub(crate) fn find_matching_impl<'db>(
    infcx: &InferCtxt<'db>,
    env: ParamEnv<'db>,
    trait_ref: TraitRef<'db>,
) -> Option<(ImplId, GenericArgs<'db>)> {
    let trait_ref = infcx.at(&ObligationCause::dummy(), env).deeply_normalize(trait_ref).ok()?;

    let obligation = Obligation::new(infcx.interner, ObligationCause::dummy(), env, trait_ref);

    let selection = infcx.select(&obligation).ok()??;

    // Currently, we use a fulfillment context to completely resolve
    // all nested obligations. This is because they can inform the
    // inference of the impl's type parameters.
    let mut ocx = ObligationCtxt::new(infcx);
    let impl_source = selection.map(|obligation| ocx.register_obligation(obligation));

    let errors = ocx.evaluate_obligations_error_on_ambiguity();
    if !errors.is_empty() {
        return None;
    }

    let impl_source = infcx.resolve_vars_if_possible(impl_source);
    if impl_source.has_non_region_infer() {
        return None;
    }

    match impl_source {
        ImplSource::UserDefined(impl_source) => Some((impl_source.impl_def_id, impl_source.args)),
        ImplSource::Param(_) | ImplSource::Builtin(..) => None,
    }
}

#[salsa::tracked(returns(ref))]
fn crates_containing_incoherent_inherent_impls(db: &dyn HirDatabase) -> Box<[Crate]> {
    // We assume that only sysroot crates contain `#[rustc_has_incoherent_inherent_impls]`
    // impls, since this is an internal feature and only std uses it.
    db.all_crates().iter().copied().filter(|krate| krate.data(db).origin.is_lang()).collect()
}

pub fn incoherent_inherent_impls(db: &dyn HirDatabase, self_ty: SimplifiedType) -> &[ImplId] {
    let has_incoherent_impls = match self_ty.def() {
        Some(def_id) => match def_id.try_into() {
            Ok(def_id) => AttrFlags::query(db, def_id)
                .contains(AttrFlags::RUSTC_HAS_INCOHERENT_INHERENT_IMPLS),
            Err(()) => true,
        },
        _ => true,
    };
    return if !has_incoherent_impls {
        &[]
    } else {
        incoherent_inherent_impls_query(db, (), self_ty)
    };

    #[salsa::tracked(returns(ref))]
    fn incoherent_inherent_impls_query(
        db: &dyn HirDatabase,
        _force_query_input_to_be_interned: (),
        self_ty: SimplifiedType,
    ) -> Box<[ImplId]> {
        let _p = tracing::info_span!("incoherent_inherent_impl_crates").entered();

        let mut result = Vec::new();
        for &krate in crates_containing_incoherent_inherent_impls(db) {
            let impls = InherentImpls::for_crate(db, krate);
            result.extend_from_slice(impls.for_self_ty(&self_ty));
        }
        result.into_boxed_slice()
    }
}

pub fn simplified_type_module(db: &dyn HirDatabase, ty: &SimplifiedType) -> Option<ModuleId> {
    match ty.def()? {
        SolverDefId::AdtId(id) => Some(id.module(db)),
        SolverDefId::TypeAliasId(id) => Some(id.module(db)),
        SolverDefId::TraitId(id) => Some(id.module(db)),
        _ => None,
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct InherentImpls {
    map: FxHashMap<SimplifiedType, Box<[ImplId]>>,
}

#[salsa::tracked]
impl<'db> InherentImpls {
    #[salsa::tracked(returns(ref))]
    pub fn for_crate(db: &'db dyn HirDatabase, krate: Crate) -> Self {
        let _p = tracing::info_span!("inherent_impls_in_crate_query", ?krate).entered();

        let crate_def_map = crate_def_map(db, krate);

        Self::collect_def_map(db, crate_def_map)
    }

    #[salsa::tracked(returns(ref))]
    pub fn for_block(db: &'db dyn HirDatabase, block: BlockIdLt<'db>) -> Option<Box<Self>> {
        let _p = tracing::info_span!("inherent_impls_in_block_query").entered();

        let block_def_map = block_def_map(db, block);
        let result = Self::collect_def_map(db, block_def_map);
        if result.map.is_empty() { None } else { Some(Box::new(result)) }
    }
}

impl InherentImpls {
    fn collect_def_map(db: &dyn HirDatabase, def_map: &DefMap) -> Self {
        let mut map = FxHashMap::default();
        collect(db, def_map, &mut map);
        let mut map = map
            .into_iter()
            .map(|(self_ty, impls)| (self_ty, impls.into_boxed_slice()))
            .collect::<FxHashMap<_, _>>();
        map.shrink_to_fit();
        return Self { map };

        fn collect(
            db: &dyn HirDatabase,
            def_map: &DefMap,
            map: &mut FxHashMap<SimplifiedType, Vec<ImplId>>,
        ) {
            for (_module_id, module_data) in def_map.modules() {
                for impl_id in module_data.scope.impls() {
                    let data = db.impl_signature(impl_id);
                    if data.target_trait.is_some() {
                        continue;
                    }

                    let interner = DbInterner::new_no_crate(db);
                    let self_ty = db.impl_self_ty(impl_id);
                    let self_ty = self_ty.instantiate_identity();
                    if let Some(self_ty) =
                        simplify_type(interner, self_ty, TreatParams::InstantiateWithInfer)
                    {
                        map.entry(self_ty).or_default().push(impl_id);
                    }
                }

                // To better support custom derives, collect impls in all unnamed const items.
                // const _: () = { ... };
                for konst in module_data.scope.unnamed_consts() {
                    let body = db.body(konst.into());
                    for (_, block_def_map) in body.blocks(db) {
                        collect(db, block_def_map, map);
                    }
                }
            }
        }
    }

    pub fn for_self_ty(&self, self_ty: &SimplifiedType) -> &[ImplId] {
        self.map.get(self_ty).map(|it| &**it).unwrap_or_default()
    }

    pub fn for_each_crate_and_block<'db>(
        db: &'db dyn HirDatabase,
        krate: Crate,
        block: Option<BlockIdLt<'db>>,
        for_each: &mut dyn FnMut(&InherentImpls),
    ) {
        let blocks = std::iter::successors(block, |block| block.module(db).block(db));
        blocks.filter_map(|block| Self::for_block(db, block).as_deref()).for_each(&mut *for_each);
        for_each(Self::for_crate(db, krate));
    }
}

#[derive(Debug, PartialEq)]
struct OneTraitImpls {
    non_blanket_impls: FxHashMap<SimplifiedType, Box<[ImplId]>>,
    blanket_impls: Box<[ImplId]>,
}

#[derive(Default)]
struct OneTraitImplsBuilder {
    non_blanket_impls: FxHashMap<SimplifiedType, Vec<ImplId>>,
    blanket_impls: Vec<ImplId>,
}

impl OneTraitImplsBuilder {
    fn finish(self) -> OneTraitImpls {
        let mut non_blanket_impls = self
            .non_blanket_impls
            .into_iter()
            .map(|(self_ty, impls)| (self_ty, impls.into_boxed_slice()))
            .collect::<FxHashMap<_, _>>();
        non_blanket_impls.shrink_to_fit();
        let blanket_impls = self.blanket_impls.into_boxed_slice();
        OneTraitImpls { non_blanket_impls, blanket_impls }
    }
}

#[derive(Debug, PartialEq)]
pub struct TraitImpls {
    map: FxHashMap<TraitId, OneTraitImpls>,
}

#[salsa::tracked]
impl<'db> TraitImpls {
    #[salsa::tracked(returns(ref))]
    pub fn for_crate(db: &'db dyn HirDatabase, krate: Crate) -> Arc<Self> {
        let _p = tracing::info_span!("inherent_impls_in_crate_query", ?krate).entered();

        let crate_def_map = crate_def_map(db, krate);
        let result = Self::collect_def_map(db, crate_def_map);
        Arc::new(result)
    }

    #[salsa::tracked(returns(ref))]
    pub fn for_block(db: &'db dyn HirDatabase, block: BlockIdLt<'db>) -> Option<Box<Self>> {
        let _p = tracing::info_span!("inherent_impls_in_block_query").entered();

        let block_def_map = block_def_map(db, block);
        let result = Self::collect_def_map(db, block_def_map);
        if result.map.is_empty() { None } else { Some(Box::new(result)) }
    }

    #[salsa::tracked(returns(ref))]
    pub fn for_crate_and_deps(db: &'db dyn HirDatabase, krate: Crate) -> Box<[Arc<Self>]> {
        krate.transitive_deps(db).iter().map(|&dep| Self::for_crate(db, dep).clone()).collect()
    }
}

impl TraitImpls {
    fn collect_def_map(db: &dyn HirDatabase, def_map: &DefMap) -> Self {
        let mut map = FxHashMap::default();
        collect(db, def_map, &mut map);
        let mut map = map
            .into_iter()
            .map(|(trait_id, trait_map)| (trait_id, trait_map.finish()))
            .collect::<FxHashMap<_, _>>();
        map.shrink_to_fit();
        return Self { map };

        fn collect(
            db: &dyn HirDatabase,
            def_map: &DefMap,
            map: &mut FxHashMap<TraitId, OneTraitImplsBuilder>,
        ) {
            for (_module_id, module_data) in def_map.modules() {
                for impl_id in module_data.scope.impls() {
                    // Reservation impls should be ignored during trait resolution, so we never need
                    // them during type analysis. See rust-lang/rust#64631 for details.
                    //
                    // FIXME: Reservation impls should be considered during coherence checks. If we are
                    // (ever) to implement coherence checks, this filtering should be done by the trait
                    // solver.
                    if AttrFlags::query(db, impl_id.into())
                        .contains(AttrFlags::RUSTC_RESERVATION_IMPL)
                    {
                        continue;
                    }
                    let trait_ref = match db.impl_trait(impl_id) {
                        Some(tr) => tr.instantiate_identity(),
                        None => continue,
                    };
                    let self_ty = trait_ref.self_ty();
                    let interner = DbInterner::new_no_crate(db);
                    let entry = map.entry(trait_ref.def_id.0).or_default();
                    match simplify_type(interner, self_ty, TreatParams::InstantiateWithInfer) {
                        Some(self_ty) => {
                            entry.non_blanket_impls.entry(self_ty).or_default().push(impl_id)
                        }
                        None => entry.blanket_impls.push(impl_id),
                    }
                }

                // To better support custom derives, collect impls in all unnamed const items.
                // const _: () = { ... };
                for konst in module_data.scope.unnamed_consts() {
                    let body = db.body(konst.into());
                    for (_, block_def_map) in body.blocks(db) {
                        collect(db, block_def_map, map);
                    }
                }
            }
        }
    }

    pub fn blanket_impls(&self, for_trait: TraitId) -> &[ImplId] {
        self.map.get(&for_trait).map(|it| &*it.blanket_impls).unwrap_or_default()
    }

    /// Queries whether `self_ty` has potentially applicable implementations of `trait_`.
    pub fn has_impls_for_trait_and_self_ty(
        &self,
        trait_: TraitId,
        self_ty: &SimplifiedType,
    ) -> bool {
        self.map.get(&trait_).is_some_and(|trait_impls| {
            trait_impls.non_blanket_impls.contains_key(self_ty)
                || !trait_impls.blanket_impls.is_empty()
        })
    }

    pub fn for_trait_and_self_ty(&self, trait_: TraitId, self_ty: &SimplifiedType) -> &[ImplId] {
        self.map
            .get(&trait_)
            .and_then(|map| map.non_blanket_impls.get(self_ty))
            .map(|it| &**it)
            .unwrap_or_default()
    }

    pub fn for_trait(&self, trait_: TraitId, mut callback: impl FnMut(&[ImplId])) {
        if let Some(impls) = self.map.get(&trait_) {
            callback(&impls.blanket_impls);
            for impls in impls.non_blanket_impls.values() {
                callback(impls);
            }
        }
    }

    pub fn for_self_ty(&self, self_ty: &SimplifiedType, mut callback: impl FnMut(&[ImplId])) {
        for for_trait in self.map.values() {
            if let Some(for_ty) = for_trait.non_blanket_impls.get(self_ty) {
                callback(for_ty);
            }
        }
    }

    pub fn for_each_crate_and_block<'db>(
        db: &'db dyn HirDatabase,
        krate: Crate,
        block: Option<BlockIdLt<'db>>,
        for_each: &mut dyn FnMut(&TraitImpls),
    ) {
        let blocks = std::iter::successors(block, |block| block.module(db).block(db));
        blocks.filter_map(|block| Self::for_block(db, block).as_deref()).for_each(&mut *for_each);
        Self::for_crate_and_deps(db, krate).iter().map(|it| &**it).for_each(for_each);
    }

    /// Like [`Self::for_each_crate_and_block()`], but takes in account two blocks, one for a trait and one for a self type.
    pub fn for_each_crate_and_block_trait_and_type<'db>(
        db: &'db dyn HirDatabase,
        krate: Crate,
        type_block: Option<BlockIdLt<'db>>,
        trait_block: Option<BlockIdLt<'db>>,
        for_each: &mut dyn FnMut(&TraitImpls),
    ) {
        let in_self_and_deps = TraitImpls::for_crate_and_deps(db, krate);
        in_self_and_deps.iter().for_each(|impls| for_each(impls));

        // We must not provide duplicate impls to the solver. Therefore we work with the following strategy:
        // start from each block, and walk ancestors until you meet the other block. If they never meet,
        // that means there can't be duplicate impls; if they meet, we stop the search of the deeper block.
        // This breaks when they are equal (both will stop immediately), therefore we handle this case
        // specifically.
        let blocks_iter = |block: Option<BlockIdLt<'db>>| {
            std::iter::successors(block, |block| block.module(db).block(db))
        };
        let for_each_block = |current_block: Option<BlockIdLt<'db>>,
                              other_block: Option<BlockIdLt<'db>>| {
            blocks_iter(current_block)
                .take_while(move |&block| {
                    other_block.is_none_or(|other_block| other_block != block)
                })
                .filter_map(move |block| TraitImpls::for_block(db, block).as_deref())
        };
        if trait_block == type_block {
            blocks_iter(trait_block)
                .filter_map(|block| TraitImpls::for_block(db, block).as_deref())
                .for_each(for_each);
        } else {
            for_each_block(trait_block, type_block).for_each(&mut *for_each);
            for_each_block(type_block, trait_block).for_each(for_each);
        }
    }
}
