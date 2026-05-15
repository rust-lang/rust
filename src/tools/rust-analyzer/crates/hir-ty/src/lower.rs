//! Methods for lowering the HIR to types. There are two main cases here:
//!
//!  - Lowering a type reference like `&usize` or `Option<foo::bar::Baz>` to a
//!    type: The entry point for this is `TyLoweringContext::lower_ty`.
//!  - Building the type for an item: This happens through the `ty` query.
//!
//! This usually involves resolving names, collecting generic arguments etc.
pub(crate) mod diagnostics;
pub(crate) mod path;

use std::{cell::OnceCell, iter, mem, sync::OnceLock};

use either::Either;
use hir_def::{
    AdtId, AssocItemId, CallableDefId, ConstId, ConstParamId, EnumId, EnumVariantId,
    ExpressionStoreOwnerId, FunctionId, GenericDefId, GenericParamId, HasModule, ImplId,
    ItemContainerId, LifetimeParamId, LocalFieldId, Lookup, StaticId, StructId, TraitId,
    TypeAliasId, TypeOrConstParamId, TypeParamId, UnionId, VariantId,
    builtin_type::BuiltinType,
    expr_store::{ExpressionStore, path::Path},
    hir::{
        ExprId, PatId,
        generics::{
            GenericParamDataRef, GenericParams, LocalTypeOrConstParamId, TypeOrConstParamData,
            TypeParamProvenance, WherePredicate,
        },
    },
    item_tree::FieldsShape,
    lang_item::LangItems,
    resolver::{HasResolver, LifetimeNs, Resolver, TypeNs},
    signatures::{
        ConstSignature, FunctionSignature, ImplSignature, StaticSignature, StructSignature,
        TraitFlags, TraitSignature, TypeAliasFlags, TypeAliasSignature,
    },
    type_ref::{
        ConstRef, FnType, LifetimeRefId, PathId, TraitBoundModifier, TraitRef as HirTraitRef,
        TypeBound, TypeRef, TypeRefId,
    },
};
use hir_expand::name::Name;
use la_arena::{Arena, ArenaMap, Idx};
use path::{PathDiagnosticCallback, PathLoweringContext};
use rustc_abi::ExternAbi;
use rustc_ast_ir::Mutability;
use rustc_hash::FxHashSet;
use rustc_type_ir::{
    AliasTyKind, BoundVarIndexKind, DebruijnIndex, ExistentialPredicate, ExistentialProjection,
    ExistentialTraitRef, FnSig, Interner, OutlivesPredicate, TermKind, TyKind, TypeFoldable,
    TypeVisitableExt, Upcast, UpcastFrom, elaborate,
    inherent::{Clause as _, GenericArgs as _, IntoKind as _, Region as _, Ty as _},
};
use smallvec::SmallVec;
use stdx::{impl_from, never};
use thin_vec::ThinVec;
use tracing::debug;

use crate::{
    ImplTraitId, Span, TyLoweringDiagnostic, TyLoweringDiagnosticKind,
    consteval::{create_anon_const, path_to_const},
    db::{AnonConstId, GeneralConstId, HirDatabase, InternedOpaqueTyId},
    generics::{Generics, SingleGenerics, generics},
    infer::unify::InferenceTable,
    next_solver::{
        AliasTy, Binder, BoundExistentialPredicates, Clause, ClauseKind, Clauses, Const, ConstKind,
        DbInterner, DefaultAny, EarlyBinder, EarlyParamRegion, ErrorGuaranteed, FnSigKind,
        FxIndexMap, GenericArg, GenericArgs, ParamConst, ParamEnv, PatList, Pattern, PolyFnSig,
        Predicate, Region, StoredClauses, StoredEarlyBinder, StoredGenericArg, StoredGenericArgs,
        StoredPolyFnSig, StoredTraitRef, StoredTy, TraitPredicate, TraitRef, Ty, Tys, Unnormalized,
        abi::Safety, util::BottomUpFolder,
    },
};

pub(crate) struct PathDiagnosticCallbackData(pub(crate) TypeRefId);

#[derive(PartialEq, Eq, Debug, Hash)]
pub struct ImplTraits {
    pub(crate) impl_traits: Arena<ImplTrait>,
}

#[derive(PartialEq, Eq, Debug, Hash)]
pub struct ImplTrait {
    pub(crate) predicates: StoredClauses,
    pub(crate) assoc_ty_bounds_start: u32,
}

pub type ImplTraitIdx = Idx<ImplTrait>;

#[derive(Debug, Default)]
struct ImplTraitLoweringState {
    /// When turning `impl Trait` into opaque types, we have to collect the
    /// bounds at the same time to get the IDs correct (without becoming too
    /// complicated).
    mode: ImplTraitLoweringMode,
    // This is structured as a struct with fields and not as an enum because it helps with the borrow checker.
    opaque_type_data: Arena<ImplTrait>,
}

impl ImplTraitLoweringState {
    fn new(mode: ImplTraitLoweringMode) -> ImplTraitLoweringState {
        Self { mode, opaque_type_data: Arena::new() }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum LifetimeElisionKind<'db> {
    /// Create a new anonymous lifetime parameter and reference it.
    ///
    /// If `report_in_path`, report an error when encountering lifetime elision in a path:
    /// ```compile_fail
    /// struct Foo<'a> { x: &'a () }
    /// async fn foo(x: Foo) {}
    /// ```
    ///
    /// Note: the error should not trigger when the elided lifetime is in a pattern or
    /// expression-position path:
    /// ```
    /// struct Foo<'a> { x: &'a () }
    /// async fn foo(Foo { x: _ }: Foo<'_>) {}
    /// ```
    AnonymousCreateParameter { report_in_path: bool },

    /// Replace all anonymous lifetimes by provided lifetime.
    Elided(Region<'db>),

    /// Give a hard error when either `&` or `'_` is written. Used to
    /// rule out things like `where T: Foo<'_>`. Does not imply an
    /// error on default object bounds (e.g., `Box<dyn Foo>`).
    AnonymousReportError,

    /// Resolves elided lifetimes to `'static` if there are no other lifetimes in scope,
    /// otherwise give a warning that the previous behavior of introducing a new early-bound
    /// lifetime is a bug and will be removed (if `only_lint` is enabled).
    StaticIfNoLifetimeInScope { only_lint: bool },

    /// Signal we cannot find which should be the anonymous lifetime.
    ElisionFailure,

    /// Infer all elided lifetimes.
    Infer,
}

impl<'db> LifetimeElisionKind<'db> {
    #[inline]
    pub(crate) fn for_const(
        interner: DbInterner<'db>,
        const_parent: ItemContainerId,
    ) -> LifetimeElisionKind<'db> {
        match const_parent {
            ItemContainerId::ExternBlockId(_) | ItemContainerId::ModuleId(_) => {
                LifetimeElisionKind::Elided(Region::new_static(interner))
            }
            ItemContainerId::ImplId(_) => {
                LifetimeElisionKind::StaticIfNoLifetimeInScope { only_lint: true }
            }
            ItemContainerId::TraitId(_) => {
                LifetimeElisionKind::StaticIfNoLifetimeInScope { only_lint: false }
            }
        }
    }

    #[inline]
    pub(crate) fn for_fn_params(data: &FunctionSignature) -> LifetimeElisionKind<'db> {
        LifetimeElisionKind::AnonymousCreateParameter { report_in_path: data.is_async() }
    }

    #[inline]
    pub(crate) fn for_fn_ret(interner: DbInterner<'db>) -> LifetimeElisionKind<'db> {
        // FIXME: We should use the elided lifetime here, or `ElisionFailure`.
        LifetimeElisionKind::Elided(Region::error(interner))
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum GenericPredicateSource {
    SelfOnly,
    AssocTyBound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ForbidParamsAfterReason {
    /// When lowering generic param defaults, you cannot refer to any param after
    /// the currently lowered param, including the current param.
    LoweringParamDefault,
    /// Most anon const (except array repeat expressions) cannot refer to any generic
    /// param.
    AnonConst,
    /// The type of a const param cannot refer to a type param.
    ConstParamTy,
}

pub trait TyLoweringInferVarsCtx<'db> {
    fn next_ty_var(&mut self, span: Span) -> Ty<'db>;
    fn next_const_var(&mut self, span: Span) -> Const<'db>;
    fn next_region_var(&mut self, span: Span) -> Region<'db>;

    #[expect(private_interfaces)]
    fn as_table(&mut self) -> Option<&mut InferenceTable<'db>> {
        None
    }
}

pub struct TyLoweringContext<'db, 'a> {
    pub db: &'db dyn HirDatabase,
    pub(crate) interner: DbInterner<'db>,
    types: &'db crate::next_solver::DefaultAny<'db>,
    lang_items: &'db LangItems,
    resolver: &'a Resolver<'db>,
    store: &'a ExpressionStore,
    def: ExpressionStoreOwnerId,
    generic_def: GenericDefId,
    generics: &'a OnceCell<Generics<'db>>,
    in_binders: DebruijnIndex,
    impl_trait_mode: ImplTraitLoweringState,
    /// Tracks types with explicit `?Sized` bounds.
    pub(crate) unsized_types: FxHashSet<Ty<'db>>,
    pub(crate) diagnostics: ThinVec<TyLoweringDiagnostic>,
    lifetime_elision: LifetimeElisionKind<'db>,
    forbid_params_after: Option<u32>,
    forbid_params_after_reason: ForbidParamsAfterReason,
    pub(crate) defined_anon_consts: ThinVec<AnonConstId>,
    infer_vars: Option<&'a mut dyn TyLoweringInferVarsCtx<'db>>,
}

impl<'db, 'a> TyLoweringContext<'db, 'a> {
    pub fn new(
        db: &'db dyn HirDatabase,
        resolver: &'a Resolver<'db>,
        store: &'a ExpressionStore,
        def: ExpressionStoreOwnerId,
        generic_def: GenericDefId,
        generics: &'a OnceCell<Generics<'db>>,
        lifetime_elision: LifetimeElisionKind<'db>,
    ) -> Self {
        let impl_trait_mode = ImplTraitLoweringState::new(ImplTraitLoweringMode::Disallowed);
        let in_binders = DebruijnIndex::ZERO;
        let interner = DbInterner::new_with(db, resolver.krate());
        Self {
            db,
            // Can provide no block since we don't use it for trait solving.
            interner,
            types: crate::next_solver::default_types(db),
            lang_items: interner.lang_items(),
            resolver,
            def,
            generic_def,
            generics,
            store,
            in_binders,
            impl_trait_mode,
            unsized_types: FxHashSet::default(),
            diagnostics: ThinVec::new(),
            lifetime_elision,
            forbid_params_after: None,
            forbid_params_after_reason: ForbidParamsAfterReason::AnonConst,
            defined_anon_consts: ThinVec::new(),
            infer_vars: None,
        }
    }

    pub(crate) fn set_lifetime_elision(&mut self, lifetime_elision: LifetimeElisionKind<'db>) {
        self.lifetime_elision = lifetime_elision;
    }

    pub(crate) fn with_debruijn<T>(
        &mut self,
        debruijn: DebruijnIndex,
        f: impl FnOnce(&mut TyLoweringContext<'db, '_>) -> T,
    ) -> T {
        let old_debruijn = mem::replace(&mut self.in_binders, debruijn);
        let result = f(self);
        self.in_binders = old_debruijn;
        result
    }

    pub(crate) fn with_shifted_in<T>(
        &mut self,
        debruijn: DebruijnIndex,
        f: impl FnOnce(&mut TyLoweringContext<'db, '_>) -> T,
    ) -> T {
        self.with_debruijn(self.in_binders.shifted_in(debruijn.as_u32()), f)
    }

    pub(crate) fn with_impl_trait_mode(self, impl_trait_mode: ImplTraitLoweringMode) -> Self {
        Self { impl_trait_mode: ImplTraitLoweringState::new(impl_trait_mode), ..self }
    }

    pub(crate) fn impl_trait_mode(&mut self, impl_trait_mode: ImplTraitLoweringMode) -> &mut Self {
        self.impl_trait_mode = ImplTraitLoweringState::new(impl_trait_mode);
        self
    }

    pub(crate) fn forbid_params_after(&mut self, index: u32, reason: ForbidParamsAfterReason) {
        self.forbid_params_after = Some(index);
        self.forbid_params_after_reason = reason;
    }

    pub fn with_infer_vars_behavior(
        mut self,
        behavior: Option<&'a mut dyn TyLoweringInferVarsCtx<'db>>,
    ) -> Self {
        self.infer_vars = behavior;
        self
    }

    pub(crate) fn push_diagnostic(&mut self, type_ref: TypeRefId, kind: TyLoweringDiagnosticKind) {
        self.diagnostics.push(TyLoweringDiagnostic { source: type_ref, kind });
    }

    #[track_caller]
    pub(crate) fn expect_table(&mut self) -> &mut InferenceTable<'db> {
        self.infer_vars.as_mut().unwrap().as_table().unwrap()
    }

    fn next_ty_var(&mut self, span: Span) -> Ty<'db> {
        match &mut self.infer_vars {
            Some(infer_vars) => infer_vars.next_ty_var(span),
            None => {
                // FIXME: Emit an error: no infer vars allowed here.
                self.types.types.error
            }
        }
    }

    fn next_const_var(&mut self, span: Span) -> Const<'db> {
        match &mut self.infer_vars {
            Some(infer_vars) => infer_vars.next_const_var(span),
            None => {
                // FIXME: Emit an error: no infer vars allowed here.
                self.types.consts.error
            }
        }
    }

    fn next_region_var(&mut self, span: Span) -> Region<'db> {
        match &mut self.infer_vars {
            Some(infer_vars) => infer_vars.next_region_var(span),
            None => {
                // FIXME: Emit an error: no infer vars allowed here.
                self.types.regions.error
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub(crate) enum ImplTraitLoweringMode {
    /// `impl Trait` gets lowered into an opaque type that doesn't unify with
    /// anything except itself. This is used in places where values flow 'out',
    /// i.e. for arguments of the function we're currently checking, and return
    /// types of functions we're calling.
    Opaque,
    /// `impl Trait` is disallowed and will be an error.
    #[default]
    Disallowed,
}

impl<'db, 'a> TyLoweringContext<'db, 'a> {
    pub fn lower_ty(&mut self, type_ref: TypeRefId) -> Ty<'db> {
        self.lower_ty_ext(type_ref).0
    }

    pub(crate) fn lower_const(&mut self, const_ref: ConstRef, const_type: Ty<'db>) -> Const<'db> {
        self.lower_expr_as_const(const_ref.expr, const_type)
    }

    pub(crate) fn lower_expr_as_const(
        &mut self,
        expr_id: ExprId,
        const_type: Ty<'db>,
    ) -> Const<'db> {
        #[expect(clippy::manual_map, reason = "a `map()` here generates a borrowck error")]
        let create_var = match &mut self.infer_vars {
            Some(infer_vars) => Some(
                (&mut |span| infer_vars.next_const_var(span)) as &mut dyn FnMut(Span) -> Const<'db>,
            ),
            None => None,
        };
        let konst = create_anon_const(
            self.interner,
            self.def,
            self.store,
            expr_id,
            self.resolver,
            const_type,
            &|| self.generics.get_or_init(|| generics(self.db, self.generic_def)),
            create_var,
            self.forbid_params_after,
        );

        if let Ok(konst) = konst
            && let ConstKind::Unevaluated(konst) = konst.kind()
            && let GeneralConstId::AnonConstId(konst) = konst.def.0
        {
            self.defined_anon_consts.push(konst);
        }

        konst.unwrap_or({
            // FIXME: Report an error.
            self.types.consts.error
        })
    }

    pub(crate) fn lower_path_as_const(&mut self, path: &Path, _const_type: Ty<'db>) -> Const<'db> {
        path_to_const(self.db, self.resolver, &|| self.generics(), self.forbid_params_after, path)
            .unwrap_or({
                // FIXME: Report an error.
                self.types.consts.error
            })
    }

    fn generics(&self) -> &Generics<'db> {
        self.generics.get_or_init(|| generics(self.db, self.generic_def))
    }

    fn param_index_is_disallowed(&self, index: u32) -> bool {
        self.forbid_params_after.is_some_and(|disallow_params_after| index >= disallow_params_after)
    }

    fn type_param(&mut self, id: TypeParamId, index: u32) -> Ty<'db> {
        if self.param_index_is_disallowed(index) {
            // FIXME: Report an error.
            self.types.types.error
        } else {
            Ty::new_param(self.interner, id, index)
        }
    }

    fn region_param(&mut self, id: LifetimeParamId, index: u32) -> Region<'db> {
        if self.param_index_is_disallowed(index) {
            // FIXME: Report an error.
            self.types.regions.error
        } else {
            Region::new_early_param(self.interner, EarlyParamRegion { id, index })
        }
    }

    #[tracing::instrument(skip(self), ret)]
    pub fn lower_ty_ext(&mut self, type_ref_id: TypeRefId) -> (Ty<'db>, Option<TypeNs>) {
        let interner = self.interner;
        let mut res = None;
        let type_ref = &self.store[type_ref_id];
        tracing::debug!(?type_ref);
        let ty = match type_ref {
            TypeRef::Never => self.types.types.never,
            TypeRef::Tuple(inner) => {
                let inner_tys = inner.iter().map(|&tr| self.lower_ty(tr));
                Ty::new_tup_from_iter(interner, inner_tys)
            }
            TypeRef::Path(path) => {
                let (ty, res_) =
                    self.lower_path(path, PathId::from_type_ref_unchecked(type_ref_id));
                res = res_;
                ty
            }
            &TypeRef::TypeParam(type_param_id) => {
                res = Some(TypeNs::GenericParam(type_param_id));

                let generics = self.generics();
                let idx = generics.type_or_const_param_idx(type_param_id.into());
                self.type_param(type_param_id, idx)
            }
            &TypeRef::RawPtr(inner, mutability) => {
                let inner_ty = self.lower_ty(inner);
                Ty::new(interner, TyKind::RawPtr(inner_ty, lower_mutability(mutability)))
            }
            TypeRef::Array(array) => {
                let inner_ty = self.lower_ty(array.ty);
                let const_len = self.lower_const(array.len, self.types.types.usize);
                Ty::new_array_with_const_len(interner, inner_ty, const_len)
            }
            &TypeRef::Slice(inner) => {
                let inner_ty = self.lower_ty(inner);
                Ty::new_slice(interner, inner_ty)
            }
            TypeRef::Reference(ref_) => {
                let inner_ty = self.lower_ty(ref_.ty);
                // FIXME: It should infer the eldided lifetimes instead of stubbing with error
                let lifetime =
                    ref_.lifetime.map_or(self.types.regions.error, |lr| self.lower_lifetime(lr));
                Ty::new_ref(interner, lifetime, inner_ty, lower_mutability(ref_.mutability))
            }
            TypeRef::Placeholder => self.next_ty_var(type_ref_id.into()),
            TypeRef::Fn(fn_) => self.lower_fn_ptr(fn_),
            TypeRef::DynTrait(bounds) => self.lower_dyn_trait(bounds),
            TypeRef::ImplTrait(bounds) => {
                match self.impl_trait_mode.mode {
                    ImplTraitLoweringMode::Opaque => {
                        let origin = match self.resolver.generic_def() {
                            Some(GenericDefId::FunctionId(it)) => Either::Left(it),
                            Some(GenericDefId::TypeAliasId(it)) => Either::Right(it),
                            _ => panic!(
                                "opaque impl trait lowering must be in function or type alias"
                            ),
                        };

                        // this dance is to make sure the data is in the right
                        // place even if we encounter more opaque types while
                        // lowering the bounds
                        let idx = self.impl_trait_mode.opaque_type_data.alloc(ImplTrait {
                            predicates: Clauses::empty(interner).store(),
                            assoc_ty_bounds_start: 0,
                        });

                        let impl_trait_id = origin.either(
                            |f| ImplTraitId::ReturnTypeImplTrait(f, idx),
                            |a| ImplTraitId::TypeAliasImplTrait(a, idx),
                        );
                        let opaque_ty_id = InternedOpaqueTyId::new(self.db, impl_trait_id);

                        // We don't want to lower the bounds inside the binders
                        // we're currently in, because they don't end up inside
                        // those binders. E.g. when we have `impl Trait<impl
                        // OtherTrait<T>>`, the `impl OtherTrait<T>` can't refer
                        // to the self parameter from `impl Trait`, and the
                        // bounds aren't actually stored nested within each
                        // other, but separately. So if the `T` refers to a type
                        // parameter of the outer function, it's just one binder
                        // away instead of two.
                        let actual_opaque_type_data = self
                            .with_debruijn(DebruijnIndex::ZERO, |ctx| {
                                ctx.lower_impl_trait(opaque_ty_id, bounds)
                            });
                        self.impl_trait_mode.opaque_type_data[idx] = actual_opaque_type_data;

                        let args =
                            GenericArgs::identity_for_item(self.interner, opaque_ty_id.into());
                        Ty::new_alias(
                            self.interner,
                            AliasTy::new_from_args(
                                self.interner,
                                AliasTyKind::Opaque { def_id: opaque_ty_id.into() },
                                args,
                            ),
                        )
                    }
                    ImplTraitLoweringMode::Disallowed => {
                        // FIXME: report error
                        self.types.types.error
                    }
                }
            }
            &TypeRef::PatternType(ty, pat) => {
                let ty = self.lower_ty(ty);
                let Some(pat) = self.lower_pattern_type(pat, ty) else {
                    // FIXME: Report an error.
                    return (self.types.types.error, res);
                };
                Ty::new_pat(self.interner, ty, pat)
            }
            TypeRef::Error => self.types.types.error,
        };
        (ty, res)
    }

    fn lower_pattern_type(&mut self, pat: PatId, ty: Ty<'db>) -> Option<Pattern<'db>> {
        let pat_kind = match self.store[pat] {
            hir_def::hir::Pat::Range { start: Some(start), end: Some(end), range_type: _ } => {
                rustc_type_ir::PatternKind::Range {
                    start: self.lower_expr_as_const(start, ty),
                    end: self.lower_expr_as_const(end, ty),
                }
            }
            hir_def::hir::Pat::NotNull => rustc_type_ir::PatternKind::NotNull,
            hir_def::hir::Pat::Or(ref pats) => rustc_type_ir::PatternKind::Or(
                PatList::new_from_iter(
                    self.interner,
                    pats.iter().map(|&pat| self.lower_pattern_type(pat, ty).ok_or(())),
                )
                .ok()?,
            ),
            _ => return None,
        };
        Some(Pattern::new(self.interner, pat_kind))
    }

    fn lower_fn_ptr(&mut self, fn_: &FnType) -> Ty<'db> {
        let interner = self.interner;
        let (params, ret_ty) = fn_.split_params_and_ret();
        let old_lifetime_elision = self.lifetime_elision;
        let mut args = Vec::with_capacity(fn_.params.len());
        self.with_shifted_in(DebruijnIndex::from_u32(1), |ctx: &mut TyLoweringContext<'_, '_>| {
            ctx.lifetime_elision =
                LifetimeElisionKind::AnonymousCreateParameter { report_in_path: false };
            args.extend(params.iter().map(|&(_, tr)| ctx.lower_ty(tr)));
            ctx.lifetime_elision = LifetimeElisionKind::for_fn_ret(interner);
            args.push(ctx.lower_ty(ret_ty));
        });
        self.lifetime_elision = old_lifetime_elision;
        Ty::new_fn_ptr(
            interner,
            Binder::dummy(FnSig {
                fn_sig_kind: FnSigKind::new(
                    fn_.abi,
                    if fn_.is_unsafe { Safety::Unsafe } else { Safety::Safe },
                    fn_.is_varargs,
                ),
                inputs_and_output: Tys::new_from_slice(&args),
            }),
        )
    }

    /// This is only for `generic_predicates_for_param`, where we can't just
    /// lower the self types of the predicates since that could lead to cycles.
    /// So we just check here if the `type_ref` resolves to a generic param, and which.
    fn lower_ty_only_param(&self, type_ref: TypeRefId) -> Option<TypeOrConstParamId> {
        let type_ref = &self.store[type_ref];
        let path = match type_ref {
            TypeRef::Path(path) => path,
            &TypeRef::TypeParam(idx) => return Some(idx.into()),
            _ => return None,
        };
        if path.type_anchor().is_some() {
            return None;
        }
        if path.segments().len() > 1 {
            return None;
        }
        let resolution = match self.resolver.resolve_path_in_type_ns(self.db, path) {
            Some((it, None, _)) => it,
            _ => return None,
        };
        match resolution {
            TypeNs::GenericParam(param_id) => Some(param_id.into()),
            _ => None,
        }
    }

    #[inline]
    fn on_path_diagnostic_callback<'b>(type_ref: TypeRefId) -> PathDiagnosticCallback<'b, 'db> {
        PathDiagnosticCallback {
            data: Either::Left(PathDiagnosticCallbackData(type_ref)),
            callback: |data, this, diag| {
                let type_ref = data.as_ref().left().unwrap().0;
                this.push_diagnostic(type_ref, TyLoweringDiagnosticKind::PathDiagnostic(diag))
            },
        }
    }

    #[inline]
    fn at_path(&mut self, path_id: PathId) -> PathLoweringContext<'_, 'a, 'db> {
        PathLoweringContext::new(
            self,
            Self::on_path_diagnostic_callback(path_id.type_ref()),
            &self.store[path_id],
        )
    }

    pub(crate) fn lower_path(&mut self, path: &Path, path_id: PathId) -> (Ty<'db>, Option<TypeNs>) {
        // Resolve the path (in type namespace)
        if let Some(type_ref) = path.type_anchor() {
            let (ty, res) = self.lower_ty_ext(type_ref);
            let mut ctx = self.at_path(path_id);
            return ctx.lower_ty_relative_path(ty, res, false, path_id.type_ref().into());
        }

        let mut ctx = self.at_path(path_id);
        let (resolution, remaining_index) = match ctx.resolve_path_in_type_ns() {
            Some(it) => it,
            None => return (self.types.types.error, None),
        };

        if matches!(resolution, TypeNs::TraitId(_)) && remaining_index.is_none() {
            // trait object type without dyn
            let bound = TypeBound::Path(path_id, TraitBoundModifier::None);
            let ty = self.lower_dyn_trait(&[bound]);
            return (ty, None);
        }

        ctx.lower_partly_resolved_path(resolution, false, path_id.type_ref().into())
    }

    fn lower_trait_ref_from_path(
        &mut self,
        path_id: PathId,
        explicit_self_ty: Ty<'db>,
    ) -> Option<(TraitRef<'db>, PathLoweringContext<'_, 'a, 'db>)> {
        let mut ctx = self.at_path(path_id);
        let resolved = match ctx.resolve_path_in_type_ns_fully()? {
            // FIXME(trait_alias): We need to handle trait alias here.
            TypeNs::TraitId(tr) => tr,
            _ => return None,
        };
        Some((
            ctx.lower_trait_ref_from_resolved_path(
                resolved,
                explicit_self_ty,
                false,
                path_id.type_ref().into(),
            ),
            ctx,
        ))
    }

    fn lower_trait_ref(
        &mut self,
        trait_ref: &HirTraitRef,
        explicit_self_ty: Ty<'db>,
    ) -> Option<TraitRef<'db>> {
        self.lower_trait_ref_from_path(trait_ref.path, explicit_self_ty).map(|it| it.0)
    }

    pub(crate) fn lower_where_predicate<'b>(
        &'b mut self,
        where_predicate: &'b WherePredicate,
        ignore_bindings: bool,
    ) -> impl Iterator<Item = (Clause<'db>, GenericPredicateSource)> + use<'a, 'b, 'db> {
        match where_predicate {
            WherePredicate::ForLifetime { target, bound, .. }
            | WherePredicate::TypeBound { target, bound } => {
                let self_ty = self.lower_ty(*target);
                Either::Left(self.lower_type_bound(bound, self_ty, ignore_bindings))
            }
            &WherePredicate::Lifetime { bound, target } => Either::Right(iter::once((
                Clause(Predicate::new(
                    self.interner,
                    Binder::dummy(rustc_type_ir::PredicateKind::Clause(
                        rustc_type_ir::ClauseKind::RegionOutlives(OutlivesPredicate(
                            self.lower_lifetime(bound),
                            self.lower_lifetime(target),
                        )),
                    )),
                )),
                GenericPredicateSource::SelfOnly,
            ))),
        }
        .into_iter()
    }

    pub(crate) fn lower_type_bound<'b>(
        &'b mut self,
        bound: &'b TypeBound,
        self_ty: Ty<'db>,
        ignore_bindings: bool,
    ) -> impl Iterator<Item = (Clause<'db>, GenericPredicateSource)> + use<'b, 'a, 'db> {
        let interner = self.interner;
        let meta_sized = self.lang_items.MetaSized;
        let pointee_sized = self.lang_items.PointeeSized;
        let mut assoc_bounds = None;
        let mut clause = None;
        match bound {
            &TypeBound::Path(path, TraitBoundModifier::None) | &TypeBound::ForLifetime(_, path) => {
                // FIXME Don't silently drop the hrtb lifetimes here
                if let Some((trait_ref, mut ctx)) = self.lower_trait_ref_from_path(path, self_ty) {
                    // FIXME(sized-hierarchy): Remove this bound modifications once we have implemented
                    // sized-hierarchy correctly.
                    if meta_sized.is_some_and(|it| it == trait_ref.def_id.0) {
                        // Ignore this bound
                    } else if pointee_sized.is_some_and(|it| it == trait_ref.def_id.0) {
                        // Regard this as `?Sized` bound
                        ctx.ty_ctx().unsized_types.insert(self_ty);
                    } else {
                        if !ignore_bindings {
                            assoc_bounds = ctx.assoc_type_bindings_from_type_bound(
                                trait_ref,
                                path.type_ref().into(),
                            );
                        }
                        clause = Some(Clause(Predicate::new(
                            interner,
                            Binder::dummy(rustc_type_ir::PredicateKind::Clause(
                                rustc_type_ir::ClauseKind::Trait(TraitPredicate {
                                    trait_ref,
                                    polarity: rustc_type_ir::PredicatePolarity::Positive,
                                }),
                            )),
                        )));
                    }
                }
            }
            &TypeBound::Path(path, TraitBoundModifier::Maybe) => {
                let sized_trait = self.lang_items.Sized;
                // Don't lower associated type bindings as the only possible relaxed trait bound
                // `?Sized` has no of them.
                // If we got another trait here ignore the bound completely.
                let trait_id = self
                    .lower_trait_ref_from_path(path, self_ty)
                    .map(|(trait_ref, _)| trait_ref.def_id.0);
                if trait_id == sized_trait {
                    self.unsized_types.insert(self_ty);
                }
            }
            &TypeBound::Lifetime(l) => {
                let lifetime = self.lower_lifetime(l);
                clause = Some(Clause(Predicate::new(
                    self.interner,
                    Binder::dummy(rustc_type_ir::PredicateKind::Clause(
                        rustc_type_ir::ClauseKind::TypeOutlives(OutlivesPredicate(
                            self_ty, lifetime,
                        )),
                    )),
                )));
            }
            TypeBound::Use(_) | TypeBound::Error => {}
        }
        clause
            .into_iter()
            .map(|pred| (pred, GenericPredicateSource::SelfOnly))
            .chain(assoc_bounds.into_iter().flatten())
    }

    fn lower_dyn_trait(&mut self, bounds: &[TypeBound]) -> Ty<'db> {
        let interner = self.interner;
        let dummy_self_ty = self.types.types.dyn_trait_dummy_self;
        let mut region = None;
        // INVARIANT: The principal trait bound, if present, must come first. Others may be in any
        // order but should be in the same order for the same set but possibly different order of
        // bounds in the input.
        // INVARIANT: If this function returns `DynTy`, there should be at least one trait bound.
        // These invariants are utilized by `TyExt::dyn_trait()` and chalk.
        let bounds = self.with_shifted_in(DebruijnIndex::from_u32(1), |ctx| {
            let mut principal = None;
            let mut auto_traits = SmallVec::<[_; 3]>::new();
            let mut projections = Vec::new();
            let mut had_error = false;

            for b in bounds {
                let db = ctx.db;
                ctx.lower_type_bound(b, dummy_self_ty, false).for_each(|(b, _)| {
                    match b.kind().skip_binder() {
                        rustc_type_ir::ClauseKind::Trait(t) => {
                            let id = t.def_id();
                            let is_auto =
                                TraitSignature::of(db, id.0).flags.contains(TraitFlags::AUTO);
                            if is_auto {
                                auto_traits.push(t.def_id().0);
                            } else {
                                if principal.is_some() {
                                    // FIXME: Report an error.
                                    had_error = true;
                                }
                                principal = Some(b.kind().rebind(t.trait_ref));
                            }
                        }
                        rustc_type_ir::ClauseKind::Projection(p) => {
                            projections.push(b.kind().rebind(p));
                        }
                        rustc_type_ir::ClauseKind::TypeOutlives(outlives_predicate) => {
                            if region.is_some() {
                                // FIXME: Report an error.
                                had_error = true;
                            }
                            region = Some(outlives_predicate.1);
                        }
                        rustc_type_ir::ClauseKind::RegionOutlives(_)
                        | rustc_type_ir::ClauseKind::ConstArgHasType(_, _)
                        | rustc_type_ir::ClauseKind::WellFormed(_)
                        | rustc_type_ir::ClauseKind::ConstEvaluatable(_)
                        | rustc_type_ir::ClauseKind::HostEffect(_)
                        | rustc_type_ir::ClauseKind::UnstableFeature(_) => unreachable!(),
                    }
                })
            }

            if had_error {
                return None;
            }

            if principal.is_none() && auto_traits.is_empty() {
                // No traits is not allowed.
                return None;
            }

            // `Send + Sync` is the same as `Sync + Send`.
            auto_traits.sort_unstable();
            // Duplicate auto traits are permitted.
            auto_traits.dedup();

            // Map the projection bounds onto a key that makes it easy to remove redundant
            // bounds that are constrained by supertraits of the principal def id.
            //
            // Also make sure we detect conflicting bounds from expanding a trait alias and
            // also specifying it manually, like:
            // ```
            // type Alias = Trait<Assoc = i32>;
            // let _: &dyn Alias<Assoc = u32> = /* ... */;
            // ```
            let mut projection_bounds = FxIndexMap::default();
            for proj in projections {
                let key = (
                    proj.skip_binder().def_id().0,
                    interner.anonymize_bound_vars(
                        proj.map_bound(|proj| proj.projection_term.trait_ref(interner)),
                    ),
                );
                if let Some(old_proj) = projection_bounds.insert(key, proj)
                    && interner.anonymize_bound_vars(proj)
                        != interner.anonymize_bound_vars(old_proj)
                {
                    // FIXME: Report "conflicting associated type" error.
                }
            }

            // A stable ordering of associated types from the principal trait and all its
            // supertraits. We use this to ensure that different substitutions of a trait
            // don't result in `dyn Trait` types with different projections lists, which
            // can be unsound: <https://github.com/rust-lang/rust/pull/136458>.
            // We achieve a stable ordering by walking over the unsubstituted principal
            // trait ref.
            let mut ordered_associated_types = vec![];

            if let Some(principal_trait) = principal {
                // Generally we should not elaborate in lowering as this can lead to cycles, but
                // here rustc cycles as well.
                for clause in elaborate::elaborate(
                    interner,
                    [Clause::upcast_from(
                        TraitRef::identity(interner, principal_trait.def_id()),
                        interner,
                    )],
                )
                .filter_only_self()
                {
                    let clause = clause.instantiate_supertrait(interner, principal_trait);
                    debug!("observing object predicate `{clause:?}`");

                    let bound_predicate = clause.kind();
                    match bound_predicate.skip_binder() {
                        ClauseKind::Trait(pred) => {
                            // FIXME(negative_bounds): Handle this correctly...
                            let trait_ref = interner
                                .anonymize_bound_vars(bound_predicate.rebind(pred.trait_ref));
                            ordered_associated_types.extend(
                                pred.trait_ref
                                    .def_id
                                    .0
                                    .trait_items(self.db)
                                    .associated_types()
                                    .map(|item| (item.into(), trait_ref)),
                            );
                        }
                        ClauseKind::Projection(pred) => {
                            let pred = bound_predicate.rebind(pred);
                            // A `Self` within the original bound will be instantiated with a
                            // `trait_object_dummy_self`, so check for that.
                            let references_self = match pred.skip_binder().term.kind() {
                                TermKind::Ty(ty) => {
                                    ty.walk().any(|arg| arg == dummy_self_ty.into())
                                }
                                // FIXME(mgca): We should walk the const instead of not doing anything
                                TermKind::Const(_) => false,
                            };

                            // If the projection output contains `Self`, force the user to
                            // elaborate it explicitly to avoid a lot of complexity.
                            //
                            // The "classically useful" case is the following:
                            // ```
                            //     trait MyTrait: FnMut() -> <Self as MyTrait>::MyOutput {
                            //         type MyOutput;
                            //     }
                            // ```
                            //
                            // Here, the user could theoretically write `dyn MyTrait<MyOutput = X>`,
                            // but actually supporting that would "expand" to an infinitely-long type
                            // `fix $ τ → dyn MyTrait<MyOutput = X, Output = <τ as MyTrait>::MyOutput`.
                            //
                            // Instead, we force the user to write
                            // `dyn MyTrait<MyOutput = X, Output = X>`, which is uglier but works. See
                            // the discussion in #56288 for alternatives.
                            if !references_self {
                                let key = (
                                    pred.skip_binder().def_id().0,
                                    interner.anonymize_bound_vars(pred.map_bound(|proj| {
                                        proj.projection_term.trait_ref(interner)
                                    })),
                                );
                                if !projection_bounds.contains_key(&key) {
                                    projection_bounds.insert(key, pred);
                                }
                            }
                        }
                        _ => (),
                    }
                }
            }

            // We compute the list of projection bounds taking the ordered associated types,
            // and check if there was an entry in the collected `projection_bounds`. Those
            // are computed by first taking the user-written associated types, then elaborating
            // the principal trait ref, and only using those if there was no user-written.
            // See note below about how we handle missing associated types with `Self: Sized`,
            // which are not required to be provided, but are still used if they are provided.
            let mut projection_bounds: Vec<_> = ordered_associated_types
                .into_iter()
                .filter_map(|key| projection_bounds.get(&key).copied())
                .collect();

            projection_bounds.sort_unstable_by_key(|proj| proj.skip_binder().def_id().0);

            let principal = principal.map(|principal| {
                principal.map_bound(|principal| {
                    // Verify that `dummy_self` did not leak inside default type parameters.
                    let args: Vec<_> = principal
                        .args
                        .iter()
                        // Skip `Self`
                        .skip(1)
                        .map(|arg| {
                            if arg.walk().any(|arg| arg == dummy_self_ty.into()) {
                                // FIXME: Report an error.
                                self.types.types.error.into()
                            } else {
                                arg
                            }
                        })
                        .collect();

                    ExistentialPredicate::Trait(ExistentialTraitRef::new(
                        interner,
                        principal.def_id,
                        args,
                    ))
                })
            });

            let projections = projection_bounds.into_iter().map(|proj| {
                proj.map_bound(|mut proj| {
                    // Like for trait refs, verify that `dummy_self` did not leak inside default type
                    // parameters.
                    let references_self = proj.projection_term.args.iter().skip(1).any(|arg| {
                        if arg.walk().any(|arg| arg == dummy_self_ty.into()) {
                            return true;
                        }
                        false
                    });
                    if references_self {
                        proj.projection_term = replace_dummy_self_with_error(
                            interner,
                            self.types,
                            proj.projection_term,
                        );
                    }

                    ExistentialPredicate::Projection(ExistentialProjection::erase_self_ty(
                        interner, proj,
                    ))
                })
            });

            let auto_traits = auto_traits.into_iter().map(|auto_trait| {
                Binder::dummy(ExistentialPredicate::AutoTrait(auto_trait.into()))
            });

            // N.b. principal, projections, auto traits
            Some(BoundExistentialPredicates::new_from_iter(
                interner,
                principal.into_iter().chain(projections).chain(auto_traits),
            ))
        });

        if let Some(bounds) = bounds {
            let region = match region {
                Some(it) => match it.kind() {
                    rustc_type_ir::RegionKind::ReBound(BoundVarIndexKind::Bound(db), var) => {
                        Region::new_bound(
                            self.interner,
                            db.shifted_out_to_binder(DebruijnIndex::from_u32(2)),
                            var,
                        )
                    }
                    _ => it,
                },
                None => Region::new_static(self.interner),
            };
            Ty::new_dynamic(self.interner, bounds, region)
        } else {
            // FIXME: report error
            // (additional non-auto traits, associated type rebound, or no resolved trait)
            self.types.types.error
        }
    }

    fn lower_impl_trait(&mut self, def_id: InternedOpaqueTyId, bounds: &[TypeBound]) -> ImplTrait {
        let interner = self.interner;
        cov_mark::hit!(lower_rpit);
        let args = GenericArgs::identity_for_item(interner, def_id.into());
        let self_ty = Ty::new_alias(
            self.interner,
            AliasTy::new_from_args(interner, rustc_type_ir::Opaque { def_id: def_id.into() }, args),
        );
        let (predicates, assoc_ty_bounds_start) =
            self.with_shifted_in(DebruijnIndex::from_u32(1), |ctx| {
                let mut predicates = Vec::new();
                let mut assoc_ty_bounds = Vec::new();
                for b in bounds {
                    for (pred, source) in ctx.lower_type_bound(b, self_ty, false) {
                        match source {
                            GenericPredicateSource::SelfOnly => predicates.push(pred),
                            GenericPredicateSource::AssocTyBound => assoc_ty_bounds.push(pred),
                        }
                    }
                }

                if !ctx.unsized_types.contains(&self_ty) {
                    let sized_trait = self.lang_items.Sized;
                    let sized_clause = sized_trait.map(|trait_id| {
                        let trait_ref = TraitRef::new_from_args(
                            interner,
                            trait_id.into(),
                            GenericArgs::new_from_slice(&[self_ty.into()]),
                        );
                        Clause(Predicate::new(
                            interner,
                            Binder::dummy(rustc_type_ir::PredicateKind::Clause(
                                rustc_type_ir::ClauseKind::Trait(TraitPredicate {
                                    trait_ref,
                                    polarity: rustc_type_ir::PredicatePolarity::Positive,
                                }),
                            )),
                        ))
                    });
                    predicates.extend(sized_clause);
                }

                let assoc_ty_bounds_start = predicates.len() as u32;
                predicates.extend(assoc_ty_bounds);
                (predicates, assoc_ty_bounds_start)
            });

        ImplTrait {
            predicates: Clauses::new_from_slice(&predicates).store(),
            assoc_ty_bounds_start,
        }
    }

    pub(crate) fn lower_lifetime(&mut self, lifetime: LifetimeRefId) -> Region<'db> {
        match self.resolver.resolve_lifetime(&self.store[lifetime]) {
            Some(resolution) => match resolution {
                LifetimeNs::Static => Region::new_static(self.interner),
                LifetimeNs::LifetimeParam(id) => {
                    let idx = self.generics().lifetime_param_idx(id);
                    self.region_param(id, idx)
                }
            },
            None => Region::error(self.interner),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct TyLoweringResult<T> {
    pub value: T,
    info: Option<Box<(ThinVec<TyLoweringDiagnostic>, ThinVec<AnonConstId>)>>,
}

impl<T: std::fmt::Debug> std::fmt::Debug for TyLoweringResult<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("TyLoweringResult");
        debug.field("value", &self.value);
        let diagnostics = self.diagnostics();
        if !diagnostics.is_empty() {
            debug.field("diagnostics", &diagnostics);
        }
        let defined_anon_consts = self.defined_anon_consts();
        if !defined_anon_consts.is_empty() {
            debug.field("defined_anon_consts", &defined_anon_consts);
        }
        debug.finish()
    }
}

impl<T> TyLoweringResult<T> {
    fn new(
        value: T,
        mut diagnostics: ThinVec<TyLoweringDiagnostic>,
        mut defined_anon_consts: ThinVec<AnonConstId>,
    ) -> Self {
        let info = if diagnostics.is_empty() && defined_anon_consts.is_empty() {
            None
        } else {
            diagnostics.shrink_to_fit();
            defined_anon_consts.shrink_to_fit();
            Some(Box::new((diagnostics, defined_anon_consts)))
        };
        Self { value, info }
    }

    fn from_ctx(value: T, ctx: TyLoweringContext<'_, '_>) -> Self {
        Self::new(value, ctx.diagnostics, ctx.defined_anon_consts)
    }

    fn empty(value: T) -> Self {
        Self { value, info: None }
    }

    #[inline]
    pub fn diagnostics(&self) -> &[TyLoweringDiagnostic] {
        match &self.info {
            Some(info) => &info.0,
            None => &[],
        }
    }

    #[inline]
    pub fn defined_anon_consts(&self) -> &[AnonConstId] {
        match &self.info {
            Some(info) => &info.1,
            None => &[],
        }
    }
}

fn replace_dummy_self_with_error<'db, T: TypeFoldable<DbInterner<'db>>>(
    interner: DbInterner<'db>,
    types: &DefaultAny<'db>,
    t: T,
) -> T {
    t.fold_with(&mut BottomUpFolder {
        interner,
        ty_op: |ty| {
            if ty == types.types.dyn_trait_dummy_self { types.types.error } else { ty }
        },
        lt_op: |lt| lt,
        ct_op: |ct| ct,
    })
}

pub(crate) fn lower_mutability(m: hir_def::type_ref::Mutability) -> Mutability {
    match m {
        hir_def::type_ref::Mutability::Shared => Mutability::Not,
        hir_def::type_ref::Mutability::Mut => Mutability::Mut,
    }
}

pub(crate) fn impl_trait_query<'db>(
    db: &'db dyn HirDatabase,
    impl_id: ImplId,
) -> Option<EarlyBinder<'db, TraitRef<'db>>> {
    impl_trait_with_diagnostics(db, impl_id)
        .as_ref()
        .map(|it| it.value.get(DbInterner::new_no_crate(db)))
}

#[salsa::tracked(returns(ref))]
pub(crate) fn impl_trait_with_diagnostics(
    db: &dyn HirDatabase,
    impl_id: ImplId,
) -> Option<TyLoweringResult<StoredEarlyBinder<StoredTraitRef>>> {
    let impl_data = ImplSignature::of(db, impl_id);
    let resolver = impl_id.resolver(db);
    let generics = OnceCell::new();
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        &impl_data.store,
        ExpressionStoreOwnerId::Signature(impl_id.into()),
        impl_id.into(),
        &generics,
        LifetimeElisionKind::AnonymousCreateParameter { report_in_path: true },
    );
    let self_ty = db.impl_self_ty(impl_id).skip_binder();
    let target_trait = impl_data.target_trait.as_ref()?;
    let trait_ref = ctx.lower_trait_ref(target_trait, self_ty)?;
    Some(TyLoweringResult::from_ctx(StoredEarlyBinder::bind(StoredTraitRef::new(trait_ref)), ctx))
}

impl ImplTraitId {
    #[inline]
    pub fn predicates<'db>(self, db: &'db dyn HirDatabase) -> EarlyBinder<'db, &'db [Clause<'db>]> {
        let (impl_traits, idx) = match self {
            ImplTraitId::ReturnTypeImplTrait(owner, idx) => {
                (ImplTraits::return_type_impl_traits(db, owner), idx)
            }
            ImplTraitId::TypeAliasImplTrait(owner, idx) => {
                (ImplTraits::type_alias_impl_traits(db, owner), idx)
            }
        };
        impl_traits
            .as_deref()
            .expect("owner should have opaque type")
            .get_with(|it| it.impl_traits[idx].predicates.as_ref().as_slice())
    }

    #[inline]
    pub fn self_predicates<'db>(
        self,
        db: &'db dyn HirDatabase,
    ) -> EarlyBinder<'db, &'db [Clause<'db>]> {
        let (impl_traits, idx) = match self {
            ImplTraitId::ReturnTypeImplTrait(owner, idx) => {
                (ImplTraits::return_type_impl_traits(db, owner), idx)
            }
            ImplTraitId::TypeAliasImplTrait(owner, idx) => {
                (ImplTraits::type_alias_impl_traits(db, owner), idx)
            }
        };
        let predicates =
            impl_traits.as_deref().expect("owner should have opaque type").get_with(|it| {
                let impl_trait = &it.impl_traits[idx];
                (
                    impl_trait.predicates.as_ref().as_slice(),
                    impl_trait.assoc_ty_bounds_start as usize,
                )
            });

        predicates.map_bound(|(preds, len)| &preds[..len])
    }
}

impl InternedOpaqueTyId {
    #[inline]
    pub fn predicates<'db>(self, db: &'db dyn HirDatabase) -> EarlyBinder<'db, &'db [Clause<'db>]> {
        self.loc(db).predicates(db)
    }

    #[inline]
    pub fn self_predicates<'db>(
        self,
        db: &'db dyn HirDatabase,
    ) -> EarlyBinder<'db, &'db [Clause<'db>]> {
        self.loc(db).self_predicates(db)
    }
}

#[salsa::tracked]
impl ImplTraits {
    #[salsa::tracked(returns(ref))]
    pub(crate) fn return_type_impl_traits(
        db: &dyn HirDatabase,
        def: hir_def::FunctionId,
    ) -> Option<Box<StoredEarlyBinder<ImplTraits>>> {
        // FIXME unify with fn_sig_for_fn instead of doing lowering twice, maybe
        let data = FunctionSignature::of(db, def);
        let resolver = def.resolver(db);
        let generics = OnceCell::new();
        let mut ctx_ret = TyLoweringContext::new(
            db,
            &resolver,
            &data.store,
            ExpressionStoreOwnerId::Signature(def.into()),
            def.into(),
            &generics,
            LifetimeElisionKind::Infer,
        )
        .with_impl_trait_mode(ImplTraitLoweringMode::Opaque);
        if let Some(ret_type) = data.ret_type {
            let _ret = ctx_ret.lower_ty(ret_type);
        }
        let mut return_type_impl_traits =
            ImplTraits { impl_traits: ctx_ret.impl_trait_mode.opaque_type_data };
        if return_type_impl_traits.impl_traits.is_empty() {
            None
        } else {
            return_type_impl_traits.impl_traits.shrink_to_fit();
            Some(Box::new(StoredEarlyBinder::bind(return_type_impl_traits)))
        }
    }

    #[salsa::tracked(returns(ref))]
    pub(crate) fn type_alias_impl_traits(
        db: &dyn HirDatabase,
        def: hir_def::TypeAliasId,
    ) -> Option<Box<StoredEarlyBinder<ImplTraits>>> {
        let data = TypeAliasSignature::of(db, def);
        let resolver = def.resolver(db);
        let generics = OnceCell::new();
        let mut ctx = TyLoweringContext::new(
            db,
            &resolver,
            &data.store,
            ExpressionStoreOwnerId::Signature(def.into()),
            def.into(),
            &generics,
            LifetimeElisionKind::AnonymousReportError,
        )
        .with_impl_trait_mode(ImplTraitLoweringMode::Opaque);
        if let Some(type_ref) = data.ty {
            let _ty = ctx.lower_ty(type_ref);
        }
        let mut type_alias_impl_traits =
            ImplTraits { impl_traits: ctx.impl_trait_mode.opaque_type_data };
        if type_alias_impl_traits.impl_traits.is_empty() {
            None
        } else {
            type_alias_impl_traits.impl_traits.shrink_to_fit();
            Some(Box::new(StoredEarlyBinder::bind(type_alias_impl_traits)))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TyDefId {
    BuiltinType(BuiltinType),
    AdtId(AdtId),
    TypeAliasId(TypeAliasId),
}
impl_from!(BuiltinType, AdtId(StructId, EnumId, UnionId), TypeAliasId for TyDefId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, salsa_macros::Supertype)]
pub enum ValueTyDefId {
    FunctionId(FunctionId),
    StructId(StructId),
    UnionId(UnionId),
    EnumVariantId(EnumVariantId),
    ConstId(ConstId),
    StaticId(StaticId),
}
impl_from!(FunctionId, StructId, UnionId, EnumVariantId, ConstId, StaticId for ValueTyDefId);

impl ValueTyDefId {
    pub(crate) fn to_generic_def_id(self, db: &dyn HirDatabase) -> GenericDefId {
        match self {
            Self::FunctionId(id) => id.into(),
            Self::StructId(id) => id.into(),
            Self::UnionId(id) => id.into(),
            Self::EnumVariantId(var) => var.lookup(db).parent.into(),
            Self::ConstId(id) => id.into(),
            Self::StaticId(id) => id.into(),
        }
    }
}

/// Build the declared type of an item. This depends on the namespace; e.g. for
/// `struct Foo(usize)`, we have two types: The type of the struct itself, and
/// the constructor function `(usize) -> Foo` which lives in the values
/// namespace.
pub(crate) fn ty_query<'db>(db: &'db dyn HirDatabase, def: TyDefId) -> EarlyBinder<'db, Ty<'db>> {
    let interner = DbInterner::new_no_crate(db);
    match def {
        TyDefId::BuiltinType(it) => EarlyBinder::bind(Ty::from_builtin_type(interner, it)),
        TyDefId::AdtId(it) => EarlyBinder::bind(Ty::new_adt(
            interner,
            it,
            GenericArgs::identity_for_item(interner, it.into()),
        )),
        TyDefId::TypeAliasId(it) => db.type_for_type_alias_with_diagnostics(it).value.get(),
    }
}

/// Build the declared type of a function. This should not need to look at the
/// function body.
fn type_for_fn<'db>(db: &'db dyn HirDatabase, def: FunctionId) -> EarlyBinder<'db, Ty<'db>> {
    let interner = DbInterner::new_no_crate(db);
    EarlyBinder::bind(Ty::new_fn_def(
        interner,
        CallableDefId::FunctionId(def).into(),
        GenericArgs::identity_for_item(interner, def.into()),
    ))
}

pub(crate) fn type_for_const<'db>(
    db: &'db dyn HirDatabase,
    def: ConstId,
) -> EarlyBinder<'db, Ty<'db>> {
    type_for_const_with_diagnostics(db, def).value.get()
}

/// Build the declared type of a const.
#[salsa_macros::tracked(returns(ref))]
pub(crate) fn type_for_const_with_diagnostics(
    db: &dyn HirDatabase,
    def: ConstId,
) -> TyLoweringResult<StoredEarlyBinder<StoredTy>> {
    let resolver = def.resolver(db);
    let data = ConstSignature::of(db, def);
    let parent = def.loc(db).container;
    let generics = OnceCell::new();
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        &data.store,
        ExpressionStoreOwnerId::Signature(def.into()),
        def.into(),
        &generics,
        LifetimeElisionKind::AnonymousReportError,
    );
    ctx.set_lifetime_elision(LifetimeElisionKind::for_const(ctx.interner, parent));
    let result = StoredEarlyBinder::bind(ctx.lower_ty(data.type_ref).store());
    TyLoweringResult::from_ctx(result, ctx)
}

pub(crate) fn type_for_static<'db>(
    db: &'db dyn HirDatabase,
    def: StaticId,
) -> EarlyBinder<'db, Ty<'db>> {
    type_for_static_with_diagnostics(db, def).value.get()
}

/// Build the declared type of a static.
#[salsa_macros::tracked(returns(ref))]
pub(crate) fn type_for_static_with_diagnostics(
    db: &dyn HirDatabase,
    def: StaticId,
) -> TyLoweringResult<StoredEarlyBinder<StoredTy>> {
    let resolver = def.resolver(db);
    let data = StaticSignature::of(db, def);
    let generics = OnceCell::new();
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        &data.store,
        ExpressionStoreOwnerId::Signature(def.into()),
        def.into(),
        &generics,
        LifetimeElisionKind::AnonymousReportError,
    );
    ctx.set_lifetime_elision(LifetimeElisionKind::Elided(Region::new_static(ctx.interner)));
    let result = StoredEarlyBinder::bind(ctx.lower_ty(data.type_ref).store());
    TyLoweringResult::from_ctx(result, ctx)
}

/// Build the type of a tuple struct constructor.
fn type_for_struct_constructor<'db>(
    db: &'db dyn HirDatabase,
    def: StructId,
) -> Option<EarlyBinder<'db, Ty<'db>>> {
    let struct_data = StructSignature::of(db, def);
    match struct_data.shape {
        FieldsShape::Record => None,
        FieldsShape::Unit => Some(type_for_adt(db, def.into())),
        FieldsShape::Tuple => {
            let interner = DbInterner::new_no_crate(db);
            Some(EarlyBinder::bind(Ty::new_fn_def(
                interner,
                CallableDefId::StructId(def).into(),
                GenericArgs::identity_for_item(interner, def.into()),
            )))
        }
    }
}

/// Build the type of a tuple enum variant constructor.
fn type_for_enum_variant_constructor<'db>(
    db: &'db dyn HirDatabase,
    def: EnumVariantId,
) -> Option<EarlyBinder<'db, Ty<'db>>> {
    let struct_data = def.fields(db);
    match struct_data.shape {
        FieldsShape::Record => None,
        FieldsShape::Unit => Some(type_for_adt(db, def.loc(db).parent.into())),
        FieldsShape::Tuple => {
            let interner = DbInterner::new_no_crate(db);
            Some(EarlyBinder::bind(Ty::new_fn_def(
                interner,
                CallableDefId::EnumVariantId(def).into(),
                GenericArgs::identity_for_item(interner, def.loc(db).parent.into()),
            )))
        }
    }
}

pub(crate) fn value_ty<'db>(
    db: &'db dyn HirDatabase,
    def: ValueTyDefId,
) -> Option<EarlyBinder<'db, Ty<'db>>> {
    match def {
        ValueTyDefId::FunctionId(it) => Some(type_for_fn(db, it)),
        ValueTyDefId::StructId(it) => type_for_struct_constructor(db, it),
        ValueTyDefId::UnionId(it) => Some(type_for_adt(db, it.into())),
        ValueTyDefId::EnumVariantId(it) => type_for_enum_variant_constructor(db, it),
        ValueTyDefId::ConstId(it) => Some(type_for_const(db, it)),
        ValueTyDefId::StaticId(it) => Some(type_for_static(db, it)),
    }
}

#[salsa::tracked(returns(ref), cycle_result = type_for_type_alias_with_diagnostics_cycle_result)]
pub(crate) fn type_for_type_alias_with_diagnostics(
    db: &dyn HirDatabase,
    t: TypeAliasId,
) -> TyLoweringResult<StoredEarlyBinder<StoredTy>> {
    let type_alias_data = TypeAliasSignature::of(db, t);
    let interner = DbInterner::new_no_crate(db);
    if type_alias_data.flags.contains(TypeAliasFlags::IS_EXTERN) {
        TyLoweringResult::empty(StoredEarlyBinder::bind(
            Ty::new_foreign(interner, t.into()).store(),
        ))
    } else {
        let resolver = t.resolver(db);
        let generics = OnceCell::new();
        let mut ctx = TyLoweringContext::new(
            db,
            &resolver,
            &type_alias_data.store,
            ExpressionStoreOwnerId::Signature(t.into()),
            t.into(),
            &generics,
            LifetimeElisionKind::AnonymousReportError,
        )
        .with_impl_trait_mode(ImplTraitLoweringMode::Opaque);
        let res = StoredEarlyBinder::bind(
            type_alias_data
                .ty
                .map(|type_ref| ctx.lower_ty(type_ref))
                .unwrap_or_else(|| Ty::new_error(interner, ErrorGuaranteed))
                .store(),
        );
        TyLoweringResult::from_ctx(res, ctx)
    }
}

pub(crate) fn type_for_type_alias_with_diagnostics_cycle_result(
    db: &dyn HirDatabase,
    _: salsa::Id,
    _adt: TypeAliasId,
) -> TyLoweringResult<StoredEarlyBinder<StoredTy>> {
    TyLoweringResult::empty(StoredEarlyBinder::bind(
        Ty::new_error(DbInterner::new_no_crate(db), ErrorGuaranteed).store(),
    ))
}

pub(crate) fn impl_self_ty_query<'db>(
    db: &'db dyn HirDatabase,
    impl_id: ImplId,
) -> EarlyBinder<'db, Ty<'db>> {
    impl_self_ty_with_diagnostics(db, impl_id).value.get()
}

#[salsa::tracked(returns(ref), cycle_result = impl_self_ty_with_diagnostics_cycle_result)]
pub(crate) fn impl_self_ty_with_diagnostics(
    db: &dyn HirDatabase,
    impl_id: ImplId,
) -> TyLoweringResult<StoredEarlyBinder<StoredTy>> {
    let resolver = impl_id.resolver(db);
    let generics = OnceCell::new();
    let impl_data = ImplSignature::of(db, impl_id);
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        &impl_data.store,
        ExpressionStoreOwnerId::Signature(impl_id.into()),
        impl_id.into(),
        &generics,
        LifetimeElisionKind::AnonymousCreateParameter { report_in_path: true },
    );
    let ty = ctx.lower_ty(impl_data.self_ty);
    assert!(!ty.has_escaping_bound_vars());
    TyLoweringResult::from_ctx(StoredEarlyBinder::bind(ty.store()), ctx)
}

pub(crate) fn impl_self_ty_with_diagnostics_cycle_result(
    db: &dyn HirDatabase,
    _: salsa::Id,
    _impl_id: ImplId,
) -> TyLoweringResult<StoredEarlyBinder<StoredTy>> {
    TyLoweringResult::empty(StoredEarlyBinder::bind(
        Ty::new_error(DbInterner::new_no_crate(db), ErrorGuaranteed).store(),
    ))
}

pub(crate) fn const_param_ty<'db>(db: &'db dyn HirDatabase, def: ConstParamId) -> Ty<'db> {
    let param_types = const_param_types(db, def.parent());
    match param_types.get(def.local_id()) {
        Some(ty) => ty.as_ref(),
        None => Ty::new_error(DbInterner::new_no_crate(db), ErrorGuaranteed),
    }
}

pub(crate) fn const_param_types(
    db: &dyn HirDatabase,
    def: GenericDefId,
) -> &ArenaMap<LocalTypeOrConstParamId, StoredTy> {
    &const_param_types_with_diagnostics(db, def).value
}

#[salsa::tracked(returns(ref), cycle_result = const_param_types_with_diagnostics_cycle_result)]
pub(crate) fn const_param_types_with_diagnostics(
    db: &dyn HirDatabase,
    def: GenericDefId,
) -> TyLoweringResult<ArenaMap<LocalTypeOrConstParamId, StoredTy>> {
    let mut result = ArenaMap::new();
    let (data, store) = GenericParams::with_store(db, def);
    let resolver = def.resolver(db);
    let generics = OnceCell::new();
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        store,
        ExpressionStoreOwnerId::Signature(def),
        def,
        &generics,
        LifetimeElisionKind::AnonymousReportError,
    );
    ctx.forbid_params_after(0, ForbidParamsAfterReason::ConstParamTy);
    for (local_id, param_data) in data.iter_type_or_consts() {
        if let TypeOrConstParamData::ConstParamData(param_data) = param_data {
            result.insert(local_id, ctx.lower_ty(param_data.ty).store());
        }
    }
    result.shrink_to_fit();
    TyLoweringResult::from_ctx(result, ctx)
}

fn const_param_types_with_diagnostics_cycle_result(
    _db: &dyn HirDatabase,
    _: salsa::Id,
    _def: GenericDefId,
) -> TyLoweringResult<ArenaMap<LocalTypeOrConstParamId, StoredTy>> {
    TyLoweringResult::empty(ArenaMap::default())
}

pub(crate) fn field_types_query(
    db: &dyn HirDatabase,
    variant_id: VariantId,
) -> &ArenaMap<LocalFieldId, StoredEarlyBinder<StoredTy>> {
    &field_types_with_diagnostics(db, variant_id).value
}

/// Build the type of all specific fields of a struct or enum variant.
#[salsa::tracked(returns(ref))]
pub(crate) fn field_types_with_diagnostics(
    db: &dyn HirDatabase,
    variant_id: VariantId,
) -> TyLoweringResult<ArenaMap<LocalFieldId, StoredEarlyBinder<StoredTy>>> {
    let var_data = variant_id.fields(db);
    let fields = var_data.fields();
    if fields.is_empty() {
        return TyLoweringResult::empty(ArenaMap::default());
    }

    let (resolver, generic_def): (_, GenericDefId) = match variant_id {
        VariantId::StructId(it) => (it.resolver(db), it.into()),
        VariantId::UnionId(it) => (it.resolver(db), it.into()),
        VariantId::EnumVariantId(it) => (it.resolver(db), it.lookup(db).parent.into()),
    };
    let generics = OnceCell::new();
    let mut res = ArenaMap::default();
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        &var_data.store,
        ExpressionStoreOwnerId::VariantFields(variant_id),
        generic_def,
        &generics,
        LifetimeElisionKind::AnonymousReportError,
    );
    for (field_id, field_data) in var_data.fields().iter() {
        res.insert(field_id, StoredEarlyBinder::bind(ctx.lower_ty(field_data.type_ref).store()));
    }
    TyLoweringResult::from_ctx(res, ctx)
}

#[derive(Debug, PartialEq, Eq, Default)]
pub(crate) struct SupertraitsInfo {
    /// This includes the trait itself.
    pub(crate) all_supertraits: Box<[TraitId]>,
    pub(crate) direct_supertraits: Box<[TraitId]>,
    pub(crate) defined_assoc_types: Box<[(Name, TypeAliasId)]>,
}

impl SupertraitsInfo {
    #[inline]
    pub(crate) fn query(db: &dyn HirDatabase, trait_: TraitId) -> &Self {
        return supertraits_info(db, trait_);

        #[salsa::tracked(returns(ref), cycle_result = supertraits_info_cycle)]
        fn supertraits_info(db: &dyn HirDatabase, trait_: TraitId) -> SupertraitsInfo {
            let mut all_supertraits = FxHashSet::default();
            let mut direct_supertraits = FxHashSet::default();
            let mut defined_assoc_types = FxHashSet::default();

            all_supertraits.insert(trait_);
            defined_assoc_types.extend(trait_.trait_items(db).items.iter().filter_map(
                |(name, id)| match *id {
                    AssocItemId::TypeAliasId(id) => Some((name.clone(), id)),
                    _ => None,
                },
            ));

            let resolver = trait_.resolver(db);
            let signature = TraitSignature::of(db, trait_);
            for pred in signature.generic_params.where_predicates() {
                let (WherePredicate::TypeBound { target, bound }
                | WherePredicate::ForLifetime { lifetimes: _, target, bound }) = pred
                else {
                    continue;
                };
                let (TypeBound::Path(bounded_trait, TraitBoundModifier::None)
                | TypeBound::ForLifetime(_, bounded_trait)) = *bound
                else {
                    continue;
                };
                let target = &signature.store[*target];
                match target {
                    TypeRef::TypeParam(param)
                        if param.local_id() == GenericParams::SELF_PARAM_ID_IN_SELF => {}
                    TypeRef::Path(path) if path.is_self_type() => {}
                    _ => continue,
                }
                let Some(TypeNs::TraitId(bounded_trait)) =
                    resolver.resolve_path_in_type_ns_fully(db, &signature.store[bounded_trait])
                else {
                    continue;
                };
                let SupertraitsInfo {
                    all_supertraits: bounded_trait_all_supertraits,
                    direct_supertraits: _,
                    defined_assoc_types: bounded_traits_defined_assoc_types,
                } = SupertraitsInfo::query(db, bounded_trait);
                all_supertraits.extend(bounded_trait_all_supertraits);
                direct_supertraits.insert(bounded_trait);
                defined_assoc_types.extend(bounded_traits_defined_assoc_types.iter().cloned());
            }

            SupertraitsInfo {
                all_supertraits: Box::from_iter(all_supertraits),
                direct_supertraits: Box::from_iter(direct_supertraits),
                defined_assoc_types: Box::from_iter(defined_assoc_types),
            }
        }

        fn supertraits_info_cycle(
            _db: &dyn HirDatabase,
            _: salsa::Id,
            _trait_: TraitId,
        ) -> SupertraitsInfo {
            SupertraitsInfo::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum AssocTypeShorthandResolution {
    Resolved(StoredEarlyBinder<(TypeAliasId, StoredGenericArgs)>),
    Ambiguous {
        /// If one resolution belongs to a sub-trait and one to a supertrait, this contains
        /// the sub-trait's resolution. This can be `None` if there is no trait inheritance
        /// relationship between the resolutions.
        sub_trait_resolution: Option<StoredEarlyBinder<(TypeAliasId, StoredGenericArgs)>>,
    },
    NotFound,
    Cycle,
}

/// Predicates for `param_id` of the form `P: SomeTrait`. If
/// `assoc_name` is provided, only return predicates referencing traits
/// that have an associated type of that name.
///
/// This query exists only to be used when resolving short-hand associated types
/// like `T::Item`.
///
/// See the analogous query in rustc and its comment:
/// <https://github.com/rust-lang/rust/blob/9150f844e2624eb013ec78ca08c1d416e6644026/src/librustc_typeck/astconv.rs#L46>
///
/// This is a query mostly to handle cycles somewhat gracefully; e.g. the
/// following bounds are disallowed: `T: Foo<U::Item>, U: Foo<T::Item>`, but
/// these are fine: `T: Foo<U::Item>, U: Foo<()>`.
#[tracing::instrument(skip(db), ret)]
#[salsa::tracked(returns(ref), cycle_result = resolve_type_param_assoc_type_shorthand_cycle_result)]
fn resolve_type_param_assoc_type_shorthand(
    db: &dyn HirDatabase,
    def: GenericDefId,
    param: TypeParamId,
    assoc_name: Name,
) -> AssocTypeShorthandResolution {
    let generics = generics(db, def);
    let store = generics.store();
    let generics = &OnceCell::from(generics);
    let resolver = def.resolver(db);
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        store,
        ExpressionStoreOwnerId::Signature(def),
        def,
        generics,
        LifetimeElisionKind::AnonymousReportError,
    );
    let interner = ctx.interner;
    let generics = generics.get().unwrap();
    let param_ty = Ty::new_param(interner, param, generics.type_or_const_param_idx(param.into()));

    let mut this_trait_resolution = None;
    if let GenericDefId::TraitId(containing_trait) = param.parent()
        && param.local_id() == GenericParams::SELF_PARAM_ID_IN_SELF
    {
        // Add the trait's own associated types.
        if let Some(assoc_type) =
            containing_trait.trait_items(db).associated_type_by_name(&assoc_name)
        {
            let args = GenericArgs::identity_for_item(interner, containing_trait.into());
            this_trait_resolution = Some(StoredEarlyBinder::bind((assoc_type, args.store())));
        }
    }

    let mut supertraits_resolution = None;
    for maybe_parent_generics in generics.iter_owners().rev() {
        ctx.store = maybe_parent_generics.store();
        for pred in maybe_parent_generics.where_predicates() {
            let (WherePredicate::TypeBound { target, bound }
            | WherePredicate::ForLifetime { lifetimes: _, target, bound }) = pred
            else {
                continue;
            };
            let (TypeBound::Path(bounded_trait_path, TraitBoundModifier::None)
            | TypeBound::ForLifetime(_, bounded_trait_path)) = *bound
            else {
                continue;
            };
            let Some(target) = ctx.lower_ty_only_param(*target) else { continue };
            if target != param.into() {
                continue;
            }
            let Some(TypeNs::TraitId(bounded_trait)) =
                resolver.resolve_path_in_type_ns_fully(db, &ctx.store[bounded_trait_path])
            else {
                continue;
            };
            if !SupertraitsInfo::query(db, bounded_trait)
                .defined_assoc_types
                .iter()
                .any(|(name, _)| *name == assoc_name)
            {
                continue;
            }

            let Some((bounded_trait_ref, _)) =
                ctx.lower_trait_ref_from_path(bounded_trait_path, param_ty)
            else {
                continue;
            };
            // Now, search from the start on the *bounded* trait like if we wrote `Self::Assoc`. Eventually, we'll get
            // the correct trait ref (or a cycle).
            let lookup_on_bounded_trait = resolve_type_param_assoc_type_shorthand(
                db,
                bounded_trait.into(),
                TypeParamId::trait_self(bounded_trait),
                assoc_name.clone(),
            );
            let assoc_type_and_args = match &lookup_on_bounded_trait {
                AssocTypeShorthandResolution::Resolved(trait_ref) => trait_ref,
                AssocTypeShorthandResolution::Ambiguous {
                    sub_trait_resolution: Some(trait_ref),
                } => trait_ref,
                AssocTypeShorthandResolution::Ambiguous { sub_trait_resolution: None } => {
                    return AssocTypeShorthandResolution::Ambiguous {
                        sub_trait_resolution: this_trait_resolution,
                    };
                }
                AssocTypeShorthandResolution::NotFound => {
                    never!("we checked that the trait defines this assoc type");
                    continue;
                }
                AssocTypeShorthandResolution::Cycle => return AssocTypeShorthandResolution::Cycle,
            };
            let (assoc_type, args) = assoc_type_and_args
                .get_with(|(assoc_type, args)| (*assoc_type, args.as_ref()))
                .skip_binder();
            let args = EarlyBinder::bind(args)
                .instantiate(interner, bounded_trait_ref.args)
                .skip_norm_wip();
            let current_result = StoredEarlyBinder::bind((assoc_type, args.store()));
            if let Some(this_trait_resolution) = &this_trait_resolution {
                if *this_trait_resolution == current_result {
                    continue;
                } else {
                    return AssocTypeShorthandResolution::Ambiguous {
                        sub_trait_resolution: Some(this_trait_resolution.clone()),
                    };
                }
            } else if let Some(prev_resolution) = &supertraits_resolution {
                if let AssocTypeShorthandResolution::Ambiguous {
                    sub_trait_resolution: Some(prev_resolution),
                }
                | AssocTypeShorthandResolution::Resolved(prev_resolution) = prev_resolution
                    && *prev_resolution == current_result
                {
                    continue;
                } else {
                    return AssocTypeShorthandResolution::Ambiguous { sub_trait_resolution: None };
                }
            } else {
                supertraits_resolution = Some(match lookup_on_bounded_trait {
                    AssocTypeShorthandResolution::Resolved(_) => {
                        AssocTypeShorthandResolution::Resolved(current_result)
                    }
                    AssocTypeShorthandResolution::Ambiguous { .. } => {
                        AssocTypeShorthandResolution::Ambiguous {
                            sub_trait_resolution: Some(current_result),
                        }
                    }
                    AssocTypeShorthandResolution::NotFound
                    | AssocTypeShorthandResolution::Cycle => unreachable!(),
                });
            }
        }
    }

    supertraits_resolution
        .or_else(|| this_trait_resolution.map(AssocTypeShorthandResolution::Resolved))
        .unwrap_or(AssocTypeShorthandResolution::NotFound)
}

fn resolve_type_param_assoc_type_shorthand_cycle_result(
    _db: &dyn HirDatabase,
    _: salsa::Id,
    _def: GenericDefId,
    _param: TypeParamId,
    _assoc_name: Name,
) -> AssocTypeShorthandResolution {
    AssocTypeShorthandResolution::Cycle
}

#[inline]
pub(crate) fn type_alias_bounds<'db>(
    db: &'db dyn HirDatabase,
    type_alias: TypeAliasId,
) -> EarlyBinder<'db, &'db [Clause<'db>]> {
    type_alias_bounds_with_diagnostics(db, type_alias)
        .value
        .predicates
        .get()
        .map_bound(|it| it.as_slice())
}

#[inline]
pub(crate) fn type_alias_self_bounds<'db>(
    db: &'db dyn HirDatabase,
    type_alias: TypeAliasId,
) -> EarlyBinder<'db, &'db [Clause<'db>]> {
    let TypeAliasBounds { predicates, assoc_ty_bounds_start } =
        &type_alias_bounds_with_diagnostics(db, type_alias).value;
    predicates.get().map_bound(|it| &it.as_slice()[..*assoc_ty_bounds_start as usize])
}

#[derive(PartialEq, Eq, Debug, Hash)]
pub struct TypeAliasBounds<T> {
    predicates: T,
    assoc_ty_bounds_start: u32,
}

#[salsa::tracked(returns(ref))]
pub(crate) fn type_alias_bounds_with_diagnostics(
    db: &dyn HirDatabase,
    type_alias: TypeAliasId,
) -> TyLoweringResult<TypeAliasBounds<StoredEarlyBinder<StoredClauses>>> {
    let type_alias_data = TypeAliasSignature::of(db, type_alias);
    let resolver = type_alias.resolver(db);
    let generics = OnceCell::new();
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        &type_alias_data.store,
        ExpressionStoreOwnerId::Signature(type_alias.into()),
        type_alias.into(),
        &generics,
        LifetimeElisionKind::AnonymousReportError,
    );
    let interner = ctx.interner;

    let item_args = GenericArgs::identity_for_item(interner, type_alias.into());
    let interner_ty = Ty::new_projection_from_args(interner, type_alias.into(), item_args);

    let mut bounds = Vec::new();
    let mut assoc_ty_bounds = Vec::new();
    for bound in &type_alias_data.bounds {
        ctx.lower_type_bound(bound, interner_ty, false).for_each(|(pred, source)| match source {
            GenericPredicateSource::SelfOnly => {
                bounds.push(pred);
            }
            GenericPredicateSource::AssocTyBound => {
                assoc_ty_bounds.push(pred);
            }
        });
    }

    if !ctx.unsized_types.contains(&interner_ty) {
        let sized_trait = ctx.lang_items.Sized;
        if let Some(sized_trait) = sized_trait {
            let trait_ref = TraitRef::new_from_args(
                interner,
                sized_trait.into(),
                GenericArgs::new_from_slice(&[interner_ty.into()]),
            );
            bounds.push(trait_ref.upcast(interner));
        };
    }

    let assoc_ty_bounds_start = bounds.len() as u32;
    bounds.extend(assoc_ty_bounds);

    TyLoweringResult::from_ctx(
        TypeAliasBounds {
            predicates: StoredEarlyBinder::bind(Clauses::new_from_slice(&bounds).store()),
            assoc_ty_bounds_start,
        },
        ctx,
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GenericPredicates {
    // The order is the following:
    //
    // 1. If `has_trait_implied_predicate == true`, the implicit trait predicate.
    // 2. The bounds of the associated types of the parents, coming from `Trait<Assoc: Trait>`.
    //    Note: associated type bounds from `Self::Assoc: Trait` on traits *won't* be included
    //    here, they are in 3.
    // 3. The explicit, self-only predicates for the parent.
    // 4. The explicit, self-only trait predicate for the child,
    // 5. The bounds of the associated types of the child.
    predicates: StoredEarlyBinder<StoredClauses>,
    // Keep this ordered according to the above.
    has_trait_implied_predicate: bool,
    parent_explicit_self_predicates_start: u32,
    own_predicates_start: u32,
    own_assoc_ty_bounds_start: u32,
}

#[salsa::tracked]
impl<'db> GenericPredicates {
    /// Resolve the where clause(s) of an item with generics.
    ///
    /// Diagnostics are computed only for this item's predicates, not for parents.
    #[salsa::tracked(returns(ref), cycle_result=generic_predicates_cycle_result)]
    pub fn query_with_diagnostics(
        db: &'db dyn HirDatabase,
        def: GenericDefId,
    ) -> TyLoweringResult<GenericPredicates> {
        generic_predicates(db, def)
    }
}

/// A cycle can occur from malformed code.
fn generic_predicates_cycle_result(
    db: &dyn HirDatabase,
    _: salsa::Id,
    _def: GenericDefId,
) -> TyLoweringResult<GenericPredicates> {
    TyLoweringResult::empty(GenericPredicates::from_explicit_own_predicates(
        StoredEarlyBinder::bind(Clauses::empty(DbInterner::new_no_crate(db)).store()),
    ))
}

impl GenericPredicates {
    #[inline]
    pub fn empty() -> &'static GenericPredicates {
        static EMPTY: OnceLock<GenericPredicates> = OnceLock::new();
        EMPTY.get_or_init(|| GenericPredicates {
            predicates: StoredEarlyBinder::bind(Clauses::new_from_slice(&[]).store()),
            has_trait_implied_predicate: false,
            parent_explicit_self_predicates_start: 0,
            own_predicates_start: 0,
            own_assoc_ty_bounds_start: 0,
        })
    }

    #[inline]
    pub(crate) fn from_explicit_own_predicates(
        predicates: StoredEarlyBinder<StoredClauses>,
    ) -> Self {
        let len = predicates.get().skip_binder().len() as u32;
        Self {
            predicates,
            has_trait_implied_predicate: false,
            parent_explicit_self_predicates_start: 0,
            own_predicates_start: 0,
            own_assoc_ty_bounds_start: len,
        }
    }

    #[inline]
    pub fn query(db: &dyn HirDatabase, def: GenericDefId) -> &GenericPredicates {
        &Self::query_with_diagnostics(db, def).value
    }

    #[inline]
    pub fn query_all<'db>(
        db: &'db dyn HirDatabase,
        def: GenericDefId,
    ) -> EarlyBinder<'db, impl Iterator<Item = Clause<'db>>> {
        Self::query(db, def).all_predicates()
    }

    #[inline]
    pub fn query_own_explicit<'db>(
        db: &'db dyn HirDatabase,
        def: GenericDefId,
    ) -> EarlyBinder<'db, impl Iterator<Item = Clause<'db>>> {
        Self::query(db, def).own_explicit_predicates()
    }

    #[inline]
    pub fn query_explicit<'db>(
        db: &'db dyn HirDatabase,
        def: GenericDefId,
    ) -> EarlyBinder<'db, impl Iterator<Item = Clause<'db>>> {
        Self::query(db, def).explicit_predicates()
    }

    #[inline]
    pub fn all_predicates(&self) -> EarlyBinder<'_, impl Iterator<Item = Clause<'_>>> {
        self.predicates.get().map_bound(|it| it.as_slice().iter().copied())
    }

    #[inline]
    pub fn own_explicit_predicates(&self) -> EarlyBinder<'_, impl Iterator<Item = Clause<'_>>> {
        self.predicates
            .get()
            .map_bound(|it| it.as_slice()[self.own_predicates_start as usize..].iter().copied())
    }

    #[inline]
    pub fn explicit_predicates(&self) -> EarlyBinder<'_, impl Iterator<Item = Clause<'_>>> {
        self.predicates.get().map_bound(|it| {
            it.as_slice()[usize::from(self.has_trait_implied_predicate)..].iter().copied()
        })
    }

    #[inline]
    pub fn explicit_non_assoc_types_predicates(
        &self,
    ) -> EarlyBinder<'_, impl Iterator<Item = Clause<'_>>> {
        self.predicates.get().map_bound(|it| {
            it.as_slice()[self.parent_explicit_self_predicates_start as usize
                ..self.own_assoc_ty_bounds_start as usize]
                .iter()
                .copied()
        })
    }

    #[inline]
    pub fn explicit_assoc_types_predicates(
        &self,
    ) -> EarlyBinder<'_, impl Iterator<Item = Clause<'_>>> {
        self.predicates.get().map_bound(|predicates| {
            let predicates = predicates.as_slice();
            predicates[usize::from(self.has_trait_implied_predicate)
                ..self.parent_explicit_self_predicates_start as usize]
                .iter()
                .copied()
                .chain(predicates[self.own_assoc_ty_bounds_start as usize..].iter().copied())
        })
    }
}

pub(crate) fn param_env_from_predicates<'db>(
    interner: DbInterner<'db>,
    predicates: &'db GenericPredicates,
) -> ParamEnv<'db> {
    let clauses = rustc_type_ir::elaborate::elaborate(
        interner,
        predicates.all_predicates().iter_identity().map(Unnormalized::skip_norm_wip),
    );
    let clauses = Clauses::new_from_iter(interner, clauses);

    // FIXME: We should normalize projections here, like rustc does.
    ParamEnv { clauses }
}

pub(crate) fn trait_environment<'db>(
    db: &'db dyn HirDatabase,
    def: ExpressionStoreOwnerId,
) -> ParamEnv<'db> {
    let def = def.generic_def(db);

    return ParamEnv { clauses: trait_environment_query(db, def).as_ref() };

    #[salsa::tracked(returns(ref))]
    pub(crate) fn trait_environment_query(
        db: &dyn HirDatabase,
        def: GenericDefId,
    ) -> StoredClauses {
        let module = def.module(db);
        let interner = DbInterner::new_with(db, module.krate(db));
        let predicates = GenericPredicates::query(db, def);
        param_env_from_predicates(interner, predicates).clauses.store()
    }
}

/// Resolve the where clause(s) of an item with generics,
/// with a given filter
#[tracing::instrument(skip(db), ret)]
fn generic_predicates(
    db: &dyn HirDatabase,
    def: GenericDefId,
) -> TyLoweringResult<GenericPredicates> {
    let generics = generics(db, def);
    let store = generics.store();
    let generics = &OnceCell::from(generics);
    let resolver = def.resolver(db);
    let interner = DbInterner::new_no_crate(db);
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        store,
        ExpressionStoreOwnerId::Signature(def),
        def,
        generics,
        LifetimeElisionKind::AnonymousReportError,
    );
    let generics = generics.get().unwrap();
    let sized_trait = ctx.lang_items.Sized;

    // We need to lower parents and self separately - see the comment below lowering of implicit
    // `Sized` predicates for why.
    let mut own_predicates = Vec::new();
    let mut parent_predicates = Vec::new();
    let mut own_assoc_ty_bounds = Vec::new();
    let mut parent_assoc_ty_bounds = Vec::new();
    let own_implicit_trait_predicate = implicit_trait_predicate(interner, def);
    let parent_implicit_trait_predicate = if let Some(parent) = generics.parent() {
        implicit_trait_predicate(interner, parent.def())
    } else {
        None
    };
    for maybe_parent_generics in generics.iter_owners() {
        // Collect only diagnostics from the child, not including parents.
        ctx.diagnostics.clear();

        ctx.store = maybe_parent_generics.store();
        for pred in maybe_parent_generics.where_predicates() {
            tracing::debug!(?pred);
            for (pred, source) in ctx.lower_where_predicate(pred, false) {
                match source {
                    GenericPredicateSource::SelfOnly => {
                        if maybe_parent_generics.def() == def {
                            own_predicates.push(pred);
                        } else {
                            parent_predicates.push(pred);
                        }
                    }
                    GenericPredicateSource::AssocTyBound => {
                        if maybe_parent_generics.def() == def {
                            own_assoc_ty_bounds.push(pred);
                        } else {
                            parent_assoc_ty_bounds.push(pred);
                        }
                    }
                }
            }
        }

        if maybe_parent_generics.def() == def {
            push_const_arg_has_type_predicates(db, &mut own_predicates, maybe_parent_generics);
        } else {
            push_const_arg_has_type_predicates(db, &mut parent_predicates, maybe_parent_generics);
        }

        if let Some(sized_trait) = sized_trait {
            let mut add_sized_clause = |param_idx, param_id, param_data| {
                let (
                    GenericParamId::TypeParamId(param_id),
                    GenericParamDataRef::TypeParamData(param_data),
                ) = (param_id, param_data)
                else {
                    return;
                };

                if param_data.provenance == TypeParamProvenance::TraitSelf {
                    return;
                }

                let param_ty = Ty::new_param(interner, param_id, param_idx);
                if ctx.unsized_types.contains(&param_ty) {
                    return;
                }
                let trait_ref = TraitRef::new_from_args(
                    interner,
                    sized_trait.into(),
                    GenericArgs::new_from_slice(&[param_ty.into()]),
                );
                let clause = Clause(Predicate::new(
                    interner,
                    Binder::dummy(rustc_type_ir::PredicateKind::Clause(
                        rustc_type_ir::ClauseKind::Trait(TraitPredicate {
                            trait_ref,
                            polarity: rustc_type_ir::PredicatePolarity::Positive,
                        }),
                    )),
                ));
                if maybe_parent_generics.def() == def {
                    own_predicates.push(clause);
                } else {
                    parent_predicates.push(clause);
                }
            };
            maybe_parent_generics.iter_with_idx().for_each(|(param_idx, param_id, param_data)| {
                add_sized_clause(param_idx, param_id, param_data);
            });
        }

        // We do not clear `ctx.unsized_types`, as the `?Sized` clause of a child (e.g. an associated type) can
        // be declared on the parent (e.g. the trait). It is nevertheless fine to register the implicit `Sized`
        // predicates before lowering the child, as a child cannot define a `?Sized` predicate for its parent.
        // But we do have to lower the parent first.
    }

    let diagnostics = mem::take(&mut ctx.diagnostics);
    let defined_anon_consts = mem::take(&mut ctx.defined_anon_consts);

    let predicates = parent_implicit_trait_predicate
        .iter()
        .chain(own_implicit_trait_predicate.iter())
        .chain(parent_assoc_ty_bounds.iter())
        .chain(parent_predicates.iter())
        .chain(own_predicates.iter())
        .chain(own_assoc_ty_bounds.iter())
        .copied()
        .collect::<Vec<_>>();
    let has_trait_implied_predicate =
        parent_implicit_trait_predicate.is_some() || own_implicit_trait_predicate.is_some();
    let parent_explicit_self_predicates_start =
        has_trait_implied_predicate as u32 + parent_assoc_ty_bounds.len() as u32;
    let own_predicates_start =
        parent_explicit_self_predicates_start + parent_predicates.len() as u32;
    let own_assoc_ty_bounds_start = own_predicates_start + own_predicates.len() as u32;

    let predicates = GenericPredicates {
        has_trait_implied_predicate,
        parent_explicit_self_predicates_start,
        own_predicates_start,
        own_assoc_ty_bounds_start,
        predicates: StoredEarlyBinder::bind(Clauses::new_from_slice(&predicates).store()),
    };
    return TyLoweringResult::new(predicates, diagnostics, defined_anon_consts);

    fn implicit_trait_predicate<'db>(
        interner: DbInterner<'db>,
        def: GenericDefId,
    ) -> Option<Clause<'db>> {
        // For traits, add `Self: Trait` predicate. This is
        // not part of the predicates that a user writes, but it
        // is something that one must prove in order to invoke a
        // method or project an associated type.
        //
        // In the chalk setup, this predicate is not part of the
        // "predicates" for a trait item. But it is useful in
        // rustc because if you directly (e.g.) invoke a trait
        // method like `Trait::method(...)`, you must naturally
        // prove that the trait applies to the types that were
        // used, and adding the predicate into this list ensures
        // that this is done.
        if let GenericDefId::TraitId(def_id) = def {
            Some(TraitRef::identity(interner, def_id.into()).upcast(interner))
        } else {
            None
        }
    }
}

fn push_const_arg_has_type_predicates<'db>(
    db: &'db dyn HirDatabase,
    predicates: &mut Vec<Clause<'db>>,
    single_generics: &SingleGenerics<'db>,
) {
    let interner = DbInterner::new_no_crate(db);
    for (param_index, param_id, _) in single_generics.iter_with_idx() {
        let GenericParamId::ConstParamId(param_id) = param_id else { continue };
        predicates.push(Clause(
            ClauseKind::ConstArgHasType(
                Const::new_param(interner, ParamConst { id: param_id, index: param_index }),
                db.const_param_ty(param_id),
            )
            .upcast(interner),
        ));
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GenericDefaults(ThinVec<Option<StoredEarlyBinder<StoredGenericArg>>>);

impl GenericDefaults {
    #[inline]
    pub fn as_ref(&self) -> GenericDefaultsRef<'_> {
        GenericDefaultsRef(&self.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GenericDefaultsRef<'db>(&'db [Option<StoredEarlyBinder<StoredGenericArg>>]);

impl<'db> GenericDefaultsRef<'db> {
    #[inline]
    pub fn get(self, idx: usize) -> Option<EarlyBinder<'db, GenericArg<'db>>> {
        Some(self.0.get(idx)?.as_ref()?.get())
    }
}

pub(crate) fn generic_defaults(db: &dyn HirDatabase, def: GenericDefId) -> GenericDefaultsRef<'_> {
    generic_defaults_with_diagnostics(db, def).value.as_ref()
}

/// Resolve the default type params from generics.
///
/// Diagnostics are only returned for this `GenericDefId` (returned defaults include parents).
#[salsa_macros::tracked(returns(ref), cycle_result = generic_defaults_with_diagnostics_cycle_result)]
pub(crate) fn generic_defaults_with_diagnostics(
    db: &dyn HirDatabase,
    def: GenericDefId,
) -> TyLoweringResult<GenericDefaults> {
    let generics = generics(db, def);
    if generics.has_no_params() {
        return TyLoweringResult::empty(GenericDefaults(ThinVec::new()));
    }
    let resolver = def.resolver(db);

    let store_for_self = generics.store();
    let generics = &OnceCell::from(generics);
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        store_for_self,
        ExpressionStoreOwnerId::Signature(def),
        def,
        generics,
        LifetimeElisionKind::AnonymousReportError,
    )
    .with_impl_trait_mode(ImplTraitLoweringMode::Disallowed);
    let generics = generics.get().unwrap();
    let mut defaults = ThinVec::new();
    if let Some(parent) = generics.parent() {
        ctx.store = parent.store();
        defaults.extend(
            parent.iter_with_idx().map(|(idx, _id, p)| handle_generic_param(&mut ctx, idx, p)),
        );
    }
    ctx.diagnostics.clear(); // Don't include diagnostics from the parent.
    ctx.defined_anon_consts.clear();
    ctx.store = store_for_self;
    defaults.extend(
        generics.iter_self_with_idx().map(|(idx, _id, p)| handle_generic_param(&mut ctx, idx, p)),
    );
    defaults.shrink_to_fit();
    return TyLoweringResult::from_ctx(GenericDefaults(defaults), ctx);

    fn handle_generic_param<'db>(
        ctx: &mut TyLoweringContext<'db, '_>,
        idx: u32,
        p: GenericParamDataRef<'_>,
    ) -> Option<StoredEarlyBinder<StoredGenericArg>> {
        ctx.forbid_params_after(idx, ForbidParamsAfterReason::LoweringParamDefault);
        match p {
            GenericParamDataRef::TypeParamData(p) => {
                let ty = p.default.map(|ty| ctx.lower_ty(ty));
                ty.map(|ty| StoredEarlyBinder::bind(GenericArg::from(ty).store()))
            }
            GenericParamDataRef::ConstParamData(p) => {
                let val = p.default.map(|c| {
                    let param_ty = ctx.lower_ty(p.ty);
                    let c = ctx.lower_const(c, param_ty);
                    GenericArg::from(c).store()
                });
                val.map(StoredEarlyBinder::bind)
            }
            GenericParamDataRef::LifetimeParamData(_) => None,
        }
    }
}

fn generic_defaults_with_diagnostics_cycle_result(
    _db: &dyn HirDatabase,
    _: salsa::Id,
    _def: GenericDefId,
) -> TyLoweringResult<GenericDefaults> {
    TyLoweringResult::empty(GenericDefaults(ThinVec::new()))
}

/// Build the signature of a callable item (function, struct or enum variant).
pub(crate) fn callable_item_signature<'db>(
    db: &'db dyn HirDatabase,
    def: CallableDefId,
) -> EarlyBinder<'db, PolyFnSig<'db>> {
    callable_item_signature_with_diagnostics(db, def).value.get()
}

#[salsa::tracked(returns(ref))]
pub(crate) fn callable_item_signature_with_diagnostics(
    db: &dyn HirDatabase,
    def: CallableDefId,
) -> TyLoweringResult<StoredEarlyBinder<StoredPolyFnSig>> {
    match def {
        CallableDefId::FunctionId(f) => fn_sig_for_fn(db, f),
        CallableDefId::StructId(s) => TyLoweringResult::empty(fn_sig_for_struct_constructor(db, s)),
        CallableDefId::EnumVariantId(e) => {
            TyLoweringResult::empty(fn_sig_for_enum_variant_constructor(db, e))
        }
    }
}

fn fn_sig_for_fn(
    db: &dyn HirDatabase,
    def: FunctionId,
) -> TyLoweringResult<StoredEarlyBinder<StoredPolyFnSig>> {
    let data = FunctionSignature::of(db, def);
    let resolver = def.resolver(db);
    let interner = DbInterner::new_no_crate(db);
    let generics = OnceCell::new();
    let mut ctx_params = TyLoweringContext::new(
        db,
        &resolver,
        &data.store,
        ExpressionStoreOwnerId::Signature(def.into()),
        def.into(),
        &generics,
        LifetimeElisionKind::for_fn_params(data),
    );
    let params = data.params.iter().map(|&tr| ctx_params.lower_ty(tr));

    let mut ctx_ret = TyLoweringContext::new(
        db,
        &resolver,
        &data.store,
        ExpressionStoreOwnerId::Signature(def.into()),
        def.into(),
        &generics,
        LifetimeElisionKind::for_fn_ret(interner),
    )
    .with_impl_trait_mode(ImplTraitLoweringMode::Opaque);
    let ret = match data.ret_type {
        Some(ret_type) => ctx_ret.lower_ty(ret_type),
        None => Ty::new_unit(interner),
    };

    let inputs_and_output = Tys::new_from_iter(interner, params.chain(Some(ret)));

    ctx_params.diagnostics.extend(ctx_ret.diagnostics);
    ctx_params.defined_anon_consts.extend(ctx_ret.defined_anon_consts);

    // If/when we track late bound vars, we need to switch this to not be `dummy`
    let result = StoredEarlyBinder::bind(StoredPolyFnSig::new(Binder::dummy(FnSig {
        inputs_and_output,
        fn_sig_kind: FnSigKind::new(
            data.abi,
            if data.is_unsafe() { Safety::Unsafe } else { Safety::Safe },
            data.is_varargs(),
        ),
    })));
    TyLoweringResult::from_ctx(result, ctx_params)
}

fn type_for_adt<'db>(db: &'db dyn HirDatabase, adt: AdtId) -> EarlyBinder<'db, Ty<'db>> {
    let interner = DbInterner::new_no_crate(db);
    let args = GenericArgs::identity_for_item(interner, adt.into());
    let ty = Ty::new_adt(interner, adt, args);
    EarlyBinder::bind(ty)
}

fn fn_sig_for_struct_constructor(
    db: &dyn HirDatabase,
    def: StructId,
) -> StoredEarlyBinder<StoredPolyFnSig> {
    let field_tys = db.field_types(def.into());
    let params = field_tys.iter().map(|(_, ty)| ty.get().skip_binder());
    let ret = type_for_adt(db, def.into()).skip_binder();

    let inputs_and_output =
        Tys::new_from_iter(DbInterner::new_no_crate(db), params.chain(Some(ret)));
    StoredEarlyBinder::bind(StoredPolyFnSig::new(Binder::dummy(FnSig {
        fn_sig_kind: FnSigKind::new(ExternAbi::Rust, Safety::Safe, false),
        inputs_and_output,
    })))
}

fn fn_sig_for_enum_variant_constructor(
    db: &dyn HirDatabase,
    def: EnumVariantId,
) -> StoredEarlyBinder<StoredPolyFnSig> {
    let field_tys = db.field_types(def.into());
    let params = field_tys.iter().map(|(_, ty)| ty.get().skip_binder());
    let parent = def.lookup(db).parent;
    let ret = type_for_adt(db, parent.into()).skip_binder();

    let inputs_and_output =
        Tys::new_from_iter(DbInterner::new_no_crate(db), params.chain(Some(ret)));
    StoredEarlyBinder::bind(StoredPolyFnSig::new(Binder::dummy(FnSig {
        fn_sig_kind: FnSigKind::new(ExternAbi::Rust, Safety::Safe, false),
        inputs_and_output,
    })))
}

// FIXME: Remove this.
pub(crate) fn associated_ty_item_bounds<'db>(
    db: &'db dyn HirDatabase,
    type_alias: TypeAliasId,
) -> EarlyBinder<'db, BoundExistentialPredicates<'db>> {
    let type_alias_data = TypeAliasSignature::of(db, type_alias);
    let resolver = type_alias.resolver(db);
    let interner = DbInterner::new_no_crate(db);
    let generics = OnceCell::new();
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        &type_alias_data.store,
        ExpressionStoreOwnerId::Signature(type_alias.into()),
        type_alias.into(),
        &generics,
        LifetimeElisionKind::AnonymousReportError,
    );
    // FIXME: we should never create non-existential predicates in the first place
    // For now, use an error type so we don't run into dummy binder issues
    let self_ty = Ty::new_error(interner, ErrorGuaranteed);

    let mut bounds = Vec::new();
    for bound in &type_alias_data.bounds {
        ctx.lower_type_bound(bound, self_ty, false).for_each(|(pred, _)| {
            if let Some(bound) = pred
                .kind()
                .map_bound(|c| match c {
                    rustc_type_ir::ClauseKind::Trait(t) => {
                        let id = t.def_id();
                        let is_auto = TraitSignature::of(db, id.0).flags.contains(TraitFlags::AUTO);
                        if is_auto {
                            Some(ExistentialPredicate::AutoTrait(t.def_id()))
                        } else {
                            Some(ExistentialPredicate::Trait(ExistentialTraitRef::new_from_args(
                                interner,
                                t.def_id(),
                                GenericArgs::new_from_slice(&t.trait_ref.args[1..]),
                            )))
                        }
                    }
                    rustc_type_ir::ClauseKind::Projection(p) => Some(
                        ExistentialPredicate::Projection(ExistentialProjection::new_from_args(
                            interner,
                            p.def_id(),
                            GenericArgs::new_from_slice(&p.projection_term.args[1..]),
                            p.term,
                        )),
                    ),
                    rustc_type_ir::ClauseKind::TypeOutlives(_) => None,
                    rustc_type_ir::ClauseKind::RegionOutlives(_)
                    | rustc_type_ir::ClauseKind::ConstArgHasType(_, _)
                    | rustc_type_ir::ClauseKind::WellFormed(_)
                    | rustc_type_ir::ClauseKind::ConstEvaluatable(_)
                    | rustc_type_ir::ClauseKind::HostEffect(_)
                    | rustc_type_ir::ClauseKind::UnstableFeature(_) => unreachable!(),
                })
                .transpose()
            {
                bounds.push(bound);
            }
        });
    }

    if !ctx.unsized_types.contains(&self_ty)
        && let Some(sized_trait) = ctx.lang_items.Sized
    {
        let sized_clause = Binder::dummy(ExistentialPredicate::Trait(ExistentialTraitRef::new(
            interner,
            sized_trait.into(),
            [] as [GenericArg<'_>; 0],
        )));
        bounds.push(sized_clause);
    }

    EarlyBinder::bind(BoundExistentialPredicates::new_from_slice(&bounds))
}

pub(crate) fn associated_type_by_name_including_super_traits_allow_ambiguity<'db>(
    db: &'db dyn HirDatabase,
    trait_ref: TraitRef<'db>,
    name: Name,
) -> Option<(TypeAliasId, GenericArgs<'db>)> {
    let (AssocTypeShorthandResolution::Resolved(assoc_type)
    | AssocTypeShorthandResolution::Ambiguous { sub_trait_resolution: Some(assoc_type) }) =
        resolve_type_param_assoc_type_shorthand(
            db,
            trait_ref.def_id.0.into(),
            TypeParamId::trait_self(trait_ref.def_id.0),
            name.clone(),
        )
    else {
        return None;
    };
    let (assoc_type, trait_args) = assoc_type
        .get_with(|(assoc_type, trait_args)| (*assoc_type, trait_args.as_ref()))
        .skip_binder();
    let interner = DbInterner::new_no_crate(db);
    Some((
        assoc_type,
        EarlyBinder::bind(trait_args).instantiate(interner, trait_ref.args).skip_norm_wip(),
    ))
}
