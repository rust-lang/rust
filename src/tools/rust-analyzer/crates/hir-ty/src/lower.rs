//! Methods for lowering the HIR to types. There are two main cases here:
//!
//!  - Lowering a type reference like `&usize` or `Option<foo::bar::Baz>` to a
//!    type: The entry point for this is `TyLoweringContext::lower_ty`.
//!  - Building the type for an item: This happens through the `ty` query.
//!
//! This usually involves resolving names, collecting generic arguments etc.
pub(crate) mod diagnostics;
pub(crate) mod path;

use std::{cell::OnceCell, iter, mem};

use arrayvec::ArrayVec;
use either::Either;
use hir_def::{
    AdtId, AssocItemId, CallableDefId, ConstId, ConstParamId, DefWithBodyId, EnumId, EnumVariantId,
    FunctionId, GeneralConstId, GenericDefId, GenericParamId, HasModule, ImplId, ItemContainerId,
    LifetimeParamId, LocalFieldId, Lookup, StaticId, StructId, TypeAliasId, TypeOrConstParamId,
    TypeParamId, UnionId, VariantId,
    builtin_type::BuiltinType,
    expr_store::{ExpressionStore, HygieneId, path::Path},
    hir::generics::{
        GenericParamDataRef, TypeOrConstParamData, TypeParamProvenance, WherePredicate,
    },
    item_tree::FieldsShape,
    lang_item::LangItems,
    resolver::{HasResolver, LifetimeNs, Resolver, TypeNs, ValueNs},
    signatures::{FunctionSignature, TraitFlags, TypeAliasFlags},
    type_ref::{
        ConstRef, LifetimeRefId, LiteralConstRef, PathId, TraitBoundModifier,
        TraitRef as HirTraitRef, TypeBound, TypeRef, TypeRefId,
    },
};
use hir_expand::name::Name;
use la_arena::{Arena, ArenaMap, Idx};
use path::{PathDiagnosticCallback, PathLoweringContext};
use rustc_ast_ir::Mutability;
use rustc_hash::FxHashSet;
use rustc_pattern_analysis::Captures;
use rustc_type_ir::{
    AliasTyKind, BoundVarIndexKind, ConstKind, DebruijnIndex, ExistentialPredicate,
    ExistentialProjection, ExistentialTraitRef, FnSig, Interner, OutlivesPredicate, TermKind,
    TyKind::{self},
    TypeFoldable, TypeVisitableExt, Upcast, UpcastFrom, elaborate,
    inherent::{
        Clause as _, GenericArg as _, GenericArgs as _, IntoKind as _, Region as _, SliceLike,
        Ty as _,
    },
};
use smallvec::{SmallVec, smallvec};
use stdx::{impl_from, never};
use tracing::debug;
use triomphe::{Arc, ThinArc};

use crate::{
    FnAbi, ImplTraitId, TyLoweringDiagnostic, TyLoweringDiagnosticKind,
    consteval::intern_const_ref,
    db::{HirDatabase, InternedOpaqueTyId},
    generics::{Generics, generics, trait_self_param_idx},
    next_solver::{
        AliasTy, Binder, BoundExistentialPredicates, Clause, ClauseKind, Clauses, Const,
        DbInterner, EarlyBinder, EarlyParamRegion, ErrorGuaranteed, FxIndexMap, GenericArg,
        GenericArgs, ParamConst, ParamEnv, PolyFnSig, Predicate, Region, SolverDefId,
        TraitPredicate, TraitRef, Ty, Tys, UnevaluatedConst, abi::Safety, util::BottomUpFolder,
    },
};

pub(crate) struct PathDiagnosticCallbackData(pub(crate) TypeRefId);

#[derive(PartialEq, Eq, Debug, Hash)]
pub struct ImplTraits<'db> {
    pub(crate) impl_traits: Arena<ImplTrait<'db>>,
}

#[derive(PartialEq, Eq, Debug, Hash)]
pub struct ImplTrait<'db> {
    pub(crate) predicates: Box<[Clause<'db>]>,
}

pub type ImplTraitIdx<'db> = Idx<ImplTrait<'db>>;

#[derive(Debug, Default)]
struct ImplTraitLoweringState<'db> {
    /// When turning `impl Trait` into opaque types, we have to collect the
    /// bounds at the same time to get the IDs correct (without becoming too
    /// complicated).
    mode: ImplTraitLoweringMode,
    // This is structured as a struct with fields and not as an enum because it helps with the borrow checker.
    opaque_type_data: Arena<ImplTrait<'db>>,
}

impl<'db> ImplTraitLoweringState<'db> {
    fn new(mode: ImplTraitLoweringMode) -> ImplTraitLoweringState<'db> {
        Self { mode, opaque_type_data: Arena::new() }
    }
}

#[derive(Debug, Clone)]
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

#[derive(Debug)]
pub struct TyLoweringContext<'db, 'a> {
    pub db: &'db dyn HirDatabase,
    interner: DbInterner<'db>,
    lang_items: &'db LangItems,
    resolver: &'a Resolver<'db>,
    store: &'a ExpressionStore,
    def: GenericDefId,
    generics: OnceCell<Generics>,
    in_binders: DebruijnIndex,
    impl_trait_mode: ImplTraitLoweringState<'db>,
    /// Tracks types with explicit `?Sized` bounds.
    pub(crate) unsized_types: FxHashSet<Ty<'db>>,
    pub(crate) diagnostics: Vec<TyLoweringDiagnostic>,
    lifetime_elision: LifetimeElisionKind<'db>,
    /// When lowering the defaults for generic params, this contains the index of the currently lowered param.
    /// We disallow referring to later params, or to ADT's `Self`.
    lowering_param_default: Option<u32>,
}

impl<'db, 'a> TyLoweringContext<'db, 'a> {
    pub fn new(
        db: &'db dyn HirDatabase,
        resolver: &'a Resolver<'db>,
        store: &'a ExpressionStore,
        def: GenericDefId,
        lifetime_elision: LifetimeElisionKind<'db>,
    ) -> Self {
        let impl_trait_mode = ImplTraitLoweringState::new(ImplTraitLoweringMode::Disallowed);
        let in_binders = DebruijnIndex::ZERO;
        let interner = DbInterner::new_with(db, resolver.krate());
        Self {
            db,
            // Can provide no block since we don't use it for trait solving.
            interner,
            lang_items: interner.lang_items(),
            resolver,
            def,
            generics: Default::default(),
            store,
            in_binders,
            impl_trait_mode,
            unsized_types: FxHashSet::default(),
            diagnostics: Vec::new(),
            lifetime_elision,
            lowering_param_default: None,
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

    pub(crate) fn lowering_param_default(&mut self, index: u32) {
        self.lowering_param_default = Some(index);
    }

    pub(crate) fn push_diagnostic(&mut self, type_ref: TypeRefId, kind: TyLoweringDiagnosticKind) {
        self.diagnostics.push(TyLoweringDiagnostic { source: type_ref, kind });
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
        let const_ref = &self.store[const_ref.expr];
        match const_ref {
            hir_def::hir::Expr::Path(path) => {
                self.path_to_const(path).unwrap_or_else(|| unknown_const(const_type))
            }
            hir_def::hir::Expr::Literal(literal) => intern_const_ref(
                self.db,
                &match *literal {
                    hir_def::hir::Literal::Float(_, _)
                    | hir_def::hir::Literal::String(_)
                    | hir_def::hir::Literal::ByteString(_)
                    | hir_def::hir::Literal::CString(_) => LiteralConstRef::Unknown,
                    hir_def::hir::Literal::Char(c) => LiteralConstRef::Char(c),
                    hir_def::hir::Literal::Bool(b) => LiteralConstRef::Bool(b),
                    hir_def::hir::Literal::Int(val, _) => LiteralConstRef::Int(val),
                    hir_def::hir::Literal::Uint(val, _) => LiteralConstRef::UInt(val),
                },
                const_type,
                self.resolver.krate(),
            ),
            hir_def::hir::Expr::UnaryOp { expr: inner_expr, op: hir_def::hir::UnaryOp::Neg } => {
                if let hir_def::hir::Expr::Literal(literal) = &self.store[*inner_expr] {
                    // Only handle negation for signed integers and floats
                    match literal {
                        hir_def::hir::Literal::Int(_, _) | hir_def::hir::Literal::Float(_, _) => {
                            if let Some(negated_literal) = literal.clone().negate() {
                                intern_const_ref(
                                    self.db,
                                    &negated_literal.into(),
                                    const_type,
                                    self.resolver.krate(),
                                )
                            } else {
                                unknown_const(const_type)
                            }
                        }
                        // For unsigned integers, chars, bools, etc., negation is not meaningful
                        _ => unknown_const(const_type),
                    }
                } else {
                    unknown_const(const_type)
                }
            }
            _ => unknown_const(const_type),
        }
    }

    pub(crate) fn path_to_const(&mut self, path: &Path) -> Option<Const<'db>> {
        match self.resolver.resolve_path_in_value_ns_fully(self.db, path, HygieneId::ROOT) {
            Some(ValueNs::GenericParam(p)) => {
                let args = self.generics();
                match args.type_or_const_param_idx(p.into()) {
                    Some(idx) => Some(self.const_param(p, idx as u32)),
                    None => {
                        never!(
                            "Generic list doesn't contain this param: {:?}, {:?}, {:?}",
                            args,
                            path,
                            p
                        );
                        None
                    }
                }
            }
            Some(ValueNs::ConstId(c)) => {
                let args = GenericArgs::new_from_iter(self.interner, []);
                Some(Const::new(
                    self.interner,
                    rustc_type_ir::ConstKind::Unevaluated(UnevaluatedConst::new(
                        GeneralConstId::ConstId(c).into(),
                        args,
                    )),
                ))
            }
            _ => None,
        }
    }

    pub(crate) fn lower_path_as_const(&mut self, path: &Path, const_type: Ty<'db>) -> Const<'db> {
        self.path_to_const(path).unwrap_or_else(|| unknown_const(const_type))
    }

    fn generics(&self) -> &Generics {
        self.generics.get_or_init(|| generics(self.db, self.def))
    }

    fn param_index_is_disallowed(&self, index: u32) -> bool {
        self.lowering_param_default
            .is_some_and(|disallow_params_after| index >= disallow_params_after)
    }

    fn type_param(&mut self, id: TypeParamId, index: u32) -> Ty<'db> {
        if self.param_index_is_disallowed(index) {
            // FIXME: Report an error.
            Ty::new_error(self.interner, ErrorGuaranteed)
        } else {
            Ty::new_param(self.interner, id, index)
        }
    }

    fn const_param(&mut self, id: ConstParamId, index: u32) -> Const<'db> {
        if self.param_index_is_disallowed(index) {
            // FIXME: Report an error.
            Const::error(self.interner)
        } else {
            Const::new_param(self.interner, ParamConst { id, index })
        }
    }

    fn region_param(&mut self, id: LifetimeParamId, index: u32) -> Region<'db> {
        if self.param_index_is_disallowed(index) {
            // FIXME: Report an error.
            Region::error(self.interner)
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
            TypeRef::Never => Ty::new(interner, TyKind::Never),
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
                let (idx, _data) =
                    generics.type_or_const_param(type_param_id.into()).expect("matching generics");
                self.type_param(type_param_id, idx as u32)
            }
            &TypeRef::RawPtr(inner, mutability) => {
                let inner_ty = self.lower_ty(inner);
                Ty::new(interner, TyKind::RawPtr(inner_ty, lower_mutability(mutability)))
            }
            TypeRef::Array(array) => {
                let inner_ty = self.lower_ty(array.ty);
                let const_len = self.lower_const(array.len, Ty::new_usize(interner));
                Ty::new_array_with_const_len(interner, inner_ty, const_len)
            }
            &TypeRef::Slice(inner) => {
                let inner_ty = self.lower_ty(inner);
                Ty::new_slice(interner, inner_ty)
            }
            TypeRef::Reference(ref_) => {
                let inner_ty = self.lower_ty(ref_.ty);
                // FIXME: It should infer the eldided lifetimes instead of stubbing with error
                let lifetime = ref_
                    .lifetime
                    .map_or_else(|| Region::error(interner), |lr| self.lower_lifetime(lr));
                Ty::new_ref(interner, lifetime, inner_ty, lower_mutability(ref_.mutability))
            }
            TypeRef::Placeholder => Ty::new_error(interner, ErrorGuaranteed),
            TypeRef::Fn(fn_) => {
                let substs = self.with_shifted_in(
                    DebruijnIndex::from_u32(1),
                    |ctx: &mut TyLoweringContext<'_, '_>| {
                        Tys::new_from_iter(
                            interner,
                            fn_.params.iter().map(|&(_, tr)| ctx.lower_ty(tr)),
                        )
                    },
                );
                Ty::new_fn_ptr(
                    interner,
                    Binder::dummy(FnSig {
                        abi: fn_.abi.as_ref().map_or(FnAbi::Rust, FnAbi::from_symbol),
                        safety: if fn_.is_unsafe { Safety::Unsafe } else { Safety::Safe },
                        c_variadic: fn_.is_varargs,
                        inputs_and_output: substs,
                    }),
                )
            }
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
                        let idx = self
                            .impl_trait_mode
                            .opaque_type_data
                            .alloc(ImplTrait { predicates: Box::default() });

                        let impl_trait_id = origin.either(
                            |f| ImplTraitId::ReturnTypeImplTrait(f, idx),
                            |a| ImplTraitId::TypeAliasImplTrait(a, idx),
                        );
                        let opaque_ty_id: SolverDefId =
                            self.db.intern_impl_trait_id(impl_trait_id).into();

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

                        let args = GenericArgs::identity_for_item(self.interner, opaque_ty_id);
                        Ty::new_alias(
                            self.interner,
                            AliasTyKind::Opaque,
                            AliasTy::new_from_args(self.interner, opaque_ty_id, args),
                        )
                    }
                    ImplTraitLoweringMode::Disallowed => {
                        // FIXME: report error
                        Ty::new_error(self.interner, ErrorGuaranteed)
                    }
                }
            }
            TypeRef::Error => Ty::new_error(self.interner, ErrorGuaranteed),
        };
        (ty, res)
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
            return ctx.lower_ty_relative_path(ty, res, false);
        }

        let mut ctx = self.at_path(path_id);
        let (resolution, remaining_index) = match ctx.resolve_path_in_type_ns() {
            Some(it) => it,
            None => return (Ty::new_error(self.interner, ErrorGuaranteed), None),
        };

        if matches!(resolution, TypeNs::TraitId(_)) && remaining_index.is_none() {
            // trait object type without dyn
            let bound = TypeBound::Path(path_id, TraitBoundModifier::None);
            let ty = self.lower_dyn_trait(&[bound]);
            return (ty, None);
        }

        ctx.lower_partly_resolved_path(resolution, false)
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
        Some((ctx.lower_trait_ref_from_resolved_path(resolved, explicit_self_ty, false), ctx))
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
        generics: &Generics,
        predicate_filter: PredicateFilter,
    ) -> impl Iterator<Item = Clause<'db>> + use<'a, 'b, 'db> {
        match where_predicate {
            WherePredicate::ForLifetime { target, bound, .. }
            | WherePredicate::TypeBound { target, bound } => {
                if let PredicateFilter::SelfTrait = predicate_filter {
                    let target_type = &self.store[*target];
                    let self_type = 'is_self: {
                        if let TypeRef::Path(path) = target_type
                            && path.is_self_type()
                        {
                            break 'is_self true;
                        }
                        if let TypeRef::TypeParam(param) = target_type
                            && generics[param.local_id()].is_trait_self()
                        {
                            break 'is_self true;
                        }
                        false
                    };
                    if !self_type {
                        return Either::Left(Either::Left(iter::empty()));
                    }
                }
                let self_ty = self.lower_ty(*target);
                Either::Left(Either::Right(self.lower_type_bound(bound, self_ty, ignore_bindings)))
            }
            &WherePredicate::Lifetime { bound, target } => {
                Either::Right(iter::once(Clause(Predicate::new(
                    self.interner,
                    Binder::dummy(rustc_type_ir::PredicateKind::Clause(
                        rustc_type_ir::ClauseKind::RegionOutlives(OutlivesPredicate(
                            self.lower_lifetime(bound),
                            self.lower_lifetime(target),
                        )),
                    )),
                ))))
            }
        }
        .into_iter()
    }

    pub(crate) fn lower_type_bound<'b>(
        &'b mut self,
        bound: &'b TypeBound,
        self_ty: Ty<'db>,
        ignore_bindings: bool,
    ) -> impl Iterator<Item = Clause<'db>> + use<'b, 'a, 'db> {
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
                            assoc_bounds = ctx.assoc_type_bindings_from_type_bound(trait_ref);
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
        clause.into_iter().chain(assoc_bounds.into_iter().flatten())
    }

    fn lower_dyn_trait(&mut self, bounds: &[TypeBound]) -> Ty<'db> {
        let interner = self.interner;
        let dummy_self_ty = dyn_trait_dummy_self(interner);
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
                ctx.lower_type_bound(b, dummy_self_ty, false).for_each(|b| {
                    match b.kind().skip_binder() {
                        rustc_type_ir::ClauseKind::Trait(t) => {
                            let id = t.def_id();
                            let is_auto = db.trait_signature(id.0).flags.contains(TraitFlags::AUTO);
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
                    proj.skip_binder().def_id().expect_type_alias(),
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
                                    .map(|item| (item, trait_ref)),
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
                                // FIXME(associated_const_equality): We should walk the const instead of not doing anything
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
                                    pred.skip_binder().projection_term.def_id.expect_type_alias(),
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

            projection_bounds.sort_unstable_by_key(|proj| proj.skip_binder().def_id());

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
                                Ty::new_error(interner, ErrorGuaranteed).into()
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
                        proj.projection_term =
                            replace_dummy_self_with_error(interner, proj.projection_term);
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
            Ty::new_error(self.interner, ErrorGuaranteed)
        }
    }

    fn lower_impl_trait(&mut self, def_id: SolverDefId, bounds: &[TypeBound]) -> ImplTrait<'db> {
        let interner = self.interner;
        cov_mark::hit!(lower_rpit);
        let args = GenericArgs::identity_for_item(interner, def_id);
        let self_ty = Ty::new_alias(
            self.interner,
            rustc_type_ir::AliasTyKind::Opaque,
            AliasTy::new_from_args(interner, def_id, args),
        );
        let predicates = self.with_shifted_in(DebruijnIndex::from_u32(1), |ctx| {
            let mut predicates = Vec::new();
            for b in bounds {
                predicates.extend(ctx.lower_type_bound(b, self_ty, false));
            }

            if !ctx.unsized_types.contains(&self_ty) {
                let sized_trait = self.lang_items.Sized;
                let sized_clause = sized_trait.map(|trait_id| {
                    let trait_ref = TraitRef::new_from_args(
                        interner,
                        trait_id.into(),
                        GenericArgs::new_from_iter(interner, [self_ty.into()]),
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
            predicates.into_boxed_slice()
        });
        ImplTrait { predicates }
    }

    pub(crate) fn lower_lifetime(&mut self, lifetime: LifetimeRefId) -> Region<'db> {
        match self.resolver.resolve_lifetime(&self.store[lifetime]) {
            Some(resolution) => match resolution {
                LifetimeNs::Static => Region::new_static(self.interner),
                LifetimeNs::LifetimeParam(id) => {
                    let idx = match self.generics().lifetime_idx(id) {
                        None => return Region::error(self.interner),
                        Some(idx) => idx,
                    };
                    self.region_param(id, idx as u32)
                }
            },
            None => Region::error(self.interner),
        }
    }
}

fn dyn_trait_dummy_self(interner: DbInterner<'_>) -> Ty<'_> {
    // This type must not appear anywhere except here.
    Ty::new_fresh(interner, 0)
}

fn replace_dummy_self_with_error<'db, T: TypeFoldable<DbInterner<'db>>>(
    interner: DbInterner<'db>,
    t: T,
) -> T {
    let dyn_trait_dummy_self = dyn_trait_dummy_self(interner);
    t.fold_with(&mut BottomUpFolder {
        interner,
        ty_op: |ty| {
            if ty == dyn_trait_dummy_self { Ty::new_error(interner, ErrorGuaranteed) } else { ty }
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

fn unknown_const(_ty: Ty<'_>) -> Const<'_> {
    Const::new(DbInterner::conjure(), ConstKind::Error(ErrorGuaranteed))
}

pub(crate) type Diagnostics = Option<ThinArc<(), TyLoweringDiagnostic>>;

pub(crate) fn create_diagnostics(diagnostics: Vec<TyLoweringDiagnostic>) -> Diagnostics {
    (!diagnostics.is_empty()).then(|| ThinArc::from_header_and_iter((), diagnostics.into_iter()))
}

pub(crate) fn impl_trait_query<'db>(
    db: &'db dyn HirDatabase,
    impl_id: ImplId,
) -> Option<EarlyBinder<'db, TraitRef<'db>>> {
    db.impl_trait_with_diagnostics(impl_id).map(|it| it.0)
}

pub(crate) fn impl_trait_with_diagnostics_query<'db>(
    db: &'db dyn HirDatabase,
    impl_id: ImplId,
) -> Option<(EarlyBinder<'db, TraitRef<'db>>, Diagnostics)> {
    let impl_data = db.impl_signature(impl_id);
    let resolver = impl_id.resolver(db);
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        &impl_data.store,
        impl_id.into(),
        LifetimeElisionKind::AnonymousCreateParameter { report_in_path: true },
    );
    let self_ty = db.impl_self_ty(impl_id).skip_binder();
    let target_trait = impl_data.target_trait.as_ref()?;
    let trait_ref = EarlyBinder::bind(ctx.lower_trait_ref(target_trait, self_ty)?);
    Some((trait_ref, create_diagnostics(ctx.diagnostics)))
}

impl<'db> ImplTraitId<'db> {
    #[inline]
    pub fn predicates(self, db: &'db dyn HirDatabase) -> EarlyBinder<'db, &'db [Clause<'db>]> {
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
            .as_ref()
            .map_bound(|it| &*it.impl_traits[idx].predicates)
    }
}

impl InternedOpaqueTyId {
    #[inline]
    pub fn predicates<'db>(self, db: &'db dyn HirDatabase) -> EarlyBinder<'db, &'db [Clause<'db>]> {
        self.loc(db).predicates(db)
    }
}

#[salsa::tracked]
impl<'db> ImplTraits<'db> {
    #[salsa::tracked(returns(ref), unsafe(non_update_return_type))]
    pub(crate) fn return_type_impl_traits(
        db: &'db dyn HirDatabase,
        def: hir_def::FunctionId,
    ) -> Option<Box<EarlyBinder<'db, ImplTraits<'db>>>> {
        // FIXME unify with fn_sig_for_fn instead of doing lowering twice, maybe
        let data = db.function_signature(def);
        let resolver = def.resolver(db);
        let mut ctx_ret = TyLoweringContext::new(
            db,
            &resolver,
            &data.store,
            def.into(),
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
            Some(Box::new(EarlyBinder::bind(return_type_impl_traits)))
        }
    }

    #[salsa::tracked(returns(ref), unsafe(non_update_return_type))]
    pub(crate) fn type_alias_impl_traits(
        db: &'db dyn HirDatabase,
        def: hir_def::TypeAliasId,
    ) -> Option<Box<EarlyBinder<'db, ImplTraits<'db>>>> {
        let data = db.type_alias_signature(def);
        let resolver = def.resolver(db);
        let mut ctx = TyLoweringContext::new(
            db,
            &resolver,
            &data.store,
            def.into(),
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
            Some(Box::new(EarlyBinder::bind(type_alias_impl_traits)))
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
        TyDefId::TypeAliasId(it) => db.type_for_type_alias_with_diagnostics(it).0,
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

/// Build the declared type of a const.
fn type_for_const<'db>(db: &'db dyn HirDatabase, def: ConstId) -> EarlyBinder<'db, Ty<'db>> {
    let resolver = def.resolver(db);
    let data = db.const_signature(def);
    let parent = def.loc(db).container;
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        &data.store,
        def.into(),
        LifetimeElisionKind::AnonymousReportError,
    );
    ctx.set_lifetime_elision(LifetimeElisionKind::for_const(ctx.interner, parent));
    EarlyBinder::bind(ctx.lower_ty(data.type_ref))
}

/// Build the declared type of a static.
fn type_for_static<'db>(db: &'db dyn HirDatabase, def: StaticId) -> EarlyBinder<'db, Ty<'db>> {
    let resolver = def.resolver(db);
    let data = db.static_signature(def);
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        &data.store,
        def.into(),
        LifetimeElisionKind::AnonymousReportError,
    );
    ctx.set_lifetime_elision(LifetimeElisionKind::Elided(Region::new_static(ctx.interner)));
    EarlyBinder::bind(ctx.lower_ty(data.type_ref))
}

/// Build the type of a tuple struct constructor.
fn type_for_struct_constructor<'db>(
    db: &'db dyn HirDatabase,
    def: StructId,
) -> Option<EarlyBinder<'db, Ty<'db>>> {
    let struct_data = def.fields(db);
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

pub(crate) fn value_ty_query<'db>(
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

pub(crate) fn type_for_type_alias_with_diagnostics_query<'db>(
    db: &'db dyn HirDatabase,
    t: TypeAliasId,
) -> (EarlyBinder<'db, Ty<'db>>, Diagnostics) {
    let type_alias_data = db.type_alias_signature(t);
    let mut diags = None;
    let resolver = t.resolver(db);
    let interner = DbInterner::new_no_crate(db);
    let inner = if type_alias_data.flags.contains(TypeAliasFlags::IS_EXTERN) {
        EarlyBinder::bind(Ty::new_foreign(interner, t.into()))
    } else {
        let mut ctx = TyLoweringContext::new(
            db,
            &resolver,
            &type_alias_data.store,
            t.into(),
            LifetimeElisionKind::AnonymousReportError,
        )
        .with_impl_trait_mode(ImplTraitLoweringMode::Opaque);
        let res = EarlyBinder::bind(
            type_alias_data
                .ty
                .map(|type_ref| ctx.lower_ty(type_ref))
                .unwrap_or_else(|| Ty::new_error(interner, ErrorGuaranteed)),
        );
        diags = create_diagnostics(ctx.diagnostics);
        res
    };
    (inner, diags)
}

pub(crate) fn type_for_type_alias_with_diagnostics_cycle_result<'db>(
    db: &'db dyn HirDatabase,
    _adt: TypeAliasId,
) -> (EarlyBinder<'db, Ty<'db>>, Diagnostics) {
    (EarlyBinder::bind(Ty::new_error(DbInterner::new_no_crate(db), ErrorGuaranteed)), None)
}

pub(crate) fn impl_self_ty_query<'db>(
    db: &'db dyn HirDatabase,
    impl_id: ImplId,
) -> EarlyBinder<'db, Ty<'db>> {
    db.impl_self_ty_with_diagnostics(impl_id).0
}

pub(crate) fn impl_self_ty_with_diagnostics_query<'db>(
    db: &'db dyn HirDatabase,
    impl_id: ImplId,
) -> (EarlyBinder<'db, Ty<'db>>, Diagnostics) {
    let resolver = impl_id.resolver(db);

    let impl_data = db.impl_signature(impl_id);
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        &impl_data.store,
        impl_id.into(),
        LifetimeElisionKind::AnonymousCreateParameter { report_in_path: true },
    );
    let ty = ctx.lower_ty(impl_data.self_ty);
    assert!(!ty.has_escaping_bound_vars());
    (EarlyBinder::bind(ty), create_diagnostics(ctx.diagnostics))
}

pub(crate) fn impl_self_ty_with_diagnostics_cycle_result(
    db: &dyn HirDatabase,
    _impl_id: ImplId,
) -> (EarlyBinder<'_, Ty<'_>>, Diagnostics) {
    (EarlyBinder::bind(Ty::new_error(DbInterner::new_no_crate(db), ErrorGuaranteed)), None)
}

pub(crate) fn const_param_ty_query<'db>(db: &'db dyn HirDatabase, def: ConstParamId) -> Ty<'db> {
    db.const_param_ty_with_diagnostics(def).0
}

// returns None if def is a type arg
pub(crate) fn const_param_ty_with_diagnostics_query<'db>(
    db: &'db dyn HirDatabase,
    def: ConstParamId,
) -> (Ty<'db>, Diagnostics) {
    let (parent_data, store) = db.generic_params_and_store(def.parent());
    let data = &parent_data[def.local_id()];
    let resolver = def.parent().resolver(db);
    let interner = DbInterner::new_no_crate(db);
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        &store,
        def.parent(),
        LifetimeElisionKind::AnonymousReportError,
    );
    let ty = match data {
        TypeOrConstParamData::TypeParamData(_) => {
            never!();
            Ty::new_error(interner, ErrorGuaranteed)
        }
        TypeOrConstParamData::ConstParamData(d) => ctx.lower_ty(d.ty),
    };
    (ty, create_diagnostics(ctx.diagnostics))
}

pub(crate) fn const_param_ty_with_diagnostics_cycle_result<'db>(
    db: &'db dyn HirDatabase,
    _: crate::db::HirDatabaseData,
    _def: ConstParamId,
) -> (Ty<'db>, Diagnostics) {
    let interner = DbInterner::new_no_crate(db);
    (Ty::new_error(interner, ErrorGuaranteed), None)
}

pub(crate) fn field_types_query<'db>(
    db: &'db dyn HirDatabase,
    variant_id: VariantId,
) -> Arc<ArenaMap<LocalFieldId, EarlyBinder<'db, Ty<'db>>>> {
    db.field_types_with_diagnostics(variant_id).0
}

/// Build the type of all specific fields of a struct or enum variant.
pub(crate) fn field_types_with_diagnostics_query<'db>(
    db: &'db dyn HirDatabase,
    variant_id: VariantId,
) -> (Arc<ArenaMap<LocalFieldId, EarlyBinder<'db, Ty<'db>>>>, Diagnostics) {
    let var_data = variant_id.fields(db);
    let fields = var_data.fields();
    if fields.is_empty() {
        return (Arc::new(ArenaMap::default()), None);
    }

    let (resolver, def): (_, GenericDefId) = match variant_id {
        VariantId::StructId(it) => (it.resolver(db), it.into()),
        VariantId::UnionId(it) => (it.resolver(db), it.into()),
        VariantId::EnumVariantId(it) => (it.resolver(db), it.lookup(db).parent.into()),
    };
    let mut res = ArenaMap::default();
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        &var_data.store,
        def,
        LifetimeElisionKind::AnonymousReportError,
    );
    for (field_id, field_data) in var_data.fields().iter() {
        res.insert(field_id, EarlyBinder::bind(ctx.lower_ty(field_data.type_ref)));
    }
    (Arc::new(res), create_diagnostics(ctx.diagnostics))
}

/// This query exists only to be used when resolving short-hand associated types
/// like `T::Item`.
///
/// See the analogous query in rustc and its comment:
/// <https://github.com/rust-lang/rust/blob/9150f844e2624eb013ec78ca08c1d416e6644026/src/librustc_typeck/astconv.rs#L46>
/// This is a query mostly to handle cycles somewhat gracefully; e.g. the
/// following bounds are disallowed: `T: Foo<U::Item>, U: Foo<T::Item>`, but
/// these are fine: `T: Foo<U::Item>, U: Foo<()>`.
#[tracing::instrument(skip(db), ret)]
#[salsa::tracked(returns(ref), unsafe(non_update_return_type), cycle_result = generic_predicates_for_param_cycle_result)]
pub(crate) fn generic_predicates_for_param<'db>(
    db: &'db dyn HirDatabase,
    def: GenericDefId,
    param_id: TypeOrConstParamId,
    assoc_name: Option<Name>,
) -> EarlyBinder<'db, Box<[Clause<'db>]>> {
    let generics = generics(db, def);
    let interner = DbInterner::new_no_crate(db);
    let resolver = def.resolver(db);
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        generics.store(),
        def,
        LifetimeElisionKind::AnonymousReportError,
    );

    // we have to filter out all other predicates *first*, before attempting to lower them
    let predicate = |pred: &_, ctx: &mut TyLoweringContext<'_, '_>| match pred {
        WherePredicate::ForLifetime { target, bound, .. }
        | WherePredicate::TypeBound { target, bound, .. } => {
            let invalid_target = { ctx.lower_ty_only_param(*target) != Some(param_id) };
            if invalid_target {
                // FIXME(sized-hierarchy): Revisit and adjust this properly once we have implemented
                // sized-hierarchy correctly.
                // If this is filtered out without lowering, `?Sized` or `PointeeSized` is not gathered into
                // `ctx.unsized_types`
                let lower = || -> bool {
                    match bound {
                        TypeBound::Path(_, TraitBoundModifier::Maybe) => true,
                        TypeBound::Path(path, _) | TypeBound::ForLifetime(_, path) => {
                            let TypeRef::Path(path) = &ctx.store[path.type_ref()] else {
                                return false;
                            };
                            let Some(pointee_sized) = ctx.lang_items.PointeeSized else {
                                return false;
                            };
                            // Lower the path directly with `Resolver` instead of PathLoweringContext`
                            // to prevent diagnostics duplications.
                            ctx.resolver.resolve_path_in_type_ns_fully(ctx.db, path).is_some_and(
                                |it| matches!(it, TypeNs::TraitId(tr) if tr == pointee_sized),
                            )
                        }
                        _ => false,
                    }
                }();
                if lower {
                    ctx.lower_where_predicate(pred, true, &generics, PredicateFilter::All)
                        .for_each(drop);
                }
                return false;
            }

            match bound {
                &TypeBound::ForLifetime(_, path) | &TypeBound::Path(path, _) => {
                    // Only lower the bound if the trait could possibly define the associated
                    // type we're looking for.
                    let path = &ctx.store[path];

                    let Some(assoc_name) = &assoc_name else { return true };
                    let Some(TypeNs::TraitId(tr)) =
                        resolver.resolve_path_in_type_ns_fully(db, path)
                    else {
                        return false;
                    };

                    rustc_type_ir::elaborate::supertrait_def_ids(interner, tr.into()).any(|tr| {
                        tr.0.trait_items(db).items.iter().any(|(name, item)| {
                            matches!(item, AssocItemId::TypeAliasId(_)) && name == assoc_name
                        })
                    })
                }
                TypeBound::Use(_) | TypeBound::Lifetime(_) | TypeBound::Error => false,
            }
        }
        WherePredicate::Lifetime { .. } => false,
    };
    let mut predicates = Vec::new();
    for maybe_parent_generics in
        std::iter::successors(Some(&generics), |generics| generics.parent_generics())
    {
        ctx.store = maybe_parent_generics.store();
        for pred in maybe_parent_generics.where_predicates() {
            if predicate(pred, &mut ctx) {
                predicates.extend(ctx.lower_where_predicate(
                    pred,
                    true,
                    maybe_parent_generics,
                    PredicateFilter::All,
                ));
            }
        }
    }

    let args = GenericArgs::identity_for_item(interner, def.into());
    if !args.is_empty() {
        let explicitly_unsized_tys = ctx.unsized_types;
        if let Some(implicitly_sized_predicates) = implicitly_sized_clauses(
            db,
            ctx.lang_items,
            param_id.parent,
            &explicitly_unsized_tys,
            &args,
        ) {
            predicates.extend(implicitly_sized_predicates);
        };
    }
    EarlyBinder::bind(predicates.into_boxed_slice())
}

pub(crate) fn generic_predicates_for_param_cycle_result<'db>(
    _db: &'db dyn HirDatabase,
    _def: GenericDefId,
    _param_id: TypeOrConstParamId,
    _assoc_name: Option<Name>,
) -> EarlyBinder<'db, Box<[Clause<'db>]>> {
    EarlyBinder::bind(Box::new([]))
}

#[inline]
pub(crate) fn type_alias_bounds<'db>(
    db: &'db dyn HirDatabase,
    type_alias: TypeAliasId,
) -> EarlyBinder<'db, &'db [Clause<'db>]> {
    type_alias_bounds_with_diagnostics(db, type_alias).0.as_ref().map_bound(|it| &**it)
}

#[salsa::tracked(returns(ref), unsafe(non_update_return_type))]
pub fn type_alias_bounds_with_diagnostics<'db>(
    db: &'db dyn HirDatabase,
    type_alias: TypeAliasId,
) -> (EarlyBinder<'db, Box<[Clause<'db>]>>, Diagnostics) {
    let type_alias_data = db.type_alias_signature(type_alias);
    let resolver = hir_def::resolver::HasResolver::resolver(type_alias, db);
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        &type_alias_data.store,
        type_alias.into(),
        LifetimeElisionKind::AnonymousReportError,
    );
    let interner = ctx.interner;
    let def_id = type_alias.into();

    let item_args = GenericArgs::identity_for_item(interner, def_id);
    let interner_ty = Ty::new_projection_from_args(interner, def_id, item_args);

    let mut bounds = Vec::new();
    for bound in &type_alias_data.bounds {
        ctx.lower_type_bound(bound, interner_ty, false).for_each(|pred| {
            bounds.push(pred);
        });
    }

    if !ctx.unsized_types.contains(&interner_ty) {
        let sized_trait = ctx.lang_items.Sized;
        if let Some(sized_trait) = sized_trait {
            let trait_ref = TraitRef::new_from_args(
                interner,
                sized_trait.into(),
                GenericArgs::new_from_iter(interner, [interner_ty.into()]),
            );
            bounds.push(trait_ref.upcast(interner));
        };
    }

    (EarlyBinder::bind(bounds.into_boxed_slice()), create_diagnostics(ctx.diagnostics))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GenericPredicates<'db> {
    // The order is the following: first, if `parent_is_trait == true`, comes the implicit trait predicate for the
    // parent. Then come the explicit predicates for the parent, then the explicit trait predicate for the child,
    // then the implicit trait predicate for the child, if `is_trait` is `true`.
    predicates: EarlyBinder<'db, Box<[Clause<'db>]>>,
    own_predicates_start: u32,
    is_trait: bool,
    parent_is_trait: bool,
}

#[salsa::tracked]
impl<'db> GenericPredicates<'db> {
    /// Resolve the where clause(s) of an item with generics.
    ///
    /// Diagnostics are computed only for this item's predicates, not for parents.
    #[salsa::tracked(returns(ref), unsafe(non_update_return_type))]
    pub fn query_with_diagnostics(
        db: &'db dyn HirDatabase,
        def: GenericDefId,
    ) -> (GenericPredicates<'db>, Diagnostics) {
        generic_predicates_filtered_by(db, def, PredicateFilter::All, |_| true)
    }
}

impl<'db> GenericPredicates<'db> {
    #[inline]
    pub fn query(db: &'db dyn HirDatabase, def: GenericDefId) -> &'db GenericPredicates<'db> {
        &Self::query_with_diagnostics(db, def).0
    }

    #[inline]
    pub fn query_all(
        db: &'db dyn HirDatabase,
        def: GenericDefId,
    ) -> EarlyBinder<'db, &'db [Clause<'db>]> {
        Self::query(db, def).all_predicates()
    }

    #[inline]
    pub fn query_own(
        db: &'db dyn HirDatabase,
        def: GenericDefId,
    ) -> EarlyBinder<'db, &'db [Clause<'db>]> {
        Self::query(db, def).own_predicates()
    }

    #[inline]
    pub fn query_explicit(
        db: &'db dyn HirDatabase,
        def: GenericDefId,
    ) -> EarlyBinder<'db, &'db [Clause<'db>]> {
        Self::query(db, def).explicit_predicates()
    }

    #[inline]
    pub fn all_predicates(&self) -> EarlyBinder<'db, &[Clause<'db>]> {
        self.predicates.as_ref().map_bound(|it| &**it)
    }

    #[inline]
    pub fn own_predicates(&self) -> EarlyBinder<'db, &[Clause<'db>]> {
        self.predicates.as_ref().map_bound(|it| &it[self.own_predicates_start as usize..])
    }

    /// Returns the predicates, minus the implicit `Self: Trait` predicate for a trait.
    #[inline]
    pub fn explicit_predicates(&self) -> EarlyBinder<'db, &[Clause<'db>]> {
        self.predicates.as_ref().map_bound(|it| {
            &it[usize::from(self.parent_is_trait)..it.len() - usize::from(self.is_trait)]
        })
    }
}

pub(crate) fn trait_environment_for_body_query(
    db: &dyn HirDatabase,
    def: DefWithBodyId,
) -> ParamEnv<'_> {
    let Some(def) = def.as_generic_def_id(db) else {
        return ParamEnv::empty();
    };
    db.trait_environment(def)
}

pub(crate) fn trait_environment_query<'db>(
    db: &'db dyn HirDatabase,
    def: GenericDefId,
) -> ParamEnv<'db> {
    let module = def.module(db);
    let interner = DbInterner::new_with(db, module.krate(db));
    let predicates = GenericPredicates::query_all(db, def);
    let clauses = rustc_type_ir::elaborate::elaborate(interner, predicates.iter_identity_copied());
    let clauses = Clauses::new_from_iter(interner, clauses);

    // FIXME: We should normalize projections here, like rustc does.
    ParamEnv { clauses }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum PredicateFilter {
    SelfTrait,
    All,
}

/// Resolve the where clause(s) of an item with generics,
/// with a given filter
#[tracing::instrument(skip(db, filter), ret)]
pub(crate) fn generic_predicates_filtered_by<'db, F>(
    db: &'db dyn HirDatabase,
    def: GenericDefId,
    predicate_filter: PredicateFilter,
    filter: F,
) -> (GenericPredicates<'db>, Diagnostics)
where
    F: Fn(GenericDefId) -> bool,
{
    let generics = generics(db, def);
    let resolver = def.resolver(db);
    let interner = DbInterner::new_no_crate(db);
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        generics.store(),
        def,
        LifetimeElisionKind::AnonymousReportError,
    );
    let sized_trait = ctx.lang_items.Sized;

    let mut predicates = Vec::new();
    let all_generics =
        std::iter::successors(Some(&generics), |generics| generics.parent_generics())
            .collect::<ArrayVec<_, 2>>();
    let mut is_trait = false;
    let mut parent_is_trait = false;
    if all_generics.len() > 1 {
        add_implicit_trait_predicate(
            interner,
            all_generics.last().unwrap().def(),
            predicate_filter,
            &mut predicates,
            &mut parent_is_trait,
        );
    }
    // We need to lower parent predicates first - see the comment below lowering of implicit `Sized` predicates
    // for why.
    let mut own_predicates_start = 0;
    for &maybe_parent_generics in all_generics.iter().rev() {
        let current_def_predicates_start = predicates.len();
        // Collect only diagnostics from the child, not including parents.
        ctx.diagnostics.clear();

        if filter(maybe_parent_generics.def()) {
            ctx.store = maybe_parent_generics.store();
            for pred in maybe_parent_generics.where_predicates() {
                tracing::debug!(?pred);
                predicates.extend(ctx.lower_where_predicate(
                    pred,
                    false,
                    maybe_parent_generics,
                    predicate_filter,
                ));
            }

            push_const_arg_has_type_predicates(db, &mut predicates, maybe_parent_generics);

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
                        GenericArgs::new_from_iter(interner, [param_ty.into()]),
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
                    predicates.push(clause);
                };
                let parent_params_len = maybe_parent_generics.len_parent();
                maybe_parent_generics.iter_self().enumerate().for_each(
                    |(param_idx, (param_id, param_data))| {
                        add_sized_clause(
                            (param_idx + parent_params_len) as u32,
                            param_id,
                            param_data,
                        );
                    },
                );
            }

            // We do not clear `ctx.unsized_types`, as the `?Sized` clause of a child (e.g. an associated type) can
            // be declared on the parent (e.g. the trait). It is nevertheless fine to register the implicit `Sized`
            // predicates before lowering the child, as a child cannot define a `?Sized` predicate for its parent.
            // But we do have to lower the parent first.
        }

        if maybe_parent_generics.def() == def {
            own_predicates_start = current_def_predicates_start as u32;
        }
    }

    add_implicit_trait_predicate(interner, def, predicate_filter, &mut predicates, &mut is_trait);

    let diagnostics = create_diagnostics(ctx.diagnostics);
    let predicates = GenericPredicates {
        own_predicates_start,
        is_trait,
        parent_is_trait,
        predicates: EarlyBinder::bind(predicates.into_boxed_slice()),
    };
    return (predicates, diagnostics);

    fn add_implicit_trait_predicate<'db>(
        interner: DbInterner<'db>,
        def: GenericDefId,
        predicate_filter: PredicateFilter,
        predicates: &mut Vec<Clause<'db>>,
        set_is_trait: &mut bool,
    ) {
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
        if let GenericDefId::TraitId(def_id) = def
            && predicate_filter == PredicateFilter::All
        {
            *set_is_trait = true;
            predicates.push(TraitRef::identity(interner, def_id.into()).upcast(interner));
        }
    }
}

fn push_const_arg_has_type_predicates<'db>(
    db: &'db dyn HirDatabase,
    predicates: &mut Vec<Clause<'db>>,
    generics: &Generics,
) {
    let interner = DbInterner::new_no_crate(db);
    let const_params_offset = generics.len_parent() + generics.len_lifetimes_self();
    for (param_index, (param_idx, param_data)) in generics.iter_self_type_or_consts().enumerate() {
        if !matches!(param_data, TypeOrConstParamData::ConstParamData(_)) {
            continue;
        }

        let param_id = ConstParamId::from_unchecked(TypeOrConstParamId {
            parent: generics.def(),
            local_id: param_idx,
        });
        predicates.push(Clause(
            ClauseKind::ConstArgHasType(
                Const::new_param(
                    interner,
                    ParamConst { id: param_id, index: (param_index + const_params_offset) as u32 },
                ),
                db.const_param_ty_ns(param_id),
            )
            .upcast(interner),
        ));
    }
}

/// Generate implicit `: Sized` predicates for all generics that has no `?Sized` bound.
/// Exception is Self of a trait def.
fn implicitly_sized_clauses<'a, 'subst, 'db>(
    db: &'db dyn HirDatabase,
    lang_items: &LangItems,
    def: GenericDefId,
    explicitly_unsized_tys: &'a FxHashSet<Ty<'db>>,
    args: &'subst GenericArgs<'db>,
) -> Option<impl Iterator<Item = Clause<'db>> + Captures<'a> + Captures<'subst>> {
    let interner = DbInterner::new_no_crate(db);
    let sized_trait = lang_items.Sized?;

    let trait_self_idx = trait_self_param_idx(db, def);

    Some(
        args.iter()
            .enumerate()
            .filter_map(
                move |(idx, generic_arg)| {
                    if Some(idx) == trait_self_idx { None } else { Some(generic_arg) }
                },
            )
            .filter_map(|generic_arg| generic_arg.as_type())
            .filter(move |self_ty| !explicitly_unsized_tys.contains(self_ty))
            .map(move |self_ty| {
                let trait_ref = TraitRef::new_from_args(
                    interner,
                    sized_trait.into(),
                    GenericArgs::new_from_iter(interner, [self_ty.into()]),
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
            }),
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GenericDefaults<'db>(Option<Arc<[Option<EarlyBinder<'db, GenericArg<'db>>>]>>);

impl<'db> GenericDefaults<'db> {
    #[inline]
    pub fn get(&self, idx: usize) -> Option<EarlyBinder<'db, GenericArg<'db>>> {
        self.0.as_ref()?[idx]
    }
}

pub(crate) fn generic_defaults_query(
    db: &dyn HirDatabase,
    def: GenericDefId,
) -> GenericDefaults<'_> {
    db.generic_defaults_with_diagnostics(def).0
}

/// Resolve the default type params from generics.
///
/// Diagnostics are only returned for this `GenericDefId` (returned defaults include parents).
pub(crate) fn generic_defaults_with_diagnostics_query(
    db: &dyn HirDatabase,
    def: GenericDefId,
) -> (GenericDefaults<'_>, Diagnostics) {
    let generic_params = generics(db, def);
    if generic_params.is_empty() {
        return (GenericDefaults(None), None);
    }
    let resolver = def.resolver(db);

    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        generic_params.store(),
        def,
        LifetimeElisionKind::AnonymousReportError,
    )
    .with_impl_trait_mode(ImplTraitLoweringMode::Disallowed);
    let mut idx = 0;
    let mut has_any_default = false;
    let mut defaults = generic_params
        .iter_parents_with_store()
        .map(|((_id, p), store)| {
            ctx.store = store;
            let (result, has_default) = handle_generic_param(&mut ctx, idx, p);
            has_any_default |= has_default;
            idx += 1;
            result
        })
        .collect::<Vec<_>>();
    ctx.diagnostics.clear(); // Don't include diagnostics from the parent.
    defaults.extend(generic_params.iter_self().map(|(_id, p)| {
        let (result, has_default) = handle_generic_param(&mut ctx, idx, p);
        has_any_default |= has_default;
        idx += 1;
        result
    }));
    let diagnostics = create_diagnostics(mem::take(&mut ctx.diagnostics));
    let defaults = if has_any_default {
        GenericDefaults(Some(Arc::from_iter(defaults)))
    } else {
        GenericDefaults(None)
    };
    return (defaults, diagnostics);

    fn handle_generic_param<'db>(
        ctx: &mut TyLoweringContext<'db, '_>,
        idx: usize,
        p: GenericParamDataRef<'_>,
    ) -> (Option<EarlyBinder<'db, GenericArg<'db>>>, bool) {
        ctx.lowering_param_default(idx as u32);
        match p {
            GenericParamDataRef::TypeParamData(p) => {
                let ty = p.default.map(|ty| ctx.lower_ty(ty));
                (ty.map(|ty| EarlyBinder::bind(ty.into())), p.default.is_some())
            }
            GenericParamDataRef::ConstParamData(p) => {
                let val = p.default.map(|c| {
                    let param_ty = ctx.lower_ty(p.ty);
                    let c = ctx.lower_const(c, param_ty);
                    c.into()
                });
                (val.map(EarlyBinder::bind), p.default.is_some())
            }
            GenericParamDataRef::LifetimeParamData(_) => (None, false),
        }
    }
}

pub(crate) fn generic_defaults_with_diagnostics_cycle_result(
    _db: &dyn HirDatabase,
    _def: GenericDefId,
) -> (GenericDefaults<'_>, Diagnostics) {
    (GenericDefaults(None), None)
}

/// Build the signature of a callable item (function, struct or enum variant).
pub(crate) fn callable_item_signature_query<'db>(
    db: &'db dyn HirDatabase,
    def: CallableDefId,
) -> EarlyBinder<'db, PolyFnSig<'db>> {
    match def {
        CallableDefId::FunctionId(f) => fn_sig_for_fn(db, f),
        CallableDefId::StructId(s) => fn_sig_for_struct_constructor(db, s),
        CallableDefId::EnumVariantId(e) => fn_sig_for_enum_variant_constructor(db, e),
    }
}

fn fn_sig_for_fn<'db>(
    db: &'db dyn HirDatabase,
    def: FunctionId,
) -> EarlyBinder<'db, PolyFnSig<'db>> {
    let data = db.function_signature(def);
    let resolver = def.resolver(db);
    let interner = DbInterner::new_no_crate(db);
    let mut ctx_params = TyLoweringContext::new(
        db,
        &resolver,
        &data.store,
        def.into(),
        LifetimeElisionKind::for_fn_params(&data),
    );
    let params = data.params.iter().map(|&tr| ctx_params.lower_ty(tr));

    let ret = match data.ret_type {
        Some(ret_type) => {
            let mut ctx_ret = TyLoweringContext::new(
                db,
                &resolver,
                &data.store,
                def.into(),
                LifetimeElisionKind::for_fn_ret(interner),
            )
            .with_impl_trait_mode(ImplTraitLoweringMode::Opaque);
            ctx_ret.lower_ty(ret_type)
        }
        None => Ty::new_tup(interner, &[]),
    };

    let inputs_and_output = Tys::new_from_iter(interner, params.chain(Some(ret)));
    // If/when we track late bound vars, we need to switch this to not be `dummy`
    EarlyBinder::bind(rustc_type_ir::Binder::dummy(FnSig {
        abi: data.abi.as_ref().map_or(FnAbi::Rust, FnAbi::from_symbol),
        c_variadic: data.is_varargs(),
        safety: if data.is_unsafe() { Safety::Unsafe } else { Safety::Safe },
        inputs_and_output,
    }))
}

fn type_for_adt<'db>(db: &'db dyn HirDatabase, adt: AdtId) -> EarlyBinder<'db, Ty<'db>> {
    let interner = DbInterner::new_no_crate(db);
    let args = GenericArgs::identity_for_item(interner, adt.into());
    let ty = Ty::new_adt(interner, adt, args);
    EarlyBinder::bind(ty)
}

fn fn_sig_for_struct_constructor<'db>(
    db: &'db dyn HirDatabase,
    def: StructId,
) -> EarlyBinder<'db, PolyFnSig<'db>> {
    let field_tys = db.field_types(def.into());
    let params = field_tys.iter().map(|(_, ty)| ty.skip_binder());
    let ret = type_for_adt(db, def.into()).skip_binder();

    let inputs_and_output =
        Tys::new_from_iter(DbInterner::new_no_crate(db), params.chain(Some(ret)));
    EarlyBinder::bind(Binder::dummy(FnSig {
        abi: FnAbi::RustCall,
        c_variadic: false,
        safety: Safety::Safe,
        inputs_and_output,
    }))
}

fn fn_sig_for_enum_variant_constructor<'db>(
    db: &'db dyn HirDatabase,
    def: EnumVariantId,
) -> EarlyBinder<'db, PolyFnSig<'db>> {
    let field_tys = db.field_types(def.into());
    let params = field_tys.iter().map(|(_, ty)| ty.skip_binder());
    let parent = def.lookup(db).parent;
    let ret = type_for_adt(db, parent.into()).skip_binder();

    let inputs_and_output =
        Tys::new_from_iter(DbInterner::new_no_crate(db), params.chain(Some(ret)));
    EarlyBinder::bind(Binder::dummy(FnSig {
        abi: FnAbi::RustCall,
        c_variadic: false,
        safety: Safety::Safe,
        inputs_and_output,
    }))
}

// FIXME(next-solver): should merge this with `explicit_item_bounds` in some way
pub(crate) fn associated_ty_item_bounds<'db>(
    db: &'db dyn HirDatabase,
    type_alias: TypeAliasId,
) -> EarlyBinder<'db, BoundExistentialPredicates<'db>> {
    let type_alias_data = db.type_alias_signature(type_alias);
    let resolver = hir_def::resolver::HasResolver::resolver(type_alias, db);
    let interner = DbInterner::new_no_crate(db);
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        &type_alias_data.store,
        type_alias.into(),
        LifetimeElisionKind::AnonymousReportError,
    );
    // FIXME: we should never create non-existential predicates in the first place
    // For now, use an error type so we don't run into dummy binder issues
    let self_ty = Ty::new_error(interner, ErrorGuaranteed);

    let mut bounds = Vec::new();
    for bound in &type_alias_data.bounds {
        ctx.lower_type_bound(bound, self_ty, false).for_each(|pred| {
            if let Some(bound) = pred
                .kind()
                .map_bound(|c| match c {
                    rustc_type_ir::ClauseKind::Trait(t) => {
                        let id = t.def_id();
                        let is_auto = db.trait_signature(id.0).flags.contains(TraitFlags::AUTO);
                        if is_auto {
                            Some(ExistentialPredicate::AutoTrait(t.def_id()))
                        } else {
                            Some(ExistentialPredicate::Trait(ExistentialTraitRef::new_from_args(
                                interner,
                                t.def_id(),
                                GenericArgs::new_from_iter(
                                    interner,
                                    t.trait_ref.args.iter().skip(1),
                                ),
                            )))
                        }
                    }
                    rustc_type_ir::ClauseKind::Projection(p) => Some(
                        ExistentialPredicate::Projection(ExistentialProjection::new_from_args(
                            interner,
                            p.def_id(),
                            GenericArgs::new_from_iter(
                                interner,
                                p.projection_term.args.iter().skip(1),
                            ),
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

    EarlyBinder::bind(BoundExistentialPredicates::new_from_iter(interner, bounds))
}

pub(crate) fn associated_type_by_name_including_super_traits<'db>(
    db: &'db dyn HirDatabase,
    trait_ref: TraitRef<'db>,
    name: &Name,
) -> Option<(TraitRef<'db>, TypeAliasId)> {
    let module = trait_ref.def_id.0.module(db);
    let interner = DbInterner::new_with(db, module.krate(db));
    rustc_type_ir::elaborate::supertraits(interner, Binder::dummy(trait_ref)).find_map(|t| {
        let trait_id = t.as_ref().skip_binder().def_id.0;
        let assoc_type = trait_id.trait_items(db).associated_type_by_name(name)?;
        Some((t.skip_binder(), assoc_type))
    })
}

pub fn associated_type_shorthand_candidates(
    db: &dyn HirDatabase,
    def: GenericDefId,
    res: TypeNs,
    mut cb: impl FnMut(&Name, TypeAliasId) -> bool,
) -> Option<TypeAliasId> {
    let interner = DbInterner::new_no_crate(db);
    named_associated_type_shorthand_candidates(interner, def, res, None, |name, _, id| {
        cb(name, id).then_some(id)
    })
}

#[tracing::instrument(skip(interner, check_alias))]
fn named_associated_type_shorthand_candidates<'db, R>(
    interner: DbInterner<'db>,
    // If the type parameter is defined in an impl and we're in a method, there
    // might be additional where clauses to consider
    def: GenericDefId,
    res: TypeNs,
    assoc_name: Option<Name>,
    mut check_alias: impl FnMut(&Name, TraitRef<'db>, TypeAliasId) -> Option<R>,
) -> Option<R> {
    let db = interner.db;
    let mut search = |t: TraitRef<'db>| -> Option<R> {
        let mut checked_traits = FxHashSet::default();
        let mut check_trait = |trait_ref: TraitRef<'db>| {
            let trait_id = trait_ref.def_id.0;
            let name = &db.trait_signature(trait_id).name;
            tracing::debug!(?trait_id, ?name);
            if !checked_traits.insert(trait_id) {
                return None;
            }
            let data = trait_id.trait_items(db);

            tracing::debug!(?data.items);
            for (name, assoc_id) in &data.items {
                if let &AssocItemId::TypeAliasId(alias) = assoc_id
                    && let Some(ty) = check_alias(name, trait_ref, alias)
                {
                    return Some(ty);
                }
            }
            None
        };
        let mut stack: SmallVec<[_; 4]> = smallvec![t];
        while let Some(trait_ref) = stack.pop() {
            if let Some(alias) = check_trait(trait_ref) {
                return Some(alias);
            }
            for pred in generic_predicates_filtered_by(
                db,
                GenericDefId::TraitId(trait_ref.def_id.0),
                PredicateFilter::SelfTrait,
                // We are likely in the midst of lowering generic predicates of `def`.
                // So, if we allow `pred == def` we might fall into an infinite recursion.
                // Actually, we have already checked for the case `pred == def` above as we started
                // with a stack including `trait_id`
                |pred| pred != def && pred == GenericDefId::TraitId(trait_ref.def_id.0),
            )
            .0
            .predicates
            .instantiate_identity()
            {
                tracing::debug!(?pred);
                let sup_trait_ref = match pred.kind().skip_binder() {
                    rustc_type_ir::ClauseKind::Trait(pred) => pred.trait_ref,
                    _ => continue,
                };
                let sup_trait_ref =
                    EarlyBinder::bind(sup_trait_ref).instantiate(interner, trait_ref.args);
                stack.push(sup_trait_ref);
            }
            tracing::debug!(?stack);
        }

        None
    };

    match res {
        TypeNs::SelfType(impl_id) => {
            let trait_ref = db.impl_trait(impl_id)?;

            // FIXME(next-solver): same method in `lower` checks for impl or not
            // Is that needed here?

            // we're _in_ the impl -- the binders get added back later. Correct,
            // but it would be nice to make this more explicit
            search(trait_ref.skip_binder())
        }
        TypeNs::GenericParam(param_id) => {
            // Handle `Self::Type` referring to own associated type in trait definitions
            // This *must* be done first to avoid cycles with
            // `generic_predicates_for_param`, but not sure that it's sufficient,
            if let GenericDefId::TraitId(trait_id) = param_id.parent() {
                let trait_name = &db.trait_signature(trait_id).name;
                tracing::debug!(?trait_name);
                let trait_generics = generics(db, trait_id.into());
                tracing::debug!(?trait_generics);
                if trait_generics[param_id.local_id()].is_trait_self() {
                    let args = GenericArgs::identity_for_item(interner, trait_id.into());
                    let trait_ref = TraitRef::new_from_args(interner, trait_id.into(), args);
                    tracing::debug!(?args, ?trait_ref);
                    return search(trait_ref);
                }
            }

            let predicates =
                generic_predicates_for_param(db, def, param_id.into(), assoc_name.clone());
            predicates
                .as_ref()
                .iter_identity_copied()
                .find_map(|pred| match pred.kind().skip_binder() {
                    rustc_type_ir::ClauseKind::Trait(trait_predicate) => Some(trait_predicate),
                    _ => None,
                })
                .and_then(|trait_predicate| {
                    let trait_ref = trait_predicate.trait_ref;
                    assert!(
                        !trait_ref.has_escaping_bound_vars(),
                        "FIXME unexpected higher-ranked trait bound"
                    );
                    search(trait_ref)
                })
        }
        _ => None,
    }
}
