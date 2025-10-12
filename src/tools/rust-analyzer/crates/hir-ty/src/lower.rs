//! Methods for lowering the HIR to types. There are two main cases here:
//!
//!  - Lowering a type reference like `&usize` or `Option<foo::bar::Baz>` to a
//!    type: The entry point for this is `TyLoweringContext::lower_ty`.
//!  - Building the type for an item: This happens through the `ty` query.
//!
//! This usually involves resolving names, collecting generic arguments etc.
pub(crate) mod diagnostics;
pub(crate) mod path;

use std::{
    cell::OnceCell,
    iter, mem,
    ops::{self, Not as _},
};

use base_db::Crate;
use chalk_ir::{
    Mutability, Safety, TypeOutlives,
    cast::Cast,
    fold::{Shift, TypeFoldable},
    interner::HasInterner,
};

use either::Either;
use hir_def::{
    AdtId, AssocItemId, ConstId, ConstParamId, EnumId, EnumVariantId, FunctionId, GenericDefId,
    GenericParamId, LocalFieldId, Lookup, StaticId, StructId, TypeAliasId, TypeOrConstParamId,
    UnionId, VariantId,
    builtin_type::BuiltinType,
    expr_store::{ExpressionStore, path::Path},
    hir::generics::{GenericParamDataRef, TypeOrConstParamData, WherePredicate},
    lang_item::LangItem,
    resolver::{HasResolver, LifetimeNs, Resolver, TypeNs},
    signatures::TraitFlags,
    type_ref::{
        ConstRef, LifetimeRefId, LiteralConstRef, PathId, TraitBoundModifier, TypeBound, TypeRef,
        TypeRefId,
    },
};
use hir_expand::name::Name;
use la_arena::{Arena, ArenaMap};
use rustc_hash::FxHashSet;
use stdx::{impl_from, never};
use triomphe::{Arc, ThinArc};

use crate::{
    AliasTy, Binders, BoundVar, Const, DebruijnIndex, DynTy, FnAbi, FnPointer, FnSig, FnSubst,
    ImplTrait, ImplTraitId, ImplTraits, Interner, Lifetime, LifetimeData, LifetimeOutlives,
    QuantifiedWhereClause, QuantifiedWhereClauses, Substitution, TraitRef, TraitRefExt, Ty,
    TyBuilder, TyKind, WhereClause, all_super_traits,
    consteval_chalk::{intern_const_ref, path_to_const, unknown_const, unknown_const_as_generic},
    db::HirDatabase,
    error_lifetime,
    generics::{Generics, generics, trait_self_param_idx},
    lower::{
        diagnostics::*,
        path::{PathDiagnosticCallback, PathLoweringContext},
    },
    make_binders,
    mapping::{from_chalk_trait_id, lt_to_placeholder_idx},
    next_solver::{
        DbInterner,
        mapping::{ChalkToNextSolver, NextSolverToChalk},
    },
    static_lifetime, to_chalk_trait_id, to_placeholder_idx,
    utils::all_super_trait_refs,
    variable_kinds_from_iter,
};

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

pub(crate) struct PathDiagnosticCallbackData(pub(crate) TypeRefId);

#[derive(Debug, Clone)]
pub(crate) enum LifetimeElisionKind {
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
    Elided(Lifetime),

    /// Give a hard error when either `&` or `'_` is written. Used to
    /// rule out things like `where T: Foo<'_>`. Does not imply an
    /// error on default object bounds (e.g., `Box<dyn Foo>`).
    AnonymousReportError,

    /// Infer all elided lifetimes.
    Infer,
}

impl LifetimeElisionKind {
    #[inline]
    pub(crate) fn for_fn_ret() -> LifetimeElisionKind {
        // FIXME: We should use the elided lifetime here, or `ElisionFailure`.
        LifetimeElisionKind::Elided(error_lifetime())
    }
}

#[derive(Debug)]
pub(crate) struct TyLoweringContext<'db> {
    pub db: &'db dyn HirDatabase,
    resolver: &'db Resolver<'db>,
    store: &'db ExpressionStore,
    def: GenericDefId,
    generics: OnceCell<Generics>,
    in_binders: DebruijnIndex,
    /// Note: Conceptually, it's thinkable that we could be in a location where
    /// some type params should be represented as placeholders, and others
    /// should be converted to variables. I think in practice, this isn't
    /// possible currently, so this should be fine for now.
    pub type_param_mode: ParamLoweringMode,
    impl_trait_mode: ImplTraitLoweringState,
    /// Tracks types with explicit `?Sized` bounds.
    pub(crate) unsized_types: FxHashSet<Ty>,
    pub(crate) diagnostics: Vec<TyLoweringDiagnostic>,
    lifetime_elision: LifetimeElisionKind,
}

impl<'db> TyLoweringContext<'db> {
    pub(crate) fn new(
        db: &'db dyn HirDatabase,
        resolver: &'db Resolver<'db>,
        store: &'db ExpressionStore,
        def: GenericDefId,
        lifetime_elision: LifetimeElisionKind,
    ) -> Self {
        let impl_trait_mode = ImplTraitLoweringState::new(ImplTraitLoweringMode::Disallowed);
        let type_param_mode = ParamLoweringMode::Placeholder;
        let in_binders = DebruijnIndex::INNERMOST;
        Self {
            db,
            resolver,
            def,
            generics: Default::default(),
            store,
            in_binders,
            impl_trait_mode,
            type_param_mode,
            unsized_types: FxHashSet::default(),
            diagnostics: Vec::new(),
            lifetime_elision,
        }
    }

    pub(crate) fn with_debruijn<T>(
        &mut self,
        debruijn: DebruijnIndex,
        f: impl FnOnce(&mut TyLoweringContext<'_>) -> T,
    ) -> T {
        let old_debruijn = mem::replace(&mut self.in_binders, debruijn);
        let result = f(self);
        self.in_binders = old_debruijn;
        result
    }

    pub(crate) fn with_shifted_in<T>(
        &mut self,
        debruijn: DebruijnIndex,
        f: impl FnOnce(&mut TyLoweringContext<'_>) -> T,
    ) -> T {
        self.with_debruijn(self.in_binders.shifted_in_from(debruijn), f)
    }

    fn with_lifetime_elision<T>(
        &mut self,
        lifetime_elision: LifetimeElisionKind,
        f: impl FnOnce(&mut TyLoweringContext<'_>) -> T,
    ) -> T {
        let old_lifetime_elision = mem::replace(&mut self.lifetime_elision, lifetime_elision);
        let result = f(self);
        self.lifetime_elision = old_lifetime_elision;
        result
    }

    pub(crate) fn with_impl_trait_mode(self, impl_trait_mode: ImplTraitLoweringMode) -> Self {
        Self { impl_trait_mode: ImplTraitLoweringState::new(impl_trait_mode), ..self }
    }

    pub(crate) fn with_type_param_mode(self, type_param_mode: ParamLoweringMode) -> Self {
        Self { type_param_mode, ..self }
    }

    pub(crate) fn push_diagnostic(&mut self, type_ref: TypeRefId, kind: TyLoweringDiagnosticKind) {
        self.diagnostics.push(TyLoweringDiagnostic { source: type_ref, kind });
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum ImplTraitLoweringMode {
    /// `impl Trait` gets lowered into an opaque type that doesn't unify with
    /// anything except itself. This is used in places where values flow 'out',
    /// i.e. for arguments of the function we're currently checking, and return
    /// types of functions we're calling.
    Opaque,
    /// `impl Trait` is disallowed and will be an error.
    #[default]
    Disallowed,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ParamLoweringMode {
    Placeholder,
    Variable,
}

impl<'db> TyLoweringContext<'db> {
    pub(crate) fn lower_ty(&mut self, type_ref: TypeRefId) -> Ty {
        self.lower_ty_ext(type_ref).0
    }

    pub(crate) fn lower_const(&mut self, const_ref: &ConstRef, const_type: Ty) -> Const {
        let const_ref = &self.store[const_ref.expr];
        match const_ref {
            hir_def::hir::Expr::Path(path) => path_to_const(
                self.db,
                self.resolver,
                path,
                self.type_param_mode,
                || self.generics(),
                self.in_binders,
                const_type.clone(),
            )
            .unwrap_or_else(|| unknown_const(const_type)),
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

    pub(crate) fn lower_path_as_const(&mut self, path: &Path, const_type: Ty) -> Const {
        path_to_const(
            self.db,
            self.resolver,
            path,
            self.type_param_mode,
            || self.generics(),
            self.in_binders,
            const_type.clone(),
        )
        .unwrap_or_else(|| unknown_const(const_type))
    }

    fn generics(&self) -> &Generics {
        self.generics.get_or_init(|| generics(self.db, self.def))
    }

    pub(crate) fn lower_ty_ext(&mut self, type_ref_id: TypeRefId) -> (Ty, Option<TypeNs>) {
        let mut res = None;
        let type_ref = &self.store[type_ref_id];
        let ty = match type_ref {
            TypeRef::Never => TyKind::Never.intern(Interner),
            TypeRef::Tuple(inner) => {
                let inner_tys = inner.iter().map(|&tr| self.lower_ty(tr));
                TyKind::Tuple(inner_tys.len(), Substitution::from_iter(Interner, inner_tys))
                    .intern(Interner)
            }
            TypeRef::Path(path) => {
                let (ty, res_) =
                    self.lower_path(path, PathId::from_type_ref_unchecked(type_ref_id));
                res = res_;
                ty
            }
            &TypeRef::TypeParam(type_param_id) => {
                res = Some(TypeNs::GenericParam(type_param_id));
                match self.type_param_mode {
                    ParamLoweringMode::Placeholder => {
                        let generics = self.generics();
                        let idx = generics.type_or_const_param_idx(type_param_id.into()).unwrap();
                        TyKind::Placeholder(to_placeholder_idx(
                            self.db,
                            type_param_id.into(),
                            idx as u32,
                        ))
                    }
                    ParamLoweringMode::Variable => {
                        let idx =
                            self.generics().type_or_const_param_idx(type_param_id.into()).unwrap();
                        TyKind::BoundVar(BoundVar::new(self.in_binders, idx))
                    }
                }
                .intern(Interner)
            }
            &TypeRef::RawPtr(inner, mutability) => {
                let inner_ty = self.lower_ty(inner);
                TyKind::Raw(lower_to_chalk_mutability(mutability), inner_ty).intern(Interner)
            }
            TypeRef::Array(array) => {
                let inner_ty = self.lower_ty(array.ty);
                let const_len = self.lower_const(&array.len, TyBuilder::usize());
                TyKind::Array(inner_ty, const_len).intern(Interner)
            }
            &TypeRef::Slice(inner) => {
                let inner_ty = self.lower_ty(inner);
                TyKind::Slice(inner_ty).intern(Interner)
            }
            TypeRef::Reference(ref_) => {
                let inner_ty = self.lower_ty(ref_.ty);
                // FIXME: It should infer the eldided lifetimes instead of stubbing with static
                let lifetime = ref_
                    .lifetime
                    .as_ref()
                    .map_or_else(error_lifetime, |&lr| self.lower_lifetime(lr));
                TyKind::Ref(lower_to_chalk_mutability(ref_.mutability), lifetime, inner_ty)
                    .intern(Interner)
            }
            TypeRef::Placeholder => TyKind::Error.intern(Interner),
            TypeRef::Fn(fn_) => {
                let substs = self.with_shifted_in(DebruijnIndex::ONE, |ctx| {
                    let (params, ret) = fn_.split_params_and_ret();
                    let mut subst = Vec::with_capacity(fn_.params.len());
                    ctx.with_lifetime_elision(
                        LifetimeElisionKind::AnonymousCreateParameter { report_in_path: false },
                        |ctx| {
                            subst.extend(params.iter().map(|&(_, tr)| ctx.lower_ty(tr)));
                        },
                    );
                    ctx.with_lifetime_elision(LifetimeElisionKind::for_fn_ret(), |ctx| {
                        subst.push(ctx.lower_ty(ret));
                    });
                    Substitution::from_iter(Interner, subst)
                });
                TyKind::Function(FnPointer {
                    num_binders: 0, // FIXME lower `for<'a> fn()` correctly
                    sig: FnSig {
                        abi: fn_.abi.as_ref().map_or(FnAbi::Rust, FnAbi::from_symbol),
                        safety: if fn_.is_unsafe { Safety::Unsafe } else { Safety::Safe },
                        variadic: fn_.is_varargs,
                    },
                    substitution: FnSubst(substs),
                })
                .intern(Interner)
            }
            TypeRef::DynTrait(bounds) => self.lower_dyn_trait(bounds),
            TypeRef::ImplTrait(bounds) => {
                match self.impl_trait_mode.mode {
                    ImplTraitLoweringMode::Opaque => {
                        let origin = match self.def {
                            GenericDefId::FunctionId(it) => Either::Left(it),
                            GenericDefId::TypeAliasId(it) => Either::Right(it),
                            _ => panic!(
                                "opaque impl trait lowering must be in function or type alias"
                            ),
                        };

                        // this dance is to make sure the data is in the right
                        // place even if we encounter more opaque types while
                        // lowering the bounds
                        let idx = self.impl_trait_mode.opaque_type_data.alloc(ImplTrait {
                            bounds: crate::make_single_type_binders(Vec::default()),
                        });
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
                            .with_debruijn(DebruijnIndex::INNERMOST, |ctx| {
                                ctx.lower_impl_trait(bounds, self.resolver.krate())
                            });
                        self.impl_trait_mode.opaque_type_data[idx] = actual_opaque_type_data;

                        let impl_trait_id = origin.either(
                            |f| ImplTraitId::ReturnTypeImplTrait(f, idx),
                            |a| ImplTraitId::TypeAliasImplTrait(a, idx),
                        );
                        let opaque_ty_id = self.db.intern_impl_trait_id(impl_trait_id).into();
                        let generics = generics(self.db, origin.either(|f| f.into(), |a| a.into()));
                        let parameters = generics.bound_vars_subst(self.db, self.in_binders);
                        TyKind::OpaqueType(opaque_ty_id, parameters).intern(Interner)
                    }
                    ImplTraitLoweringMode::Disallowed => {
                        // FIXME: report error
                        TyKind::Error.intern(Interner)
                    }
                }
            }
            TypeRef::Error => TyKind::Error.intern(Interner),
        };
        (ty, res)
    }

    /// This is only for `generic_predicates_for_param`, where we can't just
    /// lower the self types of the predicates since that could lead to cycles.
    /// So we just check here if the `type_ref` resolves to a generic param, and which.
    fn lower_ty_only_param(&mut self, type_ref_id: TypeRefId) -> Option<TypeOrConstParamId> {
        let type_ref = &self.store[type_ref_id];
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
        let mut ctx = self.at_path(PathId::from_type_ref_unchecked(type_ref_id));
        let resolution = match ctx.resolve_path_in_type_ns() {
            Some((it, None)) => it,
            _ => return None,
        };
        match resolution {
            TypeNs::GenericParam(param_id) => Some(param_id.into()),
            _ => None,
        }
    }

    #[inline]
    fn on_path_diagnostic_callback<'a>(type_ref: TypeRefId) -> PathDiagnosticCallback<'a, 'db> {
        PathDiagnosticCallback {
            data: Either::Left(PathDiagnosticCallbackData(type_ref)),
            callback: |data, this, diag| {
                let type_ref = data.as_ref().left().unwrap().0;
                this.push_diagnostic(type_ref, TyLoweringDiagnosticKind::PathDiagnostic(diag))
            },
        }
    }

    #[inline]
    fn at_path(&mut self, path_id: PathId) -> PathLoweringContext<'_, 'db> {
        PathLoweringContext::new(
            self,
            Self::on_path_diagnostic_callback(path_id.type_ref()),
            &self.store[path_id],
        )
    }

    pub(crate) fn lower_path(&mut self, path: &Path, path_id: PathId) -> (Ty, Option<TypeNs>) {
        // Resolve the path (in type namespace)
        if let Some(type_ref) = path.type_anchor() {
            let (ty, res) = self.lower_ty_ext(type_ref);
            let mut ctx = self.at_path(path_id);
            return ctx.lower_ty_relative_path(ty, res, false);
        }

        let mut ctx = self.at_path(path_id);
        let (resolution, remaining_index) = match ctx.resolve_path_in_type_ns() {
            Some(it) => it,
            None => return (TyKind::Error.intern(Interner), None),
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
        explicit_self_ty: Ty,
    ) -> Option<(TraitRef, PathLoweringContext<'_, 'db>)> {
        let mut ctx = self.at_path(path_id);
        let resolved = match ctx.resolve_path_in_type_ns_fully()? {
            // FIXME(trait_alias): We need to handle trait alias here.
            TypeNs::TraitId(tr) => tr,
            _ => return None,
        };
        Some((ctx.lower_trait_ref_from_resolved_path(resolved, explicit_self_ty, false), ctx))
    }

    /// When lowering predicates from parents (impl, traits) for children defs (fns, consts, types), `generics` should
    /// contain the `Generics` for the **child**, while `predicate_owner` should contain the `GenericDefId` of the
    /// **parent**. This is important so we generate the correct bound var/placeholder.
    pub(crate) fn lower_where_predicate<'b>(
        &'b mut self,
        where_predicate: &'b WherePredicate,
        ignore_bindings: bool,
    ) -> impl Iterator<Item = QuantifiedWhereClause> + use<'db, 'b> {
        match where_predicate {
            WherePredicate::ForLifetime { target, bound, .. }
            | WherePredicate::TypeBound { target, bound } => {
                let self_ty = self.lower_ty(*target);
                Either::Left(self.lower_type_bound(bound, self_ty, ignore_bindings))
            }
            &WherePredicate::Lifetime { bound, target } => Either::Right(iter::once(
                crate::wrap_empty_binders(WhereClause::LifetimeOutlives(LifetimeOutlives {
                    a: self.lower_lifetime(bound),
                    b: self.lower_lifetime(target),
                })),
            )),
        }
        .into_iter()
    }

    pub(crate) fn lower_type_bound<'b>(
        &'b mut self,
        bound: &'b TypeBound,
        self_ty: Ty,
        ignore_bindings: bool,
    ) -> impl Iterator<Item = QuantifiedWhereClause> + use<'b, 'db> {
        let mut assoc_bounds = None;
        let mut clause = None;
        match bound {
            &TypeBound::Path(path, TraitBoundModifier::None) | &TypeBound::ForLifetime(_, path) => {
                // FIXME Don't silently drop the hrtb lifetimes here
                if let Some((trait_ref, mut ctx)) =
                    self.lower_trait_ref_from_path(path, self_ty.clone())
                {
                    // FIXME(sized-hierarchy): Remove this bound modifications once we have implemented
                    // sized-hierarchy correctly.
                    let meta_sized = LangItem::MetaSized
                        .resolve_trait(ctx.ty_ctx().db, ctx.ty_ctx().resolver.krate());
                    let pointee_sized = LangItem::PointeeSized
                        .resolve_trait(ctx.ty_ctx().db, ctx.ty_ctx().resolver.krate());
                    let destruct = LangItem::Destruct
                        .resolve_trait(ctx.ty_ctx().db, ctx.ty_ctx().resolver.krate());
                    let hir_trait_id = trait_ref.hir_trait_id();
                    if meta_sized.is_some_and(|it| it == hir_trait_id)
                        || destruct.is_some_and(|it| it == hir_trait_id)
                    {
                        // Ignore this bound
                    } else if pointee_sized.is_some_and(|it| it == hir_trait_id) {
                        // Regard this as `?Sized` bound
                        ctx.ty_ctx().unsized_types.insert(self_ty);
                    } else {
                        if !ignore_bindings {
                            assoc_bounds =
                                ctx.assoc_type_bindings_from_type_bound(trait_ref.clone());
                        }
                        clause =
                            Some(crate::wrap_empty_binders(WhereClause::Implemented(trait_ref)));
                    }
                }
            }
            &TypeBound::Path(path, TraitBoundModifier::Maybe) => {
                let sized_trait = LangItem::Sized.resolve_trait(self.db, self.resolver.krate());
                // Don't lower associated type bindings as the only possible relaxed trait bound
                // `?Sized` has no of them.
                // If we got another trait here ignore the bound completely.
                let trait_id = self
                    .lower_trait_ref_from_path(path, self_ty.clone())
                    .map(|(trait_ref, _)| trait_ref.hir_trait_id());
                if trait_id == sized_trait {
                    self.unsized_types.insert(self_ty);
                }
            }
            &TypeBound::Lifetime(l) => {
                let lifetime = self.lower_lifetime(l);
                clause = Some(crate::wrap_empty_binders(WhereClause::TypeOutlives(TypeOutlives {
                    ty: self_ty,
                    lifetime,
                })));
            }
            TypeBound::Use(_) | TypeBound::Error => {}
        }
        clause.into_iter().chain(assoc_bounds.into_iter().flatten())
    }

    fn lower_dyn_trait(&mut self, bounds: &[TypeBound]) -> Ty {
        let self_ty = TyKind::BoundVar(BoundVar::new(DebruijnIndex::INNERMOST, 0)).intern(Interner);
        // INVARIANT: The principal trait bound, if present, must come first. Others may be in any
        // order but should be in the same order for the same set but possibly different order of
        // bounds in the input.
        // INVARIANT: If this function returns `DynTy`, there should be at least one trait bound.
        // These invariants are utilized by `TyExt::dyn_trait()` and chalk.
        let mut lifetime = None;
        let bounds = self.with_shifted_in(DebruijnIndex::ONE, |ctx| {
            let mut lowered_bounds = Vec::new();
            for b in bounds {
                ctx.lower_type_bound(b, self_ty.clone(), false).for_each(|b| {
                    let filter = match b.skip_binders() {
                        WhereClause::Implemented(_) | WhereClause::AliasEq(_) => true,
                        WhereClause::LifetimeOutlives(_) => false,
                        WhereClause::TypeOutlives(t) => {
                            lifetime = Some(t.lifetime.clone());
                            false
                        }
                    };
                    if filter {
                        lowered_bounds.push(b);
                    }
                });
            }

            let mut multiple_regular_traits = false;
            let mut multiple_same_projection = false;
            lowered_bounds.sort_unstable_by(|lhs, rhs| {
                use std::cmp::Ordering;
                match (lhs.skip_binders(), rhs.skip_binders()) {
                    (WhereClause::Implemented(lhs), WhereClause::Implemented(rhs)) => {
                        let lhs_id = lhs.trait_id;
                        let lhs_is_auto = ctx
                            .db
                            .trait_signature(from_chalk_trait_id(lhs_id))
                            .flags
                            .contains(TraitFlags::AUTO);
                        let rhs_id = rhs.trait_id;
                        let rhs_is_auto = ctx
                            .db
                            .trait_signature(from_chalk_trait_id(rhs_id))
                            .flags
                            .contains(TraitFlags::AUTO);

                        if !lhs_is_auto && !rhs_is_auto {
                            multiple_regular_traits = true;
                        }
                        // Note that the ordering here is important; this ensures the invariant
                        // mentioned above.
                        (lhs_is_auto, lhs_id).cmp(&(rhs_is_auto, rhs_id))
                    }
                    (WhereClause::Implemented(_), _) => Ordering::Less,
                    (_, WhereClause::Implemented(_)) => Ordering::Greater,
                    (WhereClause::AliasEq(lhs), WhereClause::AliasEq(rhs)) => {
                        match (&lhs.alias, &rhs.alias) {
                            (AliasTy::Projection(lhs_proj), AliasTy::Projection(rhs_proj)) => {
                                // We only compare the `associated_ty_id`s. We shouldn't have
                                // multiple bounds for an associated type in the correct Rust code,
                                // and if we do, we error out.
                                if lhs_proj.associated_ty_id == rhs_proj.associated_ty_id {
                                    multiple_same_projection = true;
                                }
                                lhs_proj.associated_ty_id.cmp(&rhs_proj.associated_ty_id)
                            }
                            // We don't produce `AliasTy::Opaque`s yet.
                            _ => unreachable!(),
                        }
                    }
                    // `WhereClause::{TypeOutlives, LifetimeOutlives}` have been filtered out
                    _ => unreachable!(),
                }
            });

            if multiple_regular_traits || multiple_same_projection {
                return None;
            }

            lowered_bounds.first().and_then(|b| b.trait_id())?;

            // As multiple occurrences of the same auto traits *are* permitted, we deduplicate the
            // bounds. We shouldn't have repeated elements besides auto traits at this point.
            lowered_bounds.dedup();

            Some(QuantifiedWhereClauses::from_iter(Interner, lowered_bounds))
        });

        if let Some(bounds) = bounds {
            let bounds = crate::make_single_type_binders(bounds);
            TyKind::Dyn(DynTy {
                bounds,
                lifetime: match lifetime {
                    Some(it) => match it.bound_var(Interner) {
                        Some(bound_var) => bound_var
                            .shifted_out_to(DebruijnIndex::new(2))
                            .map(|bound_var| LifetimeData::BoundVar(bound_var).intern(Interner))
                            .unwrap_or(it),
                        None => it,
                    },
                    None => error_lifetime(),
                },
            })
            .intern(Interner)
        } else {
            // FIXME: report error
            // (additional non-auto traits, associated type rebound, or no resolved trait)
            TyKind::Error.intern(Interner)
        }
    }

    fn lower_impl_trait(&mut self, bounds: &[TypeBound], krate: Crate) -> ImplTrait {
        cov_mark::hit!(lower_rpit);
        let self_ty = TyKind::BoundVar(BoundVar::new(DebruijnIndex::INNERMOST, 0)).intern(Interner);
        let predicates = self.with_shifted_in(DebruijnIndex::ONE, |ctx| {
            let mut predicates = Vec::new();
            for b in bounds {
                predicates.extend(ctx.lower_type_bound(b, self_ty.clone(), false));
            }

            if !ctx.unsized_types.contains(&self_ty) {
                let sized_trait =
                    LangItem::Sized.resolve_trait(ctx.db, krate).map(to_chalk_trait_id);
                let sized_clause = sized_trait.map(|trait_id| {
                    let clause = WhereClause::Implemented(TraitRef {
                        trait_id,
                        substitution: Substitution::from1(Interner, self_ty.clone()),
                    });
                    crate::wrap_empty_binders(clause)
                });
                predicates.extend(sized_clause);
            }
            predicates.shrink_to_fit();
            predicates
        });
        ImplTrait { bounds: crate::make_single_type_binders(predicates) }
    }

    pub(crate) fn lower_lifetime(&self, lifetime: LifetimeRefId) -> Lifetime {
        match self.resolver.resolve_lifetime(&self.store[lifetime]) {
            Some(resolution) => match resolution {
                LifetimeNs::Static => static_lifetime(),
                LifetimeNs::LifetimeParam(id) => match self.type_param_mode {
                    ParamLoweringMode::Placeholder => {
                        let generics = self.generics();
                        let idx = generics.lifetime_idx(id).unwrap();
                        LifetimeData::Placeholder(lt_to_placeholder_idx(self.db, id, idx as u32))
                    }
                    ParamLoweringMode::Variable => {
                        let idx = match self.generics().lifetime_idx(id) {
                            None => return error_lifetime(),
                            Some(idx) => idx,
                        };

                        LifetimeData::BoundVar(BoundVar::new(self.in_binders, idx))
                    }
                }
                .intern(Interner),
            },
            None => error_lifetime(),
        }
    }
}

fn named_associated_type_shorthand_candidates<R>(
    db: &dyn HirDatabase,
    // If the type parameter is defined in an impl and we're in a method, there
    // might be additional where clauses to consider
    def: GenericDefId,
    res: TypeNs,
    assoc_name: Option<Name>,
    // Do NOT let `cb` touch `TraitRef` outside of `TyLoweringContext`. Its substitution contains
    // free `BoundVar`s that need to be shifted and only `TyLoweringContext` knows how to do that
    // properly (see `TyLoweringContext::select_associated_type()`).
    mut cb: impl FnMut(&Name, &TraitRef, TypeAliasId) -> Option<R>,
) -> Option<R> {
    let mut search = |t| {
        all_super_trait_refs(db, t, |t| {
            let data = t.hir_trait_id().trait_items(db);

            for (name, assoc_id) in &data.items {
                if let AssocItemId::TypeAliasId(alias) = assoc_id
                    && let Some(result) = cb(name, &t, *alias)
                {
                    return Some(result);
                }
            }
            None
        })
    };

    let interner = DbInterner::new_with(db, None, None);
    match res {
        TypeNs::SelfType(impl_id) => {
            let trait_ref = db.impl_trait(impl_id)?;

            let impl_id_as_generic_def: GenericDefId = impl_id.into();
            if impl_id_as_generic_def != def {
                let subst = TyBuilder::subst_for_def(db, impl_id, None)
                    .fill_with_bound_vars(DebruijnIndex::INNERMOST, 0)
                    .build();
                let args: crate::next_solver::GenericArgs<'_> = subst.to_nextsolver(interner);
                let trait_ref = trait_ref.instantiate(interner, args).to_chalk(interner);
                search(trait_ref)
            } else {
                search(trait_ref.skip_binder().to_chalk(interner))
            }
        }
        TypeNs::GenericParam(param_id) => {
            let predicates = db.generic_predicates_for_param(def, param_id.into(), assoc_name);
            let res = predicates.iter().find_map(|pred| match pred.skip_binders().skip_binders() {
                // FIXME: how to correctly handle higher-ranked bounds here?
                WhereClause::Implemented(tr) => search(
                    tr.clone()
                        .shifted_out_to(Interner, DebruijnIndex::ONE)
                        .expect("FIXME unexpected higher-ranked trait bound"),
                ),
                _ => None,
            });
            if res.is_some() {
                return res;
            }
            // Handle `Self::Type` referring to own associated type in trait definitions
            if let GenericDefId::TraitId(trait_id) = param_id.parent() {
                let trait_generics = generics(db, trait_id.into());
                if trait_generics[param_id.local_id()].is_trait_self() {
                    let trait_ref = TyBuilder::trait_ref(db, trait_id)
                        .fill_with_bound_vars(DebruijnIndex::INNERMOST, 0)
                        .build();
                    return search(trait_ref);
                }
            }
            None
        }
        _ => None,
    }
}

pub(crate) type Diagnostics = Option<ThinArc<(), TyLoweringDiagnostic>>;

pub(crate) fn create_diagnostics(diagnostics: Vec<TyLoweringDiagnostic>) -> Diagnostics {
    (!diagnostics.is_empty()).then(|| ThinArc::from_header_and_iter((), diagnostics.into_iter()))
}

pub(crate) fn field_types_query(
    db: &dyn HirDatabase,
    variant_id: VariantId,
) -> Arc<ArenaMap<LocalFieldId, Binders<Ty>>> {
    field_types_with_diagnostics_query(db, variant_id).0
}

/// Build the type of all specific fields of a struct or enum variant.
pub(crate) fn field_types_with_diagnostics_query(
    db: &dyn HirDatabase,
    variant_id: VariantId,
) -> (Arc<ArenaMap<LocalFieldId, Binders<Ty>>>, Diagnostics) {
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
    let generics = generics(db, def);
    let mut res = ArenaMap::default();
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        &var_data.store,
        def,
        LifetimeElisionKind::AnonymousReportError,
    )
    .with_type_param_mode(ParamLoweringMode::Variable);
    for (field_id, field_data) in fields.iter() {
        res.insert(field_id, make_binders(db, &generics, ctx.lower_ty(field_data.type_ref)));
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
pub(crate) fn generic_predicates_for_param_query(
    db: &dyn HirDatabase,
    def: GenericDefId,
    param_id: TypeOrConstParamId,
    assoc_name: Option<Name>,
) -> GenericPredicates {
    let generics = generics(db, def);
    if generics.has_no_predicates() && generics.is_empty() {
        return GenericPredicates(None);
    }

    let resolver = def.resolver(db);
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        generics.store(),
        def,
        LifetimeElisionKind::AnonymousReportError,
    )
    .with_type_param_mode(ParamLoweringMode::Variable);

    // we have to filter out all other predicates *first*, before attempting to lower them
    let predicate = |pred: &_, ctx: &mut TyLoweringContext<'_>| match pred {
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
                            let Some(pointee_sized) =
                                LangItem::PointeeSized.resolve_trait(ctx.db, ctx.resolver.krate())
                            else {
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
                    ctx.lower_where_predicate(pred, true).for_each(drop);
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

                    all_super_traits(db, tr).iter().any(|tr| {
                        tr.trait_items(db).items.iter().any(|(name, item)| {
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
                predicates.extend(
                    ctx.lower_where_predicate(pred, true).map(|p| make_binders(db, &generics, p)),
                );
            }
        }
    }

    let subst = generics.bound_vars_subst(db, DebruijnIndex::INNERMOST);
    if !subst.is_empty(Interner) {
        let explicitly_unsized_tys = ctx.unsized_types;
        if let Some(implicitly_sized_predicates) = implicitly_sized_clauses(
            db,
            param_id.parent,
            &explicitly_unsized_tys,
            &subst,
            &resolver,
        ) {
            predicates.extend(
                implicitly_sized_predicates
                    .map(|p| make_binders(db, &generics, crate::wrap_empty_binders(p))),
            );
        };
    }
    GenericPredicates(predicates.is_empty().not().then(|| predicates.into()))
}

pub(crate) fn generic_predicates_for_param_cycle_result(
    _db: &dyn HirDatabase,
    _def: GenericDefId,
    _param_id: TypeOrConstParamId,
    _assoc_name: Option<Name>,
) -> GenericPredicates {
    GenericPredicates(None)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GenericPredicates(Option<Arc<[Binders<QuantifiedWhereClause>]>>);

impl ops::Deref for GenericPredicates {
    type Target = [Binders<crate::QuantifiedWhereClause>];

    fn deref(&self) -> &Self::Target {
        self.0.as_deref().unwrap_or(&[])
    }
}

/// Resolve the where clause(s) of an item with generics.
pub(crate) fn generic_predicates_query(
    db: &dyn HirDatabase,
    def: GenericDefId,
) -> GenericPredicates {
    generic_predicates_filtered_by(db, def, |_, _| true).0
}

/// Resolve the where clause(s) of an item with generics,
/// with a given filter
fn generic_predicates_filtered_by<F>(
    db: &dyn HirDatabase,
    def: GenericDefId,
    filter: F,
) -> (GenericPredicates, Diagnostics)
where
    F: Fn(&WherePredicate, GenericDefId) -> bool,
{
    let generics = generics(db, def);
    if generics.has_no_predicates() && generics.is_empty() {
        return (GenericPredicates(None), None);
    }

    let resolver = def.resolver(db);
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        generics.store(),
        def,
        LifetimeElisionKind::AnonymousReportError,
    )
    .with_type_param_mode(ParamLoweringMode::Variable);

    let mut predicates = Vec::new();
    for maybe_parent_generics in
        std::iter::successors(Some(&generics), |generics| generics.parent_generics())
    {
        ctx.store = maybe_parent_generics.store();
        for pred in maybe_parent_generics.where_predicates() {
            if filter(pred, maybe_parent_generics.def()) {
                // We deliberately use `generics` and not `maybe_parent_generics` here. This is not a mistake!
                // If we use the parent generics
                predicates.extend(
                    ctx.lower_where_predicate(pred, false).map(|p| make_binders(db, &generics, p)),
                );
            }
        }
    }

    if !generics.is_empty() {
        let subst = generics.bound_vars_subst(db, DebruijnIndex::INNERMOST);
        let explicitly_unsized_tys = ctx.unsized_types;
        if let Some(implicitly_sized_predicates) =
            implicitly_sized_clauses(db, def, &explicitly_unsized_tys, &subst, &resolver)
        {
            predicates.extend(
                implicitly_sized_predicates
                    .map(|p| make_binders(db, &generics, crate::wrap_empty_binders(p))),
            );
        };
    }

    (
        GenericPredicates(predicates.is_empty().not().then(|| predicates.into())),
        create_diagnostics(ctx.diagnostics),
    )
}

/// Generate implicit `: Sized` predicates for all generics that has no `?Sized` bound.
/// Exception is Self of a trait def.
fn implicitly_sized_clauses<'db, 'a, 'subst: 'a>(
    db: &'db dyn HirDatabase,
    def: GenericDefId,
    explicitly_unsized_tys: &'a FxHashSet<Ty>,
    substitution: &'subst Substitution,
    resolver: &Resolver<'db>,
) -> Option<impl Iterator<Item = WhereClause>> {
    let sized_trait = LangItem::Sized.resolve_trait(db, resolver.krate()).map(to_chalk_trait_id)?;

    let trait_self_idx = trait_self_param_idx(db, def);

    Some(
        substitution
            .iter(Interner)
            .enumerate()
            .filter_map(
                move |(idx, generic_arg)| {
                    if Some(idx) == trait_self_idx { None } else { Some(generic_arg) }
                },
            )
            .filter_map(|generic_arg| generic_arg.ty(Interner))
            .filter(move |&self_ty| !explicitly_unsized_tys.contains(self_ty))
            .map(move |self_ty| {
                WhereClause::Implemented(TraitRef {
                    trait_id: sized_trait,
                    substitution: Substitution::from1(Interner, self_ty.clone()),
                })
            }),
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GenericDefaults(Option<Arc<[Binders<crate::GenericArg>]>>);

impl ops::Deref for GenericDefaults {
    type Target = [Binders<crate::GenericArg>];

    fn deref(&self) -> &Self::Target {
        self.0.as_deref().unwrap_or(&[])
    }
}

pub(crate) fn generic_defaults_query(db: &dyn HirDatabase, def: GenericDefId) -> GenericDefaults {
    db.generic_defaults_with_diagnostics(def).0
}

/// Resolve the default type params from generics.
///
/// Diagnostics are only returned for this `GenericDefId` (returned defaults include parents).
pub(crate) fn generic_defaults_with_diagnostics_query(
    db: &dyn HirDatabase,
    def: GenericDefId,
) -> (GenericDefaults, Diagnostics) {
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
    .with_impl_trait_mode(ImplTraitLoweringMode::Disallowed)
    .with_type_param_mode(ParamLoweringMode::Variable);
    let mut idx = 0;
    let mut has_any_default = false;
    let mut defaults = generic_params
        .iter_parents_with_store()
        .map(|((id, p), store)| {
            ctx.store = store;
            let (result, has_default) = handle_generic_param(&mut ctx, idx, id, p, &generic_params);
            has_any_default |= has_default;
            idx += 1;
            result
        })
        .collect::<Vec<_>>();
    ctx.diagnostics.clear(); // Don't include diagnostics from the parent.
    defaults.extend(generic_params.iter_self().map(|(id, p)| {
        let (result, has_default) = handle_generic_param(&mut ctx, idx, id, p, &generic_params);
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

    fn handle_generic_param(
        ctx: &mut TyLoweringContext<'_>,
        idx: usize,
        id: GenericParamId,
        p: GenericParamDataRef<'_>,
        generic_params: &Generics,
    ) -> (Binders<crate::GenericArg>, bool) {
        let binders = variable_kinds_from_iter(ctx.db, generic_params.iter_id().take(idx));
        match p {
            GenericParamDataRef::TypeParamData(p) => {
                let ty = p.default.as_ref().map_or_else(
                    || TyKind::Error.intern(Interner),
                    |ty| {
                        // Each default can only refer to previous parameters.
                        // Type variable default referring to parameter coming
                        // after it is forbidden (FIXME: report diagnostic)
                        fallback_bound_vars(ctx.lower_ty(*ty), idx)
                    },
                );
                (Binders::new(binders, ty.cast(Interner)), p.default.is_some())
            }
            GenericParamDataRef::ConstParamData(p) => {
                let GenericParamId::ConstParamId(id) = id else {
                    unreachable!("Unexpected lifetime or type argument")
                };

                let mut val = p.default.as_ref().map_or_else(
                    || unknown_const_as_generic(ctx.db.const_param_ty(id)),
                    |c| {
                        let param_ty = ctx.lower_ty(p.ty);
                        let c = ctx.lower_const(c, param_ty);
                        c.cast(Interner)
                    },
                );
                // Each default can only refer to previous parameters, see above.
                val = fallback_bound_vars(val, idx);
                (Binders::new(binders, val), p.default.is_some())
            }
            GenericParamDataRef::LifetimeParamData(_) => {
                (Binders::new(binders, error_lifetime().cast(Interner)), false)
            }
        }
    }
}

pub(crate) fn generic_defaults_with_diagnostics_cycle_result(
    _db: &dyn HirDatabase,
    _def: GenericDefId,
) -> (GenericDefaults, Diagnostics) {
    (GenericDefaults(None), None)
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

pub(crate) fn const_param_ty_query(db: &dyn HirDatabase, def: ConstParamId) -> Ty {
    const_param_ty_with_diagnostics_query(db, def).0
}

// returns None if def is a type arg
pub(crate) fn const_param_ty_with_diagnostics_query(
    db: &dyn HirDatabase,
    def: ConstParamId,
) -> (Ty, Diagnostics) {
    let (parent_data, store) = db.generic_params_and_store(def.parent());
    let data = &parent_data[def.local_id()];
    let resolver = def.parent().resolver(db);
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
            Ty::new(Interner, TyKind::Error)
        }
        TypeOrConstParamData::ConstParamData(d) => ctx.lower_ty(d.ty),
    };
    (ty, create_diagnostics(ctx.diagnostics))
}

pub(crate) fn const_param_ty_cycle_result(
    _: &dyn HirDatabase,
    _: crate::db::HirDatabaseData,
    _: ConstParamId,
) -> Ty {
    TyKind::Error.intern(Interner)
}

pub(crate) fn return_type_impl_traits(
    db: &dyn HirDatabase,
    def: hir_def::FunctionId,
) -> Option<Arc<Binders<ImplTraits>>> {
    // FIXME unify with fn_sig_for_fn instead of doing lowering twice, maybe
    let data = db.function_signature(def);
    let resolver = def.resolver(db);
    let mut ctx_ret =
        TyLoweringContext::new(db, &resolver, &data.store, def.into(), LifetimeElisionKind::Infer)
            .with_impl_trait_mode(ImplTraitLoweringMode::Opaque)
            .with_type_param_mode(ParamLoweringMode::Variable);
    if let Some(ret_type) = data.ret_type {
        let _ret = ctx_ret.lower_ty(ret_type);
    }
    let generics = generics(db, def.into());
    let return_type_impl_traits =
        ImplTraits { impl_traits: ctx_ret.impl_trait_mode.opaque_type_data };
    if return_type_impl_traits.impl_traits.is_empty() {
        None
    } else {
        Some(Arc::new(make_binders(db, &generics, return_type_impl_traits)))
    }
}

pub(crate) fn type_alias_impl_traits(
    db: &dyn HirDatabase,
    def: hir_def::TypeAliasId,
) -> Option<Arc<Binders<ImplTraits>>> {
    let data = db.type_alias_signature(def);
    let resolver = def.resolver(db);
    let mut ctx = TyLoweringContext::new(
        db,
        &resolver,
        &data.store,
        def.into(),
        LifetimeElisionKind::AnonymousReportError,
    )
    .with_impl_trait_mode(ImplTraitLoweringMode::Opaque)
    .with_type_param_mode(ParamLoweringMode::Variable);
    if let Some(type_ref) = data.ty {
        let _ty = ctx.lower_ty(type_ref);
    }
    let type_alias_impl_traits = ImplTraits { impl_traits: ctx.impl_trait_mode.opaque_type_data };
    if type_alias_impl_traits.impl_traits.is_empty() {
        None
    } else {
        let generics = generics(db, def.into());
        Some(Arc::new(make_binders(db, &generics, type_alias_impl_traits)))
    }
}

pub(crate) fn lower_to_chalk_mutability(m: hir_def::type_ref::Mutability) -> Mutability {
    match m {
        hir_def::type_ref::Mutability::Shared => Mutability::Not,
        hir_def::type_ref::Mutability::Mut => Mutability::Mut,
    }
}

/// Replaces any 'free' `BoundVar`s in `s` by `TyKind::Error` from the perspective of generic
/// parameter whose index is `param_index`. A `BoundVar` is free when it appears after the
/// generic parameter of `param_index`.
fn fallback_bound_vars<T: TypeFoldable<Interner> + HasInterner<Interner = Interner>>(
    s: T,
    param_index: usize,
) -> T {
    let is_allowed = |index| (0..param_index).contains(&index);

    crate::fold_free_vars(
        s,
        |bound, binders| {
            if bound.index_if_innermost().is_none_or(is_allowed) {
                bound.shifted_in_from(binders).to_ty(Interner)
            } else {
                TyKind::Error.intern(Interner)
            }
        },
        |ty, bound, binders| {
            if bound.index_if_innermost().is_none_or(is_allowed) {
                bound.shifted_in_from(binders).to_const(Interner, ty)
            } else {
                unknown_const(ty)
            }
        },
    )
}
