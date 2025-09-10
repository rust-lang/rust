//! Unification and canonicalization logic.

use std::{fmt, mem};

use chalk_ir::{
    CanonicalVarKind, FloatTy, IntTy, TyVariableKind, cast::Cast, fold::TypeFoldable,
    interner::HasInterner,
};
use either::Either;
use hir_def::{AdtId, lang_item::LangItem};
use hir_expand::name::Name;
use intern::sym;
use rustc_hash::{FxHashMap, FxHashSet};
use rustc_next_trait_solver::solve::HasChanged;
use rustc_type_ir::inherent::IntoKind;
use rustc_type_ir::{
    AliasRelationDirection, FloatVid, IntVid, TyVid,
    inherent::{Span, Term as _},
    relate::{Relate, solver_relating::RelateExt},
    solve::{Certainty, NoSolution},
};
use rustc_type_ir::{TypeSuperFoldable, TypeVisitableExt};
use smallvec::SmallVec;
use triomphe::Arc;

use super::{InferOk, InferResult, InferenceContext, TypeError};
use crate::{
    AliasEq, AliasTy, BoundVar, Canonical, Const, ConstValue, DebruijnIndex, DomainGoal,
    GenericArg, GenericArgData, Goal, GoalData, InEnvironment, InferenceVar, Interner, Lifetime,
    OpaqueTyId, ParamKind, ProjectionTy, ProjectionTyExt, Scalar, Substitution, TraitEnvironment,
    TraitRef, Ty, TyBuilder, TyExt, TyKind, VariableKind, WhereClause,
    consteval::unknown_const,
    db::HirDatabase,
    fold_generic_args, fold_tys_and_consts,
    next_solver::{
        self, Binder, DbInterner, Predicate, PredicateKind, SolverDefIds, Term,
        infer::{DbInternerInferExt, InferCtxt, snapshot::CombinedSnapshot},
        mapping::{ChalkToNextSolver, InferenceVarExt, NextSolverToChalk},
    },
    to_chalk_trait_id,
    traits::{
        FnTrait, NextTraitSolveResult, next_trait_solve_canonical_in_ctxt, next_trait_solve_in_ctxt,
    },
};

impl<'db> InferenceContext<'db> {
    pub(super) fn canonicalize<T>(&mut self, t: T) -> rustc_type_ir::Canonical<DbInterner<'db>, T>
    where
        T: rustc_type_ir::TypeFoldable<DbInterner<'db>>,
    {
        self.table.canonicalize(t)
    }

    pub(super) fn clauses_for_self_ty(
        &mut self,
        self_ty: InferenceVar,
    ) -> SmallVec<[WhereClause; 4]> {
        self.table.resolve_obligations_as_possible();

        let root = InferenceVar::from_vid(self.table.infer_ctxt.root_var(self_ty.to_vid()));
        let pending_obligations = mem::take(&mut self.table.pending_obligations);
        let obligations = pending_obligations
            .iter()
            .filter_map(|obligation| match obligation.to_chalk(self.table.interner).goal.data(Interner) {
                GoalData::DomainGoal(DomainGoal::Holds(clause)) => {
                    let ty = match clause {
                        WhereClause::AliasEq(AliasEq {
                            alias: AliasTy::Projection(projection),
                            ..
                        }) => projection.self_type_parameter(self.db),
                        WhereClause::Implemented(trait_ref) => {
                            trait_ref.self_type_parameter(Interner)
                        }
                        WhereClause::TypeOutlives(to) => to.ty.clone(),
                        _ => return None,
                    };
                    let ty = self.resolve_ty_shallow(&ty);
                    if matches!(ty.kind(Interner), TyKind::InferenceVar(iv, TyVariableKind::General) if *iv == root) {
                        Some(clause.clone())
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect();
        self.table.pending_obligations = pending_obligations;

        obligations
    }
}

/// Check if types unify.
///
/// Note that we consider placeholder types to unify with everything.
/// This means that there may be some unresolved goals that actually set bounds for the placeholder
/// type for the types to unify. For example `Option<T>` and `Option<U>` unify although there is
/// unresolved goal `T = U`.
pub fn could_unify(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    tys: &Canonical<(Ty, Ty)>,
) -> bool {
    unify(db, env, tys).is_some()
}

/// Check if types unify eagerly making sure there are no unresolved goals.
///
/// This means that placeholder types are not considered to unify if there are any bounds set on
/// them. For example `Option<T>` and `Option<U>` do not unify as we cannot show that `T = U`
pub fn could_unify_deeply(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    tys: &Canonical<(Ty, Ty)>,
) -> bool {
    let mut table = InferenceTable::new(db, env);
    let vars = make_substitutions(tys, &mut table);
    let ty1_with_vars = vars.apply(tys.value.0.clone(), Interner);
    let ty2_with_vars = vars.apply(tys.value.1.clone(), Interner);
    let ty1_with_vars = table.normalize_associated_types_in(ty1_with_vars);
    let ty2_with_vars = table.normalize_associated_types_in(ty2_with_vars);
    table.resolve_obligations_as_possible();
    table.propagate_diverging_flag();
    let ty1_with_vars = table.resolve_completely(ty1_with_vars);
    let ty2_with_vars = table.resolve_completely(ty2_with_vars);
    table.unify_deeply(&ty1_with_vars, &ty2_with_vars)
}

pub(crate) fn unify(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    tys: &Canonical<(Ty, Ty)>,
) -> Option<Substitution> {
    let mut table = InferenceTable::new(db, env);
    let vars = make_substitutions(tys, &mut table);
    let ty1_with_vars = vars.apply(tys.value.0.clone(), Interner);
    let ty2_with_vars = vars.apply(tys.value.1.clone(), Interner);
    if !table.unify(&ty1_with_vars, &ty2_with_vars) {
        return None;
    }
    // default any type vars that weren't unified back to their original bound vars
    // (kind of hacky)
    let find_var = |iv| {
        vars.iter(Interner).position(|v| match v.data(Interner) {
            GenericArgData::Ty(ty) => ty.inference_var(Interner),
            GenericArgData::Lifetime(lt) => lt.inference_var(Interner),
            GenericArgData::Const(c) => c.inference_var(Interner),
        } == Some(iv))
    };
    let fallback = |iv, kind, default, binder| match kind {
        chalk_ir::VariableKind::Ty(_ty_kind) => find_var(iv)
            .map_or(default, |i| BoundVar::new(binder, i).to_ty(Interner).cast(Interner)),
        chalk_ir::VariableKind::Lifetime => find_var(iv)
            .map_or(default, |i| BoundVar::new(binder, i).to_lifetime(Interner).cast(Interner)),
        chalk_ir::VariableKind::Const(ty) => find_var(iv)
            .map_or(default, |i| BoundVar::new(binder, i).to_const(Interner, ty).cast(Interner)),
    };
    Some(Substitution::from_iter(
        Interner,
        vars.iter(Interner).map(|v| table.resolve_with_fallback(v.clone(), &fallback)),
    ))
}

fn make_substitutions(
    tys: &chalk_ir::Canonical<(chalk_ir::Ty<Interner>, chalk_ir::Ty<Interner>)>,
    table: &mut InferenceTable<'_>,
) -> chalk_ir::Substitution<Interner> {
    Substitution::from_iter(
        Interner,
        tys.binders.iter(Interner).map(|it| match &it.kind {
            chalk_ir::VariableKind::Ty(_) => table.new_type_var().cast(Interner),
            // FIXME: maybe wrong?
            chalk_ir::VariableKind::Lifetime => table.new_type_var().cast(Interner),
            chalk_ir::VariableKind::Const(ty) => table.new_const_var(ty.clone()).cast(Interner),
        }),
    )
}

bitflags::bitflags! {
    #[derive(Default, Clone, Copy)]
    pub(crate) struct TypeVariableFlags: u8 {
        const DIVERGING = 1 << 0;
        const INTEGER = 1 << 1;
        const FLOAT = 1 << 2;
    }
}

#[derive(Clone)]
pub(crate) struct InferenceTable<'a> {
    pub(crate) db: &'a dyn HirDatabase,
    pub(crate) interner: DbInterner<'a>,
    pub(crate) trait_env: Arc<TraitEnvironment>,
    pub(crate) tait_coercion_table: Option<FxHashMap<OpaqueTyId, Ty>>,
    pub(crate) infer_ctxt: InferCtxt<'a>,
    diverging_tys: FxHashSet<Ty>,
    pending_obligations: Vec<next_solver::Goal<'a, next_solver::Predicate<'a>>>,
}

pub(crate) struct InferenceTableSnapshot<'a> {
    ctxt_snapshot: CombinedSnapshot,
    diverging_tys: FxHashSet<Ty>,
    pending_obligations: Vec<next_solver::Goal<'a, next_solver::Predicate<'a>>>,
}

impl<'a> InferenceTable<'a> {
    pub(crate) fn new(db: &'a dyn HirDatabase, trait_env: Arc<TraitEnvironment>) -> Self {
        let interner = DbInterner::new_with(db, Some(trait_env.krate), trait_env.block);
        InferenceTable {
            db,
            interner,
            trait_env,
            tait_coercion_table: None,
            infer_ctxt: interner.infer_ctxt().build(rustc_type_ir::TypingMode::Analysis {
                defining_opaque_types_and_generators: SolverDefIds::new_from_iter(interner, []),
            }),
            diverging_tys: FxHashSet::default(),
            pending_obligations: Vec::new(),
        }
    }

    /// Chalk doesn't know about the `diverging` flag, so when it unifies two
    /// type variables of which one is diverging, the chosen root might not be
    /// diverging and we have no way of marking it as such at that time. This
    /// function goes through all type variables and make sure their root is
    /// marked as diverging if necessary, so that resolving them gives the right
    /// result.
    pub(super) fn propagate_diverging_flag(&mut self) {
        let mut new_tys = FxHashSet::default();
        for ty in self.diverging_tys.iter() {
            match ty.kind(Interner) {
                TyKind::InferenceVar(var, kind) => match kind {
                    TyVariableKind::General => {
                        let root = InferenceVar::from(
                            self.infer_ctxt.root_var(TyVid::from_u32(var.index())).as_u32(),
                        );
                        if root.index() != var.index() {
                            new_tys.insert(TyKind::InferenceVar(root, *kind).intern(Interner));
                        }
                    }
                    TyVariableKind::Integer => {
                        let root = InferenceVar::from(
                            self.infer_ctxt
                                .inner
                                .borrow_mut()
                                .int_unification_table()
                                .find(IntVid::from_usize(var.index() as usize))
                                .as_u32(),
                        );
                        if root.index() != var.index() {
                            new_tys.insert(TyKind::InferenceVar(root, *kind).intern(Interner));
                        }
                    }
                    TyVariableKind::Float => {
                        let root = InferenceVar::from(
                            self.infer_ctxt
                                .inner
                                .borrow_mut()
                                .float_unification_table()
                                .find(FloatVid::from_usize(var.index() as usize))
                                .as_u32(),
                        );
                        if root.index() != var.index() {
                            new_tys.insert(TyKind::InferenceVar(root, *kind).intern(Interner));
                        }
                    }
                },
                _ => {}
            }
        }
        self.diverging_tys.extend(new_tys);
    }

    pub(super) fn set_diverging(&mut self, iv: InferenceVar, kind: TyVariableKind) {
        self.diverging_tys.insert(TyKind::InferenceVar(iv, kind).intern(Interner));
    }

    fn fallback_value(&self, iv: InferenceVar, kind: TyVariableKind) -> Ty {
        let is_diverging =
            self.diverging_tys.contains(&TyKind::InferenceVar(iv, kind).intern(Interner));
        if is_diverging {
            return TyKind::Never.intern(Interner);
        }
        match kind {
            TyVariableKind::General => TyKind::Error,
            TyVariableKind::Integer => TyKind::Scalar(Scalar::Int(IntTy::I32)),
            TyVariableKind::Float => TyKind::Scalar(Scalar::Float(FloatTy::F64)),
        }
        .intern(Interner)
    }

    pub(crate) fn canonicalize<T>(&mut self, t: T) -> rustc_type_ir::Canonical<DbInterner<'a>, T>
    where
        T: rustc_type_ir::TypeFoldable<DbInterner<'a>>,
    {
        // try to resolve obligations before canonicalizing, since this might
        // result in new knowledge about variables
        self.resolve_obligations_as_possible();
        self.infer_ctxt.canonicalize_response(t)
    }

    /// Recurses through the given type, normalizing associated types mentioned
    /// in it by replacing them by type variables and registering obligations to
    /// resolve later. This should be done once for every type we get from some
    /// type annotation (e.g. from a let type annotation, field type or function
    /// call). `make_ty` handles this already, but e.g. for field types we need
    /// to do it as well.
    pub(crate) fn normalize_associated_types_in<T, U>(&mut self, ty: T) -> T
    where
        T: ChalkToNextSolver<'a, U>,
        U: NextSolverToChalk<'a, T> + rustc_type_ir::TypeFoldable<DbInterner<'a>>,
    {
        self.normalize_associated_types_in_ns(ty.to_nextsolver(self.interner))
            .to_chalk(self.interner)
    }

    pub(crate) fn normalize_associated_types_in_ns<T>(&mut self, ty: T) -> T
    where
        T: rustc_type_ir::TypeFoldable<DbInterner<'a>>,
    {
        let ty = self.resolve_vars_with_obligations(ty);
        ty.fold_with(&mut Normalizer { table: self })
    }

    /// Works almost same as [`Self::normalize_associated_types_in`], but this also resolves shallow
    /// the inference variables
    pub(crate) fn eagerly_normalize_and_resolve_shallow_in<T>(&mut self, ty: T) -> T
    where
        T: HasInterner<Interner = Interner> + TypeFoldable<Interner>,
    {
        fn eagerly_resolve_ty<const N: usize>(
            table: &mut InferenceTable<'_>,
            ty: Ty,
            mut tys: SmallVec<[Ty; N]>,
        ) -> Ty {
            if tys.contains(&ty) {
                return ty;
            }
            tys.push(ty.clone());

            match ty.kind(Interner) {
                TyKind::Alias(AliasTy::Projection(proj_ty)) => {
                    let ty = table.normalize_projection_ty(proj_ty.clone());
                    eagerly_resolve_ty(table, ty, tys)
                }
                TyKind::InferenceVar(..) => {
                    let ty = table.resolve_ty_shallow(&ty);
                    eagerly_resolve_ty(table, ty, tys)
                }
                _ => ty,
            }
        }

        fold_tys_and_consts(
            ty,
            |e, _| match e {
                Either::Left(ty) => {
                    Either::Left(eagerly_resolve_ty::<8>(self, ty, SmallVec::new()))
                }
                Either::Right(c) => Either::Right(match &c.data(Interner).value {
                    chalk_ir::ConstValue::Concrete(cc) => match &cc.interned {
                        crate::ConstScalar::UnevaluatedConst(c_id, subst) => {
                            // FIXME: same as `normalize_associated_types_in`
                            if subst.len(Interner) == 0 {
                                if let Ok(eval) = self.db.const_eval(*c_id, subst.clone(), None) {
                                    eval
                                } else {
                                    unknown_const(c.data(Interner).ty.clone())
                                }
                            } else {
                                unknown_const(c.data(Interner).ty.clone())
                            }
                        }
                        _ => c,
                    },
                    _ => c,
                }),
            },
            DebruijnIndex::INNERMOST,
        )
    }

    pub(crate) fn normalize_projection_ty(&mut self, proj_ty: ProjectionTy) -> Ty {
        let ty = TyKind::Alias(chalk_ir::AliasTy::Projection(proj_ty))
            .intern(Interner)
            .to_nextsolver(self.interner);
        self.normalize_alias_ty(ty).to_chalk(self.interner)
    }

    pub(crate) fn normalize_alias_ty(
        &mut self,
        alias: crate::next_solver::Ty<'a>,
    ) -> crate::next_solver::Ty<'a> {
        let infer_term = self.infer_ctxt.next_ty_var();
        let obligation = crate::next_solver::Predicate::new(
            self.interner,
            crate::next_solver::Binder::dummy(crate::next_solver::PredicateKind::AliasRelate(
                alias.into(),
                infer_term.into(),
                rustc_type_ir::AliasRelationDirection::Equate,
            )),
        );
        self.register_obligation(obligation);
        self.resolve_vars_with_obligations(infer_term)
    }

    fn new_var(&mut self, kind: TyVariableKind, diverging: bool) -> Ty {
        let var = match kind {
            TyVariableKind::General => {
                let var = self.infer_ctxt.next_ty_vid();
                InferenceVar::from(var.as_u32())
            }
            TyVariableKind::Integer => {
                let var = self.infer_ctxt.next_int_vid();
                InferenceVar::from(var.as_u32())
            }
            TyVariableKind::Float => {
                let var = self.infer_ctxt.next_float_vid();
                InferenceVar::from(var.as_u32())
            }
        };

        let ty = var.to_ty(Interner, kind);
        if diverging {
            self.diverging_tys.insert(ty.clone());
        }
        ty
    }

    pub(crate) fn new_type_var(&mut self) -> Ty {
        self.new_var(TyVariableKind::General, false)
    }

    pub(crate) fn new_integer_var(&mut self) -> Ty {
        self.new_var(TyVariableKind::Integer, false)
    }

    pub(crate) fn new_float_var(&mut self) -> Ty {
        self.new_var(TyVariableKind::Float, false)
    }

    pub(crate) fn new_maybe_never_var(&mut self) -> Ty {
        self.new_var(TyVariableKind::General, true)
    }

    pub(crate) fn new_const_var(&mut self, ty: Ty) -> Const {
        let var = self.infer_ctxt.next_const_vid();
        let var = InferenceVar::from(var.as_u32());
        var.to_const(Interner, ty)
    }

    pub(crate) fn new_lifetime_var(&mut self) -> Lifetime {
        let var = self.infer_ctxt.next_region_vid();
        let var = InferenceVar::from(var.as_u32());
        var.to_lifetime(Interner)
    }

    pub(crate) fn resolve_with_fallback<T>(
        &mut self,
        t: T,
        fallback: &dyn Fn(InferenceVar, VariableKind, GenericArg, DebruijnIndex) -> GenericArg,
    ) -> T
    where
        T: HasInterner<Interner = Interner> + TypeFoldable<Interner>,
    {
        self.resolve_with_fallback_inner(t, &fallback)
    }

    pub(crate) fn fresh_subst(&mut self, binders: &[CanonicalVarKind<Interner>]) -> Substitution {
        Substitution::from_iter(
            Interner,
            binders.iter().map(|kind| match &kind.kind {
                chalk_ir::VariableKind::Ty(ty_variable_kind) => {
                    self.new_var(*ty_variable_kind, false).cast(Interner)
                }
                chalk_ir::VariableKind::Lifetime => self.new_lifetime_var().cast(Interner),
                chalk_ir::VariableKind::Const(ty) => self.new_const_var(ty.clone()).cast(Interner),
            }),
        )
    }

    pub(crate) fn instantiate_canonical<T>(&mut self, canonical: Canonical<T>) -> T
    where
        T: HasInterner<Interner = Interner> + TypeFoldable<Interner> + std::fmt::Debug,
    {
        let subst = self.fresh_subst(canonical.binders.as_slice(Interner));
        subst.apply(canonical.value, Interner)
    }

    pub(crate) fn instantiate_canonical_ns<T>(
        &mut self,
        canonical: rustc_type_ir::Canonical<DbInterner<'a>, T>,
    ) -> T
    where
        T: rustc_type_ir::TypeFoldable<DbInterner<'a>>,
    {
        self.infer_ctxt.instantiate_canonical(&canonical).0
    }

    fn resolve_with_fallback_inner<T>(
        &mut self,
        t: T,
        fallback: &dyn Fn(InferenceVar, VariableKind, GenericArg, DebruijnIndex) -> GenericArg,
    ) -> T
    where
        T: HasInterner<Interner = Interner> + TypeFoldable<Interner>,
    {
        let var_stack = &mut vec![];
        t.fold_with(
            &mut resolve::Resolver { table: self, var_stack, fallback },
            DebruijnIndex::INNERMOST,
        )
    }

    pub(crate) fn resolve_completely<T, U>(&mut self, t: T) -> T
    where
        T: HasInterner<Interner = Interner> + TypeFoldable<Interner> + ChalkToNextSolver<'a, U>,
        U: NextSolverToChalk<'a, T> + rustc_type_ir::TypeFoldable<DbInterner<'a>>,
    {
        let t = self.resolve_with_fallback(t, &|_, _, d, _| d);
        let t = self.normalize_associated_types_in(t);
        // let t = self.resolve_opaque_tys_in(t);
        // Resolve again, because maybe normalization inserted infer vars.
        self.resolve_with_fallback(t, &|_, _, d, _| d)
    }

    /// Apply a fallback to unresolved scalar types. Integer type variables and float type
    /// variables are replaced with i32 and f64, respectively.
    ///
    /// This method is only intended to be called just before returning inference results (i.e. in
    /// `InferenceContext::resolve_all()`).
    ///
    /// FIXME: This method currently doesn't apply fallback to unconstrained general type variables
    /// whereas rustc replaces them with `()` or `!`.
    pub(super) fn fallback_if_possible(&mut self) {
        let int_fallback = TyKind::Scalar(Scalar::Int(IntTy::I32)).intern(Interner);
        let float_fallback = TyKind::Scalar(Scalar::Float(FloatTy::F64)).intern(Interner);

        let int_vars = self.infer_ctxt.inner.borrow_mut().int_unification_table().len();
        for v in 0..int_vars {
            let var = InferenceVar::from(v as u32).to_ty(Interner, TyVariableKind::Integer);
            let maybe_resolved = self.resolve_ty_shallow(&var);
            if let TyKind::InferenceVar(_, kind) = maybe_resolved.kind(Interner) {
                // I don't think we can ever unify these vars with float vars, but keep this here for now
                let fallback = match kind {
                    TyVariableKind::Integer => &int_fallback,
                    TyVariableKind::Float => &float_fallback,
                    TyVariableKind::General => unreachable!(),
                };
                self.unify(&var, fallback);
            }
        }
        let float_vars = self.infer_ctxt.inner.borrow_mut().float_unification_table().len();
        for v in 0..float_vars {
            let var = InferenceVar::from(v as u32).to_ty(Interner, TyVariableKind::Float);
            let maybe_resolved = self.resolve_ty_shallow(&var);
            if let TyKind::InferenceVar(_, kind) = maybe_resolved.kind(Interner) {
                // I don't think we can ever unify these vars with float vars, but keep this here for now
                let fallback = match kind {
                    TyVariableKind::Integer => &int_fallback,
                    TyVariableKind::Float => &float_fallback,
                    TyVariableKind::General => unreachable!(),
                };
                self.unify(&var, fallback);
            }
        }
    }

    /// Unify two relatable values (e.g. `Ty`) and register new trait goals that arise from that.
    pub(crate) fn unify<T: ChalkToNextSolver<'a, U>, U: Relate<DbInterner<'a>>>(
        &mut self,
        ty1: &T,
        ty2: &T,
    ) -> bool {
        let result = match self.try_unify(ty1, ty2) {
            Ok(r) => r,
            Err(_) => return false,
        };
        self.register_infer_ok(result);
        true
    }

    /// Unify two relatable values (e.g. `Ty`) and check whether trait goals which arise from that could be fulfilled
    pub(crate) fn unify_deeply<T: ChalkToNextSolver<'a, U>, U: Relate<DbInterner<'a>>>(
        &mut self,
        ty1: &T,
        ty2: &T,
    ) -> bool {
        let result = match self.try_unify(ty1, ty2) {
            Ok(r) => r,
            Err(_) => return false,
        };
        result.goals.into_iter().all(|goal| {
            matches!(next_trait_solve_in_ctxt(&self.infer_ctxt, goal), Ok((_, Certainty::Yes)))
        })
    }

    /// Unify two relatable values (e.g. `Ty`) and return new trait goals arising from it, so the
    /// caller needs to deal with them.
    pub(crate) fn try_unify<T: ChalkToNextSolver<'a, U>, U: Relate<DbInterner<'a>>>(
        &mut self,
        t1: &T,
        t2: &T,
    ) -> InferResult<'a, ()> {
        let param_env = self.trait_env.env.to_nextsolver(self.interner);
        let lhs = t1.to_nextsolver(self.interner);
        let rhs = t2.to_nextsolver(self.interner);
        let variance = rustc_type_ir::Variance::Invariant;
        let span = crate::next_solver::Span::dummy();
        match self.infer_ctxt.relate(param_env, lhs, variance, rhs, span) {
            Ok(goals) => Ok(InferOk { goals, value: () }),
            Err(_) => Err(TypeError),
        }
    }

    /// If `ty` is a type variable with known type, returns that type;
    /// otherwise, return ty.
    #[tracing::instrument(skip(self))]
    pub(crate) fn resolve_ty_shallow(&mut self, ty: &Ty) -> Ty {
        if !ty.data(Interner).flags.intersects(chalk_ir::TypeFlags::HAS_FREE_LOCAL_NAMES) {
            return ty.clone();
        }
        self.infer_ctxt
            .resolve_vars_if_possible(ty.to_nextsolver(self.interner))
            .to_chalk(self.interner)
    }

    pub(crate) fn resolve_vars_with_obligations<T>(&mut self, t: T) -> T
    where
        T: rustc_type_ir::TypeFoldable<DbInterner<'a>>,
    {
        use rustc_type_ir::TypeVisitableExt;

        if !t.has_non_region_infer() {
            return t;
        }

        let t = self.infer_ctxt.resolve_vars_if_possible(t);

        if !t.has_non_region_infer() {
            return t;
        }

        self.resolve_obligations_as_possible();
        self.infer_ctxt.resolve_vars_if_possible(t)
    }

    pub(crate) fn structurally_resolve_type(&mut self, ty: &Ty) -> Ty {
        if let TyKind::Alias(..) = ty.kind(Interner) {
            self.structurally_normalize_ty(ty)
        } else {
            self.resolve_vars_with_obligations(ty.to_nextsolver(self.interner))
                .to_chalk(self.interner)
        }
    }

    fn structurally_normalize_ty(&mut self, ty: &Ty) -> Ty {
        self.structurally_normalize_term(ty.to_nextsolver(self.interner).into())
            .expect_ty()
            .to_chalk(self.interner)
    }

    fn structurally_normalize_term(&mut self, term: Term<'a>) -> Term<'a> {
        if term.to_alias_term().is_none() {
            return term;
        }

        let new_infer = self.infer_ctxt.next_term_var_of_kind(term);

        self.register_obligation(Predicate::new(
            self.interner,
            Binder::dummy(PredicateKind::AliasRelate(
                term,
                new_infer,
                AliasRelationDirection::Equate,
            )),
        ));
        self.resolve_obligations_as_possible();
        let res = self.infer_ctxt.resolve_vars_if_possible(new_infer);
        if res == new_infer { term } else { res }
    }

    pub(crate) fn snapshot(&mut self) -> InferenceTableSnapshot<'a> {
        let ctxt_snapshot = self.infer_ctxt.start_snapshot();
        let diverging_tys = self.diverging_tys.clone();
        let pending_obligations = self.pending_obligations.clone();
        InferenceTableSnapshot { ctxt_snapshot, pending_obligations, diverging_tys }
    }

    #[tracing::instrument(skip_all)]
    pub(crate) fn rollback_to(&mut self, snapshot: InferenceTableSnapshot<'a>) {
        self.infer_ctxt.rollback_to(snapshot.ctxt_snapshot);
        self.diverging_tys = snapshot.diverging_tys;
        self.pending_obligations = snapshot.pending_obligations;
    }

    #[tracing::instrument(skip_all)]
    pub(crate) fn run_in_snapshot<T>(&mut self, f: impl FnOnce(&mut InferenceTable<'_>) -> T) -> T {
        let snapshot = self.snapshot();
        let result = f(self);
        self.rollback_to(snapshot);
        result
    }

    pub(crate) fn commit_if_ok<T, E>(
        &mut self,
        f: impl FnOnce(&mut InferenceTable<'_>) -> Result<T, E>,
    ) -> Result<T, E> {
        let snapshot = self.snapshot();
        let result = f(self);
        match result {
            Ok(_) => {}
            Err(_) => {
                self.rollback_to(snapshot);
            }
        }
        result
    }

    /// Checks an obligation without registering it. Useful mostly to check
    /// whether a trait *might* be implemented before deciding to 'lock in' the
    /// choice (during e.g. method resolution or deref).
    #[tracing::instrument(level = "debug", skip(self))]
    pub(crate) fn try_obligation(&mut self, goal: Goal) -> NextTraitSolveResult {
        let in_env = InEnvironment::new(&self.trait_env.env, goal);
        let canonicalized = self.canonicalize(in_env.to_nextsolver(self.interner));

        next_trait_solve_canonical_in_ctxt(&self.infer_ctxt, canonicalized)
    }

    #[tracing::instrument(level = "debug", skip(self))]
    pub(crate) fn solve_obligation(&mut self, goal: Goal) -> Result<Certainty, NoSolution> {
        let goal = InEnvironment::new(&self.trait_env.env, goal);
        let goal = goal.to_nextsolver(self.interner);
        let result = next_trait_solve_in_ctxt(&self.infer_ctxt, goal);
        result.map(|m| m.1)
    }

    pub(crate) fn register_obligation(&mut self, predicate: Predicate<'a>) {
        let goal = next_solver::Goal {
            param_env: self.trait_env.env.to_nextsolver(self.interner),
            predicate,
        };
        self.register_obligation_in_env(goal)
    }

    #[tracing::instrument(level = "debug", skip(self))]
    fn register_obligation_in_env(
        &mut self,
        goal: next_solver::Goal<'a, next_solver::Predicate<'a>>,
    ) {
        let result = next_trait_solve_in_ctxt(&self.infer_ctxt, goal);
        tracing::debug!(?result);
        match result {
            Ok((_, Certainty::Yes)) => {}
            Err(rustc_type_ir::solve::NoSolution) => {}
            Ok((_, Certainty::Maybe(_))) => {
                self.pending_obligations.push(goal);
            }
        }
    }

    pub(crate) fn register_infer_ok<T>(&mut self, infer_ok: InferOk<'a, T>) {
        infer_ok.goals.into_iter().for_each(|goal| self.register_obligation_in_env(goal));
    }

    pub(crate) fn resolve_obligations_as_possible(&mut self) {
        let _span = tracing::info_span!("resolve_obligations_as_possible").entered();
        let mut changed = true;
        while mem::take(&mut changed) {
            let mut obligations = mem::take(&mut self.pending_obligations);

            for goal in obligations.drain(..) {
                tracing::debug!(obligation = ?goal);

                let result = next_trait_solve_in_ctxt(&self.infer_ctxt, goal);
                let (has_changed, certainty) = match result {
                    Ok(result) => result,
                    Err(_) => {
                        continue;
                    }
                };

                if matches!(has_changed, HasChanged::Yes) {
                    changed = true;
                }

                match certainty {
                    Certainty::Yes => {}
                    Certainty::Maybe(_) => self.pending_obligations.push(goal),
                }
            }
        }
    }

    pub(crate) fn fudge_inference<T: TypeFoldable<Interner>>(
        &mut self,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        use chalk_ir::fold::TypeFolder;

        #[derive(chalk_derive::FallibleTypeFolder)]
        #[has_interner(Interner)]
        struct VarFudger<'a, 'b> {
            table: &'a mut InferenceTable<'b>,
            highest_known_var: InferenceVar,
        }
        impl TypeFolder<Interner> for VarFudger<'_, '_> {
            fn as_dyn(&mut self) -> &mut dyn TypeFolder<Interner> {
                self
            }

            fn interner(&self) -> Interner {
                Interner
            }

            fn fold_inference_ty(
                &mut self,
                var: chalk_ir::InferenceVar,
                kind: TyVariableKind,
                _outer_binder: chalk_ir::DebruijnIndex,
            ) -> chalk_ir::Ty<Interner> {
                if var < self.highest_known_var {
                    var.to_ty(Interner, kind)
                } else {
                    self.table.new_type_var()
                }
            }

            fn fold_inference_lifetime(
                &mut self,
                var: chalk_ir::InferenceVar,
                _outer_binder: chalk_ir::DebruijnIndex,
            ) -> chalk_ir::Lifetime<Interner> {
                if var < self.highest_known_var {
                    var.to_lifetime(Interner)
                } else {
                    self.table.new_lifetime_var()
                }
            }

            fn fold_inference_const(
                &mut self,
                ty: chalk_ir::Ty<Interner>,
                var: chalk_ir::InferenceVar,
                _outer_binder: chalk_ir::DebruijnIndex,
            ) -> chalk_ir::Const<Interner> {
                if var < self.highest_known_var {
                    var.to_const(Interner, ty)
                } else {
                    self.table.new_const_var(ty)
                }
            }
        }

        let snapshot = self.snapshot();
        let highest_known_var = self.new_type_var().inference_var(Interner).expect("inference_var");
        let result = f(self);
        self.rollback_to(snapshot);
        result
            .fold_with(&mut VarFudger { table: self, highest_known_var }, DebruijnIndex::INNERMOST)
    }

    pub(crate) fn callable_sig(
        &mut self,
        ty: &Ty,
        num_args: usize,
    ) -> Option<(Option<FnTrait>, Vec<Ty>, Ty)> {
        match ty.callable_sig(self.db) {
            Some(sig) => Some((None, sig.params().to_vec(), sig.ret().clone())),
            None => {
                let (f, args_ty, return_ty) = self.callable_sig_from_fn_trait(ty, num_args)?;
                Some((Some(f), args_ty, return_ty))
            }
        }
    }

    fn callable_sig_from_fn_trait(
        &mut self,
        ty: &Ty,
        num_args: usize,
    ) -> Option<(FnTrait, Vec<Ty>, Ty)> {
        for (fn_trait_name, output_assoc_name, subtraits) in [
            (FnTrait::FnOnce, sym::Output, &[FnTrait::Fn, FnTrait::FnMut][..]),
            (FnTrait::AsyncFnMut, sym::CallRefFuture, &[FnTrait::AsyncFn]),
            (FnTrait::AsyncFnOnce, sym::CallOnceFuture, &[]),
        ] {
            let krate = self.trait_env.krate;
            let fn_trait = fn_trait_name.get_id(self.db, krate)?;
            let trait_data = fn_trait.trait_items(self.db);
            let output_assoc_type =
                trait_data.associated_type_by_name(&Name::new_symbol_root(output_assoc_name))?;

            let mut arg_tys = Vec::with_capacity(num_args);
            let arg_ty = TyBuilder::tuple(num_args)
                .fill(|it| {
                    let arg = match it {
                        ParamKind::Type => self.new_type_var(),
                        ParamKind::Lifetime => unreachable!("Tuple with lifetime parameter"),
                        ParamKind::Const(_) => unreachable!("Tuple with const parameter"),
                    };
                    arg_tys.push(arg.clone());
                    arg.cast(Interner)
                })
                .build();

            let b = TyBuilder::trait_ref(self.db, fn_trait);
            if b.remaining() != 2 {
                return None;
            }
            let mut trait_ref = b.push(ty.clone()).push(arg_ty).build();

            let projection = TyBuilder::assoc_type_projection(
                self.db,
                output_assoc_type,
                Some(trait_ref.substitution.clone()),
            )
            .fill_with_unknown()
            .build();

            let goal: Goal = trait_ref.clone().cast(Interner);
            if !self.try_obligation(goal.clone()).no_solution() {
                self.register_obligation(goal.to_nextsolver(self.interner));
                let return_ty = self.normalize_projection_ty(projection);
                for &fn_x in subtraits {
                    let fn_x_trait = fn_x.get_id(self.db, krate)?;
                    trait_ref.trait_id = to_chalk_trait_id(fn_x_trait);
                    let goal = trait_ref.clone().cast(Interner);
                    if !self.try_obligation(goal).no_solution() {
                        return Some((fn_x, arg_tys, return_ty));
                    }
                }
                return Some((fn_trait_name, arg_tys, return_ty));
            }
        }
        None
    }

    pub(super) fn insert_type_vars<T>(&mut self, ty: T) -> T
    where
        T: HasInterner<Interner = Interner> + TypeFoldable<Interner>,
    {
        fold_generic_args(
            ty,
            |arg, _| match arg {
                GenericArgData::Ty(ty) => GenericArgData::Ty(self.insert_type_vars_shallow(ty)),
                // FIXME: insert lifetime vars once LifetimeData::InferenceVar
                // and specific error variant for lifetimes start being constructed
                GenericArgData::Lifetime(lt) => GenericArgData::Lifetime(lt),
                GenericArgData::Const(c) => {
                    GenericArgData::Const(self.insert_const_vars_shallow(c))
                }
            },
            DebruijnIndex::INNERMOST,
        )
    }

    /// Replaces `Ty::Error` by a new type var, so we can maybe still infer it.
    pub(super) fn insert_type_vars_shallow(&mut self, ty: Ty) -> Ty {
        match ty.kind(Interner) {
            TyKind::Error => self.new_type_var(),
            TyKind::InferenceVar(..) => {
                let ty_resolved = self.structurally_resolve_type(&ty);
                if ty_resolved.is_unknown() { self.new_type_var() } else { ty }
            }
            _ => ty,
        }
    }

    /// Whenever you lower a user-written type, you should call this.
    pub(crate) fn process_user_written_ty<T, U>(&mut self, ty: T) -> T
    where
        T: HasInterner<Interner = Interner> + TypeFoldable<Interner> + ChalkToNextSolver<'a, U>,
        U: NextSolverToChalk<'a, T> + rustc_type_ir::TypeFoldable<DbInterner<'a>>,
    {
        self.process_remote_user_written_ty(ty)
        // FIXME: Register a well-formed obligation.
    }

    /// The difference of this method from `process_user_written_ty()` is that this method doesn't register a well-formed obligation,
    /// while `process_user_written_ty()` should (but doesn't currently).
    pub(crate) fn process_remote_user_written_ty<T, U>(&mut self, ty: T) -> T
    where
        T: HasInterner<Interner = Interner> + TypeFoldable<Interner> + ChalkToNextSolver<'a, U>,
        U: NextSolverToChalk<'a, T> + rustc_type_ir::TypeFoldable<DbInterner<'a>>,
    {
        let ty = self.insert_type_vars(ty);
        // See https://github.com/rust-lang/rust/blob/cdb45c87e2cd43495379f7e867e3cc15dcee9f93/compiler/rustc_hir_typeck/src/fn_ctxt/mod.rs#L487-L495:
        // Even though the new solver only lazily normalizes usually, here we eagerly normalize so that not everything needs
        // to normalize before inspecting the `TyKind`.
        self.normalize_associated_types_in(ty)
    }

    /// Replaces ConstScalar::Unknown by a new type var, so we can maybe still infer it.
    pub(super) fn insert_const_vars_shallow(&mut self, c: Const) -> Const {
        let data = c.data(Interner);
        match &data.value {
            ConstValue::Concrete(cc) => match &cc.interned {
                crate::ConstScalar::Unknown => self.new_const_var(data.ty.clone()),
                // try to evaluate unevaluated const. Replace with new var if const eval failed.
                crate::ConstScalar::UnevaluatedConst(id, subst) => {
                    if let Ok(eval) = self.db.const_eval(*id, subst.clone(), None) {
                        eval
                    } else {
                        self.new_const_var(data.ty.clone())
                    }
                }
                _ => c,
            },
            _ => c,
        }
    }

    /// Check if given type is `Sized` or not
    pub(crate) fn is_sized(&mut self, ty: &Ty) -> bool {
        fn short_circuit_trivial_tys(ty: &Ty) -> Option<bool> {
            match ty.kind(Interner) {
                TyKind::Scalar(..)
                | TyKind::Ref(..)
                | TyKind::Raw(..)
                | TyKind::Never
                | TyKind::FnDef(..)
                | TyKind::Array(..)
                | TyKind::Function(..) => Some(true),
                TyKind::Slice(..) | TyKind::Str | TyKind::Dyn(..) => Some(false),
                _ => None,
            }
        }

        let mut ty = ty.clone();
        ty = self.eagerly_normalize_and_resolve_shallow_in(ty);
        if let Some(sized) = short_circuit_trivial_tys(&ty) {
            return sized;
        }

        {
            let mut structs = SmallVec::<[_; 8]>::new();
            // Must use a loop here and not recursion because otherwise users will conduct completely
            // artificial examples of structs that have themselves as the tail field and complain r-a crashes.
            while let Some((AdtId::StructId(id), subst)) = ty.as_adt() {
                let struct_data = id.fields(self.db);
                if let Some((last_field, _)) = struct_data.fields().iter().next_back() {
                    let last_field_ty = self.db.field_types(id.into())[last_field]
                        .clone()
                        .substitute(Interner, subst);
                    if structs.contains(&ty) {
                        // A struct recursively contains itself as a tail field somewhere.
                        return true; // Don't overload the users with too many errors.
                    }
                    structs.push(ty);
                    // Structs can have DST as its last field and such cases are not handled
                    // as unsized by the chalk, so we do this manually.
                    ty = last_field_ty;
                    ty = self.eagerly_normalize_and_resolve_shallow_in(ty);
                    if let Some(sized) = short_circuit_trivial_tys(&ty) {
                        return sized;
                    }
                } else {
                    break;
                };
            }
        }

        let Some(sized) = LangItem::Sized.resolve_trait(self.db, self.trait_env.krate) else {
            return false;
        };
        let sized_pred = WhereClause::Implemented(TraitRef {
            trait_id: to_chalk_trait_id(sized),
            substitution: Substitution::from1(Interner, ty),
        });
        let goal = GoalData::DomainGoal(chalk_ir::DomainGoal::Holds(sized_pred)).intern(Interner);
        self.try_obligation(goal).certain()
    }
}

impl fmt::Debug for InferenceTable<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InferenceTable").finish()
    }
}

mod resolve {
    use super::InferenceTable;
    use crate::{
        ConcreteConst, Const, ConstData, ConstScalar, ConstValue, DebruijnIndex, GenericArg,
        InferenceVar, Interner, Lifetime, Ty, TyVariableKind, VariableKind,
        next_solver::mapping::NextSolverToChalk,
    };
    use chalk_ir::{
        cast::Cast,
        fold::{TypeFoldable, TypeFolder},
    };
    use rustc_type_ir::{FloatVid, IntVid, TyVid};

    #[derive(Copy, Clone, PartialEq, Eq)]
    pub(super) enum VarKind {
        Ty(TyVariableKind),
        Const,
    }

    #[derive(chalk_derive::FallibleTypeFolder)]
    #[has_interner(Interner)]
    pub(super) struct Resolver<
        'a,
        'b,
        F: Fn(InferenceVar, VariableKind, GenericArg, DebruijnIndex) -> GenericArg,
    > {
        pub(super) table: &'a mut InferenceTable<'b>,
        pub(super) var_stack: &'a mut Vec<(InferenceVar, VarKind)>,
        pub(super) fallback: F,
    }
    impl<F> TypeFolder<Interner> for Resolver<'_, '_, F>
    where
        F: Fn(InferenceVar, VariableKind, GenericArg, DebruijnIndex) -> GenericArg,
    {
        fn as_dyn(&mut self) -> &mut dyn TypeFolder<Interner> {
            self
        }

        fn interner(&self) -> Interner {
            Interner
        }

        fn fold_inference_ty(
            &mut self,
            var: InferenceVar,
            kind: TyVariableKind,
            outer_binder: DebruijnIndex,
        ) -> Ty {
            match kind {
                TyVariableKind::General => {
                    let vid = self.table.infer_ctxt.root_var(TyVid::from(var.index()));
                    let var = InferenceVar::from(vid.as_u32());
                    if self.var_stack.contains(&(var, VarKind::Ty(kind))) {
                        // recursive type
                        let default = self.table.fallback_value(var, kind).cast(Interner);
                        return (self.fallback)(var, VariableKind::Ty(kind), default, outer_binder)
                            .assert_ty_ref(Interner)
                            .clone();
                    }
                    if let Ok(known_ty) = self.table.infer_ctxt.probe_ty_var(vid) {
                        let known_ty: Ty = known_ty.to_chalk(self.table.interner);
                        // known_ty may contain other variables that are known by now
                        self.var_stack.push((var, VarKind::Ty(kind)));
                        let result = known_ty.fold_with(self, outer_binder);
                        self.var_stack.pop();
                        result
                    } else {
                        let default = self.table.fallback_value(var, kind).cast(Interner);
                        (self.fallback)(var, VariableKind::Ty(kind), default, outer_binder)
                            .assert_ty_ref(Interner)
                            .clone()
                    }
                }
                TyVariableKind::Integer => {
                    let vid = self
                        .table
                        .infer_ctxt
                        .inner
                        .borrow_mut()
                        .int_unification_table()
                        .find(IntVid::from(var.index()));
                    let var = InferenceVar::from(vid.as_u32());
                    if self.var_stack.contains(&(var, VarKind::Ty(kind))) {
                        // recursive type
                        let default = self.table.fallback_value(var, kind).cast(Interner);
                        return (self.fallback)(var, VariableKind::Ty(kind), default, outer_binder)
                            .assert_ty_ref(Interner)
                            .clone();
                    }
                    if let Some(known_ty) = self.table.infer_ctxt.resolve_int_var(vid) {
                        let known_ty: Ty = known_ty.to_chalk(self.table.interner);
                        // known_ty may contain other variables that are known by now
                        self.var_stack.push((var, VarKind::Ty(kind)));
                        let result = known_ty.fold_with(self, outer_binder);
                        self.var_stack.pop();
                        result
                    } else {
                        let default = self.table.fallback_value(var, kind).cast(Interner);
                        (self.fallback)(var, VariableKind::Ty(kind), default, outer_binder)
                            .assert_ty_ref(Interner)
                            .clone()
                    }
                }
                TyVariableKind::Float => {
                    let vid = self
                        .table
                        .infer_ctxt
                        .inner
                        .borrow_mut()
                        .float_unification_table()
                        .find(FloatVid::from(var.index()));
                    let var = InferenceVar::from(vid.as_u32());
                    if self.var_stack.contains(&(var, VarKind::Ty(kind))) {
                        // recursive type
                        let default = self.table.fallback_value(var, kind).cast(Interner);
                        return (self.fallback)(var, VariableKind::Ty(kind), default, outer_binder)
                            .assert_ty_ref(Interner)
                            .clone();
                    }
                    if let Some(known_ty) = self.table.infer_ctxt.resolve_float_var(vid) {
                        let known_ty: Ty = known_ty.to_chalk(self.table.interner);
                        // known_ty may contain other variables that are known by now
                        self.var_stack.push((var, VarKind::Ty(kind)));
                        let result = known_ty.fold_with(self, outer_binder);
                        self.var_stack.pop();
                        result
                    } else {
                        let default = self.table.fallback_value(var, kind).cast(Interner);
                        (self.fallback)(var, VariableKind::Ty(kind), default, outer_binder)
                            .assert_ty_ref(Interner)
                            .clone()
                    }
                }
            }
        }

        fn fold_inference_const(
            &mut self,
            ty: Ty,
            var: InferenceVar,
            outer_binder: DebruijnIndex,
        ) -> Const {
            let vid = self
                .table
                .infer_ctxt
                .root_const_var(rustc_type_ir::ConstVid::from_u32(var.index()));
            let var = InferenceVar::from(vid.as_u32());
            let default = ConstData {
                ty: ty.clone(),
                value: ConstValue::Concrete(ConcreteConst { interned: ConstScalar::Unknown }),
            }
            .intern(Interner)
            .cast(Interner);
            if self.var_stack.contains(&(var, VarKind::Const)) {
                // recursive
                return (self.fallback)(var, VariableKind::Const(ty), default, outer_binder)
                    .assert_const_ref(Interner)
                    .clone();
            }
            if let Ok(known_const) = self.table.infer_ctxt.probe_const_var(vid) {
                let known_const: Const = known_const.to_chalk(self.table.interner);
                // known_ty may contain other variables that are known by now
                self.var_stack.push((var, VarKind::Const));
                let result = known_const.fold_with(self, outer_binder);
                self.var_stack.pop();
                result
            } else {
                (self.fallback)(var, VariableKind::Const(ty), default, outer_binder)
                    .assert_const_ref(Interner)
                    .clone()
            }
        }

        fn fold_inference_lifetime(
            &mut self,
            _var: InferenceVar,
            _outer_binder: DebruijnIndex,
        ) -> Lifetime {
            // fall back all lifetimes to 'error -- currently we don't deal
            // with any lifetimes, but we can sometimes get some lifetime
            // variables through Chalk's unification, and this at least makes
            // sure we don't leak them outside of inference
            crate::error_lifetime()
        }
    }
}

/// This expects its input to be resolved.
struct Normalizer<'a, 'b> {
    table: &'a mut InferenceTable<'b>,
}

impl<'db> Normalizer<'_, 'db> {
    fn normalize_alias_term(
        &mut self,
        alias_term: crate::next_solver::Term<'db>,
    ) -> crate::next_solver::Term<'db> {
        let infer_term = self.table.infer_ctxt.next_term_var_of_kind(alias_term);
        let obligation = crate::next_solver::Predicate::new(
            self.table.interner,
            crate::next_solver::Binder::dummy(crate::next_solver::PredicateKind::AliasRelate(
                alias_term,
                infer_term,
                rustc_type_ir::AliasRelationDirection::Equate,
            )),
        );
        self.table.register_obligation(obligation);
        let term = self.table.resolve_vars_with_obligations(infer_term);
        // Now normalize the result, because maybe it contains more aliases.
        match term {
            Term::Ty(term) => term.super_fold_with(self).into(),
            Term::Const(term) => term.super_fold_with(self).into(),
        }
    }
}

impl<'db> rustc_type_ir::TypeFolder<DbInterner<'db>> for Normalizer<'_, 'db> {
    fn cx(&self) -> DbInterner<'db> {
        self.table.interner
    }

    fn fold_ty(&mut self, ty: crate::next_solver::Ty<'db>) -> crate::next_solver::Ty<'db> {
        if !ty.has_aliases() {
            return ty;
        }

        let crate::next_solver::TyKind::Alias(..) = ty.kind() else {
            return ty.super_fold_with(self);
        };
        // FIXME: Handle escaping bound vars by replacing them with placeholders (relevant to when we handle HRTB only).
        self.normalize_alias_term(ty.into()).expect_type()
    }

    fn fold_const(&mut self, ct: crate::next_solver::Const<'db>) -> crate::next_solver::Const<'db> {
        if !ct.has_aliases() {
            return ct;
        }

        let crate::next_solver::ConstKind::Unevaluated(..) = ct.kind() else {
            return ct.super_fold_with(self);
        };
        // FIXME: Handle escaping bound vars by replacing them with placeholders (relevant to when we handle HRTB only).
        self.normalize_alias_term(ct.into()).expect_const()
    }
}
