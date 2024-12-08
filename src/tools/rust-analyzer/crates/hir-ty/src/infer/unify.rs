//! Unification and canonicalization logic.

use std::{fmt, iter, mem};

use chalk_ir::{
    cast::Cast, fold::TypeFoldable, interner::HasInterner, zip::Zip, CanonicalVarKind, FloatTy,
    IntTy, TyVariableKind, UniverseIndex,
};
use chalk_solve::infer::ParameterEnaVariableExt;
use either::Either;
use ena::unify::UnifyKey;
use hir_def::{lang_item::LangItem, AdtId};
use hir_expand::name::Name;
use intern::sym;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use triomphe::Arc;

use super::{InferOk, InferResult, InferenceContext, TypeError};
use crate::{
    consteval::unknown_const, db::HirDatabase, fold_generic_args, fold_tys_and_consts,
    to_chalk_trait_id, traits::FnTrait, AliasEq, AliasTy, BoundVar, Canonical, Const, ConstValue,
    DebruijnIndex, DomainGoal, GenericArg, GenericArgData, Goal, GoalData, Guidance, InEnvironment,
    InferenceVar, Interner, Lifetime, OpaqueTyId, ParamKind, ProjectionTy, ProjectionTyExt, Scalar,
    Solution, Substitution, TraitEnvironment, TraitRef, Ty, TyBuilder, TyExt, TyKind, VariableKind,
    WhereClause,
};

impl InferenceContext<'_> {
    pub(super) fn canonicalize<T>(&mut self, t: T) -> Canonical<T>
    where
        T: TypeFoldable<Interner> + HasInterner<Interner = Interner>,
    {
        self.table.canonicalize(t)
    }

    pub(super) fn clauses_for_self_ty(
        &mut self,
        self_ty: InferenceVar,
    ) -> SmallVec<[WhereClause; 4]> {
        self.table.resolve_obligations_as_possible();

        let root = self.table.var_unification_table.inference_var_root(self_ty);
        let pending_obligations = mem::take(&mut self.table.pending_obligations);
        let obligations = pending_obligations
            .iter()
            .filter_map(|obligation| match obligation.value.value.goal.data(Interner) {
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

                    let uncanonical =
                        chalk_ir::Substitute::apply(&obligation.free_vars, ty, Interner);
                    if matches!(
                        self.resolve_ty_shallow(&uncanonical).kind(Interner),
                        TyKind::InferenceVar(iv, TyVariableKind::General) if *iv == root,
                    ) {
                        Some(chalk_ir::Substitute::apply(
                            &obligation.free_vars,
                            clause.clone(),
                            Interner,
                        ))
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

#[derive(Debug, Clone)]
pub(crate) struct Canonicalized<T>
where
    T: HasInterner<Interner = Interner>,
{
    pub(crate) value: Canonical<T>,
    free_vars: Vec<GenericArg>,
}

impl<T: HasInterner<Interner = Interner>> Canonicalized<T> {
    pub(crate) fn apply_solution(
        &self,
        ctx: &mut InferenceTable<'_>,
        solution: Canonical<Substitution>,
    ) {
        // the solution may contain new variables, which we need to convert to new inference vars
        let new_vars = Substitution::from_iter(
            Interner,
            solution.binders.iter(Interner).map(|k| match &k.kind {
                VariableKind::Ty(TyVariableKind::General) => ctx.new_type_var().cast(Interner),
                VariableKind::Ty(TyVariableKind::Integer) => ctx.new_integer_var().cast(Interner),
                VariableKind::Ty(TyVariableKind::Float) => ctx.new_float_var().cast(Interner),
                // Chalk can sometimes return new lifetime variables. We just replace them by errors
                // for now.
                VariableKind::Lifetime => ctx.new_lifetime_var().cast(Interner),
                VariableKind::Const(ty) => ctx.new_const_var(ty.clone()).cast(Interner),
            }),
        );
        for (i, v) in solution.value.iter(Interner).enumerate() {
            let var = &self.free_vars[i];
            if let Some(ty) = v.ty(Interner) {
                // eagerly replace projections in the type; we may be getting types
                // e.g. from where clauses where this hasn't happened yet
                let ty = ctx.normalize_associated_types_in(new_vars.apply(ty.clone(), Interner));
                ctx.unify(var.assert_ty_ref(Interner), &ty);
            } else {
                let _ = ctx.try_unify(var, &new_vars.apply(v.clone(), Interner));
            }
        }
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

type ChalkInferenceTable = chalk_solve::infer::InferenceTable<Interner>;

#[derive(Clone)]
pub(crate) struct InferenceTable<'a> {
    pub(crate) db: &'a dyn HirDatabase,
    pub(crate) trait_env: Arc<TraitEnvironment>,
    pub(crate) tait_coercion_table: Option<FxHashMap<OpaqueTyId, Ty>>,
    var_unification_table: ChalkInferenceTable,
    type_variable_table: SmallVec<[TypeVariableFlags; 16]>,
    pending_obligations: Vec<Canonicalized<InEnvironment<Goal>>>,
    /// Double buffer used in [`Self::resolve_obligations_as_possible`] to cut down on
    /// temporary allocations.
    resolve_obligations_buffer: Vec<Canonicalized<InEnvironment<Goal>>>,
}

pub(crate) struct InferenceTableSnapshot {
    var_table_snapshot: chalk_solve::infer::InferenceSnapshot<Interner>,
    type_variable_table: SmallVec<[TypeVariableFlags; 16]>,
    pending_obligations: Vec<Canonicalized<InEnvironment<Goal>>>,
}

impl<'a> InferenceTable<'a> {
    pub(crate) fn new(db: &'a dyn HirDatabase, trait_env: Arc<TraitEnvironment>) -> Self {
        InferenceTable {
            db,
            trait_env,
            tait_coercion_table: None,
            var_unification_table: ChalkInferenceTable::new(),
            type_variable_table: SmallVec::new(),
            pending_obligations: Vec::new(),
            resolve_obligations_buffer: Vec::new(),
        }
    }

    /// Chalk doesn't know about the `diverging` flag, so when it unifies two
    /// type variables of which one is diverging, the chosen root might not be
    /// diverging and we have no way of marking it as such at that time. This
    /// function goes through all type variables and make sure their root is
    /// marked as diverging if necessary, so that resolving them gives the right
    /// result.
    pub(super) fn propagate_diverging_flag(&mut self) {
        for i in 0..self.type_variable_table.len() {
            if !self.type_variable_table[i].contains(TypeVariableFlags::DIVERGING) {
                continue;
            }
            let v = InferenceVar::from(i as u32);
            let root = self.var_unification_table.inference_var_root(v);
            self.modify_type_variable_flag(root, |f| {
                *f |= TypeVariableFlags::DIVERGING;
            });
        }
    }

    pub(super) fn set_diverging(&mut self, iv: InferenceVar, diverging: bool) {
        self.modify_type_variable_flag(iv, |f| {
            f.set(TypeVariableFlags::DIVERGING, diverging);
        });
    }

    fn fallback_value(&self, iv: InferenceVar, kind: TyVariableKind) -> Ty {
        let is_diverging = self
            .type_variable_table
            .get(iv.index() as usize)
            .map_or(false, |data| data.contains(TypeVariableFlags::DIVERGING));
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

    pub(crate) fn canonicalize_with_free_vars<T>(&mut self, t: T) -> Canonicalized<T>
    where
        T: TypeFoldable<Interner> + HasInterner<Interner = Interner>,
    {
        // try to resolve obligations before canonicalizing, since this might
        // result in new knowledge about variables
        self.resolve_obligations_as_possible();
        let result = self.var_unification_table.canonicalize(Interner, t);
        let free_vars = result
            .free_vars
            .into_iter()
            .map(|free_var| free_var.to_generic_arg(Interner))
            .collect();
        Canonicalized { value: result.quantified, free_vars }
    }

    pub(crate) fn canonicalize<T>(&mut self, t: T) -> Canonical<T>
    where
        T: TypeFoldable<Interner> + HasInterner<Interner = Interner>,
    {
        // try to resolve obligations before canonicalizing, since this might
        // result in new knowledge about variables
        self.resolve_obligations_as_possible();
        self.var_unification_table.canonicalize(Interner, t).quantified
    }

    /// Recurses through the given type, normalizing associated types mentioned
    /// in it by replacing them by type variables and registering obligations to
    /// resolve later. This should be done once for every type we get from some
    /// type annotation (e.g. from a let type annotation, field type or function
    /// call). `make_ty` handles this already, but e.g. for field types we need
    /// to do it as well.
    pub(crate) fn normalize_associated_types_in<T>(&mut self, ty: T) -> T
    where
        T: HasInterner<Interner = Interner> + TypeFoldable<Interner>,
    {
        fold_tys_and_consts(
            ty,
            |e, _| match e {
                Either::Left(ty) => Either::Left(match ty.kind(Interner) {
                    TyKind::Alias(AliasTy::Projection(proj_ty)) => {
                        self.normalize_projection_ty(proj_ty.clone())
                    }
                    _ => ty,
                }),
                Either::Right(c) => Either::Right(match &c.data(Interner).value {
                    chalk_ir::ConstValue::Concrete(cc) => match &cc.interned {
                        crate::ConstScalar::UnevaluatedConst(c_id, subst) => {
                            // FIXME: Ideally here we should do everything that we do with type alias, i.e. adding a variable
                            // and registering an obligation. But it needs chalk support, so we handle the most basic
                            // case (a non associated const without generic parameters) manually.
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
        let var = self.new_type_var();
        let alias_eq = AliasEq { alias: AliasTy::Projection(proj_ty), ty: var.clone() };
        let obligation = alias_eq.cast(Interner);
        self.register_obligation(obligation);
        var
    }

    fn modify_type_variable_flag<F>(&mut self, var: InferenceVar, cb: F)
    where
        F: FnOnce(&mut TypeVariableFlags),
    {
        let idx = var.index() as usize;
        if self.type_variable_table.len() <= idx {
            self.extend_type_variable_table(idx);
        }
        if let Some(f) = self.type_variable_table.get_mut(idx) {
            cb(f);
        }
    }
    fn extend_type_variable_table(&mut self, to_index: usize) {
        let count = to_index - self.type_variable_table.len() + 1;
        self.type_variable_table.extend(iter::repeat(TypeVariableFlags::default()).take(count));
    }

    fn new_var(&mut self, kind: TyVariableKind, diverging: bool) -> Ty {
        let var = self.var_unification_table.new_variable(UniverseIndex::ROOT);
        // Chalk might have created some type variables for its own purposes that we don't know about...
        self.extend_type_variable_table(var.index() as usize);
        assert_eq!(var.index() as usize, self.type_variable_table.len() - 1);
        let flags = self.type_variable_table.get_mut(var.index() as usize).unwrap();
        if diverging {
            *flags |= TypeVariableFlags::DIVERGING;
        }
        if matches!(kind, TyVariableKind::Integer) {
            *flags |= TypeVariableFlags::INTEGER;
        } else if matches!(kind, TyVariableKind::Float) {
            *flags |= TypeVariableFlags::FLOAT;
        }
        var.to_ty_with_kind(Interner, kind)
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
        let var = self.var_unification_table.new_variable(UniverseIndex::ROOT);
        var.to_const(Interner, ty)
    }

    pub(crate) fn new_lifetime_var(&mut self) -> Lifetime {
        let var = self.var_unification_table.new_variable(UniverseIndex::ROOT);
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
        self.resolve_with_fallback_inner(&mut Vec::new(), t, &fallback)
    }

    pub(crate) fn fresh_subst(&mut self, binders: &[CanonicalVarKind<Interner>]) -> Substitution {
        Substitution::from_iter(
            Interner,
            binders.iter().map(|kind| {
                let param_infer_var =
                    kind.map_ref(|&ui| self.var_unification_table.new_variable(ui));
                param_infer_var.to_generic_arg(Interner)
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

    fn resolve_with_fallback_inner<T>(
        &mut self,
        var_stack: &mut Vec<InferenceVar>,
        t: T,
        fallback: &dyn Fn(InferenceVar, VariableKind, GenericArg, DebruijnIndex) -> GenericArg,
    ) -> T
    where
        T: HasInterner<Interner = Interner> + TypeFoldable<Interner>,
    {
        t.fold_with(
            &mut resolve::Resolver { table: self, var_stack, fallback },
            DebruijnIndex::INNERMOST,
        )
    }

    pub(crate) fn resolve_completely<T>(&mut self, t: T) -> T
    where
        T: HasInterner<Interner = Interner> + TypeFoldable<Interner>,
    {
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

        let scalar_vars: Vec<_> = self
            .type_variable_table
            .iter()
            .enumerate()
            .filter_map(|(index, flags)| {
                let kind = if flags.contains(TypeVariableFlags::INTEGER) {
                    TyVariableKind::Integer
                } else if flags.contains(TypeVariableFlags::FLOAT) {
                    TyVariableKind::Float
                } else {
                    return None;
                };

                // FIXME: This is not really the nicest way to get `InferenceVar`s. Can we get them
                // without directly constructing them from `index`?
                let var = InferenceVar::from(index as u32).to_ty(Interner, kind);
                Some(var)
            })
            .collect();

        for var in scalar_vars {
            let maybe_resolved = self.resolve_ty_shallow(&var);
            if let TyKind::InferenceVar(_, kind) = maybe_resolved.kind(Interner) {
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
    #[tracing::instrument(skip_all)]
    pub(crate) fn unify<T: ?Sized + Zip<Interner>>(&mut self, ty1: &T, ty2: &T) -> bool {
        let result = match self.try_unify(ty1, ty2) {
            Ok(r) => r,
            Err(_) => return false,
        };
        self.register_infer_ok(result);
        true
    }

    /// Unify two relatable values (e.g. `Ty`) and check whether trait goals which arise from that could be fulfilled
    pub(crate) fn unify_deeply<T: ?Sized + Zip<Interner>>(&mut self, ty1: &T, ty2: &T) -> bool {
        let result = match self.try_unify(ty1, ty2) {
            Ok(r) => r,
            Err(_) => return false,
        };
        result.goals.iter().all(|goal| {
            let canonicalized = self.canonicalize_with_free_vars(goal.clone());
            self.try_resolve_obligation(&canonicalized).is_some()
        })
    }

    /// Unify two relatable values (e.g. `Ty`) and return new trait goals arising from it, so the
    /// caller needs to deal with them.
    pub(crate) fn try_unify<T: ?Sized + Zip<Interner>>(
        &mut self,
        t1: &T,
        t2: &T,
    ) -> InferResult<()> {
        match self.var_unification_table.relate(
            Interner,
            &self.db,
            &self.trait_env.env,
            chalk_ir::Variance::Invariant,
            t1,
            t2,
        ) {
            Ok(result) => Ok(InferOk { goals: result.goals, value: () }),
            Err(chalk_ir::NoSolution) => Err(TypeError),
        }
    }

    /// If `ty` is a type variable with known type, returns that type;
    /// otherwise, return ty.
    pub(crate) fn resolve_ty_shallow(&mut self, ty: &Ty) -> Ty {
        self.resolve_obligations_as_possible();
        self.var_unification_table.normalize_ty_shallow(Interner, ty).unwrap_or_else(|| ty.clone())
    }

    pub(crate) fn snapshot(&mut self) -> InferenceTableSnapshot {
        let var_table_snapshot = self.var_unification_table.snapshot();
        let type_variable_table = self.type_variable_table.clone();
        let pending_obligations = self.pending_obligations.clone();
        InferenceTableSnapshot { var_table_snapshot, pending_obligations, type_variable_table }
    }

    #[tracing::instrument(skip_all)]
    pub(crate) fn rollback_to(&mut self, snapshot: InferenceTableSnapshot) {
        self.var_unification_table.rollback_to(snapshot.var_table_snapshot);
        self.type_variable_table = snapshot.type_variable_table;
        self.pending_obligations = snapshot.pending_obligations;
    }

    #[tracing::instrument(skip_all)]
    pub(crate) fn run_in_snapshot<T>(&mut self, f: impl FnOnce(&mut InferenceTable<'_>) -> T) -> T {
        let snapshot = self.snapshot();
        let result = f(self);
        self.rollback_to(snapshot);
        result
    }

    /// Checks an obligation without registering it. Useful mostly to check
    /// whether a trait *might* be implemented before deciding to 'lock in' the
    /// choice (during e.g. method resolution or deref).
    pub(crate) fn try_obligation(&mut self, goal: Goal) -> Option<Solution> {
        let in_env = InEnvironment::new(&self.trait_env.env, goal);
        let canonicalized = self.canonicalize(in_env);

        self.db.trait_solve(self.trait_env.krate, self.trait_env.block, canonicalized)
    }

    pub(crate) fn register_obligation(&mut self, goal: Goal) {
        let in_env = InEnvironment::new(&self.trait_env.env, goal);
        self.register_obligation_in_env(in_env)
    }

    fn register_obligation_in_env(&mut self, goal: InEnvironment<Goal>) {
        let canonicalized = self.canonicalize_with_free_vars(goal);
        let solution = self.try_resolve_obligation(&canonicalized);
        if matches!(solution, Some(Solution::Ambig(_))) {
            self.pending_obligations.push(canonicalized);
        }
    }

    pub(crate) fn register_infer_ok<T>(&mut self, infer_ok: InferOk<T>) {
        infer_ok.goals.into_iter().for_each(|goal| self.register_obligation_in_env(goal));
    }

    pub(crate) fn resolve_obligations_as_possible(&mut self) {
        let _span = tracing::info_span!("resolve_obligations_as_possible").entered();
        let mut changed = true;
        let mut obligations = mem::take(&mut self.resolve_obligations_buffer);
        while mem::take(&mut changed) {
            mem::swap(&mut self.pending_obligations, &mut obligations);

            for canonicalized in obligations.drain(..) {
                if !self.check_changed(&canonicalized) {
                    self.pending_obligations.push(canonicalized);
                    continue;
                }
                changed = true;
                let uncanonical = chalk_ir::Substitute::apply(
                    &canonicalized.free_vars,
                    canonicalized.value.value,
                    Interner,
                );
                self.register_obligation_in_env(uncanonical);
            }
        }
        self.resolve_obligations_buffer = obligations;
        self.resolve_obligations_buffer.clear();
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
            fn as_dyn(&mut self) -> &mut dyn TypeFolder<Interner, Error = Self::Error> {
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

    /// This checks whether any of the free variables in the `canonicalized`
    /// have changed (either been unified with another variable, or with a
    /// value). If this is not the case, we don't need to try to solve the goal
    /// again -- it'll give the same result as last time.
    fn check_changed(&mut self, canonicalized: &Canonicalized<InEnvironment<Goal>>) -> bool {
        canonicalized.free_vars.iter().any(|var| {
            let iv = match var.data(Interner) {
                GenericArgData::Ty(ty) => ty.inference_var(Interner),
                GenericArgData::Lifetime(lt) => lt.inference_var(Interner),
                GenericArgData::Const(c) => c.inference_var(Interner),
            }
            .expect("free var is not inference var");
            if self.var_unification_table.probe_var(iv).is_some() {
                return true;
            }
            let root = self.var_unification_table.inference_var_root(iv);
            iv != root
        })
    }

    fn try_resolve_obligation(
        &mut self,
        canonicalized: &Canonicalized<InEnvironment<Goal>>,
    ) -> Option<chalk_solve::Solution<Interner>> {
        let solution = self.db.trait_solve(
            self.trait_env.krate,
            self.trait_env.block,
            canonicalized.value.clone(),
        );

        match &solution {
            Some(Solution::Unique(canonical_subst)) => {
                canonicalized.apply_solution(
                    self,
                    Canonical {
                        binders: canonical_subst.binders.clone(),
                        // FIXME: handle constraints
                        value: canonical_subst.value.subst.clone(),
                    },
                );
            }
            Some(Solution::Ambig(Guidance::Definite(substs))) => {
                canonicalized.apply_solution(self, substs.clone());
            }
            Some(_) => {
                // FIXME use this when trying to resolve everything at the end
            }
            None => {
                // FIXME obligation cannot be fulfilled => diagnostic
            }
        }
        solution
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
        let krate = self.trait_env.krate;
        let fn_once_trait = FnTrait::FnOnce.get_id(self.db, krate)?;
        let trait_data = self.db.trait_data(fn_once_trait);
        let output_assoc_type =
            trait_data.associated_type_by_name(&Name::new_symbol_root(sym::Output.clone()))?;

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

        let b = TyBuilder::trait_ref(self.db, fn_once_trait);
        if b.remaining() != 2 {
            return None;
        }
        let mut trait_ref = b.push(ty.clone()).push(arg_ty).build();

        let projection = {
            TyBuilder::assoc_type_projection(
                self.db,
                output_assoc_type,
                Some(trait_ref.substitution.clone()),
            )
            .build()
        };

        let trait_env = self.trait_env.env.clone();
        let obligation = InEnvironment {
            goal: trait_ref.clone().cast(Interner),
            environment: trait_env.clone(),
        };
        let canonical = self.canonicalize(obligation.clone());
        if self.db.trait_solve(krate, self.trait_env.block, canonical.cast(Interner)).is_some() {
            self.register_obligation(obligation.goal);
            let return_ty = self.normalize_projection_ty(projection);
            for fn_x in [FnTrait::Fn, FnTrait::FnMut, FnTrait::FnOnce] {
                let fn_x_trait = fn_x.get_id(self.db, krate)?;
                trait_ref.trait_id = to_chalk_trait_id(fn_x_trait);
                let obligation: chalk_ir::InEnvironment<chalk_ir::Goal<Interner>> = InEnvironment {
                    goal: trait_ref.clone().cast(Interner),
                    environment: trait_env.clone(),
                };
                let canonical = self.canonicalize(obligation.clone());
                if self
                    .db
                    .trait_solve(krate, self.trait_env.block, canonical.cast(Interner))
                    .is_some()
                {
                    return Some((fn_x, arg_tys, return_ty));
                }
            }
            unreachable!("It should at least implement FnOnce at this point");
        } else {
            None
        }
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
                let ty_resolved = self.resolve_ty_shallow(&ty);
                if ty_resolved.is_unknown() {
                    self.new_type_var()
                } else {
                    ty
                }
            }
            _ => ty,
        }
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
        let mut ty = ty.clone();
        {
            let mut structs = SmallVec::<[_; 8]>::new();
            // Must use a loop here and not recursion because otherwise users will conduct completely
            // artificial examples of structs that have themselves as the tail field and complain r-a crashes.
            while let Some((AdtId::StructId(id), subst)) = ty.as_adt() {
                let struct_data = self.db.struct_data(id);
                if let Some((last_field, _)) = struct_data.variant_data.fields().iter().next_back()
                {
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
                } else {
                    break;
                };
            }
        }

        // Early return for some obvious types
        if matches!(
            ty.kind(Interner),
            TyKind::Scalar(..)
                | TyKind::Ref(..)
                | TyKind::Raw(..)
                | TyKind::Never
                | TyKind::FnDef(..)
                | TyKind::Array(..)
                | TyKind::Function(_)
        ) {
            return true;
        }

        let Some(sized) = self
            .db
            .lang_item(self.trait_env.krate, LangItem::Sized)
            .and_then(|sized| sized.as_trait())
        else {
            return false;
        };
        let sized_pred = WhereClause::Implemented(TraitRef {
            trait_id: to_chalk_trait_id(sized),
            substitution: Substitution::from1(Interner, ty.clone()),
        });
        let goal = GoalData::DomainGoal(chalk_ir::DomainGoal::Holds(sized_pred)).intern(Interner);
        matches!(self.try_obligation(goal), Some(Solution::Unique(_)))
    }
}

impl fmt::Debug for InferenceTable<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InferenceTable").field("num_vars", &self.type_variable_table.len()).finish()
    }
}

mod resolve {
    use super::InferenceTable;
    use crate::{
        ConcreteConst, Const, ConstData, ConstScalar, ConstValue, DebruijnIndex, GenericArg,
        InferenceVar, Interner, Lifetime, Ty, TyVariableKind, VariableKind,
    };
    use chalk_ir::{
        cast::Cast,
        fold::{TypeFoldable, TypeFolder},
    };

    #[derive(chalk_derive::FallibleTypeFolder)]
    #[has_interner(Interner)]
    pub(super) struct Resolver<
        'a,
        'b,
        F: Fn(InferenceVar, VariableKind, GenericArg, DebruijnIndex) -> GenericArg,
    > {
        pub(super) table: &'a mut InferenceTable<'b>,
        pub(super) var_stack: &'a mut Vec<InferenceVar>,
        pub(super) fallback: F,
    }
    impl<F> TypeFolder<Interner> for Resolver<'_, '_, F>
    where
        F: Fn(InferenceVar, VariableKind, GenericArg, DebruijnIndex) -> GenericArg,
    {
        fn as_dyn(&mut self) -> &mut dyn TypeFolder<Interner, Error = Self::Error> {
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
            let var = self.table.var_unification_table.inference_var_root(var);
            if self.var_stack.contains(&var) {
                // recursive type
                let default = self.table.fallback_value(var, kind).cast(Interner);
                return (self.fallback)(var, VariableKind::Ty(kind), default, outer_binder)
                    .assert_ty_ref(Interner)
                    .clone();
            }
            let result = if let Some(known_ty) = self.table.var_unification_table.probe_var(var) {
                // known_ty may contain other variables that are known by now
                self.var_stack.push(var);
                let result = known_ty.fold_with(self, outer_binder);
                self.var_stack.pop();
                result.assert_ty_ref(Interner).clone()
            } else {
                let default = self.table.fallback_value(var, kind).cast(Interner);
                (self.fallback)(var, VariableKind::Ty(kind), default, outer_binder)
                    .assert_ty_ref(Interner)
                    .clone()
            };
            result
        }

        fn fold_inference_const(
            &mut self,
            ty: Ty,
            var: InferenceVar,
            outer_binder: DebruijnIndex,
        ) -> Const {
            let var = self.table.var_unification_table.inference_var_root(var);
            let default = ConstData {
                ty: ty.clone(),
                value: ConstValue::Concrete(ConcreteConst { interned: ConstScalar::Unknown }),
            }
            .intern(Interner)
            .cast(Interner);
            if self.var_stack.contains(&var) {
                // recursive
                return (self.fallback)(var, VariableKind::Const(ty), default, outer_binder)
                    .assert_const_ref(Interner)
                    .clone();
            }
            if let Some(known_ty) = self.table.var_unification_table.probe_var(var) {
                // known_ty may contain other variables that are known by now
                self.var_stack.push(var);
                let result = known_ty.fold_with(self, outer_binder);
                self.var_stack.pop();
                result.assert_const_ref(Interner).clone()
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
