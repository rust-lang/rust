//! Unification and canonicalization logic.

use std::{borrow::Cow, fmt, sync::Arc};

use chalk_ir::{
    cast::Cast, fold::Fold, interner::HasInterner, zip::Zip, FloatTy, IntTy, TyVariableKind,
    UniverseIndex,
};
use chalk_solve::infer::ParameterEnaVariableExt;
use ena::unify::UnifyKey;

use super::{InferOk, InferResult, InferenceContext, TypeError};
use crate::{
    db::HirDatabase, fold_tys, static_lifetime, BoundVar, Canonical, DebruijnIndex, GenericArg,
    InferenceVar, Interner, Scalar, Substitution, TraitEnvironment, Ty, TyKind, VariableKind,
};

impl<'a> InferenceContext<'a> {
    pub(super) fn canonicalize<T: Fold<Interner> + HasInterner<Interner = Interner>>(
        &mut self,
        t: T,
    ) -> Canonicalized<T::Result>
    where
        T::Result: HasInterner<Interner = Interner>,
    {
        let result = self.table.var_unification_table.canonicalize(&Interner, t);
        let free_vars = result
            .free_vars
            .into_iter()
            .map(|free_var| free_var.to_generic_arg(&Interner))
            .collect();
        Canonicalized { value: result.quantified, free_vars }
    }
}

#[derive(Debug)]
pub(super) struct Canonicalized<T>
where
    T: HasInterner<Interner = Interner>,
{
    pub(super) value: Canonical<T>,
    free_vars: Vec<GenericArg>,
}

impl<T: HasInterner<Interner = Interner>> Canonicalized<T> {
    pub(super) fn decanonicalize_ty(&self, ty: Ty) -> Ty {
        chalk_ir::Substitute::apply(&self.free_vars, ty, &Interner)
    }

    pub(super) fn apply_solution(
        &self,
        ctx: &mut InferenceContext<'_>,
        solution: Canonical<Substitution>,
    ) {
        // the solution may contain new variables, which we need to convert to new inference vars
        let new_vars = Substitution::from_iter(
            &Interner,
            solution.binders.iter(&Interner).map(|k| match k.kind {
                VariableKind::Ty(TyVariableKind::General) => {
                    ctx.table.new_type_var().cast(&Interner)
                }
                VariableKind::Ty(TyVariableKind::Integer) => {
                    ctx.table.new_integer_var().cast(&Interner)
                }
                VariableKind::Ty(TyVariableKind::Float) => {
                    ctx.table.new_float_var().cast(&Interner)
                }
                // Chalk can sometimes return new lifetime variables. We just use the static lifetime everywhere
                VariableKind::Lifetime => static_lifetime().cast(&Interner),
                _ => panic!("const variable in solution"),
            }),
        );
        for (i, v) in solution.value.iter(&Interner).enumerate() {
            let var = self.free_vars[i].clone();
            if let Some(ty) = v.ty(&Interner) {
                // eagerly replace projections in the type; we may be getting types
                // e.g. from where clauses where this hasn't happened yet
                let ty = ctx.normalize_associated_types_in(new_vars.apply(ty.clone(), &Interner));
                ctx.table.unify(var.assert_ty_ref(&Interner), &ty);
            } else {
                let _ = ctx.table.unify_inner(&var, &new_vars.apply(v.clone(), &Interner));
            }
        }
    }
}

pub fn could_unify(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    tys: &Canonical<(Ty, Ty)>,
) -> bool {
    unify(db, env, tys).is_some()
}

pub(crate) fn unify(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    tys: &Canonical<(Ty, Ty)>,
) -> Option<Substitution> {
    let mut table = InferenceTable::new(db, env);
    let vars = Substitution::from_iter(
        &Interner,
        tys.binders
            .iter(&Interner)
            // we always use type vars here because we want everything to
            // fallback to Unknown in the end (kind of hacky, as below)
            .map(|_| table.new_type_var()),
    );
    let ty1_with_vars = vars.apply(tys.value.0.clone(), &Interner);
    let ty2_with_vars = vars.apply(tys.value.1.clone(), &Interner);
    if !table.unify(&ty1_with_vars, &ty2_with_vars) {
        return None;
    }
    // default any type vars that weren't unified back to their original bound vars
    // (kind of hacky)
    let find_var = |iv| {
        vars.iter(&Interner).position(|v| match v.interned() {
            chalk_ir::GenericArgData::Ty(ty) => ty.inference_var(&Interner),
            chalk_ir::GenericArgData::Lifetime(lt) => lt.inference_var(&Interner),
            chalk_ir::GenericArgData::Const(c) => c.inference_var(&Interner),
        } == Some(iv))
    };
    let fallback = |iv, kind, default, binder| match kind {
        chalk_ir::VariableKind::Ty(_ty_kind) => find_var(iv)
            .map_or(default, |i| BoundVar::new(binder, i).to_ty(&Interner).cast(&Interner)),
        chalk_ir::VariableKind::Lifetime => find_var(iv)
            .map_or(default, |i| BoundVar::new(binder, i).to_lifetime(&Interner).cast(&Interner)),
        chalk_ir::VariableKind::Const(ty) => find_var(iv)
            .map_or(default, |i| BoundVar::new(binder, i).to_const(&Interner, ty).cast(&Interner)),
    };
    Some(Substitution::from_iter(
        &Interner,
        vars.iter(&Interner)
            .map(|v| table.resolve_with_fallback(v.assert_ty_ref(&Interner).clone(), fallback)),
    ))
}

#[derive(Clone, Debug)]
pub(super) struct TypeVariableTable {
    inner: Vec<TypeVariableData>,
}

impl TypeVariableTable {
    pub(super) fn set_diverging(&mut self, iv: InferenceVar, diverging: bool) {
        self.inner[iv.index() as usize].diverging = diverging;
    }

    fn fallback_value(&self, iv: InferenceVar, kind: TyVariableKind) -> Ty {
        match kind {
            _ if self.inner.get(iv.index() as usize).map_or(false, |data| data.diverging) => {
                TyKind::Never
            }
            TyVariableKind::General => TyKind::Error,
            TyVariableKind::Integer => TyKind::Scalar(Scalar::Int(IntTy::I32)),
            TyVariableKind::Float => TyKind::Scalar(Scalar::Float(FloatTy::F64)),
        }
        .intern(&Interner)
    }
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct TypeVariableData {
    diverging: bool,
}

type ChalkInferenceTable = chalk_solve::infer::InferenceTable<Interner>;

#[derive(Clone)]
pub(crate) struct InferenceTable<'a> {
    db: &'a dyn HirDatabase,
    trait_env: Arc<TraitEnvironment>,
    pub(super) var_unification_table: ChalkInferenceTable,
    pub(super) type_variable_table: TypeVariableTable,
}

impl<'a> InferenceTable<'a> {
    pub(crate) fn new(db: &'a dyn HirDatabase, trait_env: Arc<TraitEnvironment>) -> Self {
        InferenceTable {
            db,
            trait_env,
            var_unification_table: ChalkInferenceTable::new(),
            type_variable_table: TypeVariableTable { inner: Vec::new() },
        }
    }

    /// Chalk doesn't know about the `diverging` flag, so when it unifies two
    /// type variables of which one is diverging, the chosen root might not be
    /// diverging and we have no way of marking it as such at that time. This
    /// function goes through all type variables and make sure their root is
    /// marked as diverging if necessary, so that resolving them gives the right
    /// result.
    pub(super) fn propagate_diverging_flag(&mut self) {
        for i in 0..self.type_variable_table.inner.len() {
            if !self.type_variable_table.inner[i].diverging {
                continue;
            }
            let v = InferenceVar::from(i as u32);
            let root = self.var_unification_table.inference_var_root(v);
            if let Some(data) = self.type_variable_table.inner.get_mut(root.index() as usize) {
                data.diverging = true;
            }
        }
    }

    fn new_var(&mut self, kind: TyVariableKind, diverging: bool) -> Ty {
        let var = self.var_unification_table.new_variable(UniverseIndex::ROOT);
        // Chalk might have created some type variables for its own purposes that we don't know about...
        // TODO refactor this?
        self.type_variable_table.inner.extend(
            (0..1 + var.index() as usize - self.type_variable_table.inner.len())
                .map(|_| TypeVariableData { diverging: false }),
        );
        assert_eq!(var.index() as usize, self.type_variable_table.inner.len() - 1);
        self.type_variable_table.inner[var.index() as usize].diverging = diverging;
        var.to_ty_with_kind(&Interner, kind)
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

    pub(crate) fn resolve_with_fallback<T>(
        &mut self,
        t: T,
        fallback: impl Fn(InferenceVar, VariableKind, GenericArg, DebruijnIndex) -> GenericArg,
    ) -> T::Result
    where
        T: HasInterner<Interner = Interner> + Fold<Interner>,
    {
        self.resolve_with_fallback_inner(&mut Vec::new(), t, &fallback)
    }

    fn resolve_with_fallback_inner<T>(
        &mut self,
        var_stack: &mut Vec<InferenceVar>,
        t: T,
        fallback: &impl Fn(InferenceVar, VariableKind, GenericArg, DebruijnIndex) -> GenericArg,
    ) -> T::Result
    where
        T: HasInterner<Interner = Interner> + Fold<Interner>,
    {
        t.fold_with(
            &mut resolve::Resolver {
                type_variable_table: &self.type_variable_table,
                var_unification_table: &mut self.var_unification_table,
                var_stack,
                fallback,
            },
            DebruijnIndex::INNERMOST,
        )
        .expect("fold failed unexpectedly")
    }

    pub(crate) fn resolve_ty_completely(&mut self, ty: Ty) -> Ty {
        self.resolve_with_fallback(ty, |_, _, d, _| d)
    }

    // FIXME get rid of this, instead resolve shallowly where necessary
    pub(crate) fn resolve_ty_as_possible(&mut self, ty: Ty) -> Ty {
        self.resolve_ty_as_possible_inner(&mut Vec::new(), ty)
    }

    /// Unify two types and register new trait goals that arise from that.
    // TODO give these two functions better names
    pub(crate) fn unify(&mut self, ty1: &Ty, ty2: &Ty) -> bool {
        let _result = if let Ok(r) = self.unify_inner(ty1, ty2) {
            r
        } else {
            return false;
        };
        // TODO deal with new goals
        true
    }

    /// Unify two types and return new trait goals arising from it, so the
    /// caller needs to deal with them.
    pub(crate) fn unify_inner<T: Zip<Interner>>(&mut self, t1: &T, t2: &T) -> InferResult {
        match self.var_unification_table.relate(
            &Interner,
            &self.db,
            &self.trait_env.env,
            chalk_ir::Variance::Invariant,
            t1,
            t2,
        ) {
            Ok(_result) => {
                // TODO deal with new goals
                Ok(InferOk {})
            }
            Err(chalk_ir::NoSolution) => Err(TypeError),
        }
    }

    /// If `ty` is a type variable with known type, returns that type;
    /// otherwise, return ty.
    // FIXME this could probably just return Ty
    pub(crate) fn resolve_ty_shallow<'b>(&mut self, ty: &'b Ty) -> Cow<'b, Ty> {
        self.var_unification_table
            .normalize_ty_shallow(&Interner, ty)
            .map_or(Cow::Borrowed(ty), Cow::Owned)
    }

    /// Resolves the type as far as currently possible, replacing type variables
    /// by their known types.
    fn resolve_ty_as_possible_inner(&mut self, tv_stack: &mut Vec<InferenceVar>, ty: Ty) -> Ty {
        fold_tys(
            ty,
            |ty, _| match ty.kind(&Interner) {
                &TyKind::InferenceVar(tv, kind) => {
                    if tv_stack.contains(&tv) {
                        // recursive type
                        return self.type_variable_table.fallback_value(tv, kind);
                    }
                    if let Some(known_ty) = self.var_unification_table.probe_var(tv) {
                        // known_ty may contain other variables that are known by now
                        tv_stack.push(tv);
                        let result = self.resolve_ty_as_possible_inner(
                            tv_stack,
                            known_ty.assert_ty_ref(&Interner).clone(),
                        );
                        tv_stack.pop();
                        result
                    } else {
                        ty
                    }
                }
                _ => ty,
            },
            DebruijnIndex::INNERMOST,
        )
    }
}

impl<'a> fmt::Debug for InferenceTable<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InferenceTable")
            .field("num_vars", &self.type_variable_table.inner.len())
            .finish()
    }
}

mod resolve {
    use super::{ChalkInferenceTable, TypeVariableTable};
    use crate::{
        ConcreteConst, Const, ConstData, ConstValue, DebruijnIndex, GenericArg, InferenceVar,
        Interner, Ty, TyVariableKind, VariableKind,
    };
    use chalk_ir::{
        cast::Cast,
        fold::{Fold, Folder},
        Fallible,
    };
    use hir_def::type_ref::ConstScalar;

    pub(super) struct Resolver<'a, F> {
        pub type_variable_table: &'a TypeVariableTable,
        pub var_unification_table: &'a mut ChalkInferenceTable,
        pub var_stack: &'a mut Vec<InferenceVar>,
        pub fallback: F,
    }
    impl<'a, 'i, F> Folder<'i, Interner> for Resolver<'a, F>
    where
        F: Fn(InferenceVar, VariableKind, GenericArg, DebruijnIndex) -> GenericArg + 'i,
    {
        fn as_dyn(&mut self) -> &mut dyn Folder<'i, Interner> {
            self
        }

        fn interner(&self) -> &'i Interner {
            &Interner
        }

        fn fold_inference_ty(
            &mut self,
            var: InferenceVar,
            kind: TyVariableKind,
            outer_binder: DebruijnIndex,
        ) -> Fallible<Ty> {
            let var = self.var_unification_table.inference_var_root(var);
            if self.var_stack.contains(&var) {
                // recursive type
                let default = self.type_variable_table.fallback_value(var, kind).cast(&Interner);
                return Ok((self.fallback)(var, VariableKind::Ty(kind), default, outer_binder)
                    .assert_ty_ref(&Interner)
                    .clone());
            }
            let result = if let Some(known_ty) = self.var_unification_table.probe_var(var) {
                // known_ty may contain other variables that are known by now
                self.var_stack.push(var);
                let result =
                    known_ty.fold_with(self, outer_binder).expect("fold failed unexpectedly");
                self.var_stack.pop();
                result.assert_ty_ref(&Interner).clone()
            } else {
                let default = self.type_variable_table.fallback_value(var, kind).cast(&Interner);
                (self.fallback)(var, VariableKind::Ty(kind), default, outer_binder)
                    .assert_ty_ref(&Interner)
                    .clone()
            };
            Ok(result)
        }

        fn fold_inference_const(
            &mut self,
            ty: Ty,
            var: InferenceVar,
            outer_binder: DebruijnIndex,
        ) -> Fallible<Const> {
            let var = self.var_unification_table.inference_var_root(var);
            let default = ConstData {
                ty: ty.clone(),
                value: ConstValue::Concrete(ConcreteConst { interned: ConstScalar::Unknown }),
            }
            .intern(&Interner)
            .cast(&Interner);
            if self.var_stack.contains(&var) {
                // recursive
                return Ok((self.fallback)(var, VariableKind::Const(ty), default, outer_binder)
                    .assert_const_ref(&Interner)
                    .clone());
            }
            let result = if let Some(known_ty) = self.var_unification_table.probe_var(var) {
                // known_ty may contain other variables that are known by now
                self.var_stack.push(var);
                let result =
                    known_ty.fold_with(self, outer_binder).expect("fold failed unexpectedly");
                self.var_stack.pop();
                result.assert_const_ref(&Interner).clone()
            } else {
                (self.fallback)(var, VariableKind::Const(ty), default, outer_binder)
                    .assert_const_ref(&Interner)
                    .clone()
            };
            Ok(result)
        }
    }
}
