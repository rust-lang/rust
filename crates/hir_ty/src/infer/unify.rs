//! Unification and canonicalization logic.

use std::{borrow::Cow, fmt, sync::Arc};

use chalk_ir::{
    cast::Cast, fold::Fold, interner::HasInterner, FloatTy, IntTy, TyVariableKind, UniverseIndex,
    VariableKind,
};
use chalk_solve::infer::ParameterEnaVariableExt;
use ena::unify::UnifyKey;

use super::{InferOk, InferResult, InferenceContext, TypeError};
use crate::{
    db::HirDatabase, fold_tys, static_lifetime, BoundVar, Canonical, DebruijnIndex, GenericArg,
    InferenceVar, Interner, Scalar, Substitution, TraitEnvironment, Ty, TyKind,
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
        crate::fold_free_vars(ty, |bound, _binders| {
            let var = self.free_vars[bound.index].clone();
            var.assert_ty_ref(&Interner).clone()
        })
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
        for (i, ty) in solution.value.iter(&Interner).enumerate() {
            // FIXME: deal with non-type vars here -- the only problematic part is the normalization
            // and maybe we don't need that with lazy normalization?
            let var = self.free_vars[i].clone();
            // eagerly replace projections in the type; we may be getting types
            // e.g. from where clauses where this hasn't happened yet
            let ty = ctx.normalize_associated_types_in(
                new_vars.apply(ty.assert_ty_ref(&Interner).clone(), &Interner),
            );
            ctx.table.unify(var.assert_ty_ref(&Interner), &ty);
        }
    }
}

pub fn could_unify(db: &dyn HirDatabase, env: Arc<TraitEnvironment>, t1: &Ty, t2: &Ty) -> bool {
    InferenceTable::new(db, env).unify(t1, t2)
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
    for (i, var) in vars.iter(&Interner).enumerate() {
        let var = var.assert_ty_ref(&Interner);
        if &*table.resolve_ty_shallow(var) == var {
            table.unify(
                var,
                &TyKind::BoundVar(BoundVar::new(DebruijnIndex::INNERMOST, i)).intern(&Interner),
            );
        }
    }
    Some(Substitution::from_iter(
        &Interner,
        vars.iter(&Interner)
            .map(|v| table.resolve_ty_completely(v.assert_ty_ref(&Interner).clone())),
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
            _ if self.inner[iv.index() as usize].diverging => TyKind::Never,
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

    fn new_var(&mut self, kind: TyVariableKind, diverging: bool) -> Ty {
        let var = self.var_unification_table.new_variable(UniverseIndex::ROOT);
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

    pub(crate) fn resolve_ty_completely(&mut self, ty: Ty) -> Ty {
        self.resolve_ty_completely_inner(&mut Vec::new(), ty)
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
    pub(crate) fn unify_inner(&mut self, ty1: &Ty, ty2: &Ty) -> InferResult {
        match self.var_unification_table.relate(
            &Interner,
            &self.db,
            &self.trait_env.env,
            chalk_ir::Variance::Invariant,
            ty1,
            ty2,
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
    /// by their known types. All types returned by the infer_* functions should
    /// be resolved as far as possible, i.e. contain no type variables with
    /// known type.
    fn resolve_ty_as_possible_inner(&mut self, tv_stack: &mut Vec<InferenceVar>, ty: Ty) -> Ty {
        fold_tys(
            ty,
            |ty, _| match ty.kind(&Interner) {
                &TyKind::InferenceVar(tv, kind) => {
                    if tv_stack.contains(&tv) {
                        cov_mark::hit!(type_var_cycles_resolve_as_possible);
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

    /// Resolves the type completely; type variables without known type are
    /// replaced by TyKind::Unknown.
    fn resolve_ty_completely_inner(&mut self, tv_stack: &mut Vec<InferenceVar>, ty: Ty) -> Ty {
        // FIXME implement as a proper Folder, handle lifetimes and consts as well
        fold_tys(
            ty,
            |ty, _| match ty.kind(&Interner) {
                &TyKind::InferenceVar(tv, kind) => {
                    if tv_stack.contains(&tv) {
                        cov_mark::hit!(type_var_cycles_resolve_completely);
                        // recursive type
                        return self.type_variable_table.fallback_value(tv, kind);
                    }
                    if let Some(known_ty) = self.var_unification_table.probe_var(tv) {
                        // known_ty may contain other variables that are known by now
                        tv_stack.push(tv);
                        let result = self.resolve_ty_completely_inner(
                            tv_stack,
                            known_ty.assert_ty_ref(&Interner).clone(),
                        );
                        tv_stack.pop();
                        result
                    } else {
                        self.type_variable_table.fallback_value(tv, kind)
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
