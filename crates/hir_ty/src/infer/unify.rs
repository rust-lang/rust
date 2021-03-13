//! Unification and canonicalization logic.

use std::borrow::Cow;

use chalk_ir::{FloatTy, IntTy, TyVariableKind};
use ena::unify::{InPlaceUnificationTable, NoError, UnifyKey, UnifyValue};

use super::{InferenceContext, Obligation};
use crate::{
    BoundVar, Canonical, DebruijnIndex, GenericPredicate, InEnvironment, InferenceVar, Interner,
    Scalar, Substs, Ty, TyKind, TypeWalk,
};

impl<'a> InferenceContext<'a> {
    pub(super) fn canonicalizer<'b>(&'b mut self) -> Canonicalizer<'a, 'b>
    where
        'a: 'b,
    {
        Canonicalizer { ctx: self, free_vars: Vec::new(), var_stack: Vec::new() }
    }
}

pub(super) struct Canonicalizer<'a, 'b>
where
    'a: 'b,
{
    ctx: &'b mut InferenceContext<'a>,
    free_vars: Vec<(InferenceVar, TyVariableKind)>,
    /// A stack of type variables that is used to detect recursive types (which
    /// are an error, but we need to protect against them to avoid stack
    /// overflows).
    var_stack: Vec<TypeVarId>,
}

#[derive(Debug)]
pub(super) struct Canonicalized<T> {
    pub(super) value: Canonical<T>,
    free_vars: Vec<(InferenceVar, TyVariableKind)>,
}

impl<'a, 'b> Canonicalizer<'a, 'b> {
    fn add(&mut self, free_var: InferenceVar, kind: TyVariableKind) -> usize {
        self.free_vars.iter().position(|&(v, _)| v == free_var).unwrap_or_else(|| {
            let next_index = self.free_vars.len();
            self.free_vars.push((free_var, kind));
            next_index
        })
    }

    fn do_canonicalize<T: TypeWalk>(&mut self, t: T, binders: DebruijnIndex) -> T {
        t.fold_binders(
            &mut |ty, binders| match ty.interned(&Interner) {
                &TyKind::InferenceVar(var, kind) => {
                    let inner = var.to_inner();
                    if self.var_stack.contains(&inner) {
                        // recursive type
                        return self.ctx.table.type_variable_table.fallback_value(var, kind);
                    }
                    if let Some(known_ty) =
                        self.ctx.table.var_unification_table.inlined_probe_value(inner).known()
                    {
                        self.var_stack.push(inner);
                        let result = self.do_canonicalize(known_ty.clone(), binders);
                        self.var_stack.pop();
                        result
                    } else {
                        let root = self.ctx.table.var_unification_table.find(inner);
                        let position = self.add(InferenceVar::from_inner(root), kind);
                        TyKind::BoundVar(BoundVar::new(binders, position)).intern(&Interner)
                    }
                }
                _ => ty,
            },
            binders,
        )
    }

    fn into_canonicalized<T>(self, result: T) -> Canonicalized<T> {
        let kinds = self.free_vars.iter().map(|&(_, k)| k).collect();
        Canonicalized { value: Canonical { value: result, kinds }, free_vars: self.free_vars }
    }

    pub(crate) fn canonicalize_ty(mut self, ty: Ty) -> Canonicalized<Ty> {
        let result = self.do_canonicalize(ty, DebruijnIndex::INNERMOST);
        self.into_canonicalized(result)
    }

    pub(crate) fn canonicalize_obligation(
        mut self,
        obligation: InEnvironment<Obligation>,
    ) -> Canonicalized<InEnvironment<Obligation>> {
        let result = match obligation.value {
            Obligation::Trait(tr) => {
                Obligation::Trait(self.do_canonicalize(tr, DebruijnIndex::INNERMOST))
            }
            Obligation::Projection(pr) => {
                Obligation::Projection(self.do_canonicalize(pr, DebruijnIndex::INNERMOST))
            }
        };
        self.into_canonicalized(InEnvironment {
            value: result,
            environment: obligation.environment,
        })
    }
}

impl<T> Canonicalized<T> {
    pub(super) fn decanonicalize_ty(&self, mut ty: Ty) -> Ty {
        ty.walk_mut_binders(
            &mut |ty, binders| {
                if let &mut TyKind::BoundVar(bound) = &mut ty.0 {
                    if bound.debruijn >= binders {
                        let (v, k) = self.free_vars[bound.index];
                        *ty = TyKind::InferenceVar(v, k).intern(&Interner);
                    }
                }
            },
            DebruijnIndex::INNERMOST,
        );
        ty
    }

    pub(super) fn apply_solution(
        &self,
        ctx: &mut InferenceContext<'_>,
        solution: Canonical<Substs>,
    ) {
        // the solution may contain new variables, which we need to convert to new inference vars
        let new_vars = Substs(
            solution
                .kinds
                .iter()
                .map(|k| match k {
                    TyVariableKind::General => ctx.table.new_type_var(),
                    TyVariableKind::Integer => ctx.table.new_integer_var(),
                    TyVariableKind::Float => ctx.table.new_float_var(),
                })
                .collect(),
        );
        for (i, ty) in solution.value.into_iter().enumerate() {
            let (v, k) = self.free_vars[i];
            // eagerly replace projections in the type; we may be getting types
            // e.g. from where clauses where this hasn't happened yet
            let ty = ctx.normalize_associated_types_in(ty.clone().subst_bound_vars(&new_vars));
            ctx.table.unify(&TyKind::InferenceVar(v, k).intern(&Interner), &ty);
        }
    }
}

pub(crate) fn unify(tys: &Canonical<(Ty, Ty)>) -> Option<Substs> {
    let mut table = InferenceTable::new();
    let vars = Substs(
        tys.kinds
            .iter()
            // we always use type vars here because we want everything to
            // fallback to Unknown in the end (kind of hacky, as below)
            .map(|_| table.new_type_var())
            .collect(),
    );
    let ty1_with_vars = tys.value.0.clone().subst_bound_vars(&vars);
    let ty2_with_vars = tys.value.1.clone().subst_bound_vars(&vars);
    if !table.unify(&ty1_with_vars, &ty2_with_vars) {
        return None;
    }
    // default any type vars that weren't unified back to their original bound vars
    // (kind of hacky)
    for (i, var) in vars.iter().enumerate() {
        if &*table.resolve_ty_shallow(var) == var {
            table.unify(
                var,
                &TyKind::BoundVar(BoundVar::new(DebruijnIndex::INNERMOST, i)).intern(&Interner),
            );
        }
    }
    Some(
        Substs::builder(tys.kinds.len())
            .fill(vars.iter().map(|v| table.resolve_ty_completely(v.clone())))
            .build(),
    )
}

#[derive(Clone, Debug)]
pub(super) struct TypeVariableTable {
    inner: Vec<TypeVariableData>,
}

impl TypeVariableTable {
    fn push(&mut self, data: TypeVariableData) {
        self.inner.push(data);
    }

    pub(super) fn set_diverging(&mut self, iv: InferenceVar, diverging: bool) {
        self.inner[iv.to_inner().0 as usize].diverging = diverging;
    }

    fn is_diverging(&mut self, iv: InferenceVar) -> bool {
        self.inner[iv.to_inner().0 as usize].diverging
    }

    fn fallback_value(&self, iv: InferenceVar, kind: TyVariableKind) -> Ty {
        match kind {
            _ if self.inner[iv.to_inner().0 as usize].diverging => TyKind::Never,
            TyVariableKind::General => TyKind::Unknown,
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

#[derive(Clone, Debug)]
pub(crate) struct InferenceTable {
    pub(super) var_unification_table: InPlaceUnificationTable<TypeVarId>,
    pub(super) type_variable_table: TypeVariableTable,
}

impl InferenceTable {
    pub(crate) fn new() -> Self {
        InferenceTable {
            var_unification_table: InPlaceUnificationTable::new(),
            type_variable_table: TypeVariableTable { inner: Vec::new() },
        }
    }

    fn new_var(&mut self, kind: TyVariableKind, diverging: bool) -> Ty {
        self.type_variable_table.push(TypeVariableData { diverging });
        let key = self.var_unification_table.new_key(TypeVarValue::Unknown);
        assert_eq!(key.0 as usize, self.type_variable_table.inner.len() - 1);
        TyKind::InferenceVar(InferenceVar::from_inner(key), kind).intern(&Interner)
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

    pub(crate) fn resolve_ty_as_possible(&mut self, ty: Ty) -> Ty {
        self.resolve_ty_as_possible_inner(&mut Vec::new(), ty)
    }

    pub(crate) fn unify(&mut self, ty1: &Ty, ty2: &Ty) -> bool {
        self.unify_inner(ty1, ty2, 0)
    }

    pub(crate) fn unify_substs(
        &mut self,
        substs1: &Substs,
        substs2: &Substs,
        depth: usize,
    ) -> bool {
        substs1.0.iter().zip(substs2.0.iter()).all(|(t1, t2)| self.unify_inner(t1, t2, depth))
    }

    fn unify_inner(&mut self, ty1: &Ty, ty2: &Ty, depth: usize) -> bool {
        if depth > 1000 {
            // prevent stackoverflows
            panic!("infinite recursion in unification");
        }
        if ty1 == ty2 {
            return true;
        }
        // try to resolve type vars first
        let ty1 = self.resolve_ty_shallow(ty1);
        let ty2 = self.resolve_ty_shallow(ty2);
        if ty1.equals_ctor(&ty2) {
            match (ty1.substs(), ty2.substs()) {
                (Some(st1), Some(st2)) => self.unify_substs(st1, st2, depth + 1),
                (None, None) => true,
                _ => false,
            }
        } else {
            self.unify_inner_trivial(&ty1, &ty2, depth)
        }
    }

    pub(super) fn unify_inner_trivial(&mut self, ty1: &Ty, ty2: &Ty, depth: usize) -> bool {
        match (ty1.interned(&Interner), ty2.interned(&Interner)) {
            (TyKind::Unknown, _) | (_, TyKind::Unknown) => true,

            (TyKind::Placeholder(p1), TyKind::Placeholder(p2)) if *p1 == *p2 => true,

            (TyKind::Dyn(dyn1), TyKind::Dyn(dyn2)) if dyn1.len() == dyn2.len() => {
                for (pred1, pred2) in dyn1.iter().zip(dyn2.iter()) {
                    if !self.unify_preds(pred1, pred2, depth + 1) {
                        return false;
                    }
                }
                true
            }

            (
                TyKind::InferenceVar(tv1, TyVariableKind::General),
                TyKind::InferenceVar(tv2, TyVariableKind::General),
            )
            | (
                TyKind::InferenceVar(tv1, TyVariableKind::Integer),
                TyKind::InferenceVar(tv2, TyVariableKind::Integer),
            )
            | (
                TyKind::InferenceVar(tv1, TyVariableKind::Float),
                TyKind::InferenceVar(tv2, TyVariableKind::Float),
            ) if self.type_variable_table.is_diverging(*tv1)
                == self.type_variable_table.is_diverging(*tv2) =>
            {
                // both type vars are unknown since we tried to resolve them
                self.var_unification_table.union(tv1.to_inner(), tv2.to_inner());
                true
            }

            // The order of MaybeNeverTypeVar matters here.
            // Unifying MaybeNeverTypeVar and TypeVar will let the latter become MaybeNeverTypeVar.
            // Unifying MaybeNeverTypeVar and other concrete type will let the former become it.
            (TyKind::InferenceVar(tv, TyVariableKind::General), other)
            | (other, TyKind::InferenceVar(tv, TyVariableKind::General))
            | (
                TyKind::InferenceVar(tv, TyVariableKind::Integer),
                other @ TyKind::Scalar(Scalar::Int(_)),
            )
            | (
                other @ TyKind::Scalar(Scalar::Int(_)),
                TyKind::InferenceVar(tv, TyVariableKind::Integer),
            )
            | (
                TyKind::InferenceVar(tv, TyVariableKind::Integer),
                other @ TyKind::Scalar(Scalar::Uint(_)),
            )
            | (
                other @ TyKind::Scalar(Scalar::Uint(_)),
                TyKind::InferenceVar(tv, TyVariableKind::Integer),
            )
            | (
                TyKind::InferenceVar(tv, TyVariableKind::Float),
                other @ TyKind::Scalar(Scalar::Float(_)),
            )
            | (
                other @ TyKind::Scalar(Scalar::Float(_)),
                TyKind::InferenceVar(tv, TyVariableKind::Float),
            ) => {
                // the type var is unknown since we tried to resolve it
                self.var_unification_table.union_value(
                    tv.to_inner(),
                    TypeVarValue::Known(other.clone().intern(&Interner)),
                );
                true
            }

            _ => false,
        }
    }

    fn unify_preds(
        &mut self,
        pred1: &GenericPredicate,
        pred2: &GenericPredicate,
        depth: usize,
    ) -> bool {
        match (pred1, pred2) {
            (GenericPredicate::Implemented(tr1), GenericPredicate::Implemented(tr2))
                if tr1.trait_ == tr2.trait_ =>
            {
                self.unify_substs(&tr1.substs, &tr2.substs, depth + 1)
            }
            (GenericPredicate::Projection(proj1), GenericPredicate::Projection(proj2))
                if proj1.projection_ty.associated_ty == proj2.projection_ty.associated_ty =>
            {
                self.unify_substs(
                    &proj1.projection_ty.parameters,
                    &proj2.projection_ty.parameters,
                    depth + 1,
                ) && self.unify_inner(&proj1.ty, &proj2.ty, depth + 1)
            }
            _ => false,
        }
    }

    /// If `ty` is a type variable with known type, returns that type;
    /// otherwise, return ty.
    pub(crate) fn resolve_ty_shallow<'b>(&mut self, ty: &'b Ty) -> Cow<'b, Ty> {
        let mut ty = Cow::Borrowed(ty);
        // The type variable could resolve to a int/float variable. Hence try
        // resolving up to three times; each type of variable shouldn't occur
        // more than once
        for i in 0..3 {
            if i > 0 {
                cov_mark::hit!(type_var_resolves_to_int_var);
            }
            match &ty.0 {
                TyKind::InferenceVar(tv, _) => {
                    let inner = tv.to_inner();
                    match self.var_unification_table.inlined_probe_value(inner).known() {
                        Some(known_ty) => {
                            // The known_ty can't be a type var itself
                            ty = Cow::Owned(known_ty.clone());
                        }
                        _ => return ty,
                    }
                }
                _ => return ty,
            }
        }
        log::error!("Inference variable still not resolved: {:?}", ty);
        ty
    }

    /// Resolves the type as far as currently possible, replacing type variables
    /// by their known types. All types returned by the infer_* functions should
    /// be resolved as far as possible, i.e. contain no type variables with
    /// known type.
    fn resolve_ty_as_possible_inner(&mut self, tv_stack: &mut Vec<TypeVarId>, ty: Ty) -> Ty {
        ty.fold(&mut |ty| match ty.interned(&Interner) {
            &TyKind::InferenceVar(tv, kind) => {
                let inner = tv.to_inner();
                if tv_stack.contains(&inner) {
                    cov_mark::hit!(type_var_cycles_resolve_as_possible);
                    // recursive type
                    return self.type_variable_table.fallback_value(tv, kind);
                }
                if let Some(known_ty) =
                    self.var_unification_table.inlined_probe_value(inner).known()
                {
                    // known_ty may contain other variables that are known by now
                    tv_stack.push(inner);
                    let result = self.resolve_ty_as_possible_inner(tv_stack, known_ty.clone());
                    tv_stack.pop();
                    result
                } else {
                    ty
                }
            }
            _ => ty,
        })
    }

    /// Resolves the type completely; type variables without known type are
    /// replaced by TyKind::Unknown.
    fn resolve_ty_completely_inner(&mut self, tv_stack: &mut Vec<TypeVarId>, ty: Ty) -> Ty {
        ty.fold(&mut |ty| match ty.interned(&Interner) {
            &TyKind::InferenceVar(tv, kind) => {
                let inner = tv.to_inner();
                if tv_stack.contains(&inner) {
                    cov_mark::hit!(type_var_cycles_resolve_completely);
                    // recursive type
                    return self.type_variable_table.fallback_value(tv, kind);
                }
                if let Some(known_ty) =
                    self.var_unification_table.inlined_probe_value(inner).known()
                {
                    // known_ty may contain other variables that are known by now
                    tv_stack.push(inner);
                    let result = self.resolve_ty_completely_inner(tv_stack, known_ty.clone());
                    tv_stack.pop();
                    result
                } else {
                    self.type_variable_table.fallback_value(tv, kind)
                }
            }
            _ => ty,
        })
    }
}

/// The ID of a type variable.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub(super) struct TypeVarId(pub(super) u32);

impl UnifyKey for TypeVarId {
    type Value = TypeVarValue;

    fn index(&self) -> u32 {
        self.0
    }

    fn from_index(i: u32) -> Self {
        TypeVarId(i)
    }

    fn tag() -> &'static str {
        "TypeVarId"
    }
}

/// The value of a type variable: either we already know the type, or we don't
/// know it yet.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum TypeVarValue {
    Known(Ty),
    Unknown,
}

impl TypeVarValue {
    fn known(&self) -> Option<&Ty> {
        match self {
            TypeVarValue::Known(ty) => Some(ty),
            TypeVarValue::Unknown => None,
        }
    }
}

impl UnifyValue for TypeVarValue {
    type Error = NoError;

    fn unify_values(value1: &Self, value2: &Self) -> Result<Self, NoError> {
        match (value1, value2) {
            // We should never equate two type variables, both of which have
            // known types. Instead, we recursively equate those types.
            (TypeVarValue::Known(t1), TypeVarValue::Known(t2)) => panic!(
                "equating two type variables, both of which have known types: {:?} and {:?}",
                t1, t2
            ),

            // If one side is known, prefer that one.
            (TypeVarValue::Known(..), TypeVarValue::Unknown) => Ok(value1.clone()),
            (TypeVarValue::Unknown, TypeVarValue::Known(..)) => Ok(value2.clone()),

            (TypeVarValue::Unknown, TypeVarValue::Unknown) => Ok(TypeVarValue::Unknown),
        }
    }
}
