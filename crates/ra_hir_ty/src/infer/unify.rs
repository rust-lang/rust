//! Unification and canonicalization logic.

use std::borrow::Cow;

use ena::unify::{InPlaceUnificationTable, NoError, UnifyKey, UnifyValue};

use test_utils::tested_by;

use super::{InferenceContext, Obligation};
use crate::{
    db::HirDatabase, utils::make_mut_slice, Canonical, InEnvironment, InferTy, ProjectionPredicate,
    ProjectionTy, Substs, TraitRef, Ty, TypeCtor, TypeWalk,
};

impl<'a, D: HirDatabase> InferenceContext<'a, D> {
    pub(super) fn canonicalizer<'b>(&'b mut self) -> Canonicalizer<'a, 'b, D>
    where
        'a: 'b,
    {
        Canonicalizer { ctx: self, free_vars: Vec::new(), var_stack: Vec::new() }
    }
}

pub(super) struct Canonicalizer<'a, 'b, D: HirDatabase>
where
    'a: 'b,
{
    ctx: &'b mut InferenceContext<'a, D>,
    free_vars: Vec<InferTy>,
    /// A stack of type variables that is used to detect recursive types (which
    /// are an error, but we need to protect against them to avoid stack
    /// overflows).
    var_stack: Vec<TypeVarId>,
}

pub(super) struct Canonicalized<T> {
    pub value: Canonical<T>,
    free_vars: Vec<InferTy>,
}

impl<'a, 'b, D: HirDatabase> Canonicalizer<'a, 'b, D>
where
    'a: 'b,
{
    fn add(&mut self, free_var: InferTy) -> usize {
        self.free_vars.iter().position(|&v| v == free_var).unwrap_or_else(|| {
            let next_index = self.free_vars.len();
            self.free_vars.push(free_var);
            next_index
        })
    }

    fn do_canonicalize_ty(&mut self, ty: Ty) -> Ty {
        ty.fold(&mut |ty| match ty {
            Ty::Infer(tv) => {
                let inner = tv.to_inner();
                if self.var_stack.contains(&inner) {
                    // recursive type
                    return tv.fallback_value();
                }
                if let Some(known_ty) =
                    self.ctx.table.var_unification_table.inlined_probe_value(inner).known()
                {
                    self.var_stack.push(inner);
                    let result = self.do_canonicalize_ty(known_ty.clone());
                    self.var_stack.pop();
                    result
                } else {
                    let root = self.ctx.table.var_unification_table.find(inner);
                    let free_var = match tv {
                        InferTy::TypeVar(_) => InferTy::TypeVar(root),
                        InferTy::IntVar(_) => InferTy::IntVar(root),
                        InferTy::FloatVar(_) => InferTy::FloatVar(root),
                        InferTy::MaybeNeverTypeVar(_) => InferTy::MaybeNeverTypeVar(root),
                    };
                    let position = self.add(free_var);
                    Ty::Bound(position as u32)
                }
            }
            _ => ty,
        })
    }

    fn do_canonicalize_trait_ref(&mut self, mut trait_ref: TraitRef) -> TraitRef {
        for ty in make_mut_slice(&mut trait_ref.substs.0) {
            *ty = self.do_canonicalize_ty(ty.clone());
        }
        trait_ref
    }

    fn into_canonicalized<T>(self, result: T) -> Canonicalized<T> {
        Canonicalized {
            value: Canonical { value: result, num_vars: self.free_vars.len() },
            free_vars: self.free_vars,
        }
    }

    fn do_canonicalize_projection_ty(&mut self, mut projection_ty: ProjectionTy) -> ProjectionTy {
        for ty in make_mut_slice(&mut projection_ty.parameters.0) {
            *ty = self.do_canonicalize_ty(ty.clone());
        }
        projection_ty
    }

    fn do_canonicalize_projection_predicate(
        &mut self,
        projection: ProjectionPredicate,
    ) -> ProjectionPredicate {
        let ty = self.do_canonicalize_ty(projection.ty);
        let projection_ty = self.do_canonicalize_projection_ty(projection.projection_ty);

        ProjectionPredicate { ty, projection_ty }
    }

    // FIXME: add some point, we need to introduce a `Fold` trait that abstracts
    // over all the things that can be canonicalized (like Chalk and rustc have)

    pub(crate) fn canonicalize_ty(mut self, ty: Ty) -> Canonicalized<Ty> {
        let result = self.do_canonicalize_ty(ty);
        self.into_canonicalized(result)
    }

    pub(crate) fn canonicalize_obligation(
        mut self,
        obligation: InEnvironment<Obligation>,
    ) -> Canonicalized<InEnvironment<Obligation>> {
        let result = match obligation.value {
            Obligation::Trait(tr) => Obligation::Trait(self.do_canonicalize_trait_ref(tr)),
            Obligation::Projection(pr) => {
                Obligation::Projection(self.do_canonicalize_projection_predicate(pr))
            }
        };
        self.into_canonicalized(InEnvironment {
            value: result,
            environment: obligation.environment,
        })
    }
}

impl<T> Canonicalized<T> {
    pub fn decanonicalize_ty(&self, mut ty: Ty) -> Ty {
        ty.walk_mut_binders(
            &mut |ty, binders| match ty {
                &mut Ty::Bound(idx) => {
                    if idx as usize >= binders && (idx as usize - binders) < self.free_vars.len() {
                        *ty = Ty::Infer(self.free_vars[idx as usize - binders]);
                    }
                }
                _ => {}
            },
            0,
        );
        ty
    }

    pub fn apply_solution(
        &self,
        ctx: &mut InferenceContext<'_, impl HirDatabase>,
        solution: Canonical<Vec<Ty>>,
    ) {
        // the solution may contain new variables, which we need to convert to new inference vars
        let new_vars = Substs((0..solution.num_vars).map(|_| ctx.table.new_type_var()).collect());
        for (i, ty) in solution.value.into_iter().enumerate() {
            let var = self.free_vars[i];
            ctx.table.unify(&Ty::Infer(var), &ty.subst_bound_vars(&new_vars));
        }
    }
}

pub fn unify(ty1: &Canonical<Ty>, ty2: &Canonical<Ty>) -> Option<Substs> {
    let mut table = InferenceTable::new();
    let vars =
        Substs::builder(ty1.num_vars).fill(std::iter::repeat_with(|| table.new_type_var())).build();
    let ty_with_vars = ty1.value.clone().subst_bound_vars(&vars);
    if !table.unify(&ty_with_vars, &ty2.value) {
        return None;
    }
    Some(
        Substs::builder(ty1.num_vars)
            .fill(vars.iter().map(|v| table.resolve_ty_completely(v.clone())))
            .build(),
    )
}

#[derive(Clone, Debug)]
pub(crate) struct InferenceTable {
    pub(super) var_unification_table: InPlaceUnificationTable<TypeVarId>,
}

impl InferenceTable {
    pub fn new() -> Self {
        InferenceTable { var_unification_table: InPlaceUnificationTable::new() }
    }

    pub fn new_type_var(&mut self) -> Ty {
        Ty::Infer(InferTy::TypeVar(self.var_unification_table.new_key(TypeVarValue::Unknown)))
    }

    pub fn new_integer_var(&mut self) -> Ty {
        Ty::Infer(InferTy::IntVar(self.var_unification_table.new_key(TypeVarValue::Unknown)))
    }

    pub fn new_float_var(&mut self) -> Ty {
        Ty::Infer(InferTy::FloatVar(self.var_unification_table.new_key(TypeVarValue::Unknown)))
    }

    pub fn new_maybe_never_type_var(&mut self) -> Ty {
        Ty::Infer(InferTy::MaybeNeverTypeVar(
            self.var_unification_table.new_key(TypeVarValue::Unknown),
        ))
    }

    pub fn resolve_ty_completely(&mut self, ty: Ty) -> Ty {
        self.resolve_ty_completely_inner(&mut Vec::new(), ty)
    }

    pub fn resolve_ty_as_possible(&mut self, ty: Ty) -> Ty {
        self.resolve_ty_as_possible_inner(&mut Vec::new(), ty)
    }

    pub fn unify(&mut self, ty1: &Ty, ty2: &Ty) -> bool {
        self.unify_inner(ty1, ty2, 0)
    }

    pub fn unify_substs(&mut self, substs1: &Substs, substs2: &Substs, depth: usize) -> bool {
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
        match (&*ty1, &*ty2) {
            (Ty::Apply(a_ty1), Ty::Apply(a_ty2)) if a_ty1.ctor == a_ty2.ctor => {
                self.unify_substs(&a_ty1.parameters, &a_ty2.parameters, depth + 1)
            }
            _ => self.unify_inner_trivial(&ty1, &ty2),
        }
    }

    pub(super) fn unify_inner_trivial(&mut self, ty1: &Ty, ty2: &Ty) -> bool {
        match (ty1, ty2) {
            (Ty::Unknown, _) | (_, Ty::Unknown) => true,

            (Ty::Infer(InferTy::TypeVar(tv1)), Ty::Infer(InferTy::TypeVar(tv2)))
            | (Ty::Infer(InferTy::IntVar(tv1)), Ty::Infer(InferTy::IntVar(tv2)))
            | (Ty::Infer(InferTy::FloatVar(tv1)), Ty::Infer(InferTy::FloatVar(tv2)))
            | (
                Ty::Infer(InferTy::MaybeNeverTypeVar(tv1)),
                Ty::Infer(InferTy::MaybeNeverTypeVar(tv2)),
            ) => {
                // both type vars are unknown since we tried to resolve them
                self.var_unification_table.union(*tv1, *tv2);
                true
            }

            // The order of MaybeNeverTypeVar matters here.
            // Unifying MaybeNeverTypeVar and TypeVar will let the latter become MaybeNeverTypeVar.
            // Unifying MaybeNeverTypeVar and other concrete type will let the former become it.
            (Ty::Infer(InferTy::TypeVar(tv)), other)
            | (other, Ty::Infer(InferTy::TypeVar(tv)))
            | (Ty::Infer(InferTy::MaybeNeverTypeVar(tv)), other)
            | (other, Ty::Infer(InferTy::MaybeNeverTypeVar(tv)))
            | (Ty::Infer(InferTy::IntVar(tv)), other @ ty_app!(TypeCtor::Int(_)))
            | (other @ ty_app!(TypeCtor::Int(_)), Ty::Infer(InferTy::IntVar(tv)))
            | (Ty::Infer(InferTy::FloatVar(tv)), other @ ty_app!(TypeCtor::Float(_)))
            | (other @ ty_app!(TypeCtor::Float(_)), Ty::Infer(InferTy::FloatVar(tv))) => {
                // the type var is unknown since we tried to resolve it
                self.var_unification_table.union_value(*tv, TypeVarValue::Known(other.clone()));
                true
            }

            _ => false,
        }
    }

    /// If `ty` is a type variable with known type, returns that type;
    /// otherwise, return ty.
    pub fn resolve_ty_shallow<'b>(&mut self, ty: &'b Ty) -> Cow<'b, Ty> {
        let mut ty = Cow::Borrowed(ty);
        // The type variable could resolve to a int/float variable. Hence try
        // resolving up to three times; each type of variable shouldn't occur
        // more than once
        for i in 0..3 {
            if i > 0 {
                tested_by!(type_var_resolves_to_int_var);
            }
            match &*ty {
                Ty::Infer(tv) => {
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
        ty.fold(&mut |ty| match ty {
            Ty::Infer(tv) => {
                let inner = tv.to_inner();
                if tv_stack.contains(&inner) {
                    tested_by!(type_var_cycles_resolve_as_possible);
                    // recursive type
                    return tv.fallback_value();
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
    /// replaced by Ty::Unknown.
    fn resolve_ty_completely_inner(&mut self, tv_stack: &mut Vec<TypeVarId>, ty: Ty) -> Ty {
        ty.fold(&mut |ty| match ty {
            Ty::Infer(tv) => {
                let inner = tv.to_inner();
                if tv_stack.contains(&inner) {
                    tested_by!(type_var_cycles_resolve_completely);
                    // recursive type
                    return tv.fallback_value();
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
                    tv.fallback_value()
                }
            }
            _ => ty,
        })
    }
}

/// The ID of a type variable.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct TypeVarId(pub(super) u32);

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
pub enum TypeVarValue {
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
