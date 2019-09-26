//! Unification and canonicalization logic.

use super::{InferenceContext, Obligation};
use crate::db::HirDatabase;
use crate::ty::{
    Canonical, InEnvironment, InferTy, ProjectionPredicate, ProjectionTy, Substs, TraitRef, Ty,
    TypeWalk,
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
    var_stack: Vec<super::TypeVarId>,
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
                if let Some(known_ty) = self.ctx.var_unification_table.probe_value(inner).known() {
                    self.var_stack.push(inner);
                    let result = self.do_canonicalize_ty(known_ty.clone());
                    self.var_stack.pop();
                    result
                } else {
                    let root = self.ctx.var_unification_table.find(inner);
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

    fn do_canonicalize_trait_ref(&mut self, trait_ref: TraitRef) -> TraitRef {
        let substs =
            trait_ref.substs.iter().map(|ty| self.do_canonicalize_ty(ty.clone())).collect();
        TraitRef { trait_: trait_ref.trait_, substs: Substs(substs) }
    }

    fn into_canonicalized<T>(self, result: T) -> Canonicalized<T> {
        Canonicalized {
            value: Canonical { value: result, num_vars: self.free_vars.len() },
            free_vars: self.free_vars,
        }
    }

    fn do_canonicalize_projection_ty(&mut self, projection_ty: ProjectionTy) -> ProjectionTy {
        let params =
            projection_ty.parameters.iter().map(|ty| self.do_canonicalize_ty(ty.clone())).collect();
        ProjectionTy { associated_ty: projection_ty.associated_ty, parameters: Substs(params) }
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
    pub fn decanonicalize_ty(&self, ty: Ty) -> Ty {
        ty.fold(&mut |ty| match ty {
            Ty::Bound(idx) => {
                if (idx as usize) < self.free_vars.len() {
                    Ty::Infer(self.free_vars[idx as usize])
                } else {
                    Ty::Bound(idx)
                }
            }
            ty => ty,
        })
    }

    pub fn apply_solution(
        &self,
        ctx: &mut InferenceContext<'_, impl HirDatabase>,
        solution: Canonical<Vec<Ty>>,
    ) {
        // the solution may contain new variables, which we need to convert to new inference vars
        let new_vars = Substs((0..solution.num_vars).map(|_| ctx.new_type_var()).collect());
        for (i, ty) in solution.value.into_iter().enumerate() {
            let var = self.free_vars[i];
            ctx.unify(&Ty::Infer(var), &ty.subst_bound_vars(&new_vars));
        }
    }
}
