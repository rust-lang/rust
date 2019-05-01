//! Unification and canonicalization logic.

use crate::db::HirDatabase;
use crate::ty::{Ty, Canonical, TraitRef, InferTy};
use super::InferenceContext;

impl<'a, D: HirDatabase> InferenceContext<'a, D> {
    pub(super) fn canonicalizer<'b>(&'b mut self) -> Canonicalizer<'a, 'b, D>
    where
        'a: 'b,
    {
        Canonicalizer { ctx: self, free_vars: Vec::new() }
    }
}

// TODO improve the interface of this

pub(super) struct Canonicalizer<'a, 'b, D: HirDatabase>
where
    'a: 'b,
{
    pub ctx: &'b mut InferenceContext<'a, D>,
    pub free_vars: Vec<InferTy>,
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

    pub fn canonicalize_ty(&mut self, ty: Ty) -> Canonical<Ty> {
        let value = ty.fold(&mut |ty| match ty {
            Ty::Infer(tv) => {
                let inner = tv.to_inner();
                // TODO prevent infinite loops? => keep var stack
                if let Some(known_ty) = self.ctx.var_unification_table.probe_value(inner).known() {
                    self.canonicalize_ty(known_ty.clone()).value
                } else {
                    let free_var = InferTy::TypeVar(self.ctx.var_unification_table.find(inner));
                    let position = self.add(free_var);
                    Ty::Bound(position as u32)
                }
            }
            _ => ty,
        });
        Canonical { value, num_vars: self.free_vars.len() }
    }

    pub fn canonicalize_trait_ref(&mut self, trait_ref: TraitRef) -> Canonical<TraitRef> {
        let substs = trait_ref
            .substs
            .iter()
            .map(|ty| self.canonicalize_ty(ty.clone()).value)
            .collect::<Vec<_>>();
        let value = TraitRef { trait_: trait_ref.trait_, substs: substs.into() };
        Canonical { value, num_vars: self.free_vars.len() }
    }

    pub fn decanonicalize_ty(&self, ty: Ty) -> Ty {
        ty.fold(&mut |ty| match ty {
            Ty::Bound(idx) => {
                if (idx as usize) < self.free_vars.len() {
                    Ty::Infer(self.free_vars[idx as usize].clone())
                } else {
                    Ty::Bound(idx)
                }
            }
            ty => ty,
        })
    }

    pub fn apply_solution(&mut self, solution: Canonical<Vec<Ty>>) {
        // the solution may contain new variables, which we need to convert to new inference vars
        let new_vars =
            (0..solution.num_vars).map(|_| self.ctx.new_type_var()).collect::<Vec<_>>().into();
        for (i, ty) in solution.value.into_iter().enumerate() {
            let var = self.free_vars[i].clone();
            self.ctx.unify(&Ty::Infer(var), &ty.subst_bound_vars(&new_vars));
        }
    }
}
