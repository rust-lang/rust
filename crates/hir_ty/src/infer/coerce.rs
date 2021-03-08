//! Coercion logic. Coercions are certain type conversions that can implicitly
//! happen in certain places, e.g. weakening `&mut` to `&` or deref coercions
//! like going from `&Vec<T>` to `&[T]`.
//!
//! See: https://doc.rust-lang.org/nomicon/coercions.html

use chalk_ir::{Mutability, TyVariableKind};
use hir_def::lang_item::LangItemTarget;

use crate::{autoderef, traits::Solution, Obligation, Substs, TraitRef, Ty};

use super::{InEnvironment, InferenceContext};

impl<'a> InferenceContext<'a> {
    /// Unify two types, but may coerce the first one to the second one
    /// using "implicit coercion rules" if needed.
    pub(super) fn coerce(&mut self, from_ty: &Ty, to_ty: &Ty) -> bool {
        let from_ty = self.resolve_ty_shallow(from_ty).into_owned();
        let to_ty = self.resolve_ty_shallow(to_ty);
        self.coerce_inner(from_ty, &to_ty)
    }

    /// Merge two types from different branches, with possible coercion.
    ///
    /// Mostly this means trying to coerce one to the other, but
    ///  - if we have two function types for different functions, we need to
    ///    coerce both to function pointers;
    ///  - if we were concerned with lifetime subtyping, we'd need to look for a
    ///    least upper bound.
    pub(super) fn coerce_merge_branch(&mut self, ty1: &Ty, ty2: &Ty) -> Ty {
        if self.coerce(ty1, ty2) {
            ty2.clone()
        } else if self.coerce(ty2, ty1) {
            ty1.clone()
        } else {
            if let (Ty::FnDef(..), Ty::FnDef(..)) = (ty1, ty2) {
                cov_mark::hit!(coerce_fn_reification);
                // Special case: two function types. Try to coerce both to
                // pointers to have a chance at getting a match. See
                // https://github.com/rust-lang/rust/blob/7b805396bf46dce972692a6846ce2ad8481c5f85/src/librustc_typeck/check/coercion.rs#L877-L916
                let sig1 = ty1.callable_sig(self.db).expect("FnDef without callable sig");
                let sig2 = ty2.callable_sig(self.db).expect("FnDef without callable sig");
                let ptr_ty1 = Ty::fn_ptr(sig1);
                let ptr_ty2 = Ty::fn_ptr(sig2);
                self.coerce_merge_branch(&ptr_ty1, &ptr_ty2)
            } else {
                cov_mark::hit!(coerce_merge_fail_fallback);
                ty1.clone()
            }
        }
    }

    fn coerce_inner(&mut self, mut from_ty: Ty, to_ty: &Ty) -> bool {
        match (&from_ty, to_ty) {
            // Never type will make type variable to fallback to Never Type instead of Unknown.
            (Ty::Never, Ty::InferenceVar(tv, TyVariableKind::General)) => {
                self.table.type_variable_table.set_diverging(*tv, true);
                return true;
            }
            (Ty::Never, _) => return true,

            // Trivial cases, this should go after `never` check to
            // avoid infer result type to be never
            _ => {
                if self.table.unify_inner_trivial(&from_ty, &to_ty, 0) {
                    return true;
                }
            }
        }

        // Pointer weakening and function to pointer
        match (&mut from_ty, to_ty) {
            // `*mut T` -> `*const T`
            // `&mut T` -> `&T`
            (Ty::Raw(m1, ..), Ty::Raw(m2 @ Mutability::Not, ..))
            | (Ty::Ref(m1, ..), Ty::Ref(m2 @ Mutability::Not, ..)) => {
                *m1 = *m2;
            }
            // `&T` -> `*const T`
            // `&mut T` -> `*mut T`/`*const T`
            (Ty::Ref(.., substs), &Ty::Raw(m2 @ Mutability::Not, ..))
            | (Ty::Ref(Mutability::Mut, substs), &Ty::Raw(m2, ..)) => {
                from_ty = Ty::Raw(m2, substs.clone());
            }

            // Illegal mutability conversion
            (Ty::Raw(Mutability::Not, ..), Ty::Raw(Mutability::Mut, ..))
            | (Ty::Ref(Mutability::Not, ..), Ty::Ref(Mutability::Mut, ..)) => return false,

            // `{function_type}` -> `fn()`
            (Ty::FnDef(..), Ty::Function { .. }) => match from_ty.callable_sig(self.db) {
                None => return false,
                Some(sig) => {
                    from_ty = Ty::fn_ptr(sig);
                }
            },

            (Ty::Closure(.., substs), Ty::Function { .. }) => {
                from_ty = substs[0].clone();
            }

            _ => {}
        }

        if let Some(ret) = self.try_coerce_unsized(&from_ty, &to_ty) {
            return ret;
        }

        // Auto Deref if cannot coerce
        match (&from_ty, to_ty) {
            // FIXME: DerefMut
            (Ty::Ref(_, st1), Ty::Ref(_, st2)) => self.unify_autoderef_behind_ref(&st1[0], &st2[0]),

            // Otherwise, normal unify
            _ => self.unify(&from_ty, to_ty),
        }
    }

    /// Coerce a type using `from_ty: CoerceUnsized<ty_ty>`
    ///
    /// See: https://doc.rust-lang.org/nightly/std/marker/trait.CoerceUnsized.html
    fn try_coerce_unsized(&mut self, from_ty: &Ty, to_ty: &Ty) -> Option<bool> {
        let krate = self.resolver.krate().unwrap();
        let coerce_unsized_trait = match self.db.lang_item(krate, "coerce_unsized".into()) {
            Some(LangItemTarget::TraitId(trait_)) => trait_,
            _ => return None,
        };

        let generic_params = crate::utils::generics(self.db.upcast(), coerce_unsized_trait.into());
        if generic_params.len() != 2 {
            // The CoerceUnsized trait should have two generic params: Self and T.
            return None;
        }

        let substs = Substs::build_for_generics(&generic_params)
            .push(from_ty.clone())
            .push(to_ty.clone())
            .build();
        let trait_ref = TraitRef { trait_: coerce_unsized_trait, substs };
        let goal = InEnvironment::new(self.trait_env.clone(), Obligation::Trait(trait_ref));

        let canonicalizer = self.canonicalizer();
        let canonicalized = canonicalizer.canonicalize_obligation(goal);

        let solution = self.db.trait_solve(krate, canonicalized.value.clone())?;

        match solution {
            Solution::Unique(v) => {
                canonicalized.apply_solution(self, v.0);
            }
            _ => return None,
        };

        Some(true)
    }

    /// Unify `from_ty` to `to_ty` with optional auto Deref
    ///
    /// Note that the parameters are already stripped the outer reference.
    fn unify_autoderef_behind_ref(&mut self, from_ty: &Ty, to_ty: &Ty) -> bool {
        let canonicalized = self.canonicalizer().canonicalize_ty(from_ty.clone());
        let to_ty = self.resolve_ty_shallow(&to_ty);
        // FIXME: Auto DerefMut
        for derefed_ty in autoderef::autoderef(
            self.db,
            self.resolver.krate(),
            InEnvironment {
                value: canonicalized.value.clone(),
                environment: self.trait_env.clone(),
            },
        ) {
            let derefed_ty = canonicalized.decanonicalize_ty(derefed_ty.value);
            let from_ty = self.resolve_ty_shallow(&derefed_ty);
            // Stop when constructor matches.
            if from_ty.equals_ctor(&to_ty) {
                // It will not recurse to `coerce`.
                return match (from_ty.substs(), to_ty.substs()) {
                    (Some(st1), Some(st2)) => self.table.unify_substs(st1, st2, 0),
                    (None, None) => true,
                    _ => false,
                };
            } else if self.table.unify_inner_trivial(&derefed_ty, &to_ty, 0) {
                return true;
            }
        }

        false
    }
}
