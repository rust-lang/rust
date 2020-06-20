//! Coercion logic. Coercions are certain type conversions that can implicitly
//! happen in certain places, e.g. weakening `&mut` to `&` or deref coercions
//! like going from `&Vec<T>` to `&[T]`.
//!
//! See: https://doc.rust-lang.org/nomicon/coercions.html

use hir_def::{lang_item::LangItemTarget, type_ref::Mutability};
use test_utils::mark;

use crate::{autoderef, traits::Solution, Obligation, Substs, TraitRef, Ty, TypeCtor};

use super::{unify::TypeVarValue, InEnvironment, InferTy, InferenceContext};

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
            if let (ty_app!(TypeCtor::FnDef(_)), ty_app!(TypeCtor::FnDef(_))) = (ty1, ty2) {
                mark::hit!(coerce_fn_reification);
                // Special case: two function types. Try to coerce both to
                // pointers to have a chance at getting a match. See
                // https://github.com/rust-lang/rust/blob/7b805396bf46dce972692a6846ce2ad8481c5f85/src/librustc_typeck/check/coercion.rs#L877-L916
                let sig1 = ty1.callable_sig(self.db).expect("FnDef without callable sig");
                let sig2 = ty2.callable_sig(self.db).expect("FnDef without callable sig");
                let ptr_ty1 = Ty::fn_ptr(sig1);
                let ptr_ty2 = Ty::fn_ptr(sig2);
                self.coerce_merge_branch(&ptr_ty1, &ptr_ty2)
            } else {
                mark::hit!(coerce_merge_fail_fallback);
                ty1.clone()
            }
        }
    }

    fn coerce_inner(&mut self, mut from_ty: Ty, to_ty: &Ty) -> bool {
        match (&from_ty, to_ty) {
            // Never type will make type variable to fallback to Never Type instead of Unknown.
            (ty_app!(TypeCtor::Never), Ty::Infer(InferTy::TypeVar(tv))) => {
                let var = self.table.new_maybe_never_type_var();
                self.table.var_unification_table.union_value(*tv, TypeVarValue::Known(var));
                return true;
            }
            (ty_app!(TypeCtor::Never), _) => return true,

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
            // `*mut T`, `&mut T, `&T`` -> `*const T`
            // `&mut T` -> `&T`
            // `&mut T` -> `*mut T`
            (ty_app!(c1@TypeCtor::RawPtr(_)), ty_app!(c2@TypeCtor::RawPtr(Mutability::Shared)))
            | (ty_app!(c1@TypeCtor::Ref(_)), ty_app!(c2@TypeCtor::RawPtr(Mutability::Shared)))
            | (ty_app!(c1@TypeCtor::Ref(_)), ty_app!(c2@TypeCtor::Ref(Mutability::Shared)))
            | (ty_app!(c1@TypeCtor::Ref(Mutability::Mut)), ty_app!(c2@TypeCtor::RawPtr(_))) => {
                *c1 = *c2;
            }

            // Illegal mutablity conversion
            (
                ty_app!(TypeCtor::RawPtr(Mutability::Shared)),
                ty_app!(TypeCtor::RawPtr(Mutability::Mut)),
            )
            | (
                ty_app!(TypeCtor::Ref(Mutability::Shared)),
                ty_app!(TypeCtor::Ref(Mutability::Mut)),
            ) => return false,

            // `{function_type}` -> `fn()`
            (ty_app!(TypeCtor::FnDef(_)), ty_app!(TypeCtor::FnPtr { .. })) => {
                match from_ty.callable_sig(self.db) {
                    None => return false,
                    Some(sig) => {
                        from_ty = Ty::fn_ptr(sig);
                    }
                }
            }

            (ty_app!(TypeCtor::Closure { .. }, params), ty_app!(TypeCtor::FnPtr { .. })) => {
                from_ty = params[0].clone();
            }

            _ => {}
        }

        if let Some(ret) = self.try_coerce_unsized(&from_ty, &to_ty) {
            return ret;
        }

        // Auto Deref if cannot coerce
        match (&from_ty, to_ty) {
            // FIXME: DerefMut
            (ty_app!(TypeCtor::Ref(_), st1), ty_app!(TypeCtor::Ref(_), st2)) => {
                self.unify_autoderef_behind_ref(&st1[0], &st2[0])
            }

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
            match (&*self.resolve_ty_shallow(&derefed_ty), &*to_ty) {
                // Stop when constructor matches.
                (ty_app!(from_ctor, st1), ty_app!(to_ctor, st2)) if from_ctor == to_ctor => {
                    // It will not recurse to `coerce`.
                    return self.table.unify_substs(st1, st2, 0);
                }
                _ => {
                    if self.table.unify_inner_trivial(&derefed_ty, &to_ty, 0) {
                        return true;
                    }
                }
            }
        }

        false
    }
}
