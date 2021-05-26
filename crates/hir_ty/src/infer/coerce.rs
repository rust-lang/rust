//! Coercion logic. Coercions are certain type conversions that can implicitly
//! happen in certain places, e.g. weakening `&mut` to `&` or deref coercions
//! like going from `&Vec<T>` to `&[T]`.
//!
//! See https://doc.rust-lang.org/nomicon/coercions.html and
//! librustc_typeck/check/coercion.rs.

use chalk_ir::{cast::Cast, Mutability, TyVariableKind};
use hir_def::{expr::ExprId, lang_item::LangItemTarget};

use crate::{
    autoderef, infer::TypeMismatch, static_lifetime, Canonical, DomainGoal, FnPointer, FnSig,
    Interner, Solution, Substitution, Ty, TyBuilder, TyExt, TyKind,
};

use super::{InEnvironment, InferOk, InferResult, InferenceContext, TypeError};

impl<'a> InferenceContext<'a> {
    /// Unify two types, but may coerce the first one to the second one
    /// using "implicit coercion rules" if needed.
    pub(super) fn coerce(&mut self, from_ty: &Ty, to_ty: &Ty) -> bool {
        let from_ty = self.resolve_ty_shallow(from_ty);
        let to_ty = self.resolve_ty_shallow(to_ty);
        match self.coerce_inner(from_ty, &to_ty) {
            Ok(result) => {
                self.table.register_infer_ok(result);
                true
            }
            Err(_) => {
                // FIXME deal with error
                false
            }
        }
    }

    /// Merge two types from different branches, with possible coercion.
    ///
    /// Mostly this means trying to coerce one to the other, but
    ///  - if we have two function types for different functions or closures, we need to
    ///    coerce both to function pointers;
    ///  - if we were concerned with lifetime subtyping, we'd need to look for a
    ///    least upper bound.
    pub(super) fn coerce_merge_branch(&mut self, id: Option<ExprId>, ty1: &Ty, ty2: &Ty) -> Ty {
        let ty1 = self.resolve_ty_shallow(ty1);
        let ty2 = self.resolve_ty_shallow(ty2);
        // Special case: two function types. Try to coerce both to
        // pointers to have a chance at getting a match. See
        // https://github.com/rust-lang/rust/blob/7b805396bf46dce972692a6846ce2ad8481c5f85/src/librustc_typeck/check/coercion.rs#L877-L916
        let sig = match (ty1.kind(&Interner), ty2.kind(&Interner)) {
            (TyKind::FnDef(..), TyKind::FnDef(..))
            | (TyKind::Closure(..), TyKind::FnDef(..))
            | (TyKind::FnDef(..), TyKind::Closure(..))
            | (TyKind::Closure(..), TyKind::Closure(..)) => {
                // FIXME: we're ignoring safety here. To be more correct, if we have one FnDef and one Closure,
                // we should be coercing the closure to a fn pointer of the safety of the FnDef
                cov_mark::hit!(coerce_fn_reification);
                let sig = ty1.callable_sig(self.db).expect("FnDef without callable sig");
                Some(sig)
            }
            _ => None,
        };
        if let Some(sig) = sig {
            let target_ty = TyKind::Function(sig.to_fn_ptr()).intern(&Interner);
            let result1 = self.coerce_inner(ty1.clone(), &target_ty);
            let result2 = self.coerce_inner(ty2.clone(), &target_ty);
            if let (Ok(result1), Ok(result2)) = (result1, result2) {
                self.table.register_infer_ok(result1);
                self.table.register_infer_ok(result2);
                return target_ty;
            }
        }

        // It might not seem like it, but order is important here: ty1 is our
        // "previous" type, ty2 is the "new" one being added. If the previous
        // type is a type variable and the new one is `!`, trying it the other
        // way around first would mean we make the type variable `!`, instead of
        // just marking it as possibly diverging.
        if self.coerce(&ty2, &ty1) {
            ty1
        } else if self.coerce(&ty1, &ty2) {
            ty2
        } else {
            if let Some(id) = id {
                self.result
                    .type_mismatches
                    .insert(id.into(), TypeMismatch { expected: ty1.clone(), actual: ty2 });
            }
            cov_mark::hit!(coerce_merge_fail_fallback);
            ty1
        }
    }

    fn coerce_inner(&mut self, from_ty: Ty, to_ty: &Ty) -> InferResult {
        if from_ty.is_never() {
            // Subtle: If we are coercing from `!` to `?T`, where `?T` is an unbound
            // type variable, we want `?T` to fallback to `!` if not
            // otherwise constrained. An example where this arises:
            //
            //     let _: Option<?T> = Some({ return; });
            //
            // here, we would coerce from `!` to `?T`.
            match to_ty.kind(&Interner) {
                TyKind::InferenceVar(tv, TyVariableKind::General) => {
                    self.table.set_diverging(*tv, true);
                }
                _ => {}
            }
            return Ok(InferOk { goals: Vec::new() });
        }

        // Consider coercing the subtype to a DST
        if let Ok(ret) = self.try_coerce_unsized(&from_ty, &to_ty) {
            return Ok(ret);
        }

        // Examine the supertype and consider auto-borrowing.
        match to_ty.kind(&Interner) {
            TyKind::Raw(mt, _) => {
                return self.coerce_ptr(from_ty, to_ty, *mt);
            }
            TyKind::Ref(mt, _, _) => {
                return self.coerce_ref(from_ty, to_ty, *mt);
            }
            _ => {}
        }

        match from_ty.kind(&Interner) {
            TyKind::FnDef(..) => {
                // Function items are coercible to any closure
                // type; function pointers are not (that would
                // require double indirection).
                // Additionally, we permit coercion of function
                // items to drop the unsafe qualifier.
                self.coerce_from_fn_item(from_ty, to_ty)
            }
            TyKind::Function(from_fn_ptr) => {
                // We permit coercion of fn pointers to drop the
                // unsafe qualifier.
                self.coerce_from_fn_pointer(from_ty.clone(), from_fn_ptr, to_ty)
            }
            TyKind::Closure(_, from_substs) => {
                // Non-capturing closures are coercible to
                // function pointers or unsafe function pointers.
                // It cannot convert closures that require unsafe.
                self.coerce_closure_to_fn(from_ty.clone(), from_substs, to_ty)
            }
            _ => {
                // Otherwise, just use unification rules.
                self.table.try_unify(&from_ty, to_ty)
            }
        }
    }

    fn coerce_ptr(&mut self, from_ty: Ty, to_ty: &Ty, to_mt: Mutability) -> InferResult {
        let (_is_ref, from_mt, from_inner) = match from_ty.kind(&Interner) {
            TyKind::Ref(mt, _, ty) => (true, mt, ty),
            TyKind::Raw(mt, ty) => (false, mt, ty),
            _ => return self.table.try_unify(&from_ty, to_ty),
        };

        coerce_mutabilities(*from_mt, to_mt)?;

        // Check that the types which they point at are compatible.
        let from_raw = TyKind::Raw(to_mt, from_inner.clone()).intern(&Interner);
        // FIXME: behavior differs based on is_ref once we're computing adjustments
        self.table.try_unify(&from_raw, to_ty)
    }

    /// Reborrows `&mut A` to `&mut B` and `&(mut) A` to `&B`.
    /// To match `A` with `B`, autoderef will be performed,
    /// calling `deref`/`deref_mut` where necessary.
    fn coerce_ref(&mut self, from_ty: Ty, to_ty: &Ty, to_mt: Mutability) -> InferResult {
        match from_ty.kind(&Interner) {
            TyKind::Ref(mt, _, _) => {
                coerce_mutabilities(*mt, to_mt)?;
            }
            _ => return self.table.try_unify(&from_ty, to_ty),
        };

        // NOTE: this code is mostly copied and adapted from rustc, and
        // currently more complicated than necessary, carrying errors around
        // etc.. This complication will become necessary when we actually track
        // details of coercion errors though, so I think it's useful to leave
        // the structure like it is.

        let canonicalized = self.canonicalize(from_ty);
        let autoderef = autoderef::autoderef(
            self.db,
            self.resolver.krate(),
            InEnvironment {
                goal: canonicalized.value.clone(),
                environment: self.trait_env.env.clone(),
            },
        );
        let mut first_error = None;
        let mut found = None;

        for (autoderefs, referent_ty) in autoderef.enumerate() {
            if autoderefs == 0 {
                // Don't let this pass, otherwise it would cause
                // &T to autoref to &&T.
                continue;
            }

            let referent_ty = canonicalized.decanonicalize_ty(referent_ty.value);

            // At this point, we have deref'd `a` to `referent_ty`.  So
            // imagine we are coercing from `&'a mut Vec<T>` to `&'b mut [T]`.
            // In the autoderef loop for `&'a mut Vec<T>`, we would get
            // three callbacks:
            //
            // - `&'a mut Vec<T>` -- 0 derefs, just ignore it
            // - `Vec<T>` -- 1 deref
            // - `[T]` -- 2 deref
            //
            // At each point after the first callback, we want to
            // check to see whether this would match out target type
            // (`&'b mut [T]`) if we autoref'd it. We can't just
            // compare the referent types, though, because we still
            // have to consider the mutability. E.g., in the case
            // we've been considering, we have an `&mut` reference, so
            // the `T` in `[T]` needs to be unified with equality.
            //
            // Therefore, we construct reference types reflecting what
            // the types will be after we do the final auto-ref and
            // compare those. Note that this means we use the target
            // mutability [1], since it may be that we are coercing
            // from `&mut T` to `&U`.
            let lt = static_lifetime(); // FIXME: handle lifetimes correctly, see rustc
            let derefd_from_ty = TyKind::Ref(to_mt, lt, referent_ty).intern(&Interner);
            match self.table.try_unify(&derefd_from_ty, to_ty) {
                Ok(result) => {
                    found = Some(result);
                    break;
                }
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        }

        // Extract type or return an error. We return the first error
        // we got, which should be from relating the "base" type
        // (e.g., in example above, the failure from relating `Vec<T>`
        // to the target type), since that should be the least
        // confusing.
        let result = match found {
            Some(d) => d,
            None => {
                let err = first_error.expect("coerce_borrowed_pointer had no error");
                return Err(err);
            }
        };

        Ok(result)
    }

    /// Attempts to coerce from the type of a Rust function item into a function pointer.
    fn coerce_from_fn_item(&mut self, from_ty: Ty, to_ty: &Ty) -> InferResult {
        match to_ty.kind(&Interner) {
            TyKind::Function(_) => {
                let from_sig = from_ty.callable_sig(self.db).expect("FnDef had no sig");

                // FIXME check ABI: Intrinsics are not coercible to function pointers
                // FIXME Safe `#[target_feature]` functions are not assignable to safe fn pointers (RFC 2396)

                // FIXME rustc normalizes assoc types in the sig here, not sure if necessary

                let from_sig = from_sig.to_fn_ptr();
                let from_fn_pointer = TyKind::Function(from_sig.clone()).intern(&Interner);
                let ok = self.coerce_from_safe_fn(from_fn_pointer, &from_sig, to_ty)?;

                Ok(ok)
            }
            _ => self.table.try_unify(&from_ty, to_ty),
        }
    }

    fn coerce_from_fn_pointer(
        &mut self,
        from_ty: Ty,
        from_f: &FnPointer,
        to_ty: &Ty,
    ) -> InferResult {
        self.coerce_from_safe_fn(from_ty, from_f, to_ty)
    }

    fn coerce_from_safe_fn(
        &mut self,
        from_ty: Ty,
        from_fn_ptr: &FnPointer,
        to_ty: &Ty,
    ) -> InferResult {
        if let TyKind::Function(to_fn_ptr) = to_ty.kind(&Interner) {
            if let (chalk_ir::Safety::Safe, chalk_ir::Safety::Unsafe) =
                (from_fn_ptr.sig.safety, to_fn_ptr.sig.safety)
            {
                let from_unsafe =
                    TyKind::Function(safe_to_unsafe_fn_ty(from_fn_ptr.clone())).intern(&Interner);
                return self.table.try_unify(&from_unsafe, to_ty);
            }
        }
        self.table.try_unify(&from_ty, to_ty)
    }

    /// Attempts to coerce from the type of a non-capturing closure into a
    /// function pointer.
    fn coerce_closure_to_fn(
        &mut self,
        from_ty: Ty,
        from_substs: &Substitution,
        to_ty: &Ty,
    ) -> InferResult {
        match to_ty.kind(&Interner) {
            TyKind::Function(fn_ty) /* if from_substs is non-capturing (FIXME) */ => {
                // We coerce the closure, which has fn type
                //     `extern "rust-call" fn((arg0,arg1,...)) -> _`
                // to
                //     `fn(arg0,arg1,...) -> _`
                // or
                //     `unsafe fn(arg0,arg1,...) -> _`
                let safety = fn_ty.sig.safety;
                let pointer_ty = coerce_closure_fn_ty(from_substs, safety);
                self.table.try_unify(&pointer_ty, to_ty)
            }
            _ => self.table.try_unify(&from_ty, to_ty),
        }
    }

    /// Coerce a type using `from_ty: CoerceUnsized<ty_ty>`
    ///
    /// See: https://doc.rust-lang.org/nightly/std/marker/trait.CoerceUnsized.html
    fn try_coerce_unsized(&mut self, from_ty: &Ty, to_ty: &Ty) -> InferResult {
        // These 'if' statements require some explanation.
        // The `CoerceUnsized` trait is special - it is only
        // possible to write `impl CoerceUnsized<B> for A` where
        // A and B have 'matching' fields. This rules out the following
        // two types of blanket impls:
        //
        // `impl<T> CoerceUnsized<T> for SomeType`
        // `impl<T> CoerceUnsized<SomeType> for T`
        //
        // Both of these trigger a special `CoerceUnsized`-related error (E0376)
        //
        // We can take advantage of this fact to avoid performing unecessary work.
        // If either `source` or `target` is a type variable, then any applicable impl
        // would need to be generic over the self-type (`impl<T> CoerceUnsized<SomeType> for T`)
        // or generic over the `CoerceUnsized` type parameter (`impl<T> CoerceUnsized<T> for
        // SomeType`).
        //
        // However, these are exactly the kinds of impls which are forbidden by
        // the compiler! Therefore, we can be sure that coercion will always fail
        // when either the source or target type is a type variable. This allows us
        // to skip performing any trait selection, and immediately bail out.
        if from_ty.is_ty_var() {
            return Err(TypeError);
        }
        if to_ty.is_ty_var() {
            return Err(TypeError);
        }

        // Handle reborrows before trying to solve `Source: CoerceUnsized<Target>`.
        let coerce_from = match (from_ty.kind(&Interner), to_ty.kind(&Interner)) {
            (TyKind::Ref(from_mt, _, from_inner), TyKind::Ref(to_mt, _, _)) => {
                coerce_mutabilities(*from_mt, *to_mt)?;

                let lt = static_lifetime();
                TyKind::Ref(*to_mt, lt, from_inner.clone()).intern(&Interner)
            }
            (TyKind::Ref(from_mt, _, from_inner), TyKind::Raw(to_mt, _)) => {
                coerce_mutabilities(*from_mt, *to_mt)?;

                TyKind::Raw(*to_mt, from_inner.clone()).intern(&Interner)
            }
            _ => from_ty.clone(),
        };

        let krate = self.resolver.krate().unwrap();
        let coerce_unsized_trait = match self.db.lang_item(krate, "coerce_unsized".into()) {
            Some(LangItemTarget::TraitId(trait_)) => trait_,
            _ => return Err(TypeError),
        };

        let trait_ref = {
            let b = TyBuilder::trait_ref(self.db, coerce_unsized_trait);
            if b.remaining() != 2 {
                // The CoerceUnsized trait should have two generic params: Self and T.
                return Err(TypeError);
            }
            b.push(coerce_from).push(to_ty.clone()).build()
        };

        let goal: InEnvironment<DomainGoal> =
            InEnvironment::new(&self.trait_env.env, trait_ref.cast(&Interner));

        let canonicalized = self.canonicalize(goal);

        // FIXME: rustc's coerce_unsized is more specialized -- it only tries to
        // solve `CoerceUnsized` and `Unsize` goals at this point and leaves the
        // rest for later. Also, there's some logic about sized type variables.
        // Need to find out in what cases this is necessary
        let solution = self
            .db
            .trait_solve(krate, canonicalized.value.clone().cast(&Interner))
            .ok_or(TypeError)?;

        match solution {
            Solution::Unique(v) => {
                canonicalized.apply_solution(
                    &mut self.table,
                    Canonical {
                        binders: v.binders,
                        // FIXME handle constraints
                        value: v.value.subst,
                    },
                );
            }
            // FIXME: should we accept ambiguous results here?
            _ => return Err(TypeError),
        };

        Ok(InferOk { goals: Vec::new() })
    }
}

fn coerce_closure_fn_ty(closure_substs: &Substitution, safety: chalk_ir::Safety) -> Ty {
    let closure_sig = closure_substs.at(&Interner, 0).assert_ty_ref(&Interner).clone();
    match closure_sig.kind(&Interner) {
        TyKind::Function(fn_ty) => TyKind::Function(FnPointer {
            num_binders: fn_ty.num_binders,
            sig: FnSig { safety, ..fn_ty.sig },
            substitution: fn_ty.substitution.clone(),
        })
        .intern(&Interner),
        _ => TyKind::Error.intern(&Interner),
    }
}

fn safe_to_unsafe_fn_ty(fn_ty: FnPointer) -> FnPointer {
    FnPointer {
        num_binders: fn_ty.num_binders,
        sig: FnSig { safety: chalk_ir::Safety::Unsafe, ..fn_ty.sig },
        substitution: fn_ty.substitution,
    }
}

fn coerce_mutabilities(from: Mutability, to: Mutability) -> Result<(), TypeError> {
    match (from, to) {
        (Mutability::Mut, Mutability::Mut)
        | (Mutability::Mut, Mutability::Not)
        | (Mutability::Not, Mutability::Not) => Ok(()),
        (Mutability::Not, Mutability::Mut) => Err(TypeError),
    }
}
