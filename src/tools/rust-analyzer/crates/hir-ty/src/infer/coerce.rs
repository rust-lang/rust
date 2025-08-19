//! Coercion logic. Coercions are certain type conversions that can implicitly
//! happen in certain places, e.g. weakening `&mut` to `&` or deref coercions
//! like going from `&Vec<T>` to `&[T]`.
//!
//! See <https://doc.rust-lang.org/nomicon/coercions.html> and
//! `rustc_hir_analysis/check/coercion.rs`.

use std::iter;

use chalk_ir::{BoundVar, Goal, Mutability, TyKind, TyVariableKind, cast::Cast};
use hir_def::{hir::ExprId, lang_item::LangItem};
use stdx::always;
use triomphe::Arc;

use crate::{
    Canonical, DomainGoal, FnAbi, FnPointer, FnSig, Guidance, InEnvironment, Interner, Lifetime,
    Solution, Substitution, TraitEnvironment, Ty, TyBuilder, TyExt,
    autoderef::{Autoderef, AutoderefKind},
    db::HirDatabase,
    infer::{
        Adjust, Adjustment, AutoBorrow, InferOk, InferenceContext, OverloadedDeref, PointerCast,
        TypeError, TypeMismatch,
    },
    utils::ClosureSubst,
};

use super::unify::InferenceTable;

pub(crate) type CoerceResult = Result<InferOk<(Vec<Adjustment>, Ty)>, TypeError>;

/// Do not require any adjustments, i.e. coerce `x -> x`.
fn identity(_: Ty) -> Vec<Adjustment> {
    vec![]
}

fn simple(kind: Adjust) -> impl FnOnce(Ty) -> Vec<Adjustment> {
    move |target| vec![Adjustment { kind, target }]
}

/// This always returns `Ok(...)`.
fn success(
    adj: Vec<Adjustment>,
    target: Ty,
    goals: Vec<InEnvironment<Goal<Interner>>>,
) -> CoerceResult {
    Ok(InferOk { goals, value: (adj, target) })
}

pub(super) enum CoercionCause {
    // FIXME: Make better use of this. Right now things like return and break without a value
    // use it to point to themselves, causing us to report a mismatch on those expressions even
    // though technically they themselves are `!`
    Expr(ExprId),
}

#[derive(Clone, Debug)]
pub(super) struct CoerceMany {
    expected_ty: Ty,
    final_ty: Option<Ty>,
    expressions: Vec<ExprId>,
}

impl CoerceMany {
    pub(super) fn new(expected: Ty) -> Self {
        CoerceMany { expected_ty: expected, final_ty: None, expressions: vec![] }
    }

    /// Returns the "expected type" with which this coercion was
    /// constructed. This represents the "downward propagated" type
    /// that was given to us at the start of typing whatever construct
    /// we are typing (e.g., the match expression).
    ///
    /// Typically, this is used as the expected type when
    /// type-checking each of the alternative expressions whose types
    /// we are trying to merge.
    pub(super) fn expected_ty(&self) -> Ty {
        self.expected_ty.clone()
    }

    /// Returns the current "merged type", representing our best-guess
    /// at the LUB of the expressions we've seen so far (if any). This
    /// isn't *final* until you call `self.complete()`, which will return
    /// the merged type.
    pub(super) fn merged_ty(&self) -> Ty {
        self.final_ty.clone().unwrap_or_else(|| self.expected_ty.clone())
    }

    pub(super) fn complete(self, ctx: &mut InferenceContext<'_>) -> Ty {
        if let Some(final_ty) = self.final_ty {
            final_ty
        } else {
            ctx.result.standard_types.never.clone()
        }
    }

    pub(super) fn coerce_forced_unit(
        &mut self,
        ctx: &mut InferenceContext<'_>,
        cause: CoercionCause,
    ) {
        self.coerce(ctx, None, &ctx.result.standard_types.unit.clone(), cause)
    }

    /// Merge two types from different branches, with possible coercion.
    ///
    /// Mostly this means trying to coerce one to the other, but
    ///  - if we have two function types for different functions or closures, we need to
    ///    coerce both to function pointers;
    ///  - if we were concerned with lifetime subtyping, we'd need to look for a
    ///    least upper bound.
    pub(super) fn coerce(
        &mut self,
        ctx: &mut InferenceContext<'_>,
        expr: Option<ExprId>,
        expr_ty: &Ty,
        cause: CoercionCause,
    ) {
        let expr_ty = ctx.resolve_ty_shallow(expr_ty);
        self.expected_ty = ctx.resolve_ty_shallow(&self.expected_ty);

        // Special case: two function types. Try to coerce both to
        // pointers to have a chance at getting a match. See
        // https://github.com/rust-lang/rust/blob/7b805396bf46dce972692a6846ce2ad8481c5f85/src/librustc_typeck/check/coercion.rs#L877-L916
        let sig = match (self.merged_ty().kind(Interner), expr_ty.kind(Interner)) {
            (TyKind::FnDef(x, _), TyKind::FnDef(y, _))
                if x == y && ctx.table.unify(&self.merged_ty(), &expr_ty) =>
            {
                None
            }
            (TyKind::Closure(x, _), TyKind::Closure(y, _)) if x == y => None,
            (TyKind::FnDef(..) | TyKind::Closure(..), TyKind::FnDef(..) | TyKind::Closure(..)) => {
                // FIXME: we're ignoring safety here. To be more correct, if we have one FnDef and one Closure,
                // we should be coercing the closure to a fn pointer of the safety of the FnDef
                cov_mark::hit!(coerce_fn_reification);
                let sig =
                    self.merged_ty().callable_sig(ctx.db).expect("FnDef without callable sig");
                Some(sig)
            }
            _ => None,
        };
        if let Some(sig) = sig {
            let target_ty = TyKind::Function(sig.to_fn_ptr()).intern(Interner);
            let result1 = ctx.table.coerce_inner(self.merged_ty(), &target_ty, CoerceNever::Yes);
            let result2 = ctx.table.coerce_inner(expr_ty.clone(), &target_ty, CoerceNever::Yes);
            if let (Ok(result1), Ok(result2)) = (result1, result2) {
                ctx.table.register_infer_ok(InferOk { value: (), goals: result1.goals });
                for &e in &self.expressions {
                    ctx.write_expr_adj(e, result1.value.0.clone().into_boxed_slice());
                }
                ctx.table.register_infer_ok(InferOk { value: (), goals: result2.goals });
                if let Some(expr) = expr {
                    ctx.write_expr_adj(expr, result2.value.0.into_boxed_slice());
                    self.expressions.push(expr);
                }
                return self.final_ty = Some(target_ty);
            }
        }

        // It might not seem like it, but order is important here: If the expected
        // type is a type variable and the new one is `!`, trying it the other
        // way around first would mean we make the type variable `!`, instead of
        // just marking it as possibly diverging.
        //
        // - [Comment from rustc](https://github.com/rust-lang/rust/blob/5ff18d0eaefd1bd9ab8ec33dab2404a44e7631ed/compiler/rustc_hir_typeck/src/coercion.rs#L1334-L1335)
        // First try to coerce the new expression to the type of the previous ones,
        // but only if the new expression has no coercion already applied to it.
        if expr.is_none_or(|expr| !ctx.result.expr_adjustments.contains_key(&expr))
            && let Ok(res) = ctx.coerce(expr, &expr_ty, &self.merged_ty(), CoerceNever::Yes)
        {
            self.final_ty = Some(res);
            if let Some(expr) = expr {
                self.expressions.push(expr);
            }
            return;
        }

        if let Ok((adjustments, res)) =
            ctx.coerce_inner(&self.merged_ty(), &expr_ty, CoerceNever::Yes)
        {
            self.final_ty = Some(res);
            for &e in &self.expressions {
                ctx.write_expr_adj(e, adjustments.clone().into_boxed_slice());
            }
        } else {
            match cause {
                CoercionCause::Expr(id) => {
                    ctx.result.type_mismatches.insert(
                        id.into(),
                        TypeMismatch { expected: self.merged_ty(), actual: expr_ty.clone() },
                    );
                }
            }
            cov_mark::hit!(coerce_merge_fail_fallback);
        }
        if let Some(expr) = expr {
            self.expressions.push(expr);
        }
    }
}

pub fn could_coerce(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    tys: &Canonical<(Ty, Ty)>,
) -> bool {
    coerce(db, env, tys).is_ok()
}

pub(crate) fn coerce(
    db: &dyn HirDatabase,
    env: Arc<TraitEnvironment>,
    tys: &Canonical<(Ty, Ty)>,
) -> Result<(Vec<Adjustment>, Ty), TypeError> {
    let mut table = InferenceTable::new(db, env);
    let vars = table.fresh_subst(tys.binders.as_slice(Interner));
    let ty1_with_vars = vars.apply(tys.value.0.clone(), Interner);
    let ty2_with_vars = vars.apply(tys.value.1.clone(), Interner);
    let (adjustments, ty) = table.coerce(&ty1_with_vars, &ty2_with_vars, CoerceNever::Yes)?;
    // default any type vars that weren't unified back to their original bound vars
    // (kind of hacky)
    let find_var = |iv| {
        vars.iter(Interner).position(|v| match v.interned() {
            chalk_ir::GenericArgData::Ty(ty) => ty.inference_var(Interner),
            chalk_ir::GenericArgData::Lifetime(lt) => lt.inference_var(Interner),
            chalk_ir::GenericArgData::Const(c) => c.inference_var(Interner),
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
    // FIXME also map the types in the adjustments
    Ok((adjustments, table.resolve_with_fallback(ty, &fallback)))
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum CoerceNever {
    Yes,
    No,
}

impl InferenceContext<'_> {
    /// Unify two types, but may coerce the first one to the second one
    /// using "implicit coercion rules" if needed.
    pub(super) fn coerce(
        &mut self,
        expr: Option<ExprId>,
        from_ty: &Ty,
        to_ty: &Ty,
        // [Comment from rustc](https://github.com/rust-lang/rust/blob/4cc494bbfe9911d24f3ee521f98d5c6bb7e3ffe8/compiler/rustc_hir_typeck/src/coercion.rs#L85-L89)
        // Whether we allow `NeverToAny` coercions. This is unsound if we're
        // coercing a place expression without it counting as a read in the MIR.
        // This is a side-effect of HIR not really having a great distinction
        // between places and values.
        coerce_never: CoerceNever,
    ) -> Result<Ty, TypeError> {
        let (adjustments, ty) = self.coerce_inner(from_ty, to_ty, coerce_never)?;
        if let Some(expr) = expr {
            self.write_expr_adj(expr, adjustments.into_boxed_slice());
        }
        Ok(ty)
    }

    fn coerce_inner(
        &mut self,
        from_ty: &Ty,
        to_ty: &Ty,
        coerce_never: CoerceNever,
    ) -> Result<(Vec<Adjustment>, Ty), TypeError> {
        let from_ty = self.resolve_ty_shallow(from_ty);
        let to_ty = self.resolve_ty_shallow(to_ty);
        self.table.coerce(&from_ty, &to_ty, coerce_never)
    }
}

impl InferenceTable<'_> {
    /// Unify two types, but may coerce the first one to the second one
    /// using "implicit coercion rules" if needed.
    pub(crate) fn coerce(
        &mut self,
        from_ty: &Ty,
        to_ty: &Ty,
        coerce_never: CoerceNever,
    ) -> Result<(Vec<Adjustment>, Ty), TypeError> {
        let from_ty = self.resolve_ty_shallow(from_ty);
        let to_ty = self.resolve_ty_shallow(to_ty);
        match self.coerce_inner(from_ty, &to_ty, coerce_never) {
            Ok(InferOk { value: (adjustments, ty), goals }) => {
                self.register_infer_ok(InferOk { value: (), goals });
                Ok((adjustments, ty))
            }
            Err(e) => {
                // FIXME deal with error
                Err(e)
            }
        }
    }

    fn coerce_inner(&mut self, from_ty: Ty, to_ty: &Ty, coerce_never: CoerceNever) -> CoerceResult {
        if from_ty.is_never() {
            if let TyKind::InferenceVar(tv, TyVariableKind::General) = to_ty.kind(Interner) {
                self.set_diverging(*tv, true);
            }
            if coerce_never == CoerceNever::Yes {
                // Subtle: If we are coercing from `!` to `?T`, where `?T` is an unbound
                // type variable, we want `?T` to fallback to `!` if not
                // otherwise constrained. An example where this arises:
                //
                //     let _: Option<?T> = Some({ return; });
                //
                // here, we would coerce from `!` to `?T`.
                return success(simple(Adjust::NeverToAny)(to_ty.clone()), to_ty.clone(), vec![]);
            } else {
                return self.unify_and(&from_ty, to_ty, identity);
            }
        }

        // If we are coercing into a TAIT, coerce into its proxy inference var, instead.
        let mut to_ty = to_ty;
        let _to;
        if let Some(tait_table) = &self.tait_coercion_table
            && let TyKind::OpaqueType(opaque_ty_id, _) = to_ty.kind(Interner)
            && !matches!(from_ty.kind(Interner), TyKind::InferenceVar(..) | TyKind::OpaqueType(..))
            && let Some(ty) = tait_table.get(opaque_ty_id)
        {
            _to = ty.clone();
            to_ty = &_to;
        }

        // Consider coercing the subtype to a DST
        if let Ok(ret) = self.try_coerce_unsized(&from_ty, to_ty) {
            return Ok(ret);
        }

        // Examine the supertype and consider auto-borrowing.
        match to_ty.kind(Interner) {
            TyKind::Raw(mt, _) => return self.coerce_ptr(from_ty, to_ty, *mt),
            TyKind::Ref(mt, lt, _) => return self.coerce_ref(from_ty, to_ty, *mt, lt),
            _ => {}
        }

        match from_ty.kind(Interner) {
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
                self.unify_and(&from_ty, to_ty, identity)
            }
        }
    }

    /// Unify two types (using sub or lub) and produce a specific coercion.
    fn unify_and<F>(&mut self, t1: &Ty, t2: &Ty, f: F) -> CoerceResult
    where
        F: FnOnce(Ty) -> Vec<Adjustment>,
    {
        self.try_unify(t1, t2)
            .and_then(|InferOk { goals, .. }| success(f(t1.clone()), t1.clone(), goals))
    }

    fn coerce_ptr(&mut self, from_ty: Ty, to_ty: &Ty, to_mt: Mutability) -> CoerceResult {
        let (is_ref, from_mt, from_inner) = match from_ty.kind(Interner) {
            TyKind::Ref(mt, _, ty) => (true, mt, ty),
            TyKind::Raw(mt, ty) => (false, mt, ty),
            _ => return self.unify_and(&from_ty, to_ty, identity),
        };

        coerce_mutabilities(*from_mt, to_mt)?;

        // Check that the types which they point at are compatible.
        let from_raw = TyKind::Raw(to_mt, from_inner.clone()).intern(Interner);

        // Although references and raw ptrs have the same
        // representation, we still register an Adjust::DerefRef so that
        // regionck knows that the region for `a` must be valid here.
        if is_ref {
            self.unify_and(&from_raw, to_ty, |target| {
                vec![
                    Adjustment { kind: Adjust::Deref(None), target: from_inner.clone() },
                    Adjustment { kind: Adjust::Borrow(AutoBorrow::RawPtr(to_mt)), target },
                ]
            })
        } else if *from_mt != to_mt {
            self.unify_and(
                &from_raw,
                to_ty,
                simple(Adjust::Pointer(PointerCast::MutToConstPointer)),
            )
        } else {
            self.unify_and(&from_raw, to_ty, identity)
        }
    }

    /// Reborrows `&mut A` to `&mut B` and `&(mut) A` to `&B`.
    /// To match `A` with `B`, autoderef will be performed,
    /// calling `deref`/`deref_mut` where necessary.
    fn coerce_ref(
        &mut self,
        from_ty: Ty,
        to_ty: &Ty,
        to_mt: Mutability,
        to_lt: &Lifetime,
    ) -> CoerceResult {
        let (_from_lt, from_mt) = match from_ty.kind(Interner) {
            TyKind::Ref(mt, lt, _) => {
                coerce_mutabilities(*mt, to_mt)?;
                (lt.clone(), *mt) // clone is probably not good?
            }
            _ => return self.unify_and(&from_ty, to_ty, identity),
        };

        // NOTE: this code is mostly copied and adapted from rustc, and
        // currently more complicated than necessary, carrying errors around
        // etc.. This complication will become necessary when we actually track
        // details of coercion errors though, so I think it's useful to leave
        // the structure like it is.

        let snapshot = self.snapshot();

        let mut autoderef = Autoderef::new(self, from_ty.clone(), false, false);
        let mut first_error = None;
        let mut found = None;

        while let Some((referent_ty, autoderefs)) = autoderef.next() {
            if autoderefs == 0 {
                // Don't let this pass, otherwise it would cause
                // &T to autoref to &&T.
                continue;
            }

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
            let lt = to_lt; // FIXME: Involve rustc LUB and SUB flag checks
            let derefd_from_ty = TyKind::Ref(to_mt, lt.clone(), referent_ty).intern(Interner);
            match autoderef.table.try_unify(&derefd_from_ty, to_ty) {
                Ok(result) => {
                    found = Some(result.map(|()| derefd_from_ty));
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
        let InferOk { value: ty, goals } = match found {
            Some(d) => d,
            None => {
                self.rollback_to(snapshot);
                let err = first_error.expect("coerce_borrowed_pointer had no error");
                return Err(err);
            }
        };
        if ty == from_ty && from_mt == Mutability::Not && autoderef.step_count() == 1 {
            // As a special case, if we would produce `&'a *x`, that's
            // a total no-op. We end up with the type `&'a T` just as
            // we started with.  In that case, just skip it
            // altogether. This is just an optimization.
            //
            // Note that for `&mut`, we DO want to reborrow --
            // otherwise, this would be a move, which might be an
            // error. For example `foo(self.x)` where `self` and
            // `self.x` both have `&mut `type would be a move of
            // `self.x`, but we auto-coerce it to `foo(&mut *self.x)`,
            // which is a borrow.
            always!(to_mt == Mutability::Not); // can only coerce &T -> &U
            return success(vec![], ty, goals);
        }

        let mut adjustments = auto_deref_adjust_steps(&autoderef);
        adjustments.push(Adjustment {
            kind: Adjust::Borrow(AutoBorrow::Ref(to_lt.clone(), to_mt)),
            target: ty.clone(),
        });

        success(adjustments, ty, goals)
    }

    /// Attempts to coerce from the type of a Rust function item into a function pointer.
    fn coerce_from_fn_item(&mut self, from_ty: Ty, to_ty: &Ty) -> CoerceResult {
        match to_ty.kind(Interner) {
            TyKind::Function(_) => {
                let from_sig = from_ty.callable_sig(self.db).expect("FnDef had no sig");

                // FIXME check ABI: Intrinsics are not coercible to function pointers
                // FIXME Safe `#[target_feature]` functions are not assignable to safe fn pointers (RFC 2396)

                // FIXME rustc normalizes assoc types in the sig here, not sure if necessary

                let from_sig = from_sig.to_fn_ptr();
                let from_fn_pointer = TyKind::Function(from_sig.clone()).intern(Interner);
                let ok = self.coerce_from_safe_fn(
                    from_fn_pointer.clone(),
                    &from_sig,
                    to_ty,
                    |unsafe_ty| {
                        vec![
                            Adjustment {
                                kind: Adjust::Pointer(PointerCast::ReifyFnPointer),
                                target: from_fn_pointer,
                            },
                            Adjustment {
                                kind: Adjust::Pointer(PointerCast::UnsafeFnPointer),
                                target: unsafe_ty,
                            },
                        ]
                    },
                    simple(Adjust::Pointer(PointerCast::ReifyFnPointer)),
                )?;

                Ok(ok)
            }
            _ => self.unify_and(&from_ty, to_ty, identity),
        }
    }

    fn coerce_from_fn_pointer(
        &mut self,
        from_ty: Ty,
        from_f: &FnPointer,
        to_ty: &Ty,
    ) -> CoerceResult {
        self.coerce_from_safe_fn(
            from_ty,
            from_f,
            to_ty,
            simple(Adjust::Pointer(PointerCast::UnsafeFnPointer)),
            identity,
        )
    }

    fn coerce_from_safe_fn<F, G>(
        &mut self,
        from_ty: Ty,
        from_fn_ptr: &FnPointer,
        to_ty: &Ty,
        to_unsafe: F,
        normal: G,
    ) -> CoerceResult
    where
        F: FnOnce(Ty) -> Vec<Adjustment>,
        G: FnOnce(Ty) -> Vec<Adjustment>,
    {
        if let TyKind::Function(to_fn_ptr) = to_ty.kind(Interner)
            && let (chalk_ir::Safety::Safe, chalk_ir::Safety::Unsafe) =
                (from_fn_ptr.sig.safety, to_fn_ptr.sig.safety)
        {
            let from_unsafe =
                TyKind::Function(safe_to_unsafe_fn_ty(from_fn_ptr.clone())).intern(Interner);
            return self.unify_and(&from_unsafe, to_ty, to_unsafe);
        }
        self.unify_and(&from_ty, to_ty, normal)
    }

    /// Attempts to coerce from the type of a non-capturing closure into a
    /// function pointer.
    fn coerce_closure_to_fn(
        &mut self,
        from_ty: Ty,
        from_substs: &Substitution,
        to_ty: &Ty,
    ) -> CoerceResult {
        match to_ty.kind(Interner) {
            // if from_substs is non-capturing (FIXME)
            TyKind::Function(fn_ty) => {
                // We coerce the closure, which has fn type
                //     `extern "rust-call" fn((arg0,arg1,...)) -> _`
                // to
                //     `fn(arg0,arg1,...) -> _`
                // or
                //     `unsafe fn(arg0,arg1,...) -> _`
                let safety = fn_ty.sig.safety;
                let pointer_ty = coerce_closure_fn_ty(from_substs, safety);
                self.unify_and(
                    &pointer_ty,
                    to_ty,
                    simple(Adjust::Pointer(PointerCast::ClosureFnPointer(safety))),
                )
            }
            _ => self.unify_and(&from_ty, to_ty, identity),
        }
    }

    /// Coerce a type using `from_ty: CoerceUnsized<ty_ty>`
    ///
    /// See: <https://doc.rust-lang.org/nightly/std/marker/trait.CoerceUnsized.html>
    fn try_coerce_unsized(&mut self, from_ty: &Ty, to_ty: &Ty) -> CoerceResult {
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
        // We can take advantage of this fact to avoid performing unnecessary work.
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
        let reborrow = match (from_ty.kind(Interner), to_ty.kind(Interner)) {
            (TyKind::Ref(from_mt, _, from_inner), &TyKind::Ref(to_mt, _, _)) => {
                coerce_mutabilities(*from_mt, to_mt)?;

                let lt = self.new_lifetime_var();
                Some((
                    Adjustment { kind: Adjust::Deref(None), target: from_inner.clone() },
                    Adjustment {
                        kind: Adjust::Borrow(AutoBorrow::Ref(lt.clone(), to_mt)),
                        target: TyKind::Ref(to_mt, lt, from_inner.clone()).intern(Interner),
                    },
                ))
            }
            (TyKind::Ref(from_mt, _, from_inner), &TyKind::Raw(to_mt, _)) => {
                coerce_mutabilities(*from_mt, to_mt)?;

                Some((
                    Adjustment { kind: Adjust::Deref(None), target: from_inner.clone() },
                    Adjustment {
                        kind: Adjust::Borrow(AutoBorrow::RawPtr(to_mt)),
                        target: TyKind::Raw(to_mt, from_inner.clone()).intern(Interner),
                    },
                ))
            }
            _ => None,
        };
        let coerce_from =
            reborrow.as_ref().map_or_else(|| from_ty.clone(), |(_, adj)| adj.target.clone());

        let krate = self.trait_env.krate;
        let coerce_unsized_trait = match LangItem::CoerceUnsized.resolve_trait(self.db, krate) {
            Some(trait_) => trait_,
            _ => return Err(TypeError),
        };

        let coerce_unsized_tref = {
            let b = TyBuilder::trait_ref(self.db, coerce_unsized_trait);
            if b.remaining() != 2 {
                // The CoerceUnsized trait should have two generic params: Self and T.
                return Err(TypeError);
            }
            b.push(coerce_from).push(to_ty.clone()).build()
        };

        let goal: InEnvironment<DomainGoal> =
            InEnvironment::new(&self.trait_env.env, coerce_unsized_tref.cast(Interner));

        let canonicalized = self.canonicalize_with_free_vars(goal);

        // FIXME: rustc's coerce_unsized is more specialized -- it only tries to
        // solve `CoerceUnsized` and `Unsize` goals at this point and leaves the
        // rest for later. Also, there's some logic about sized type variables.
        // Need to find out in what cases this is necessary
        let solution = self
            .db
            .trait_solve(krate, self.trait_env.block, canonicalized.value.clone().cast(Interner))
            .ok_or(TypeError)?;

        match solution {
            Solution::Unique(v) => {
                canonicalized.apply_solution(
                    self,
                    Canonical {
                        binders: v.binders,
                        // FIXME handle constraints
                        value: v.value.subst,
                    },
                );
            }
            Solution::Ambig(Guidance::Definite(subst)) => {
                // FIXME need to record an obligation here
                canonicalized.apply_solution(self, subst)
            }
            // FIXME actually we maybe should also accept unknown guidance here
            _ => return Err(TypeError),
        };
        let unsize =
            Adjustment { kind: Adjust::Pointer(PointerCast::Unsize), target: to_ty.clone() };
        let adjustments = match reborrow {
            None => vec![unsize],
            Some((deref, autoref)) => vec![deref, autoref, unsize],
        };
        success(adjustments, to_ty.clone(), vec![])
    }
}

fn coerce_closure_fn_ty(closure_substs: &Substitution, safety: chalk_ir::Safety) -> Ty {
    let closure_sig = ClosureSubst(closure_substs).sig_ty().clone();
    match closure_sig.kind(Interner) {
        TyKind::Function(fn_ty) => TyKind::Function(FnPointer {
            num_binders: fn_ty.num_binders,
            sig: FnSig { safety, abi: FnAbi::Rust, variadic: fn_ty.sig.variadic },
            substitution: fn_ty.substitution.clone(),
        })
        .intern(Interner),
        _ => TyKind::Error.intern(Interner),
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
        (Mutability::Mut, Mutability::Mut | Mutability::Not)
        | (Mutability::Not, Mutability::Not) => Ok(()),
        (Mutability::Not, Mutability::Mut) => Err(TypeError),
    }
}

pub(super) fn auto_deref_adjust_steps(autoderef: &Autoderef<'_, '_>) -> Vec<Adjustment> {
    let steps = autoderef.steps();
    let targets =
        steps.iter().skip(1).map(|(_, ty)| ty.clone()).chain(iter::once(autoderef.final_ty()));
    steps
        .iter()
        .map(|(kind, _source)| match kind {
            // We do not know what kind of deref we require at this point yet
            AutoderefKind::Overloaded => Some(OverloadedDeref(None)),
            AutoderefKind::Builtin => None,
        })
        .zip(targets)
        .map(|(autoderef, target)| Adjustment { kind: Adjust::Deref(autoderef), target })
        .collect()
}
