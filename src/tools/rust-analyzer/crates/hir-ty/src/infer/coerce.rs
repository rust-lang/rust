//! # Type Coercion
//!
//! Under certain circumstances we will coerce from one type to another,
//! for example by auto-borrowing. This occurs in situations where the
//! compiler has a firm 'expected type' that was supplied from the user,
//! and where the actual type is similar to that expected type in purpose
//! but not in representation (so actual subtyping is inappropriate).
//!
//! ## Reborrowing
//!
//! Note that if we are expecting a reference, we will *reborrow*
//! even if the argument provided was already a reference. This is
//! useful for freezing mut things (that is, when the expected type is &T
//! but you have &mut T) and also for avoiding the linearity
//! of mut things (when the expected is &mut T and you have &mut T). See
//! the various `tests/ui/coerce/*.rs` tests for
//! examples of where this is useful.
//!
//! ## Subtle note
//!
//! When inferring the generic arguments of functions, the argument
//! order is relevant, which can lead to the following edge case:
//!
//! ```ignore (illustrative)
//! fn foo<T>(a: T, b: T) {
//!     // ...
//! }
//!
//! foo(&7i32, &mut 7i32);
//! // This compiles, as we first infer `T` to be `&i32`,
//! // and then coerce `&mut 7i32` to `&7i32`.
//!
//! foo(&mut 7i32, &7i32);
//! // This does not compile, as we first infer `T` to be `&mut i32`
//! // and are then unable to coerce `&7i32` to `&mut i32`.
//! ```

use hir_def::{
    CallableDefId,
    hir::{ExprId, ExprOrPatId},
    lang_item::LangItem,
    signatures::FunctionSignature,
};
use intern::sym;
use rustc_ast_ir::Mutability;
use rustc_type_ir::{
    BoundVar, TypeAndMut,
    error::TypeError,
    inherent::{Const as _, GenericArg as _, IntoKind, Safety, SliceLike, Ty as _},
};
use smallvec::{SmallVec, smallvec};
use tracing::{debug, instrument};
use triomphe::Arc;

use crate::{
    Adjust, Adjustment, AutoBorrow, PointerCast, TargetFeatures, TraitEnvironment,
    autoderef::Autoderef,
    db::{HirDatabase, InternedClosureId},
    infer::{AllowTwoPhase, InferenceContext, TypeMismatch, unify::InferenceTable},
    next_solver::{
        Binder, BoundConst, BoundRegion, BoundRegionKind, BoundTy, BoundTyKind, CallableIdWrapper,
        Canonical, ClauseKind, CoercePredicate, Const, ConstKind, DbInterner, ErrorGuaranteed,
        GenericArgs, PolyFnSig, PredicateKind, Region, RegionKind, TraitRef, Ty, TyKind,
        infer::{
            InferCtxt, InferOk, InferResult,
            relate::RelateResult,
            select::{ImplSource, SelectionError},
            traits::{Obligation, ObligationCause, PredicateObligation, PredicateObligations},
        },
        obligation_ctxt::ObligationCtxt,
    },
    utils::TargetFeatureIsSafeInTarget,
};

struct Coerce<'a, 'b, 'db> {
    table: &'a mut InferenceTable<'db>,
    has_errors: &'a mut bool,
    target_features: &'a mut dyn FnMut() -> (&'b TargetFeatures, TargetFeatureIsSafeInTarget),
    use_lub: bool,
    /// Determines whether or not allow_two_phase_borrow is set on any
    /// autoref adjustments we create while coercing. We don't want to
    /// allow deref coercions to create two-phase borrows, at least initially,
    /// but we do need two-phase borrows for function argument reborrows.
    /// See rust#47489 and rust#48598
    /// See docs on the "AllowTwoPhase" type for a more detailed discussion
    allow_two_phase: AllowTwoPhase,
    /// Whether we allow `NeverToAny` coercions. This is unsound if we're
    /// coercing a place expression without it counting as a read in the MIR.
    /// This is a side-effect of HIR not really having a great distinction
    /// between places and values.
    coerce_never: bool,
    cause: ObligationCause,
}

type CoerceResult<'db> = InferResult<'db, (Vec<Adjustment<'db>>, Ty<'db>)>;

/// Coercing a mutable reference to an immutable works, while
/// coercing `&T` to `&mut T` should be forbidden.
fn coerce_mutbls<'db>(from_mutbl: Mutability, to_mutbl: Mutability) -> RelateResult<'db, ()> {
    if from_mutbl >= to_mutbl { Ok(()) } else { Err(TypeError::Mutability) }
}

/// This always returns `Ok(...)`.
fn success<'db>(
    adj: Vec<Adjustment<'db>>,
    target: Ty<'db>,
    obligations: PredicateObligations<'db>,
) -> CoerceResult<'db> {
    Ok(InferOk { value: (adj, target), obligations })
}

impl<'a, 'b, 'db> Coerce<'a, 'b, 'db> {
    #[inline]
    fn set_tainted_by_errors(&mut self) {
        *self.has_errors = true;
    }

    #[inline]
    fn interner(&self) -> DbInterner<'db> {
        self.table.interner()
    }

    #[inline]
    fn infer_ctxt(&self) -> &InferCtxt<'db> {
        &self.table.infer_ctxt
    }

    pub(crate) fn commit_if_ok<T, E>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, E>,
    ) -> Result<T, E> {
        let snapshot = self.table.snapshot();
        let result = f(self);
        match result {
            Ok(_) => {}
            Err(_) => {
                self.table.rollback_to(snapshot);
            }
        }
        result
    }

    fn unify_raw(&mut self, a: Ty<'db>, b: Ty<'db>) -> InferResult<'db, Ty<'db>> {
        debug!("unify(a: {:?}, b: {:?}, use_lub: {})", a, b, self.use_lub);
        self.commit_if_ok(|this| {
            let at = this.infer_ctxt().at(&this.cause, this.table.trait_env.env);

            let res = if this.use_lub {
                at.lub(b, a)
            } else {
                at.sup(b, a)
                    .map(|InferOk { value: (), obligations }| InferOk { value: b, obligations })
            };

            // In the new solver, lazy norm may allow us to shallowly equate
            // more types, but we emit possibly impossible-to-satisfy obligations.
            // Filter these cases out to make sure our coercion is more accurate.
            match res {
                Ok(InferOk { value, obligations }) => {
                    let mut ocx = ObligationCtxt::new(this.infer_ctxt());
                    ocx.register_obligations(obligations);
                    if ocx.try_evaluate_obligations().is_empty() {
                        Ok(InferOk { value, obligations: ocx.into_pending_obligations() })
                    } else {
                        Err(TypeError::Mismatch)
                    }
                }
                res => res,
            }
        })
    }

    /// Unify two types (using sub or lub).
    fn unify(&mut self, a: Ty<'db>, b: Ty<'db>) -> CoerceResult<'db> {
        self.unify_raw(a, b)
            .and_then(|InferOk { value: ty, obligations }| success(vec![], ty, obligations))
    }

    /// Unify two types (using sub or lub) and produce a specific coercion.
    fn unify_and(
        &mut self,
        a: Ty<'db>,
        b: Ty<'db>,
        adjustments: impl IntoIterator<Item = Adjustment<'db>>,
        final_adjustment: Adjust<'db>,
    ) -> CoerceResult<'db> {
        self.unify_raw(a, b).and_then(|InferOk { value: ty, obligations }| {
            success(
                adjustments
                    .into_iter()
                    .chain(std::iter::once(Adjustment { target: ty, kind: final_adjustment }))
                    .collect(),
                ty,
                obligations,
            )
        })
    }

    #[instrument(skip(self))]
    fn coerce(&mut self, a: Ty<'db>, b: Ty<'db>) -> CoerceResult<'db> {
        // First, remove any resolved type variables (at the top level, at least):
        let a = self.table.shallow_resolve(a);
        let b = self.table.shallow_resolve(b);
        debug!("Coerce.tys({:?} => {:?})", a, b);

        // Coercing from `!` to any type is allowed:
        if a.is_never() {
            // If we're coercing into an inference var, mark it as possibly diverging.
            if b.is_infer() {
                self.table.set_diverging(b);
            }

            if self.coerce_never {
                return success(
                    vec![Adjustment { kind: Adjust::NeverToAny, target: b }],
                    b,
                    PredicateObligations::new(),
                );
            } else {
                // Otherwise the only coercion we can do is unification.
                return self.unify(a, b);
            }
        }

        // Coercing *from* an unresolved inference variable means that
        // we have no information about the source type. This will always
        // ultimately fall back to some form of subtyping.
        if a.is_infer() {
            return self.coerce_from_inference_variable(a, b);
        }

        // Consider coercing the subtype to a DST
        //
        // NOTE: this is wrapped in a `commit_if_ok` because it creates
        // a "spurious" type variable, and we don't want to have that
        // type variable in memory if the coercion fails.
        let unsize = self.commit_if_ok(|this| this.coerce_unsized(a, b));
        match unsize {
            Ok(_) => {
                debug!("coerce: unsize successful");
                return unsize;
            }
            Err(error) => {
                debug!(?error, "coerce: unsize failed");
            }
        }

        // Examine the supertype and consider type-specific coercions, such
        // as auto-borrowing, coercing pointer mutability, a `dyn*` coercion,
        // or pin-ergonomics.
        match b.kind() {
            TyKind::RawPtr(_, b_mutbl) => {
                return self.coerce_raw_ptr(a, b, b_mutbl);
            }
            TyKind::Ref(r_b, _, mutbl_b) => {
                return self.coerce_borrowed_pointer(a, b, r_b, mutbl_b);
            }
            _ => {}
        }

        match a.kind() {
            TyKind::FnDef(..) => {
                // Function items are coercible to any closure
                // type; function pointers are not (that would
                // require double indirection).
                // Additionally, we permit coercion of function
                // items to drop the unsafe qualifier.
                self.coerce_from_fn_item(a, b)
            }
            TyKind::FnPtr(a_sig_tys, a_hdr) => {
                // We permit coercion of fn pointers to drop the
                // unsafe qualifier.
                self.coerce_from_fn_pointer(a_sig_tys.with(a_hdr), b)
            }
            TyKind::Closure(closure_def_id_a, args_a) => {
                // Non-capturing closures are coercible to
                // function pointers or unsafe function pointers.
                // It cannot convert closures that require unsafe.
                self.coerce_closure_to_fn(a, closure_def_id_a.0, args_a, b)
            }
            _ => {
                // Otherwise, just use unification rules.
                self.unify(a, b)
            }
        }
    }

    /// Coercing *from* an inference variable. In this case, we have no information
    /// about the source type, so we can't really do a true coercion and we always
    /// fall back to subtyping (`unify_and`).
    fn coerce_from_inference_variable(&mut self, a: Ty<'db>, b: Ty<'db>) -> CoerceResult<'db> {
        debug!("coerce_from_inference_variable(a={:?}, b={:?})", a, b);
        debug_assert!(a.is_infer() && self.table.shallow_resolve(a) == a);
        debug_assert!(self.table.shallow_resolve(b) == b);

        if b.is_infer() {
            // Two unresolved type variables: create a `Coerce` predicate.
            let target_ty = if self.use_lub { self.table.next_ty_var() } else { b };

            let mut obligations = PredicateObligations::with_capacity(2);
            for &source_ty in &[a, b] {
                if source_ty != target_ty {
                    obligations.push(Obligation::new(
                        self.interner(),
                        self.cause.clone(),
                        self.table.trait_env.env,
                        Binder::dummy(PredicateKind::Coerce(CoercePredicate {
                            a: source_ty,
                            b: target_ty,
                        })),
                    ));
                }
            }

            debug!(
                "coerce_from_inference_variable: two inference variables, target_ty={:?}, obligations={:?}",
                target_ty, obligations
            );
            success(vec![], target_ty, obligations)
        } else {
            // One unresolved type variable: just apply subtyping, we may be able
            // to do something useful.
            self.unify(a, b)
        }
    }

    /// Reborrows `&mut A` to `&mut B` and `&(mut) A` to `&B`.
    /// To match `A` with `B`, autoderef will be performed,
    /// calling `deref`/`deref_mut` where necessary.
    fn coerce_borrowed_pointer(
        &mut self,
        a: Ty<'db>,
        b: Ty<'db>,
        r_b: Region<'db>,
        mutbl_b: Mutability,
    ) -> CoerceResult<'db> {
        debug!("coerce_borrowed_pointer(a={:?}, b={:?})", a, b);
        debug_assert!(self.table.shallow_resolve(a) == a);
        debug_assert!(self.table.shallow_resolve(b) == b);

        // If we have a parameter of type `&M T_a` and the value
        // provided is `expr`, we will be adding an implicit borrow,
        // meaning that we convert `f(expr)` to `f(&M *expr)`. Therefore,
        // to type check, we will construct the type that `&M*expr` would
        // yield.

        let (r_a, mt_a) = match a.kind() {
            TyKind::Ref(r_a, ty, mutbl) => {
                let mt_a = TypeAndMut::<DbInterner<'db>> { ty, mutbl };
                coerce_mutbls(mt_a.mutbl, mutbl_b)?;
                (r_a, mt_a)
            }
            _ => return self.unify(a, b),
        };

        let mut first_error = None;
        let mut r_borrow_var = None;
        let mut autoderef = Autoderef::new(self.table, a);
        let mut found = None;

        while let Some((referent_ty, autoderefs)) = autoderef.next() {
            if autoderefs == 0 {
                // Don't let this pass, otherwise it would cause
                // &T to autoref to &&T.
                continue;
            }

            // At this point, we have deref'd `a` to `referent_ty`. So
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
            //
            // One fine point concerns the region that we use. We
            // choose the region such that the region of the final
            // type that results from `unify` will be the region we
            // want for the autoref:
            //
            // - if in sub mode, that means we want to use `'b` (the
            //   region from the target reference) for both
            //   pointers [2]. This is because sub mode (somewhat
            //   arbitrarily) returns the subtype region. In the case
            //   where we are coercing to a target type, we know we
            //   want to use that target type region (`'b`) because --
            //   for the program to type-check -- it must be the
            //   smaller of the two.
            //   - One fine point. It may be surprising that we can
            //     use `'b` without relating `'a` and `'b`. The reason
            //     that this is ok is that what we produce is
            //     effectively a `&'b *x` expression (if you could
            //     annotate the region of a borrow), and regionck has
            //     code that adds edges from the region of a borrow
            //     (`'b`, here) into the regions in the borrowed
            //     expression (`*x`, here). (Search for "link".)
            // - if in lub mode, things can get fairly complicated. The
            //   easiest thing is just to make a fresh
            //   region variable [4], which effectively means we defer
            //   the decision to region inference (and regionck, which will add
            //   some more edges to this variable). However, this can wind up
            //   creating a crippling number of variables in some cases --
            //   e.g., #32278 -- so we optimize one particular case [3].
            //   Let me try to explain with some examples:
            //   - The "running example" above represents the simple case,
            //     where we have one `&` reference at the outer level and
            //     ownership all the rest of the way down. In this case,
            //     we want `LUB('a, 'b)` as the resulting region.
            //   - However, if there are nested borrows, that region is
            //     too strong. Consider a coercion from `&'a &'x Rc<T>` to
            //     `&'b T`. In this case, `'a` is actually irrelevant.
            //     The pointer we want is `LUB('x, 'b`). If we choose `LUB('a,'b)`
            //     we get spurious errors (`ui/regions-lub-ref-ref-rc.rs`).
            //     (The errors actually show up in borrowck, typically, because
            //     this extra edge causes the region `'a` to be inferred to something
            //     too big, which then results in borrowck errors.)
            //   - We could track the innermost shared reference, but there is already
            //     code in regionck that has the job of creating links between
            //     the region of a borrow and the regions in the thing being
            //     borrowed (here, `'a` and `'x`), and it knows how to handle
            //     all the various cases. So instead we just make a region variable
            //     and let regionck figure it out.
            let r = if !self.use_lub {
                r_b // [2] above
            } else if autoderefs == 1 {
                r_a // [3] above
            } else {
                if r_borrow_var.is_none() {
                    // create var lazily, at most once
                    let r = autoderef.table.next_region_var();
                    r_borrow_var = Some(r); // [4] above
                }
                r_borrow_var.unwrap()
            };
            let derefd_ty_a = Ty::new_ref(
                autoderef.table.interner(),
                r,
                referent_ty,
                mutbl_b, // [1] above
            );
            // We need to construct a new `Coerce` because of lifetimes.
            let mut coerce = Coerce {
                table: autoderef.table,
                has_errors: self.has_errors,
                target_features: self.target_features,
                use_lub: self.use_lub,
                allow_two_phase: self.allow_two_phase,
                coerce_never: self.coerce_never,
                cause: self.cause.clone(),
            };
            match coerce.unify_raw(derefd_ty_a, b) {
                Ok(ok) => {
                    found = Some(ok);
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
        let Some(InferOk { value: ty, mut obligations }) = found else {
            if let Some(first_error) = first_error {
                debug!("coerce_borrowed_pointer: failed with err = {:?}", first_error);
                return Err(first_error);
            } else {
                // This may happen in the new trait solver since autoderef requires
                // the pointee to be structurally normalizable, or else it'll just bail.
                // So when we have a type like `&<not well formed>`, then we get no
                // autoderef steps (even though there should be at least one). That means
                // we get no type mismatches, since the loop above just exits early.
                return Err(TypeError::Mismatch);
            }
        };

        if ty == a && mt_a.mutbl.is_not() && autoderef.step_count() == 1 {
            // As a special case, if we would produce `&'a *x`, that's
            // a total no-op. We end up with the type `&'a T` just as
            // we started with. In that case, just skip it
            // altogether. This is just an optimization.
            //
            // Note that for `&mut`, we DO want to reborrow --
            // otherwise, this would be a move, which might be an
            // error. For example `foo(self.x)` where `self` and
            // `self.x` both have `&mut `type would be a move of
            // `self.x`, but we auto-coerce it to `foo(&mut *self.x)`,
            // which is a borrow.
            assert!(mutbl_b.is_not()); // can only coerce &T -> &U
            return success(vec![], ty, obligations);
        }

        let InferOk { value: mut adjustments, obligations: o } =
            autoderef.adjust_steps_as_infer_ok();
        obligations.extend(o);

        // Now apply the autoref. We have to extract the region out of
        // the final ref type we got.
        let TyKind::Ref(region, _, _) = ty.kind() else {
            panic!("expected a ref type, got {:?}", ty);
        };
        adjustments.push(Adjustment {
            kind: Adjust::Borrow(AutoBorrow::Ref(region, mutbl_b)),
            target: ty,
        });

        debug!("coerce_borrowed_pointer: succeeded ty={:?} adjustments={:?}", ty, adjustments);

        success(adjustments, ty, obligations)
    }

    /// Performs [unsized coercion] by emulating a fulfillment loop on a
    /// `CoerceUnsized` goal until all `CoerceUnsized` and `Unsize` goals
    /// are successfully selected.
    ///
    /// [unsized coercion](https://doc.rust-lang.org/reference/type-coercions.html#unsized-coercions)
    #[instrument(skip(self), level = "debug")]
    fn coerce_unsized(&mut self, source: Ty<'db>, target: Ty<'db>) -> CoerceResult<'db> {
        debug!(?source, ?target);
        debug_assert!(self.table.shallow_resolve(source) == source);
        debug_assert!(self.table.shallow_resolve(target) == target);

        // We don't apply any coercions incase either the source or target
        // aren't sufficiently well known but tend to instead just equate
        // them both.
        if source.is_infer() {
            debug!("coerce_unsized: source is a TyVar, bailing out");
            return Err(TypeError::Mismatch);
        }
        if target.is_infer() {
            debug!("coerce_unsized: target is a TyVar, bailing out");
            return Err(TypeError::Mismatch);
        }

        // This is an optimization because coercion is one of the most common
        // operations that we do in typeck, since it happens at every assignment
        // and call arg (among other positions).
        //
        // These targets are known to never be RHS in `LHS: CoerceUnsized<RHS>`.
        // That's because these are built-in types for which a core-provided impl
        // doesn't exist, and for which a user-written impl is invalid.
        //
        // This is technically incomplete when users write impossible bounds like
        // `where T: CoerceUnsized<usize>`, for example, but that trait is unstable
        // and coercion is allowed to be incomplete. The only case where this matters
        // is impossible bounds.
        //
        // Note that some of these types implement `LHS: Unsize<RHS>`, but they
        // do not implement *`CoerceUnsized`* which is the root obligation of the
        // check below.
        match target.kind() {
            TyKind::Bool
            | TyKind::Char
            | TyKind::Int(_)
            | TyKind::Uint(_)
            | TyKind::Float(_)
            | TyKind::Infer(rustc_type_ir::IntVar(_) | rustc_type_ir::FloatVar(_))
            | TyKind::Str
            | TyKind::Array(_, _)
            | TyKind::Slice(_)
            | TyKind::FnDef(_, _)
            | TyKind::FnPtr(_, _)
            | TyKind::Dynamic(_, _)
            | TyKind::Closure(_, _)
            | TyKind::CoroutineClosure(_, _)
            | TyKind::Coroutine(_, _)
            | TyKind::CoroutineWitness(_, _)
            | TyKind::Never
            | TyKind::Tuple(_) => return Err(TypeError::Mismatch),
            _ => {}
        }
        // Additionally, we ignore `&str -> &str` coercions, which happen very
        // commonly since strings are one of the most used argument types in Rust,
        // we do coercions when type checking call expressions.
        if let TyKind::Ref(_, source_pointee, Mutability::Not) = source.kind()
            && source_pointee.is_str()
            && let TyKind::Ref(_, target_pointee, Mutability::Not) = target.kind()
            && target_pointee.is_str()
        {
            return Err(TypeError::Mismatch);
        }

        let traits = (
            LangItem::Unsize.resolve_trait(self.table.db, self.table.trait_env.krate),
            LangItem::CoerceUnsized.resolve_trait(self.table.db, self.table.trait_env.krate),
        );
        let (Some(unsize_did), Some(coerce_unsized_did)) = traits else {
            debug!("missing Unsize or CoerceUnsized traits");
            return Err(TypeError::Mismatch);
        };

        // Note, we want to avoid unnecessary unsizing. We don't want to coerce to
        // a DST unless we have to. This currently comes out in the wash since
        // we can't unify [T] with U. But to properly support DST, we need to allow
        // that, at which point we will need extra checks on the target here.

        // Handle reborrows before selecting `Source: CoerceUnsized<Target>`.
        let reborrow = match (source.kind(), target.kind()) {
            (TyKind::Ref(_, ty_a, mutbl_a), TyKind::Ref(_, _, mutbl_b)) => {
                coerce_mutbls(mutbl_a, mutbl_b)?;

                let r_borrow = self.table.next_region_var();

                // We don't allow two-phase borrows here, at least for initial
                // implementation. If it happens that this coercion is a function argument,
                // the reborrow in coerce_borrowed_ptr will pick it up.
                // let mutbl = AutoBorrowMutability::new(mutbl_b, AllowTwoPhase::No);
                let mutbl = mutbl_b;

                Some((
                    Adjustment { kind: Adjust::Deref(None), target: ty_a },
                    Adjustment {
                        kind: Adjust::Borrow(AutoBorrow::Ref(r_borrow, mutbl)),
                        target: Ty::new_ref(self.interner(), r_borrow, ty_a, mutbl_b),
                    },
                ))
            }
            (TyKind::Ref(_, ty_a, mt_a), TyKind::RawPtr(_, mt_b)) => {
                coerce_mutbls(mt_a, mt_b)?;

                Some((
                    Adjustment { kind: Adjust::Deref(None), target: ty_a },
                    Adjustment {
                        kind: Adjust::Borrow(AutoBorrow::RawPtr(mt_b)),
                        target: Ty::new_ptr(self.interner(), ty_a, mt_b),
                    },
                ))
            }
            _ => None,
        };
        let coerce_source = reborrow.as_ref().map_or(source, |(_, r)| r.target);

        // Setup either a subtyping or a LUB relationship between
        // the `CoerceUnsized` target type and the expected type.
        // We only have the latter, so we use an inference variable
        // for the former and let type inference do the rest.
        let coerce_target = self.table.next_ty_var();

        let mut coercion = self.unify_and(
            coerce_target,
            target,
            reborrow.into_iter().flat_map(|(deref, autoref)| [deref, autoref]),
            Adjust::Pointer(PointerCast::Unsize),
        )?;

        // Create an obligation for `Source: CoerceUnsized<Target>`.
        let cause = self.cause.clone();

        // Use a FIFO queue for this custom fulfillment procedure.
        //
        // A Vec (or SmallVec) is not a natural choice for a queue. However,
        // this code path is hot, and this queue usually has a max length of 1
        // and almost never more than 3. By using a SmallVec we avoid an
        // allocation, at the (very small) cost of (occasionally) having to
        // shift subsequent elements down when removing the front element.
        let mut queue: SmallVec<[PredicateObligation<'db>; 4]> = smallvec![Obligation::new(
            self.interner(),
            cause,
            self.table.trait_env.env,
            TraitRef::new(
                self.interner(),
                coerce_unsized_did.into(),
                [coerce_source, coerce_target]
            )
        )];
        // Keep resolving `CoerceUnsized` and `Unsize` predicates to avoid
        // emitting a coercion in cases like `Foo<$1>` -> `Foo<$2>`, where
        // inference might unify those two inner type variables later.
        let traits = [coerce_unsized_did, unsize_did];
        while !queue.is_empty() {
            let obligation = queue.remove(0);
            let trait_pred = match obligation.predicate.kind().no_bound_vars() {
                Some(PredicateKind::Clause(ClauseKind::Trait(trait_pred)))
                    if traits.contains(&trait_pred.def_id().0) =>
                {
                    self.infer_ctxt().resolve_vars_if_possible(trait_pred)
                }
                // Eagerly process alias-relate obligations in new trait solver,
                // since these can be emitted in the process of solving trait goals,
                // but we need to constrain vars before processing goals mentioning
                // them.
                Some(PredicateKind::AliasRelate(..)) => {
                    let mut ocx = ObligationCtxt::new(self.infer_ctxt());
                    ocx.register_obligation(obligation);
                    if !ocx.try_evaluate_obligations().is_empty() {
                        return Err(TypeError::Mismatch);
                    }
                    coercion.obligations.extend(ocx.into_pending_obligations());
                    continue;
                }
                _ => {
                    coercion.obligations.push(obligation);
                    continue;
                }
            };
            debug!("coerce_unsized resolve step: {:?}", trait_pred);
            match self.infer_ctxt().select(&obligation.with(self.interner(), trait_pred)) {
                // Uncertain or unimplemented.
                Ok(None) => {
                    if trait_pred.def_id().0 == unsize_did {
                        let self_ty = trait_pred.self_ty();
                        let unsize_ty = trait_pred.trait_ref.args.inner()[1].expect_ty();
                        debug!("coerce_unsized: ambiguous unsize case for {:?}", trait_pred);
                        match (self_ty.kind(), unsize_ty.kind()) {
                            (TyKind::Infer(rustc_type_ir::TyVar(v)), TyKind::Dynamic(..))
                                if self.table.type_var_is_sized(v) =>
                            {
                                debug!("coerce_unsized: have sized infer {:?}", v);
                                coercion.obligations.push(obligation);
                                // `$0: Unsize<dyn Trait>` where we know that `$0: Sized`, try going
                                // for unsizing.
                            }
                            _ => {
                                // Some other case for `$0: Unsize<Something>`. Note that we
                                // hit this case even if `Something` is a sized type, so just
                                // don't do the coercion.
                                debug!("coerce_unsized: ambiguous unsize");
                                return Err(TypeError::Mismatch);
                            }
                        }
                    } else {
                        debug!("coerce_unsized: early return - ambiguous");
                        if !coerce_source.references_non_lt_error()
                            && !coerce_target.references_non_lt_error()
                        {
                            // rustc always early-returns here, even when the types contains errors. However not bailing
                            // improves error recovery, and while we don't implement generic consts properly, it also helps
                            // correct code.
                            return Err(TypeError::Mismatch);
                        }
                    }
                }
                Err(SelectionError::Unimplemented) => {
                    debug!("coerce_unsized: early return - can't prove obligation");
                    return Err(TypeError::Mismatch);
                }

                Err(SelectionError::TraitDynIncompatible(_)) => {
                    // Dyn compatibility errors in coercion will *always* be due to the
                    // fact that the RHS of the coercion is a non-dyn compatible `dyn Trait`
                    // written in source somewhere (otherwise we will never have lowered
                    // the dyn trait from HIR to middle).
                    //
                    // There's no reason to emit yet another dyn compatibility error,
                    // especially since the span will differ slightly and thus not be
                    // deduplicated at all!
                    self.set_tainted_by_errors();
                }
                Err(_err) => {
                    // FIXME: Report an error:
                    // let guar = self.err_ctxt().report_selection_error(
                    //     obligation.clone(),
                    //     &obligation,
                    //     &err,
                    // );
                    self.set_tainted_by_errors();
                    // Treat this like an obligation and follow through
                    // with the unsizing - the lack of a coercion should
                    // be silent, as it causes a type mismatch later.
                }

                Ok(Some(ImplSource::UserDefined(impl_source))) => {
                    queue.extend(impl_source.nested);
                }
                Ok(Some(impl_source)) => queue.extend(impl_source.nested_obligations()),
            }
        }

        Ok(coercion)
    }

    fn coerce_from_safe_fn(
        &mut self,
        fn_ty_a: PolyFnSig<'db>,
        b: Ty<'db>,
        adjustment: Option<Adjust<'db>>,
    ) -> CoerceResult<'db> {
        debug_assert!(self.table.shallow_resolve(b) == b);

        self.commit_if_ok(|this| {
            if let TyKind::FnPtr(_, hdr_b) = b.kind()
                && fn_ty_a.safety().is_safe()
                && !hdr_b.safety.is_safe()
            {
                let unsafe_a = Ty::safe_to_unsafe_fn_ty(this.interner(), fn_ty_a);
                this.unify_and(
                    unsafe_a,
                    b,
                    adjustment.map(|kind| Adjustment {
                        kind,
                        target: Ty::new_fn_ptr(this.interner(), fn_ty_a),
                    }),
                    Adjust::Pointer(PointerCast::UnsafeFnPointer),
                )
            } else {
                let a = Ty::new_fn_ptr(this.interner(), fn_ty_a);
                match adjustment {
                    Some(adjust) => this.unify_and(a, b, [], adjust),
                    None => this.unify(a, b),
                }
            }
        })
    }

    fn coerce_from_fn_pointer(&mut self, fn_ty_a: PolyFnSig<'db>, b: Ty<'db>) -> CoerceResult<'db> {
        debug!(?fn_ty_a, ?b, "coerce_from_fn_pointer");
        debug_assert!(self.table.shallow_resolve(b) == b);

        self.coerce_from_safe_fn(fn_ty_a, b, None)
    }

    fn coerce_from_fn_item(&mut self, a: Ty<'db>, b: Ty<'db>) -> CoerceResult<'db> {
        debug!("coerce_from_fn_item(a={:?}, b={:?})", a, b);
        debug_assert!(self.table.shallow_resolve(a) == a);
        debug_assert!(self.table.shallow_resolve(b) == b);

        match b.kind() {
            TyKind::FnPtr(_, b_hdr) => {
                let a_sig = a.fn_sig(self.interner());
                if let TyKind::FnDef(def_id, _) = a.kind() {
                    // Intrinsics are not coercible to function pointers
                    if let CallableDefId::FunctionId(def_id) = def_id.0 {
                        if FunctionSignature::is_intrinsic(self.table.db, def_id) {
                            return Err(TypeError::IntrinsicCast);
                        }

                        let attrs = self.table.db.attrs(def_id.into());
                        if attrs.by_key(sym::rustc_force_inline).exists() {
                            return Err(TypeError::ForceInlineCast);
                        }

                        if b_hdr.safety.is_safe() && attrs.by_key(sym::target_feature).exists() {
                            let fn_target_features =
                                TargetFeatures::from_attrs_no_implications(&attrs);
                            // Allow the coercion if the current function has all the features that would be
                            // needed to call the coercee safely.
                            let (target_features, target_feature_is_safe) =
                                (self.target_features)();
                            if target_feature_is_safe == TargetFeatureIsSafeInTarget::No
                                && !target_features.enabled.is_superset(&fn_target_features.enabled)
                            {
                                return Err(TypeError::TargetFeatureCast(
                                    CallableIdWrapper(def_id.into()).into(),
                                ));
                            }
                        }
                    }
                }

                self.coerce_from_safe_fn(
                    a_sig,
                    b,
                    Some(Adjust::Pointer(PointerCast::ReifyFnPointer)),
                )
            }
            _ => self.unify(a, b),
        }
    }

    /// Attempts to coerce from the type of a non-capturing closure
    /// into a function pointer.
    fn coerce_closure_to_fn(
        &mut self,
        a: Ty<'db>,
        _closure_def_id_a: InternedClosureId,
        args_a: GenericArgs<'db>,
        b: Ty<'db>,
    ) -> CoerceResult<'db> {
        debug_assert!(self.table.shallow_resolve(a) == a);
        debug_assert!(self.table.shallow_resolve(b) == b);

        match b.kind() {
            // FIXME: We need to have an `upvars_mentioned()` query:
            // At this point we haven't done capture analysis, which means
            // that the ClosureArgs just contains an inference variable instead
            // of tuple of captured types.
            //
            // All we care here is if any variable is being captured and not the exact paths,
            // so we check `upvars_mentioned` for root variables being captured.
            TyKind::FnPtr(_, hdr) =>
            // if self
            //     .db
            //     .upvars_mentioned(closure_def_id_a.expect_local())
            //     .is_none_or(|u| u.is_empty()) =>
            {
                // We coerce the closure, which has fn type
                //     `extern "rust-call" fn((arg0,arg1,...)) -> _`
                // to
                //     `fn(arg0,arg1,...) -> _`
                // or
                //     `unsafe fn(arg0,arg1,...) -> _`
                let safety = hdr.safety;
                let closure_sig = args_a.closure_sig_untupled().map_bound(|mut sig| {
                    sig.safety = hdr.safety;
                    sig
                });
                let pointer_ty = Ty::new_fn_ptr(self.interner(), closure_sig);
                debug!("coerce_closure_to_fn(a={:?}, b={:?}, pty={:?})", a, b, pointer_ty);
                self.unify_and(
                    pointer_ty,
                    b,
                    [],
                    Adjust::Pointer(PointerCast::ClosureFnPointer(safety)),
                )
            }
            _ => self.unify(a, b),
        }
    }

    fn coerce_raw_ptr(&mut self, a: Ty<'db>, b: Ty<'db>, mutbl_b: Mutability) -> CoerceResult<'db> {
        debug!("coerce_raw_ptr(a={:?}, b={:?})", a, b);
        debug_assert!(self.table.shallow_resolve(a) == a);
        debug_assert!(self.table.shallow_resolve(b) == b);

        let (is_ref, mt_a) = match a.kind() {
            TyKind::Ref(_, ty, mutbl) => (true, TypeAndMut::<DbInterner<'db>> { ty, mutbl }),
            TyKind::RawPtr(ty, mutbl) => (false, TypeAndMut { ty, mutbl }),
            _ => return self.unify(a, b),
        };
        coerce_mutbls(mt_a.mutbl, mutbl_b)?;

        // Check that the types which they point at are compatible.
        let a_raw = Ty::new_ptr(self.interner(), mt_a.ty, mutbl_b);
        // Although references and raw ptrs have the same
        // representation, we still register an Adjust::DerefRef so that
        // regionck knows that the region for `a` must be valid here.
        if is_ref {
            self.unify_and(
                a_raw,
                b,
                [Adjustment { kind: Adjust::Deref(None), target: mt_a.ty }],
                Adjust::Borrow(AutoBorrow::RawPtr(mutbl_b)),
            )
        } else if mt_a.mutbl != mutbl_b {
            self.unify_and(a_raw, b, [], Adjust::Pointer(PointerCast::MutToConstPointer))
        } else {
            self.unify(a_raw, b)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CoerceNever {
    No,
    Yes,
}

impl<'db> InferenceContext<'_, 'db> {
    /// Attempt to coerce an expression to a type, and return the
    /// adjusted type of the expression, if successful.
    /// Adjustments are only recorded if the coercion succeeded.
    /// The expressions *must not* have any preexisting adjustments.
    pub(crate) fn coerce(
        &mut self,
        expr: ExprOrPatId,
        expr_ty: Ty<'db>,
        mut target: Ty<'db>,
        allow_two_phase: AllowTwoPhase,
        coerce_never: CoerceNever,
    ) -> RelateResult<'db, Ty<'db>> {
        let source = self.table.try_structurally_resolve_type(expr_ty);
        target = self.table.try_structurally_resolve_type(target);
        debug!("coercion::try({:?}: {:?} -> {:?})", expr, source, target);

        let cause = ObligationCause::new();
        let krate = self.krate();
        let mut coerce = Coerce {
            table: &mut self.table,
            has_errors: &mut self.result.has_errors,
            cause,
            allow_two_phase,
            coerce_never: matches!(coerce_never, CoerceNever::Yes),
            use_lub: false,
            target_features: &mut || {
                Self::target_features(self.db, &self.target_features, self.owner, krate)
            },
        };
        let ok = coerce.commit_if_ok(|coerce| coerce.coerce(source, target))?;

        let (adjustments, _) = self.table.register_infer_ok(ok);
        match expr {
            ExprOrPatId::ExprId(expr) => self.write_expr_adj(expr, adjustments.into_boxed_slice()),
            ExprOrPatId::PatId(pat) => self
                .write_pat_adj(pat, adjustments.into_iter().map(|adjust| adjust.target).collect()),
        }
        Ok(target)
    }

    /// Given some expressions, their known unified type and another expression,
    /// tries to unify the types, potentially inserting coercions on any of the
    /// provided expressions and returns their LUB (aka "common supertype").
    ///
    /// This is really an internal helper. From outside the coercion
    /// module, you should instantiate a `CoerceMany` instance.
    fn try_find_coercion_lub(
        &mut self,
        exprs: &[ExprId],
        prev_ty: Ty<'db>,
        new: ExprId,
        new_ty: Ty<'db>,
    ) -> RelateResult<'db, Ty<'db>> {
        let prev_ty = self.table.try_structurally_resolve_type(prev_ty);
        let new_ty = self.table.try_structurally_resolve_type(new_ty);
        debug!(
            "coercion::try_find_coercion_lub({:?}, {:?}, exprs={:?} exprs)",
            prev_ty,
            new_ty,
            exprs.len()
        );

        // The following check fixes #88097, where the compiler erroneously
        // attempted to coerce a closure type to itself via a function pointer.
        if prev_ty == new_ty {
            return Ok(prev_ty);
        }

        let is_force_inline = |ty: Ty<'db>| {
            if let TyKind::FnDef(CallableIdWrapper(CallableDefId::FunctionId(did)), _) = ty.kind() {
                self.db.attrs(did.into()).by_key(sym::rustc_force_inline).exists()
            } else {
                false
            }
        };
        if is_force_inline(prev_ty) || is_force_inline(new_ty) {
            return Err(TypeError::ForceInlineCast);
        }

        // Special-case that coercion alone cannot handle:
        // Function items or non-capturing closures of differing IDs or GenericArgs.
        let (a_sig, b_sig) = {
            let is_capturing_closure = |_ty: Ty<'db>| {
                // FIXME:
                // if let TyKind::Closure(closure_def_id, _args) = ty.kind() {
                //     self.db.upvars_mentioned(closure_def_id.expect_local()).is_some()
                // } else {
                //     false
                // }
                false
            };
            if is_capturing_closure(prev_ty) || is_capturing_closure(new_ty) {
                (None, None)
            } else {
                match (prev_ty.kind(), new_ty.kind()) {
                    (TyKind::FnDef(..), TyKind::FnDef(..)) => {
                        // Don't reify if the function types have a LUB, i.e., they
                        // are the same function and their parameters have a LUB.
                        match self.table.commit_if_ok(|table| {
                            // We need to eagerly handle nested obligations due to lazy norm.
                            let mut ocx = ObligationCtxt::new(&table.infer_ctxt);
                            let value = ocx.lub(
                                &ObligationCause::new(),
                                table.trait_env.env,
                                prev_ty,
                                new_ty,
                            )?;
                            if ocx.try_evaluate_obligations().is_empty() {
                                Ok(InferOk { value, obligations: ocx.into_pending_obligations() })
                            } else {
                                Err(TypeError::Mismatch)
                            }
                        }) {
                            // We have a LUB of prev_ty and new_ty, just return it.
                            Ok(ok) => return Ok(self.table.register_infer_ok(ok)),
                            Err(_) => (
                                Some(prev_ty.fn_sig(self.table.interner())),
                                Some(new_ty.fn_sig(self.table.interner())),
                            ),
                        }
                    }
                    (TyKind::Closure(_, args), TyKind::FnDef(..)) => {
                        let b_sig = new_ty.fn_sig(self.table.interner());
                        let a_sig = args.closure_sig_untupled().map_bound(|mut sig| {
                            sig.safety = b_sig.safety();
                            sig
                        });
                        (Some(a_sig), Some(b_sig))
                    }
                    (TyKind::FnDef(..), TyKind::Closure(_, args)) => {
                        let a_sig = prev_ty.fn_sig(self.table.interner());
                        let b_sig = args.closure_sig_untupled().map_bound(|mut sig| {
                            sig.safety = a_sig.safety();
                            sig
                        });
                        (Some(a_sig), Some(b_sig))
                    }
                    (TyKind::Closure(_, args_a), TyKind::Closure(_, args_b)) => {
                        (Some(args_a.closure_sig_untupled()), Some(args_b.closure_sig_untupled()))
                    }
                    _ => (None, None),
                }
            }
        };
        if let (Some(a_sig), Some(b_sig)) = (a_sig, b_sig) {
            // The signature must match.
            let sig = self
                .table
                .infer_ctxt
                .at(&ObligationCause::new(), self.table.trait_env.env)
                .lub(a_sig, b_sig)
                .map(|ok| self.table.register_infer_ok(ok))?;

            // Reify both sides and return the reified fn pointer type.
            let fn_ptr = Ty::new_fn_ptr(self.table.interner(), sig);
            let prev_adjustment = match prev_ty.kind() {
                TyKind::Closure(..) => {
                    Adjust::Pointer(PointerCast::ClosureFnPointer(a_sig.safety()))
                }
                TyKind::FnDef(..) => Adjust::Pointer(PointerCast::ReifyFnPointer),
                _ => panic!("should not try to coerce a {prev_ty:?} to a fn pointer"),
            };
            let next_adjustment = match new_ty.kind() {
                TyKind::Closure(..) => {
                    Adjust::Pointer(PointerCast::ClosureFnPointer(b_sig.safety()))
                }
                TyKind::FnDef(..) => Adjust::Pointer(PointerCast::ReifyFnPointer),
                _ => panic!("should not try to coerce a {new_ty:?} to a fn pointer"),
            };
            for &expr in exprs {
                self.write_expr_adj(
                    expr,
                    Box::new([Adjustment { kind: prev_adjustment.clone(), target: fn_ptr }]),
                );
            }
            self.write_expr_adj(
                new,
                Box::new([Adjustment { kind: next_adjustment, target: fn_ptr }]),
            );
            return Ok(fn_ptr);
        }

        // Configure a Coerce instance to compute the LUB.
        // We don't allow two-phase borrows on any autorefs this creates since we
        // probably aren't processing function arguments here and even if we were,
        // they're going to get autorefed again anyway and we can apply 2-phase borrows
        // at that time.
        //
        // NOTE: we set `coerce_never` to `true` here because coercion LUBs only
        // operate on values and not places, so a never coercion is valid.
        let krate = self.krate();
        let mut coerce = Coerce {
            table: &mut self.table,
            has_errors: &mut self.result.has_errors,
            cause: ObligationCause::new(),
            allow_two_phase: AllowTwoPhase::No,
            coerce_never: true,
            use_lub: true,
            target_features: &mut || {
                Self::target_features(self.db, &self.target_features, self.owner, krate)
            },
        };

        // First try to coerce the new expression to the type of the previous ones,
        // but only if the new expression has no coercion already applied to it.
        let mut first_error = None;
        if !self.result.expr_adjustments.contains_key(&new) {
            let result = coerce.commit_if_ok(|coerce| coerce.coerce(new_ty, prev_ty));
            match result {
                Ok(ok) => {
                    let (adjustments, target) = self.table.register_infer_ok(ok);
                    self.write_expr_adj(new, adjustments.into_boxed_slice());
                    debug!(
                        "coercion::try_find_coercion_lub: was able to coerce from new type {:?} to previous type {:?} ({:?})",
                        new_ty, prev_ty, target
                    );
                    return Ok(target);
                }
                Err(e) => first_error = Some(e),
            }
        }

        match coerce.commit_if_ok(|coerce| coerce.coerce(prev_ty, new_ty)) {
            Err(_) => {
                // Avoid giving strange errors on failed attempts.
                if let Some(e) = first_error {
                    Err(e)
                } else {
                    Err(self
                        .table
                        .commit_if_ok(|table| {
                            table
                                .infer_ctxt
                                .at(&ObligationCause::new(), table.trait_env.env)
                                .lub(prev_ty, new_ty)
                        })
                        .unwrap_err())
                }
            }
            Ok(ok) => {
                let (adjustments, target) = self.table.register_infer_ok(ok);
                for &expr in exprs {
                    self.write_expr_adj(expr, adjustments.as_slice().into());
                }
                debug!(
                    "coercion::try_find_coercion_lub: was able to coerce previous type {:?} to new type {:?} ({:?})",
                    prev_ty, new_ty, target
                );
                Ok(target)
            }
        }
    }
}

/// CoerceMany encapsulates the pattern you should use when you have
/// many expressions that are all getting coerced to a common
/// type. This arises, for example, when you have a match (the result
/// of each arm is coerced to a common type). It also arises in less
/// obvious places, such as when you have many `break foo` expressions
/// that target the same loop, or the various `return` expressions in
/// a function.
///
/// The basic protocol is as follows:
///
/// - Instantiate the `CoerceMany` with an initial `expected_ty`.
///   This will also serve as the "starting LUB". The expectation is
///   that this type is something which all of the expressions *must*
///   be coercible to. Use a fresh type variable if needed.
/// - For each expression whose result is to be coerced, invoke `coerce()` with.
///   - In some cases we wish to coerce "non-expressions" whose types are implicitly
///     unit. This happens for example if you have a `break` with no expression,
///     or an `if` with no `else`. In that case, invoke `coerce_forced_unit()`.
///   - `coerce()` and `coerce_forced_unit()` may report errors. They hide this
///     from you so that you don't have to worry your pretty head about it.
///     But if an error is reported, the final type will be `err`.
///   - Invoking `coerce()` may cause us to go and adjust the "adjustments" on
///     previously coerced expressions.
/// - When all done, invoke `complete()`. This will return the LUB of
///   all your expressions.
///   - WARNING: I don't believe this final type is guaranteed to be
///     related to your initial `expected_ty` in any particular way,
///     although it will typically be a subtype, so you should check it.
///   - Invoking `complete()` may cause us to go and adjust the "adjustments" on
///     previously coerced expressions.
///
/// Example:
///
/// ```ignore (illustrative)
/// let mut coerce = CoerceMany::new(expected_ty);
/// for expr in exprs {
///     let expr_ty = fcx.check_expr_with_expectation(expr, expected);
///     coerce.coerce(fcx, &cause, expr, expr_ty);
/// }
/// let final_ty = coerce.complete(fcx);
/// ```
#[derive(Debug, Clone)]
pub(crate) struct CoerceMany<'db, 'exprs> {
    expected_ty: Ty<'db>,
    final_ty: Option<Ty<'db>>,
    expressions: Expressions<'exprs>,
    pushed: usize,
}

/// The type of a `CoerceMany` that is storing up the expressions into
/// a buffer. We use this for things like `break`.
pub(crate) type DynamicCoerceMany<'db> = CoerceMany<'db, 'db>;

#[derive(Debug, Clone)]
enum Expressions<'exprs> {
    Dynamic(SmallVec<[ExprId; 4]>),
    UpFront(&'exprs [ExprId]),
}

impl<'db, 'exprs> CoerceMany<'db, 'exprs> {
    /// The usual case; collect the set of expressions dynamically.
    /// If the full set of coercion sites is known before hand,
    /// consider `with_coercion_sites()` instead to avoid allocation.
    pub(crate) fn new(expected_ty: Ty<'db>) -> Self {
        Self::make(expected_ty, Expressions::Dynamic(SmallVec::new()))
    }

    /// As an optimization, you can create a `CoerceMany` with a
    /// preexisting slice of expressions. In this case, you are
    /// expected to pass each element in the slice to `coerce(...)` in
    /// order. This is used with arrays in particular to avoid
    /// needlessly cloning the slice.
    pub(crate) fn with_coercion_sites(
        expected_ty: Ty<'db>,
        coercion_sites: &'exprs [ExprId],
    ) -> Self {
        Self::make(expected_ty, Expressions::UpFront(coercion_sites))
    }

    fn make(expected_ty: Ty<'db>, expressions: Expressions<'exprs>) -> Self {
        CoerceMany { expected_ty, final_ty: None, expressions, pushed: 0 }
    }

    /// Returns the "expected type" with which this coercion was
    /// constructed. This represents the "downward propagated" type
    /// that was given to us at the start of typing whatever construct
    /// we are typing (e.g., the match expression).
    ///
    /// Typically, this is used as the expected type when
    /// type-checking each of the alternative expressions whose types
    /// we are trying to merge.
    pub(crate) fn expected_ty(&self) -> Ty<'db> {
        self.expected_ty
    }

    /// Returns the current "merged type", representing our best-guess
    /// at the LUB of the expressions we've seen so far (if any). This
    /// isn't *final* until you call `self.complete()`, which will return
    /// the merged type.
    pub(crate) fn merged_ty(&self) -> Ty<'db> {
        self.final_ty.unwrap_or(self.expected_ty)
    }

    /// Indicates that the value generated by `expression`, which is
    /// of type `expression_ty`, is one of the possibilities that we
    /// could coerce from. This will record `expression`, and later
    /// calls to `coerce` may come back and add adjustments and things
    /// if necessary.
    pub(crate) fn coerce(
        &mut self,
        icx: &mut InferenceContext<'_, 'db>,
        cause: &ObligationCause,
        expression: ExprId,
        expression_ty: Ty<'db>,
    ) {
        self.coerce_inner(icx, cause, expression, expression_ty, false, false)
    }

    /// Indicates that one of the inputs is a "forced unit". This
    /// occurs in a case like `if foo { ... };`, where the missing else
    /// generates a "forced unit". Another example is a `loop { break;
    /// }`, where the `break` has no argument expression. We treat
    /// these cases slightly differently for error-reporting
    /// purposes. Note that these tend to correspond to cases where
    /// the `()` expression is implicit in the source, and hence we do
    /// not take an expression argument.
    ///
    /// The `augment_error` gives you a chance to extend the error
    /// message, in case any results (e.g., we use this to suggest
    /// removing a `;`).
    pub(crate) fn coerce_forced_unit(
        &mut self,
        icx: &mut InferenceContext<'_, 'db>,
        expr: ExprId,
        cause: &ObligationCause,
        label_unit_as_expected: bool,
    ) {
        self.coerce_inner(icx, cause, expr, icx.types.unit, true, label_unit_as_expected)
    }

    /// The inner coercion "engine". If `expression` is `None`, this
    /// is a forced-unit case, and hence `expression_ty` must be
    /// `Nil`.
    pub(crate) fn coerce_inner(
        &mut self,
        icx: &mut InferenceContext<'_, 'db>,
        cause: &ObligationCause,
        expression: ExprId,
        mut expression_ty: Ty<'db>,
        force_unit: bool,
        label_expression_as_expected: bool,
    ) {
        // Incorporate whatever type inference information we have
        // until now; in principle we might also want to process
        // pending obligations, but doing so should only improve
        // compatibility (hopefully that is true) by helping us
        // uncover never types better.
        if expression_ty.is_ty_var() {
            expression_ty = icx.shallow_resolve(expression_ty);
        }

        let (expected, found) = if label_expression_as_expected {
            // In the case where this is a "forced unit", like
            // `break`, we want to call the `()` "expected"
            // since it is implied by the syntax.
            // (Note: not all force-units work this way.)"
            (expression_ty, self.merged_ty())
        } else {
            // Otherwise, the "expected" type for error
            // reporting is the current unification type,
            // which is basically the LUB of the expressions
            // we've seen so far (combined with the expected
            // type)
            (self.merged_ty(), expression_ty)
        };

        // Handle the actual type unification etc.
        let result = if !force_unit {
            if self.pushed == 0 {
                // Special-case the first expression we are coercing.
                // To be honest, I'm not entirely sure why we do this.
                // We don't allow two-phase borrows, see comment in try_find_coercion_lub for why
                icx.coerce(
                    expression.into(),
                    expression_ty,
                    self.expected_ty,
                    AllowTwoPhase::No,
                    CoerceNever::Yes,
                )
            } else {
                match self.expressions {
                    Expressions::Dynamic(ref exprs) => icx.try_find_coercion_lub(
                        exprs,
                        self.merged_ty(),
                        expression,
                        expression_ty,
                    ),
                    Expressions::UpFront(coercion_sites) => icx.try_find_coercion_lub(
                        &coercion_sites[0..self.pushed],
                        self.merged_ty(),
                        expression,
                        expression_ty,
                    ),
                }
            }
        } else {
            // this is a hack for cases where we default to `()` because
            // the expression etc has been omitted from the source. An
            // example is an `if let` without an else:
            //
            //     if let Some(x) = ... { }
            //
            // we wind up with a second match arm that is like `_ =>
            // ()`. That is the case we are considering here. We take
            // a different path to get the right "expected, found"
            // message and so forth (and because we know that
            // `expression_ty` will be unit).
            //
            // Another example is `break` with no argument expression.
            assert!(expression_ty.is_unit(), "if let hack without unit type");
            icx.table.infer_ctxt.at(cause, icx.table.trait_env.env).eq(expected, found).map(
                |infer_ok| {
                    icx.table.register_infer_ok(infer_ok);
                    expression_ty
                },
            )
        };

        debug!(?result);
        match result {
            Ok(v) => {
                self.final_ty = Some(v);
                match self.expressions {
                    Expressions::Dynamic(ref mut buffer) => buffer.push(expression),
                    Expressions::UpFront(coercion_sites) => {
                        // if the user gave us an array to validate, check that we got
                        // the next expression in the list, as expected
                        assert_eq!(coercion_sites[self.pushed], expression);
                    }
                }
            }
            Err(_coercion_error) => {
                // Mark that we've failed to coerce the types here to suppress
                // any superfluous errors we might encounter while trying to
                // emit or provide suggestions on how to fix the initial error.
                icx.set_tainted_by_errors();

                self.final_ty = Some(icx.types.error);

                icx.result.type_mismatches.insert(
                    expression.into(),
                    if label_expression_as_expected {
                        TypeMismatch { expected: found, actual: expected }
                    } else {
                        TypeMismatch { expected, actual: found }
                    },
                );
            }
        }

        self.pushed += 1;
    }

    pub(crate) fn complete(self, icx: &mut InferenceContext<'_, 'db>) -> Ty<'db> {
        if let Some(final_ty) = self.final_ty {
            final_ty
        } else {
            // If we only had inputs that were of type `!` (or no
            // inputs at all), then the final type is `!`.
            assert_eq!(self.pushed, 0);
            icx.types.never
        }
    }
}

pub fn could_coerce<'db>(
    db: &'db dyn HirDatabase,
    env: Arc<TraitEnvironment<'db>>,
    tys: &Canonical<'db, (Ty<'db>, Ty<'db>)>,
) -> bool {
    coerce(db, env, tys).is_ok()
}

fn coerce<'db>(
    db: &'db dyn HirDatabase,
    env: Arc<TraitEnvironment<'db>>,
    tys: &Canonical<'db, (Ty<'db>, Ty<'db>)>,
) -> Result<(Vec<Adjustment<'db>>, Ty<'db>), TypeError<DbInterner<'db>>> {
    let mut table = InferenceTable::new(db, env, None);
    let interner = table.interner();
    let ((ty1_with_vars, ty2_with_vars), vars) = table.infer_ctxt.instantiate_canonical(tys);

    let cause = ObligationCause::new();
    // FIXME: Target features.
    let target_features = TargetFeatures::default();
    let mut coerce = Coerce {
        table: &mut table,
        has_errors: &mut false,
        cause,
        allow_two_phase: AllowTwoPhase::No,
        coerce_never: true,
        use_lub: false,
        target_features: &mut || (&target_features, TargetFeatureIsSafeInTarget::No),
    };
    let InferOk { value: (adjustments, ty), obligations } =
        coerce.coerce(ty1_with_vars, ty2_with_vars)?;
    table.register_predicates(obligations);

    // default any type vars that weren't unified back to their original bound vars
    // (kind of hacky)
    let mut fallback_ty = |debruijn, infer| {
        let var = vars.var_values.iter().position(|arg| {
            arg.as_type().is_some_and(|ty| match ty.kind() {
                TyKind::Infer(it) => infer == it,
                _ => false,
            })
        });
        var.map_or_else(
            || Ty::new_error(interner, ErrorGuaranteed),
            |i| {
                Ty::new_bound(
                    interner,
                    debruijn,
                    BoundTy { kind: BoundTyKind::Anon, var: BoundVar::from_usize(i) },
                )
            },
        )
    };
    let mut fallback_const = |debruijn, infer| {
        let var = vars.var_values.iter().position(|arg| {
            arg.as_const().is_some_and(|ty| match ty.kind() {
                ConstKind::Infer(it) => infer == it,
                _ => false,
            })
        });
        var.map_or_else(
            || Const::new_error(interner, ErrorGuaranteed),
            |i| Const::new_bound(interner, debruijn, BoundConst { var: BoundVar::from_usize(i) }),
        )
    };
    let mut fallback_region = |debruijn, infer| {
        let var = vars.var_values.iter().position(|arg| {
            arg.as_region().is_some_and(|ty| match ty.kind() {
                RegionKind::ReVar(it) => infer == it,
                _ => false,
            })
        });
        var.map_or_else(
            || Region::error(interner),
            |i| {
                Region::new_bound(
                    interner,
                    debruijn,
                    BoundRegion { kind: BoundRegionKind::Anon, var: BoundVar::from_usize(i) },
                )
            },
        )
    };
    // FIXME also map the types in the adjustments
    // FIXME: We don't fallback correctly since this is done on `InferenceContext` and we only have `InferenceTable`.
    let ty = table.resolve_with_fallback(
        ty,
        &mut fallback_ty,
        &mut fallback_const,
        &mut fallback_region,
    );
    Ok((adjustments, ty))
}
