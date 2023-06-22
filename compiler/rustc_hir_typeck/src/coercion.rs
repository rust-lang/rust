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

use crate::FnCtxt;
use rustc_errors::{struct_span_err, Diagnostic, DiagnosticBuilder, ErrorGuaranteed, MultiSpan};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::Expr;
use rustc_hir_analysis::astconv::AstConv;
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_infer::infer::{Coercion, DefineOpaqueTypes, InferOk, InferResult};
use rustc_infer::traits::{Obligation, PredicateObligation};
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::adjustment::{
    Adjust, Adjustment, AllowTwoPhase, AutoBorrow, AutoBorrowMutability, PointerCast,
};
use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::relate::RelateResult;
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::visit::TypeVisitableExt;
use rustc_middle::ty::{self, Ty, TypeAndMut};
use rustc_session::parse::feature_err;
use rustc_span::symbol::sym;
use rustc_span::{self, DesugaringKind};
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::infer::InferCtxtExt as _;
use rustc_trait_selection::traits::error_reporting::TypeErrCtxtExt as _;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt;
use rustc_trait_selection::traits::{
    self, NormalizeExt, ObligationCause, ObligationCauseCode, ObligationCtxt,
};

use smallvec::{smallvec, SmallVec};
use std::ops::Deref;

struct Coerce<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    cause: ObligationCause<'tcx>,
    use_lub: bool,
    /// Determines whether or not allow_two_phase_borrow is set on any
    /// autoref adjustments we create while coercing. We don't want to
    /// allow deref coercions to create two-phase borrows, at least initially,
    /// but we do need two-phase borrows for function argument reborrows.
    /// See #47489 and #48598
    /// See docs on the "AllowTwoPhase" type for a more detailed discussion
    allow_two_phase: AllowTwoPhase,
}

impl<'a, 'tcx> Deref for Coerce<'a, 'tcx> {
    type Target = FnCtxt<'a, 'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.fcx
    }
}

type CoerceResult<'tcx> = InferResult<'tcx, (Vec<Adjustment<'tcx>>, Ty<'tcx>)>;

struct CollectRetsVisitor<'tcx> {
    ret_exprs: Vec<&'tcx hir::Expr<'tcx>>,
}

impl<'tcx> Visitor<'tcx> for CollectRetsVisitor<'tcx> {
    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        if let hir::ExprKind::Ret(_) = expr.kind {
            self.ret_exprs.push(expr);
        }
        intravisit::walk_expr(self, expr);
    }
}

/// Coercing a mutable reference to an immutable works, while
/// coercing `&T` to `&mut T` should be forbidden.
fn coerce_mutbls<'tcx>(
    from_mutbl: hir::Mutability,
    to_mutbl: hir::Mutability,
) -> RelateResult<'tcx, ()> {
    if from_mutbl >= to_mutbl { Ok(()) } else { Err(TypeError::Mutability) }
}

/// Do not require any adjustments, i.e. coerce `x -> x`.
fn identity(_: Ty<'_>) -> Vec<Adjustment<'_>> {
    vec![]
}

fn simple<'tcx>(kind: Adjust<'tcx>) -> impl FnOnce(Ty<'tcx>) -> Vec<Adjustment<'_>> {
    move |target| vec![Adjustment { kind, target }]
}

/// This always returns `Ok(...)`.
fn success<'tcx>(
    adj: Vec<Adjustment<'tcx>>,
    target: Ty<'tcx>,
    obligations: traits::PredicateObligations<'tcx>,
) -> CoerceResult<'tcx> {
    Ok(InferOk { value: (adj, target), obligations })
}

impl<'f, 'tcx> Coerce<'f, 'tcx> {
    fn new(
        fcx: &'f FnCtxt<'f, 'tcx>,
        cause: ObligationCause<'tcx>,
        allow_two_phase: AllowTwoPhase,
    ) -> Self {
        Coerce { fcx, cause, allow_two_phase, use_lub: false }
    }

    fn unify(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> InferResult<'tcx, Ty<'tcx>> {
        debug!("unify(a: {:?}, b: {:?}, use_lub: {})", a, b, self.use_lub);
        self.commit_if_ok(|_| {
            let at = self.at(&self.cause, self.fcx.param_env);

            let res = if self.use_lub {
                at.lub(DefineOpaqueTypes::Yes, b, a)
            } else {
                at.sup(DefineOpaqueTypes::Yes, b, a)
                    .map(|InferOk { value: (), obligations }| InferOk { value: a, obligations })
            };

            // In the new solver, lazy norm may allow us to shallowly equate
            // more types, but we emit possibly impossible-to-satisfy obligations.
            // Filter these cases out to make sure our coercion is more accurate.
            if self.next_trait_solver() {
                if let Ok(res) = &res {
                    for obligation in &res.obligations {
                        if !self.predicate_may_hold(&obligation) {
                            return Err(TypeError::Mismatch);
                        }
                    }
                }
            }

            res
        })
    }

    /// Unify two types (using sub or lub) and produce a specific coercion.
    fn unify_and<F>(&self, a: Ty<'tcx>, b: Ty<'tcx>, f: F) -> CoerceResult<'tcx>
    where
        F: FnOnce(Ty<'tcx>) -> Vec<Adjustment<'tcx>>,
    {
        self.unify(a, b)
            .and_then(|InferOk { value: ty, obligations }| success(f(ty), ty, obligations))
    }

    #[instrument(skip(self))]
    fn coerce(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> CoerceResult<'tcx> {
        // First, remove any resolved type variables (at the top level, at least):
        let a = self.shallow_resolve(a);
        let b = self.shallow_resolve(b);
        debug!("Coerce.tys({:?} => {:?})", a, b);

        // Just ignore error types.
        if let Err(guar) = (a, b).error_reported() {
            // Best-effort try to unify these types -- we're already on the error path,
            // so this will have the side-effect of making sure we have no ambiguities
            // due to `[type error]` and `_` not coercing together.
            let _ = self.commit_if_ok(|_| {
                self.at(&self.cause, self.param_env).eq(DefineOpaqueTypes::Yes, a, b)
            });
            return success(vec![], self.fcx.tcx.ty_error(guar), vec![]);
        }

        // Coercing from `!` to any type is allowed:
        if a.is_never() {
            return success(simple(Adjust::NeverToAny)(b), b, vec![]);
        }

        // Coercing *from* an unresolved inference variable means that
        // we have no information about the source type. This will always
        // ultimately fall back to some form of subtyping.
        if a.is_ty_var() {
            return self.coerce_from_inference_variable(a, b, identity);
        }

        // Consider coercing the subtype to a DST
        //
        // NOTE: this is wrapped in a `commit_if_ok` because it creates
        // a "spurious" type variable, and we don't want to have that
        // type variable in memory if the coercion fails.
        let unsize = self.commit_if_ok(|_| self.coerce_unsized(a, b));
        match unsize {
            Ok(_) => {
                debug!("coerce: unsize successful");
                return unsize;
            }
            Err(error) => {
                debug!(?error, "coerce: unsize failed");
            }
        }

        // Examine the supertype and consider auto-borrowing.
        match *b.kind() {
            ty::RawPtr(mt_b) => {
                return self.coerce_unsafe_ptr(a, b, mt_b.mutbl);
            }
            ty::Ref(r_b, _, mutbl_b) => {
                return self.coerce_borrowed_pointer(a, b, r_b, mutbl_b);
            }
            ty::Dynamic(predicates, region, ty::DynStar) if self.tcx.features().dyn_star => {
                return self.coerce_dyn_star(a, b, predicates, region);
            }
            _ => {}
        }

        match *a.kind() {
            ty::FnDef(..) => {
                // Function items are coercible to any closure
                // type; function pointers are not (that would
                // require double indirection).
                // Additionally, we permit coercion of function
                // items to drop the unsafe qualifier.
                self.coerce_from_fn_item(a, b)
            }
            ty::FnPtr(a_f) => {
                // We permit coercion of fn pointers to drop the
                // unsafe qualifier.
                self.coerce_from_fn_pointer(a, a_f, b)
            }
            ty::Closure(closure_def_id_a, substs_a) => {
                // Non-capturing closures are coercible to
                // function pointers or unsafe function pointers.
                // It cannot convert closures that require unsafe.
                self.coerce_closure_to_fn(a, closure_def_id_a, substs_a, b)
            }
            _ => {
                // Otherwise, just use unification rules.
                self.unify_and(a, b, identity)
            }
        }
    }

    /// Coercing *from* an inference variable. In this case, we have no information
    /// about the source type, so we can't really do a true coercion and we always
    /// fall back to subtyping (`unify_and`).
    fn coerce_from_inference_variable(
        &self,
        a: Ty<'tcx>,
        b: Ty<'tcx>,
        make_adjustments: impl FnOnce(Ty<'tcx>) -> Vec<Adjustment<'tcx>>,
    ) -> CoerceResult<'tcx> {
        debug!("coerce_from_inference_variable(a={:?}, b={:?})", a, b);
        assert!(a.is_ty_var() && self.shallow_resolve(a) == a);
        assert!(self.shallow_resolve(b) == b);

        if b.is_ty_var() {
            // Two unresolved type variables: create a `Coerce` predicate.
            let target_ty = if self.use_lub {
                self.next_ty_var(TypeVariableOrigin {
                    kind: TypeVariableOriginKind::LatticeVariable,
                    span: self.cause.span,
                })
            } else {
                b
            };

            let mut obligations = Vec::with_capacity(2);
            for &source_ty in &[a, b] {
                if source_ty != target_ty {
                    obligations.push(Obligation::new(
                        self.tcx(),
                        self.cause.clone(),
                        self.param_env,
                        ty::Binder::dummy(ty::PredicateKind::Coerce(ty::CoercePredicate {
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
            let adjustments = make_adjustments(target_ty);
            InferResult::Ok(InferOk { value: (adjustments, target_ty), obligations })
        } else {
            // One unresolved type variable: just apply subtyping, we may be able
            // to do something useful.
            self.unify_and(a, b, make_adjustments)
        }
    }

    /// Reborrows `&mut A` to `&mut B` and `&(mut) A` to `&B`.
    /// To match `A` with `B`, autoderef will be performed,
    /// calling `deref`/`deref_mut` where necessary.
    fn coerce_borrowed_pointer(
        &self,
        a: Ty<'tcx>,
        b: Ty<'tcx>,
        r_b: ty::Region<'tcx>,
        mutbl_b: hir::Mutability,
    ) -> CoerceResult<'tcx> {
        debug!("coerce_borrowed_pointer(a={:?}, b={:?})", a, b);

        // If we have a parameter of type `&M T_a` and the value
        // provided is `expr`, we will be adding an implicit borrow,
        // meaning that we convert `f(expr)` to `f(&M *expr)`. Therefore,
        // to type check, we will construct the type that `&M*expr` would
        // yield.

        let (r_a, mt_a) = match *a.kind() {
            ty::Ref(r_a, ty, mutbl) => {
                let mt_a = ty::TypeAndMut { ty, mutbl };
                coerce_mutbls(mt_a.mutbl, mutbl_b)?;
                (r_a, mt_a)
            }
            _ => return self.unify_and(a, b, identity),
        };

        let span = self.cause.span;

        let mut first_error = None;
        let mut r_borrow_var = None;
        let mut autoderef = self.autoderef(span, a);
        let mut found = None;

        for (referent_ty, autoderefs) in autoderef.by_ref() {
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
                    let coercion = Coercion(span);
                    let r = self.next_region_var(coercion);
                    r_borrow_var = Some(r); // [4] above
                }
                r_borrow_var.unwrap()
            };
            let derefd_ty_a = self.tcx.mk_ref(
                r,
                TypeAndMut {
                    ty: referent_ty,
                    mutbl: mutbl_b, // [1] above
                },
            );
            match self.unify(derefd_ty_a, b) {
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
            let err = first_error.expect("coerce_borrowed_pointer had no error");
            debug!("coerce_borrowed_pointer: failed with err = {:?}", err);
            return Err(err);
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
            self.adjust_steps_as_infer_ok(&autoderef);
        obligations.extend(o);
        obligations.extend(autoderef.into_obligations());

        // Now apply the autoref. We have to extract the region out of
        // the final ref type we got.
        let ty::Ref(r_borrow, _, _) = ty.kind() else {
            span_bug!(span, "expected a ref type, got {:?}", ty);
        };
        let mutbl = AutoBorrowMutability::new(mutbl_b, self.allow_two_phase);
        adjustments.push(Adjustment {
            kind: Adjust::Borrow(AutoBorrow::Ref(*r_borrow, mutbl)),
            target: ty,
        });

        debug!("coerce_borrowed_pointer: succeeded ty={:?} adjustments={:?}", ty, adjustments);

        success(adjustments, ty, obligations)
    }

    // &[T; n] or &mut [T; n] -> &[T]
    // or &mut [T; n] -> &mut [T]
    // or &Concrete -> &Trait, etc.
    #[instrument(skip(self), level = "debug")]
    fn coerce_unsized(&self, mut source: Ty<'tcx>, mut target: Ty<'tcx>) -> CoerceResult<'tcx> {
        source = self.shallow_resolve(source);
        target = self.shallow_resolve(target);
        debug!(?source, ?target);

        // We don't apply any coercions incase either the source or target
        // aren't sufficiently well known but tend to instead just equate
        // them both.
        if source.is_ty_var() {
            debug!("coerce_unsized: source is a TyVar, bailing out");
            return Err(TypeError::Mismatch);
        }
        if target.is_ty_var() {
            debug!("coerce_unsized: target is a TyVar, bailing out");
            return Err(TypeError::Mismatch);
        }

        let traits =
            (self.tcx.lang_items().unsize_trait(), self.tcx.lang_items().coerce_unsized_trait());
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
            (&ty::Ref(_, ty_a, mutbl_a), &ty::Ref(_, _, mutbl_b)) => {
                coerce_mutbls(mutbl_a, mutbl_b)?;

                let coercion = Coercion(self.cause.span);
                let r_borrow = self.next_region_var(coercion);

                // We don't allow two-phase borrows here, at least for initial
                // implementation. If it happens that this coercion is a function argument,
                // the reborrow in coerce_borrowed_ptr will pick it up.
                let mutbl = AutoBorrowMutability::new(mutbl_b, AllowTwoPhase::No);

                Some((
                    Adjustment { kind: Adjust::Deref(None), target: ty_a },
                    Adjustment {
                        kind: Adjust::Borrow(AutoBorrow::Ref(r_borrow, mutbl)),
                        target: self
                            .tcx
                            .mk_ref(r_borrow, ty::TypeAndMut { mutbl: mutbl_b, ty: ty_a }),
                    },
                ))
            }
            (&ty::Ref(_, ty_a, mt_a), &ty::RawPtr(ty::TypeAndMut { mutbl: mt_b, .. })) => {
                coerce_mutbls(mt_a, mt_b)?;

                Some((
                    Adjustment { kind: Adjust::Deref(None), target: ty_a },
                    Adjustment {
                        kind: Adjust::Borrow(AutoBorrow::RawPtr(mt_b)),
                        target: self.tcx.mk_ptr(ty::TypeAndMut { mutbl: mt_b, ty: ty_a }),
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
        let origin = TypeVariableOrigin {
            kind: TypeVariableOriginKind::MiscVariable,
            span: self.cause.span,
        };
        let coerce_target = self.next_ty_var(origin);
        let mut coercion = self.unify_and(coerce_target, target, |target| {
            let unsize = Adjustment { kind: Adjust::Pointer(PointerCast::Unsize), target };
            match reborrow {
                None => vec![unsize],
                Some((ref deref, ref autoref)) => vec![deref.clone(), autoref.clone(), unsize],
            }
        })?;

        let mut selcx = traits::SelectionContext::new(self);

        // Create an obligation for `Source: CoerceUnsized<Target>`.
        let cause = ObligationCause::new(
            self.cause.span,
            self.body_id,
            ObligationCauseCode::Coercion { source, target },
        );

        // Use a FIFO queue for this custom fulfillment procedure.
        //
        // A Vec (or SmallVec) is not a natural choice for a queue. However,
        // this code path is hot, and this queue usually has a max length of 1
        // and almost never more than 3. By using a SmallVec we avoid an
        // allocation, at the (very small) cost of (occasionally) having to
        // shift subsequent elements down when removing the front element.
        let mut queue: SmallVec<[PredicateObligation<'tcx>; 4]> = smallvec![Obligation::new(
            self.tcx,
            cause,
            self.fcx.param_env,
            ty::TraitRef::new(self.tcx, coerce_unsized_did, [coerce_source, coerce_target])
        )];

        let mut has_unsized_tuple_coercion = false;
        let mut has_trait_upcasting_coercion = None;

        // Keep resolving `CoerceUnsized` and `Unsize` predicates to avoid
        // emitting a coercion in cases like `Foo<$1>` -> `Foo<$2>`, where
        // inference might unify those two inner type variables later.
        let traits = [coerce_unsized_did, unsize_did];
        while !queue.is_empty() {
            let obligation = queue.remove(0);
            debug!("coerce_unsized resolve step: {:?}", obligation);
            let bound_predicate = obligation.predicate.kind();
            let trait_pred = match bound_predicate.skip_binder() {
                ty::PredicateKind::Clause(ty::ClauseKind::Trait(trait_pred))
                    if traits.contains(&trait_pred.def_id()) =>
                {
                    if unsize_did == trait_pred.def_id() {
                        let self_ty = trait_pred.self_ty();
                        let unsize_ty = trait_pred.trait_ref.substs[1].expect_ty();
                        if let (ty::Dynamic(ref data_a, ..), ty::Dynamic(ref data_b, ..)) =
                            (self_ty.kind(), unsize_ty.kind())
                            && data_a.principal_def_id() != data_b.principal_def_id()
                        {
                            debug!("coerce_unsized: found trait upcasting coercion");
                            has_trait_upcasting_coercion = Some((self_ty, unsize_ty));
                        }
                        if let ty::Tuple(..) = unsize_ty.kind() {
                            debug!("coerce_unsized: found unsized tuple coercion");
                            has_unsized_tuple_coercion = true;
                        }
                    }
                    bound_predicate.rebind(trait_pred)
                }
                _ => {
                    coercion.obligations.push(obligation);
                    continue;
                }
            };
            match selcx.select(&obligation.with(selcx.tcx(), trait_pred)) {
                // Uncertain or unimplemented.
                Ok(None) => {
                    if trait_pred.def_id() == unsize_did {
                        let trait_pred = self.resolve_vars_if_possible(trait_pred);
                        let self_ty = trait_pred.skip_binder().self_ty();
                        let unsize_ty = trait_pred.skip_binder().trait_ref.substs[1].expect_ty();
                        debug!("coerce_unsized: ambiguous unsize case for {:?}", trait_pred);
                        match (self_ty.kind(), unsize_ty.kind()) {
                            (&ty::Infer(ty::TyVar(v)), ty::Dynamic(..))
                                if self.type_var_is_sized(v) =>
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
                        return Err(TypeError::Mismatch);
                    }
                }
                Err(traits::Unimplemented) => {
                    debug!("coerce_unsized: early return - can't prove obligation");
                    return Err(TypeError::Mismatch);
                }

                // Object safety violations or miscellaneous.
                Err(err) => {
                    self.err_ctxt().report_selection_error(obligation.clone(), &obligation, &err);
                    // Treat this like an obligation and follow through
                    // with the unsizing - the lack of a coercion should
                    // be silent, as it causes a type mismatch later.
                }

                Ok(Some(impl_source)) => queue.extend(impl_source.nested_obligations()),
            }
        }

        if has_unsized_tuple_coercion && !self.tcx.features().unsized_tuple_coercion {
            feature_err(
                &self.tcx.sess.parse_sess,
                sym::unsized_tuple_coercion,
                self.cause.span,
                "unsized tuple coercion is not stable enough for use and is subject to change",
            )
            .emit();
        }

        if let Some((sub, sup)) = has_trait_upcasting_coercion
            && !self.tcx().features().trait_upcasting
        {
            // Renders better when we erase regions, since they're not really the point here.
            let (sub, sup) = self.tcx.erase_regions((sub, sup));
            let mut err = feature_err(
                &self.tcx.sess.parse_sess,
                sym::trait_upcasting,
                self.cause.span,
                format!("cannot cast `{sub}` to `{sup}`, trait upcasting coercion is experimental"),
            );
            err.note(format!("required when coercing `{source}` into `{target}`"));
            err.emit();
        }

        Ok(coercion)
    }

    fn coerce_dyn_star(
        &self,
        a: Ty<'tcx>,
        b: Ty<'tcx>,
        predicates: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
        b_region: ty::Region<'tcx>,
    ) -> CoerceResult<'tcx> {
        if !self.tcx.features().dyn_star {
            return Err(TypeError::Mismatch);
        }

        if let ty::Dynamic(a_data, _, _) = a.kind()
            && let ty::Dynamic(b_data, _, _) = b.kind()
            && a_data.principal_def_id() == b_data.principal_def_id()
        {
            return self.unify_and(a, b, |_| vec![]);
        }

        // Check the obligations of the cast -- for example, when casting
        // `usize` to `dyn* Clone + 'static`:
        let mut obligations: Vec<_> = predicates
            .iter()
            .map(|predicate| {
                // For each existential predicate (e.g., `?Self: Clone`) substitute
                // the type of the expression (e.g., `usize` in our example above)
                // and then require that the resulting predicate (e.g., `usize: Clone`)
                // holds (it does).
                let predicate = predicate.with_self_ty(self.tcx, a);
                Obligation::new(self.tcx, self.cause.clone(), self.param_env, predicate)
            })
            .chain([
                // Enforce the region bound (e.g., `usize: 'static`, in our example).
                Obligation::new(
                    self.tcx,
                    self.cause.clone(),
                    self.param_env,
                    ty::Binder::dummy(ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(
                        ty::OutlivesPredicate(a, b_region),
                    ))),
                ),
            ])
            .collect();

        // Enforce that the type is `usize`/pointer-sized.
        obligations.push(Obligation::new(
            self.tcx,
            self.cause.clone(),
            self.param_env,
            ty::TraitRef::from_lang_item(
                self.tcx,
                hir::LangItem::PointerLike,
                self.cause.span,
                [a],
            ),
        ));

        Ok(InferOk {
            value: (vec![Adjustment { kind: Adjust::DynStar, target: b }], b),
            obligations,
        })
    }

    fn coerce_from_safe_fn<F, G>(
        &self,
        a: Ty<'tcx>,
        fn_ty_a: ty::PolyFnSig<'tcx>,
        b: Ty<'tcx>,
        to_unsafe: F,
        normal: G,
    ) -> CoerceResult<'tcx>
    where
        F: FnOnce(Ty<'tcx>) -> Vec<Adjustment<'tcx>>,
        G: FnOnce(Ty<'tcx>) -> Vec<Adjustment<'tcx>>,
    {
        self.commit_if_ok(|snapshot| {
            let outer_universe = self.infcx.universe();

            let result = if let ty::FnPtr(fn_ty_b) = b.kind()
                && let (hir::Unsafety::Normal, hir::Unsafety::Unsafe) =
                    (fn_ty_a.unsafety(), fn_ty_b.unsafety())
            {
                let unsafe_a = self.tcx.safe_to_unsafe_fn_ty(fn_ty_a);
                self.unify_and(unsafe_a, b, to_unsafe)
            } else {
                self.unify_and(a, b, normal)
            };

            // FIXME(#73154): This is a hack. Currently LUB can generate
            // unsolvable constraints. Additionally, it returns `a`
            // unconditionally, even when the "LUB" is `b`. In the future, we
            // want the coerced type to be the actual supertype of these two,
            // but for now, we want to just error to ensure we don't lock
            // ourselves into a specific behavior with NLL.
            self.leak_check(outer_universe, Some(snapshot))?;

            result
        })
    }

    fn coerce_from_fn_pointer(
        &self,
        a: Ty<'tcx>,
        fn_ty_a: ty::PolyFnSig<'tcx>,
        b: Ty<'tcx>,
    ) -> CoerceResult<'tcx> {
        //! Attempts to coerce from the type of a Rust function item
        //! into a closure or a `proc`.
        //!

        let b = self.shallow_resolve(b);
        debug!("coerce_from_fn_pointer(a={:?}, b={:?})", a, b);

        self.coerce_from_safe_fn(
            a,
            fn_ty_a,
            b,
            simple(Adjust::Pointer(PointerCast::UnsafeFnPointer)),
            identity,
        )
    }

    fn coerce_from_fn_item(&self, a: Ty<'tcx>, b: Ty<'tcx>) -> CoerceResult<'tcx> {
        //! Attempts to coerce from the type of a Rust function item
        //! into a closure or a `proc`.

        let b = self.shallow_resolve(b);
        let InferOk { value: b, mut obligations } =
            self.at(&self.cause, self.param_env).normalize(b);
        debug!("coerce_from_fn_item(a={:?}, b={:?})", a, b);

        match b.kind() {
            ty::FnPtr(b_sig) => {
                let a_sig = a.fn_sig(self.tcx);
                if let ty::FnDef(def_id, _) = *a.kind() {
                    // Intrinsics are not coercible to function pointers
                    if self.tcx.is_intrinsic(def_id) {
                        return Err(TypeError::IntrinsicCast);
                    }

                    // Safe `#[target_feature]` functions are not assignable to safe fn pointers (RFC 2396).

                    if b_sig.unsafety() == hir::Unsafety::Normal
                        && !self.tcx.codegen_fn_attrs(def_id).target_features.is_empty()
                    {
                        return Err(TypeError::TargetFeatureCast(def_id));
                    }
                }

                let InferOk { value: a_sig, obligations: o1 } =
                    self.at(&self.cause, self.param_env).normalize(a_sig);
                obligations.extend(o1);

                let a_fn_pointer = self.tcx.mk_fn_ptr(a_sig);
                let InferOk { value, obligations: o2 } = self.coerce_from_safe_fn(
                    a_fn_pointer,
                    a_sig,
                    b,
                    |unsafe_ty| {
                        vec![
                            Adjustment {
                                kind: Adjust::Pointer(PointerCast::ReifyFnPointer),
                                target: a_fn_pointer,
                            },
                            Adjustment {
                                kind: Adjust::Pointer(PointerCast::UnsafeFnPointer),
                                target: unsafe_ty,
                            },
                        ]
                    },
                    simple(Adjust::Pointer(PointerCast::ReifyFnPointer)),
                )?;

                obligations.extend(o2);
                Ok(InferOk { value, obligations })
            }
            _ => self.unify_and(a, b, identity),
        }
    }

    fn coerce_closure_to_fn(
        &self,
        a: Ty<'tcx>,
        closure_def_id_a: DefId,
        substs_a: SubstsRef<'tcx>,
        b: Ty<'tcx>,
    ) -> CoerceResult<'tcx> {
        //! Attempts to coerce from the type of a non-capturing closure
        //! into a function pointer.
        //!

        let b = self.shallow_resolve(b);

        match b.kind() {
            // At this point we haven't done capture analysis, which means
            // that the ClosureSubsts just contains an inference variable instead
            // of tuple of captured types.
            //
            // All we care here is if any variable is being captured and not the exact paths,
            // so we check `upvars_mentioned` for root variables being captured.
            ty::FnPtr(fn_ty)
                if self
                    .tcx
                    .upvars_mentioned(closure_def_id_a.expect_local())
                    .map_or(true, |u| u.is_empty()) =>
            {
                // We coerce the closure, which has fn type
                //     `extern "rust-call" fn((arg0,arg1,...)) -> _`
                // to
                //     `fn(arg0,arg1,...) -> _`
                // or
                //     `unsafe fn(arg0,arg1,...) -> _`
                let closure_sig = substs_a.as_closure().sig();
                let unsafety = fn_ty.unsafety();
                let pointer_ty =
                    self.tcx.mk_fn_ptr(self.tcx.signature_unclosure(closure_sig, unsafety));
                debug!("coerce_closure_to_fn(a={:?}, b={:?}, pty={:?})", a, b, pointer_ty);
                self.unify_and(
                    pointer_ty,
                    b,
                    simple(Adjust::Pointer(PointerCast::ClosureFnPointer(unsafety))),
                )
            }
            _ => self.unify_and(a, b, identity),
        }
    }

    fn coerce_unsafe_ptr(
        &self,
        a: Ty<'tcx>,
        b: Ty<'tcx>,
        mutbl_b: hir::Mutability,
    ) -> CoerceResult<'tcx> {
        debug!("coerce_unsafe_ptr(a={:?}, b={:?})", a, b);

        let (is_ref, mt_a) = match *a.kind() {
            ty::Ref(_, ty, mutbl) => (true, ty::TypeAndMut { ty, mutbl }),
            ty::RawPtr(mt) => (false, mt),
            _ => return self.unify_and(a, b, identity),
        };
        coerce_mutbls(mt_a.mutbl, mutbl_b)?;

        // Check that the types which they point at are compatible.
        let a_unsafe = self.tcx.mk_ptr(ty::TypeAndMut { mutbl: mutbl_b, ty: mt_a.ty });
        // Although references and unsafe ptrs have the same
        // representation, we still register an Adjust::DerefRef so that
        // regionck knows that the region for `a` must be valid here.
        if is_ref {
            self.unify_and(a_unsafe, b, |target| {
                vec![
                    Adjustment { kind: Adjust::Deref(None), target: mt_a.ty },
                    Adjustment { kind: Adjust::Borrow(AutoBorrow::RawPtr(mutbl_b)), target },
                ]
            })
        } else if mt_a.mutbl != mutbl_b {
            self.unify_and(a_unsafe, b, simple(Adjust::Pointer(PointerCast::MutToConstPointer)))
        } else {
            self.unify_and(a_unsafe, b, identity)
        }
    }
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    /// Attempt to coerce an expression to a type, and return the
    /// adjusted type of the expression, if successful.
    /// Adjustments are only recorded if the coercion succeeded.
    /// The expressions *must not* have any preexisting adjustments.
    pub fn try_coerce(
        &self,
        expr: &hir::Expr<'_>,
        expr_ty: Ty<'tcx>,
        target: Ty<'tcx>,
        allow_two_phase: AllowTwoPhase,
        cause: Option<ObligationCause<'tcx>>,
    ) -> RelateResult<'tcx, Ty<'tcx>> {
        let source = self.resolve_vars_with_obligations(expr_ty);
        debug!("coercion::try({:?}: {:?} -> {:?})", expr, source, target);

        let cause =
            cause.unwrap_or_else(|| self.cause(expr.span, ObligationCauseCode::ExprAssignable));
        let coerce = Coerce::new(self, cause, allow_two_phase);
        let ok = self.commit_if_ok(|_| coerce.coerce(source, target))?;

        let (adjustments, _) = self.register_infer_ok_obligations(ok);
        self.apply_adjustments(expr, adjustments);
        Ok(if let Err(guar) = expr_ty.error_reported() { self.tcx.ty_error(guar) } else { target })
    }

    /// Same as `try_coerce()`, but without side-effects.
    ///
    /// Returns false if the coercion creates any obligations that result in
    /// errors.
    pub fn can_coerce(&self, expr_ty: Ty<'tcx>, target: Ty<'tcx>) -> bool {
        let source = self.resolve_vars_with_obligations(expr_ty);
        debug!("coercion::can_with_predicates({:?} -> {:?})", source, target);

        let cause = self.cause(rustc_span::DUMMY_SP, ObligationCauseCode::ExprAssignable);
        // We don't ever need two-phase here since we throw out the result of the coercion
        let coerce = Coerce::new(self, cause, AllowTwoPhase::No);
        self.probe(|_| {
            let Ok(ok) = coerce.coerce(source, target) else {
                return false;
            };
            let ocx = ObligationCtxt::new_in_snapshot(self);
            ocx.register_obligations(ok.obligations);
            ocx.select_where_possible().is_empty()
        })
    }

    /// Given a type and a target type, this function will calculate and return
    /// how many dereference steps needed to achieve `expr_ty <: target`. If
    /// it's not possible, return `None`.
    pub fn deref_steps(&self, expr_ty: Ty<'tcx>, target: Ty<'tcx>) -> Option<usize> {
        let cause = self.cause(rustc_span::DUMMY_SP, ObligationCauseCode::ExprAssignable);
        // We don't ever need two-phase here since we throw out the result of the coercion
        let coerce = Coerce::new(self, cause, AllowTwoPhase::No);
        coerce
            .autoderef(rustc_span::DUMMY_SP, expr_ty)
            .find_map(|(ty, steps)| self.probe(|_| coerce.unify(ty, target)).ok().map(|_| steps))
    }

    /// Given a type, this function will calculate and return the type given
    /// for `<Ty as Deref>::Target` only if `Ty` also implements `DerefMut`.
    ///
    /// This function is for diagnostics only, since it does not register
    /// trait or region sub-obligations. (presumably we could, but it's not
    /// particularly important for diagnostics...)
    pub fn deref_once_mutably_for_diagnostic(&self, expr_ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
        self.autoderef(rustc_span::DUMMY_SP, expr_ty).nth(1).and_then(|(deref_ty, _)| {
            self.infcx
                .type_implements_trait(
                    self.tcx.lang_items().deref_mut_trait()?,
                    [expr_ty],
                    self.param_env,
                )
                .may_apply()
                .then_some(deref_ty)
        })
    }

    /// Given some expressions, their known unified type and another expression,
    /// tries to unify the types, potentially inserting coercions on any of the
    /// provided expressions and returns their LUB (aka "common supertype").
    ///
    /// This is really an internal helper. From outside the coercion
    /// module, you should instantiate a `CoerceMany` instance.
    fn try_find_coercion_lub<E>(
        &self,
        cause: &ObligationCause<'tcx>,
        exprs: &[E],
        prev_ty: Ty<'tcx>,
        new: &hir::Expr<'_>,
        new_ty: Ty<'tcx>,
    ) -> RelateResult<'tcx, Ty<'tcx>>
    where
        E: AsCoercionSite,
    {
        let prev_ty = self.resolve_vars_with_obligations(prev_ty);
        let new_ty = self.resolve_vars_with_obligations(new_ty);
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

        // Special-case that coercion alone cannot handle:
        // Function items or non-capturing closures of differing IDs or InternalSubsts.
        let (a_sig, b_sig) = {
            let is_capturing_closure = |ty: Ty<'tcx>| {
                if let &ty::Closure(closure_def_id, _substs) = ty.kind() {
                    self.tcx.upvars_mentioned(closure_def_id.expect_local()).is_some()
                } else {
                    false
                }
            };
            if is_capturing_closure(prev_ty) || is_capturing_closure(new_ty) {
                (None, None)
            } else {
                match (prev_ty.kind(), new_ty.kind()) {
                    (ty::FnDef(..), ty::FnDef(..)) => {
                        // Don't reify if the function types have a LUB, i.e., they
                        // are the same function and their parameters have a LUB.
                        match self.commit_if_ok(|_| {
                            self.at(cause, self.param_env).lub(
                                DefineOpaqueTypes::No,
                                prev_ty,
                                new_ty,
                            )
                        }) {
                            // We have a LUB of prev_ty and new_ty, just return it.
                            Ok(ok) => return Ok(self.register_infer_ok_obligations(ok)),
                            Err(_) => {
                                (Some(prev_ty.fn_sig(self.tcx)), Some(new_ty.fn_sig(self.tcx)))
                            }
                        }
                    }
                    (ty::Closure(_, substs), ty::FnDef(..)) => {
                        let b_sig = new_ty.fn_sig(self.tcx);
                        let a_sig = self
                            .tcx
                            .signature_unclosure(substs.as_closure().sig(), b_sig.unsafety());
                        (Some(a_sig), Some(b_sig))
                    }
                    (ty::FnDef(..), ty::Closure(_, substs)) => {
                        let a_sig = prev_ty.fn_sig(self.tcx);
                        let b_sig = self
                            .tcx
                            .signature_unclosure(substs.as_closure().sig(), a_sig.unsafety());
                        (Some(a_sig), Some(b_sig))
                    }
                    (ty::Closure(_, substs_a), ty::Closure(_, substs_b)) => (
                        Some(self.tcx.signature_unclosure(
                            substs_a.as_closure().sig(),
                            hir::Unsafety::Normal,
                        )),
                        Some(self.tcx.signature_unclosure(
                            substs_b.as_closure().sig(),
                            hir::Unsafety::Normal,
                        )),
                    ),
                    _ => (None, None),
                }
            }
        };
        if let (Some(a_sig), Some(b_sig)) = (a_sig, b_sig) {
            // Intrinsics are not coercible to function pointers.
            if a_sig.abi() == Abi::RustIntrinsic
                || a_sig.abi() == Abi::PlatformIntrinsic
                || b_sig.abi() == Abi::RustIntrinsic
                || b_sig.abi() == Abi::PlatformIntrinsic
            {
                return Err(TypeError::IntrinsicCast);
            }
            // The signature must match.
            let (a_sig, b_sig) = self.normalize(new.span, (a_sig, b_sig));
            let sig = self
                .at(cause, self.param_env)
                .trace(prev_ty, new_ty)
                .lub(DefineOpaqueTypes::No, a_sig, b_sig)
                .map(|ok| self.register_infer_ok_obligations(ok))?;

            // Reify both sides and return the reified fn pointer type.
            let fn_ptr = self.tcx.mk_fn_ptr(sig);
            let prev_adjustment = match prev_ty.kind() {
                ty::Closure(..) => Adjust::Pointer(PointerCast::ClosureFnPointer(a_sig.unsafety())),
                ty::FnDef(..) => Adjust::Pointer(PointerCast::ReifyFnPointer),
                _ => unreachable!(),
            };
            let next_adjustment = match new_ty.kind() {
                ty::Closure(..) => Adjust::Pointer(PointerCast::ClosureFnPointer(b_sig.unsafety())),
                ty::FnDef(..) => Adjust::Pointer(PointerCast::ReifyFnPointer),
                _ => unreachable!(),
            };
            for expr in exprs.iter().map(|e| e.as_coercion_site()) {
                self.apply_adjustments(
                    expr,
                    vec![Adjustment { kind: prev_adjustment.clone(), target: fn_ptr }],
                );
            }
            self.apply_adjustments(new, vec![Adjustment { kind: next_adjustment, target: fn_ptr }]);
            return Ok(fn_ptr);
        }

        // Configure a Coerce instance to compute the LUB.
        // We don't allow two-phase borrows on any autorefs this creates since we
        // probably aren't processing function arguments here and even if we were,
        // they're going to get autorefed again anyway and we can apply 2-phase borrows
        // at that time.
        let mut coerce = Coerce::new(self, cause.clone(), AllowTwoPhase::No);
        coerce.use_lub = true;

        // First try to coerce the new expression to the type of the previous ones,
        // but only if the new expression has no coercion already applied to it.
        let mut first_error = None;
        if !self.typeck_results.borrow().adjustments().contains_key(new.hir_id) {
            let result = self.commit_if_ok(|_| coerce.coerce(new_ty, prev_ty));
            match result {
                Ok(ok) => {
                    let (adjustments, target) = self.register_infer_ok_obligations(ok);
                    self.apply_adjustments(new, adjustments);
                    debug!(
                        "coercion::try_find_coercion_lub: was able to coerce from new type {:?} to previous type {:?} ({:?})",
                        new_ty, prev_ty, target
                    );
                    return Ok(target);
                }
                Err(e) => first_error = Some(e),
            }
        }

        // Then try to coerce the previous expressions to the type of the new one.
        // This requires ensuring there are no coercions applied to *any* of the
        // previous expressions, other than noop reborrows (ignoring lifetimes).
        for expr in exprs {
            let expr = expr.as_coercion_site();
            let noop = match self.typeck_results.borrow().expr_adjustments(expr) {
                &[
                    Adjustment { kind: Adjust::Deref(_), .. },
                    Adjustment { kind: Adjust::Borrow(AutoBorrow::Ref(_, mutbl_adj)), .. },
                ] => {
                    match *self.node_ty(expr.hir_id).kind() {
                        ty::Ref(_, _, mt_orig) => {
                            let mutbl_adj: hir::Mutability = mutbl_adj.into();
                            // Reborrow that we can safely ignore, because
                            // the next adjustment can only be a Deref
                            // which will be merged into it.
                            mutbl_adj == mt_orig
                        }
                        _ => false,
                    }
                }
                &[Adjustment { kind: Adjust::NeverToAny, .. }] | &[] => true,
                _ => false,
            };

            if !noop {
                debug!(
                    "coercion::try_find_coercion_lub: older expression {:?} had adjustments, requiring LUB",
                    expr,
                );

                return self
                    .commit_if_ok(|_| {
                        self.at(cause, self.param_env).lub(DefineOpaqueTypes::No, prev_ty, new_ty)
                    })
                    .map(|ok| self.register_infer_ok_obligations(ok));
            }
        }

        match self.commit_if_ok(|_| coerce.coerce(prev_ty, new_ty)) {
            Err(_) => {
                // Avoid giving strange errors on failed attempts.
                if let Some(e) = first_error {
                    Err(e)
                } else {
                    self.commit_if_ok(|_| {
                        self.at(cause, self.param_env).lub(DefineOpaqueTypes::No, prev_ty, new_ty)
                    })
                    .map(|ok| self.register_infer_ok_obligations(ok))
                }
            }
            Ok(ok) => {
                let (adjustments, target) = self.register_infer_ok_obligations(ok);
                for expr in exprs {
                    let expr = expr.as_coercion_site();
                    self.apply_adjustments(expr, adjustments.clone());
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
pub struct CoerceMany<'tcx, 'exprs, E: AsCoercionSite> {
    expected_ty: Ty<'tcx>,
    final_ty: Option<Ty<'tcx>>,
    expressions: Expressions<'tcx, 'exprs, E>,
    pushed: usize,
}

/// The type of a `CoerceMany` that is storing up the expressions into
/// a buffer. We use this in `check/mod.rs` for things like `break`.
pub type DynamicCoerceMany<'tcx> = CoerceMany<'tcx, 'tcx, &'tcx hir::Expr<'tcx>>;

enum Expressions<'tcx, 'exprs, E: AsCoercionSite> {
    Dynamic(Vec<&'tcx hir::Expr<'tcx>>),
    UpFront(&'exprs [E]),
}

impl<'tcx, 'exprs, E: AsCoercionSite> CoerceMany<'tcx, 'exprs, E> {
    /// The usual case; collect the set of expressions dynamically.
    /// If the full set of coercion sites is known before hand,
    /// consider `with_coercion_sites()` instead to avoid allocation.
    pub fn new(expected_ty: Ty<'tcx>) -> Self {
        Self::make(expected_ty, Expressions::Dynamic(vec![]))
    }

    /// As an optimization, you can create a `CoerceMany` with a
    /// preexisting slice of expressions. In this case, you are
    /// expected to pass each element in the slice to `coerce(...)` in
    /// order. This is used with arrays in particular to avoid
    /// needlessly cloning the slice.
    pub fn with_coercion_sites(expected_ty: Ty<'tcx>, coercion_sites: &'exprs [E]) -> Self {
        Self::make(expected_ty, Expressions::UpFront(coercion_sites))
    }

    fn make(expected_ty: Ty<'tcx>, expressions: Expressions<'tcx, 'exprs, E>) -> Self {
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
    pub fn expected_ty(&self) -> Ty<'tcx> {
        self.expected_ty
    }

    /// Returns the current "merged type", representing our best-guess
    /// at the LUB of the expressions we've seen so far (if any). This
    /// isn't *final* until you call `self.complete()`, which will return
    /// the merged type.
    pub fn merged_ty(&self) -> Ty<'tcx> {
        self.final_ty.unwrap_or(self.expected_ty)
    }

    /// Indicates that the value generated by `expression`, which is
    /// of type `expression_ty`, is one of the possibilities that we
    /// could coerce from. This will record `expression`, and later
    /// calls to `coerce` may come back and add adjustments and things
    /// if necessary.
    pub fn coerce<'a>(
        &mut self,
        fcx: &FnCtxt<'a, 'tcx>,
        cause: &ObligationCause<'tcx>,
        expression: &'tcx hir::Expr<'tcx>,
        expression_ty: Ty<'tcx>,
    ) {
        self.coerce_inner(fcx, cause, Some(expression), expression_ty, None, false)
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
    pub fn coerce_forced_unit<'a>(
        &mut self,
        fcx: &FnCtxt<'a, 'tcx>,
        cause: &ObligationCause<'tcx>,
        augment_error: &mut dyn FnMut(&mut Diagnostic),
        label_unit_as_expected: bool,
    ) {
        self.coerce_inner(
            fcx,
            cause,
            None,
            fcx.tcx.mk_unit(),
            Some(augment_error),
            label_unit_as_expected,
        )
    }

    /// The inner coercion "engine". If `expression` is `None`, this
    /// is a forced-unit case, and hence `expression_ty` must be
    /// `Nil`.
    #[instrument(skip(self, fcx, augment_error, label_expression_as_expected), level = "debug")]
    pub(crate) fn coerce_inner<'a>(
        &mut self,
        fcx: &FnCtxt<'a, 'tcx>,
        cause: &ObligationCause<'tcx>,
        expression: Option<&'tcx hir::Expr<'tcx>>,
        mut expression_ty: Ty<'tcx>,
        augment_error: Option<&mut dyn FnMut(&mut Diagnostic)>,
        label_expression_as_expected: bool,
    ) {
        // Incorporate whatever type inference information we have
        // until now; in principle we might also want to process
        // pending obligations, but doing so should only improve
        // compatibility (hopefully that is true) by helping us
        // uncover never types better.
        if expression_ty.is_ty_var() {
            expression_ty = fcx.infcx.shallow_resolve(expression_ty);
        }

        // If we see any error types, just propagate that error
        // upwards.
        if let Err(guar) = (expression_ty, self.merged_ty()).error_reported() {
            self.final_ty = Some(fcx.tcx.ty_error(guar));
            return;
        }

        // Handle the actual type unification etc.
        let result = if let Some(expression) = expression {
            if self.pushed == 0 {
                // Special-case the first expression we are coercing.
                // To be honest, I'm not entirely sure why we do this.
                // We don't allow two-phase borrows, see comment in try_find_coercion_lub for why
                fcx.try_coerce(
                    expression,
                    expression_ty,
                    self.expected_ty,
                    AllowTwoPhase::No,
                    Some(cause.clone()),
                )
            } else {
                match self.expressions {
                    Expressions::Dynamic(ref exprs) => fcx.try_find_coercion_lub(
                        cause,
                        exprs,
                        self.merged_ty(),
                        expression,
                        expression_ty,
                    ),
                    Expressions::UpFront(ref coercion_sites) => fcx.try_find_coercion_lub(
                        cause,
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
            fcx.at(cause, fcx.param_env)
                // needed for tests/ui/type-alias-impl-trait/issue-65679-inst-opaque-ty-from-val-twice.rs
                .eq_exp(
                    DefineOpaqueTypes::Yes,
                    label_expression_as_expected,
                    expression_ty,
                    self.merged_ty(),
                )
                .map(|infer_ok| {
                    fcx.register_infer_ok_obligations(infer_ok);
                    expression_ty
                })
        };

        debug!(?result);
        match result {
            Ok(v) => {
                self.final_ty = Some(v);
                if let Some(e) = expression {
                    match self.expressions {
                        Expressions::Dynamic(ref mut buffer) => buffer.push(e),
                        Expressions::UpFront(coercion_sites) => {
                            // if the user gave us an array to validate, check that we got
                            // the next expression in the list, as expected
                            assert_eq!(
                                coercion_sites[self.pushed].as_coercion_site().hir_id,
                                e.hir_id
                            );
                        }
                    }
                    self.pushed += 1;
                }
            }
            Err(coercion_error) => {
                // Mark that we've failed to coerce the types here to suppress
                // any superfluous errors we might encounter while trying to
                // emit or provide suggestions on how to fix the initial error.
                fcx.set_tainted_by_errors(
                    fcx.tcx.sess.delay_span_bug(cause.span, "coercion error but no error emitted"),
                );
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
                let (expected, found) = fcx.resolve_vars_if_possible((expected, found));

                let mut err;
                let mut unsized_return = false;
                let mut visitor = CollectRetsVisitor { ret_exprs: vec![] };
                match *cause.code() {
                    ObligationCauseCode::ReturnNoExpression => {
                        err = struct_span_err!(
                            fcx.tcx.sess,
                            cause.span,
                            E0069,
                            "`return;` in a function whose return type is not `()`"
                        );
                        err.span_label(cause.span, "return type is not `()`");
                    }
                    ObligationCauseCode::BlockTailExpression(blk_id) => {
                        let parent_id = fcx.tcx.hir().parent_id(blk_id);
                        err = self.report_return_mismatched_types(
                            cause,
                            expected,
                            found,
                            coercion_error,
                            fcx,
                            parent_id,
                            expression,
                            Some(blk_id),
                        );
                        if !fcx.tcx.features().unsized_locals {
                            unsized_return = self.is_return_ty_definitely_unsized(fcx);
                        }
                        if let Some(expression) = expression
                            && let hir::ExprKind::Loop(loop_blk, ..) = expression.kind {
                              intravisit::walk_block(& mut visitor, loop_blk);
                        }
                    }
                    ObligationCauseCode::ReturnValue(id) => {
                        err = self.report_return_mismatched_types(
                            cause,
                            expected,
                            found,
                            coercion_error,
                            fcx,
                            id,
                            expression,
                            None,
                        );
                        if !fcx.tcx.features().unsized_locals {
                            unsized_return = self.is_return_ty_definitely_unsized(fcx);
                        }
                    }
                    _ => {
                        err = fcx.err_ctxt().report_mismatched_types(
                            cause,
                            expected,
                            found,
                            coercion_error,
                        );
                    }
                }

                if let Some(augment_error) = augment_error {
                    augment_error(&mut err);
                }

                let is_insufficiently_polymorphic =
                    matches!(coercion_error, TypeError::RegionsInsufficientlyPolymorphic(..));

                if !is_insufficiently_polymorphic && let Some(expr) = expression {
                    fcx.emit_coerce_suggestions(
                        &mut err,
                        expr,
                        found,
                        expected,
                        None,
                        Some(coercion_error),
                    );
                }

                if visitor.ret_exprs.len() > 0 && let Some(expr) = expression {
                    self.note_unreachable_loop_return(&mut err, &expr, &visitor.ret_exprs);
                }

                let reported = err.emit_unless(unsized_return);

                self.final_ty = Some(fcx.tcx.ty_error(reported));
            }
        }
    }

    fn note_unreachable_loop_return(
        &self,
        err: &mut Diagnostic,
        expr: &hir::Expr<'tcx>,
        ret_exprs: &Vec<&'tcx hir::Expr<'tcx>>,
    ) {
        let hir::ExprKind::Loop(_, _, _, loop_span) = expr.kind else { return;};
        let mut span: MultiSpan = vec![loop_span].into();
        span.push_span_label(loop_span, "this might have zero elements to iterate on");
        const MAXITER: usize = 3;
        let iter = ret_exprs.iter().take(MAXITER);
        for ret_expr in iter {
            span.push_span_label(
                ret_expr.span,
                "if the loop doesn't execute, this value would never get returned",
            );
        }
        err.span_note(
            span,
            "the function expects a value to always be returned, but loops might run zero times",
        );
        if MAXITER < ret_exprs.len() {
            err.note(format!(
                "if the loop doesn't execute, {} other values would never get returned",
                ret_exprs.len() - MAXITER
            ));
        }
        err.help(
            "return a value for the case when the loop has zero elements to iterate on, or \
           consider changing the return type to account for that possibility",
        );
    }

    fn report_return_mismatched_types<'a>(
        &self,
        cause: &ObligationCause<'tcx>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        ty_err: TypeError<'tcx>,
        fcx: &FnCtxt<'a, 'tcx>,
        id: hir::HirId,
        expression: Option<&'tcx hir::Expr<'tcx>>,
        blk_id: Option<hir::HirId>,
    ) -> DiagnosticBuilder<'a, ErrorGuaranteed> {
        let mut err = fcx.err_ctxt().report_mismatched_types(cause, expected, found, ty_err);

        let parent_id = fcx.tcx.hir().parent_id(id);
        let parent = fcx.tcx.hir().get(parent_id);
        if let Some(expr) = expression
            && let hir::Node::Expr(hir::Expr { kind: hir::ExprKind::Closure(&hir::Closure { body, .. }), .. }) = parent
            && !matches!(fcx.tcx.hir().body(body).value.kind, hir::ExprKind::Block(..))
        {
            fcx.suggest_missing_semicolon(&mut err, expr, expected, true);
        }
        // Verify that this is a tail expression of a function, otherwise the
        // label pointing out the cause for the type coercion will be wrong
        // as prior return coercions would not be relevant (#57664).
        let fn_decl = if let (Some(expr), Some(blk_id)) = (expression, blk_id) {
            let pointing_at_return_type =
                fcx.suggest_mismatched_types_on_tail(&mut err, expr, expected, found, blk_id);
            if let (Some(cond_expr), true, false) = (
                fcx.tcx.hir().get_if_cause(expr.hir_id),
                expected.is_unit(),
                pointing_at_return_type,
            )
                // If the block is from an external macro or try (`?`) desugaring, then
                // do not suggest adding a semicolon, because there's nowhere to put it.
                // See issues #81943 and #87051.
                && matches!(
                    cond_expr.span.desugaring_kind(),
                    None | Some(DesugaringKind::WhileLoop)
                ) && !in_external_macro(fcx.tcx.sess, cond_expr.span)
                    && !matches!(
                        cond_expr.kind,
                        hir::ExprKind::Match(.., hir::MatchSource::TryDesugar)
                    )
            {
                err.span_label(cond_expr.span, "expected this to be `()`");
                if expr.can_have_side_effects() {
                    fcx.suggest_semicolon_at_end(cond_expr.span, &mut err);
                }
            }
            fcx.get_node_fn_decl(parent)
                .map(|(fn_id, fn_decl, _, is_main)| (fn_id, fn_decl, is_main))
        } else {
            fcx.get_fn_decl(parent_id)
        };

        if let Some((fn_id, fn_decl, can_suggest)) = fn_decl {
            if blk_id.is_none() {
                fcx.suggest_missing_return_type(
                    &mut err,
                    &fn_decl,
                    expected,
                    found,
                    can_suggest,
                    fn_id,
                );
            }
        }

        let parent_id = fcx.tcx.hir().get_parent_item(id);
        let parent_item = fcx.tcx.hir().get_by_def_id(parent_id.def_id);

        if let (Some(expr), Some(_), Some((fn_id, fn_decl, _, _))) =
            (expression, blk_id, fcx.get_node_fn_decl(parent_item))
        {
            fcx.suggest_missing_break_or_return_expr(
                &mut err, expr, fn_decl, expected, found, id, fn_id,
            );
        }

        let ret_coercion_span = fcx.ret_coercion_span.get();

        if let Some(sp) = ret_coercion_span
            // If the closure has an explicit return type annotation, or if
            // the closure's return type has been inferred from outside
            // requirements (such as an Fn* trait bound), then a type error
            // may occur at the first return expression we see in the closure
            // (if it conflicts with the declared return type). Skip adding a
            // note in this case, since it would be incorrect.
            && let Some(fn_sig) = fcx.body_fn_sig()
            && fn_sig.output().is_ty_var()
        {
            err.span_note(
                sp,
                format!(
                    "return type inferred to be `{}` here",
                    expected
                ),
            );
        }

        err
    }

    /// Checks whether the return type is unsized via an obligation, which makes
    /// sure we consider `dyn Trait: Sized` where clauses, which are trivially
    /// false but technically valid for typeck.
    fn is_return_ty_definitely_unsized(&self, fcx: &FnCtxt<'_, 'tcx>) -> bool {
        if let Some(sig) = fcx.body_fn_sig() {
            !fcx.predicate_may_hold(&Obligation::new(
                fcx.tcx,
                ObligationCause::dummy(),
                fcx.param_env,
                ty::TraitRef::new(
                    fcx.tcx,
                    fcx.tcx.require_lang_item(hir::LangItem::Sized, None),
                    [sig.output()],
                ),
            ))
        } else {
            false
        }
    }

    pub fn complete<'a>(self, fcx: &FnCtxt<'a, 'tcx>) -> Ty<'tcx> {
        if let Some(final_ty) = self.final_ty {
            final_ty
        } else {
            // If we only had inputs that were of type `!` (or no
            // inputs at all), then the final type is `!`.
            assert_eq!(self.pushed, 0);
            fcx.tcx.types.never
        }
    }
}

/// Something that can be converted into an expression to which we can
/// apply a coercion.
pub trait AsCoercionSite {
    fn as_coercion_site(&self) -> &hir::Expr<'_>;
}

impl AsCoercionSite for hir::Expr<'_> {
    fn as_coercion_site(&self) -> &hir::Expr<'_> {
        self
    }
}

impl<'a, T> AsCoercionSite for &'a T
where
    T: AsCoercionSite,
{
    fn as_coercion_site(&self) -> &hir::Expr<'_> {
        (**self).as_coercion_site()
    }
}

impl AsCoercionSite for ! {
    fn as_coercion_site(&self) -> &hir::Expr<'_> {
        unreachable!()
    }
}

impl AsCoercionSite for hir::Arm<'_> {
    fn as_coercion_site(&self) -> &hir::Expr<'_> {
        &self.body
    }
}
