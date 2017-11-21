// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Code for type-checking closure expressions.

use super::{check_fn, Expectation, FnCtxt, GeneratorTypes};

use astconv::AstConv;
use rustc::hir::def_id::DefId;
use rustc::infer::{InferOk, InferResult};
use rustc::infer::LateBoundRegionConversionTime;
use rustc::infer::type_variable::TypeVariableOrigin;
use rustc::ty::{self, ToPolyTraitRef, Ty};
use rustc::ty::subst::Substs;
use rustc::ty::TypeFoldable;
use std::cmp;
use std::iter;
use syntax::abi::Abi;
use rustc::hir;

struct ClosureSignatures<'tcx> {
    bound_sig: ty::PolyFnSig<'tcx>,
    liberated_sig: ty::FnSig<'tcx>,
}

impl<'a, 'gcx, 'tcx> FnCtxt<'a, 'gcx, 'tcx> {
    pub fn check_expr_closure(
        &self,
        expr: &hir::Expr,
        _capture: hir::CaptureClause,
        decl: &'gcx hir::FnDecl,
        body_id: hir::BodyId,
        expected: Expectation<'tcx>,
    ) -> Ty<'tcx> {
        debug!(
            "check_expr_closure(expr={:?},expected={:?})",
            expr,
            expected
        );

        // It's always helpful for inference if we know the kind of
        // closure sooner rather than later, so first examine the expected
        // type, and see if can glean a closure kind from there.
        let (expected_sig, expected_kind) = match expected.to_option(self) {
            Some(ty) => self.deduce_expectations_from_expected_type(ty),
            None => (None, None),
        };
        let body = self.tcx.hir.body(body_id);
        self.check_closure(expr, expected_kind, decl, body, expected_sig)
    }

    fn check_closure(
        &self,
        expr: &hir::Expr,
        opt_kind: Option<ty::ClosureKind>,
        decl: &'gcx hir::FnDecl,
        body: &'gcx hir::Body,
        expected_sig: Option<ty::FnSig<'tcx>>,
    ) -> Ty<'tcx> {
        debug!(
            "check_closure(opt_kind={:?}, expected_sig={:?})",
            opt_kind,
            expected_sig
        );

        let expr_def_id = self.tcx.hir.local_def_id(expr.id);

        let ClosureSignatures {
            bound_sig,
            liberated_sig,
        } = self.sig_of_closure(expr_def_id, decl, body, expected_sig);

        debug!("check_closure: ty_of_closure returns {:?}", liberated_sig);

        let generator_types = check_fn(
            self,
            self.param_env,
            liberated_sig,
            decl,
            expr.id,
            body,
            true,
        ).1;

        // Create type variables (for now) to represent the transformed
        // types of upvars. These will be unified during the upvar
        // inference phase (`upvar.rs`).
        let base_substs =
            Substs::identity_for_item(self.tcx, self.tcx.closure_base_def_id(expr_def_id));
        let substs = base_substs.extend_to(
            self.tcx,
            expr_def_id,
            |_, _| span_bug!(expr.span, "closure has region param"),
            |_, _| {
                self.infcx
                    .next_ty_var(TypeVariableOrigin::ClosureSynthetic(expr.span))
            },
        );
        let substs = ty::ClosureSubsts { substs };
        let closure_type = self.tcx.mk_closure(expr_def_id, substs);

        if let Some(GeneratorTypes { yield_ty, interior }) = generator_types {
            self.demand_eqtype(expr.span,
                               yield_ty,
                               substs.generator_yield_ty(expr_def_id, self.tcx));
            self.demand_eqtype(expr.span,
                               liberated_sig.output(),
                               substs.generator_return_ty(expr_def_id, self.tcx));
            return self.tcx.mk_generator(expr_def_id, substs, interior);
        }

        debug!(
            "check_closure: expr.id={:?} closure_type={:?}",
            expr.id,
            closure_type
        );

        // Tuple up the arguments and insert the resulting function type into
        // the `closures` table.
        let sig = bound_sig.map_bound(|sig| {
            self.tcx.mk_fn_sig(
                iter::once(self.tcx.intern_tup(sig.inputs(), false)),
                sig.output(),
                sig.variadic,
                sig.unsafety,
                sig.abi,
            )
        });

        debug!(
            "check_closure: expr_def_id={:?}, sig={:?}, opt_kind={:?}",
            expr_def_id,
            sig,
            opt_kind
        );

        let sig_fn_ptr_ty = self.tcx.mk_fn_ptr(sig);
        self.demand_eqtype(expr.span,
                           sig_fn_ptr_ty,
                           substs.closure_sig_ty(expr_def_id, self.tcx));

        if let Some(kind) = opt_kind {
            self.demand_eqtype(expr.span,
                               kind.to_ty(self.tcx),
                               substs.closure_kind_ty(expr_def_id, self.tcx));
        }

        closure_type
    }

    fn deduce_expectations_from_expected_type(
        &self,
        expected_ty: Ty<'tcx>,
    ) -> (Option<ty::FnSig<'tcx>>, Option<ty::ClosureKind>) {
        debug!(
            "deduce_expectations_from_expected_type(expected_ty={:?})",
            expected_ty
        );

        match expected_ty.sty {
            ty::TyDynamic(ref object_type, ..) => {
                let sig = object_type
                    .projection_bounds()
                    .filter_map(|pb| {
                        let pb = pb.with_self_ty(self.tcx, self.tcx.types.err);
                        self.deduce_sig_from_projection(&pb)
                    })
                    .next();
                let kind = object_type
                    .principal()
                    .and_then(|p| self.tcx.lang_items().fn_trait_kind(p.def_id()));
                (sig, kind)
            }
            ty::TyInfer(ty::TyVar(vid)) => self.deduce_expectations_from_obligations(vid),
            ty::TyFnPtr(sig) => (Some(sig.skip_binder().clone()), Some(ty::ClosureKind::Fn)),
            _ => (None, None),
        }
    }

    fn deduce_expectations_from_obligations(
        &self,
        expected_vid: ty::TyVid,
    ) -> (Option<ty::FnSig<'tcx>>, Option<ty::ClosureKind>) {
        let fulfillment_cx = self.fulfillment_cx.borrow();
        // Here `expected_ty` is known to be a type inference variable.

        let expected_sig = fulfillment_cx
            .pending_obligations()
            .iter()
            .map(|obligation| &obligation.obligation)
            .filter_map(|obligation| {
                debug!(
                    "deduce_expectations_from_obligations: obligation.predicate={:?}",
                    obligation.predicate
                );

                match obligation.predicate {
                    // Given a Projection predicate, we can potentially infer
                    // the complete signature.
                    ty::Predicate::Projection(ref proj_predicate) => {
                        let trait_ref = proj_predicate.to_poly_trait_ref(self.tcx);
                        self.self_type_matches_expected_vid(trait_ref, expected_vid)
                            .and_then(|_| self.deduce_sig_from_projection(proj_predicate))
                    }
                    _ => None,
                }
            })
            .next();

        // Even if we can't infer the full signature, we may be able to
        // infer the kind. This can occur if there is a trait-reference
        // like `F : Fn<A>`. Note that due to subtyping we could encounter
        // many viable options, so pick the most restrictive.
        let expected_kind = fulfillment_cx
            .pending_obligations()
            .iter()
            .map(|obligation| &obligation.obligation)
            .filter_map(|obligation| {
                let opt_trait_ref = match obligation.predicate {
                    ty::Predicate::Projection(ref data) => Some(data.to_poly_trait_ref(self.tcx)),
                    ty::Predicate::Trait(ref data) => Some(data.to_poly_trait_ref()),
                    ty::Predicate::Equate(..) => None,
                    ty::Predicate::Subtype(..) => None,
                    ty::Predicate::RegionOutlives(..) => None,
                    ty::Predicate::TypeOutlives(..) => None,
                    ty::Predicate::WellFormed(..) => None,
                    ty::Predicate::ObjectSafe(..) => None,
                    ty::Predicate::ConstEvaluatable(..) => None,

                    // NB: This predicate is created by breaking down a
                    // `ClosureType: FnFoo()` predicate, where
                    // `ClosureType` represents some `TyClosure`. It can't
                    // possibly be referring to the current closure,
                    // because we haven't produced the `TyClosure` for
                    // this closure yet; this is exactly why the other
                    // code is looking for a self type of a unresolved
                    // inference variable.
                    ty::Predicate::ClosureKind(..) => None,
                };
                opt_trait_ref
                    .and_then(|tr| self.self_type_matches_expected_vid(tr, expected_vid))
                    .and_then(|tr| self.tcx.lang_items().fn_trait_kind(tr.def_id()))
            })
            .fold(None, |best, cur| {
                Some(best.map_or(cur, |best| cmp::min(best, cur)))
            });

        (expected_sig, expected_kind)
    }

    /// Given a projection like "<F as Fn(X)>::Result == Y", we can deduce
    /// everything we need to know about a closure.
    fn deduce_sig_from_projection(
        &self,
        projection: &ty::PolyProjectionPredicate<'tcx>,
    ) -> Option<ty::FnSig<'tcx>> {
        let tcx = self.tcx;

        debug!("deduce_sig_from_projection({:?})", projection);

        let trait_ref = projection.to_poly_trait_ref(tcx);

        if tcx.lang_items().fn_trait_kind(trait_ref.def_id()).is_none() {
            return None;
        }

        let arg_param_ty = trait_ref.substs().type_at(1);
        let arg_param_ty = self.resolve_type_vars_if_possible(&arg_param_ty);
        debug!(
            "deduce_sig_from_projection: arg_param_ty {:?}",
            arg_param_ty
        );

        let input_tys = match arg_param_ty.sty {
            ty::TyTuple(tys, _) => tys.into_iter(),
            _ => {
                return None;
            }
        };

        let ret_param_ty = projection.0.ty;
        let ret_param_ty = self.resolve_type_vars_if_possible(&ret_param_ty);
        debug!(
            "deduce_sig_from_projection: ret_param_ty {:?}",
            ret_param_ty
        );

        let fn_sig = self.tcx.mk_fn_sig(
            input_tys.cloned(),
            ret_param_ty,
            false,
            hir::Unsafety::Normal,
            Abi::Rust,
        );
        debug!("deduce_sig_from_projection: fn_sig {:?}", fn_sig);

        Some(fn_sig)
    }

    fn self_type_matches_expected_vid(
        &self,
        trait_ref: ty::PolyTraitRef<'tcx>,
        expected_vid: ty::TyVid,
    ) -> Option<ty::PolyTraitRef<'tcx>> {
        let self_ty = self.shallow_resolve(trait_ref.self_ty());
        debug!(
            "self_type_matches_expected_vid(trait_ref={:?}, self_ty={:?})",
            trait_ref,
            self_ty
        );
        match self_ty.sty {
            ty::TyInfer(ty::TyVar(v)) if expected_vid == v => Some(trait_ref),
            _ => None,
        }
    }

    fn sig_of_closure(
        &self,
        expr_def_id: DefId,
        decl: &hir::FnDecl,
        body: &hir::Body,
        expected_sig: Option<ty::FnSig<'tcx>>,
    ) -> ClosureSignatures<'tcx> {
        if let Some(e) = expected_sig {
            self.sig_of_closure_with_expectation(expr_def_id, decl, body, e)
        } else {
            self.sig_of_closure_no_expectation(expr_def_id, decl, body)
        }
    }

    /// If there is no expected signature, then we will convert the
    /// types that the user gave into a signature.
    fn sig_of_closure_no_expectation(
        &self,
        expr_def_id: DefId,
        decl: &hir::FnDecl,
        body: &hir::Body,
    ) -> ClosureSignatures<'tcx> {
        debug!("sig_of_closure_no_expectation()");

        let bound_sig = self.supplied_sig_of_closure(decl);

        self.closure_sigs(expr_def_id, body, bound_sig)
    }

    /// Invoked to compute the signature of a closure expression. This
    /// combines any user-provided type annotations (e.g., `|x: u32|
    /// -> u32 { .. }`) with the expected signature.
    ///
    /// The approach is as follows:
    ///
    /// - Let `S` be the (higher-ranked) signature that we derive from the user's annotations.
    /// - Let `E` be the (higher-ranked) signature that we derive from the expectations, if any.
    ///   - If we have no expectation `E`, then the signature of the closure is `S`.
    ///   - Otherwise, the signature of the closure is E. Moreover:
    ///     - Skolemize the late-bound regions in `E`, yielding `E'`.
    ///     - Instantiate all the late-bound regions bound in the closure within `S`
    ///       with fresh (existential) variables, yielding `S'`
    ///     - Require that `E' = S'`
    ///       - We could use some kind of subtyping relationship here,
    ///         I imagine, but equality is easier and works fine for
    ///         our purposes.
    ///
    /// The key intuition here is that the user's types must be valid
    /// from "the inside" of the closure, but the expectation
    /// ultimately drives the overall signature.
    ///
    /// # Examples
    ///
    /// ```
    /// fn with_closure<F>(_: F)
    ///   where F: Fn(&u32) -> &u32 { .. }
    ///
    /// with_closure(|x: &u32| { ... })
    /// ```
    ///
    /// Here:
    /// - E would be `fn(&u32) -> &u32`.
    /// - S would be `fn(&u32) ->
    /// - E' is `&'!0 u32 -> &'!0 u32`
    /// - S' is `&'?0 u32 -> ?T`
    ///
    /// S' can be unified with E' with `['?0 = '!0, ?T = &'!10 u32]`.
    ///
    /// # Arguments
    ///
    /// - `expr_def_id`: the def-id of the closure expression
    /// - `decl`: the HIR declaration of the closure
    /// - `body`: the body of the closure
    /// - `expected_sig`: the expected signature (if any). Note that
    ///   this is missing a binder: that is, there may be late-bound
    ///   regions with depth 1, which are bound then by the closure.
    fn sig_of_closure_with_expectation(
        &self,
        expr_def_id: DefId,
        decl: &hir::FnDecl,
        body: &hir::Body,
        expected_sig: ty::FnSig<'tcx>,
    ) -> ClosureSignatures<'tcx> {
        debug!(
            "sig_of_closure_with_expectation(expected_sig={:?})",
            expected_sig
        );

        // Watch out for some surprises and just ignore the
        // expectation if things don't see to match up with what we
        // expect.
        if expected_sig.variadic != decl.variadic {
            return self.sig_of_closure_no_expectation(expr_def_id, decl, body);
        } else if expected_sig.inputs_and_output.len() != decl.inputs.len() + 1 {
            // we could probably handle this case more gracefully
            return self.sig_of_closure_no_expectation(expr_def_id, decl, body);
        }

        // Create a `PolyFnSig`. Note the oddity that late bound
        // regions appearing free in `expected_sig` are now bound up
        // in this binder we are creating.
        assert!(!expected_sig.has_regions_escaping_depth(1));
        let bound_sig = ty::Binder(self.tcx.mk_fn_sig(
            expected_sig.inputs().iter().cloned(),
            expected_sig.output(),
            decl.variadic,
            hir::Unsafety::Normal,
            Abi::RustCall,
        ));

        // `deduce_expectations_from_expected_type` introduces
        // late-bound lifetimes defined elsewhere, which we now
        // anonymize away, so as not to confuse the user.
        let bound_sig = self.tcx.anonymize_late_bound_regions(&bound_sig);

        let closure_sigs = self.closure_sigs(expr_def_id, body, bound_sig);

        // Up till this point, we have ignored the annotations that the user
        // gave. This function will check that they unify successfully.
        // Along the way, it also writes out entries for types that the user
        // wrote into our tables, which are then later used by the privacy
        // check.
        match self.check_supplied_sig_against_expectation(decl, &closure_sigs) {
            Ok(infer_ok) => self.register_infer_ok_obligations(infer_ok),
            Err(_) => return self.sig_of_closure_no_expectation(expr_def_id, decl, body),
        }

        closure_sigs
    }

    /// Enforce the user's types against the expectation.  See
    /// `sig_of_closure_with_expectation` for details on the overall
    /// strategy.
    fn check_supplied_sig_against_expectation(
        &self,
        decl: &hir::FnDecl,
        expected_sigs: &ClosureSignatures<'tcx>,
    ) -> InferResult<'tcx, ()> {
        // Get the signature S that the user gave.
        //
        // (See comment on `sig_of_closure_with_expectation` for the
        // meaning of these letters.)
        let supplied_sig = self.supplied_sig_of_closure(decl);

        debug!(
            "check_supplied_sig_against_expectation: supplied_sig={:?}",
            supplied_sig
        );

        // FIXME(#45727): As discussed in [this comment][c1], naively
        // forcing equality here actually results in suboptimal error
        // messages in some cases.  For now, if there would have been
        // an obvious error, we fallback to declaring the type of the
        // closure to be the one the user gave, which allows other
        // error message code to trigger.
        //
        // However, I think [there is potential to do even better
        // here][c2], since in *this* code we have the precise span of
        // the type parameter in question in hand when we report the
        // error.
        //
        // [c1]: https://github.com/rust-lang/rust/pull/45072#issuecomment-341089706
        // [c2]: https://github.com/rust-lang/rust/pull/45072#issuecomment-341096796
        self.infcx.commit_if_ok(|_| {
            let mut all_obligations = vec![];

            // The liberated version of this signature should be be a subtype
            // of the liberated form of the expectation.
            for ((hir_ty, &supplied_ty), expected_ty) in decl.inputs.iter()
                           .zip(*supplied_sig.inputs().skip_binder()) // binder moved to (*) below
                           .zip(expected_sigs.liberated_sig.inputs())
            // `liberated_sig` is E'.
            {
                // Instantiate (this part of..) S to S', i.e., with fresh variables.
                let (supplied_ty, _) = self.infcx.replace_late_bound_regions_with_fresh_var(
                    hir_ty.span,
                    LateBoundRegionConversionTime::FnCall,
                    &ty::Binder(supplied_ty),
                ); // recreated from (*) above

                // Check that E' = S'.
                let cause = &self.misc(hir_ty.span);
                let InferOk {
                    value: (),
                    obligations,
                } = self.at(cause, self.param_env)
                    .eq(*expected_ty, supplied_ty)?;
                all_obligations.extend(obligations);
            }

            let (supplied_output_ty, _) = self.infcx.replace_late_bound_regions_with_fresh_var(
                decl.output.span(),
                LateBoundRegionConversionTime::FnCall,
                &supplied_sig.output(),
            );
            let cause = &self.misc(decl.output.span());
            let InferOk {
                value: (),
                obligations,
            } = self.at(cause, self.param_env)
                .eq(expected_sigs.liberated_sig.output(), supplied_output_ty)?;
            all_obligations.extend(obligations);

            Ok(InferOk {
                value: (),
                obligations: all_obligations,
            })
        })
    }

    /// If there is no expected signature, then we will convert the
    /// types that the user gave into a signature.
    fn supplied_sig_of_closure(&self, decl: &hir::FnDecl) -> ty::PolyFnSig<'tcx> {
        let astconv: &AstConv = self;

        // First, convert the types that the user supplied (if any).
        let supplied_arguments = decl.inputs.iter().map(|a| astconv.ast_ty_to_ty(a));
        let supplied_return = match decl.output {
            hir::Return(ref output) => astconv.ast_ty_to_ty(&output),
            hir::DefaultReturn(_) => astconv.ty_infer(decl.output.span()),
        };

        let result = ty::Binder(self.tcx.mk_fn_sig(
            supplied_arguments,
            supplied_return,
            decl.variadic,
            hir::Unsafety::Normal,
            Abi::RustCall,
        ));

        debug!("supplied_sig_of_closure: result={:?}", result);

        result
    }

    fn closure_sigs(
        &self,
        expr_def_id: DefId,
        body: &hir::Body,
        bound_sig: ty::PolyFnSig<'tcx>,
    ) -> ClosureSignatures<'tcx> {
        let liberated_sig = self.tcx().liberate_late_bound_regions(expr_def_id, &bound_sig);
        let liberated_sig = self.inh.normalize_associated_types_in(
            body.value.span,
            body.value.id,
            self.param_env,
            &liberated_sig,
        );
        ClosureSignatures {
            bound_sig,
            liberated_sig,
        }
    }
}
