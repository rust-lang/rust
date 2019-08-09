use super::{probe, MethodCallee};

use crate::astconv::AstConv;
use crate::check::{FnCtxt, PlaceOp, callee, Needs};
use crate::hir::GenericArg;
use crate::hir::def_id::DefId;
use rustc::ty::subst::{Subst, SubstsRef};
use rustc::traits;
use rustc::ty::{self, Ty, GenericParamDefKind};
use rustc::ty::adjustment::{Adjustment, Adjust, OverloadedDeref, PointerCast};
use rustc::ty::adjustment::{AllowTwoPhase, AutoBorrow, AutoBorrowMutability};
use rustc::ty::fold::TypeFoldable;
use rustc::infer::{self, InferOk};
use rustc::hir;
use syntax_pos::Span;

use std::ops::Deref;

struct ConfirmContext<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    span: Span,
    self_expr: &'tcx hir::Expr,
    call_expr: &'tcx hir::Expr,
}

impl<'a, 'tcx> Deref for ConfirmContext<'a, 'tcx> {
    type Target = FnCtxt<'a, 'tcx>;
    fn deref(&self) -> &Self::Target {
        &self.fcx
    }
}

pub struct ConfirmResult<'tcx> {
    pub callee: MethodCallee<'tcx>,
    pub illegal_sized_bound: bool,
}

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub fn confirm_method(
        &self,
        span: Span,
        self_expr: &'tcx hir::Expr,
        call_expr: &'tcx hir::Expr,
        unadjusted_self_ty: Ty<'tcx>,
        pick: probe::Pick<'tcx>,
        segment: &hir::PathSegment,
    ) -> ConfirmResult<'tcx> {
        debug!(
            "confirm(unadjusted_self_ty={:?}, pick={:?}, generic_args={:?})",
            unadjusted_self_ty,
            pick,
            segment.args,
        );

        let mut confirm_cx = ConfirmContext::new(self, span, self_expr, call_expr);
        confirm_cx.confirm(unadjusted_self_ty, pick, segment)
    }
}

impl<'a, 'tcx> ConfirmContext<'a, 'tcx> {
    fn new(
        fcx: &'a FnCtxt<'a, 'tcx>,
        span: Span,
        self_expr: &'tcx hir::Expr,
        call_expr: &'tcx hir::Expr,
    ) -> ConfirmContext<'a, 'tcx> {
        ConfirmContext {
            fcx,
            span,
            self_expr,
            call_expr,
        }
    }

    fn confirm(
        &mut self,
        unadjusted_self_ty: Ty<'tcx>,
        pick: probe::Pick<'tcx>,
        segment: &hir::PathSegment,
    ) -> ConfirmResult<'tcx> {
        // Adjust the self expression the user provided and obtain the adjusted type.
        let self_ty = self.adjust_self_ty(unadjusted_self_ty, &pick);

        // Create substitutions for the method's type parameters.
        let rcvr_substs = self.fresh_receiver_substs(self_ty, &pick);
        let all_substs = self.instantiate_method_substs(&pick, segment, rcvr_substs);

        debug!("all_substs={:?}", all_substs);

        // Create the final signature for the method, replacing late-bound regions.
        let (method_sig, method_predicates) = self.instantiate_method_sig(&pick, all_substs);

        // Unify the (adjusted) self type with what the method expects.
        //
        // SUBTLE: if we want good error messages, because of "guessing" while matching
        // traits, no trait system method can be called before this point because they
        // could alter our Self-type, except for normalizing the receiver from the
        // signature (which is also done during probing).
        let method_sig_rcvr =
            self.normalize_associated_types_in(self.span, &method_sig.inputs()[0]);
        self.unify_receivers(self_ty, method_sig_rcvr);

        let (method_sig, method_predicates) =
            self.normalize_associated_types_in(self.span, &(method_sig, method_predicates));

        // Make sure nobody calls `drop()` explicitly.
        self.enforce_illegal_method_limitations(&pick);

        // If there is a `Self: Sized` bound and `Self` is a trait object, it is possible that
        // something which derefs to `Self` actually implements the trait and the caller
        // wanted to make a static dispatch on it but forgot to import the trait.
        // See test `src/test/ui/issue-35976.rs`.
        //
        // In that case, we'll error anyway, but we'll also re-run the search with all traits
        // in scope, and if we find another method which can be used, we'll output an
        // appropriate hint suggesting to import the trait.
        let illegal_sized_bound = self.predicates_require_illegal_sized_bound(&method_predicates);

        // Add any trait/regions obligations specified on the method's type parameters.
        // We won't add these if we encountered an illegal sized bound, so that we can use
        // a custom error in that case.
        if !illegal_sized_bound {
            let method_ty = self.tcx.mk_fn_ptr(ty::Binder::bind(method_sig));
            self.add_obligations(method_ty, all_substs, &method_predicates);
        }

        // Create the final `MethodCallee`.
        let callee = MethodCallee {
            def_id: pick.item.def_id,
            substs: all_substs,
            sig: method_sig,
        };

        if let Some(hir::MutMutable) = pick.autoref {
            self.convert_place_derefs_to_mutable();
        }

        ConfirmResult { callee, illegal_sized_bound }
    }

    ///////////////////////////////////////////////////////////////////////////
    // ADJUSTMENTS

    fn adjust_self_ty(&mut self,
                      unadjusted_self_ty: Ty<'tcx>,
                      pick: &probe::Pick<'tcx>)
                      -> Ty<'tcx> {
        // Commit the autoderefs by calling `autoderef` again, but this
        // time writing the results into the various tables.
        let mut autoderef = self.autoderef(self.span, unadjusted_self_ty);
        let (_, n) = autoderef.nth(pick.autoderefs).unwrap();
        assert_eq!(n, pick.autoderefs);

        let mut adjustments = autoderef.adjust_steps(self, Needs::None);

        let mut target = autoderef.unambiguous_final_ty(self);

        if let Some(mutbl) = pick.autoref {
            let region = self.next_region_var(infer::Autoref(self.span));
            target = self.tcx.mk_ref(region, ty::TypeAndMut {
                mutbl,
                ty: target
            });
            let mutbl = match mutbl {
                hir::MutImmutable => AutoBorrowMutability::Immutable,
                hir::MutMutable => AutoBorrowMutability::Mutable {
                    // Method call receivers are the primary use case
                    // for two-phase borrows.
                    allow_two_phase_borrow: AllowTwoPhase::Yes,
                }
            };
            adjustments.push(Adjustment {
                kind: Adjust::Borrow(AutoBorrow::Ref(region, mutbl)),
                target
            });

            if let Some(unsize_target) = pick.unsize {
                target = self.tcx.mk_ref(region, ty::TypeAndMut {
                    mutbl: mutbl.into(),
                    ty: unsize_target
                });
                adjustments.push(Adjustment {
                    kind: Adjust::Pointer(PointerCast::Unsize),
                    target
                });
            }
        } else {
            // No unsizing should be performed without autoref (at
            // least during method dispach). This is because we
            // currently only unsize `[T;N]` to `[T]`, and naturally
            // that must occur being a reference.
            assert!(pick.unsize.is_none());
        }

        autoderef.finalize(self);

        // Write out the final adjustments.
        self.apply_adjustments(self.self_expr, adjustments);

        target
    }

    /// Returns a set of substitutions for the method *receiver* where all type and region
    /// parameters are instantiated with fresh variables. This substitution does not include any
    /// parameters declared on the method itself.
    ///
    /// Note that this substitution may include late-bound regions from the impl level. If so,
    /// these are instantiated later in the `instantiate_method_sig` routine.
    fn fresh_receiver_substs(&mut self,
                             self_ty: Ty<'tcx>,
                             pick: &probe::Pick<'tcx>)
                             -> SubstsRef<'tcx> {
        match pick.kind {
            probe::InherentImplPick => {
                let impl_def_id = pick.item.container.id();
                assert!(self.tcx.impl_trait_ref(impl_def_id).is_none(),
                        "impl {:?} is not an inherent impl",
                        impl_def_id);
                self.impl_self_ty(self.span, impl_def_id).substs
            }

            probe::ObjectPick => {
                let trait_def_id = pick.item.container.id();
                self.extract_existential_trait_ref(self_ty, |this, object_ty, principal| {
                    // The object data has no entry for the Self
                    // Type. For the purposes of this method call, we
                    // substitute the object type itself. This
                    // wouldn't be a sound substitution in all cases,
                    // since each instance of the object type is a
                    // different existential and hence could match
                    // distinct types (e.g., if `Self` appeared as an
                    // argument type), but those cases have already
                    // been ruled out when we deemed the trait to be
                    // "object safe".
                    let original_poly_trait_ref = principal.with_self_ty(this.tcx, object_ty);
                    let upcast_poly_trait_ref = this.upcast(original_poly_trait_ref, trait_def_id);
                    let upcast_trait_ref =
                        this.replace_bound_vars_with_fresh_vars(&upcast_poly_trait_ref);
                    debug!("original_poly_trait_ref={:?} upcast_trait_ref={:?} target_trait={:?}",
                           original_poly_trait_ref,
                           upcast_trait_ref,
                           trait_def_id);
                    upcast_trait_ref.substs
                })
            }

            probe::TraitPick => {
                let trait_def_id = pick.item.container.id();

                // Make a trait reference `$0 : Trait<$1...$n>`
                // consisting entirely of type variables. Later on in
                // the process we will unify the transformed-self-type
                // of the method with the actual type in order to
                // unify some of these variables.
                self.fresh_substs_for_item(self.span, trait_def_id)
            }

            probe::WhereClausePick(ref poly_trait_ref) => {
                // Where clauses can have bound regions in them. We need to instantiate
                // those to convert from a poly-trait-ref to a trait-ref.
                self.replace_bound_vars_with_fresh_vars(&poly_trait_ref).substs
            }
        }
    }

    fn extract_existential_trait_ref<R, F>(&mut self, self_ty: Ty<'tcx>, mut closure: F) -> R
    where
        F: FnMut(&mut ConfirmContext<'a, 'tcx>, Ty<'tcx>, ty::PolyExistentialTraitRef<'tcx>) -> R,
    {
        // If we specified that this is an object method, then the
        // self-type ought to be something that can be dereferenced to
        // yield an object-type (e.g., `&Object` or `Box<Object>`
        // etc).

        // FIXME: this feels, like, super dubious
        self.fcx
            .autoderef(self.span, self_ty)
            .include_raw_pointers()
            .filter_map(|(ty, _)|
                match ty.sty {
                    ty::Dynamic(ref data, ..) => {
                        Some(closure(self, ty, data.principal().unwrap_or_else(|| {
                            span_bug!(self.span, "calling trait method on empty object?")
                        })))
                    },
                    _ => None,
                }
            )
            .next()
            .unwrap_or_else(||
                span_bug!(self.span,
                          "self-type `{}` for ObjectPick never dereferenced to an object",
                          self_ty)
            )
    }

    fn instantiate_method_substs(
        &mut self,
        pick: &probe::Pick<'tcx>,
        seg: &hir::PathSegment,
        parent_substs: SubstsRef<'tcx>,
    ) -> SubstsRef<'tcx> {
        // Determine the values for the generic parameters of the method.
        // If they were not explicitly supplied, just construct fresh
        // variables.
        let generics = self.tcx.generics_of(pick.item.def_id);
        AstConv::check_generic_arg_count_for_call(
            self.tcx,
            self.span,
            &generics,
            &seg,
            true, // `is_method_call`
        );

        // Create subst for early-bound lifetime parameters, combining
        // parameters from the type and those from the method.
        assert_eq!(generics.parent_count, parent_substs.len());

        AstConv::create_substs_for_generic_args(
            self.tcx,
            pick.item.def_id,
            parent_substs,
            false,
            None,
            // Provide the generic args, and whether types should be inferred.
            |_| {
                // The last argument of the returned tuple here is unimportant.
                if let Some(ref data) = seg.args {
                    (Some(data), false)
                } else {
                    (None, false)
                }
            },
            // Provide substitutions for parameters for which (valid) arguments have been provided.
            |param, arg| {
                match (&param.kind, arg) {
                    (GenericParamDefKind::Lifetime, GenericArg::Lifetime(lt)) => {
                        AstConv::ast_region_to_region(self.fcx, lt, Some(param)).into()
                    }
                    (GenericParamDefKind::Type { .. }, GenericArg::Type(ty)) => {
                        self.to_ty(ty).into()
                    }
                    (GenericParamDefKind::Const, GenericArg::Const(ct)) => {
                        self.to_const(&ct.value, self.tcx.type_of(param.def_id)).into()
                    }
                    _ => unreachable!(),
                }
            },
            // Provide substitutions for parameters for which arguments are inferred.
            |_, param, _| self.var_for_def(self.span, param),
        )
    }

    fn unify_receivers(&mut self, self_ty: Ty<'tcx>, method_self_ty: Ty<'tcx>) {
        match self.at(&self.misc(self.span), self.param_env).sup(method_self_ty, self_ty) {
            Ok(InferOk { obligations, value: () }) => {
                self.register_predicates(obligations);
            }
            Err(_) => {
                span_bug!(self.span,
                          "{} was a subtype of {} but now is not?",
                          self_ty,
                          method_self_ty);
            }
        }
    }

    // NOTE: this returns the *unnormalized* predicates and method sig. Because of
    // inference guessing, the predicates and method signature can't be normalized
    // until we unify the `Self` type.
    fn instantiate_method_sig(&mut self,
                              pick: &probe::Pick<'tcx>,
                              all_substs: SubstsRef<'tcx>)
                              -> (ty::FnSig<'tcx>, ty::InstantiatedPredicates<'tcx>) {
        debug!("instantiate_method_sig(pick={:?}, all_substs={:?})",
               pick,
               all_substs);

        // Instantiate the bounds on the method with the
        // type/early-bound-regions substitutions performed. There can
        // be no late-bound regions appearing here.
        let def_id = pick.item.def_id;
        let method_predicates = self.tcx.predicates_of(def_id)
                                    .instantiate(self.tcx, all_substs);

        debug!("method_predicates after subst = {:?}", method_predicates);

        let sig = self.tcx.fn_sig(def_id);

        // Instantiate late-bound regions and substitute the trait
        // parameters into the method type to get the actual method type.
        //
        // N.B., instantiate late-bound regions first so that
        // `instantiate_type_scheme` can normalize associated types that
        // may reference those regions.
        let method_sig = self.replace_bound_vars_with_fresh_vars(&sig);
        debug!("late-bound lifetimes from method instantiated, method_sig={:?}",
               method_sig);

        let method_sig = method_sig.subst(self.tcx, all_substs);
        debug!("type scheme substituted, method_sig={:?}", method_sig);

        (method_sig, method_predicates)
    }

    fn add_obligations(&mut self,
                       fty: Ty<'tcx>,
                       all_substs: SubstsRef<'tcx>,
                       method_predicates: &ty::InstantiatedPredicates<'tcx>) {
        debug!("add_obligations: fty={:?} all_substs={:?} method_predicates={:?}",
               fty,
               all_substs,
               method_predicates);

        self.add_obligations_for_parameters(traits::ObligationCause::misc(self.span, self.body_id),
                                            method_predicates);

        // this is a projection from a trait reference, so we have to
        // make sure that the trait reference inputs are well-formed.
        self.add_wf_bounds(all_substs, self.call_expr);

        // the function type must also be well-formed (this is not
        // implied by the substs being well-formed because of inherent
        // impls and late-bound regions - see issue #28609).
        self.register_wf_obligation(fty, self.span, traits::MiscObligation);
    }

    ///////////////////////////////////////////////////////////////////////////
    // RECONCILIATION

    /// When we select a method with a mutable autoref, we have to go convert any
    /// auto-derefs, indices, etc from `Deref` and `Index` into `DerefMut` and `IndexMut`
    /// respectively.
    fn convert_place_derefs_to_mutable(&self) {
        // Gather up expressions we want to munge.
        let mut exprs = vec![self.self_expr];

        loop {
            match exprs.last().unwrap().node {
                hir::ExprKind::Field(ref expr, _) |
                hir::ExprKind::Index(ref expr, _) |
                hir::ExprKind::Unary(hir::UnDeref, ref expr) => exprs.push(&expr),
                _ => break,
            }
        }

        debug!("convert_place_derefs_to_mutable: exprs={:?}", exprs);

        // Fix up autoderefs and derefs.
        for (i, &expr) in exprs.iter().rev().enumerate() {
            debug!("convert_place_derefs_to_mutable: i={} expr={:?}", i, expr);

            // Fix up the autoderefs. Autorefs can only occur immediately preceding
            // overloaded place ops, and will be fixed by them in order to get
            // the correct region.
            let mut source = self.node_ty(expr.hir_id);
            // Do not mutate adjustments in place, but rather take them,
            // and replace them after mutating them, to avoid having the
            // tables borrowed during (`deref_mut`) method resolution.
            let previous_adjustments = self.tables
                                           .borrow_mut()
                                           .adjustments_mut()
                                           .remove(expr.hir_id);
            if let Some(mut adjustments) = previous_adjustments {
                let needs = Needs::MutPlace;
                for adjustment in &mut adjustments {
                    if let Adjust::Deref(Some(ref mut deref)) = adjustment.kind {
                        if let Some(ok) = self.try_overloaded_deref(expr.span, source, needs) {
                            let method = self.register_infer_ok_obligations(ok);
                            if let ty::Ref(region, _, mutbl) = method.sig.output().sty {
                                *deref = OverloadedDeref {
                                    region,
                                    mutbl,
                                };
                            }
                        }
                    }
                    source = adjustment.target;
                }
                self.tables.borrow_mut().adjustments_mut().insert(expr.hir_id, adjustments);
            }

            match expr.node {
                hir::ExprKind::Index(ref base_expr, ref index_expr) => {
                    let index_expr_ty = self.node_ty(index_expr.hir_id);
                    self.convert_place_op_to_mutable(
                        PlaceOp::Index, expr, base_expr, &[index_expr_ty]);
                }
                hir::ExprKind::Unary(hir::UnDeref, ref base_expr) => {
                    self.convert_place_op_to_mutable(
                        PlaceOp::Deref, expr, base_expr, &[]);
                }
                _ => {}
            }
        }
    }

    fn convert_place_op_to_mutable(&self,
                                    op: PlaceOp,
                                    expr: &hir::Expr,
                                    base_expr: &hir::Expr,
                                    arg_tys: &[Ty<'tcx>])
    {
        debug!("convert_place_op_to_mutable({:?}, {:?}, {:?}, {:?})",
               op, expr, base_expr, arg_tys);
        if !self.tables.borrow().is_method_call(expr) {
            debug!("convert_place_op_to_mutable - builtin, nothing to do");
            return
        }

        let base_ty = self.tables.borrow().expr_adjustments(base_expr).last()
            .map_or_else(|| self.node_ty(expr.hir_id), |adj| adj.target);
        let base_ty = self.resolve_vars_if_possible(&base_ty);

        // Need to deref because overloaded place ops take self by-reference.
        let base_ty = base_ty.builtin_deref(false)
            .expect("place op takes something that is not a ref")
            .ty;

        let method = self.try_overloaded_place_op(
            expr.span, base_ty, arg_tys, Needs::MutPlace, op);
        let method = match method {
            Some(ok) => self.register_infer_ok_obligations(ok),
            None => return self.tcx.sess.delay_span_bug(expr.span, "re-trying op failed")
        };
        debug!("convert_place_op_to_mutable: method={:?}", method);
        self.write_method_call(expr.hir_id, method);

        let (region, mutbl) = if let ty::Ref(r, _, mutbl) = method.sig.inputs()[0].sty {
            (r, mutbl)
        } else {
            span_bug!(expr.span, "input to place op is not a ref?");
        };

        // Convert the autoref in the base expr to mutable with the correct
        // region and mutability.
        let base_expr_ty = self.node_ty(base_expr.hir_id);
        if let Some(adjustments) = self.tables
                                       .borrow_mut()
                                       .adjustments_mut()
                                       .get_mut(base_expr.hir_id) {
            let mut source = base_expr_ty;
            for adjustment in &mut adjustments[..] {
                if let Adjust::Borrow(AutoBorrow::Ref(..)) = adjustment.kind {
                    debug!("convert_place_op_to_mutable: converting autoref {:?}", adjustment);
                    let mutbl = match mutbl {
                        hir::MutImmutable => AutoBorrowMutability::Immutable,
                        hir::MutMutable => AutoBorrowMutability::Mutable {
                            // For initial two-phase borrow
                            // deployment, conservatively omit
                            // overloaded operators.
                            allow_two_phase_borrow: AllowTwoPhase::No,
                        }
                    };
                    adjustment.kind = Adjust::Borrow(AutoBorrow::Ref(region, mutbl));
                    adjustment.target = self.tcx.mk_ref(region, ty::TypeAndMut {
                        ty: source,
                        mutbl: mutbl.into(),
                    });
                }
                source = adjustment.target;
            }

            // If we have an autoref followed by unsizing at the end, fix the unsize target.
            match adjustments[..] {
                [.., Adjustment { kind: Adjust::Borrow(AutoBorrow::Ref(..)), .. },
                 Adjustment { kind: Adjust::Pointer(PointerCast::Unsize), ref mut target }] => {
                    *target = method.sig.inputs()[0];
                }
                _ => {}
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // MISCELLANY

    fn predicates_require_illegal_sized_bound(&self,
                                              predicates: &ty::InstantiatedPredicates<'tcx>)
                                              -> bool {
        let sized_def_id = match self.tcx.lang_items().sized_trait() {
            Some(def_id) => def_id,
            None => return false,
        };

        traits::elaborate_predicates(self.tcx, predicates.predicates.clone())
            .filter_map(|predicate| {
                match predicate {
                    ty::Predicate::Trait(trait_pred) if trait_pred.def_id() == sized_def_id =>
                        Some(trait_pred),
                    _ => None,
                }
            })
            .any(|trait_pred| {
                match trait_pred.skip_binder().self_ty().sty {
                    ty::Dynamic(..) => true,
                    _ => false,
                }
            })
    }

    fn enforce_illegal_method_limitations(&self, pick: &probe::Pick<'_>) {
        // Disallow calls to the method `drop` defined in the `Drop` trait.
        match pick.item.container {
            ty::TraitContainer(trait_def_id) => {
                callee::check_legal_trait_for_method_call(self.tcx, self.span, trait_def_id)
            }
            ty::ImplContainer(..) => {}
        }
    }

    fn upcast(&mut self,
              source_trait_ref: ty::PolyTraitRef<'tcx>,
              target_trait_def_id: DefId)
              -> ty::PolyTraitRef<'tcx> {
        let upcast_trait_refs = self.tcx
            .upcast_choices(source_trait_ref.clone(), target_trait_def_id);

        // must be exactly one trait ref or we'd get an ambig error etc
        if upcast_trait_refs.len() != 1 {
            span_bug!(self.span,
                      "cannot uniquely upcast `{:?}` to `{:?}`: `{:?}`",
                      source_trait_ref,
                      target_trait_def_id,
                      upcast_trait_refs);
        }

        upcast_trait_refs.into_iter().next().unwrap()
    }

    fn replace_bound_vars_with_fresh_vars<T>(&self, value: &ty::Binder<T>) -> T
        where T: TypeFoldable<'tcx>
    {
        self.fcx.replace_bound_vars_with_fresh_vars(self.span, infer::FnCall, value).0
    }
}
