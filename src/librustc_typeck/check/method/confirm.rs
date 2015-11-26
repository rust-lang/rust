// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::probe;

use check::{self, FnCtxt, callee, demand};
use check::UnresolvedTypeAction;
use middle::def_id::DefId;
use middle::subst::{self};
use middle::traits;
use middle::ty::{self, NoPreference, PreferMutLvalue, Ty};
use middle::ty::adjustment::{AdjustDerefRef, AutoDerefRef, AutoPtr};
use middle::ty::fold::TypeFoldable;
use middle::infer;
use middle::infer::{InferCtxt, TypeOrigin};
use syntax::codemap::Span;
use rustc_front::hir;

struct ConfirmContext<'a, 'tcx:'a> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    span: Span,
    self_expr: &'tcx hir::Expr,
    call_expr: &'tcx hir::Expr,
}

struct InstantiatedMethodSig<'tcx> {
    /// Function signature of the method being invoked. The 0th
    /// argument is the receiver.
    method_sig: ty::FnSig<'tcx>,

    /// Substitutions for all types/early-bound-regions declared on
    /// the method.
    all_substs: subst::Substs<'tcx>,

    /// Generic bounds on the method's parameters which must be added
    /// as pending obligations.
    method_predicates: ty::InstantiatedPredicates<'tcx>,
}

pub fn confirm<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                         span: Span,
                         self_expr: &'tcx hir::Expr,
                         call_expr: &'tcx hir::Expr,
                         unadjusted_self_ty: Ty<'tcx>,
                         pick: probe::Pick<'tcx>,
                         supplied_method_types: Vec<Ty<'tcx>>)
                         -> ty::MethodCallee<'tcx>
{
    debug!("confirm(unadjusted_self_ty={:?}, pick={:?}, supplied_method_types={:?})",
           unadjusted_self_ty,
           pick,
           supplied_method_types);

    let mut confirm_cx = ConfirmContext::new(fcx, span, self_expr, call_expr);
    confirm_cx.confirm(unadjusted_self_ty, pick, supplied_method_types)
}

impl<'a,'tcx> ConfirmContext<'a,'tcx> {
    fn new(fcx: &'a FnCtxt<'a, 'tcx>,
           span: Span,
           self_expr: &'tcx hir::Expr,
           call_expr: &'tcx hir::Expr)
           -> ConfirmContext<'a, 'tcx>
    {
        ConfirmContext { fcx: fcx, span: span, self_expr: self_expr, call_expr: call_expr }
    }

    fn confirm(&mut self,
               unadjusted_self_ty: Ty<'tcx>,
               pick: probe::Pick<'tcx>,
               supplied_method_types: Vec<Ty<'tcx>>)
               -> ty::MethodCallee<'tcx>
    {
        // Adjust the self expression the user provided and obtain the adjusted type.
        let self_ty = self.adjust_self_ty(unadjusted_self_ty, &pick);

        // Make sure nobody calls `drop()` explicitly.
        self.enforce_illegal_method_limitations(&pick);

        // Create substitutions for the method's type parameters.
        let rcvr_substs = self.fresh_receiver_substs(self_ty, &pick);
        let all_substs =
            self.instantiate_method_substs(
                &pick,
                supplied_method_types,
                rcvr_substs);

        debug!("all_substs={:?}", all_substs);

        // Create the final signature for the method, replacing late-bound regions.
        let InstantiatedMethodSig {
            method_sig, all_substs, method_predicates
        } = self.instantiate_method_sig(&pick, all_substs);
        let method_self_ty = method_sig.inputs[0];

        // Unify the (adjusted) self type with what the method expects.
        self.unify_receivers(self_ty, method_self_ty);

        // Create the method type
        let method_ty = pick.item.as_opt_method().unwrap();
        let fty = self.tcx().mk_fn(None, self.tcx().mk_bare_fn(ty::BareFnTy {
            sig: ty::Binder(method_sig),
            unsafety: method_ty.fty.unsafety,
            abi: method_ty.fty.abi.clone(),
        }));

        // Add any trait/regions obligations specified on the method's type parameters.
        self.add_obligations(fty, &all_substs, &method_predicates);

        // Create the final `MethodCallee`.
        let callee = ty::MethodCallee {
            def_id: pick.item.def_id(),
            ty: fty,
            substs: self.tcx().mk_substs(all_substs)
        };
        // If this is an `&mut self` method, bias the receiver
        // expression towards mutability (this will switch
        // e.g. `Deref` to `DerefMut` in overloaded derefs and so on).
        self.fixup_derefs_on_method_receiver_if_necessary(&callee);

        callee
    }

    ///////////////////////////////////////////////////////////////////////////
    // ADJUSTMENTS

    fn adjust_self_ty(&mut self,
                      unadjusted_self_ty: Ty<'tcx>,
                      pick: &probe::Pick<'tcx>)
                      -> Ty<'tcx>
    {
        let (autoref, unsize) = if let Some(mutbl) = pick.autoref {
            let region = self.infcx().next_region_var(infer::Autoref(self.span));
            let autoref = AutoPtr(self.tcx().mk_region(region), mutbl);
            (Some(autoref), pick.unsize.map(|target| {
                target.adjust_for_autoref(self.tcx(), Some(autoref))
            }))
        } else {
            // No unsizing should be performed without autoref (at
            // least during method dispach). This is because we
            // currently only unsize `[T;N]` to `[T]`, and naturally
            // that must occur being a reference.
            assert!(pick.unsize.is_none());
            (None, None)
        };

        // Commit the autoderefs by calling `autoderef again, but this
        // time writing the results into the various tables.
        let (autoderefd_ty, n, result) = check::autoderef(self.fcx,
                                                          self.span,
                                                          unadjusted_self_ty,
                                                          Some(self.self_expr),
                                                          UnresolvedTypeAction::Error,
                                                          NoPreference,
                                                          |_, n| {
            if n == pick.autoderefs {
                Some(())
            } else {
                None
            }
        });
        assert_eq!(n, pick.autoderefs);
        assert_eq!(result, Some(()));

        // Write out the final adjustment.
        self.fcx.write_adjustment(self.self_expr.id,
                                  AdjustDerefRef(AutoDerefRef {
            autoderefs: pick.autoderefs,
            autoref: autoref,
            unsize: unsize
        }));

        if let Some(target) = unsize {
            target
        } else {
            autoderefd_ty.adjust_for_autoref(self.tcx(), autoref)
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    //

    /// Returns a set of substitutions for the method *receiver* where all type and region
    /// parameters are instantiated with fresh variables. This substitution does not include any
    /// parameters declared on the method itself.
    ///
    /// Note that this substitution may include late-bound regions from the impl level. If so,
    /// these are instantiated later in the `instantiate_method_sig` routine.
    fn fresh_receiver_substs(&mut self,
                             self_ty: Ty<'tcx>,
                             pick: &probe::Pick<'tcx>)
                             -> subst::Substs<'tcx>
    {
        match pick.kind {
            probe::InherentImplPick => {
                let impl_def_id = pick.item.container().id();
                assert!(self.tcx().impl_trait_ref(impl_def_id).is_none(),
                        "impl {:?} is not an inherent impl", impl_def_id);
                check::impl_self_ty(self.fcx, self.span, impl_def_id).substs
            }

            probe::ObjectPick => {
                let trait_def_id = pick.item.container().id();
                self.extract_trait_ref(self_ty, |this, object_ty, data| {
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
                    let original_poly_trait_ref =
                        data.principal_trait_ref_with_self_ty(this.tcx(), object_ty);
                    let upcast_poly_trait_ref =
                        this.upcast(original_poly_trait_ref.clone(), trait_def_id);
                    let upcast_trait_ref =
                        this.replace_late_bound_regions_with_fresh_var(&upcast_poly_trait_ref);
                    debug!("original_poly_trait_ref={:?} upcast_trait_ref={:?} target_trait={:?}",
                           original_poly_trait_ref,
                           upcast_trait_ref,
                           trait_def_id);
                    upcast_trait_ref.substs.clone()
                })
            }

            probe::ExtensionImplPick(impl_def_id) => {
                // The method being invoked is the method as defined on the trait,
                // so return the substitutions from the trait. Consider:
                //
                //     impl<A,B,C> Trait<A,B> for Foo<C> { ... }
                //
                // If we instantiate A, B, and C with $A, $B, and $C
                // respectively, then we want to return the type
                // parameters from the trait ([$A,$B]), not those from
                // the impl ([$A,$B,$C]) not the receiver type ([$C]).
                let impl_polytype = check::impl_self_ty(self.fcx, self.span, impl_def_id);
                let impl_trait_ref =
                    self.fcx.instantiate_type_scheme(
                        self.span,
                        &impl_polytype.substs,
                        &self.tcx().impl_trait_ref(impl_def_id).unwrap());
                impl_trait_ref.substs.clone()
            }

            probe::TraitPick => {
                let trait_def_id = pick.item.container().id();
                let trait_def = self.tcx().lookup_trait_def(trait_def_id);

                // Make a trait reference `$0 : Trait<$1...$n>`
                // consisting entirely of type variables. Later on in
                // the process we will unify the transformed-self-type
                // of the method with the actual type in order to
                // unify some of these variables.
                self.infcx().fresh_substs_for_trait(self.span,
                                                    &trait_def.generics,
                                                    self.infcx().next_ty_var())
            }

            probe::WhereClausePick(ref poly_trait_ref) => {
                // Where clauses can have bound regions in them. We need to instantiate
                // those to convert from a poly-trait-ref to a trait-ref.
                self.replace_late_bound_regions_with_fresh_var(&*poly_trait_ref).substs.clone()
            }
        }
    }

    fn extract_trait_ref<R, F>(&mut self, self_ty: Ty<'tcx>, mut closure: F) -> R where
        F: FnMut(&mut ConfirmContext<'a, 'tcx>, Ty<'tcx>, &ty::TraitTy<'tcx>) -> R,
    {
        // If we specified that this is an object method, then the
        // self-type ought to be something that can be dereferenced to
        // yield an object-type (e.g., `&Object` or `Box<Object>`
        // etc).

        let (_, _, result) = check::autoderef(self.fcx,
                                              self.span,
                                              self_ty,
                                              None,
                                              UnresolvedTypeAction::Error,
                                              NoPreference,
                                              |ty, _| {
            match ty.sty {
                ty::TyTrait(ref data) => Some(closure(self, ty, &**data)),
                _ => None,
            }
        });

        match result {
            Some(r) => r,
            None => {
                self.tcx().sess.span_bug(
                    self.span,
                    &format!("self-type `{}` for ObjectPick never dereferenced to an object",
                            self_ty))
            }
        }
    }

    fn instantiate_method_substs(&mut self,
                                 pick: &probe::Pick<'tcx>,
                                 supplied_method_types: Vec<Ty<'tcx>>,
                                 substs: subst::Substs<'tcx>)
                                 -> subst::Substs<'tcx>
    {
        // Determine the values for the generic parameters of the method.
        // If they were not explicitly supplied, just construct fresh
        // variables.
        let num_supplied_types = supplied_method_types.len();
        let method = pick.item.as_opt_method().unwrap();
        let method_types = method.generics.types.get_slice(subst::FnSpace);
        let num_method_types = method_types.len();


        // Create subst for early-bound lifetime parameters, combining
        // parameters from the type and those from the method.
        //
        // FIXME -- permit users to manually specify lifetimes
        let method_regions =
            self.fcx.infcx().region_vars_for_defs(
                self.span,
                pick.item.as_opt_method().unwrap()
                    .generics.regions.get_slice(subst::FnSpace));

        let subst::Substs { types, regions } = substs;
        let regions = regions.map(|r| r.with_vec(subst::FnSpace, method_regions));
        let mut final_substs = subst::Substs { types: types, regions: regions };

        if num_supplied_types == 0 {
            self.fcx.infcx().type_vars_for_defs(
                self.span,
                subst::FnSpace,
                &mut final_substs,
                method_types);
        } else if num_method_types == 0 {
            span_err!(self.tcx().sess, self.span, E0035,
                "does not take type parameters");
            self.fcx.infcx().type_vars_for_defs(
                self.span,
                subst::FnSpace,
                &mut final_substs,
                method_types);
        } else if num_supplied_types != num_method_types {
            span_err!(self.tcx().sess, self.span, E0036,
                "incorrect number of type parameters given for this method: expected {}, found {}",
                num_method_types, num_supplied_types);
            final_substs.types.replace(
                subst::FnSpace,
                vec![self.tcx().types.err; num_method_types]);
        } else {
            final_substs.types.replace(subst::FnSpace, supplied_method_types);
        }

        return final_substs;
    }

    fn unify_receivers(&mut self,
                       self_ty: Ty<'tcx>,
                       method_self_ty: Ty<'tcx>)
    {
        match self.fcx.mk_subty(false, TypeOrigin::Misc(self.span), self_ty, method_self_ty) {
            Ok(_) => {}
            Err(_) => {
                self.tcx().sess.span_bug(
                    self.span,
                    &format!("{} was a subtype of {} but now is not?",
                             self_ty, method_self_ty));
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    //

    fn instantiate_method_sig(&mut self,
                              pick: &probe::Pick<'tcx>,
                              all_substs: subst::Substs<'tcx>)
                              -> InstantiatedMethodSig<'tcx>
    {
        debug!("instantiate_method_sig(pick={:?}, all_substs={:?})",
               pick,
               all_substs);

        // Instantiate the bounds on the method with the
        // type/early-bound-regions substitutions performed. There can
        // be no late-bound regions appearing here.
        let method_predicates = pick.item.as_opt_method().unwrap()
                                    .predicates.instantiate(self.tcx(), &all_substs);
        let method_predicates = self.fcx.normalize_associated_types_in(self.span,
                                                                       &method_predicates);

        debug!("method_predicates after subst = {:?}",
               method_predicates);

        // Instantiate late-bound regions and substitute the trait
        // parameters into the method type to get the actual method type.
        //
        // NB: Instantiate late-bound regions first so that
        // `instantiate_type_scheme` can normalize associated types that
        // may reference those regions.
        let method_sig = self.replace_late_bound_regions_with_fresh_var(
            &pick.item.as_opt_method().unwrap().fty.sig);
        debug!("late-bound lifetimes from method instantiated, method_sig={:?}",
               method_sig);

        let method_sig = self.fcx.instantiate_type_scheme(self.span, &all_substs, &method_sig);
        debug!("type scheme substituted, method_sig={:?}",
               method_sig);

        InstantiatedMethodSig {
            method_sig: method_sig,
            all_substs: all_substs,
            method_predicates: method_predicates,
        }
    }

    fn add_obligations(&mut self,
                       fty: Ty<'tcx>,
                       all_substs: &subst::Substs<'tcx>,
                       method_predicates: &ty::InstantiatedPredicates<'tcx>) {
        debug!("add_obligations: fty={:?} all_substs={:?} method_predicates={:?}",
               fty,
               all_substs,
               method_predicates);

        self.fcx.add_obligations_for_parameters(
            traits::ObligationCause::misc(self.span, self.fcx.body_id),
            method_predicates);

        // this is a projection from a trait reference, so we have to
        // make sure that the trait reference inputs are well-formed.
        self.fcx.add_wf_bounds(
            all_substs,
            self.call_expr);

        // the function type must also be well-formed (this is not
        // implied by the substs being well-formed because of inherent
        // impls and late-bound regions - see issue #28609).
        self.fcx.register_wf_obligation(fty, self.span, traits::MiscObligation);
    }

    ///////////////////////////////////////////////////////////////////////////
    // RECONCILIATION

    /// When we select a method with an `&mut self` receiver, we have to go convert any
    /// auto-derefs, indices, etc from `Deref` and `Index` into `DerefMut` and `IndexMut`
    /// respectively.
    fn fixup_derefs_on_method_receiver_if_necessary(&self,
                                                    method_callee: &ty::MethodCallee) {
        let sig = match method_callee.ty.sty {
            ty::TyBareFn(_, ref f) => f.sig.clone(),
            _ => return,
        };

        match sig.0.inputs[0].sty {
            ty::TyRef(_, ty::TypeAndMut {
                ty: _,
                mutbl: hir::MutMutable,
            }) => {}
            _ => return,
        }

        // Gather up expressions we want to munge.
        let mut exprs = Vec::new();
        exprs.push(self.self_expr);
        loop {
            let last = exprs[exprs.len() - 1];
            match last.node {
                hir::ExprField(ref expr, _) |
                hir::ExprTupField(ref expr, _) |
                hir::ExprIndex(ref expr, _) |
                hir::ExprUnary(hir::UnDeref, ref expr) => exprs.push(&**expr),
                _ => break,
            }
        }

        debug!("fixup_derefs_on_method_receiver_if_necessary: exprs={:?}",
               exprs);

        // Fix up autoderefs and derefs.
        for (i, &expr) in exprs.iter().rev().enumerate() {
            // Count autoderefs.
            let autoderef_count = match self.fcx
                                            .inh
                                            .tables
                                            .borrow()
                                            .adjustments
                                            .get(&expr.id) {
                Some(&AdjustDerefRef(ref adj)) => adj.autoderefs,
                Some(_) | None => 0,
            };

            debug!("fixup_derefs_on_method_receiver_if_necessary: i={} expr={:?} \
                                                                  autoderef_count={}",
                   i, expr, autoderef_count);

            if autoderef_count > 0 {
                check::autoderef(self.fcx,
                                 expr.span,
                                 self.fcx.expr_ty(expr),
                                 Some(expr),
                                 UnresolvedTypeAction::Error,
                                 PreferMutLvalue,
                                 |_, autoderefs| {
                                     if autoderefs == autoderef_count + 1 {
                                         Some(())
                                     } else {
                                         None
                                     }
                                 });
            }

            // Don't retry the first one or we might infinite loop!
            if i != 0 {
                match expr.node {
                    hir::ExprIndex(ref base_expr, ref index_expr) => {
                        // If this is an overloaded index, the
                        // adjustment will include an extra layer of
                        // autoref because the method is an &self/&mut
                        // self method. We have to peel it off to get
                        // the raw adjustment that `try_index_step`
                        // expects. This is annoying and horrible. We
                        // ought to recode this routine so it doesn't
                        // (ab)use the normal type checking paths.
                        let adj = self.fcx.inh.tables.borrow().adjustments.get(&base_expr.id)
                                                                          .cloned();
                        let (autoderefs, unsize) = match adj {
                            Some(AdjustDerefRef(adr)) => match adr.autoref {
                                None => {
                                    assert!(adr.unsize.is_none());
                                    (adr.autoderefs, None)
                                }
                                Some(AutoPtr(_, _)) => {
                                    (adr.autoderefs, adr.unsize.map(|target| {
                                        target.builtin_deref(false, NoPreference)
                                              .expect("fixup: AutoPtr is not &T").ty
                                    }))
                                }
                                Some(_) => {
                                    self.tcx().sess.span_bug(
                                        base_expr.span,
                                        &format!("unexpected adjustment autoref {:?}",
                                                adr));
                                }
                            },
                            None => (0, None),
                            Some(_) => {
                                self.tcx().sess.span_bug(
                                    base_expr.span,
                                    "unexpected adjustment type");
                            }
                        };

                        let (adjusted_base_ty, unsize) = if let Some(target) = unsize {
                            (target, true)
                        } else {
                            (self.fcx.adjust_expr_ty(base_expr,
                                Some(&AdjustDerefRef(AutoDerefRef {
                                    autoderefs: autoderefs,
                                    autoref: None,
                                    unsize: None
                                }))), false)
                        };
                        let index_expr_ty = self.fcx.expr_ty(&**index_expr);

                        let result = check::try_index_step(
                            self.fcx,
                            ty::MethodCall::expr(expr.id),
                            expr,
                            &**base_expr,
                            adjusted_base_ty,
                            autoderefs,
                            unsize,
                            PreferMutLvalue,
                            index_expr_ty);

                        if let Some((input_ty, return_ty)) = result {
                            demand::suptype(self.fcx, index_expr.span, input_ty, index_expr_ty);

                            let expr_ty = self.fcx.expr_ty(&*expr);
                            demand::suptype(self.fcx, expr.span, expr_ty, return_ty);
                        }
                    }
                    hir::ExprUnary(hir::UnDeref, ref base_expr) => {
                        // if this is an overloaded deref, then re-evaluate with
                        // a preference for mut
                        let method_call = ty::MethodCall::expr(expr.id);
                        if self.fcx.inh.tables.borrow().method_map.contains_key(&method_call) {
                            check::try_overloaded_deref(
                                self.fcx,
                                expr.span,
                                Some(method_call),
                                Some(&**base_expr),
                                self.fcx.expr_ty(&**base_expr),
                                PreferMutLvalue);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // MISCELLANY

    fn tcx(&self) -> &'a ty::ctxt<'tcx> {
        self.fcx.tcx()
    }

    fn infcx(&self) -> &'a InferCtxt<'a, 'tcx> {
        self.fcx.infcx()
    }

    fn enforce_illegal_method_limitations(&self, pick: &probe::Pick) {
        // Disallow calls to the method `drop` defined in the `Drop` trait.
        match pick.item.container() {
            ty::TraitContainer(trait_def_id) => {
                callee::check_legal_trait_for_method_call(self.fcx.ccx, self.span, trait_def_id)
            }
            ty::ImplContainer(..) => {}
        }
    }

    fn upcast(&mut self,
              source_trait_ref: ty::PolyTraitRef<'tcx>,
              target_trait_def_id: DefId)
              -> ty::PolyTraitRef<'tcx>
    {
        let upcast_trait_refs = traits::upcast(self.tcx(),
                                               source_trait_ref.clone(),
                                               target_trait_def_id);

        // must be exactly one trait ref or we'd get an ambig error etc
        if upcast_trait_refs.len() != 1 {
            self.tcx().sess.span_bug(
                self.span,
                &format!("cannot uniquely upcast `{:?}` to `{:?}`: `{:?}`",
                         source_trait_ref,
                         target_trait_def_id,
                         upcast_trait_refs));
        }

        upcast_trait_refs.into_iter().next().unwrap()
    }

    fn replace_late_bound_regions_with_fresh_var<T>(&self, value: &ty::Binder<T>) -> T
        where T : TypeFoldable<'tcx>
    {
        self.infcx().replace_late_bound_regions_with_fresh_var(
            self.span, infer::FnCall, value).0
    }
}
