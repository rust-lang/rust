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

use middle::subst::{mod, Subst};
use middle::traits;
use middle::ty::{mod, Ty};
use middle::ty::{MethodCall, MethodCallee, MethodObject, MethodOrigin,
                 MethodParam, MethodStatic, MethodTraitObject, MethodTypeParam};
use middle::typeck::check::{mod, FnCtxt, NoPreference, PreferMutLvalue};
use middle::infer;
use middle::infer::InferCtxt;
use middle::ty_fold::HigherRankedFoldable;
use syntax::ast;
use syntax::codemap::Span;
use std::rc::Rc;
use std::mem;
use util::ppaux::Repr;

struct ConfirmContext<'a, 'tcx:'a> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    span: Span,
    self_expr: &'a ast::Expr,
    call_expr: &'a ast::Expr,
}

struct InstantiatedMethodSig<'tcx> {
    /// Function signature of the method being invoked. The 0th
    /// argument is the receiver.
    method_sig: ty::FnSig<'tcx>,

    /// Substitutions for all types/early-bound-regions declared on
    /// the method.
    all_substs: subst::Substs<'tcx>,

    /// Substitution to use when adding obligations from the method
    /// bounds. Normally equal to `all_substs` except for object
    /// receivers. See FIXME in instantiate_method_sig() for
    /// explanation.
    method_bounds_substs: subst::Substs<'tcx>,

    /// Generic bounds on the method's parameters which must be added
    /// as pending obligations.
    method_bounds: ty::GenericBounds<'tcx>,
}

pub fn confirm<'a, 'tcx>(fcx: &FnCtxt<'a, 'tcx>,
                         span: Span,
                         self_expr: &ast::Expr,
                         call_expr: &ast::Expr,
                         unadjusted_self_ty: Ty<'tcx>,
                         pick: probe::Pick<'tcx>,
                         supplied_method_types: Vec<Ty<'tcx>>)
                         -> MethodCallee<'tcx>
{
    debug!("confirm(unadjusted_self_ty={}, pick={}, supplied_method_types={})",
           unadjusted_self_ty.repr(fcx.tcx()),
           pick.repr(fcx.tcx()),
           supplied_method_types.repr(fcx.tcx()));

    let mut confirm_cx = ConfirmContext::new(fcx, span, self_expr, call_expr);
    confirm_cx.confirm(unadjusted_self_ty, pick, supplied_method_types)
}

impl<'a,'tcx> ConfirmContext<'a,'tcx> {
    fn new(fcx: &'a FnCtxt<'a, 'tcx>,
           span: Span,
           self_expr: &'a ast::Expr,
           call_expr: &'a ast::Expr)
           -> ConfirmContext<'a, 'tcx>
    {
        ConfirmContext { fcx: fcx, span: span, self_expr: self_expr, call_expr: call_expr }
    }

    fn confirm(&mut self,
               unadjusted_self_ty: Ty<'tcx>,
               pick: probe::Pick<'tcx>,
               supplied_method_types: Vec<Ty<'tcx>>)
               -> MethodCallee<'tcx>
    {
        // Adjust the self expression the user provided and obtain the adjusted type.
        let self_ty = self.adjust_self_ty(unadjusted_self_ty, &pick.adjustment);

        // Make sure nobody calls `drop()` explicitly.
        self.enforce_drop_trait_limitations(&pick);

        // Create substitutions for the method's type parameters.
        let (rcvr_substs, method_origin) =
            self.fresh_receiver_substs(self_ty, &pick);
        let (method_types, method_regions) =
            self.instantiate_method_substs(&pick, supplied_method_types);
        let all_substs = rcvr_substs.with_method(method_types, method_regions);
        debug!("all_substs={}", all_substs.repr(self.tcx()));

        // Create the final signature for the method, replacing late-bound regions.
        let InstantiatedMethodSig {
            method_sig, all_substs, method_bounds_substs, method_bounds
        } = self.instantiate_method_sig(&pick, all_substs);
        let method_self_ty = method_sig.inputs[0];

        // Unify the (adjusted) self type with what the method expects.
        self.unify_receivers(self_ty, method_self_ty);

        // Add any trait/regions obligations specified on the method's type parameters.
        self.add_obligations(&pick, &method_bounds_substs, &method_bounds);

        // Create the final `MethodCallee`.
        let fty = ty::mk_bare_fn(self.tcx(), ty::BareFnTy {
            sig: method_sig,
            fn_style: pick.method_ty.fty.fn_style,
            abi: pick.method_ty.fty.abi.clone(),
        });
        let callee = MethodCallee {
            origin: method_origin,
            ty: fty,
            substs: all_substs
        };

        // If this is an `&mut self` method, bias the receiver
        // expression towards mutability (this will switch
        // e.g. `Deref` to `DerefMut` in oveloaded derefs and so on).
        self.fixup_derefs_on_method_receiver_if_necessary(&callee);

        callee
    }

    ///////////////////////////////////////////////////////////////////////////
    // ADJUSTMENTS

    fn adjust_self_ty(&mut self,
                      unadjusted_self_ty: Ty<'tcx>,
                      adjustment: &probe::PickAdjustment)
                      -> Ty<'tcx>
    {
        // Construct the actual adjustment and write it into the table
        let auto_deref_ref = self.create_ty_adjustment(adjustment);

        // Commit the autoderefs by calling `autoderef again, but this
        // time writing the results into the various tables.
        let (autoderefd_ty, n, result) =
            check::autoderef(
                self.fcx, self.span, unadjusted_self_ty, Some(self.self_expr.id), NoPreference,
                |_, n| if n == auto_deref_ref.autoderefs { Some(()) } else { None });
        assert_eq!(n, auto_deref_ref.autoderefs);
        assert_eq!(result, Some(()));

        let final_ty =
            ty::adjust_ty_for_autoref(self.tcx(), self.span, autoderefd_ty,
                                      auto_deref_ref.autoref.as_ref());

        // Write out the final adjustment.
        self.fcx.write_adjustment(self.self_expr.id, self.span, ty::AdjustDerefRef(auto_deref_ref));

        final_ty
    }

    fn create_ty_adjustment(&mut self,
                            adjustment: &probe::PickAdjustment)
                            -> ty::AutoDerefRef<'tcx>
    {
        match *adjustment {
            probe::AutoDeref(num) => {
                ty::AutoDerefRef {
                    autoderefs: num,
                    autoref: None
                }
            }
            probe::AutoUnsizeLength(autoderefs, len) => {
                ty::AutoDerefRef {
                    autoderefs: autoderefs,
                    autoref: Some(ty::AutoUnsize(ty::UnsizeLength(len)))
                }
            }
            probe::AutoRef(mutability, ref sub_adjustment) => {
                let deref = self.create_ty_adjustment(&**sub_adjustment);
                let region = self.infcx().next_region_var(infer::Autoref(self.span));
                wrap_autoref(deref, |base| ty::AutoPtr(region, mutability, base))
            }
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
                             -> (subst::Substs<'tcx>, MethodOrigin<'tcx>)
    {
        match pick.kind {
            probe::InherentImplPick(impl_def_id) => {
                assert!(ty::impl_trait_ref(self.tcx(), impl_def_id).is_none(),
                        "impl {} is not an inherent impl", impl_def_id);
                let impl_polytype = check::impl_self_ty(self.fcx, self.span, impl_def_id);

                (impl_polytype.substs, MethodStatic(pick.method_ty.def_id))
            }

            probe::ObjectPick(trait_def_id, method_num, real_index) => {
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
                    let substs = data.principal.substs.clone().with_self_ty(object_ty);
                    let original_trait_ref =
                        Rc::new(ty::TraitRef::new(data.principal.def_id, substs));
                    let upcast_trait_ref = this.upcast(original_trait_ref.clone(), trait_def_id);
                    debug!("original_trait_ref={} upcast_trait_ref={} target_trait={}",
                           original_trait_ref.repr(this.tcx()),
                           upcast_trait_ref.repr(this.tcx()),
                           trait_def_id.repr(this.tcx()));
                    let substs = upcast_trait_ref.substs.clone();
                    let origin = MethodTraitObject(MethodObject {
                        trait_ref: upcast_trait_ref,
                        object_trait_id: trait_def_id,
                        method_num: method_num,
                        real_index: real_index,
                    });
                    (substs, origin)
                })
            }

            probe::ExtensionImplPick(impl_def_id, method_num) => {
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
                let impl_trait_ref = ty::impl_trait_ref(self.tcx(), impl_def_id)
                                     .unwrap()
                                     .subst(self.tcx(), &impl_polytype.substs);
                let origin = MethodTypeParam(MethodParam { trait_ref: impl_trait_ref.clone(),
                                                           method_num: method_num });
                (impl_trait_ref.substs.clone(), origin)
            }

            probe::TraitPick(trait_def_id, method_num) => {
                let trait_def = ty::lookup_trait_def(self.tcx(), trait_def_id);

                // Make a trait reference `$0 : Trait<$1...$n>`
                // consisting entirely of type variables. Later on in
                // the process we will unify the transformed-self-type
                // of the method with the actual type in order to
                // unify some of these variables.
                let substs = self.infcx().fresh_substs_for_trait(self.span,
                                                                 &trait_def.generics,
                                                                 self.infcx().next_ty_var());

                let trait_ref = Rc::new(ty::TraitRef::new(trait_def_id, substs.clone()));
                let origin = MethodTypeParam(MethodParam { trait_ref: trait_ref,
                                                           method_num: method_num });
                (substs, origin)
            }

            probe::WhereClausePick(ref trait_ref, method_num) => {
                let origin = MethodTypeParam(MethodParam { trait_ref: (*trait_ref).clone(),
                                                           method_num: method_num });
                (trait_ref.substs.clone(), origin)
            }
        }
    }

    fn extract_trait_ref<R>(&mut self,
                            self_ty: Ty<'tcx>,
                            closure: |&mut ConfirmContext<'a,'tcx>,
                                      Ty<'tcx>, &ty::TyTrait<'tcx>| -> R)
                            -> R
    {
        // If we specified that this is an object method, then the
        // self-type ought to be something that can be dereferenced to
        // yield an object-type (e.g., `&Object` or `Box<Object>`
        // etc).

        let (_, _, result) =
            check::autoderef(
                self.fcx, self.span, self_ty, None, NoPreference,
                |ty, _| {
                    match ty.sty {
                        ty::ty_trait(ref data) => Some(closure(self, ty, &**data)),
                        _ => None,
                    }
                });

        match result {
            Some(r) => r,
            None => {
                self.tcx().sess.span_bug(
                    self.span,
                    format!("self-type `{}` for ObjectPick never dereferenced to an object",
                            self_ty.repr(self.tcx()))[])
            }
        }
    }

    fn instantiate_method_substs(&mut self,
                                 pick: &probe::Pick<'tcx>,
                                 supplied_method_types: Vec<Ty<'tcx>>)
                                 -> (Vec<Ty<'tcx>>, Vec<ty::Region>)
    {
        // Determine the values for the generic parameters of the method.
        // If they were not explicitly supplied, just construct fresh
        // variables.
        let num_supplied_types = supplied_method_types.len();
        let num_method_types = pick.method_ty.generics.types.len(subst::FnSpace);
        let method_types = {
            if num_supplied_types == 0u {
                self.fcx.infcx().next_ty_vars(num_method_types)
            } else if num_method_types == 0u {
                span_err!(self.tcx().sess, self.span, E0035,
                    "does not take type parameters");
                self.fcx.infcx().next_ty_vars(num_method_types)
            } else if num_supplied_types != num_method_types {
                span_err!(self.tcx().sess, self.span, E0036,
                    "incorrect number of type parameters given for this method");
                Vec::from_elem(num_method_types, ty::mk_err())
            } else {
                supplied_method_types
            }
        };

        // Create subst for early-bound lifetime parameters, combining
        // parameters from the type and those from the method.
        //
        // FIXME -- permit users to manually specify lifetimes
        let method_regions =
            self.fcx.infcx().region_vars_for_defs(
                self.span,
                pick.method_ty.generics.regions.get_slice(subst::FnSpace));

        (method_types, method_regions)
    }

    fn unify_receivers(&mut self,
                       self_ty: Ty<'tcx>,
                       method_self_ty: Ty<'tcx>)
    {
        match self.fcx.mk_subty(false, infer::Misc(self.span), self_ty, method_self_ty) {
            Ok(_) => {}
            Err(_) => {
                self.tcx().sess.span_bug(
                    self.span,
                    format!(
                        "{} was a subtype of {} but now is not?",
                        self_ty.repr(self.tcx()),
                        method_self_ty.repr(self.tcx()))[]);
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
        // If this method comes from an impl (as opposed to a trait),
        // it may have late-bound regions from the impl that appear in
        // the substitutions, method signature, and
        // bounds. Instantiate those at this point. (If it comes from
        // a trait, this step has no effect, as there are no
        // late-bound regions to instantiate.)
        //
        // The binder level here corresponds to the impl.
        let (all_substs, (method_sig, method_generics)) =
            self.replace_late_bound_regions_with_fresh_var(
                &ty::bind((all_substs,
                           (pick.method_ty.fty.sig.clone(),
                            pick.method_ty.generics.clone())))).value;

        debug!("late-bound lifetimes from impl instantiated, \
                all_substs={} method_sig={} method_generics={}",
               all_substs.repr(self.tcx()),
               method_sig.repr(self.tcx()),
               method_generics.repr(self.tcx()));

        // Instantiate the bounds on the method with the
        // type/early-bound-regions substitutions performed.  The only
        // late-bound-regions that can appear in bounds are from the
        // impl, and those were already instantiated above.
        //
        // FIXME(DST). Super hack. For a method on a trait object
        // `Trait`, the generic signature requires that
        // `Self:Trait`. Since, for an object, we bind `Self` to the
        // type `Trait`, this leads to an obligation
        // `Trait:Trait`. Until such time we DST is fully implemented,
        // that obligation is not necessarily satisfied. (In the
        // future, it would be.)
        //
        // To sidestep this, we overwrite the binding for `Self` with
        // `err` (just for trait objects) when we generate the
        // obligations.  This causes us to generate the obligation
        // `err:Trait`, and the error type is considered to implement
        // all traits, so we're all good. Hack hack hack.
        let method_bounds_substs = match pick.kind {
            probe::ObjectPick(..) => {
                let mut temp_substs = all_substs.clone();
                temp_substs.types.get_mut_slice(subst::SelfSpace)[0] = ty::mk_err();
                temp_substs
            }
            _ => {
                all_substs.clone()
            }
        };
        let method_bounds =
            method_generics.to_bounds(self.tcx(), &method_bounds_substs);

        debug!("method_bounds after subst = {}",
               method_bounds.repr(self.tcx()));

        // Substitute the type/early-bound-regions into the method
        // signature. In addition, the method signature may bind
        // late-bound regions, so instantiate those.
        let method_sig = method_sig.subst(self.tcx(), &all_substs);
        let method_sig = self.replace_late_bound_regions_with_fresh_var(&method_sig);

        debug!("late-bound lifetimes from method instantiated, method_sig={}",
               method_sig.repr(self.tcx()));

        InstantiatedMethodSig {
            method_sig: method_sig,
            all_substs: all_substs,
            method_bounds_substs: method_bounds_substs,
            method_bounds: method_bounds,
        }
    }

    fn add_obligations(&mut self,
                       pick: &probe::Pick<'tcx>,
                       method_bounds_substs: &subst::Substs<'tcx>,
                       method_bounds: &ty::GenericBounds<'tcx>) {
        debug!("add_obligations: pick={} method_bounds_substs={} method_bounds={}",
               pick.repr(self.tcx()),
               method_bounds_substs.repr(self.tcx()),
               method_bounds.repr(self.tcx()));

        self.fcx.add_obligations_for_parameters(
            traits::ObligationCause::misc(self.span),
            method_bounds_substs,
            method_bounds);

        self.fcx.add_default_region_param_bounds(
            method_bounds_substs,
            self.call_expr);
    }

    ///////////////////////////////////////////////////////////////////////////
    // RECONCILIATION

    /// When we select a method with an `&mut self` receiver, we have to go convert any
    /// auto-derefs, indices, etc from `Deref` and `Index` into `DerefMut` and `IndexMut`
    /// respectively.
    fn fixup_derefs_on_method_receiver_if_necessary(&self,
                                                    method_callee: &MethodCallee) {
        let sig = match method_callee.ty.sty {
            ty::ty_bare_fn(ref f) => f.sig.clone(),
            ty::ty_closure(ref f) => f.sig.clone(),
            _ => return,
        };

        match sig.inputs[0].sty {
            ty::ty_rptr(_, ty::mt {
                ty: _,
                mutbl: ast::MutMutable,
            }) => {}
            _ => return,
        }

        // Gather up expressions we want to munge.
        let mut exprs = Vec::new();
        exprs.push(self.self_expr);
        loop {
            let last = exprs[exprs.len() - 1];
            match last.node {
                ast::ExprParen(ref expr) |
                ast::ExprField(ref expr, _) |
                ast::ExprTupField(ref expr, _) |
                ast::ExprSlice(ref expr, _, _, _) |
                ast::ExprIndex(ref expr, _) |
                ast::ExprUnary(ast::UnDeref, ref expr) => exprs.push(&**expr),
                _ => break,
            }
        }

        debug!("fixup_derefs_on_method_receiver_if_necessary: exprs={}",
               exprs.repr(self.tcx()));

        // Fix up autoderefs and derefs.
        for (i, expr) in exprs.iter().rev().enumerate() {
            // Count autoderefs.
            let autoderef_count = match self.fcx
                                            .inh
                                            .adjustments
                                            .borrow()
                                            .get(&expr.id) {
                Some(&ty::AdjustDerefRef(ty::AutoDerefRef {
                    autoderefs: autoderef_count,
                    autoref: _
                })) => autoderef_count,
                Some(_) | None => 0,
            };

            debug!("fixup_derefs_on_method_receiver_if_necessary: i={} expr={} autoderef_count={}",
                   i, expr.repr(self.tcx()), autoderef_count);

            if autoderef_count > 0 {
                check::autoderef(self.fcx,
                                 expr.span,
                                 self.fcx.expr_ty(*expr),
                                 Some(expr.id),
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
                    ast::ExprIndex(ref base_expr, _) => {
                        let mut base_adjustment =
                            match self.fcx.inh.adjustments.borrow().get(&base_expr.id) {
                                Some(&ty::AdjustDerefRef(ref adr)) => (*adr).clone(),
                                None => ty::AutoDerefRef { autoderefs: 0, autoref: None },
                                Some(_) => {
                                    self.tcx().sess.span_bug(
                                        base_expr.span,
                                        "unexpected adjustment type");
                                }
                            };

                        // If this is an overloaded index, the
                        // adjustment will include an extra layer of
                        // autoref because the method is an &self/&mut
                        // self method. We have to peel it off to get
                        // the raw adjustment that `try_index_step`
                        // expects. This is annoying and horrible. We
                        // ought to recode this routine so it doesn't
                        // (ab)use the normal type checking paths.
                        base_adjustment.autoref = match base_adjustment.autoref {
                            None => { None }
                            Some(ty::AutoPtr(_, _, None)) => { None }
                            Some(ty::AutoPtr(_, _, Some(box r))) => { Some(r) }
                            Some(_) => {
                                self.tcx().sess.span_bug(
                                    base_expr.span,
                                    "unexpected adjustment autoref");
                            }
                        };

                        let adjusted_base_ty =
                            self.fcx.adjust_expr_ty(
                                &**base_expr,
                                Some(&ty::AdjustDerefRef(base_adjustment.clone())));

                        check::try_index_step(
                            self.fcx,
                            MethodCall::expr(expr.id),
                            *expr,
                            &**base_expr,
                            adjusted_base_ty,
                            base_adjustment,
                            PreferMutLvalue);
                    }
                    ast::ExprUnary(ast::UnDeref, ref base_expr) => {
                        // if this is an overloaded deref, then re-evaluate with
                        // a preference for mut
                        let method_call = MethodCall::expr(expr.id);
                        if self.fcx.inh.method_map.borrow().contains_key(&method_call) {
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

    fn enforce_drop_trait_limitations(&self, pick: &probe::Pick) {
        // Disallow calls to the method `drop` defined in the `Drop` trait.
        match pick.method_ty.container {
            ty::TraitContainer(trait_def_id) => {
                if Some(trait_def_id) == self.tcx().lang_items.drop_trait() {
                    span_err!(self.tcx().sess, self.span, E0040,
                              "explicit call to destructor");
                }
            }
            ty::ImplContainer(..) => {
                // Since `drop` is a trait method, we expect that any
                // potential calls to it will wind up in the other
                // arm. But just to be sure, check that the method id
                // does not appear in the list of destructors.
                assert!(!self.tcx().destructors.borrow().contains(&pick.method_ty.def_id));
            }
        }
    }

    fn upcast(&mut self,
              source_trait_ref: Rc<ty::TraitRef<'tcx>>,
              target_trait_def_id: ast::DefId)
              -> Rc<ty::TraitRef<'tcx>>
    {
        for super_trait_ref in traits::supertraits(self.tcx(), source_trait_ref.clone()) {
            if super_trait_ref.def_id == target_trait_def_id {
                return super_trait_ref;
            }
        }

        self.tcx().sess.span_bug(
            self.span,
            format!("cannot upcast `{}` to `{}`",
                    source_trait_ref.repr(self.tcx()),
                    target_trait_def_id.repr(self.tcx()))[]);
    }

    fn replace_late_bound_regions_with_fresh_var<T>(&self, value: &T) -> T
        where T : HigherRankedFoldable<'tcx>
    {
        self.infcx().replace_late_bound_regions_with_fresh_var(
            self.span, infer::FnCall, value).0
    }
}

fn wrap_autoref<'tcx>(mut deref: ty::AutoDerefRef<'tcx>,
                      base_fn: |Option<Box<ty::AutoRef<'tcx>>>| -> ty::AutoRef<'tcx>)
                      -> ty::AutoDerefRef<'tcx> {
    let autoref = mem::replace(&mut deref.autoref, None);
    let autoref = autoref.map(|r| box r);
    deref.autoref = Some(base_fn(autoref));
    deref
}
