// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::free_region::FreeRegionMap;
use middle::infer::{self, TypeOrigin};
use middle::traits;
use middle::ty::{self};
use middle::subst::{self, Subst, Substs, VecPerParamSpace};

use syntax::ast;
use syntax::codemap::Span;

use super::assoc;

/// Checks that a method from an impl conforms to the signature of
/// the same method as declared in the trait.
///
/// # Parameters
///
/// - impl_m: type of the method we are checking
/// - impl_m_span: span to use for reporting errors
/// - impl_m_body_id: id of the method body
/// - trait_m: the method in the trait
/// - impl_trait_ref: the TraitRef corresponding to the trait implementation

pub fn compare_impl_method<'tcx>(tcx: &ty::ctxt<'tcx>,
                                 impl_m: &ty::Method<'tcx>,
                                 impl_m_span: Span,
                                 impl_m_body_id: ast::NodeId,
                                 trait_m: &ty::Method<'tcx>,
                                 impl_trait_ref: &ty::TraitRef<'tcx>) {
    debug!("compare_impl_method(impl_trait_ref={:?})",
           impl_trait_ref);

    debug!("compare_impl_method: impl_trait_ref (liberated) = {:?}",
           impl_trait_ref);

    let mut infcx = infer::new_infer_ctxt(tcx, &tcx.tables, None);
    let mut fulfillment_cx = infcx.fulfillment_cx.borrow_mut();

    let trait_to_impl_substs = &impl_trait_ref.substs;

    // Try to give more informative error messages about self typing
    // mismatches.  Note that any mismatch will also be detected
    // below, where we construct a canonical function type that
    // includes the self parameter as a normal parameter.  It's just
    // that the error messages you get out of this code are a bit more
    // inscrutable, particularly for cases where one method has no
    // self.
    match (&trait_m.explicit_self, &impl_m.explicit_self) {
        (&ty::ExplicitSelfCategory::Static,
         &ty::ExplicitSelfCategory::Static) => {}
        (&ty::ExplicitSelfCategory::Static, _) => {
            span_err!(tcx.sess, impl_m_span, E0185,
                "method `{}` has a `{}` declaration in the impl, \
                        but not in the trait",
                        trait_m.name,
                        impl_m.explicit_self);
            return;
        }
        (_, &ty::ExplicitSelfCategory::Static) => {
            span_err!(tcx.sess, impl_m_span, E0186,
                "method `{}` has a `{}` declaration in the trait, \
                        but not in the impl",
                        trait_m.name,
                        trait_m.explicit_self);
            return;
        }
        _ => {
            // Let the type checker catch other errors below
        }
    }

    let num_impl_m_type_params = impl_m.generics.types.len(subst::FnSpace);
    let num_trait_m_type_params = trait_m.generics.types.len(subst::FnSpace);
    if num_impl_m_type_params != num_trait_m_type_params {
        span_err!(tcx.sess, impl_m_span, E0049,
            "method `{}` has {} type parameter{} \
             but its trait declaration has {} type parameter{}",
            trait_m.name,
            num_impl_m_type_params,
            if num_impl_m_type_params == 1 {""} else {"s"},
            num_trait_m_type_params,
            if num_trait_m_type_params == 1 {""} else {"s"});
        return;
    }

    if impl_m.fty.sig.0.inputs.len() != trait_m.fty.sig.0.inputs.len() {
        span_err!(tcx.sess, impl_m_span, E0050,
            "method `{}` has {} parameter{} \
             but the declaration in trait `{}` has {}",
            trait_m.name,
            impl_m.fty.sig.0.inputs.len(),
            if impl_m.fty.sig.0.inputs.len() == 1 {""} else {"s"},
            tcx.item_path_str(trait_m.def_id),
            trait_m.fty.sig.0.inputs.len());
        return;
    }

    // This code is best explained by example. Consider a trait:
    //
    //     trait Trait<'t,T> {
    //          fn method<'a,M>(t: &'t T, m: &'a M) -> Self;
    //     }
    //
    // And an impl:
    //
    //     impl<'i, 'j, U> Trait<'j, &'i U> for Foo {
    //          fn method<'b,N>(t: &'j &'i U, m: &'b N) -> Foo;
    //     }
    //
    // We wish to decide if those two method types are compatible.
    //
    // We start out with trait_to_impl_substs, that maps the trait
    // type parameters to impl type parameters. This is taken from the
    // impl trait reference:
    //
    //     trait_to_impl_substs = {'t => 'j, T => &'i U, Self => Foo}
    //
    // We create a mapping `dummy_substs` that maps from the impl type
    // parameters to fresh types and regions. For type parameters,
    // this is the identity transform, but we could as well use any
    // skolemized types. For regions, we convert from bound to free
    // regions (Note: but only early-bound regions, i.e., those
    // declared on the impl or used in type parameter bounds).
    //
    //     impl_to_skol_substs = {'i => 'i0, U => U0, N => N0 }
    //
    // Now we can apply skol_substs to the type of the impl method
    // to yield a new function type in terms of our fresh, skolemized
    // types:
    //
    //     <'b> fn(t: &'i0 U0, m: &'b) -> Foo
    //
    // We now want to extract and substitute the type of the *trait*
    // method and compare it. To do so, we must create a compound
    // substitution by combining trait_to_impl_substs and
    // impl_to_skol_substs, and also adding a mapping for the method
    // type parameters. We extend the mapping to also include
    // the method parameters.
    //
    //     trait_to_skol_substs = { T => &'i0 U0, Self => Foo, M => N0 }
    //
    // Applying this to the trait method type yields:
    //
    //     <'a> fn(t: &'i0 U0, m: &'a) -> Foo
    //
    // This type is also the same but the name of the bound region ('a
    // vs 'b).  However, the normal subtyping rules on fn types handle
    // this kind of equivalency just fine.
    //
    // We now use these substitutions to ensure that all declared bounds are
    // satisfied by the implementation's method.
    //
    // We do this by creating a parameter environment which contains a
    // substitution corresponding to impl_to_skol_substs. We then build
    // trait_to_skol_substs and use it to convert the predicates contained
    // in the trait_m.generics to the skolemized form.
    //
    // Finally we register each of these predicates as an obligation in
    // a fresh FulfillmentCtxt, and invoke select_all_or_error.

    // Create a parameter environment that represents the implementation's
    // method.
    let impl_m_node_id = tcx.map.as_local_node_id(impl_m.def_id).unwrap();
    let impl_param_env = ty::ParameterEnvironment::for_item(tcx, impl_m_node_id);

    // Create mapping from impl to skolemized.
    let impl_to_skol_substs = &impl_param_env.free_substs;

    // Create mapping from trait to skolemized.
    let trait_to_skol_substs =
        trait_to_impl_substs
        .subst(tcx, impl_to_skol_substs)
        .with_method(impl_to_skol_substs.types.get_slice(subst::FnSpace).to_vec(),
                     impl_to_skol_substs.regions().get_slice(subst::FnSpace).to_vec());
    debug!("compare_impl_method: trait_to_skol_substs={:?}",
           trait_to_skol_substs);

    // Check region bounds. FIXME(@jroesch) refactor this away when removing
    // ParamBounds.
    if !check_region_bounds_on_impl_method(tcx,
                                           impl_m_span,
                                           impl_m,
                                           &trait_m.generics,
                                           &impl_m.generics,
                                           &trait_to_skol_substs,
                                           impl_to_skol_substs) {
        return;
    }

    // Create obligations for each predicate declared by the impl
    // definition in the context of the trait's parameter
    // environment. We can't just use `impl_env.caller_bounds`,
    // however, because we want to replace all late-bound regions with
    // region variables.
    let impl_bounds =
        impl_m.predicates.instantiate(tcx, impl_to_skol_substs);

    let (impl_bounds, _) =
        infcx.replace_late_bound_regions_with_fresh_var(
            impl_m_span,
            infer::HigherRankedType,
            &ty::Binder(impl_bounds));
    debug!("compare_impl_method: impl_bounds={:?}",
           impl_bounds);

    // Normalize the associated types in the trait_bounds.
    let trait_bounds = trait_m.predicates.instantiate(tcx, &trait_to_skol_substs);

    // Obtain the predicate split predicate sets for each.
    let trait_pred = trait_bounds.predicates.split();
    let impl_pred = impl_bounds.predicates.split();

    // This is the only tricky bit of the new way we check implementation methods
    // We need to build a set of predicates where only the FnSpace bounds
    // are from the trait and we assume all other bounds from the implementation
    // to be previously satisfied.
    //
    // We then register the obligations from the impl_m and check to see
    // if all constraints hold.
    let hybrid_preds = VecPerParamSpace::new(
        impl_pred.types,
        impl_pred.selfs,
        trait_pred.fns
    );

    // Construct trait parameter environment and then shift it into the skolemized viewpoint.
    // The key step here is to update the caller_bounds's predicates to be
    // the new hybrid bounds we computed.
    let normalize_cause = traits::ObligationCause::misc(impl_m_span, impl_m_body_id);
    let trait_param_env = impl_param_env.with_caller_bounds(hybrid_preds.into_vec());
    let trait_param_env = traits::normalize_param_env_or_error(trait_param_env,
                                                               normalize_cause.clone());
    // FIXME(@jroesch) this seems ugly, but is a temporary change
    infcx.parameter_environment = trait_param_env;

    debug!("compare_impl_method: trait_bounds={:?}",
        infcx.parameter_environment.caller_bounds);

    let mut selcx = traits::SelectionContext::new(&infcx);

    for predicate in impl_pred.fns {
        let traits::Normalized { value: predicate, .. } =
            traits::normalize(&mut selcx, normalize_cause.clone(), &predicate);

        let cause = traits::ObligationCause {
            span: impl_m_span,
            body_id: impl_m_body_id,
            code: traits::ObligationCauseCode::CompareImplMethodObligation
        };

        fulfillment_cx.register_predicate_obligation(
            &infcx,
            traits::Obligation::new(cause, predicate));
    }

    // We now need to check that the signature of the impl method is
    // compatible with that of the trait method. We do this by
    // checking that `impl_fty <: trait_fty`.
    //
    // FIXME. Unfortunately, this doesn't quite work right now because
    // associated type normalization is not integrated into subtype
    // checks. For the comparison to be valid, we need to
    // normalize the associated types in the impl/trait methods
    // first. However, because function types bind regions, just
    // calling `normalize_associated_types_in` would have no effect on
    // any associated types appearing in the fn arguments or return
    // type.

    // Compute skolemized form of impl and trait method tys.
    let impl_fty = tcx.mk_fn(None, tcx.mk_bare_fn(impl_m.fty.clone()));
    let impl_fty = impl_fty.subst(tcx, impl_to_skol_substs);
    let trait_fty = tcx.mk_fn(None, tcx.mk_bare_fn(trait_m.fty.clone()));
    let trait_fty = trait_fty.subst(tcx, &trait_to_skol_substs);

    let err = infcx.commit_if_ok(|snapshot| {
        let origin = TypeOrigin::MethodCompatCheck(impl_m_span);

        let (impl_sig, _) =
            infcx.replace_late_bound_regions_with_fresh_var(impl_m_span,
                                                            infer::HigherRankedType,
                                                            &impl_m.fty.sig);
        let impl_sig =
            impl_sig.subst(tcx, impl_to_skol_substs);
        let impl_sig =
            assoc::normalize_associated_types_in(&infcx,
                                                 &mut fulfillment_cx,
                                                 impl_m_span,
                                                 impl_m_body_id,
                                                 &impl_sig);
        let impl_fty = tcx.mk_fn(None, tcx.mk_bare_fn(ty::BareFnTy {
            unsafety: impl_m.fty.unsafety,
            abi: impl_m.fty.abi,
            sig: ty::Binder(impl_sig)
        }));
        debug!("compare_impl_method: impl_fty={:?}",
               impl_fty);

        let (trait_sig, skol_map) =
            infcx.skolemize_late_bound_regions(&trait_m.fty.sig, snapshot);
        let trait_sig =
            trait_sig.subst(tcx, &trait_to_skol_substs);
        let trait_sig =
            assoc::normalize_associated_types_in(&infcx,
                                                 &mut fulfillment_cx,
                                                 impl_m_span,
                                                 impl_m_body_id,
                                                 &trait_sig);
        let trait_fty = tcx.mk_fn(None, tcx.mk_bare_fn(ty::BareFnTy {
            unsafety: trait_m.fty.unsafety,
            abi: trait_m.fty.abi,
            sig: ty::Binder(trait_sig)
        }));

        debug!("compare_impl_method: trait_fty={:?}",
               trait_fty);

        try!(infer::mk_subty(&infcx, false, origin, impl_fty, trait_fty));

        infcx.leak_check(&skol_map, snapshot)
    });

    match err {
        Ok(()) => { }
        Err(terr) => {
            debug!("checking trait method for compatibility: impl ty {:?}, trait ty {:?}",
                   impl_fty,
                   trait_fty);
            span_err!(tcx.sess, impl_m_span, E0053,
                      "method `{}` has an incompatible type for trait: {}",
                      trait_m.name,
                      terr);
            return;
        }
    }

    // Check that all obligations are satisfied by the implementation's
    // version.
    match fulfillment_cx.select_all_or_error(&infcx) {
        Err(ref errors) => { traits::report_fulfillment_errors(&infcx, errors) }
        Ok(_) => {}
    }

    // Finally, resolve all regions. This catches wily misuses of
    // lifetime parameters. We have to build up a plausible lifetime
    // environment based on what we find in the trait. We could also
    // include the obligations derived from the method argument types,
    // but I don't think it's necessary -- after all, those are still
    // in effect when type-checking the body, and all the
    // where-clauses in the header etc should be implied by the trait
    // anyway, so it shouldn't be needed there either. Anyway, we can
    // always add more relations later (it's backwards compat).
    let mut free_regions = FreeRegionMap::new();
    free_regions.relate_free_regions_from_predicates(tcx,
                                                     &infcx.parameter_environment.caller_bounds);

    infcx.resolve_regions_and_report_errors(&free_regions, impl_m_body_id);

    fn check_region_bounds_on_impl_method<'tcx>(tcx: &ty::ctxt<'tcx>,
                                                span: Span,
                                                impl_m: &ty::Method<'tcx>,
                                                trait_generics: &ty::Generics<'tcx>,
                                                impl_generics: &ty::Generics<'tcx>,
                                                trait_to_skol_substs: &Substs<'tcx>,
                                                impl_to_skol_substs: &Substs<'tcx>)
                                                -> bool
    {

        let trait_params = trait_generics.regions.get_slice(subst::FnSpace);
        let impl_params = impl_generics.regions.get_slice(subst::FnSpace);

        debug!("check_region_bounds_on_impl_method: \
               trait_generics={:?} \
               impl_generics={:?} \
               trait_to_skol_substs={:?} \
               impl_to_skol_substs={:?}",
               trait_generics,
               impl_generics,
               trait_to_skol_substs,
               impl_to_skol_substs);

        // Must have same number of early-bound lifetime parameters.
        // Unfortunately, if the user screws up the bounds, then this
        // will change classification between early and late.  E.g.,
        // if in trait we have `<'a,'b:'a>`, and in impl we just have
        // `<'a,'b>`, then we have 2 early-bound lifetime parameters
        // in trait but 0 in the impl. But if we report "expected 2
        // but found 0" it's confusing, because it looks like there
        // are zero. Since I don't quite know how to phrase things at
        // the moment, give a kind of vague error message.
        if trait_params.len() != impl_params.len() {
            span_err!(tcx.sess, span, E0195,
                "lifetime parameters or bounds on method `{}` do \
                         not match the trait declaration",
                         impl_m.name);
            return false;
        }

        return true;
    }
}

pub fn compare_const_impl<'tcx>(tcx: &ty::ctxt<'tcx>,
                                impl_c: &ty::AssociatedConst<'tcx>,
                                impl_c_span: Span,
                                trait_c: &ty::AssociatedConst<'tcx>,
                                impl_trait_ref: &ty::TraitRef<'tcx>) {
    debug!("compare_const_impl(impl_trait_ref={:?})",
           impl_trait_ref);

    let infcx = infer::new_infer_ctxt(tcx, &tcx.tables, None);
    let mut fulfillment_cx = infcx.fulfillment_cx.borrow_mut();

    // The below is for the most part highly similar to the procedure
    // for methods above. It is simpler in many respects, especially
    // because we shouldn't really have to deal with lifetimes or
    // predicates. In fact some of this should probably be put into
    // shared functions because of DRY violations...
    let trait_to_impl_substs = &impl_trait_ref.substs;

    // Create a parameter environment that represents the implementation's
    // method.
    let impl_c_node_id = tcx.map.as_local_node_id(impl_c.def_id).unwrap();
    let impl_param_env = ty::ParameterEnvironment::for_item(tcx, impl_c_node_id);

    // Create mapping from impl to skolemized.
    let impl_to_skol_substs = &impl_param_env.free_substs;

    // Create mapping from trait to skolemized.
    let trait_to_skol_substs =
        trait_to_impl_substs
        .subst(tcx, impl_to_skol_substs)
        .with_method(impl_to_skol_substs.types.get_slice(subst::FnSpace).to_vec(),
                     impl_to_skol_substs.regions().get_slice(subst::FnSpace).to_vec());
    debug!("compare_const_impl: trait_to_skol_substs={:?}",
           trait_to_skol_substs);

    // Compute skolemized form of impl and trait const tys.
    let impl_ty = impl_c.ty.subst(tcx, impl_to_skol_substs);
    let trait_ty = trait_c.ty.subst(tcx, &trait_to_skol_substs);

    let err = infcx.commit_if_ok(|_| {
        let origin = TypeOrigin::Misc(impl_c_span);

        // There is no "body" here, so just pass dummy id.
        let impl_ty =
            assoc::normalize_associated_types_in(&infcx,
                                                 &mut fulfillment_cx,
                                                 impl_c_span,
                                                 0,
                                                 &impl_ty);

        debug!("compare_const_impl: impl_ty={:?}",
               impl_ty);

        let trait_ty =
            assoc::normalize_associated_types_in(&infcx,
                                                 &mut fulfillment_cx,
                                                 impl_c_span,
                                                 0,
                                                 &trait_ty);

        debug!("compare_const_impl: trait_ty={:?}",
               trait_ty);

        infer::mk_subty(&infcx, false, origin, impl_ty, trait_ty)
    });

    match err {
        Ok(()) => { }
        Err(terr) => {
            debug!("checking associated const for compatibility: impl ty {:?}, trait ty {:?}",
                   impl_ty,
                   trait_ty);
            span_err!(tcx.sess, impl_c_span, E0326,
                      "implemented const `{}` has an incompatible type for \
                      trait: {}",
                      trait_c.name,
                      terr);
            return;
        }
    }
}
