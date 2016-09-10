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
use rustc::infer::{self, InferOk, TypeOrigin};
use rustc::ty;
use rustc::traits::{self, Reveal};
use rustc::ty::error::{ExpectedFound, TypeError};
use rustc::ty::subst::{Subst, Substs};
use rustc::hir::{ImplItemKind, TraitItem_, Ty_};

use syntax::ast;
use syntax_pos::Span;

use CrateCtxt;
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

pub fn compare_impl_method<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                     impl_m: &ty::Method<'tcx>,
                                     impl_m_span: Span,
                                     impl_m_body_id: ast::NodeId,
                                     trait_m: &ty::Method<'tcx>,
                                     impl_trait_ref: &ty::TraitRef<'tcx>,
                                     trait_item_span: Option<Span>) {
    debug!("compare_impl_method(impl_trait_ref={:?})",
           impl_trait_ref);

    debug!("compare_impl_method: impl_trait_ref (liberated) = {:?}",
           impl_trait_ref);

    let tcx = ccx.tcx;

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
            let mut err = struct_span_err!(tcx.sess, impl_m_span, E0185,
                "method `{}` has a `{}` declaration in the impl, \
                        but not in the trait",
                        trait_m.name,
                        impl_m.explicit_self);
            err.span_label(impl_m_span, &format!("`{}` used in impl",
                                                 impl_m.explicit_self));
            if let Some(span) = tcx.map.span_if_local(trait_m.def_id) {
                err.span_label(span, &format!("trait declared without `{}`",
                                              impl_m.explicit_self));
            }
            err.emit();
            return;
        }
        (_, &ty::ExplicitSelfCategory::Static) => {
            let mut err = struct_span_err!(tcx.sess, impl_m_span, E0186,
                "method `{}` has a `{}` declaration in the trait, \
                        but not in the impl",
                        trait_m.name,
                        trait_m.explicit_self);
            err.span_label(impl_m_span, &format!("expected `{}` in impl",
                                                  trait_m.explicit_self));
            if let Some(span) = tcx.map.span_if_local(trait_m.def_id) {
                err.span_label(span, & format!("`{}` used in trait",
                                               trait_m.explicit_self));
            }
            err.emit();
            return;
        }
        _ => {
            // Let the type checker catch other errors below
        }
    }

    let num_impl_m_type_params = impl_m.generics.types.len();
    let num_trait_m_type_params = trait_m.generics.types.len();
    if num_impl_m_type_params != num_trait_m_type_params {
        let impl_m_node_id = tcx.map.as_local_node_id(impl_m.def_id).unwrap();
        let span = match tcx.map.expect_impl_item(impl_m_node_id).node {
            ImplItemKind::Method(ref impl_m_sig, _) => {
                if impl_m_sig.generics.is_parameterized() {
                    impl_m_sig.generics.span
                } else {
                    impl_m_span
                }
            }
            _ => bug!("{:?} is not a method", impl_m)
        };

        struct_span_err!(tcx.sess, span, E0049,
            "method `{}` has {} type parameter{} \
             but its trait declaration has {} type parameter{}",
            trait_m.name,
            num_impl_m_type_params,
            if num_impl_m_type_params == 1 {""} else {"s"},
            num_trait_m_type_params,
            if num_trait_m_type_params == 1 {""} else {"s"})
            .span_label(trait_item_span.unwrap(),
                        &format!("expected {}",
                                 &if num_trait_m_type_params != 1 {
                                     format!("{} type parameters",
                                             num_trait_m_type_params)
                                 } else {
                                     format!("{} type parameter",
                                             num_trait_m_type_params)
                                 }))
            .span_label(span, &format!("found {}",
                                       &if num_impl_m_type_params != 1 {
                                           format!("{} type parameters", num_impl_m_type_params)
                                       } else {
                                           format!("1 type parameter")
                                       }))
            .emit();
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
        impl_to_skol_substs.rebase_onto(tcx, impl_m.container_id(),
                                        trait_to_impl_substs.subst(tcx, impl_to_skol_substs));
    debug!("compare_impl_method: trait_to_skol_substs={:?}",
           trait_to_skol_substs);

    // Check region bounds. FIXME(@jroesch) refactor this away when removing
    // ParamBounds.
    if !check_region_bounds_on_impl_method(ccx,
                                           impl_m_span,
                                           impl_m,
                                           &trait_m.generics,
                                           &impl_m.generics,
                                           trait_to_skol_substs,
                                           impl_to_skol_substs) {
        return;
    }

    tcx.infer_ctxt(None, None, Reveal::NotSpecializable).enter(|mut infcx| {
        let mut fulfillment_cx = traits::FulfillmentContext::new();

        // Create obligations for each predicate declared by the impl
        // definition in the context of the trait's parameter
        // environment. We can't just use `impl_env.caller_bounds`,
        // however, because we want to replace all late-bound regions with
        // region variables.
        let impl_predicates = tcx.lookup_predicates(impl_m.predicates.parent.unwrap());
        let mut hybrid_preds = impl_predicates.instantiate(tcx, impl_to_skol_substs);

        debug!("compare_impl_method: impl_bounds={:?}", hybrid_preds);

        // This is the only tricky bit of the new way we check implementation methods
        // We need to build a set of predicates where only the method-level bounds
        // are from the trait and we assume all other bounds from the implementation
        // to be previously satisfied.
        //
        // We then register the obligations from the impl_m and check to see
        // if all constraints hold.
        hybrid_preds.predicates.extend(
            trait_m.predicates.instantiate_own(tcx, trait_to_skol_substs).predicates);

        // Construct trait parameter environment and then shift it into the skolemized viewpoint.
        // The key step here is to update the caller_bounds's predicates to be
        // the new hybrid bounds we computed.
        let normalize_cause = traits::ObligationCause::misc(impl_m_span, impl_m_body_id);
        let trait_param_env = impl_param_env.with_caller_bounds(hybrid_preds.predicates);
        let trait_param_env = traits::normalize_param_env_or_error(tcx,
                                                                   trait_param_env,
                                                                   normalize_cause.clone());
        // FIXME(@jroesch) this seems ugly, but is a temporary change
        infcx.parameter_environment = trait_param_env;

        debug!("compare_impl_method: caller_bounds={:?}",
            infcx.parameter_environment.caller_bounds);

        let mut selcx = traits::SelectionContext::new(&infcx);

        let impl_m_own_bounds = impl_m.predicates.instantiate_own(tcx, impl_to_skol_substs);
        let (impl_m_own_bounds, _) =
            infcx.replace_late_bound_regions_with_fresh_var(
                impl_m_span,
                infer::HigherRankedType,
                &ty::Binder(impl_m_own_bounds.predicates));
        for predicate in impl_m_own_bounds {
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
        let tcx = infcx.tcx;
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
        let impl_fty = tcx.mk_fn_ptr(tcx.mk_bare_fn(ty::BareFnTy {
            unsafety: impl_m.fty.unsafety,
            abi: impl_m.fty.abi,
            sig: ty::Binder(impl_sig.clone())
        }));
        debug!("compare_impl_method: impl_fty={:?}", impl_fty);

        let trait_sig = tcx.liberate_late_bound_regions(
            infcx.parameter_environment.free_id_outlive,
            &trait_m.fty.sig);
        let trait_sig =
            trait_sig.subst(tcx, trait_to_skol_substs);
        let trait_sig =
            assoc::normalize_associated_types_in(&infcx,
                                                 &mut fulfillment_cx,
                                                 impl_m_span,
                                                 impl_m_body_id,
                                                 &trait_sig);
        let trait_fty = tcx.mk_fn_ptr(tcx.mk_bare_fn(ty::BareFnTy {
            unsafety: trait_m.fty.unsafety,
            abi: trait_m.fty.abi,
            sig: ty::Binder(trait_sig.clone())
        }));

        debug!("compare_impl_method: trait_fty={:?}", trait_fty);

        if let Err(terr) = infcx.sub_types(false, origin, impl_fty, trait_fty) {
            debug!("sub_types failed: impl ty {:?}, trait ty {:?}",
                   impl_fty,
                   trait_fty);

            let (impl_err_span, trait_err_span) =
                extract_spans_for_error_reporting(&infcx, &terr, origin, impl_m,
                    impl_sig, trait_m, trait_sig);

            let origin = TypeOrigin::MethodCompatCheck(impl_err_span);

            let mut diag = struct_span_err!(
                tcx.sess, origin.span(), E0053,
                "method `{}` has an incompatible type for trait", trait_m.name
            );

            infcx.note_type_err(
                &mut diag,
                origin,
                trait_err_span.map(|sp| (sp, format!("type in trait"))),
                Some(infer::ValuePairs::Types(ExpectedFound {
                     expected: trait_fty,
                     found: impl_fty
                })),
                &terr
            );
            diag.emit();
            return
        }

        // Check that all obligations are satisfied by the implementation's
        // version.
        if let Err(ref errors) = fulfillment_cx.select_all_or_error(&infcx) {
            infcx.report_fulfillment_errors(errors);
            return
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
        free_regions.relate_free_regions_from_predicates(
            &infcx.parameter_environment.caller_bounds);

        infcx.resolve_regions_and_report_errors(&free_regions, impl_m_body_id);
    });

    fn check_region_bounds_on_impl_method<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                                    span: Span,
                                                    impl_m: &ty::Method<'tcx>,
                                                    trait_generics: &ty::Generics<'tcx>,
                                                    impl_generics: &ty::Generics<'tcx>,
                                                    trait_to_skol_substs: &Substs<'tcx>,
                                                    impl_to_skol_substs: &Substs<'tcx>)
                                                    -> bool
    {

        let trait_params = &trait_generics.regions[..];
        let impl_params = &impl_generics.regions[..];

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
            struct_span_err!(ccx.tcx.sess, span, E0195,
                "lifetime parameters or bounds on method `{}` do \
                 not match the trait declaration",impl_m.name)
                .span_label(span, &format!("lifetimes do not match trait"))
                .emit();
            return false;
        }

        return true;
    }

    fn extract_spans_for_error_reporting<'a, 'gcx, 'tcx>(infcx: &infer::InferCtxt<'a, 'gcx, 'tcx>,
                                                         terr: &TypeError,
                                                         origin: TypeOrigin,
                                                         impl_m: &ty::Method,
                                                         impl_sig: ty::FnSig<'tcx>,
                                                         trait_m: &ty::Method,
                                                         trait_sig: ty::FnSig<'tcx>)
                                                        -> (Span, Option<Span>) {
        let tcx = infcx.tcx;
        let impl_m_node_id = tcx.map.as_local_node_id(impl_m.def_id).unwrap();
        let (impl_m_output, impl_m_iter) = match tcx.map.expect_impl_item(impl_m_node_id).node {
            ImplItemKind::Method(ref impl_m_sig, _) =>
                (&impl_m_sig.decl.output, impl_m_sig.decl.inputs.iter()),
            _ => bug!("{:?} is not a method", impl_m)
        };

        match *terr {
            TypeError::Mutability => {
                if let Some(trait_m_node_id) = tcx.map.as_local_node_id(trait_m.def_id) {
                    let trait_m_iter = match tcx.map.expect_trait_item(trait_m_node_id).node {
                        TraitItem_::MethodTraitItem(ref trait_m_sig, _) =>
                            trait_m_sig.decl.inputs.iter(),
                        _ => bug!("{:?} is not a MethodTraitItem", trait_m)
                    };

                    impl_m_iter.zip(trait_m_iter).find(|&(ref impl_arg, ref trait_arg)| {
                        match (&impl_arg.ty.node, &trait_arg.ty.node) {
                            (&Ty_::TyRptr(_, ref impl_mt), &Ty_::TyRptr(_, ref trait_mt)) |
                            (&Ty_::TyPtr(ref impl_mt), &Ty_::TyPtr(ref trait_mt)) =>
                                impl_mt.mutbl != trait_mt.mutbl,
                            _ => false
                        }
                    }).map(|(ref impl_arg, ref trait_arg)| {
                        match (impl_arg.to_self(), trait_arg.to_self()) {
                            (Some(impl_self), Some(trait_self)) =>
                                (impl_self.span, Some(trait_self.span)),
                            (None, None) => (impl_arg.ty.span, Some(trait_arg.ty.span)),
                            _ => bug!("impl and trait fns have different first args, \
                                       impl: {:?}, trait: {:?}", impl_arg, trait_arg)
                        }
                    }).unwrap_or((origin.span(), tcx.map.span_if_local(trait_m.def_id)))
                } else {
                    (origin.span(), tcx.map.span_if_local(trait_m.def_id))
                }
            }
            TypeError::Sorts(ExpectedFound { .. }) => {
                if let Some(trait_m_node_id) = tcx.map.as_local_node_id(trait_m.def_id) {
                    let (trait_m_output, trait_m_iter) =
                    match tcx.map.expect_trait_item(trait_m_node_id).node {
                        TraitItem_::MethodTraitItem(ref trait_m_sig, _) =>
                            (&trait_m_sig.decl.output, trait_m_sig.decl.inputs.iter()),
                        _ => bug!("{:?} is not a MethodTraitItem", trait_m)
                    };

                    let impl_iter = impl_sig.inputs.iter();
                    let trait_iter = trait_sig.inputs.iter();
                    impl_iter.zip(trait_iter).zip(impl_m_iter).zip(trait_m_iter)
                        .filter_map(|(((impl_arg_ty, trait_arg_ty), impl_arg), trait_arg)| {
                            match infcx.sub_types(true, origin, trait_arg_ty, impl_arg_ty) {
                                Ok(_) => None,
                                Err(_) => Some((impl_arg.ty.span, Some(trait_arg.ty.span)))
                            }
                        })
                        .next()
                        .unwrap_or_else(|| {
                            if infcx.sub_types(false, origin, impl_sig.output,
                                               trait_sig.output).is_err() {
                                (impl_m_output.span(), Some(trait_m_output.span()))
                            } else {
                                (origin.span(), tcx.map.span_if_local(trait_m.def_id))
                            }
                        })
                } else {
                    (origin.span(), tcx.map.span_if_local(trait_m.def_id))
                }
            }
            _ => (origin.span(), tcx.map.span_if_local(trait_m.def_id))
        }
    }
}

pub fn compare_const_impl<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                    impl_c: &ty::AssociatedConst<'tcx>,
                                    impl_c_span: Span,
                                    trait_c: &ty::AssociatedConst<'tcx>,
                                    impl_trait_ref: &ty::TraitRef<'tcx>) {
    debug!("compare_const_impl(impl_trait_ref={:?})",
           impl_trait_ref);

    let tcx = ccx.tcx;
    tcx.infer_ctxt(None, None, Reveal::NotSpecializable).enter(|infcx| {
        let mut fulfillment_cx = traits::FulfillmentContext::new();

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
            impl_to_skol_substs.rebase_onto(tcx, impl_c.container.id(),
                                            trait_to_impl_substs.subst(tcx, impl_to_skol_substs));
        debug!("compare_const_impl: trait_to_skol_substs={:?}",
            trait_to_skol_substs);

        // Compute skolemized form of impl and trait const tys.
        let impl_ty = impl_c.ty.subst(tcx, impl_to_skol_substs);
        let trait_ty = trait_c.ty.subst(tcx, trait_to_skol_substs);
        let mut origin = TypeOrigin::Misc(impl_c_span);

        let err = infcx.commit_if_ok(|_| {
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

            infcx.sub_types(false, origin, impl_ty, trait_ty)
                 .map(|InferOk { obligations, .. }| {
                // FIXME(#32730) propagate obligations
                assert!(obligations.is_empty())
            })
        });

        if let Err(terr) = err {
            debug!("checking associated const for compatibility: impl ty {:?}, trait ty {:?}",
                   impl_ty,
                   trait_ty);

            // Locate the Span containing just the type of the offending impl
            match tcx.map.expect_impl_item(impl_c_node_id).node {
                ImplItemKind::Const(ref ty, _) => origin = TypeOrigin::Misc(ty.span),
                _ => bug!("{:?} is not a impl const", impl_c)
            }

            let mut diag = struct_span_err!(
                tcx.sess, origin.span(), E0326,
                "implemented const `{}` has an incompatible type for trait",
                trait_c.name
            );

            // Add a label to the Span containing just the type of the item
            let trait_c_node_id = tcx.map.as_local_node_id(trait_c.def_id).unwrap();
            let trait_c_span = match tcx.map.expect_trait_item(trait_c_node_id).node {
                TraitItem_::ConstTraitItem(ref ty, _) => ty.span,
                _ => bug!("{:?} is not a trait const", trait_c)
            };

            infcx.note_type_err(
                &mut diag,
                origin,
                Some((trait_c_span, format!("type in trait"))),
                Some(infer::ValuePairs::Types(ExpectedFound {
                    expected: trait_ty,
                    found: impl_ty
                })), &terr
            );
            diag.emit();
        }
    });
}
