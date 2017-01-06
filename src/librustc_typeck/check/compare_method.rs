// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::{self, ImplItemKind, TraitItemKind};
use rustc::infer::{self, InferOk};
use rustc::middle::free_region::FreeRegionMap;
use rustc::ty;
use rustc::traits::{self, ObligationCause, ObligationCauseCode, Reveal};
use rustc::ty::error::{ExpectedFound, TypeError};
use rustc::ty::subst::{Subst, Substs};
use rustc::util::common::ErrorReported;

use syntax::ast;
use syntax_pos::Span;

use CrateCtxt;
use super::assoc;
use super::{Inherited, FnCtxt};
use astconv::ExplicitSelf;

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
                                     impl_m: &ty::AssociatedItem,
                                     impl_m_span: Span,
                                     impl_m_body_id: ast::NodeId,
                                     trait_m: &ty::AssociatedItem,
                                     impl_trait_ref: ty::TraitRef<'tcx>,
                                     trait_item_span: Option<Span>,
                                     old_broken_mode: bool) {
    debug!("compare_impl_method(impl_trait_ref={:?})",
           impl_trait_ref);

    if let Err(ErrorReported) = compare_self_type(ccx,
                                                  impl_m,
                                                  impl_m_span,
                                                  trait_m,
                                                  impl_trait_ref) {
        return;
    }

    if let Err(ErrorReported) = compare_number_of_generics(ccx,
                                                           impl_m,
                                                           impl_m_span,
                                                           trait_m,
                                                           trait_item_span) {
        return;
    }

    if let Err(ErrorReported) = compare_number_of_method_arguments(ccx,
                                                                   impl_m,
                                                                   impl_m_span,
                                                                   trait_m,
                                                                   trait_item_span) {
        return;
    }

    if let Err(ErrorReported) = compare_predicate_entailment(ccx,
                                                             impl_m,
                                                             impl_m_span,
                                                             impl_m_body_id,
                                                             trait_m,
                                                             impl_trait_ref,
                                                             old_broken_mode) {
        return;
    }
}

fn compare_predicate_entailment<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                          impl_m: &ty::AssociatedItem,
                                          impl_m_span: Span,
                                          impl_m_body_id: ast::NodeId,
                                          trait_m: &ty::AssociatedItem,
                                          impl_trait_ref: ty::TraitRef<'tcx>,
                                          old_broken_mode: bool)
                                          -> Result<(), ErrorReported> {
    let tcx = ccx.tcx;

    let trait_to_impl_substs = impl_trait_ref.substs;

    let cause = ObligationCause {
        span: impl_m_span,
        body_id: impl_m_body_id,
        code: ObligationCauseCode::CompareImplMethodObligation {
            item_name: impl_m.name,
            impl_item_def_id: impl_m.def_id,
            trait_item_def_id: trait_m.def_id,
            lint_id: if !old_broken_mode { Some(impl_m_body_id) } else { None },
        },
    };

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
    let trait_to_skol_substs = impl_to_skol_substs.rebase_onto(tcx,
                                                               impl_m.container.id(),
                                                               trait_to_impl_substs.subst(tcx,
                                                                          impl_to_skol_substs));
    debug!("compare_impl_method: trait_to_skol_substs={:?}",
           trait_to_skol_substs);

    let impl_m_generics = tcx.item_generics(impl_m.def_id);
    let trait_m_generics = tcx.item_generics(trait_m.def_id);
    let impl_m_predicates = tcx.item_predicates(impl_m.def_id);
    let trait_m_predicates = tcx.item_predicates(trait_m.def_id);

    // Check region bounds.
    check_region_bounds_on_impl_method(ccx,
                                       impl_m_span,
                                       impl_m,
                                       &trait_m_generics,
                                       &impl_m_generics,
                                       trait_to_skol_substs,
                                       impl_to_skol_substs)?;

    // Create obligations for each predicate declared by the impl
    // definition in the context of the trait's parameter
    // environment. We can't just use `impl_env.caller_bounds`,
    // however, because we want to replace all late-bound regions with
    // region variables.
    let impl_predicates = tcx.item_predicates(impl_m_predicates.parent.unwrap());
    let mut hybrid_preds = impl_predicates.instantiate(tcx, impl_to_skol_substs);

    debug!("compare_impl_method: impl_bounds={:?}", hybrid_preds);

    // This is the only tricky bit of the new way we check implementation methods
    // We need to build a set of predicates where only the method-level bounds
    // are from the trait and we assume all other bounds from the implementation
    // to be previously satisfied.
    //
    // We then register the obligations from the impl_m and check to see
    // if all constraints hold.
    hybrid_preds.predicates
                .extend(trait_m_predicates.instantiate_own(tcx, trait_to_skol_substs).predicates);

    // Construct trait parameter environment and then shift it into the skolemized viewpoint.
    // The key step here is to update the caller_bounds's predicates to be
    // the new hybrid bounds we computed.
    let normalize_cause = traits::ObligationCause::misc(impl_m_span, impl_m_body_id);
    let trait_param_env = impl_param_env.with_caller_bounds(hybrid_preds.predicates);
    let trait_param_env = traits::normalize_param_env_or_error(tcx,
                                                               trait_param_env,
                                                               normalize_cause.clone());

    tcx.infer_ctxt(trait_param_env, Reveal::NotSpecializable).enter(|infcx| {
        let inh = Inherited::new(ccx, infcx);
        let infcx = &inh.infcx;
        let fulfillment_cx = &inh.fulfillment_cx;

        debug!("compare_impl_method: caller_bounds={:?}",
               infcx.parameter_environment.caller_bounds);

        let mut selcx = traits::SelectionContext::new(&infcx);

        let impl_m_own_bounds = impl_m_predicates.instantiate_own(tcx, impl_to_skol_substs);
        let (impl_m_own_bounds, _) = infcx.replace_late_bound_regions_with_fresh_var(impl_m_span,
                                                       infer::HigherRankedType,
                                                       &ty::Binder(impl_m_own_bounds.predicates));
        for predicate in impl_m_own_bounds {
            let traits::Normalized { value: predicate, .. } =
                traits::normalize(&mut selcx, normalize_cause.clone(), &predicate);

            fulfillment_cx.borrow_mut().register_predicate_obligation(
                &infcx,
                traits::Obligation::new(cause.clone(), predicate));
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

        let m_fty = |method: &ty::AssociatedItem| {
            match tcx.item_type(method.def_id).sty {
                ty::TyFnDef(_, _, f) => f,
                _ => bug!()
            }
        };
        let impl_m_fty = m_fty(impl_m);
        let trait_m_fty = m_fty(trait_m);

        let (impl_sig, _) =
            infcx.replace_late_bound_regions_with_fresh_var(impl_m_span,
                                                            infer::HigherRankedType,
                                                            &impl_m_fty.sig);
        let impl_sig =
            impl_sig.subst(tcx, impl_to_skol_substs);
        let impl_sig =
            assoc::normalize_associated_types_in(&infcx,
                                                 &mut fulfillment_cx.borrow_mut(),
                                                 impl_m_span,
                                                 impl_m_body_id,
                                                 &impl_sig);
        let impl_fty = tcx.mk_fn_ptr(tcx.mk_bare_fn(ty::BareFnTy {
            unsafety: impl_m_fty.unsafety,
            abi: impl_m_fty.abi,
            sig: ty::Binder(impl_sig.clone()),
        }));
        debug!("compare_impl_method: impl_fty={:?}", impl_fty);

        let trait_sig = tcx.liberate_late_bound_regions(
            infcx.parameter_environment.free_id_outlive,
            &trait_m_fty.sig);
        let trait_sig =
            trait_sig.subst(tcx, trait_to_skol_substs);
        let trait_sig =
            assoc::normalize_associated_types_in(&infcx,
                                                 &mut fulfillment_cx.borrow_mut(),
                                                 impl_m_span,
                                                 impl_m_body_id,
                                                 &trait_sig);
        let trait_fty = tcx.mk_fn_ptr(tcx.mk_bare_fn(ty::BareFnTy {
            unsafety: trait_m_fty.unsafety,
            abi: trait_m_fty.abi,
            sig: ty::Binder(trait_sig.clone()),
        }));

        debug!("compare_impl_method: trait_fty={:?}", trait_fty);

        let sub_result = infcx.sub_types(false, &cause, impl_fty, trait_fty)
            .map(|InferOk { obligations, .. }| {
                // FIXME(#32730) propagate obligations
                assert!(obligations.is_empty());
            });

        if let Err(terr) = sub_result {
            debug!("sub_types failed: impl ty {:?}, trait ty {:?}",
                   impl_fty,
                   trait_fty);

            let (impl_err_span, trait_err_span) = extract_spans_for_error_reporting(&infcx,
                                                                                    &terr,
                                                                                    &cause,
                                                                                    impl_m,
                                                                                    impl_sig,
                                                                                    trait_m,
                                                                                    trait_sig);

            let cause = ObligationCause {
                span: impl_err_span,
                ..cause.clone()
            };

            let mut diag = struct_span_err!(tcx.sess,
                                            cause.span,
                                            E0053,
                                            "method `{}` has an incompatible type for trait",
                                            trait_m.name);

            infcx.note_type_err(&mut diag,
                                &cause,
                                trait_err_span.map(|sp| (sp, format!("type in trait"))),
                                Some(infer::ValuePairs::Types(ExpectedFound {
                                    expected: trait_fty,
                                    found: impl_fty,
                                })),
                                &terr);
            diag.emit();
            return Err(ErrorReported);
        }

        // Check that all obligations are satisfied by the implementation's
        // version.
        if let Err(ref errors) = fulfillment_cx.borrow_mut().select_all_or_error(&infcx) {
            infcx.report_fulfillment_errors(errors);
            return Err(ErrorReported);
        }

        // Finally, resolve all regions. This catches wily misuses of
        // lifetime parameters.
        if old_broken_mode {
            // FIXME(#18937) -- this is how the code used to
            // work. This is buggy because the fulfillment cx creates
            // region obligations that get overlooked.  The right
            // thing to do is the code below. But we keep this old
            // pass around temporarily.
            let mut free_regions = FreeRegionMap::new();
            free_regions.relate_free_regions_from_predicates(
                &infcx.parameter_environment.caller_bounds);
            infcx.resolve_regions_and_report_errors(&free_regions, impl_m_body_id);
        } else {
            let fcx = FnCtxt::new(&inh, Some(tcx.types.err), impl_m_body_id);
            fcx.regionck_item(impl_m_body_id, impl_m_span, &[]);
        }

        Ok(())
    })
}

fn check_region_bounds_on_impl_method<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                                span: Span,
                                                impl_m: &ty::AssociatedItem,
                                                trait_generics: &ty::Generics<'tcx>,
                                                impl_generics: &ty::Generics<'tcx>,
                                                trait_to_skol_substs: &Substs<'tcx>,
                                                impl_to_skol_substs: &Substs<'tcx>)
                                                -> Result<(), ErrorReported> {
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
        struct_span_err!(ccx.tcx.sess,
                         span,
                         E0195,
                         "lifetime parameters or bounds on method `{}` do not match the \
                          trait declaration",
                         impl_m.name)
            .span_label(span, &format!("lifetimes do not match trait"))
            .emit();
        return Err(ErrorReported);
    }

    return Ok(());
}

fn extract_spans_for_error_reporting<'a, 'gcx, 'tcx>(infcx: &infer::InferCtxt<'a, 'gcx, 'tcx>,
                                                     terr: &TypeError,
                                                     cause: &ObligationCause<'tcx>,
                                                     impl_m: &ty::AssociatedItem,
                                                     impl_sig: ty::FnSig<'tcx>,
                                                     trait_m: &ty::AssociatedItem,
                                                     trait_sig: ty::FnSig<'tcx>)
                                                     -> (Span, Option<Span>) {
    let tcx = infcx.tcx;
    let impl_m_node_id = tcx.map.as_local_node_id(impl_m.def_id).unwrap();
    let (impl_m_output, impl_m_iter) = match tcx.map.expect_impl_item(impl_m_node_id).node {
        ImplItemKind::Method(ref impl_m_sig, _) => {
            (&impl_m_sig.decl.output, impl_m_sig.decl.inputs.iter())
        }
        _ => bug!("{:?} is not a method", impl_m),
    };

    match *terr {
        TypeError::Mutability => {
            if let Some(trait_m_node_id) = tcx.map.as_local_node_id(trait_m.def_id) {
                let trait_m_iter = match tcx.map.expect_trait_item(trait_m_node_id).node {
                    TraitItemKind::Method(ref trait_m_sig, _) => {
                        trait_m_sig.decl.inputs.iter()
                    }
                    _ => bug!("{:?} is not a TraitItemKind::Method", trait_m),
                };

                impl_m_iter.zip(trait_m_iter).find(|&(ref impl_arg, ref trait_arg)| {
                    match (&impl_arg.node, &trait_arg.node) {
                        (&hir::TyRptr(_, ref impl_mt), &hir::TyRptr(_, ref trait_mt)) |
                        (&hir::TyPtr(ref impl_mt), &hir::TyPtr(ref trait_mt)) => {
                            impl_mt.mutbl != trait_mt.mutbl
                        }
                        _ => false,
                    }
                }).map(|(ref impl_arg, ref trait_arg)| {
                    (impl_arg.span, Some(trait_arg.span))
                })
                .unwrap_or_else(|| (cause.span, tcx.map.span_if_local(trait_m.def_id)))
            } else {
                (cause.span, tcx.map.span_if_local(trait_m.def_id))
            }
        }
        TypeError::Sorts(ExpectedFound { .. }) => {
            if let Some(trait_m_node_id) = tcx.map.as_local_node_id(trait_m.def_id) {
                let (trait_m_output, trait_m_iter) =
                    match tcx.map.expect_trait_item(trait_m_node_id).node {
                        TraitItemKind::Method(ref trait_m_sig, _) => {
                            (&trait_m_sig.decl.output, trait_m_sig.decl.inputs.iter())
                        }
                        _ => bug!("{:?} is not a TraitItemKind::Method", trait_m),
                    };

                let impl_iter = impl_sig.inputs().iter();
                let trait_iter = trait_sig.inputs().iter();
                impl_iter.zip(trait_iter)
                         .zip(impl_m_iter)
                         .zip(trait_m_iter)
                         .filter_map(|(((impl_arg_ty, trait_arg_ty), impl_arg), trait_arg)| {
                             match infcx.sub_types(true, &cause, trait_arg_ty, impl_arg_ty) {
                                 Ok(_) => None,
                                 Err(_) => Some((impl_arg.span, Some(trait_arg.span))),
                             }
                         })
                         .next()
                         .unwrap_or_else(|| {
                             if infcx.sub_types(false, &cause, impl_sig.output(),
                                                trait_sig.output())
                                     .is_err() {
                                         (impl_m_output.span(), Some(trait_m_output.span()))
                                     } else {
                                         (cause.span, tcx.map.span_if_local(trait_m.def_id))
                                     }
                         })
            } else {
                (cause.span, tcx.map.span_if_local(trait_m.def_id))
            }
        }
        _ => (cause.span, tcx.map.span_if_local(trait_m.def_id)),
    }
}

fn compare_self_type<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                               impl_m: &ty::AssociatedItem,
                               impl_m_span: Span,
                               trait_m: &ty::AssociatedItem,
                               impl_trait_ref: ty::TraitRef<'tcx>)
                               -> Result<(), ErrorReported>
{
    let tcx = ccx.tcx;
    // Try to give more informative error messages about self typing
    // mismatches.  Note that any mismatch will also be detected
    // below, where we construct a canonical function type that
    // includes the self parameter as a normal parameter.  It's just
    // that the error messages you get out of this code are a bit more
    // inscrutable, particularly for cases where one method has no
    // self.

    let self_string = |method: &ty::AssociatedItem| {
        let untransformed_self_ty = match method.container {
            ty::ImplContainer(_) => impl_trait_ref.self_ty(),
            ty::TraitContainer(_) => tcx.mk_self_type()
        };
        let method_ty = tcx.item_type(method.def_id);
        let self_arg_ty = *method_ty.fn_sig().input(0).skip_binder();
        match ExplicitSelf::determine(untransformed_self_ty, self_arg_ty) {
            ExplicitSelf::ByValue => "self".to_string(),
            ExplicitSelf::ByReference(_, hir::MutImmutable) => "&self".to_string(),
            ExplicitSelf::ByReference(_, hir::MutMutable) => "&mut self".to_string(),
            _ => format!("self: {}", self_arg_ty)
        }
    };

    match (trait_m.method_has_self_argument, impl_m.method_has_self_argument) {
        (false, false) | (true, true) => {}

        (false, true) => {
            let self_descr = self_string(impl_m);
            let mut err = struct_span_err!(tcx.sess,
                                           impl_m_span,
                                           E0185,
                                           "method `{}` has a `{}` declaration in the impl, but \
                                            not in the trait",
                                           trait_m.name,
                                           self_descr);
            err.span_label(impl_m_span, &format!("`{}` used in impl", self_descr));
            if let Some(span) = tcx.map.span_if_local(trait_m.def_id) {
                err.span_label(span, &format!("trait declared without `{}`", self_descr));
            }
            err.emit();
            return Err(ErrorReported);
        }

        (true, false) => {
            let self_descr = self_string(trait_m);
            let mut err = struct_span_err!(tcx.sess,
                                           impl_m_span,
                                           E0186,
                                           "method `{}` has a `{}` declaration in the trait, but \
                                            not in the impl",
                                           trait_m.name,
                                           self_descr);
            err.span_label(impl_m_span,
                           &format!("expected `{}` in impl", self_descr));
            if let Some(span) = tcx.map.span_if_local(trait_m.def_id) {
                err.span_label(span, &format!("`{}` used in trait", self_descr));
            }
            err.emit();
            return Err(ErrorReported);
        }
    }

    Ok(())
}

fn compare_number_of_generics<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                        impl_m: &ty::AssociatedItem,
                                        impl_m_span: Span,
                                        trait_m: &ty::AssociatedItem,
                                        trait_item_span: Option<Span>)
                                        -> Result<(), ErrorReported> {
    let tcx = ccx.tcx;
    let impl_m_generics = tcx.item_generics(impl_m.def_id);
    let trait_m_generics = tcx.item_generics(trait_m.def_id);
    let num_impl_m_type_params = impl_m_generics.types.len();
    let num_trait_m_type_params = trait_m_generics.types.len();
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
            _ => bug!("{:?} is not a method", impl_m),
        };

        let mut err = struct_span_err!(tcx.sess,
                                       span,
                                       E0049,
                                       "method `{}` has {} type parameter{} but its trait \
                                        declaration has {} type parameter{}",
                                       trait_m.name,
                                       num_impl_m_type_params,
                                       if num_impl_m_type_params == 1 { "" } else { "s" },
                                       num_trait_m_type_params,
                                       if num_trait_m_type_params == 1 {
                                           ""
                                       } else {
                                           "s"
                                       });

        let mut suffix = None;

        if let Some(span) = trait_item_span {
            err.span_label(span,
                           &format!("expected {}",
                                    &if num_trait_m_type_params != 1 {
                                        format!("{} type parameters", num_trait_m_type_params)
                                    } else {
                                        format!("{} type parameter", num_trait_m_type_params)
                                    }));
        } else {
            suffix = Some(format!(", expected {}", num_trait_m_type_params));
        }

        err.span_label(span,
                       &format!("found {}{}",
                                &if num_impl_m_type_params != 1 {
                                    format!("{} type parameters", num_impl_m_type_params)
                                } else {
                                    format!("1 type parameter")
                                },
                                suffix.as_ref().map(|s| &s[..]).unwrap_or("")));

        err.emit();

        return Err(ErrorReported);
    }

    Ok(())
}

fn compare_number_of_method_arguments<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                                impl_m: &ty::AssociatedItem,
                                                impl_m_span: Span,
                                                trait_m: &ty::AssociatedItem,
                                                trait_item_span: Option<Span>)
                                                -> Result<(), ErrorReported> {
    let tcx = ccx.tcx;
    let m_fty = |method: &ty::AssociatedItem| {
        match tcx.item_type(method.def_id).sty {
            ty::TyFnDef(_, _, f) => f,
            _ => bug!()
        }
    };
    let impl_m_fty = m_fty(impl_m);
    let trait_m_fty = m_fty(trait_m);
    let trait_number_args = trait_m_fty.sig.inputs().skip_binder().len();
    let impl_number_args = impl_m_fty.sig.inputs().skip_binder().len();
    if trait_number_args != impl_number_args {
        let trait_m_node_id = tcx.map.as_local_node_id(trait_m.def_id);
        let trait_span = if let Some(trait_id) = trait_m_node_id {
            match tcx.map.expect_trait_item(trait_id).node {
                TraitItemKind::Method(ref trait_m_sig, _) => {
                    if let Some(arg) = trait_m_sig.decl.inputs.get(if trait_number_args > 0 {
                        trait_number_args - 1
                    } else {
                        0
                    }) {
                        Some(arg.span)
                    } else {
                        trait_item_span
                    }
                }
                _ => bug!("{:?} is not a method", impl_m),
            }
        } else {
            trait_item_span
        };
        let impl_m_node_id = tcx.map.as_local_node_id(impl_m.def_id).unwrap();
        let impl_span = match tcx.map.expect_impl_item(impl_m_node_id).node {
            ImplItemKind::Method(ref impl_m_sig, _) => {
                if let Some(arg) = impl_m_sig.decl.inputs.get(if impl_number_args > 0 {
                    impl_number_args - 1
                } else {
                    0
                }) {
                    arg.span
                } else {
                    impl_m_span
                }
            }
            _ => bug!("{:?} is not a method", impl_m),
        };
        let mut err = struct_span_err!(tcx.sess,
                                       impl_span,
                                       E0050,
                                       "method `{}` has {} parameter{} but the declaration in \
                                        trait `{}` has {}",
                                       trait_m.name,
                                       impl_number_args,
                                       if impl_number_args == 1 { "" } else { "s" },
                                       tcx.item_path_str(trait_m.def_id),
                                       trait_number_args);
        if let Some(trait_span) = trait_span {
            err.span_label(trait_span,
                           &format!("trait requires {}",
                                    &if trait_number_args != 1 {
                                        format!("{} parameters", trait_number_args)
                                    } else {
                                        format!("{} parameter", trait_number_args)
                                    }));
        }
        err.span_label(impl_span,
                       &format!("expected {}, found {}",
                                &if trait_number_args != 1 {
                                    format!("{} parameters", trait_number_args)
                                } else {
                                    format!("{} parameter", trait_number_args)
                                },
                                impl_number_args));
        err.emit();
        return Err(ErrorReported);
    }

    Ok(())
}

pub fn compare_const_impl<'a, 'tcx>(ccx: &CrateCtxt<'a, 'tcx>,
                                    impl_c: &ty::AssociatedItem,
                                    impl_c_span: Span,
                                    trait_c: &ty::AssociatedItem,
                                    impl_trait_ref: ty::TraitRef<'tcx>) {
    debug!("compare_const_impl(impl_trait_ref={:?})", impl_trait_ref);

    let tcx = ccx.tcx;
    tcx.infer_ctxt((), Reveal::NotSpecializable).enter(|infcx| {
        let mut fulfillment_cx = traits::FulfillmentContext::new();

        // The below is for the most part highly similar to the procedure
        // for methods above. It is simpler in many respects, especially
        // because we shouldn't really have to deal with lifetimes or
        // predicates. In fact some of this should probably be put into
        // shared functions because of DRY violations...
        let trait_to_impl_substs = impl_trait_ref.substs;

        // Create a parameter environment that represents the implementation's
        // method.
        let impl_c_node_id = tcx.map.as_local_node_id(impl_c.def_id).unwrap();
        let impl_param_env = ty::ParameterEnvironment::for_item(tcx, impl_c_node_id);

        // Create mapping from impl to skolemized.
        let impl_to_skol_substs = &impl_param_env.free_substs;

        // Create mapping from trait to skolemized.
        let trait_to_skol_substs = impl_to_skol_substs.rebase_onto(tcx,
                                                                   impl_c.container.id(),
                                                                   trait_to_impl_substs.subst(tcx,
                                                                              impl_to_skol_substs));
        debug!("compare_const_impl: trait_to_skol_substs={:?}",
               trait_to_skol_substs);

        // Compute skolemized form of impl and trait const tys.
        let impl_ty = tcx.item_type(impl_c.def_id).subst(tcx, impl_to_skol_substs);
        let trait_ty = tcx.item_type(trait_c.def_id).subst(tcx, trait_to_skol_substs);
        let mut cause = ObligationCause::misc(impl_c_span, impl_c_node_id);

        let err = infcx.commit_if_ok(|_| {
            // There is no "body" here, so just pass dummy id.
            let impl_ty = assoc::normalize_associated_types_in(&infcx,
                                                               &mut fulfillment_cx,
                                                               impl_c_span,
                                                               ast::CRATE_NODE_ID,
                                                               &impl_ty);

            debug!("compare_const_impl: impl_ty={:?}", impl_ty);

            let trait_ty = assoc::normalize_associated_types_in(&infcx,
                                                                &mut fulfillment_cx,
                                                                impl_c_span,
                                                                ast::CRATE_NODE_ID,
                                                                &trait_ty);

            debug!("compare_const_impl: trait_ty={:?}", trait_ty);

            infcx.sub_types(false, &cause, impl_ty, trait_ty)
                 .map(|InferOk { obligations, value: () }| {
                     for obligation in obligations {
                         fulfillment_cx.register_predicate_obligation(&infcx, obligation);
                     }
                 })
        });

        if let Err(terr) = err {
            debug!("checking associated const for compatibility: impl ty {:?}, trait ty {:?}",
                   impl_ty,
                   trait_ty);

            // Locate the Span containing just the type of the offending impl
            match tcx.map.expect_impl_item(impl_c_node_id).node {
                ImplItemKind::Const(ref ty, _) => cause.span = ty.span,
                _ => bug!("{:?} is not a impl const", impl_c),
            }

            let mut diag = struct_span_err!(tcx.sess,
                                            cause.span,
                                            E0326,
                                            "implemented const `{}` has an incompatible type for \
                                             trait",
                                            trait_c.name);

            // Add a label to the Span containing just the type of the item
            let trait_c_node_id = tcx.map.as_local_node_id(trait_c.def_id).unwrap();
            let trait_c_span = match tcx.map.expect_trait_item(trait_c_node_id).node {
                TraitItemKind::Const(ref ty, _) => ty.span,
                _ => bug!("{:?} is not a trait const", trait_c),
            };

            infcx.note_type_err(&mut diag,
                                &cause,
                                Some((trait_c_span, format!("type in trait"))),
                                Some(infer::ValuePairs::Types(ExpectedFound {
                                    expected: trait_ty,
                                    found: impl_ty,
                                })),
                                &terr);
            diag.emit();
        }
    });
}
