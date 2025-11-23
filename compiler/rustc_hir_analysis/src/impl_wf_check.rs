//! This pass enforces various "well-formedness constraints" on impls.
//! Logically, it is part of wfcheck -- but we do it early so that we
//! can stop compilation afterwards, since part of the trait matching
//! infrastructure gets very grumpy if these conditions don't hold. In
//! particular, if there are type parameters that are not part of the
//! impl, then coherence will report strange inference ambiguity
//! errors; if impls have duplicate items, we get misleading
//! specialization errors. These things can (and probably should) be
//! fixed, but for the moment it's easier to do these checks early.

use std::assert_matches::debug_assert_matches;
use std::ops::ControlFlow;

use min_specialization::check_min_specialization;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::codes::*;
use rustc_errors::{Applicability, Diag};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::{self, Visitor, walk_lifetime};
use rustc_hir::{HirId, LifetimeKind, Path, QPath, Ty, TyKind};
use rustc_middle::hir::nested_filter::All;
use rustc_middle::ty::{self, GenericParamDef, TyCtxt, TypeVisitableExt};
use rustc_span::{ErrorGuaranteed, kw};

use crate::constrained_generic_params as cgp;
use crate::errors::UnconstrainedGenericParameter;
use crate::hir::def::Res;

mod min_specialization;

/// Checks that all the type/lifetime parameters on an impl also
/// appear in the trait ref or self type (or are constrained by a
/// where-clause). These rules are needed to ensure that, given a
/// trait ref like `<T as Trait<U>>`, we can derive the values of all
/// parameters on the impl (which is needed to make specialization
/// possible).
///
/// However, in the case of lifetimes, we only enforce these rules if
/// the lifetime parameter is used in an associated type. This is a
/// concession to backwards compatibility; see comment at the end of
/// the fn for details.
///
/// Example:
///
/// ```rust,ignore (pseudo-Rust)
/// impl<T> Trait<Foo> for Bar { ... }
/// //   ^ T does not appear in `Foo` or `Bar`, error!
///
/// impl<T> Trait<Foo<T>> for Bar { ... }
/// //   ^ T appears in `Foo<T>`, ok.
///
/// impl<T> Trait<Foo> for Bar where Bar: Iterator<Item = T> { ... }
/// //   ^ T is bound to `<Bar as Iterator>::Item`, ok.
///
/// impl<'a> Trait<Foo> for Bar { }
/// //   ^ 'a is unused, but for back-compat we allow it
///
/// impl<'a> Trait<Foo> for Bar { type X = &'a i32; }
/// //   ^ 'a is unused and appears in assoc type, error
/// ```
pub(crate) fn check_impl_wf(
    tcx: TyCtxt<'_>,
    impl_def_id: LocalDefId,
    of_trait: bool,
) -> Result<(), ErrorGuaranteed> {
    debug_assert_matches!(tcx.def_kind(impl_def_id), DefKind::Impl { .. });

    // Check that the args are constrained. We queryfied the check for ty/const params
    // since unconstrained type/const params cause ICEs in projection, so we want to
    // detect those specifically and project those to `TyKind::Error`.
    let mut res = tcx.ensure_ok().enforce_impl_non_lifetime_params_are_constrained(impl_def_id);
    res = res.and(enforce_impl_lifetime_params_are_constrained(tcx, impl_def_id, of_trait));

    if of_trait && tcx.features().min_specialization() {
        res = res.and(check_min_specialization(tcx, impl_def_id));
    }
    res
}

pub(crate) fn enforce_impl_lifetime_params_are_constrained(
    tcx: TyCtxt<'_>,
    impl_def_id: LocalDefId,
    of_trait: bool,
) -> Result<(), ErrorGuaranteed> {
    let impl_self_ty = tcx.type_of(impl_def_id).instantiate_identity();

    // Don't complain about unconstrained type params when self ty isn't known due to errors.
    // (#36836)
    impl_self_ty.error_reported()?;

    let impl_generics = tcx.generics_of(impl_def_id);
    let impl_predicates = tcx.predicates_of(impl_def_id);
    let impl_trait_ref = of_trait.then(|| tcx.impl_trait_ref(impl_def_id).instantiate_identity());

    impl_trait_ref.error_reported()?;

    let mut input_parameters = cgp::parameters_for_impl(tcx, impl_self_ty, impl_trait_ref);
    cgp::identify_constrained_generic_params(
        tcx,
        impl_predicates,
        impl_trait_ref,
        &mut input_parameters,
    );

    // Disallow unconstrained lifetimes, but only if they appear in assoc types.
    let lifetimes_in_associated_types: FxHashSet<_> = tcx
        .associated_item_def_ids(impl_def_id)
        .iter()
        .flat_map(|def_id| {
            let item = tcx.associated_item(def_id);
            match item.kind {
                ty::AssocKind::Type { .. } => {
                    if item.defaultness(tcx).has_value() {
                        cgp::parameters_for(tcx, tcx.type_of(def_id).instantiate_identity(), true)
                    } else {
                        vec![]
                    }
                }
                ty::AssocKind::Fn { .. } | ty::AssocKind::Const { .. } => vec![],
            }
        })
        .collect();

    let mut res = Ok(());
    for param in &impl_generics.own_params {
        match param.kind {
            ty::GenericParamDefKind::Lifetime => {
                // This is a horrible concession to reality. I think it'd be
                // better to just ban unconstrained lifetimes outright, but in
                // practice people do non-hygienic macros like:
                //
                // ```
                // macro_rules! __impl_slice_eq1 {
                //   ($Lhs: ty, $Rhs: ty, $Bound: ident) => {
                //     impl<'a, 'b, A: $Bound, B> PartialEq<$Rhs> for $Lhs where A: PartialEq<B> {
                //        ....
                //     }
                //   }
                // }
                // ```
                //
                // In a concession to backwards compatibility, we continue to
                // permit those, so long as the lifetimes aren't used in
                // associated types. I believe this is sound, because lifetimes
                // used elsewhere are not projected back out.
                let param_lt = cgp::Parameter::from(param.to_early_bound_region_data());
                if lifetimes_in_associated_types.contains(&param_lt)
                    && !input_parameters.contains(&param_lt)
                {
                    let mut diag = tcx.dcx().create_err(UnconstrainedGenericParameter {
                        span: tcx.def_span(param.def_id),
                        param_name: tcx.item_ident(param.def_id),
                        param_def_kind: tcx.def_descr(param.def_id),
                        const_param_note: false,
                        const_param_note2: false,
                    });
                    diag.code(E0207);
                    for p in &impl_generics.own_params {
                        if p.name == kw::UnderscoreLifetime {
                            let span = tcx.def_span(p.def_id);
                            let Ok(snippet) = tcx.sess.source_map().span_to_snippet(span) else {
                                continue;
                            };

                            let (span, sugg) = if &snippet == "'_" {
                                (span, param.name.to_string())
                            } else {
                                (span.shrink_to_hi(), format!("{} ", param.name))
                            };
                            diag.span_suggestion_verbose(
                                span,
                                "consider using the named lifetime here instead of an implicit \
                                 lifetime",
                                sugg,
                                Applicability::MaybeIncorrect,
                            );
                        }
                    }
                    suggest_to_remove_or_use_generic(
                        tcx,
                        &mut diag,
                        impl_def_id,
                        param,
                        "lifetime",
                    );
                    res = Err(diag.emit());
                }
            }
            ty::GenericParamDefKind::Type { .. } | ty::GenericParamDefKind::Const { .. } => {
                // Enforced in `enforce_impl_non_lifetime_params_are_constrained`.
            }
        }
    }
    res
}

pub(crate) fn enforce_impl_non_lifetime_params_are_constrained(
    tcx: TyCtxt<'_>,
    impl_def_id: LocalDefId,
) -> Result<(), ErrorGuaranteed> {
    let impl_self_ty = tcx.type_of(impl_def_id).instantiate_identity();

    // Don't complain about unconstrained type params when self ty isn't known due to errors.
    // (#36836)
    impl_self_ty.error_reported()?;

    let impl_generics = tcx.generics_of(impl_def_id);
    let impl_predicates = tcx.predicates_of(impl_def_id);
    let impl_trait_ref =
        tcx.impl_opt_trait_ref(impl_def_id).map(ty::EarlyBinder::instantiate_identity);

    impl_trait_ref.error_reported()?;

    let mut input_parameters = cgp::parameters_for_impl(tcx, impl_self_ty, impl_trait_ref);
    cgp::identify_constrained_generic_params(
        tcx,
        impl_predicates,
        impl_trait_ref,
        &mut input_parameters,
    );

    let mut res = Ok(());
    for param in &impl_generics.own_params {
        let err = match param.kind {
            // Disallow ANY unconstrained type parameters.
            ty::GenericParamDefKind::Type { .. } => {
                let param_ty = ty::ParamTy::for_def(param);
                !input_parameters.contains(&cgp::Parameter::from(param_ty))
            }
            ty::GenericParamDefKind::Const { .. } => {
                let param_ct = ty::ParamConst::for_def(param);
                !input_parameters.contains(&cgp::Parameter::from(param_ct))
            }
            ty::GenericParamDefKind::Lifetime => {
                // Enforced in `enforce_impl_type_params_are_constrained`.
                false
            }
        };
        if err {
            let const_param_note = matches!(param.kind, ty::GenericParamDefKind::Const { .. });
            let mut diag = tcx.dcx().create_err(UnconstrainedGenericParameter {
                span: tcx.def_span(param.def_id),
                param_name: tcx.item_ident(param.def_id),
                param_def_kind: tcx.def_descr(param.def_id),
                const_param_note,
                const_param_note2: const_param_note,
            });
            diag.code(E0207);
            suggest_to_remove_or_use_generic(tcx, &mut diag, impl_def_id, &param, "type");
            res = Err(diag.emit());
        }
    }
    res
}

/// Use a Visitor to find usages of the type or lifetime parameter
struct ParamUsageVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    /// The `DefId` of the generic parameter we are looking for.
    param_def_id: DefId,
    found: bool,
}

// todo: maybe this can be done more efficiently by only searching for generics OR lifetimes and searching more effectively
impl<'tcx> Visitor<'tcx> for ParamUsageVisitor<'tcx> {
    type NestedFilter = All;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }

    type Result = ControlFlow<()>;

    fn visit_path(&mut self, path: &Path<'tcx>, _id: HirId) -> Self::Result {
        if let Some(res_def_id) = path.res.opt_def_id() {
            if res_def_id == self.param_def_id {
                self.found = true;
                return ControlFlow::Break(()); // Found it, stop visiting.
            }
        }
        // If not found, continue walking down the HIR tree.
        intravisit::walk_path(self, path)
    }

    fn visit_lifetime(&mut self, lifetime: &'tcx rustc_hir::Lifetime) -> Self::Result {
        if let LifetimeKind::Param(id) = lifetime.kind {
            if let Some(local_def_id) = self.param_def_id.as_local() {
                if id == local_def_id {
                    self.found = true;
                    return ControlFlow::Break(()); // Found it, stop visiting.
                }
            }
        }
        walk_lifetime(self, lifetime)
    }
}

fn suggest_to_remove_or_use_generic(
    tcx: TyCtxt<'_>,
    diag: &mut Diag<'_>,
    impl_def_id: LocalDefId,
    param: &GenericParamDef,
    parameter_type: &str,
) {
    let node = tcx.hir_node_by_def_id(impl_def_id);
    let hir_impl = node.expect_item().expect_impl();

    let Some((index, _)) = hir_impl
        .generics
        .params
        .iter()
        .enumerate()
        .find(|(_, par)| par.def_id.to_def_id() == param.def_id)
    else {
        return;
    };

    // get the struct_def_id from the self type
    let Some(struct_def_id) = (|| {
        let ty = hir_impl.self_ty;
        if let TyKind::Path(QPath::Resolved(_, path)) = ty.kind
            && let Res::Def(_, def_id) = path.res
        {
            Some(def_id)
        } else {
            None
        }
    })() else {
        return;
    };
    let generics = tcx.generics_of(struct_def_id);
    // println!("number of struct generics: {}", generics.own_params.len());
    // println!("number of impl generics: {}", hir_impl.generics.params.len());

    // search if the parameter is used in the impl body
    let mut visitor = ParamUsageVisitor { tcx, param_def_id: param.def_id, found: false };

    for item_ref in hir_impl.items {
        let _ = visitor.visit_impl_item_ref(item_ref);
        if visitor.found {
            break;
        }
    }

    let is_param_used = visitor.found;

    // Suggestion for removing the type parameter.
    let mut suggestions = vec![];
    if !is_param_used {
        // Only suggest removing it if it's not used anywhere.
        suggestions.push(vec![(hir_impl.generics.span_for_param_removal(index), String::new())]);
    }

    // Suggestion for making use of the type parameter.
    if let Some(path) = extract_ty_as_path(hir_impl.self_ty) {
        let seg = path.segments.last().unwrap();
        if let Some(args) = seg.args {
            suggestions
                .push(vec![(args.span().unwrap().shrink_to_hi(), format!(", {}", param.name))]);
        } else {
            suggestions.push(vec![(seg.ident.span.shrink_to_hi(), format!("<{}>", param.name))]);
        }
    }

    let msg = if is_param_used {
        // If it's used, the only valid fix is to constrain it.
        format!("make use of the {} parameter `{}` in the `self` type", parameter_type, param.name)
    } else {
        format!(
            "either remove the unused {} parameter `{}`, or make use of it",
            parameter_type, param.name
        )
    };

    diag.multipart_suggestions(msg, suggestions, Applicability::MaybeIncorrect);
}

fn extract_ty_as_path<'hir>(ty: &Ty<'hir>) -> Option<&'hir Path<'hir>> {
    match ty.kind {
        TyKind::Path(QPath::Resolved(_, path)) => Some(path),
        TyKind::Slice(ty) | TyKind::Array(ty, _) => extract_ty_as_path(ty),
        TyKind::Ptr(ty) | TyKind::Ref(_, ty) => extract_ty_as_path(ty.ty),
        _ => None,
    }
}
