//! Error Reporting for Anonymous Region Lifetime Errors
//! where both the regions are anonymous.

use crate::infer::error_reporting::nice_region_error::find_anon_type::find_anon_type;
use crate::infer::error_reporting::nice_region_error::util::AnonymousParamInfo;
use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use crate::infer::SubregionOrigin;

use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder, ErrorReported};
use rustc_hir as hir;
use rustc_hir::{GenericParamKind, Ty};
use rustc_middle::ty::Region;

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    /// Print the error message for lifetime errors when both the concerned regions are anonymous.
    ///
    /// Consider a case where we have
    ///
    /// ```no_run
    /// fn foo(x: &mut Vec<&u8>, y: &u8) {
    ///     x.push(y);
    /// }
    /// ```
    ///
    /// The example gives
    ///
    /// ```text
    /// fn foo(x: &mut Vec<&u8>, y: &u8) {
    ///                    ---      --- these references are declared with different lifetimes...
    ///     x.push(y);
    ///     ^ ...but data from `y` flows into `x` here
    /// ```
    ///
    /// It has been extended for the case of structs too.
    ///
    /// Consider the example
    ///
    /// ```no_run
    /// struct Ref<'a> { x: &'a u32 }
    /// ```
    ///
    /// ```text
    /// fn foo(mut x: Vec<Ref>, y: Ref) {
    ///                   ---      --- these structs are declared with different lifetimes...
    ///     x.push(y);
    ///     ^ ...but data from `y` flows into `x` here
    /// }
    /// ```
    ///
    /// It will later be extended to trait objects.
    pub(super) fn try_report_anon_anon_conflict(&self) -> Option<ErrorReported> {
        let (span, sub, sup) = self.regions()?;

        if let Some(RegionResolutionError::ConcreteFailure(
            SubregionOrigin::ReferenceOutlivesReferent(..),
            ..,
        )) = self.error
        {
            // This error doesn't make much sense in this case.
            return None;
        }

        // Determine whether the sub and sup consist of both anonymous (elided) regions.
        let anon_reg_sup = self.tcx().is_suitable_region(sup)?;

        let anon_reg_sub = self.tcx().is_suitable_region(sub)?;
        let scope_def_id_sup = anon_reg_sup.def_id;
        let bregion_sup = anon_reg_sup.boundregion;
        let scope_def_id_sub = anon_reg_sub.def_id;
        let bregion_sub = anon_reg_sub.boundregion;

        let ty_sup = find_anon_type(self.tcx(), sup, &bregion_sup)?;

        let ty_sub = find_anon_type(self.tcx(), sub, &bregion_sub)?;

        debug!(
            "try_report_anon_anon_conflict: found_param1={:?} sup={:?} br1={:?}",
            ty_sub, sup, bregion_sup
        );
        debug!(
            "try_report_anon_anon_conflict: found_param2={:?} sub={:?} br2={:?}",
            ty_sup, sub, bregion_sub
        );

        let (ty_sup, ty_fndecl_sup) = ty_sup;
        let (ty_sub, ty_fndecl_sub) = ty_sub;

        let AnonymousParamInfo { param: anon_param_sup, .. } =
            self.find_param_with_region(sup, sup)?;
        let AnonymousParamInfo { param: anon_param_sub, .. } =
            self.find_param_with_region(sub, sub)?;

        let sup_is_ret_type =
            self.is_return_type_anon(scope_def_id_sup, bregion_sup, ty_fndecl_sup);
        let sub_is_ret_type =
            self.is_return_type_anon(scope_def_id_sub, bregion_sub, ty_fndecl_sub);

        let span_label_var1 = match anon_param_sup.pat.simple_ident() {
            Some(simple_ident) => format!(" from `{}`", simple_ident),
            None => String::new(),
        };

        let span_label_var2 = match anon_param_sub.pat.simple_ident() {
            Some(simple_ident) => format!(" into `{}`", simple_ident),
            None => String::new(),
        };

        debug!(
            "try_report_anon_anon_conflict: sub_is_ret_type={:?} sup_is_ret_type={:?}",
            sub_is_ret_type, sup_is_ret_type
        );

        let mut err = struct_span_err!(self.tcx().sess, span, E0623, "lifetime mismatch");

        match (sup_is_ret_type, sub_is_ret_type) {
            (ret_capture @ Some(ret_span), _) | (_, ret_capture @ Some(ret_span)) => {
                let param_span =
                    if sup_is_ret_type == ret_capture { ty_sub.span } else { ty_sup.span };

                err.span_label(
                    param_span,
                    "this parameter and the return type are declared with different lifetimes...",
                );
                err.span_label(ret_span, "");
                err.span_label(span, format!("...but data{} is returned here", span_label_var1));
            }

            (None, None) => {
                if ty_sup.hir_id == ty_sub.hir_id {
                    err.span_label(ty_sup.span, "this type is declared with multiple lifetimes...");
                    err.span_label(ty_sub.span, "");
                    err.span_label(span, "...but data with one lifetime flows into the other here");
                } else {
                    err.span_label(
                        ty_sup.span,
                        "these two types are declared with different lifetimes...",
                    );
                    err.span_label(ty_sub.span, "");
                    err.span_label(
                        span,
                        format!("...but data{} flows{} here", span_label_var1, span_label_var2),
                    );
                }
            }
        }

        self.suggest_adding_lifetime_params(sub, ty_sup, ty_sub, &mut err);

        err.emit();
        Some(ErrorReported)
    }

    fn suggest_adding_lifetime_params(
        &self,
        sub: Region<'tcx>,
        ty_sup: &Ty<'_>,
        ty_sub: &Ty<'_>,
        err: &mut DiagnosticBuilder<'_>,
    ) {
        if let (
            hir::Ty { kind: hir::TyKind::Rptr(lifetime_sub, _), .. },
            hir::Ty { kind: hir::TyKind::Rptr(lifetime_sup, _), .. },
        ) = (ty_sub, ty_sup)
        {
            if lifetime_sub.name.is_elided() && lifetime_sup.name.is_elided() {
                if let Some(anon_reg) = self.tcx().is_suitable_region(sub) {
                    let hir_id = self.tcx().hir().local_def_id_to_hir_id(anon_reg.def_id);
                    if let hir::Node::Item(&hir::Item {
                        kind: hir::ItemKind::Fn(_, ref generics, ..),
                        ..
                    }) = self.tcx().hir().get(hir_id)
                    {
                        let (suggestion_param_name, introduce_new) = generics
                            .params
                            .iter()
                            .find(|p| matches!(p.kind, GenericParamKind::Lifetime { .. }))
                            .and_then(|p| self.tcx().sess.source_map().span_to_snippet(p.span).ok())
                            .map(|name| (name, false))
                            .unwrap_or_else(|| ("'a".to_string(), true));

                        let mut suggestions = vec![
                            if let hir::LifetimeName::Underscore = lifetime_sub.name {
                                (lifetime_sub.span, suggestion_param_name.clone())
                            } else {
                                (
                                    lifetime_sub.span.shrink_to_hi(),
                                    suggestion_param_name.clone() + " ",
                                )
                            },
                            if let hir::LifetimeName::Underscore = lifetime_sup.name {
                                (lifetime_sup.span, suggestion_param_name.clone())
                            } else {
                                (
                                    lifetime_sup.span.shrink_to_hi(),
                                    suggestion_param_name.clone() + " ",
                                )
                            },
                        ];

                        if introduce_new {
                            let new_param_suggestion = match &generics.params {
                                [] => (generics.span, format!("<{}>", suggestion_param_name)),
                                [first, ..] => (
                                    first.span.shrink_to_lo(),
                                    format!("{}, ", suggestion_param_name),
                                ),
                            };

                            suggestions.push(new_param_suggestion);
                        }

                        err.multipart_suggestion(
                            "consider introducing a named lifetime parameter",
                            suggestions,
                            Applicability::MaybeIncorrect,
                        );
                        err.note(
                            "each elided lifetime in input position becomes a distinct lifetime",
                        );
                    }
                }
            }
        }
    }
}
