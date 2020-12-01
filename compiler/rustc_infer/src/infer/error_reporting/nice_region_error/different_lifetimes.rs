//! Error Reporting for Anonymous Region Lifetime Errors
//! where both the regions are anonymous.

use crate::infer::error_reporting::nice_region_error::util::AnonymousParamInfo;
use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use crate::infer::SubregionOrigin;

use rustc_errors::{struct_span_err, ErrorReported};

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

        let ty_sup = self.find_anon_type(sup, &bregion_sup)?;

        let ty_sub = self.find_anon_type(sub, &bregion_sub)?;

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

        let (span_1, span_2, main_label, span_label, future_return_type) =
            match (sup_is_ret_type, sub_is_ret_type) {
                (None, None) => {
                    let (main_label_1, span_label_1) = if ty_sup.hir_id == ty_sub.hir_id {
                        (
                            "this type is declared with multiple lifetimes...".to_owned(),
                            "...but data with one lifetime flows into the other here".to_owned(),
                        )
                    } else {
                        (
                            "these two types are declared with different lifetimes...".to_owned(),
                            format!("...but data{} flows{} here", span_label_var1, span_label_var2),
                        )
                    };
                    (ty_sup.span, ty_sub.span, main_label_1, span_label_1, None)
                }

                (Some(ret_span), _) => {
                    let sup_future = self.future_return_type(scope_def_id_sup);
                    let (return_type, action) = if let Some(_) = sup_future {
                        ("returned future", "held across an await point")
                    } else {
                        ("return type", "returned")
                    };

                    (
                        ty_sub.span,
                        ret_span,
                        format!(
                            "this parameter and the {} are declared with different lifetimes...",
                            return_type
                        ),
                        format!("...but data{} is {} here", span_label_var1, action),
                        sup_future,
                    )
                }
                (_, Some(ret_span)) => {
                    let sub_future = self.future_return_type(scope_def_id_sub);
                    let (return_type, action) = if let Some(_) = sub_future {
                        ("returned future", "held across an await point")
                    } else {
                        ("return type", "returned")
                    };

                    (
                        ty_sup.span,
                        ret_span,
                        format!(
                            "this parameter and the {} are declared with different lifetimes...",
                            return_type
                        ),
                        format!("...but data{} is {} here", span_label_var1, action),
                        sub_future,
                    )
                }
            };

        let mut e = struct_span_err!(self.tcx().sess, span, E0623, "lifetime mismatch");

        e.span_label(span_1, main_label);
        e.span_label(span_2, String::new());
        e.span_label(span, span_label);

        if let Some(t) = future_return_type {
            let snip = self
                .tcx()
                .sess
                .source_map()
                .span_to_snippet(t.span)
                .ok()
                .and_then(|s| match (&t.kind, s.as_str()) {
                    (rustc_hir::TyKind::Tup(&[]), "") => Some("()".to_string()),
                    (_, "") => None,
                    _ => Some(s),
                })
                .unwrap_or("{unnamed_type}".to_string());

            e.span_label(
                t.span,
                &format!("this `async fn` implicitly returns an `impl Future<Output = {}>`", snip),
            );
        }
        e.emit();
        Some(ErrorReported)
    }
}
