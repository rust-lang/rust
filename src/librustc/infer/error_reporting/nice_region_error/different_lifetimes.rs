//! Error Reporting for Anonymous Region Lifetime Errors
//! where both the regions are anonymous.

use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::error_reporting::nice_region_error::util::AnonymousArgInfo;
use crate::util::common::ErrorReported;

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
        let (span, sub, sup) = self.get_regions();

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
            "try_report_anon_anon_conflict: found_arg1={:?} sup={:?} br1={:?}",
            ty_sub,
            sup,
            bregion_sup
        );
        debug!(
            "try_report_anon_anon_conflict: found_arg2={:?} sub={:?} br2={:?}",
            ty_sup,
            sub,
            bregion_sub
        );

        let (ty_sup, ty_fndecl_sup) = ty_sup;
        let (ty_sub, ty_fndecl_sub) = ty_sub;

        let AnonymousArgInfo {
            arg: anon_arg_sup, ..
        } = self.find_arg_with_region(sup, sup)?;
        let AnonymousArgInfo {
            arg: anon_arg_sub, ..
        } = self.find_arg_with_region(sub, sub)?;

        let sup_is_ret_type =
            self.is_return_type_anon(scope_def_id_sup, bregion_sup, ty_fndecl_sup);
        let sub_is_ret_type =
            self.is_return_type_anon(scope_def_id_sub, bregion_sub, ty_fndecl_sub);

        let span_label_var1 = match anon_arg_sup.pat.simple_ident() {
            Some(simple_ident) => format!(" from `{}`", simple_ident),
            None => String::new(),
        };

        let span_label_var2 = match anon_arg_sub.pat.simple_ident() {
            Some(simple_ident) => format!(" into `{}`", simple_ident),
            None => String::new(),
        };

        let (span_1, span_2, main_label, span_label) = match (sup_is_ret_type, sub_is_ret_type) {
            (None, None) => {
                let (main_label_1, span_label_1) = if ty_sup.hir_id == ty_sub.hir_id {
                    (
                        "this type is declared with multiple lifetimes...".to_owned(),
                        "...but data with one lifetime flows into the other here".to_owned()
                    )
                } else {
                    (
                        "these two types are declared with different lifetimes...".to_owned(),
                        format!(
                            "...but data{} flows{} here",
                            span_label_var1,
                            span_label_var2
                        ),
                    )
                };
                (ty_sup.span, ty_sub.span, main_label_1, span_label_1)
            }

            (Some(ret_span), _) => (
                ty_sub.span,
                ret_span,
                "this parameter and the return type are declared \
                 with different lifetimes...".to_owned()
                ,
                format!("...but data{} is returned here", span_label_var1),
            ),
            (_, Some(ret_span)) => (
                ty_sup.span,
                ret_span,
                "this parameter and the return type are declared \
                 with different lifetimes...".to_owned()
                ,
                format!("...but data{} is returned here", span_label_var1),
            ),
        };


        struct_span_err!(self.tcx().sess, span, E0623, "lifetime mismatch")
            .span_label(span_1, main_label)
            .span_label(span_2, String::new())
            .span_label(span, span_label)
            .emit();
        return Some(ErrorReported);
    }
}
