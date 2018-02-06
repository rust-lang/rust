// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Error Reporting for Anonymous Region Lifetime Errors
//! where both the regions are anonymous.

use infer::error_reporting::nice_region_error::NiceRegionError;
use infer::error_reporting::nice_region_error::util::AnonymousArgInfo;
use util::common::ErrorReported;

impl<'a, 'gcx, 'tcx> NiceRegionError<'a, 'gcx, 'tcx> {
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
    /// ````
    ///
    /// It will later be extended to trait objects.
    pub(super) fn try_report_anon_anon_conflict(&self) -> Option<ErrorReported> {
        let NiceRegionError { span, sub, sup, .. } = *self;

        // Determine whether the sub and sup consist of both anonymous (elided) regions.
        let anon_reg_sup = self.is_suitable_region(sup)?;

        let anon_reg_sub = self.is_suitable_region(sub)?;
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

        let span_label_var1 = if let Some(simple_name) = anon_arg_sup.pat.simple_name() {
            format!(" from `{}`", simple_name)
        } else {
            format!("")
        };

        let span_label_var2 = if let Some(simple_name) = anon_arg_sub.pat.simple_name() {
            format!(" into `{}`", simple_name)
        } else {
            format!("")
        };


        let (span_1, span_2, main_label, span_label) = match (sup_is_ret_type, sub_is_ret_type) {
            (None, None) => {
                let (main_label_1, span_label_1) = if ty_sup == ty_sub {
                    (
                        format!("this type is declared with multiple lifetimes..."),
                        format!(
                            "...but data{} flows{} here",
                            format!(" with one lifetime"),
                            format!(" into the other")
                        ),
                    )
                } else {
                    (
                        format!("these two types are declared with different lifetimes..."),
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
                format!(
                    "this parameter and the return type are declared \
                     with different lifetimes...",
                ),
                format!("...but data{} is returned here", span_label_var1),
            ),
            (_, Some(ret_span)) => (
                ty_sup.span,
                ret_span,
                format!(
                    "this parameter and the return type are declared \
                     with different lifetimes...",
                ),
                format!("...but data{} is returned here", span_label_var1),
            ),
        };


        struct_span_err!(self.tcx.sess, span, E0623, "lifetime mismatch")
            .span_label(span_1, main_label)
            .span_label(span_2, format!(""))
            .span_label(span, span_label)
            .emit();
        return Some(ErrorReported);
    }
}
