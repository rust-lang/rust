// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Error Reporting for static impl Traits.

use infer::error_reporting::nice_region_error::NiceRegionError;
use infer::lexical_region_resolve::RegionResolutionError;
use ty::{BoundRegion, FreeRegion, RegionKind};
use util::common::ErrorReported;

impl<'a, 'gcx, 'tcx> NiceRegionError<'a, 'gcx, 'tcx> {
    /// Print the error message for lifetime errors when the return type is a static impl Trait.
    pub(super) fn try_report_static_impl_trait(&self) -> Option<ErrorReported> {
        if let Some(ref error) = self.error {
            match error.clone() {
                RegionResolutionError::SubSupConflict(
                    var_origin,
                    sub_origin,
                    sub_r,
                    sup_origin,
                    sup_r,
                ) => {
                    let anon_reg_sup = self.is_suitable_region(sup_r)?;
                    if sub_r == &RegionKind::ReStatic &&
                        self.is_return_type_impl_trait(anon_reg_sup.def_id)
                    {
                        let sp = var_origin.span();
                        let return_sp = sub_origin.span();
                        let mut err = self.tcx.sess.struct_span_err(
                            sp,
                            "cannot infer an appropriate lifetime",
                        );
                        err.span_label(
                            return_sp,
                            "this return type evaluates to the `'static` lifetime...",
                        );
                        err.span_label(
                            sup_origin.span(),
                            "...but this borrow...",
                        );

                        let (lifetime, lt_sp_opt) = self.tcx.msg_span_from_free_region(sup_r);
                        if let Some(lifetime_sp) = lt_sp_opt {
                            err.span_note(
                                lifetime_sp,
                                &format!("...can't outlive {}", lifetime),
                            );
                        }

                        let lifetime_name = match sup_r {
                            RegionKind::ReFree(FreeRegion {
                                bound_region: BoundRegion::BrNamed(_, ref name), ..
                            }) => name.to_string(),
                            _ => "'_".to_owned(),
                        };
                        if let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(return_sp) {
                            err.span_suggestion(
                                return_sp,
                                &format!(
                                    "you can add a constraint to the return type to make it last \
                                     less than `'static` and match {}",
                                    lifetime,
                                ),
                                format!("{} + {}", snippet, lifetime_name),
                            );
                        }
                        err.emit();
                        return Some(ErrorReported);
                    }
                }
                _ => {}
            }
        }
        None
    }
}
