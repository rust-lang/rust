//! Error Reporting for static impl Traits.

use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use rustc_errors::{Applicability, ErrorReported};
use rustc_hir::{GenericBound, ItemKind, Lifetime, LifetimeName, TyKind};
use rustc_middle::ty::RegionKind;

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    /// Print the error message for lifetime errors when the return type is a static impl Trait.
    pub(super) fn try_report_static_impl_trait(&self) -> Option<ErrorReported> {
        debug!("try_report_static_impl_trait(error={:?})", self.error);
        if let Some(ref error) = self.error {
            if let RegionResolutionError::SubSupConflict(
                _,
                var_origin,
                sub_origin,
                sub_r,
                sup_origin,
                sup_r,
            ) = error
            {
                debug!(
                    "try_report_static_impl_trait(var={:?}, sub={:?} {:?} sup={:?} {:?})",
                    var_origin, sub_origin, sub_r, sup_origin, sup_r
                );
                let anon_reg_sup = self.tcx().is_suitable_region(sup_r)?;
                debug!("try_report_static_impl_trait: anon_reg_sup={:?}", anon_reg_sup);
                let fn_return = self.tcx().return_type_impl_or_dyn_trait(anon_reg_sup.def_id)?;
                debug!("try_report_static_impl_trait: fn_return={:?}", fn_return);
                if **sub_r == RegionKind::ReStatic {
                    let sp = var_origin.span();
                    let return_sp = sub_origin.span();
                    let param_info = self.find_param_with_region(sup_r, sub_r)?;
                    let (lifetime_name, lifetime) = if sup_r.has_name() {
                        (sup_r.to_string(), format!("lifetime `{}`", sup_r))
                    } else {
                        ("'_".to_owned(), "the anonymous lifetime `'_`".to_string())
                    };
                    let mut err =
                        self.tcx().sess.struct_span_err(sp, "cannot infer an appropriate lifetime");
                    err.span_label(
                        param_info.param_ty_span,
                        &format!("this data with {}...", lifetime),
                    );
                    debug!("try_report_static_impl_trait: param_info={:?}", param_info);

                    // We try to make the output have fewer overlapping spans if possible.
                    if (sp == sup_origin.span() || !return_sp.overlaps(sup_origin.span()))
                        && sup_origin.span() != return_sp
                    {
                        // FIXME: account for `async fn` like in `async-await/issues/issue-62097.rs`

                        // Customize the spans and labels depending on their relative order so
                        // that split sentences flow correctly.
                        if sup_origin.span().overlaps(return_sp) && sp == sup_origin.span() {
                            // Avoid the following:
                            //
                            // error: cannot infer an appropriate lifetime
                            //   --> $DIR/must_outlive_least_region_or_bound.rs:18:50
                            //    |
                            // LL | fn foo(x: &i32) -> Box<dyn Debug> { Box::new(x) }
                            //    |           ----                      ---------^-
                            //    |           |                         |        |
                            //    |           |                         |   ...and is captured here
                            //    |           |           ...is required to be `'static` by this...
                            //    |           this data with the anonymous lifetime `'_`...
                            //
                            // and instead show:
                            //
                            // error: cannot infer an appropriate lifetime
                            //   --> $DIR/must_outlive_least_region_or_bound.rs:18:50
                            //    |
                            // LL | fn foo(x: &i32) -> Box<dyn Debug> { Box::new(x) }
                            //    |           ----                               ^ ...is captured here with a `'static` requirement
                            //    |           |
                            //    |           this data with the anonymous lifetime `'_`...
                            //    |
                            err.span_label(
                                sup_origin.span(),
                                "...is captured here with a `'static` requirement",
                            );
                        } else if sup_origin.span() <= return_sp {
                            err.span_label(sup_origin.span(), "...is captured here...");
                            err.span_label(return_sp, "...and required to be `'static` by this");
                        } else {
                            err.span_label(return_sp, "...is required to be `'static` by this...");
                            err.span_label(sup_origin.span(), "...and is captured here");
                        }
                    } else {
                        err.span_label(
                            return_sp,
                            "...is captured and required to be `'static` here",
                        );
                    }

                    // only apply this suggestion onto functions with
                    // explicit non-desugar'able return.
                    if fn_return.span.desugaring_kind().is_none() {
                        // FIXME: account for the need of parens in `&(dyn Trait + '_)`
                        match fn_return.kind {
                            TyKind::Def(item_id, _) => {
                                let item = self.tcx().hir().item(item_id.id);
                                let opaque = if let ItemKind::OpaqueTy(opaque) = &item.kind {
                                    opaque
                                } else {
                                    err.emit();
                                    return Some(ErrorReported);
                                };

                                if let Some(span) = opaque
                                    .bounds
                                    .iter()
                                    .filter_map(|arg| match arg {
                                        GenericBound::Outlives(Lifetime {
                                            name: LifetimeName::Static,
                                            span,
                                            ..
                                        }) => Some(*span),
                                        _ => None,
                                    })
                                    .next()
                                {
                                    err.span_suggestion_verbose(
                                        span,
                                        &format!(
                                            "consider changing the `impl Trait`'s explicit \
                                             `'static` bound to {}",
                                            lifetime,
                                        ),
                                        lifetime_name,
                                        Applicability::MaybeIncorrect,
                                    );
                                    err.span_suggestion_verbose(
                                        param_info.param_ty_span,
                                        "alternatively, set an explicit `'static` lifetime to \
                                         this parameter",
                                        param_info.param_ty.to_string(),
                                        Applicability::MaybeIncorrect,
                                    );
                                } else {
                                    err.span_suggestion_verbose(
                                        fn_return.span.shrink_to_hi(),
                                        &format!(
                                            "to permit non-static references in an `impl Trait` \
                                             value, you can add an explicit bound for {}",
                                            lifetime,
                                        ),
                                        format!(" + {}", lifetime_name),
                                        Applicability::MaybeIncorrect,
                                    );
                                };
                            }
                            TyKind::TraitObject(_, lt) => match lt.name {
                                LifetimeName::ImplicitObjectLifetimeDefault => {
                                    err.span_suggestion_verbose(
                                        fn_return.span.shrink_to_hi(),
                                        &format!(
                                            "to permit non-static references in a trait object \
                                             value, you can add an explicit bound for {}",
                                            lifetime,
                                        ),
                                        format!(" + {}", lifetime_name),
                                        Applicability::MaybeIncorrect,
                                    );
                                }
                                _ => {
                                    err.span_suggestion_verbose(
                                        lt.span,
                                        &format!(
                                            "consider changing the trait object's explicit \
                                             `'static` bound to {}",
                                            lifetime,
                                        ),
                                        lifetime_name,
                                        Applicability::MaybeIncorrect,
                                    );
                                    err.span_suggestion_verbose(
                                        param_info.param_ty_span,
                                        &format!(
                                            "alternatively, set an explicit `'static` lifetime \
                                             in this parameter",
                                        ),
                                        param_info.param_ty.to_string(),
                                        Applicability::MaybeIncorrect,
                                    );
                                }
                            },
                            _ => {}
                        }
                    }
                    err.emit();
                    return Some(ErrorReported);
                }
            }
        }
        None
    }
}
