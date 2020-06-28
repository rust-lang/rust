//! Error Reporting for static impl Traits.

use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use rustc_errors::{struct_span_err, Applicability, ErrorReported};
use rustc_hir::{GenericBound, ItemKind, Lifetime, LifetimeName, TyKind};
use rustc_middle::ty::RegionKind;

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    /// Print the error message for lifetime errors when the return type is a static impl Trait.
    pub(super) fn try_report_static_impl_trait(&self) -> Option<ErrorReported> {
        debug!("try_report_static_impl_trait(error={:?})", self.error);
        if let Some(RegionResolutionError::SubSupConflict(
            _,
            var_origin,
            ref sub_origin,
            sub_r,
            ref sup_origin,
            sup_r,
        )) = self.error
        {
            debug!(
                "try_report_static_impl_trait(var={:?}, sub={:?} {:?} sup={:?} {:?})",
                var_origin, sub_origin, sub_r, sup_origin, sup_r
            );
            let anon_reg_sup = self.tcx().is_suitable_region(sup_r)?;
            debug!("try_report_static_impl_trait: anon_reg_sup={:?}", anon_reg_sup);
            let fn_returns = self.tcx().return_type_impl_or_dyn_traits(anon_reg_sup.def_id);
            if fn_returns.is_empty() {
                return None;
            }
            debug!("try_report_static_impl_trait: fn_return={:?}", fn_returns);
            if *sub_r == RegionKind::ReStatic {
                let sp = var_origin.span();
                let return_sp = sub_origin.span();
                let param_info = self.find_param_with_region(sup_r, sub_r)?;
                let (lifetime_name, lifetime) = if sup_r.has_name() {
                    (sup_r.to_string(), format!("lifetime `{}`", sup_r))
                } else {
                    ("'_".to_owned(), "an anonymous lifetime `'_`".to_string())
                };
                let mut err = struct_span_err!(
                    self.tcx().sess,
                    sp,
                    E0759,
                    "cannot infer an appropriate lifetime"
                );
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
                        //
                        // and instead show:
                        //
                        // error: cannot infer an appropriate lifetime
                        //   --> $DIR/must_outlive_least_region_or_bound.rs:18:50
                        //    |
                        // LL | fn foo(x: &i32) -> Box<dyn Debug> { Box::new(x) }
                        //    |           ----                               ^
                        err.span_label(
                            sup_origin.span(),
                            "...is captured here, requiring it to live as long as `'static`",
                        );
                    } else {
                        err.span_label(sup_origin.span(), "...is captured here...");
                        if return_sp < sup_origin.span() {
                            err.span_note(
                                return_sp,
                                "...and is required to live as long as `'static` here",
                            );
                        } else {
                            err.span_label(
                                return_sp,
                                "...and is required to live as long as `'static` here",
                            );
                        }
                    }
                } else {
                    err.span_label(
                        return_sp,
                        "...is captured and required to live as long as `'static` here",
                    );
                }

                // FIXME: account for the need of parens in `&(dyn Trait + '_)`
                let consider = "consider changing the";
                let declare = "to declare that the";
                let arg = match param_info.param.pat.simple_ident() {
                    Some(simple_ident) => format!("argument `{}`", simple_ident),
                    None => "the argument".to_string(),
                };
                let explicit =
                    format!("you can add an explicit `{}` lifetime bound", lifetime_name);
                let explicit_static =
                    format!("explicit `'static` bound to the lifetime of {}", arg);
                let captures = format!("captures data from {}", arg);
                let add_static_bound =
                    "alternatively, add an explicit `'static` bound to this reference";
                let plus_lt = format!(" + {}", lifetime_name);
                for fn_return in fn_returns {
                    if fn_return.span.desugaring_kind().is_some() {
                        // Skip `async` desugaring `impl Future`.
                        continue;
                    }
                    match fn_return.kind {
                        TyKind::OpaqueDef(item_id, _) => {
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
                                    &format!("{} `impl Trait`'s {}", consider, explicit_static),
                                    lifetime_name.clone(),
                                    Applicability::MaybeIncorrect,
                                );
                                err.span_suggestion_verbose(
                                    param_info.param_ty_span,
                                    add_static_bound,
                                    param_info.param_ty.to_string(),
                                    Applicability::MaybeIncorrect,
                                );
                            } else if let Some(_) = opaque
                                .bounds
                                .iter()
                                .filter_map(|arg| match arg {
                                    GenericBound::Outlives(Lifetime { name, span, .. })
                                        if name.ident().to_string() == lifetime_name =>
                                    {
                                        Some(*span)
                                    }
                                    _ => None,
                                })
                                .next()
                            {
                            } else {
                                err.span_suggestion_verbose(
                                    fn_return.span.shrink_to_hi(),
                                    &format!(
                                        "{declare} `impl Trait` {captures}, {explicit}",
                                        declare = declare,
                                        captures = captures,
                                        explicit = explicit,
                                    ),
                                    plus_lt.clone(),
                                    Applicability::MaybeIncorrect,
                                );
                            }
                        }
                        TyKind::TraitObject(_, lt) => match lt.name {
                            LifetimeName::ImplicitObjectLifetimeDefault => {
                                err.span_suggestion_verbose(
                                    fn_return.span.shrink_to_hi(),
                                    &format!(
                                        "{declare} trait object {captures}, {explicit}",
                                        declare = declare,
                                        captures = captures,
                                        explicit = explicit,
                                    ),
                                    plus_lt.clone(),
                                    Applicability::MaybeIncorrect,
                                );
                            }
                            name if name.ident().to_string() != lifetime_name => {
                                // With this check we avoid suggesting redundant bounds. This
                                // would happen if there are nested impl/dyn traits and only
                                // one of them has the bound we'd suggest already there, like
                                // in `impl Foo<X = dyn Bar> + '_`.
                                err.span_suggestion_verbose(
                                    lt.span,
                                    &format!("{} trait object's {}", consider, explicit_static),
                                    lifetime_name.clone(),
                                    Applicability::MaybeIncorrect,
                                );
                                err.span_suggestion_verbose(
                                    param_info.param_ty_span,
                                    add_static_bound,
                                    param_info.param_ty.to_string(),
                                    Applicability::MaybeIncorrect,
                                );
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }
                err.emit();
                return Some(ErrorReported);
            }
        }
        None
    }
}
