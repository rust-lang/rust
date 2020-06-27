//! Error Reporting for static impl Traits.

use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder, ErrorReported};
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::{
    GenericBound, Item, ItemKind, Lifetime, LifetimeName, Node, Path, PolyTraitRef, TraitRef,
    TyKind,
};
use rustc_middle::ty::{self, RegionKind, Ty, TypeFoldable, TypeVisitor};

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    /// Print the error message for lifetime errors when the return type is a static impl Trait.
    pub(super) fn try_report_static_impl_trait(&self) -> Option<ErrorReported> {
        debug!("try_report_static_impl_trait(error={:?})", self.error);
        let tcx = self.tcx();
        let (var_origin, sub_origin, sub_r, sup_origin, sup_r) = match self.error.as_ref()? {
            RegionResolutionError::SubSupConflict(
                _,
                var_origin,
                sub_origin,
                sub_r,
                sup_origin,
                sup_r,
            ) if **sub_r == RegionKind::ReStatic => {
                (var_origin, sub_origin, sub_r, sup_origin, sup_r)
            }
            _ => return None,
        };
        debug!(
            "try_report_static_impl_trait(var={:?}, sub={:?} {:?} sup={:?} {:?})",
            var_origin, sub_origin, sub_r, sup_origin, sup_r
        );
        let anon_reg_sup = tcx.is_suitable_region(sup_r)?;
        debug!("try_report_static_impl_trait: anon_reg_sup={:?}", anon_reg_sup);
        let sp = var_origin.span();
        let return_sp = sub_origin.span();
        let param = self.find_param_with_region(sup_r, sub_r)?;
        let (lifetime_name, lifetime) = if sup_r.has_name() {
            (sup_r.to_string(), format!("lifetime `{}`", sup_r))
        } else {
            ("'_".to_owned(), "an anonymous lifetime `'_`".to_string())
        };
        let mut err = struct_span_err!(tcx.sess, sp, E0759, "cannot infer an appropriate lifetime");
        err.span_label(param.param_ty_span, &format!("this data with {}...", lifetime));
        debug!("try_report_static_impl_trait: param_info={:?}", param);

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

        self.find_impl_on_dyn_trait(&mut err, param.param_ty);

        let fn_returns = tcx.return_type_impl_or_dyn_traits(anon_reg_sup.def_id);
        debug!("try_report_static_impl_trait: fn_return={:?}", fn_returns);
        // FIXME: account for the need of parens in `&(dyn Trait + '_)`
        let consider = "consider changing the";
        let declare = "to declare that the";
        let arg = match param.param.pat.simple_ident() {
            Some(simple_ident) => format!("argument `{}`", simple_ident),
            None => "the argument".to_string(),
        };
        let explicit = format!("you can add an explicit `{}` lifetime bound", lifetime_name);
        let explicit_static = format!("explicit `'static` bound to the lifetime of {}", arg);
        let captures = format!("captures data from {}", arg);
        let add_static_bound = "alternatively, add an explicit `'static` bound to this reference";
        let plus_lt = format!(" + {}", lifetime_name);
        for fn_return in fn_returns {
            if fn_return.span.desugaring_kind().is_some() {
                // Skip `async` desugaring `impl Future`.
                continue;
            }
            match fn_return.kind {
                TyKind::OpaqueDef(item_id, _) => {
                    let item = tcx.hir().item(item_id.id);
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
                            param.param_ty_span,
                            add_static_bound,
                            param.param_ty.to_string(),
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
                            param.param_ty_span,
                            add_static_bound,
                            param.param_ty.to_string(),
                            Applicability::MaybeIncorrect,
                        );
                    }
                    _ => {}
                },
                _ => {}
            }
        }
        err.emit();
        Some(ErrorReported)
    }

    /// When we call a method coming from an `impl Foo for dyn Bar`, `dyn Bar` introduces a default
    /// `'static` obligation. Find `impl` blocks that are implemented
    fn find_impl_on_dyn_trait(&self, err: &mut DiagnosticBuilder<'_>, ty: Ty<'_>) -> bool {
        let tcx = self.tcx();

        // Find the trait object types in the argument.
        let mut v = TraitObjectVisitor(vec![]);
        v.visit_ty(ty);
        debug!("TraitObjectVisitor {:?}", v.0);

        // Find all the `impl`s in the local scope that can be called on the type parameter.
        // FIXME: this doesn't find `impl dyn Trait { /**/ }`.
        let impl_self_tys = tcx
            .all_traits(LOCAL_CRATE)
            .iter()
            .flat_map(|trait_did| tcx.hir().trait_impls(*trait_did))
            .filter_map(|impl_node| {
                let impl_did = tcx.hir().local_def_id(*impl_node);
                if let Some(Node::Item(Item { kind: ItemKind::Impl { self_ty, .. }, .. })) =
                    tcx.hir().get_if_local(impl_did.to_def_id())
                {
                    Some(self_ty)
                } else {
                    None
                }
            });
        let mut suggested = false;
        for self_ty in impl_self_tys {
            if let TyKind::TraitObject(
                poly_trait_refs,
                Lifetime { name: LifetimeName::ImplicitObjectLifetimeDefault, .. },
            ) = self_ty.kind
            {
                for p in poly_trait_refs {
                    if let PolyTraitRef {
                        trait_ref:
                            TraitRef { path: Path { res: Res::Def(DefKind::Trait, did), .. }, .. },
                        ..
                    } = p
                    {
                        for found_did in &v.0 {
                            if did == found_did {
                                // We've found an `impl Foo for dyn Bar {}`.
                                // FIXME: we should change this so it also works for
                                // `impl Foo for Box<dyn Bar> {}`.
                                err.span_suggestion_verbose(
                                    self_ty.span.shrink_to_hi(),
                                    "this `impl` introduces an implicit `'static` requirement, \
                                     consider changing it",
                                    " + '_".to_string(),
                                    Applicability::MaybeIncorrect,
                                );
                                suggested = true;
                            }
                        }
                    }
                }
                err.emit();
                return Some(ErrorReported);
            }
        }
        suggested
    }
}

/// Collect all the trait objects in a type that could have received an implicit `'static` lifetime.
struct TraitObjectVisitor(Vec<DefId>);

impl TypeVisitor<'_> for TraitObjectVisitor {
    fn visit_ty(&mut self, t: Ty<'_>) -> bool {
        match t.kind {
            ty::Dynamic(preds, RegionKind::ReStatic) => {
                if let Some(def_id) = preds.principal_def_id() {
                    self.0.push(def_id);
                }
                false
            }
            _ => t.super_visit_with(self),
        }
    }
}
