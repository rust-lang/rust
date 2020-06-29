//! Error Reporting for static impl Traits.

use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use crate::infer::{SubregionOrigin, TypeTrace};
use crate::traits::ObligationCauseCode;
use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder, ErrorReported};
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_hir::intravisit::{walk_ty, ErasedMap, NestedVisitorMap, Visitor};
use rustc_hir::{self as hir, GenericBound, Item, ItemKind, Lifetime, LifetimeName, Node, TyKind};
use rustc_middle::ty::{
    self, AssocItem, AssocItemContainer, RegionKind, Ty, TypeFoldable, TypeVisitor,
};
use rustc_span::{MultiSpan, Span};

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
            RegionResolutionError::ConcreteFailure(
                SubregionOrigin::Subtype(box TypeTrace { cause, .. }),
                sub_r,
                sup_r,
            ) if **sub_r == RegionKind::ReStatic => {
                // This is for the implicit `'static` requirement coming from `impl dyn Trait {}`.
                if let ObligationCauseCode::UnifyReceiver(assoc) = &cause.code {
                    let param = self.find_param_with_region(sup_r, sub_r)?;
                    let lifetime = if sup_r.has_name() {
                        format!("lifetime `{}`", sup_r)
                    } else {
                        "an anonymous lifetime `'_`".to_string()
                    };
                    let mut err = struct_span_err!(
                        tcx.sess,
                        cause.span,
                        E0767,
                        "{} has {} but calling `{}` introduces an implicit `'static` lifetime \
                         requirement",
                        param
                            .param
                            .pat
                            .simple_ident()
                            .map(|s| format!("`{}`", s))
                            .unwrap_or_else(|| "`fn` parameter".to_string()),
                        lifetime,
                        assoc.ident,
                    );
                    err.span_label(param.param_ty_span, &format!("this data with {}...", lifetime));
                    err.span_label(
                        cause.span,
                        &format!(
                            "...is captured and required to live as long as `'static` here \
                             because of an implicit lifetime bound on the {}",
                            match assoc.container {
                                AssocItemContainer::TraitContainer(id) =>
                                    format!("`impl` of `{}`", tcx.def_path_str(id)),
                                AssocItemContainer::ImplContainer(_) =>
                                    "inherent `impl`".to_string(),
                            },
                        ),
                    );
                    if self.find_impl_on_dyn_trait(&mut err, param.param_ty, assoc) {
                        err.emit();
                        return Some(ErrorReported);
                    } else {
                        err.cancel();
                    }
                }
                return None;
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
        let param_name = param
            .param
            .pat
            .simple_ident()
            .map(|s| format!("`{}`", s))
            .unwrap_or_else(|| "`fn` parameter".to_string());
        let mut err = struct_span_err!(
            tcx.sess,
            sp,
            E0759,
            "{} has {} but it needs to satisfy a `'static` lifetime requirement",
            param_name,
            lifetime,
        );
        err.span_label(param.param_ty_span, &format!("this data with {}...", lifetime));
        debug!("try_report_static_impl_trait: param_info={:?}", param);

        let fn_returns = tcx.return_type_impl_or_dyn_traits(anon_reg_sup.def_id);

        let mut postfix = String::new();
        if let SubregionOrigin::Subtype(box TypeTrace { cause, .. }) = &sup_origin {
            if let ObligationCauseCode::UnifyReceiver(assoc) = &cause.code {
                if self.find_impl_on_dyn_trait(&mut err, param.param_ty, assoc)
                    && fn_returns.is_empty()
                {
                    err.code(rustc_errors::error_code!(E0767));
                    err.set_primary_message(&format!(
                        "{} has {} but calling `{}` introduces an implicit `'static` lifetime \
                         requirement",
                        param_name, lifetime, assoc.ident,
                    ));
                    postfix = format!(
                        " because of an implicit lifetime on the {}",
                        match assoc.container {
                            AssocItemContainer::TraitContainer(id) =>
                                format!("`impl` of `{}`", tcx.def_path_str(id)),
                            AssocItemContainer::ImplContainer(_) => "inherent `impl`".to_string(),
                        },
                    );
                }
            }
        }

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
                    &format!(
                        "...is captured here, requiring it to live as long as `'static`{}",
                        postfix
                    ),
                );
            } else {
                err.span_label(sup_origin.span(), "...is captured here...");
                if return_sp < sup_origin.span() {
                    err.span_note(
                        return_sp,
                        &format!("...and is required to live as long as `'static` here{}", postfix),
                    );
                } else {
                    err.span_label(
                        return_sp,
                        &format!("...and is required to live as long as `'static` here{}", postfix),
                    );
                }
            }
        } else {
            err.span_label(
                return_sp,
                &format!(
                    "...is captured and required to live as long as `'static` here{}",
                    postfix
                ),
            );
        }

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
    fn find_impl_on_dyn_trait(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        ty: Ty<'_>,
        assoc: &AssocItem,
    ) -> bool {
        let tcx = self.tcx();
        let mut suggested = false;

        // Find the trait object types in the argument.
        let mut v = TraitObjectVisitor(vec![]);
        v.visit_ty(ty);

        let container_id = match assoc.container {
            // When the obligation comes from an `impl Foo for dyn Bar {}`, we
            // have the `DefId` of the `trait` itself, not the relevant `impl`
            // block. Because of this, we have to look at all the `trait`s
            // available, and filter out all that are not of `Foo` (this `def_id`)
            // and not of `Bar` (the `filter_map` later in this method).
            AssocItemContainer::TraitContainer(def_id) => def_id,

            // When the obligation comes from an `impl dyn Trait {}`, we already
            // have the `DefId` of the relevant `Item`, so we use it directly.
            AssocItemContainer::ImplContainer(def_id) => {
                if let Some(Node::Item(Item { kind: ItemKind::Impl { self_ty, .. }, .. })) =
                    tcx.hir().get_if_local(def_id)
                {
                    for found_did in &v.0 {
                        let mut hir_v = HirTraitObjectVisitor(vec![], *found_did);
                        hir_v.visit_ty(self_ty);
                        if let [span] = &hir_v.0[..] {
                            let mut multi_span: MultiSpan = vec![*span].into();
                            multi_span.push_span_label(
                                *span,
                                "this has an implicit `'static` lifetime requirement".to_string(),
                            );
                            multi_span.push_span_label(
                                assoc.ident.span,
                                "`'static` requirement is introduced when calling this method"
                                    .to_string(),
                            );
                            err.span_note(
                                multi_span,
                                &format!(
                                    "`{}`'s inherent `impl` has a `'static` requirement",
                                    tcx.def_path_str(*found_did),
                                ),
                            );
                            err.span_suggestion_verbose(
                                span.shrink_to_hi(),
                                "consider relaxing the implicit `'static` requirement",
                                " + '_".to_string(),
                                Applicability::MaybeIncorrect,
                            );
                            suggested = true;
                        }
                    }
                }
                return suggested;
            }
        };

        // Find all the `impl`s in the local scope that can be called on the type parameter. And
        // retain all that are `impl`s of the trait that originated the `'static` obligation.
        // This doesn't find `impl dyn Trait { /**/ }`, but that case is handled above.
        let impl_self_tys = tcx
            .all_traits(LOCAL_CRATE)
            .iter()
            .flat_map(|trait_did| tcx.hir().trait_impls(*trait_did))
            .filter_map(|impl_node| {
                let impl_did = tcx.hir().local_def_id(*impl_node);
                match tcx.hir().get_if_local(impl_did.to_def_id()) {
                    Some(Node::Item(Item {
                        kind: ItemKind::Impl { self_ty, of_trait: Some(of_trait), items, .. },
                        ..
                    })) if of_trait.trait_def_id() == Some(container_id) => Some((
                        self_ty,
                        // Get the ident of the method, in order to use its `Span`.
                        items
                            .iter()
                            .filter(|item| item.ident == assoc.ident)
                            .map(|item| item.ident)
                            .next()
                            .unwrap_or(assoc.ident),
                    )),
                    _ => None,
                }
            });

        // Given all the `impl`s of the relevant `trait`, look for those that are implemented for
        // the trait object in the `fn` parameter type.
        for (self_ty, method) in impl_self_tys {
            for found_did in &v.0 {
                let mut hir_v = HirTraitObjectVisitor(vec![], *found_did);
                hir_v.visit_ty(self_ty);
                if let [span] = &hir_v.0[..] {
                    let mut multi_span: MultiSpan = vec![*span].into();
                    multi_span.push_span_label(
                        *span,
                        "this has an implicit `'static` lifetime requirement".to_string(),
                    );
                    multi_span.push_span_label(
                        method.span,
                        "`'static` requirement is introduced when calling this method".to_string(),
                    );
                    err.span_note(
                        multi_span,
                        &format!(
                            "`{}`'s `impl` of `{}` has an implicit `'static` requirement",
                            tcx.def_path_str(*found_did),
                            tcx.def_path_str(container_id),
                        ),
                    );
                    err.span_suggestion_verbose(
                        span.shrink_to_hi(),
                        "consider relaxing the implicit `'static` requirement",
                        " + '_".to_string(),
                        Applicability::MaybeIncorrect,
                    );
                    suggested = true;
                }
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

/// Collect all `hir::Ty<'_>` `Span`s for trait objects with an implicit lifetime.
struct HirTraitObjectVisitor(Vec<Span>, DefId);

impl<'tcx> Visitor<'tcx> for HirTraitObjectVisitor {
    type Map = ErasedMap<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }

    fn visit_ty(&mut self, t: &'tcx hir::Ty<'tcx>) {
        match t.kind {
            TyKind::TraitObject(
                poly_trait_refs,
                Lifetime { name: LifetimeName::ImplicitObjectLifetimeDefault, .. },
            ) => {
                for ptr in poly_trait_refs {
                    if Some(self.1) == ptr.trait_ref.trait_def_id() {
                        self.0.push(ptr.span);
                    }
                }
            }
            _ => {}
        }
        walk_ty(self, t);
    }
}
