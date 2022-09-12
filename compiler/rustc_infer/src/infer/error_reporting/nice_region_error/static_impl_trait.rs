//! Error Reporting for static impl Traits.

use crate::infer::error_reporting::nice_region_error::NiceRegionError;
use crate::infer::lexical_region_resolve::RegionResolutionError;
use crate::infer::{SubregionOrigin, TypeTrace};
use crate::traits::{ObligationCauseCode, UnifyReceiverContext};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{struct_span_err, Applicability, Diagnostic, ErrorGuaranteed, MultiSpan};
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{walk_ty, Visitor};
use rustc_hir::{self as hir, GenericBound, Item, ItemKind, Lifetime, LifetimeName, Node, TyKind};
use rustc_middle::ty::{
    self, AssocItemContainer, StaticLifetimeVisitor, Ty, TyCtxt, TypeSuperVisitable, TypeVisitor,
};
use rustc_span::symbol::Ident;
use rustc_span::Span;

use std::ops::ControlFlow;

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    /// Print the error message for lifetime errors when the return type is a static `impl Trait`,
    /// `dyn Trait` or if a method call on a trait object introduces a static requirement.
    pub(super) fn try_report_static_impl_trait(&self) -> Option<ErrorGuaranteed> {
        debug!("try_report_static_impl_trait(error={:?})", self.error);
        let tcx = self.tcx();
        let (var_origin, sub_origin, sub_r, sup_origin, sup_r, spans) = match self.error.as_ref()? {
            RegionResolutionError::SubSupConflict(
                _,
                var_origin,
                sub_origin,
                sub_r,
                sup_origin,
                sup_r,
                spans,
            ) if sub_r.is_static() => (var_origin, sub_origin, sub_r, sup_origin, sup_r, spans),
            RegionResolutionError::ConcreteFailure(
                SubregionOrigin::Subtype(box TypeTrace { cause, .. }),
                sub_r,
                sup_r,
            ) if sub_r.is_static() => {
                // This is for an implicit `'static` requirement coming from `impl dyn Trait {}`.
                if let ObligationCauseCode::UnifyReceiver(ctxt) = cause.code() {
                    // This may have a closure and it would cause ICE
                    // through `find_param_with_region` (#78262).
                    let anon_reg_sup = tcx.is_suitable_region(*sup_r)?;
                    let fn_returns = tcx.return_type_impl_or_dyn_traits(anon_reg_sup.def_id);
                    if fn_returns.is_empty() {
                        return None;
                    }

                    let param = self.find_param_with_region(*sup_r, *sub_r)?;
                    let lifetime = if sup_r.has_name() {
                        format!("lifetime `{}`", sup_r)
                    } else {
                        "an anonymous lifetime `'_`".to_string()
                    };
                    let mut err = struct_span_err!(
                        tcx.sess,
                        cause.span,
                        E0772,
                        "{} has {} but calling `{}` introduces an implicit `'static` lifetime \
                         requirement",
                        param
                            .param
                            .pat
                            .simple_ident()
                            .map(|s| format!("`{}`", s))
                            .unwrap_or_else(|| "`fn` parameter".to_string()),
                        lifetime,
                        ctxt.assoc_item.name,
                    );
                    err.span_label(param.param_ty_span, &format!("this data with {}...", lifetime));
                    err.span_label(
                        cause.span,
                        &format!(
                            "...is used and required to live as long as `'static` here \
                             because of an implicit lifetime bound on the {}",
                            match ctxt.assoc_item.container {
                                AssocItemContainer::TraitContainer => {
                                    let id = ctxt.assoc_item.container_id(tcx);
                                    format!("`impl` of `{}`", tcx.def_path_str(id))
                                }
                                AssocItemContainer::ImplContainer => "inherent `impl`".to_string(),
                            },
                        ),
                    );
                    if self.find_impl_on_dyn_trait(&mut err, param.param_ty, &ctxt) {
                        let reported = err.emit();
                        return Some(reported);
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
        let anon_reg_sup = tcx.is_suitable_region(*sup_r)?;
        debug!("try_report_static_impl_trait: anon_reg_sup={:?}", anon_reg_sup);
        let sp = var_origin.span();
        let return_sp = sub_origin.span();
        let param = self.find_param_with_region(*sup_r, *sub_r)?;
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

        let (mention_influencer, influencer_point) =
            if sup_origin.span().overlaps(param.param_ty_span) {
                // Account for `async fn` like in `async-await/issues/issue-62097.rs`.
                // The desugaring of `async `fn`s causes `sup_origin` and `param` to point at the same
                // place (but with different `ctxt`, hence `overlaps` instead of `==` above).
                //
                // This avoids the following:
                //
                // LL |     pub async fn run_dummy_fn(&self) {
                //    |                               ^^^^^
                //    |                               |
                //    |                               this data with an anonymous lifetime `'_`...
                //    |                               ...is captured here...
                (false, sup_origin.span())
            } else {
                (!sup_origin.span().overlaps(return_sp), param.param_ty_span)
            };
        err.span_label(influencer_point, &format!("this data with {}...", lifetime));

        debug!("try_report_static_impl_trait: param_info={:?}", param);

        let mut spans = spans.clone();

        if mention_influencer {
            spans.push(sup_origin.span());
        }
        // We dedup the spans *ignoring* expansion context.
        spans.sort();
        spans.dedup_by_key(|span| (span.lo(), span.hi()));

        // We try to make the output have fewer overlapping spans if possible.
        let require_msg = if spans.is_empty() {
            "...is used and required to live as long as `'static` here"
        } else {
            "...and is required to live as long as `'static` here"
        };
        let require_span =
            if sup_origin.span().overlaps(return_sp) { sup_origin.span() } else { return_sp };

        for span in &spans {
            err.span_label(*span, "...is used here...");
        }

        if spans.iter().any(|sp| sp.overlaps(return_sp) || *sp > return_sp) {
            // If any of the "captured here" labels appears on the same line or after
            // `require_span`, we put it on a note to ensure the text flows by appearing
            // always at the end.
            err.span_note(require_span, require_msg);
        } else {
            // We don't need a note, it's already at the end, it can be shown as a `span_label`.
            err.span_label(require_span, require_msg);
        }

        if let SubregionOrigin::RelateParamBound(_, _, Some(bound)) = sub_origin {
            err.span_note(*bound, "`'static` lifetime requirement introduced by this bound");
        }
        if let SubregionOrigin::Subtype(box TypeTrace { cause, .. }) = sub_origin {
            if let ObligationCauseCode::ReturnValue(hir_id)
            | ObligationCauseCode::BlockTailExpression(hir_id) = cause.code()
            {
                let parent_id = tcx.hir().get_parent_item(*hir_id);
                let parent_id = tcx.hir().local_def_id_to_hir_id(parent_id);
                if let Some(fn_decl) = tcx.hir().fn_decl_by_hir_id(parent_id) {
                    let mut span: MultiSpan = fn_decl.output.span().into();
                    let mut add_label = true;
                    if let hir::FnRetTy::Return(ty) = fn_decl.output {
                        let mut v = StaticLifetimeVisitor(vec![], tcx.hir());
                        v.visit_ty(ty);
                        if !v.0.is_empty() {
                            span = v.0.clone().into();
                            for sp in v.0 {
                                span.push_span_label(sp, "`'static` requirement introduced here");
                            }
                            add_label = false;
                        }
                    }
                    if add_label {
                        span.push_span_label(
                            fn_decl.output.span(),
                            "requirement introduced by this return type",
                        );
                    }
                    span.push_span_label(cause.span, "because of this returned expression");
                    err.span_note(
                        span,
                        "`'static` lifetime requirement introduced by the return type",
                    );
                }
            }
        }

        let fn_returns = tcx.return_type_impl_or_dyn_traits(anon_reg_sup.def_id);

        let mut override_error_code = None;
        if let SubregionOrigin::Subtype(box TypeTrace { cause, .. }) = &sup_origin
            && let ObligationCauseCode::UnifyReceiver(ctxt) = cause.code()
            // Handle case of `impl Foo for dyn Bar { fn qux(&self) {} }` introducing a
            // `'static` lifetime when called as a method on a binding: `bar.qux()`.
            && self.find_impl_on_dyn_trait(&mut err, param.param_ty, &ctxt)
        {
            override_error_code = Some(ctxt.assoc_item.name);
        }

        if let SubregionOrigin::Subtype(box TypeTrace { cause, .. }) = &sub_origin
            && let code = match cause.code() {
                ObligationCauseCode::MatchImpl(parent, ..) => parent.code(),
                _ => cause.code(),
            }
            && let (&ObligationCauseCode::ItemObligation(item_def_id) | &ObligationCauseCode::ExprItemObligation(item_def_id, ..), None) = (code, override_error_code)
        {
            // Same case of `impl Foo for dyn Bar { fn qux(&self) {} }` introducing a `'static`
            // lifetime as above, but called using a fully-qualified path to the method:
            // `Foo::qux(bar)`.
            let mut v = TraitObjectVisitor(FxHashSet::default());
            v.visit_ty(param.param_ty);
            if let Some((ident, self_ty)) =
                self.get_impl_ident_and_self_ty_from_trait(item_def_id, &v.0)
                && self.suggest_constrain_dyn_trait_in_impl(&mut err, &v.0, ident, self_ty)
            {
                override_error_code = Some(ident.name);
            }
        }
        if let (Some(ident), true) = (override_error_code, fn_returns.is_empty()) {
            // Provide a more targeted error code and description.
            err.code(rustc_errors::error_code!(E0772));
            err.set_primary_message(&format!(
                "{} has {} but calling `{}` introduces an implicit `'static` lifetime \
                requirement",
                param_name, lifetime, ident,
            ));
        }

        let arg = match param.param.pat.simple_ident() {
            Some(simple_ident) => format!("argument `{}`", simple_ident),
            None => "the argument".to_string(),
        };
        let captures = format!("captures data from {}", arg);
        suggest_new_region_bound(
            tcx,
            &mut err,
            fn_returns,
            lifetime_name,
            Some(arg),
            captures,
            Some((param.param_ty_span, param.param_ty.to_string())),
        );

        let reported = err.emit();
        Some(reported)
    }
}

pub fn suggest_new_region_bound(
    tcx: TyCtxt<'_>,
    err: &mut Diagnostic,
    fn_returns: Vec<&rustc_hir::Ty<'_>>,
    lifetime_name: String,
    arg: Option<String>,
    captures: String,
    param: Option<(Span, String)>,
) {
    debug!("try_report_static_impl_trait: fn_return={:?}", fn_returns);
    // FIXME: account for the need of parens in `&(dyn Trait + '_)`
    let consider = "consider changing the";
    let declare = "to declare that the";
    let explicit = format!("you can add an explicit `{}` lifetime bound", lifetime_name);
    let explicit_static =
        arg.map(|arg| format!("explicit `'static` bound to the lifetime of {}", arg));
    let add_static_bound = "alternatively, add an explicit `'static` bound to this reference";
    let plus_lt = format!(" + {}", lifetime_name);
    for fn_return in fn_returns {
        if fn_return.span.desugaring_kind().is_some() {
            // Skip `async` desugaring `impl Future`.
            continue;
        }
        match fn_return.kind {
            TyKind::OpaqueDef(item_id, _, _) => {
                let item = tcx.hir().item(item_id);
                let ItemKind::OpaqueTy(opaque) = &item.kind else {
                    return;
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
                    if let Some(explicit_static) = &explicit_static {
                        err.span_suggestion_verbose(
                            span,
                            &format!("{} `impl Trait`'s {}", consider, explicit_static),
                            &lifetime_name,
                            Applicability::MaybeIncorrect,
                        );
                    }
                    if let Some((param_span, param_ty)) = param.clone() {
                        err.span_suggestion_verbose(
                            param_span,
                            add_static_bound,
                            param_ty,
                            Applicability::MaybeIncorrect,
                        );
                    }
                } else if opaque
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
                    .is_some()
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
                        &plus_lt,
                        Applicability::MaybeIncorrect,
                    );
                }
            }
            TyKind::TraitObject(_, lt, _) => match lt.name {
                LifetimeName::ImplicitObjectLifetimeDefault => {
                    err.span_suggestion_verbose(
                        fn_return.span.shrink_to_hi(),
                        &format!(
                            "{declare} trait object {captures}, {explicit}",
                            declare = declare,
                            captures = captures,
                            explicit = explicit,
                        ),
                        &plus_lt,
                        Applicability::MaybeIncorrect,
                    );
                }
                name if name.ident().to_string() != lifetime_name => {
                    // With this check we avoid suggesting redundant bounds. This
                    // would happen if there are nested impl/dyn traits and only
                    // one of them has the bound we'd suggest already there, like
                    // in `impl Foo<X = dyn Bar> + '_`.
                    if let Some(explicit_static) = &explicit_static {
                        err.span_suggestion_verbose(
                            lt.span,
                            &format!("{} trait object's {}", consider, explicit_static),
                            &lifetime_name,
                            Applicability::MaybeIncorrect,
                        );
                    }
                    if let Some((param_span, param_ty)) = param.clone() {
                        err.span_suggestion_verbose(
                            param_span,
                            add_static_bound,
                            param_ty,
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }
}

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    fn get_impl_ident_and_self_ty_from_trait(
        &self,
        def_id: DefId,
        trait_objects: &FxHashSet<DefId>,
    ) -> Option<(Ident, &'tcx hir::Ty<'tcx>)> {
        let tcx = self.tcx();
        match tcx.hir().get_if_local(def_id) {
            Some(Node::ImplItem(impl_item)) => {
                match tcx.hir().find_by_def_id(tcx.hir().get_parent_item(impl_item.hir_id())) {
                    Some(Node::Item(Item {
                        kind: ItemKind::Impl(hir::Impl { self_ty, .. }),
                        ..
                    })) => Some((impl_item.ident, self_ty)),
                    _ => None,
                }
            }
            Some(Node::TraitItem(trait_item)) => {
                let trait_did = tcx.hir().get_parent_item(trait_item.hir_id());
                match tcx.hir().find_by_def_id(trait_did) {
                    Some(Node::Item(Item { kind: ItemKind::Trait(..), .. })) => {
                        // The method being called is defined in the `trait`, but the `'static`
                        // obligation comes from the `impl`. Find that `impl` so that we can point
                        // at it in the suggestion.
                        let trait_did = trait_did.to_def_id();
                        match tcx
                            .hir()
                            .trait_impls(trait_did)
                            .iter()
                            .filter_map(|&impl_did| {
                                match tcx.hir().get_if_local(impl_did.to_def_id()) {
                                    Some(Node::Item(Item {
                                        kind: ItemKind::Impl(hir::Impl { self_ty, .. }),
                                        ..
                                    })) if trait_objects.iter().all(|did| {
                                        // FIXME: we should check `self_ty` against the receiver
                                        // type in the `UnifyReceiver` context, but for now, use
                                        // this imperfect proxy. This will fail if there are
                                        // multiple `impl`s for the same trait like
                                        // `impl Foo for Box<dyn Bar>` and `impl Foo for dyn Bar`.
                                        // In that case, only the first one will get suggestions.
                                        let mut traits = vec![];
                                        let mut hir_v = HirTraitObjectVisitor(&mut traits, *did);
                                        hir_v.visit_ty(self_ty);
                                        !traits.is_empty()
                                    }) =>
                                    {
                                        Some(self_ty)
                                    }
                                    _ => None,
                                }
                            })
                            .next()
                        {
                            Some(self_ty) => Some((trait_item.ident, self_ty)),
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// When we call a method coming from an `impl Foo for dyn Bar`, `dyn Bar` introduces a default
    /// `'static` obligation. Suggest relaxing that implicit bound.
    fn find_impl_on_dyn_trait(
        &self,
        err: &mut Diagnostic,
        ty: Ty<'_>,
        ctxt: &UnifyReceiverContext<'tcx>,
    ) -> bool {
        let tcx = self.tcx();

        // Find the method being called.
        let Ok(Some(instance)) = ty::Instance::resolve(
            tcx,
            ctxt.param_env,
            ctxt.assoc_item.def_id,
            self.infcx.resolve_vars_if_possible(ctxt.substs),
        ) else {
            return false;
        };

        let mut v = TraitObjectVisitor(FxHashSet::default());
        v.visit_ty(ty);

        // Get the `Ident` of the method being called and the corresponding `impl` (to point at
        // `Bar` in `impl Foo for dyn Bar {}` and the definition of the method being called).
        let Some((ident, self_ty)) = self.get_impl_ident_and_self_ty_from_trait(instance.def_id(), &v.0) else {
            return false;
        };

        // Find the trait object types in the argument, so we point at *only* the trait object.
        self.suggest_constrain_dyn_trait_in_impl(err, &v.0, ident, self_ty)
    }

    fn suggest_constrain_dyn_trait_in_impl(
        &self,
        err: &mut Diagnostic,
        found_dids: &FxHashSet<DefId>,
        ident: Ident,
        self_ty: &hir::Ty<'_>,
    ) -> bool {
        let mut suggested = false;
        for found_did in found_dids {
            let mut traits = vec![];
            let mut hir_v = HirTraitObjectVisitor(&mut traits, *found_did);
            hir_v.visit_ty(&self_ty);
            for span in &traits {
                let mut multi_span: MultiSpan = vec![*span].into();
                multi_span
                    .push_span_label(*span, "this has an implicit `'static` lifetime requirement");
                multi_span.push_span_label(
                    ident.span,
                    "calling this method introduces the `impl`'s 'static` requirement",
                );
                err.span_note(multi_span, "the used `impl` has a `'static` requirement");
                err.span_suggestion_verbose(
                    span.shrink_to_hi(),
                    "consider relaxing the implicit `'static` requirement",
                    " + '_",
                    Applicability::MaybeIncorrect,
                );
                suggested = true;
            }
        }
        suggested
    }
}

/// Collect all the trait objects in a type that could have received an implicit `'static` lifetime.
pub struct TraitObjectVisitor(pub FxHashSet<DefId>);

impl<'tcx> TypeVisitor<'tcx> for TraitObjectVisitor {
    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
        match t.kind() {
            ty::Dynamic(preds, re) if re.is_static() => {
                if let Some(def_id) = preds.principal_def_id() {
                    self.0.insert(def_id);
                }
                ControlFlow::CONTINUE
            }
            _ => t.super_visit_with(self),
        }
    }
}

/// Collect all `hir::Ty<'_>` `Span`s for trait objects with an implicit lifetime.
pub struct HirTraitObjectVisitor<'a>(pub &'a mut Vec<Span>, pub DefId);

impl<'a, 'tcx> Visitor<'tcx> for HirTraitObjectVisitor<'a> {
    fn visit_ty(&mut self, t: &'tcx hir::Ty<'tcx>) {
        if let TyKind::TraitObject(
            poly_trait_refs,
            Lifetime { name: LifetimeName::ImplicitObjectLifetimeDefault, .. },
            _,
        ) = t.kind
        {
            for ptr in poly_trait_refs {
                if Some(self.1) == ptr.trait_ref.trait_def_id() {
                    self.0.push(ptr.span);
                }
            }
        }
        walk_ty(self, t);
    }
}
