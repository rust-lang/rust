//! Error Reporting for static impl Traits.

use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::{Applicability, Diag, ErrorGuaranteed};
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{Visitor, VisitorExt, walk_ty};
use rustc_hir::{
    self as hir, AmbigArg, GenericBound, GenericParam, GenericParamKind, Item, ItemKind, Lifetime,
    LifetimeKind, LifetimeParamKind, MissingLifetimeKind, Node, TyKind,
};
use rustc_middle::ty::{self, Ty, TyCtxt, TypeSuperVisitable, TypeVisitor};
use rustc_span::def_id::LocalDefId;
use rustc_span::{Ident, Span};
use tracing::debug;

use crate::error_reporting::infer::nice_region_error::NiceRegionError;
use crate::errors::ButNeedsToSatisfy;
use crate::infer::{RegionResolutionError, SubregionOrigin};

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
            _ => return None,
        };
        debug!(
            "try_report_static_impl_trait(var={:?}, sub={:?} {:?} sup={:?} {:?})",
            var_origin, sub_origin, sub_r, sup_origin, sup_r
        );
        let anon_reg_sup = tcx.is_suitable_region(self.generic_param_scope, *sup_r)?;
        debug!("try_report_static_impl_trait: anon_reg_sup={:?}", anon_reg_sup);
        let sp = var_origin.span();
        let return_sp = sub_origin.span();
        let param = self.find_param_with_region(*sup_r, *sub_r)?;
        let simple_ident = param.param.pat.simple_ident();
        let lifetime_name =
            if sup_r.is_named(self.tcx()) { sup_r.to_string() } else { "'_".to_owned() };

        let (mention_influencer, influencer_point) =
            if sup_origin.span().overlaps(param.param_ty_span) {
                // Account for `async fn` like in `async-await/issues/issue-62097.rs`.
                // The desugaring of `async fn`s causes `sup_origin` and `param` to point at the same
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

        debug!("try_report_static_impl_trait: param_info={:?}", param);

        let mut spans = spans.clone();

        if mention_influencer {
            spans.push(sup_origin.span());
        }
        // We dedup the spans *ignoring* expansion context.
        spans.sort();
        spans.dedup_by_key(|span| (span.lo(), span.hi()));

        // We try to make the output have fewer overlapping spans if possible.
        let require_span =
            if sup_origin.span().overlaps(return_sp) { sup_origin.span() } else { return_sp };

        let spans_empty = spans.is_empty();
        let require_as_note = spans.iter().any(|sp| sp.overlaps(return_sp) || *sp > return_sp);
        let bound = if let SubregionOrigin::RelateParamBound(_, _, Some(bound)) = sub_origin {
            Some(*bound)
        } else {
            None
        };

        let diag = ButNeedsToSatisfy {
            sp,
            influencer_point,
            spans: spans.clone(),
            // If any of the "captured here" labels appears on the same line or after
            // `require_span`, we put it on a note to ensure the text flows by appearing
            // always at the end.
            require_span_as_note: require_as_note.then_some(require_span),
            // We don't need a note, it's already at the end, it can be shown as a `span_label`.
            require_span_as_label: (!require_as_note).then_some(require_span),

            has_lifetime: sup_r.is_named(self.tcx()),
            lifetime: lifetime_name.clone(),
            has_param_name: simple_ident.is_some(),
            param_name: simple_ident.map(|x| x.to_string()).unwrap_or_default(),
            spans_empty,
            bound,
        };

        let mut err = self.tcx().dcx().create_err(diag);

        let fn_returns = tcx.return_type_impl_or_dyn_traits(anon_reg_sup.scope);

        let arg = match param.param.pat.simple_ident() {
            Some(simple_ident) => format!("argument `{simple_ident}`"),
            None => "the argument".to_string(),
        };
        let captures = format!("captures data from {arg}");
        suggest_new_region_bound(
            tcx,
            &mut err,
            fn_returns,
            lifetime_name,
            Some(arg),
            captures,
            Some((param.param_ty_span, param.param_ty.to_string())),
            Some(anon_reg_sup.scope),
        );

        let reported = err.emit();
        Some(reported)
    }
}

pub fn suggest_new_region_bound(
    tcx: TyCtxt<'_>,
    err: &mut Diag<'_>,
    fn_returns: Vec<&rustc_hir::Ty<'_>>,
    lifetime_name: String,
    arg: Option<String>,
    captures: String,
    param: Option<(Span, String)>,
    scope_def_id: Option<LocalDefId>,
) {
    debug!("try_report_static_impl_trait: fn_return={:?}", fn_returns);
    // FIXME: account for the need of parens in `&(dyn Trait + '_)`
    let consider = "consider changing";
    let declare = "to declare that";
    let explicit = format!("you can add an explicit `{lifetime_name}` lifetime bound");
    let explicit_static =
        arg.map(|arg| format!("explicit `'static` bound to the lifetime of {arg}"));
    let add_static_bound = "alternatively, add an explicit `'static` bound to this reference";
    let plus_lt = format!(" + {lifetime_name}");
    for fn_return in fn_returns {
        if fn_return.span.desugaring_kind().is_some() {
            // Skip `async` desugaring `impl Future`.
            continue;
        }
        match fn_return.kind {
            // FIXME(precise_captures): Suggest adding to `use<...>` list instead.
            TyKind::OpaqueDef(opaque) => {
                // Get the identity type for this RPIT
                let did = opaque.def_id.to_def_id();
                let ty = Ty::new_opaque(tcx, did, ty::GenericArgs::identity_for_item(tcx, did));

                if let Some(span) = opaque.bounds.iter().find_map(|arg| match arg {
                    GenericBound::Outlives(Lifetime {
                        kind: LifetimeKind::Static, ident, ..
                    }) => Some(ident.span),
                    _ => None,
                }) {
                    if let Some(explicit_static) = &explicit_static {
                        err.span_suggestion_verbose(
                            span,
                            format!("{consider} `{ty}`'s {explicit_static}"),
                            &lifetime_name,
                            Applicability::MaybeIncorrect,
                        );
                    }
                    if let Some((param_span, ref param_ty)) = param {
                        err.span_suggestion_verbose(
                            param_span,
                            add_static_bound,
                            param_ty,
                            Applicability::MaybeIncorrect,
                        );
                    }
                } else if opaque.bounds.iter().any(|arg| {
                    matches!(arg,
                        GenericBound::Outlives(Lifetime { ident, .. })
                        if ident.name.to_string() == lifetime_name )
                }) {
                } else {
                    // get a lifetime name of existing named lifetimes if any
                    let existing_lt_name = if let Some(id) = scope_def_id
                        && let Some(generics) = tcx.hir_get_generics(id)
                        && let named_lifetimes = generics
                            .params
                            .iter()
                            .filter(|p| {
                                matches!(
                                    p.kind,
                                    GenericParamKind::Lifetime {
                                        kind: hir::LifetimeParamKind::Explicit
                                    }
                                )
                            })
                            .map(|p| {
                                if let hir::ParamName::Plain(name) = p.name {
                                    Some(name.to_string())
                                } else {
                                    None
                                }
                            })
                            .filter(|n| !matches!(n, None))
                            .collect::<Vec<_>>()
                        && named_lifetimes.len() > 0
                    {
                        named_lifetimes[0].clone()
                    } else {
                        None
                    };
                    let name = if let Some(name) = &existing_lt_name { name } else { "'a" };
                    // if there are more than one elided lifetimes in inputs, the explicit `'_` lifetime cannot be used.
                    // introducing a new lifetime `'a` or making use of one from existing named lifetimes if any
                    if let Some(id) = scope_def_id
                        && let Some(generics) = tcx.hir_get_generics(id)
                        && let mut spans_suggs =
                            make_elided_region_spans_suggs(name, generics.params.iter())
                        && spans_suggs.len() > 1
                    {
                        let use_lt = if existing_lt_name == None {
                            spans_suggs.push((generics.span.shrink_to_hi(), format!("<{name}>")));
                            format!("you can introduce a named lifetime parameter `{name}`")
                        } else {
                            // make use the existing named lifetime
                            format!("you can use the named lifetime parameter `{name}`")
                        };
                        spans_suggs.push((fn_return.span.shrink_to_hi(), format!(" + {name} ")));
                        err.multipart_suggestion_verbose(
                            format!("{declare} `{ty}` {captures}, {use_lt}",),
                            spans_suggs,
                            Applicability::MaybeIncorrect,
                        );
                    } else {
                        err.span_suggestion_verbose(
                            fn_return.span.shrink_to_hi(),
                            format!("{declare} `{ty}` {captures}, {explicit}",),
                            &plus_lt,
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
            }
            TyKind::TraitObject(_, lt) => {
                if let LifetimeKind::ImplicitObjectLifetimeDefault = lt.kind {
                    err.span_suggestion_verbose(
                        fn_return.span.shrink_to_hi(),
                        format!("{declare} the trait object {captures}, {explicit}",),
                        &plus_lt,
                        Applicability::MaybeIncorrect,
                    );
                } else if lt.ident.name.to_string() != lifetime_name {
                    // With this check we avoid suggesting redundant bounds. This
                    // would happen if there are nested impl/dyn traits and only
                    // one of them has the bound we'd suggest already there, like
                    // in `impl Foo<X = dyn Bar> + '_`.
                    if let Some(explicit_static) = &explicit_static {
                        err.span_suggestion_verbose(
                            lt.ident.span,
                            format!("{consider} the trait object's {explicit_static}"),
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
            }
            _ => {}
        }
    }
}

fn make_elided_region_spans_suggs<'a>(
    name: &str,
    generic_params: impl Iterator<Item = &'a GenericParam<'a>>,
) -> Vec<(Span, String)> {
    let mut spans_suggs = Vec::new();
    let mut bracket_span = None;
    let mut consecutive_brackets = 0;

    let mut process_consecutive_brackets =
        |span: Option<Span>, spans_suggs: &mut Vec<(Span, String)>| {
            if let Some(span) = span
                && bracket_span.is_none_or(|bracket_span| span == bracket_span)
            {
                consecutive_brackets += 1;
            } else if let Some(bracket_span) = bracket_span.take() {
                let sugg = std::iter::once("<")
                    .chain(std::iter::repeat(name).take(consecutive_brackets).intersperse(", "))
                    .chain([">"])
                    .collect();
                spans_suggs.push((bracket_span.shrink_to_hi(), sugg));
                consecutive_brackets = 0;
            }
            bracket_span = span;
        };

    for p in generic_params {
        if let GenericParamKind::Lifetime { kind: LifetimeParamKind::Elided(kind) } = p.kind {
            match kind {
                MissingLifetimeKind::Underscore => {
                    process_consecutive_brackets(None, &mut spans_suggs);
                    spans_suggs.push((p.span, name.to_string()))
                }
                MissingLifetimeKind::Ampersand => {
                    process_consecutive_brackets(None, &mut spans_suggs);
                    spans_suggs.push((p.span.shrink_to_hi(), format!("{name} ")));
                }
                MissingLifetimeKind::Comma => {
                    process_consecutive_brackets(None, &mut spans_suggs);
                    spans_suggs.push((p.span.shrink_to_hi(), format!("{name}, ")));
                }
                MissingLifetimeKind::Brackets => {
                    process_consecutive_brackets(Some(p.span), &mut spans_suggs);
                }
            }
        }
    }
    process_consecutive_brackets(None, &mut spans_suggs);

    spans_suggs
}

impl<'a, 'tcx> NiceRegionError<'a, 'tcx> {
    pub fn get_impl_ident_and_self_ty_from_trait(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        trait_objects: &FxIndexSet<DefId>,
    ) -> Option<(Ident, &'tcx hir::Ty<'tcx>)> {
        match tcx.hir_get_if_local(def_id)? {
            Node::ImplItem(impl_item) => {
                let impl_did = tcx.hir_get_parent_item(impl_item.hir_id());
                if let hir::OwnerNode::Item(Item {
                    kind: ItemKind::Impl(hir::Impl { self_ty, .. }),
                    ..
                }) = tcx.hir_owner_node(impl_did)
                {
                    Some((impl_item.ident, self_ty))
                } else {
                    None
                }
            }
            Node::TraitItem(trait_item) => {
                let trait_id = tcx.hir_get_parent_item(trait_item.hir_id());
                debug_assert_eq!(tcx.def_kind(trait_id.def_id), hir::def::DefKind::Trait);
                // The method being called is defined in the `trait`, but the `'static`
                // obligation comes from the `impl`. Find that `impl` so that we can point
                // at it in the suggestion.
                let trait_did = trait_id.to_def_id();
                tcx.local_trait_impls(trait_did).iter().find_map(|&impl_did| {
                    if let Node::Item(Item {
                        kind: ItemKind::Impl(hir::Impl { self_ty, .. }), ..
                    }) = tcx.hir_node_by_def_id(impl_did)
                        && trait_objects.iter().all(|did| {
                            // FIXME: we should check `self_ty`, but for now, use
                            // this imperfect proxy. This will fail if there are
                            // multiple `impl`s for the same trait like
                            // `impl Foo for Box<dyn Bar>` and `impl Foo for dyn Bar`.
                            // In that case, only the first one will get suggestions.
                            let mut traits = vec![];
                            let mut hir_v = HirTraitObjectVisitor(&mut traits, *did);
                            hir_v.visit_ty_unambig(self_ty);
                            !traits.is_empty()
                        })
                    {
                        Some((trait_item.ident, *self_ty))
                    } else {
                        None
                    }
                })
            }
            _ => None,
        }
    }
}

/// Collect all the trait objects in a type that could have received an implicit `'static` lifetime.
pub struct TraitObjectVisitor(pub FxIndexSet<DefId>);

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for TraitObjectVisitor {
    fn visit_ty(&mut self, t: Ty<'tcx>) {
        match t.kind() {
            ty::Dynamic(preds, re) if re.is_static() => {
                if let Some(def_id) = preds.principal_def_id() {
                    self.0.insert(def_id);
                }
            }
            _ => t.super_visit_with(self),
        }
    }
}

/// Collect all `hir::Ty<'_>` `Span`s for trait objects with an implicit lifetime.
pub struct HirTraitObjectVisitor<'a>(pub &'a mut Vec<Span>, pub DefId);

impl<'a, 'tcx> Visitor<'tcx> for HirTraitObjectVisitor<'a> {
    fn visit_ty(&mut self, t: &'tcx hir::Ty<'tcx, AmbigArg>) {
        if let TyKind::TraitObject(poly_trait_refs, lifetime_ptr) = t.kind
            && let Lifetime { kind: LifetimeKind::ImplicitObjectLifetimeDefault, .. } =
                lifetime_ptr.pointer()
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
