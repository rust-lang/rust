//! Give useful errors and suggestions to users when an item can't be
//! found or is otherwise invalid.

use crate::check::FnCtxt;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::{
    pluralize, struct_span_err, Applicability, Diagnostic, DiagnosticBuilder, ErrorGuaranteed,
    MultiSpan,
};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItem;
use rustc_hir::{ExprKind, Node, QPath};
use rustc_infer::infer::type_variable::{TypeVariableOrigin, TypeVariableOriginKind};
use rustc_middle::traits::util::supertraits;
use rustc_middle::ty::fast_reject::{simplify_type, TreatParams};
use rustc_middle::ty::print::with_crate_prefix;
use rustc_middle::ty::{self, DefIdTree, ToPredicate, Ty, TyCtxt, TypeVisitable};
use rustc_middle::ty::{IsSuggestable, ToPolyTraitRef};
use rustc_span::symbol::{kw, sym, Ident};
use rustc_span::Symbol;
use rustc_span::{lev_distance, source_map, ExpnKind, FileName, MacroKind, Span};
use rustc_trait_selection::traits::error_reporting::on_unimplemented::InferCtxtExt as _;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt as _;
use rustc_trait_selection::traits::{
    FulfillmentError, Obligation, ObligationCause, ObligationCauseCode, OnUnimplementedNote,
};

use std::cmp::Ordering;
use std::iter;

use super::probe::{IsSuggestion, Mode, ProbeScope};
use super::{CandidateSource, MethodError, NoMatchData};

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    fn is_fn_ty(&self, ty: Ty<'tcx>, span: Span) -> bool {
        let tcx = self.tcx;
        match ty.kind() {
            // Not all of these (e.g., unsafe fns) implement `FnOnce`,
            // so we look for these beforehand.
            ty::Closure(..) | ty::FnDef(..) | ty::FnPtr(_) => true,
            // If it's not a simple function, look for things which implement `FnOnce`.
            _ => {
                let Some(fn_once) = tcx.lang_items().fn_once_trait() else {
                    return false;
                };

                // This conditional prevents us from asking to call errors and unresolved types.
                // It might seem that we can use `predicate_must_hold_modulo_regions`,
                // but since a Dummy binder is used to fill in the FnOnce trait's arguments,
                // type resolution always gives a "maybe" here.
                if self.autoderef(span, ty).any(|(ty, _)| {
                    info!("check deref {:?} error", ty);
                    matches!(ty.kind(), ty::Error(_) | ty::Infer(_))
                }) {
                    return false;
                }

                self.autoderef(span, ty).any(|(ty, _)| {
                    info!("check deref {:?} impl FnOnce", ty);
                    self.probe(|_| {
                        let fn_once_substs = tcx.mk_substs_trait(
                            ty,
                            &[self
                                .next_ty_var(TypeVariableOrigin {
                                    kind: TypeVariableOriginKind::MiscVariable,
                                    span,
                                })
                                .into()],
                        );
                        let trait_ref = ty::TraitRef::new(fn_once, fn_once_substs);
                        let poly_trait_ref = ty::Binder::dummy(trait_ref);
                        let obligation = Obligation::misc(
                            span,
                            self.body_id,
                            self.param_env,
                            poly_trait_ref.without_const().to_predicate(tcx),
                        );
                        self.predicate_may_hold(&obligation)
                    })
                })
            }
        }
    }

    fn is_slice_ty(&self, ty: Ty<'tcx>, span: Span) -> bool {
        self.autoderef(span, ty).any(|(ty, _)| matches!(ty.kind(), ty::Slice(..) | ty::Array(..)))
    }

    pub fn report_method_error(
        &self,
        mut span: Span,
        rcvr_ty: Ty<'tcx>,
        item_name: Ident,
        source: SelfSource<'tcx>,
        error: MethodError<'tcx>,
        args: Option<&'tcx [hir::Expr<'tcx>]>,
    ) -> Option<DiagnosticBuilder<'_, ErrorGuaranteed>> {
        // Avoid suggestions when we don't know what's going on.
        if rcvr_ty.references_error() {
            return None;
        }

        let report_candidates = |span: Span,
                                 err: &mut Diagnostic,
                                 mut sources: Vec<CandidateSource>,
                                 sugg_span: Span| {
            sources.sort();
            sources.dedup();
            // Dynamic limit to avoid hiding just one candidate, which is silly.
            let limit = if sources.len() == 5 { 5 } else { 4 };

            for (idx, source) in sources.iter().take(limit).enumerate() {
                match *source {
                    CandidateSource::Impl(impl_did) => {
                        // Provide the best span we can. Use the item, if local to crate, else
                        // the impl, if local to crate (item may be defaulted), else nothing.
                        let Some(item) = self.associated_value(impl_did, item_name).or_else(|| {
                            let impl_trait_ref = self.tcx.impl_trait_ref(impl_did)?;
                            self.associated_value(impl_trait_ref.def_id, item_name)
                        }) else {
                            continue;
                        };

                        let note_span = if item.def_id.is_local() {
                            Some(self.tcx.def_span(item.def_id))
                        } else if impl_did.is_local() {
                            Some(self.tcx.def_span(impl_did))
                        } else {
                            None
                        };

                        let impl_ty = self.tcx.at(span).type_of(impl_did);

                        let insertion = match self.tcx.impl_trait_ref(impl_did) {
                            None => String::new(),
                            Some(trait_ref) => format!(
                                " of the trait `{}`",
                                self.tcx.def_path_str(trait_ref.def_id)
                            ),
                        };

                        let (note_str, idx) = if sources.len() > 1 {
                            (
                                format!(
                                    "candidate #{} is defined in an impl{} for the type `{}`",
                                    idx + 1,
                                    insertion,
                                    impl_ty,
                                ),
                                Some(idx + 1),
                            )
                        } else {
                            (
                                format!(
                                    "the candidate is defined in an impl{} for the type `{}`",
                                    insertion, impl_ty,
                                ),
                                None,
                            )
                        };
                        if let Some(note_span) = note_span {
                            // We have a span pointing to the method. Show note with snippet.
                            err.span_note(note_span, &note_str);
                        } else {
                            err.note(&note_str);
                        }
                        if let Some(trait_ref) = self.tcx.impl_trait_ref(impl_did) {
                            let path = self.tcx.def_path_str(trait_ref.def_id);

                            let ty = match item.kind {
                                ty::AssocKind::Const | ty::AssocKind::Type => rcvr_ty,
                                ty::AssocKind::Fn => self
                                    .tcx
                                    .fn_sig(item.def_id)
                                    .inputs()
                                    .skip_binder()
                                    .get(0)
                                    .filter(|ty| ty.is_region_ptr() && !rcvr_ty.is_region_ptr())
                                    .copied()
                                    .unwrap_or(rcvr_ty),
                            };
                            print_disambiguation_help(
                                item_name,
                                args,
                                err,
                                path,
                                ty,
                                item.kind,
                                item.def_id,
                                sugg_span,
                                idx,
                                self.tcx.sess.source_map(),
                                item.fn_has_self_parameter,
                            );
                        }
                    }
                    CandidateSource::Trait(trait_did) => {
                        let Some(item) = self.associated_value(trait_did, item_name) else { continue };
                        let item_span = self.tcx.def_span(item.def_id);
                        let idx = if sources.len() > 1 {
                            let msg = &format!(
                                "candidate #{} is defined in the trait `{}`",
                                idx + 1,
                                self.tcx.def_path_str(trait_did)
                            );
                            err.span_note(item_span, msg);
                            Some(idx + 1)
                        } else {
                            let msg = &format!(
                                "the candidate is defined in the trait `{}`",
                                self.tcx.def_path_str(trait_did)
                            );
                            err.span_note(item_span, msg);
                            None
                        };
                        let path = self.tcx.def_path_str(trait_did);
                        print_disambiguation_help(
                            item_name,
                            args,
                            err,
                            path,
                            rcvr_ty,
                            item.kind,
                            item.def_id,
                            sugg_span,
                            idx,
                            self.tcx.sess.source_map(),
                            item.fn_has_self_parameter,
                        );
                    }
                }
            }
            if sources.len() > limit {
                err.note(&format!("and {} others", sources.len() - limit));
            }
        };

        let sugg_span = if let SelfSource::MethodCall(expr) = source {
            // Given `foo.bar(baz)`, `expr` is `bar`, but we want to point to the whole thing.
            self.tcx.hir().expect_expr(self.tcx.hir().get_parent_node(expr.hir_id)).span
        } else {
            span
        };

        match error {
            MethodError::NoMatch(NoMatchData {
                static_candidates: static_sources,
                unsatisfied_predicates,
                out_of_scope_traits,
                lev_candidate,
                mode,
            }) => {
                let tcx = self.tcx;

                let actual = self.resolve_vars_if_possible(rcvr_ty);
                let ty_str = self.ty_to_string(actual);
                let is_method = mode == Mode::MethodCall;
                let item_kind = if is_method {
                    "method"
                } else if actual.is_enum() {
                    "variant or associated item"
                } else {
                    match (item_name.as_str().chars().next(), actual.is_fresh_ty()) {
                        (Some(name), false) if name.is_lowercase() => "function or associated item",
                        (Some(_), false) => "associated item",
                        (Some(_), true) | (None, false) => "variant or associated item",
                        (None, true) => "variant",
                    }
                };

                if self.suggest_constraining_numerical_ty(
                    tcx, actual, source, span, item_kind, item_name, &ty_str,
                ) {
                    return None;
                }

                span = item_name.span;

                // Don't show generic arguments when the method can't be found in any implementation (#81576).
                let mut ty_str_reported = ty_str.clone();
                if let ty::Adt(_, generics) = actual.kind() {
                    if generics.len() > 0 {
                        let mut autoderef = self.autoderef(span, actual);
                        let candidate_found = autoderef.any(|(ty, _)| {
                            if let ty::Adt(adt_deref, _) = ty.kind() {
                                self.tcx
                                    .inherent_impls(adt_deref.did())
                                    .iter()
                                    .filter_map(|def_id| self.associated_value(*def_id, item_name))
                                    .count()
                                    >= 1
                            } else {
                                false
                            }
                        });
                        let has_deref = autoderef.step_count() > 0;
                        if !candidate_found && !has_deref && unsatisfied_predicates.is_empty() {
                            if let Some((path_string, _)) = ty_str.split_once('<') {
                                ty_str_reported = path_string.to_string();
                            }
                        }
                    }
                }

                let mut err = struct_span_err!(
                    tcx.sess,
                    span,
                    E0599,
                    "no {} named `{}` found for {} `{}` in the current scope",
                    item_kind,
                    item_name,
                    actual.prefix_string(self.tcx),
                    ty_str_reported,
                );
                if actual.references_error() {
                    err.downgrade_to_delayed_bug();
                }

                if let Mode::MethodCall = mode && let SelfSource::MethodCall(cal) = source {
                    self.suggest_await_before_method(
                        &mut err, item_name, actual, cal, span,
                    );
                }
                if let Some(span) = tcx.resolutions(()).confused_type_with_std_module.get(&span) {
                    err.span_suggestion(
                        span.shrink_to_lo(),
                        "you are looking for the module in `std`, not the primitive type",
                        "std::",
                        Applicability::MachineApplicable,
                    );
                }
                if let ty::RawPtr(_) = &actual.kind() {
                    err.note(
                        "try using `<*const T>::as_ref()` to get a reference to the \
                         type behind the pointer: https://doc.rust-lang.org/std/\
                         primitive.pointer.html#method.as_ref",
                    );
                    err.note(
                        "using `<*const T>::as_ref()` on a pointer which is unaligned or points \
                         to invalid or uninitialized memory is undefined behavior",
                    );
                }

                let ty_span = match actual.kind() {
                    ty::Param(param_type) => {
                        let generics = self.tcx.generics_of(self.body_id.owner.to_def_id());
                        let type_param = generics.type_param(param_type, self.tcx);
                        Some(self.tcx.def_span(type_param.def_id))
                    }
                    ty::Adt(def, _) if def.did().is_local() => Some(tcx.def_span(def.did())),
                    _ => None,
                };

                if let Some(span) = ty_span {
                    err.span_label(
                        span,
                        format!(
                            "{item_kind} `{item_name}` not found for this {}",
                            actual.prefix_string(self.tcx)
                        ),
                    );
                }

                if let SelfSource::MethodCall(rcvr_expr) = source {
                    self.suggest_fn_call(&mut err, rcvr_expr, rcvr_ty, |output_ty| {
                        let call_expr = self
                            .tcx
                            .hir()
                            .expect_expr(self.tcx.hir().get_parent_node(rcvr_expr.hir_id));
                        let probe = self.lookup_probe(
                            span,
                            item_name,
                            output_ty,
                            call_expr,
                            ProbeScope::AllTraits,
                        );
                        probe.is_ok()
                    });
                }

                let mut custom_span_label = false;

                if !static_sources.is_empty() {
                    err.note(
                        "found the following associated functions; to be used as methods, \
                         functions must have a `self` parameter",
                    );
                    err.span_label(span, "this is an associated function, not a method");
                    custom_span_label = true;
                }
                if static_sources.len() == 1 {
                    let ty_str =
                        if let Some(CandidateSource::Impl(impl_did)) = static_sources.get(0) {
                            // When the "method" is resolved through dereferencing, we really want the
                            // original type that has the associated function for accurate suggestions.
                            // (#61411)
                            let ty = tcx.at(span).type_of(*impl_did);
                            match (&ty.peel_refs().kind(), &actual.peel_refs().kind()) {
                                (ty::Adt(def, _), ty::Adt(def_actual, _)) if def == def_actual => {
                                    // Use `actual` as it will have more `substs` filled in.
                                    self.ty_to_value_string(actual.peel_refs())
                                }
                                _ => self.ty_to_value_string(ty.peel_refs()),
                            }
                        } else {
                            self.ty_to_value_string(actual.peel_refs())
                        };
                    if let SelfSource::MethodCall(expr) = source {
                        err.span_suggestion(
                            expr.span.to(span),
                            "use associated function syntax instead",
                            format!("{}::{}", ty_str, item_name),
                            Applicability::MachineApplicable,
                        );
                    } else {
                        err.help(&format!("try with `{}::{}`", ty_str, item_name,));
                    }

                    report_candidates(span, &mut err, static_sources, sugg_span);
                } else if static_sources.len() > 1 {
                    report_candidates(span, &mut err, static_sources, sugg_span);
                }

                let mut bound_spans = vec![];
                let mut restrict_type_params = false;
                let mut unsatisfied_bounds = false;
                if item_name.name == sym::count && self.is_slice_ty(actual, span) {
                    let msg = "consider using `len` instead";
                    if let SelfSource::MethodCall(_expr) = source {
                        err.span_suggestion_short(
                            span,
                            msg,
                            "len",
                            Applicability::MachineApplicable,
                        );
                    } else {
                        err.span_label(span, msg);
                    }
                    if let Some(iterator_trait) = self.tcx.get_diagnostic_item(sym::Iterator) {
                        let iterator_trait = self.tcx.def_path_str(iterator_trait);
                        err.note(&format!("`count` is defined on `{iterator_trait}`, which `{actual}` does not implement"));
                    }
                } else if !unsatisfied_predicates.is_empty() {
                    let mut type_params = FxHashMap::default();

                    // Pick out the list of unimplemented traits on the receiver.
                    // This is used for custom error messages with the `#[rustc_on_unimplemented]` attribute.
                    let mut unimplemented_traits = FxHashMap::default();
                    let mut unimplemented_traits_only = true;
                    for (predicate, _parent_pred, cause) in &unsatisfied_predicates {
                        if let (ty::PredicateKind::Trait(p), Some(cause)) =
                            (predicate.kind().skip_binder(), cause.as_ref())
                        {
                            if p.trait_ref.self_ty() != rcvr_ty {
                                // This is necessary, not just to keep the errors clean, but also
                                // because our derived obligations can wind up with a trait ref that
                                // requires a different param_env to be correctly compared.
                                continue;
                            }
                            unimplemented_traits.entry(p.trait_ref.def_id).or_insert((
                                predicate.kind().rebind(p.trait_ref),
                                Obligation {
                                    cause: cause.clone(),
                                    param_env: self.param_env,
                                    predicate: *predicate,
                                    recursion_depth: 0,
                                },
                            ));
                        }
                    }

                    // Make sure that, if any traits other than the found ones were involved,
                    // we don't don't report an unimplemented trait.
                    // We don't want to say that `iter::Cloned` is not an iterator, just
                    // because of some non-Clone item being iterated over.
                    for (predicate, _parent_pred, _cause) in &unsatisfied_predicates {
                        match predicate.kind().skip_binder() {
                            ty::PredicateKind::Trait(p)
                                if unimplemented_traits.contains_key(&p.trait_ref.def_id) => {}
                            _ => {
                                unimplemented_traits_only = false;
                                break;
                            }
                        }
                    }

                    let mut collect_type_param_suggestions =
                        |self_ty: Ty<'tcx>, parent_pred: ty::Predicate<'tcx>, obligation: &str| {
                            // We don't care about regions here, so it's fine to skip the binder here.
                            if let (ty::Param(_), ty::PredicateKind::Trait(p)) =
                                (self_ty.kind(), parent_pred.kind().skip_binder())
                            {
                                let node = match p.trait_ref.self_ty().kind() {
                                    ty::Param(_) => {
                                        // Account for `fn` items like in `issue-35677.rs` to
                                        // suggest restricting its type params.
                                        let did = self.tcx.hir().body_owner_def_id(hir::BodyId {
                                            hir_id: self.body_id,
                                        });
                                        Some(
                                            self.tcx
                                                .hir()
                                                .get(self.tcx.hir().local_def_id_to_hir_id(did)),
                                        )
                                    }
                                    ty::Adt(def, _) => def.did().as_local().map(|def_id| {
                                        self.tcx
                                            .hir()
                                            .get(self.tcx.hir().local_def_id_to_hir_id(def_id))
                                    }),
                                    _ => None,
                                };
                                if let Some(hir::Node::Item(hir::Item { kind, .. })) = node {
                                    if let Some(g) = kind.generics() {
                                        let key = (
                                            g.tail_span_for_predicate_suggestion(),
                                            g.add_where_or_trailing_comma(),
                                        );
                                        type_params
                                            .entry(key)
                                            .or_insert_with(FxHashSet::default)
                                            .insert(obligation.to_owned());
                                    }
                                }
                            }
                        };
                    let mut bound_span_label = |self_ty: Ty<'_>, obligation: &str, quiet: &str| {
                        let msg = format!(
                            "doesn't satisfy `{}`",
                            if obligation.len() > 50 { quiet } else { obligation }
                        );
                        match &self_ty.kind() {
                            // Point at the type that couldn't satisfy the bound.
                            ty::Adt(def, _) => {
                                bound_spans.push((self.tcx.def_span(def.did()), msg))
                            }
                            // Point at the trait object that couldn't satisfy the bound.
                            ty::Dynamic(preds, _) => {
                                for pred in preds.iter() {
                                    match pred.skip_binder() {
                                        ty::ExistentialPredicate::Trait(tr) => bound_spans
                                            .push((self.tcx.def_span(tr.def_id), msg.clone())),
                                        ty::ExistentialPredicate::Projection(_)
                                        | ty::ExistentialPredicate::AutoTrait(_) => {}
                                    }
                                }
                            }
                            // Point at the closure that couldn't satisfy the bound.
                            ty::Closure(def_id, _) => bound_spans.push((
                                tcx.def_span(*def_id),
                                format!("doesn't satisfy `{}`", quiet),
                            )),
                            _ => {}
                        }
                    };
                    let mut format_pred = |pred: ty::Predicate<'tcx>| {
                        let bound_predicate = pred.kind();
                        match bound_predicate.skip_binder() {
                            ty::PredicateKind::Projection(pred) => {
                                let pred = bound_predicate.rebind(pred);
                                // `<Foo as Iterator>::Item = String`.
                                let projection_ty = pred.skip_binder().projection_ty;

                                let substs_with_infer_self = tcx.mk_substs(
                                    iter::once(tcx.mk_ty_var(ty::TyVid::from_u32(0)).into())
                                        .chain(projection_ty.substs.iter().skip(1)),
                                );

                                let quiet_projection_ty = ty::ProjectionTy {
                                    substs: substs_with_infer_self,
                                    item_def_id: projection_ty.item_def_id,
                                };

                                let term = pred.skip_binder().term;

                                let obligation = format!("{} = {}", projection_ty, term);
                                let quiet = format!("{} = {}", quiet_projection_ty, term);

                                bound_span_label(projection_ty.self_ty(), &obligation, &quiet);
                                Some((obligation, projection_ty.self_ty()))
                            }
                            ty::PredicateKind::Trait(poly_trait_ref) => {
                                let p = poly_trait_ref.trait_ref;
                                let self_ty = p.self_ty();
                                let path = p.print_only_trait_path();
                                let obligation = format!("{}: {}", self_ty, path);
                                let quiet = format!("_: {}", path);
                                bound_span_label(self_ty, &obligation, &quiet);
                                Some((obligation, self_ty))
                            }
                            _ => None,
                        }
                    };

                    // Find all the requirements that come from a local `impl` block.
                    let mut skip_list: FxHashSet<_> = Default::default();
                    let mut spanned_predicates: FxHashMap<MultiSpan, _> = Default::default();
                    for (data, p, parent_p, impl_def_id, cause) in unsatisfied_predicates
                        .iter()
                        .filter_map(|(p, parent, c)| c.as_ref().map(|c| (p, parent, c)))
                        .filter_map(|(p, parent, c)| match c.code() {
                            ObligationCauseCode::ImplDerivedObligation(ref data) => {
                                Some((&data.derived, p, parent, data.impl_def_id, data))
                            }
                            _ => None,
                        })
                    {
                        let parent_trait_ref = data.parent_trait_pred;
                        let path = parent_trait_ref.print_modifiers_and_trait_path();
                        let tr_self_ty = parent_trait_ref.skip_binder().self_ty();
                        let unsatisfied_msg = "unsatisfied trait bound introduced here";
                        let derive_msg =
                            "unsatisfied trait bound introduced in this `derive` macro";
                        match self.tcx.hir().get_if_local(impl_def_id) {
                            // Unmet obligation comes from a `derive` macro, point at it once to
                            // avoid multiple span labels pointing at the same place.
                            Some(Node::Item(hir::Item {
                                kind: hir::ItemKind::Trait(..),
                                ident,
                                ..
                            })) if matches!(
                                ident.span.ctxt().outer_expn_data().kind,
                                ExpnKind::Macro(MacroKind::Derive, _)
                            ) =>
                            {
                                let span = ident.span.ctxt().outer_expn_data().call_site;
                                let mut spans: MultiSpan = span.into();
                                spans.push_span_label(span, derive_msg);
                                let entry = spanned_predicates.entry(spans);
                                entry.or_insert_with(|| (path, tr_self_ty, Vec::new())).2.push(p);
                            }

                            Some(Node::Item(hir::Item {
                                kind: hir::ItemKind::Impl(hir::Impl { of_trait, self_ty, .. }),
                                ..
                            })) if matches!(
                                self_ty.span.ctxt().outer_expn_data().kind,
                                ExpnKind::Macro(MacroKind::Derive, _)
                            ) || matches!(
                                of_trait.as_ref().map(|t| t
                                    .path
                                    .span
                                    .ctxt()
                                    .outer_expn_data()
                                    .kind),
                                Some(ExpnKind::Macro(MacroKind::Derive, _))
                            ) =>
                            {
                                let span = self_ty.span.ctxt().outer_expn_data().call_site;
                                let mut spans: MultiSpan = span.into();
                                spans.push_span_label(span, derive_msg);
                                let entry = spanned_predicates.entry(spans);
                                entry.or_insert_with(|| (path, tr_self_ty, Vec::new())).2.push(p);
                            }

                            // Unmet obligation coming from a `trait`.
                            Some(Node::Item(hir::Item {
                                kind: hir::ItemKind::Trait(..),
                                ident,
                                span: item_span,
                                ..
                            })) if !matches!(
                                ident.span.ctxt().outer_expn_data().kind,
                                ExpnKind::Macro(MacroKind::Derive, _)
                            ) =>
                            {
                                if let Some(pred) = parent_p {
                                    // Done to add the "doesn't satisfy" `span_label`.
                                    let _ = format_pred(*pred);
                                }
                                skip_list.insert(p);
                                let mut spans = if cause.span != *item_span {
                                    let mut spans: MultiSpan = cause.span.into();
                                    spans.push_span_label(cause.span, unsatisfied_msg);
                                    spans
                                } else {
                                    ident.span.into()
                                };
                                spans.push_span_label(ident.span, "in this trait");
                                let entry = spanned_predicates.entry(spans);
                                entry.or_insert_with(|| (path, tr_self_ty, Vec::new())).2.push(p);
                            }

                            // Unmet obligation coming from an `impl`.
                            Some(Node::Item(hir::Item {
                                kind:
                                    hir::ItemKind::Impl(hir::Impl {
                                        of_trait, self_ty, generics, ..
                                    }),
                                span: item_span,
                                ..
                            })) if !matches!(
                                self_ty.span.ctxt().outer_expn_data().kind,
                                ExpnKind::Macro(MacroKind::Derive, _)
                            ) && !matches!(
                                of_trait.as_ref().map(|t| t
                                    .path
                                    .span
                                    .ctxt()
                                    .outer_expn_data()
                                    .kind),
                                Some(ExpnKind::Macro(MacroKind::Derive, _))
                            ) =>
                            {
                                let sized_pred =
                                    unsatisfied_predicates.iter().any(|(pred, _, _)| {
                                        match pred.kind().skip_binder() {
                                            ty::PredicateKind::Trait(pred) => {
                                                Some(pred.def_id())
                                                    == self.tcx.lang_items().sized_trait()
                                                    && pred.polarity == ty::ImplPolarity::Positive
                                            }
                                            _ => false,
                                        }
                                    });
                                for param in generics.params {
                                    if param.span == cause.span && sized_pred {
                                        let (sp, sugg) = match param.colon_span {
                                            Some(sp) => (sp.shrink_to_hi(), " ?Sized +"),
                                            None => (param.span.shrink_to_hi(), ": ?Sized"),
                                        };
                                        err.span_suggestion_verbose(
                                            sp,
                                            "consider relaxing the type parameter's implicit \
                                             `Sized` bound",
                                            sugg,
                                            Applicability::MachineApplicable,
                                        );
                                    }
                                }
                                if let Some(pred) = parent_p {
                                    // Done to add the "doesn't satisfy" `span_label`.
                                    let _ = format_pred(*pred);
                                }
                                skip_list.insert(p);
                                let mut spans = if cause.span != *item_span {
                                    let mut spans: MultiSpan = cause.span.into();
                                    spans.push_span_label(cause.span, unsatisfied_msg);
                                    spans
                                } else {
                                    let mut spans = Vec::with_capacity(2);
                                    if let Some(trait_ref) = of_trait {
                                        spans.push(trait_ref.path.span);
                                    }
                                    spans.push(self_ty.span);
                                    spans.into()
                                };
                                if let Some(trait_ref) = of_trait {
                                    spans.push_span_label(trait_ref.path.span, "");
                                }
                                spans.push_span_label(self_ty.span, "");

                                let entry = spanned_predicates.entry(spans);
                                entry.or_insert_with(|| (path, tr_self_ty, Vec::new())).2.push(p);
                            }
                            _ => {}
                        }
                    }
                    let mut spanned_predicates: Vec<_> = spanned_predicates.into_iter().collect();
                    spanned_predicates.sort_by_key(|(span, (_, _, _))| span.primary_span());
                    for (span, (_path, _self_ty, preds)) in spanned_predicates {
                        let mut preds: Vec<_> = preds
                            .into_iter()
                            .filter_map(|pred| format_pred(*pred))
                            .map(|(p, _)| format!("`{}`", p))
                            .collect();
                        preds.sort();
                        preds.dedup();
                        let msg = if let [pred] = &preds[..] {
                            format!("trait bound {} was not satisfied", pred)
                        } else {
                            format!(
                                "the following trait bounds were not satisfied:\n{}",
                                preds.join("\n"),
                            )
                        };
                        err.span_note(span, &msg);
                        unsatisfied_bounds = true;
                    }

                    // The requirements that didn't have an `impl` span to show.
                    let mut bound_list = unsatisfied_predicates
                        .iter()
                        .filter_map(|(pred, parent_pred, _cause)| {
                            format_pred(*pred).map(|(p, self_ty)| {
                                collect_type_param_suggestions(self_ty, *pred, &p);
                                (
                                    match parent_pred {
                                        None => format!("`{}`", &p),
                                        Some(parent_pred) => match format_pred(*parent_pred) {
                                            None => format!("`{}`", &p),
                                            Some((parent_p, _)) => {
                                                collect_type_param_suggestions(
                                                    self_ty,
                                                    *parent_pred,
                                                    &p,
                                                );
                                                format!(
                                                    "`{}`\nwhich is required by `{}`",
                                                    p, parent_p
                                                )
                                            }
                                        },
                                    },
                                    *pred,
                                )
                            })
                        })
                        .filter(|(_, pred)| !skip_list.contains(&pred))
                        .map(|(t, _)| t)
                        .enumerate()
                        .collect::<Vec<(usize, String)>>();

                    for ((span, add_where_or_comma), obligations) in type_params.into_iter() {
                        restrict_type_params = true;
                        // #74886: Sort here so that the output is always the same.
                        let mut obligations = obligations.into_iter().collect::<Vec<_>>();
                        obligations.sort();
                        err.span_suggestion_verbose(
                            span,
                            &format!(
                                "consider restricting the type parameter{s} to satisfy the \
                                 trait bound{s}",
                                s = pluralize!(obligations.len())
                            ),
                            format!("{} {}", add_where_or_comma, obligations.join(", ")),
                            Applicability::MaybeIncorrect,
                        );
                    }

                    bound_list.sort_by(|(_, a), (_, b)| a.cmp(b)); // Sort alphabetically.
                    bound_list.dedup_by(|(_, a), (_, b)| a == b); // #35677
                    bound_list.sort_by_key(|(pos, _)| *pos); // Keep the original predicate order.

                    if !bound_list.is_empty() || !skip_list.is_empty() {
                        let bound_list = bound_list
                            .into_iter()
                            .map(|(_, path)| path)
                            .collect::<Vec<_>>()
                            .join("\n");
                        let actual_prefix = actual.prefix_string(self.tcx);
                        info!("unimplemented_traits.len() == {}", unimplemented_traits.len());
                        let (primary_message, label) =
                            if unimplemented_traits.len() == 1 && unimplemented_traits_only {
                                unimplemented_traits
                                    .into_iter()
                                    .next()
                                    .map(|(_, (trait_ref, obligation))| {
                                        if trait_ref.self_ty().references_error()
                                            || actual.references_error()
                                        {
                                            // Avoid crashing.
                                            return (None, None);
                                        }
                                        let OnUnimplementedNote { message, label, .. } =
                                            self.on_unimplemented_note(trait_ref, &obligation);
                                        (message, label)
                                    })
                                    .unwrap_or((None, None))
                            } else {
                                (None, None)
                            };
                        let primary_message = primary_message.unwrap_or_else(|| format!(
                            "the {item_kind} `{item_name}` exists for {actual_prefix} `{ty_str}`, but its trait bounds were not satisfied"
                        ));
                        err.set_primary_message(&primary_message);
                        if let Some(label) = label {
                            custom_span_label = true;
                            err.span_label(span, label);
                        }
                        if !bound_list.is_empty() {
                            err.note(&format!(
                                "the following trait bounds were not satisfied:\n{bound_list}"
                            ));
                        }
                        self.suggest_derive(&mut err, &unsatisfied_predicates);

                        unsatisfied_bounds = true;
                    }
                }

                let label_span_not_found = |err: &mut Diagnostic| {
                    if unsatisfied_predicates.is_empty() {
                        err.span_label(span, format!("{item_kind} not found in `{ty_str}`"));
                        let is_string_or_ref_str = match actual.kind() {
                            ty::Ref(_, ty, _) => {
                                ty.is_str()
                                    || matches!(
                                        ty.kind(),
                                        ty::Adt(adt, _) if self.tcx.is_diagnostic_item(sym::String, adt.did())
                                    )
                            }
                            ty::Adt(adt, _) => self.tcx.is_diagnostic_item(sym::String, adt.did()),
                            _ => false,
                        };
                        if is_string_or_ref_str && item_name.name == sym::iter {
                            err.span_suggestion_verbose(
                                item_name.span,
                                "because of the in-memory representation of `&str`, to obtain \
                                 an `Iterator` over each of its codepoint use method `chars`",
                                "chars",
                                Applicability::MachineApplicable,
                            );
                        }
                        if let ty::Adt(adt, _) = rcvr_ty.kind() {
                            let mut inherent_impls_candidate = self
                                .tcx
                                .inherent_impls(adt.did())
                                .iter()
                                .copied()
                                .filter(|def_id| {
                                    if let Some(assoc) = self.associated_value(*def_id, item_name) {
                                        // Check for both mode is the same so we avoid suggesting
                                        // incorrect associated item.
                                        match (mode, assoc.fn_has_self_parameter, source) {
                                            (Mode::MethodCall, true, SelfSource::MethodCall(_)) => {
                                                // We check that the suggest type is actually
                                                // different from the received one
                                                // So we avoid suggestion method with Box<Self>
                                                // for instance
                                                self.tcx.at(span).type_of(*def_id) != actual
                                                    && self.tcx.at(span).type_of(*def_id) != rcvr_ty
                                            }
                                            (Mode::Path, false, _) => true,
                                            _ => false,
                                        }
                                    } else {
                                        false
                                    }
                                })
                                .collect::<Vec<_>>();
                            if !inherent_impls_candidate.is_empty() {
                                inherent_impls_candidate.sort();
                                inherent_impls_candidate.dedup();

                                // number of type to shows at most.
                                let limit = if inherent_impls_candidate.len() == 5 { 5 } else { 4 };
                                let type_candidates = inherent_impls_candidate
                                    .iter()
                                    .take(limit)
                                    .map(|impl_item| {
                                        format!("- `{}`", self.tcx.at(span).type_of(*impl_item))
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n");
                                let additional_types = if inherent_impls_candidate.len() > limit {
                                    format!(
                                        "\nand {} more types",
                                        inherent_impls_candidate.len() - limit
                                    )
                                } else {
                                    "".to_string()
                                };
                                err.note(&format!(
                                    "the {item_kind} was found for\n{}{}",
                                    type_candidates, additional_types
                                ));
                            }
                        }
                    } else {
                        err.span_label(span, format!("{item_kind} cannot be called on `{ty_str}` due to unsatisfied trait bounds"));
                    }
                };

                // If the method name is the name of a field with a function or closure type,
                // give a helping note that it has to be called as `(x.f)(...)`.
                if let SelfSource::MethodCall(expr) = source {
                    if !self.suggest_field_call(span, rcvr_ty, expr, item_name, &mut err)
                        && lev_candidate.is_none()
                        && !custom_span_label
                    {
                        label_span_not_found(&mut err);
                    }
                } else if !custom_span_label {
                    label_span_not_found(&mut err);
                }

                // Don't suggest (for example) `expr.field.method()` if `expr.method()`
                // doesn't exist due to unsatisfied predicates.
                if unsatisfied_predicates.is_empty() {
                    self.check_for_field_method(&mut err, source, span, actual, item_name);
                }

                self.check_for_unwrap_self(&mut err, source, span, actual, item_name);

                bound_spans.sort();
                bound_spans.dedup();
                for (span, msg) in bound_spans.into_iter() {
                    err.span_label(span, &msg);
                }

                if actual.is_numeric() && actual.is_fresh() || restrict_type_params {
                } else {
                    self.suggest_traits_to_import(
                        &mut err,
                        span,
                        rcvr_ty,
                        item_name,
                        args.map(|args| args.len()),
                        source,
                        out_of_scope_traits,
                        &unsatisfied_predicates,
                        unsatisfied_bounds,
                    );
                }

                // Don't emit a suggestion if we found an actual method
                // that had unsatisfied trait bounds
                if unsatisfied_predicates.is_empty() && actual.is_enum() {
                    let adt_def = actual.ty_adt_def().expect("enum is not an ADT");
                    if let Some(suggestion) = lev_distance::find_best_match_for_name(
                        &adt_def.variants().iter().map(|s| s.name).collect::<Vec<_>>(),
                        item_name.name,
                        None,
                    ) {
                        err.span_suggestion(
                            span,
                            "there is a variant with a similar name",
                            suggestion,
                            Applicability::MaybeIncorrect,
                        );
                    }
                }

                if item_name.name == sym::as_str && actual.peel_refs().is_str() {
                    let msg = "remove this method call";
                    let mut fallback_span = true;
                    if let SelfSource::MethodCall(expr) = source {
                        let call_expr =
                            self.tcx.hir().expect_expr(self.tcx.hir().get_parent_node(expr.hir_id));
                        if let Some(span) = call_expr.span.trim_start(expr.span) {
                            err.span_suggestion(span, msg, "", Applicability::MachineApplicable);
                            fallback_span = false;
                        }
                    }
                    if fallback_span {
                        err.span_label(span, msg);
                    }
                } else if let Some(lev_candidate) = lev_candidate {
                    // Don't emit a suggestion if we found an actual method
                    // that had unsatisfied trait bounds
                    if unsatisfied_predicates.is_empty() {
                        let def_kind = lev_candidate.kind.as_def_kind();
                        // Methods are defined within the context of a struct and their first parameter is always self,
                        // which represents the instance of the struct the method is being called on
                        // Associated functions don’t take self as a parameter and
                        // they are not methods because they don’t have an instance of the struct to work with.
                        if def_kind == DefKind::AssocFn && lev_candidate.fn_has_self_parameter {
                            err.span_suggestion(
                                span,
                                &format!("there is a method with a similar name",),
                                lev_candidate.name,
                                Applicability::MaybeIncorrect,
                            );
                        } else {
                            err.span_suggestion(
                                span,
                                &format!(
                                    "there is {} {} with a similar name",
                                    def_kind.article(),
                                    def_kind.descr(lev_candidate.def_id),
                                ),
                                lev_candidate.name,
                                Applicability::MaybeIncorrect,
                            );
                        }
                    }
                }

                self.check_for_deref_method(&mut err, source, rcvr_ty, item_name);

                return Some(err);
            }

            MethodError::Ambiguity(sources) => {
                let mut err = struct_span_err!(
                    self.sess(),
                    item_name.span,
                    E0034,
                    "multiple applicable items in scope"
                );
                err.span_label(item_name.span, format!("multiple `{}` found", item_name));

                report_candidates(span, &mut err, sources, sugg_span);
                err.emit();
            }

            MethodError::PrivateMatch(kind, def_id, out_of_scope_traits) => {
                let kind = kind.descr(def_id);
                let mut err = struct_span_err!(
                    self.tcx.sess,
                    item_name.span,
                    E0624,
                    "{} `{}` is private",
                    kind,
                    item_name
                );
                err.span_label(item_name.span, &format!("private {}", kind));
                let sp = self
                    .tcx
                    .hir()
                    .span_if_local(def_id)
                    .unwrap_or_else(|| self.tcx.def_span(def_id));
                err.span_label(sp, &format!("private {} defined here", kind));
                self.suggest_valid_traits(&mut err, out_of_scope_traits);
                err.emit();
            }

            MethodError::IllegalSizedBound(candidates, needs_mut, bound_span) => {
                let msg = format!("the `{}` method cannot be invoked on a trait object", item_name);
                let mut err = self.sess().struct_span_err(span, &msg);
                err.span_label(bound_span, "this has a `Sized` requirement");
                if !candidates.is_empty() {
                    let help = format!(
                        "{an}other candidate{s} {were} found in the following trait{s}, perhaps \
                         add a `use` for {one_of_them}:",
                        an = if candidates.len() == 1 { "an" } else { "" },
                        s = pluralize!(candidates.len()),
                        were = pluralize!("was", candidates.len()),
                        one_of_them = if candidates.len() == 1 { "it" } else { "one_of_them" },
                    );
                    self.suggest_use_candidates(&mut err, help, candidates);
                }
                if let ty::Ref(region, t_type, mutability) = rcvr_ty.kind() {
                    if needs_mut {
                        let trait_type = self.tcx.mk_ref(
                            *region,
                            ty::TypeAndMut { ty: *t_type, mutbl: mutability.invert() },
                        );
                        err.note(&format!("you need `{}` instead of `{}`", trait_type, rcvr_ty));
                    }
                }
                err.emit();
            }

            MethodError::BadReturnType => bug!("no return type expectations but got BadReturnType"),
        }
        None
    }

    fn suggest_field_call(
        &self,
        span: Span,
        rcvr_ty: Ty<'tcx>,
        expr: &hir::Expr<'_>,
        item_name: Ident,
        err: &mut Diagnostic,
    ) -> bool {
        let tcx = self.tcx;
        let field_receiver = self.autoderef(span, rcvr_ty).find_map(|(ty, _)| match ty.kind() {
            ty::Adt(def, substs) if !def.is_enum() => {
                let variant = &def.non_enum_variant();
                tcx.find_field_index(item_name, variant).map(|index| {
                    let field = &variant.fields[index];
                    let field_ty = field.ty(tcx, substs);
                    (field, field_ty)
                })
            }
            _ => None,
        });
        if let Some((field, field_ty)) = field_receiver {
            let scope = tcx.parent_module(self.body_id).to_def_id();
            let is_accessible = field.vis.is_accessible_from(scope, tcx);

            if is_accessible {
                if self.is_fn_ty(field_ty, span) {
                    let expr_span = expr.span.to(item_name.span);
                    err.multipart_suggestion(
                        &format!(
                            "to call the function stored in `{}`, \
                                         surround the field access with parentheses",
                            item_name,
                        ),
                        vec![
                            (expr_span.shrink_to_lo(), '('.to_string()),
                            (expr_span.shrink_to_hi(), ')'.to_string()),
                        ],
                        Applicability::MachineApplicable,
                    );
                } else {
                    let call_expr = tcx.hir().expect_expr(tcx.hir().get_parent_node(expr.hir_id));

                    if let Some(span) = call_expr.span.trim_start(item_name.span) {
                        err.span_suggestion(
                            span,
                            "remove the arguments",
                            "",
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
            }

            let field_kind = if is_accessible { "field" } else { "private field" };
            err.span_label(item_name.span, format!("{}, not a method", field_kind));
            return true;
        }
        false
    }

    fn suggest_constraining_numerical_ty(
        &self,
        tcx: TyCtxt<'tcx>,
        actual: Ty<'tcx>,
        source: SelfSource<'_>,
        span: Span,
        item_kind: &str,
        item_name: Ident,
        ty_str: &str,
    ) -> bool {
        let found_candidate = all_traits(self.tcx)
            .into_iter()
            .any(|info| self.associated_value(info.def_id, item_name).is_some());
        let found_assoc = |ty: Ty<'tcx>| {
            simplify_type(tcx, ty, TreatParams::AsInfer)
                .and_then(|simp| {
                    tcx.incoherent_impls(simp)
                        .iter()
                        .find_map(|&id| self.associated_value(id, item_name))
                })
                .is_some()
        };
        let found_candidate = found_candidate
            || found_assoc(tcx.types.i8)
            || found_assoc(tcx.types.i16)
            || found_assoc(tcx.types.i32)
            || found_assoc(tcx.types.i64)
            || found_assoc(tcx.types.i128)
            || found_assoc(tcx.types.u8)
            || found_assoc(tcx.types.u16)
            || found_assoc(tcx.types.u32)
            || found_assoc(tcx.types.u64)
            || found_assoc(tcx.types.u128)
            || found_assoc(tcx.types.f32)
            || found_assoc(tcx.types.f32);
        if found_candidate
            && actual.is_numeric()
            && !actual.has_concrete_skeleton()
            && let SelfSource::MethodCall(expr) = source
        {
            let mut err = struct_span_err!(
                tcx.sess,
                span,
                E0689,
                "can't call {} `{}` on ambiguous numeric type `{}`",
                item_kind,
                item_name,
                ty_str
            );
            let concrete_type = if actual.is_integral() { "i32" } else { "f32" };
            match expr.kind {
                ExprKind::Lit(ref lit) => {
                    // numeric literal
                    let snippet = tcx
                        .sess
                        .source_map()
                        .span_to_snippet(lit.span)
                        .unwrap_or_else(|_| "<numeric literal>".to_owned());

                    // If this is a floating point literal that ends with '.',
                    // get rid of it to stop this from becoming a member access.
                    let snippet = snippet.strip_suffix('.').unwrap_or(&snippet);

                    err.span_suggestion(
                        lit.span,
                        &format!(
                            "you must specify a concrete type for this numeric value, \
                                         like `{}`",
                            concrete_type
                        ),
                        format!("{snippet}_{concrete_type}"),
                        Applicability::MaybeIncorrect,
                    );
                }
                ExprKind::Path(QPath::Resolved(_, path)) => {
                    // local binding
                    if let hir::def::Res::Local(hir_id) = path.res {
                        let span = tcx.hir().span(hir_id);
                        let filename = tcx.sess.source_map().span_to_filename(span);

                        let parent_node =
                            self.tcx.hir().get(self.tcx.hir().get_parent_node(hir_id));
                        let msg = format!(
                            "you must specify a type for this binding, like `{}`",
                            concrete_type,
                        );

                        match (filename, parent_node) {
                            (
                                FileName::Real(_),
                                Node::Local(hir::Local {
                                    source: hir::LocalSource::Normal,
                                    ty,
                                    ..
                                }),
                            ) => {
                                let type_span = ty.map(|ty| ty.span.with_lo(span.hi())).unwrap_or(span.shrink_to_hi());
                                err.span_suggestion(
                                    // account for `let x: _ = 42;`
                                    //                   ^^^
                                    type_span,
                                    &msg,
                                    format!(": {concrete_type}"),
                                    Applicability::MaybeIncorrect,
                                );
                            }
                            _ => {
                                err.span_label(span, msg);
                            }
                        }
                    }
                }
                _ => {}
            }
            err.emit();
            return true;
        }
        false
    }

    fn check_for_field_method(
        &self,
        err: &mut Diagnostic,
        source: SelfSource<'tcx>,
        span: Span,
        actual: Ty<'tcx>,
        item_name: Ident,
    ) {
        if let SelfSource::MethodCall(expr) = source
        && let mod_id = self.tcx.parent_module(expr.hir_id).to_def_id()
        && let Some((fields, substs)) =
            self.get_field_candidates_considering_privacy(span, actual, mod_id)
        {
            let call_expr = self.tcx.hir().expect_expr(self.tcx.hir().get_parent_node(expr.hir_id));

            let lang_items = self.tcx.lang_items();
            let never_mention_traits = [
                lang_items.clone_trait(),
                lang_items.deref_trait(),
                lang_items.deref_mut_trait(),
                self.tcx.get_diagnostic_item(sym::AsRef),
                self.tcx.get_diagnostic_item(sym::AsMut),
                self.tcx.get_diagnostic_item(sym::Borrow),
                self.tcx.get_diagnostic_item(sym::BorrowMut),
            ];
            let candidate_fields: Vec<_> = fields
                .filter_map(|candidate_field| {
                    self.check_for_nested_field_satisfying(
                        span,
                        &|_, field_ty| {
                            self.lookup_probe(
                                span,
                                item_name,
                                field_ty,
                                call_expr,
                                ProbeScope::TraitsInScope,
                            )
                            .map_or(false, |pick| {
                                !never_mention_traits
                                    .iter()
                                    .flatten()
                                    .any(|def_id| self.tcx.parent(pick.item.def_id) == *def_id)
                            })
                        },
                        candidate_field,
                        substs,
                        vec![],
                        mod_id,
                    )
                })
                .map(|field_path| {
                    field_path
                        .iter()
                        .map(|id| id.name.to_ident_string())
                        .collect::<Vec<String>>()
                        .join(".")
                })
                .collect();

            let len = candidate_fields.len();
            if len > 0 {
                err.span_suggestions(
                    item_name.span.shrink_to_lo(),
                    format!(
                        "{} of the expressions' fields {} a method of the same name",
                        if len > 1 { "some" } else { "one" },
                        if len > 1 { "have" } else { "has" },
                    ),
                    candidate_fields.iter().map(|path| format!("{path}.")),
                    Applicability::MaybeIncorrect,
                );
            }
        }
    }

    fn check_for_unwrap_self(
        &self,
        err: &mut Diagnostic,
        source: SelfSource<'tcx>,
        span: Span,
        actual: Ty<'tcx>,
        item_name: Ident,
    ) {
        let tcx = self.tcx;
        let SelfSource::MethodCall(expr) = source else { return; };
        let call_expr = tcx.hir().expect_expr(tcx.hir().get_parent_node(expr.hir_id));

        let ty::Adt(kind, substs) = actual.kind() else { return; };
        if !kind.is_enum() {
            return;
        }

        let matching_variants: Vec<_> = kind
            .variants()
            .iter()
            .flat_map(|variant| {
                let [field] = &variant.fields[..] else { return None; };
                let field_ty = field.ty(tcx, substs);

                // Skip `_`, since that'll just lead to ambiguity.
                if self.resolve_vars_if_possible(field_ty).is_ty_var() {
                    return None;
                }

                self.lookup_probe(span, item_name, field_ty, call_expr, ProbeScope::AllTraits)
                    .ok()
                    .map(|pick| (variant, field, pick))
            })
            .collect();

        let ret_ty_matches = |diagnostic_item| {
            if let Some(ret_ty) = self
                .ret_coercion
                .as_ref()
                .map(|c| self.resolve_vars_if_possible(c.borrow().expected_ty()))
                && let ty::Adt(kind, _) = ret_ty.kind()
                && tcx.get_diagnostic_item(diagnostic_item) == Some(kind.did())
            {
                true
            } else {
                false
            }
        };

        match &matching_variants[..] {
            [(_, field, pick)] => {
                let self_ty = field.ty(tcx, substs);
                err.span_note(
                    tcx.def_span(pick.item.def_id),
                    &format!("the method `{item_name}` exists on the type `{self_ty}`"),
                );
                let (article, kind, variant, question) =
                    if Some(kind.did()) == tcx.get_diagnostic_item(sym::Result) {
                        ("a", "Result", "Err", ret_ty_matches(sym::Result))
                    } else if Some(kind.did()) == tcx.get_diagnostic_item(sym::Option) {
                        ("an", "Option", "None", ret_ty_matches(sym::Option))
                    } else {
                        return;
                    };
                if question {
                    err.span_suggestion_verbose(
                        expr.span.shrink_to_hi(),
                        format!(
                            "use the `?` operator to extract the `{self_ty}` value, propagating \
                            {article} `{kind}::{variant}` value to the caller"
                        ),
                        "?",
                        Applicability::MachineApplicable,
                    );
                } else {
                    err.span_suggestion_verbose(
                        expr.span.shrink_to_hi(),
                        format!(
                            "consider using `{kind}::expect` to unwrap the `{self_ty}` value, \
                             panicking if the value is {article} `{kind}::{variant}`"
                        ),
                        ".expect(\"REASON\")",
                        Applicability::HasPlaceholders,
                    );
                }
            }
            // FIXME(compiler-errors): Support suggestions for other matching enum variants
            _ => {}
        }
    }

    pub(crate) fn note_unmet_impls_on_type(
        &self,
        err: &mut Diagnostic,
        errors: Vec<FulfillmentError<'tcx>>,
    ) {
        let all_local_types_needing_impls =
            errors.iter().all(|e| match e.obligation.predicate.kind().skip_binder() {
                ty::PredicateKind::Trait(pred) => match pred.self_ty().kind() {
                    ty::Adt(def, _) => def.did().is_local(),
                    _ => false,
                },
                _ => false,
            });
        let mut preds: Vec<_> = errors
            .iter()
            .filter_map(|e| match e.obligation.predicate.kind().skip_binder() {
                ty::PredicateKind::Trait(pred) => Some(pred),
                _ => None,
            })
            .collect();
        preds.sort_by_key(|pred| (pred.def_id(), pred.self_ty()));
        let def_ids = preds
            .iter()
            .filter_map(|pred| match pred.self_ty().kind() {
                ty::Adt(def, _) => Some(def.did()),
                _ => None,
            })
            .collect::<FxHashSet<_>>();
        let mut spans: MultiSpan = def_ids
            .iter()
            .filter_map(|def_id| {
                let span = self.tcx.def_span(*def_id);
                if span.is_dummy() { None } else { Some(span) }
            })
            .collect::<Vec<_>>()
            .into();

        for pred in &preds {
            match pred.self_ty().kind() {
                ty::Adt(def, _) if def.did().is_local() => {
                    spans.push_span_label(
                        self.tcx.def_span(def.did()),
                        format!("must implement `{}`", pred.trait_ref.print_only_trait_path()),
                    );
                }
                _ => {}
            }
        }

        if all_local_types_needing_impls && spans.primary_span().is_some() {
            let msg = if preds.len() == 1 {
                format!(
                    "an implementation of `{}` might be missing for `{}`",
                    preds[0].trait_ref.print_only_trait_path(),
                    preds[0].self_ty()
                )
            } else {
                format!(
                    "the following type{} would have to `impl` {} required trait{} for this \
                     operation to be valid",
                    pluralize!(def_ids.len()),
                    if def_ids.len() == 1 { "its" } else { "their" },
                    pluralize!(preds.len()),
                )
            };
            err.span_note(spans, &msg);
        }

        let preds: Vec<_> = errors
            .iter()
            .map(|e| (e.obligation.predicate, None, Some(e.obligation.cause.clone())))
            .collect();
        self.suggest_derive(err, &preds);
    }

    fn suggest_derive(
        &self,
        err: &mut Diagnostic,
        unsatisfied_predicates: &[(
            ty::Predicate<'tcx>,
            Option<ty::Predicate<'tcx>>,
            Option<ObligationCause<'tcx>>,
        )],
    ) {
        let mut derives = Vec::<(String, Span, Symbol)>::new();
        let mut traits = Vec::<Span>::new();
        for (pred, _, _) in unsatisfied_predicates {
            let ty::PredicateKind::Trait(trait_pred) = pred.kind().skip_binder() else { continue };
            let adt = match trait_pred.self_ty().ty_adt_def() {
                Some(adt) if adt.did().is_local() => adt,
                _ => continue,
            };
            if let Some(diagnostic_name) = self.tcx.get_diagnostic_name(trait_pred.def_id()) {
                let can_derive = match diagnostic_name {
                    sym::Default => !adt.is_enum(),
                    sym::Eq
                    | sym::PartialEq
                    | sym::Ord
                    | sym::PartialOrd
                    | sym::Clone
                    | sym::Copy
                    | sym::Hash
                    | sym::Debug => true,
                    _ => false,
                };
                if can_derive {
                    let self_name = trait_pred.self_ty().to_string();
                    let self_span = self.tcx.def_span(adt.did());
                    if let Some(poly_trait_ref) = pred.to_opt_poly_trait_pred() {
                        for super_trait in supertraits(self.tcx, poly_trait_ref.to_poly_trait_ref())
                        {
                            if let Some(parent_diagnostic_name) =
                                self.tcx.get_diagnostic_name(super_trait.def_id())
                            {
                                derives.push((
                                    self_name.clone(),
                                    self_span,
                                    parent_diagnostic_name,
                                ));
                            }
                        }
                    }
                    derives.push((self_name, self_span, diagnostic_name));
                } else {
                    traits.push(self.tcx.def_span(trait_pred.def_id()));
                }
            } else {
                traits.push(self.tcx.def_span(trait_pred.def_id()));
            }
        }
        traits.sort();
        traits.dedup();

        derives.sort();
        derives.dedup();

        let mut derives_grouped = Vec::<(String, Span, String)>::new();
        for (self_name, self_span, trait_name) in derives.into_iter() {
            if let Some((last_self_name, _, ref mut last_trait_names)) = derives_grouped.last_mut()
            {
                if last_self_name == &self_name {
                    last_trait_names.push_str(format!(", {}", trait_name).as_str());
                    continue;
                }
            }
            derives_grouped.push((self_name, self_span, trait_name.to_string()));
        }

        let len = traits.len();
        if len > 0 {
            let span: MultiSpan = traits.into();
            err.span_note(
                span,
                &format!("the following trait{} must be implemented", pluralize!(len),),
            );
        }

        for (self_name, self_span, traits) in &derives_grouped {
            err.span_suggestion_verbose(
                self_span.shrink_to_lo(),
                &format!("consider annotating `{}` with `#[derive({})]`", self_name, traits),
                format!("#[derive({})]\n", traits),
                Applicability::MaybeIncorrect,
            );
        }
    }

    fn check_for_deref_method(
        &self,
        err: &mut Diagnostic,
        self_source: SelfSource<'tcx>,
        rcvr_ty: Ty<'tcx>,
        item_name: Ident,
    ) {
        let SelfSource::QPath(ty) = self_source else { return; };
        for (deref_ty, _) in self.autoderef(rustc_span::DUMMY_SP, rcvr_ty).skip(1) {
            if let Ok(pick) = self.probe_for_name(
                ty.span,
                Mode::Path,
                item_name,
                IsSuggestion(true),
                deref_ty,
                ty.hir_id,
                ProbeScope::TraitsInScope,
            ) {
                if deref_ty.is_suggestable(self.tcx, true)
                    // If this method receives `&self`, then the provided
                    // argument _should_ coerce, so it's valid to suggest
                    // just changing the path.
                    && pick.item.fn_has_self_parameter
                    && let Some(self_ty) =
                        self.tcx.fn_sig(pick.item.def_id).inputs().skip_binder().get(0)
                    && self_ty.is_ref()
                {
                    let suggested_path = match deref_ty.kind() {
                        ty::Bool
                        | ty::Char
                        | ty::Int(_)
                        | ty::Uint(_)
                        | ty::Float(_)
                        | ty::Adt(_, _)
                        | ty::Str
                        | ty::Projection(_)
                        | ty::Param(_) => format!("{deref_ty}"),
                        _ => format!("<{deref_ty}>"),
                    };
                    err.span_suggestion_verbose(
                        ty.span,
                        format!("the function `{item_name}` is implemented on `{deref_ty}`"),
                        suggested_path,
                        Applicability::MaybeIncorrect,
                    );
                } else {
                    err.span_note(
                        ty.span,
                        format!("the function `{item_name}` is implemented on `{deref_ty}`"),
                    );
                }
                return;
            }
        }
    }

    /// Print out the type for use in value namespace.
    fn ty_to_value_string(&self, ty: Ty<'tcx>) -> String {
        match ty.kind() {
            ty::Adt(def, substs) => format!("{}", ty::Instance::new(def.did(), substs)),
            _ => self.ty_to_string(ty),
        }
    }

    fn suggest_await_before_method(
        &self,
        err: &mut Diagnostic,
        item_name: Ident,
        ty: Ty<'tcx>,
        call: &hir::Expr<'_>,
        span: Span,
    ) {
        let output_ty = match self.get_impl_future_output_ty(ty) {
            Some(output_ty) => self.resolve_vars_if_possible(output_ty).skip_binder(),
            _ => return,
        };
        let method_exists = self.method_exists(item_name, output_ty, call.hir_id, true);
        debug!("suggest_await_before_method: is_method_exist={}", method_exists);
        if method_exists {
            err.span_suggestion_verbose(
                span.shrink_to_lo(),
                "consider `await`ing on the `Future` and calling the method on its `Output`",
                "await.",
                Applicability::MaybeIncorrect,
            );
        }
    }

    fn suggest_use_candidates(&self, err: &mut Diagnostic, msg: String, candidates: Vec<DefId>) {
        let parent_map = self.tcx.visible_parent_map(());

        // Separate out candidates that must be imported with a glob, because they are named `_`
        // and cannot be referred with their identifier.
        let (candidates, globs): (Vec<_>, Vec<_>) = candidates.into_iter().partition(|trait_did| {
            if let Some(parent_did) = parent_map.get(trait_did) {
                // If the item is re-exported as `_`, we should suggest a glob-import instead.
                if *parent_did != self.tcx.parent(*trait_did)
                    && self
                        .tcx
                        .module_children(*parent_did)
                        .iter()
                        .filter(|child| child.res.opt_def_id() == Some(*trait_did))
                        .all(|child| child.ident.name == kw::Underscore)
                {
                    return false;
                }
            }

            true
        });

        let module_did = self.tcx.parent_module(self.body_id);
        let (module, _, _) = self.tcx.hir().get_module(module_did);
        let span = module.spans.inject_use_span;

        let path_strings = candidates.iter().map(|trait_did| {
            format!("use {};\n", with_crate_prefix!(self.tcx.def_path_str(*trait_did)),)
        });

        let glob_path_strings = globs.iter().map(|trait_did| {
            let parent_did = parent_map.get(trait_did).unwrap();
            format!(
                "use {}::*; // trait {}\n",
                with_crate_prefix!(self.tcx.def_path_str(*parent_did)),
                self.tcx.item_name(*trait_did),
            )
        });

        err.span_suggestions(
            span,
            &msg,
            path_strings.chain(glob_path_strings),
            Applicability::MaybeIncorrect,
        );
    }

    fn suggest_valid_traits(
        &self,
        err: &mut Diagnostic,
        valid_out_of_scope_traits: Vec<DefId>,
    ) -> bool {
        if !valid_out_of_scope_traits.is_empty() {
            let mut candidates = valid_out_of_scope_traits;
            candidates.sort();
            candidates.dedup();

            // `TryFrom` and `FromIterator` have no methods
            let edition_fix = candidates
                .iter()
                .find(|did| self.tcx.is_diagnostic_item(sym::TryInto, **did))
                .copied();

            err.help("items from traits can only be used if the trait is in scope");
            let msg = format!(
                "the following {traits_are} implemented but not in scope; \
                 perhaps add a `use` for {one_of_them}:",
                traits_are = if candidates.len() == 1 { "trait is" } else { "traits are" },
                one_of_them = if candidates.len() == 1 { "it" } else { "one of them" },
            );

            self.suggest_use_candidates(err, msg, candidates);
            if let Some(did) = edition_fix {
                err.note(&format!(
                    "'{}' is included in the prelude starting in Edition 2021",
                    with_crate_prefix!(self.tcx.def_path_str(did))
                ));
            }

            true
        } else {
            false
        }
    }

    fn suggest_traits_to_import(
        &self,
        err: &mut Diagnostic,
        span: Span,
        rcvr_ty: Ty<'tcx>,
        item_name: Ident,
        inputs_len: Option<usize>,
        source: SelfSource<'tcx>,
        valid_out_of_scope_traits: Vec<DefId>,
        unsatisfied_predicates: &[(
            ty::Predicate<'tcx>,
            Option<ty::Predicate<'tcx>>,
            Option<ObligationCause<'tcx>>,
        )],
        unsatisfied_bounds: bool,
    ) {
        let mut alt_rcvr_sugg = false;
        if let (SelfSource::MethodCall(rcvr), false) = (source, unsatisfied_bounds) {
            debug!(?span, ?item_name, ?rcvr_ty, ?rcvr);
            let skippable = [
                self.tcx.lang_items().clone_trait(),
                self.tcx.lang_items().deref_trait(),
                self.tcx.lang_items().deref_mut_trait(),
                self.tcx.lang_items().drop_trait(),
                self.tcx.get_diagnostic_item(sym::AsRef),
            ];
            // Try alternative arbitrary self types that could fulfill this call.
            // FIXME: probe for all types that *could* be arbitrary self-types, not
            // just this list.
            for (rcvr_ty, post) in &[
                (rcvr_ty, ""),
                (self.tcx.mk_mut_ref(self.tcx.lifetimes.re_erased, rcvr_ty), "&mut "),
                (self.tcx.mk_imm_ref(self.tcx.lifetimes.re_erased, rcvr_ty), "&"),
            ] {
                match self.lookup_probe(span, item_name, *rcvr_ty, rcvr, ProbeScope::AllTraits) {
                    Ok(pick) => {
                        // If the method is defined for the receiver we have, it likely wasn't `use`d.
                        // We point at the method, but we just skip the rest of the check for arbitrary
                        // self types and rely on the suggestion to `use` the trait from
                        // `suggest_valid_traits`.
                        let did = Some(pick.item.container_id(self.tcx));
                        let skip = skippable.contains(&did);
                        if pick.autoderefs == 0 && !skip {
                            err.span_label(
                                pick.item.ident(self.tcx).span,
                                &format!("the method is available for `{}` here", rcvr_ty),
                            );
                        }
                        break;
                    }
                    Err(MethodError::Ambiguity(_)) => {
                        // If the method is defined (but ambiguous) for the receiver we have, it is also
                        // likely we haven't `use`d it. It may be possible that if we `Box`/`Pin`/etc.
                        // the receiver, then it might disambiguate this method, but I think these
                        // suggestions are generally misleading (see #94218).
                        break;
                    }
                    _ => {}
                }

                for (rcvr_ty, pre) in &[
                    (self.tcx.mk_lang_item(*rcvr_ty, LangItem::OwnedBox), "Box::new"),
                    (self.tcx.mk_lang_item(*rcvr_ty, LangItem::Pin), "Pin::new"),
                    (self.tcx.mk_diagnostic_item(*rcvr_ty, sym::Arc), "Arc::new"),
                    (self.tcx.mk_diagnostic_item(*rcvr_ty, sym::Rc), "Rc::new"),
                ] {
                    if let Some(new_rcvr_t) = *rcvr_ty
                        && let Ok(pick) = self.lookup_probe(
                            span,
                            item_name,
                            new_rcvr_t,
                            rcvr,
                            ProbeScope::AllTraits,
                        )
                    {
                        debug!("try_alt_rcvr: pick candidate {:?}", pick);
                        let did = Some(pick.item.container_id(self.tcx));
                        // We don't want to suggest a container type when the missing
                        // method is `.clone()` or `.deref()` otherwise we'd suggest
                        // `Arc::new(foo).clone()`, which is far from what the user wants.
                        // Explicitly ignore the `Pin::as_ref()` method as `Pin` does not
                        // implement the `AsRef` trait.
                        let skip = skippable.contains(&did)
                            || (("Pin::new" == *pre) && (sym::as_ref == item_name.name))
                            || inputs_len.map_or(false, |inputs_len| pick.item.kind == ty::AssocKind::Fn && self.tcx.fn_sig(pick.item.def_id).skip_binder().inputs().len() != inputs_len);
                        // Make sure the method is defined for the *actual* receiver: we don't
                        // want to treat `Box<Self>` as a receiver if it only works because of
                        // an autoderef to `&self`
                        if pick.autoderefs == 0 && !skip {
                            err.span_label(
                                pick.item.ident(self.tcx).span,
                                &format!("the method is available for `{}` here", new_rcvr_t),
                            );
                            err.multipart_suggestion(
                                "consider wrapping the receiver expression with the \
                                    appropriate type",
                                vec![
                                    (rcvr.span.shrink_to_lo(), format!("{}({}", pre, post)),
                                    (rcvr.span.shrink_to_hi(), ")".to_string()),
                                ],
                                Applicability::MaybeIncorrect,
                            );
                            // We don't care about the other suggestions.
                            alt_rcvr_sugg = true;
                        }
                    }
                }
            }
        }
        if self.suggest_valid_traits(err, valid_out_of_scope_traits) {
            return;
        }

        let type_is_local = self.type_derefs_to_local(span, rcvr_ty, source);

        let mut arbitrary_rcvr = vec![];
        // There are no traits implemented, so lets suggest some traits to
        // implement, by finding ones that have the item name, and are
        // legal to implement.
        let mut candidates = all_traits(self.tcx)
            .into_iter()
            // Don't issue suggestions for unstable traits since they're
            // unlikely to be implementable anyway
            .filter(|info| match self.tcx.lookup_stability(info.def_id) {
                Some(attr) => attr.level.is_stable(),
                None => true,
            })
            .filter(|info| {
                // We approximate the coherence rules to only suggest
                // traits that are legal to implement by requiring that
                // either the type or trait is local. Multi-dispatch means
                // this isn't perfect (that is, there are cases when
                // implementing a trait would be legal but is rejected
                // here).
                unsatisfied_predicates.iter().all(|(p, _, _)| {
                    match p.kind().skip_binder() {
                        // Hide traits if they are present in predicates as they can be fixed without
                        // having to implement them.
                        ty::PredicateKind::Trait(t) => t.def_id() == info.def_id,
                        ty::PredicateKind::Projection(p) => {
                            p.projection_ty.item_def_id == info.def_id
                        }
                        _ => false,
                    }
                }) && (type_is_local || info.def_id.is_local())
                    && self
                        .associated_value(info.def_id, item_name)
                        .filter(|item| {
                            if let ty::AssocKind::Fn = item.kind {
                                let id = item
                                    .def_id
                                    .as_local()
                                    .map(|def_id| self.tcx.hir().local_def_id_to_hir_id(def_id));
                                if let Some(hir::Node::TraitItem(hir::TraitItem {
                                    kind: hir::TraitItemKind::Fn(fn_sig, method),
                                    ..
                                })) = id.map(|id| self.tcx.hir().get(id))
                                {
                                    let self_first_arg = match method {
                                        hir::TraitFn::Required([ident, ..]) => {
                                            ident.name == kw::SelfLower
                                        }
                                        hir::TraitFn::Provided(body_id) => {
                                            self.tcx.hir().body(*body_id).params.first().map_or(
                                                false,
                                                |param| {
                                                    matches!(
                                                        param.pat.kind,
                                                        hir::PatKind::Binding(_, _, ident, _)
                                                            if ident.name == kw::SelfLower
                                                    )
                                                },
                                            )
                                        }
                                        _ => false,
                                    };

                                    if !fn_sig.decl.implicit_self.has_implicit_self()
                                        && self_first_arg
                                    {
                                        if let Some(ty) = fn_sig.decl.inputs.get(0) {
                                            arbitrary_rcvr.push(ty.span);
                                        }
                                        return false;
                                    }
                                }
                            }
                            // We only want to suggest public or local traits (#45781).
                            item.visibility(self.tcx).is_public() || info.def_id.is_local()
                        })
                        .is_some()
            })
            .collect::<Vec<_>>();
        for span in &arbitrary_rcvr {
            err.span_label(
                *span,
                "the method might not be found because of this arbitrary self type",
            );
        }
        if alt_rcvr_sugg {
            return;
        }

        if !candidates.is_empty() {
            // Sort from most relevant to least relevant.
            candidates.sort_by(|a, b| a.cmp(b).reverse());
            candidates.dedup();

            let param_type = match rcvr_ty.kind() {
                ty::Param(param) => Some(param),
                ty::Ref(_, ty, _) => match ty.kind() {
                    ty::Param(param) => Some(param),
                    _ => None,
                },
                _ => None,
            };
            err.help(if param_type.is_some() {
                "items from traits can only be used if the type parameter is bounded by the trait"
            } else {
                "items from traits can only be used if the trait is implemented and in scope"
            });
            let candidates_len = candidates.len();
            let message = |action| {
                format!(
                    "the following {traits_define} an item `{name}`, perhaps you need to {action} \
                     {one_of_them}:",
                    traits_define =
                        if candidates_len == 1 { "trait defines" } else { "traits define" },
                    action = action,
                    one_of_them = if candidates_len == 1 { "it" } else { "one of them" },
                    name = item_name,
                )
            };
            // Obtain the span for `param` and use it for a structured suggestion.
            if let Some(param) = param_type {
                let generics = self.tcx.generics_of(self.body_id.owner.to_def_id());
                let type_param = generics.type_param(param, self.tcx);
                let hir = self.tcx.hir();
                if let Some(def_id) = type_param.def_id.as_local() {
                    let id = hir.local_def_id_to_hir_id(def_id);
                    // Get the `hir::Param` to verify whether it already has any bounds.
                    // We do this to avoid suggesting code that ends up as `T: FooBar`,
                    // instead we suggest `T: Foo + Bar` in that case.
                    match hir.get(id) {
                        Node::GenericParam(param) => {
                            enum Introducer {
                                Plus,
                                Colon,
                                Nothing,
                            }
                            let ast_generics = hir.get_generics(id.owner).unwrap();
                            let (sp, mut introducer) = if let Some(span) =
                                ast_generics.bounds_span_for_suggestions(def_id)
                            {
                                (span, Introducer::Plus)
                            } else if let Some(colon_span) = param.colon_span {
                                (colon_span.shrink_to_hi(), Introducer::Nothing)
                            } else {
                                (param.span.shrink_to_hi(), Introducer::Colon)
                            };
                            if matches!(
                                param.kind,
                                hir::GenericParamKind::Type { synthetic: true, .. },
                            ) {
                                introducer = Introducer::Plus
                            }
                            let trait_def_ids: FxHashSet<DefId> = ast_generics
                                .bounds_for_param(def_id)
                                .flat_map(|bp| bp.bounds.iter())
                                .filter_map(|bound| bound.trait_ref()?.trait_def_id())
                                .collect();
                            if !candidates.iter().any(|t| trait_def_ids.contains(&t.def_id)) {
                                err.span_suggestions(
                                    sp,
                                    &message(format!(
                                        "restrict type parameter `{}` with",
                                        param.name.ident(),
                                    )),
                                    candidates.iter().map(|t| {
                                        format!(
                                            "{} {}",
                                            match introducer {
                                                Introducer::Plus => " +",
                                                Introducer::Colon => ":",
                                                Introducer::Nothing => "",
                                            },
                                            self.tcx.def_path_str(t.def_id),
                                        )
                                    }),
                                    Applicability::MaybeIncorrect,
                                );
                            }
                            return;
                        }
                        Node::Item(hir::Item {
                            kind: hir::ItemKind::Trait(.., bounds, _),
                            ident,
                            ..
                        }) => {
                            let (sp, sep, article) = if bounds.is_empty() {
                                (ident.span.shrink_to_hi(), ":", "a")
                            } else {
                                (bounds.last().unwrap().span().shrink_to_hi(), " +", "another")
                            };
                            err.span_suggestions(
                                sp,
                                &message(format!("add {} supertrait for", article)),
                                candidates.iter().map(|t| {
                                    format!("{} {}", sep, self.tcx.def_path_str(t.def_id),)
                                }),
                                Applicability::MaybeIncorrect,
                            );
                            return;
                        }
                        _ => {}
                    }
                }
            }

            let (potential_candidates, explicitly_negative) = if param_type.is_some() {
                // FIXME: Even though negative bounds are not implemented, we could maybe handle
                // cases where a positive bound implies a negative impl.
                (candidates, Vec::new())
            } else if let Some(simp_rcvr_ty) =
                simplify_type(self.tcx, rcvr_ty, TreatParams::AsPlaceholder)
            {
                let mut potential_candidates = Vec::new();
                let mut explicitly_negative = Vec::new();
                for candidate in candidates {
                    // Check if there's a negative impl of `candidate` for `rcvr_ty`
                    if self
                        .tcx
                        .all_impls(candidate.def_id)
                        .filter(|imp_did| {
                            self.tcx.impl_polarity(*imp_did) == ty::ImplPolarity::Negative
                        })
                        .any(|imp_did| {
                            let imp = self.tcx.impl_trait_ref(imp_did).unwrap();
                            let imp_simp =
                                simplify_type(self.tcx, imp.self_ty(), TreatParams::AsPlaceholder);
                            imp_simp.map_or(false, |s| s == simp_rcvr_ty)
                        })
                    {
                        explicitly_negative.push(candidate);
                    } else {
                        potential_candidates.push(candidate);
                    }
                }
                (potential_candidates, explicitly_negative)
            } else {
                // We don't know enough about `recv_ty` to make proper suggestions.
                (candidates, Vec::new())
            };

            let action = if let Some(param) = param_type {
                format!("restrict type parameter `{}` with", param)
            } else {
                // FIXME: it might only need to be imported into scope, not implemented.
                "implement".to_string()
            };
            match &potential_candidates[..] {
                [] => {}
                [trait_info] if trait_info.def_id.is_local() => {
                    err.span_note(
                        self.tcx.def_span(trait_info.def_id),
                        &format!(
                            "`{}` defines an item `{}`, perhaps you need to {} it",
                            self.tcx.def_path_str(trait_info.def_id),
                            item_name,
                            action
                        ),
                    );
                }
                trait_infos => {
                    let mut msg = message(action);
                    for (i, trait_info) in trait_infos.iter().enumerate() {
                        msg.push_str(&format!(
                            "\ncandidate #{}: `{}`",
                            i + 1,
                            self.tcx.def_path_str(trait_info.def_id),
                        ));
                    }
                    err.note(&msg);
                }
            }
            match &explicitly_negative[..] {
                [] => {}
                [trait_info] => {
                    let msg = format!(
                        "the trait `{}` defines an item `{}`, but is explicitly unimplemented",
                        self.tcx.def_path_str(trait_info.def_id),
                        item_name
                    );
                    err.note(&msg);
                }
                trait_infos => {
                    let mut msg = format!(
                        "the following traits define an item `{}`, but are explicitly unimplemented:",
                        item_name
                    );
                    for trait_info in trait_infos {
                        msg.push_str(&format!("\n{}", self.tcx.def_path_str(trait_info.def_id)));
                    }
                    err.note(&msg);
                }
            }
        }
    }

    /// Checks whether there is a local type somewhere in the chain of
    /// autoderefs of `rcvr_ty`.
    fn type_derefs_to_local(
        &self,
        span: Span,
        rcvr_ty: Ty<'tcx>,
        source: SelfSource<'tcx>,
    ) -> bool {
        fn is_local(ty: Ty<'_>) -> bool {
            match ty.kind() {
                ty::Adt(def, _) => def.did().is_local(),
                ty::Foreign(did) => did.is_local(),
                ty::Dynamic(tr, ..) => tr.principal().map_or(false, |d| d.def_id().is_local()),
                ty::Param(_) => true,

                // Everything else (primitive types, etc.) is effectively
                // non-local (there are "edge" cases, e.g., `(LocalType,)`, but
                // the noise from these sort of types is usually just really
                // annoying, rather than any sort of help).
                _ => false,
            }
        }

        // This occurs for UFCS desugaring of `T::method`, where there is no
        // receiver expression for the method call, and thus no autoderef.
        if let SelfSource::QPath(_) = source {
            return is_local(self.resolve_vars_with_obligations(rcvr_ty));
        }

        self.autoderef(span, rcvr_ty).any(|(ty, _)| is_local(ty))
    }
}

#[derive(Copy, Clone, Debug)]
pub enum SelfSource<'a> {
    QPath(&'a hir::Ty<'a>),
    MethodCall(&'a hir::Expr<'a> /* rcvr */),
}

#[derive(Copy, Clone)]
pub struct TraitInfo {
    pub def_id: DefId,
}

impl PartialEq for TraitInfo {
    fn eq(&self, other: &TraitInfo) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for TraitInfo {}
impl PartialOrd for TraitInfo {
    fn partial_cmp(&self, other: &TraitInfo) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TraitInfo {
    fn cmp(&self, other: &TraitInfo) -> Ordering {
        // Local crates are more important than remote ones (local:
        // `cnum == 0`), and otherwise we throw in the defid for totality.

        let lhs = (other.def_id.krate, other.def_id);
        let rhs = (self.def_id.krate, self.def_id);
        lhs.cmp(&rhs)
    }
}

/// Retrieves all traits in this crate and any dependent crates,
/// and wraps them into `TraitInfo` for custom sorting.
pub fn all_traits(tcx: TyCtxt<'_>) -> Vec<TraitInfo> {
    tcx.all_traits().map(|def_id| TraitInfo { def_id }).collect()
}

fn print_disambiguation_help<'tcx>(
    item_name: Ident,
    args: Option<&'tcx [hir::Expr<'tcx>]>,
    err: &mut Diagnostic,
    trait_name: String,
    rcvr_ty: Ty<'_>,
    kind: ty::AssocKind,
    def_id: DefId,
    span: Span,
    candidate: Option<usize>,
    source_map: &source_map::SourceMap,
    fn_has_self_parameter: bool,
) {
    let mut applicability = Applicability::MachineApplicable;
    let (span, sugg) = if let (ty::AssocKind::Fn, Some(args)) = (kind, args) {
        let args = format!(
            "({}{})",
            if rcvr_ty.is_region_ptr() {
                if rcvr_ty.is_mutable_ptr() { "&mut " } else { "&" }
            } else {
                ""
            },
            args.iter()
                .map(|arg| source_map.span_to_snippet(arg.span).unwrap_or_else(|_| {
                    applicability = Applicability::HasPlaceholders;
                    "_".to_owned()
                }))
                .collect::<Vec<_>>()
                .join(", "),
        );
        let trait_name = if !fn_has_self_parameter {
            format!("<{} as {}>", rcvr_ty, trait_name)
        } else {
            trait_name
        };
        (span, format!("{}::{}{}", trait_name, item_name, args))
    } else {
        (span.with_hi(item_name.span.lo()), format!("<{} as {}>::", rcvr_ty, trait_name))
    };
    err.span_suggestion_verbose(
        span,
        &format!(
            "disambiguate the {} for {}",
            kind.as_def_kind().descr(def_id),
            if let Some(candidate) = candidate {
                format!("candidate #{}", candidate)
            } else {
                "the candidate".to_string()
            },
        ),
        sugg,
        applicability,
    );
}
