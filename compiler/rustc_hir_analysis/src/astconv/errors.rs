use crate::astconv::AstConv;
use crate::errors::{ManualImplementation, MissingTypeParams};
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{pluralize, struct_span_err, Applicability, Diagnostic, ErrorGuaranteed};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_infer::traits::FulfillmentError;
use rustc_middle::ty::{self, Ty};
use rustc_session::parse::feature_err;
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::symbol::{sym, Ident};
use rustc_span::{Span, Symbol, DUMMY_SP};

use std::collections::BTreeSet;

impl<'o, 'tcx> dyn AstConv<'tcx> + 'o {
    /// On missing type parameters, emit an E0393 error and provide a structured suggestion using
    /// the type parameter's name as a placeholder.
    pub(crate) fn complain_about_missing_type_params(
        &self,
        missing_type_params: Vec<Symbol>,
        def_id: DefId,
        span: Span,
        empty_generic_args: bool,
    ) {
        if missing_type_params.is_empty() {
            return;
        }

        self.tcx().sess.emit_err(MissingTypeParams {
            span,
            def_span: self.tcx().def_span(def_id),
            span_snippet: self.tcx().sess.source_map().span_to_snippet(span).ok(),
            missing_type_params,
            empty_generic_args,
        });
    }

    /// When the code is using the `Fn` traits directly, instead of the `Fn(A) -> B` syntax, emit
    /// an error and attempt to build a reasonable structured suggestion.
    pub(crate) fn complain_about_internal_fn_trait(
        &self,
        span: Span,
        trait_def_id: DefId,
        trait_segment: &'_ hir::PathSegment<'_>,
        is_impl: bool,
    ) {
        if self.tcx().features().unboxed_closures {
            return;
        }

        let trait_def = self.tcx().trait_def(trait_def_id);
        if !trait_def.paren_sugar {
            if trait_segment.args().parenthesized {
                // For now, require that parenthetical notation be used only with `Fn()` etc.
                let mut err = feature_err(
                    &self.tcx().sess.parse_sess,
                    sym::unboxed_closures,
                    span,
                    "parenthetical notation is only stable when used with `Fn`-family traits",
                );
                err.emit();
            }

            return;
        }

        let sess = self.tcx().sess;

        if !trait_segment.args().parenthesized {
            // For now, require that parenthetical notation be used only with `Fn()` etc.
            let mut err = feature_err(
                &sess.parse_sess,
                sym::unboxed_closures,
                span,
                "the precise format of `Fn`-family traits' type parameters is subject to change",
            );
            // Do not suggest the other syntax if we are in trait impl:
            // the desugaring would contain an associated type constraint.
            if !is_impl {
                let args = trait_segment
                    .args
                    .as_ref()
                    .and_then(|args| args.args.get(0))
                    .and_then(|arg| match arg {
                        hir::GenericArg::Type(ty) => match ty.kind {
                            hir::TyKind::Tup(t) => t
                                .iter()
                                .map(|e| sess.source_map().span_to_snippet(e.span))
                                .collect::<Result<Vec<_>, _>>()
                                .map(|a| a.join(", ")),
                            _ => sess.source_map().span_to_snippet(ty.span),
                        }
                        .map(|s| format!("({})", s))
                        .ok(),
                        _ => None,
                    })
                    .unwrap_or_else(|| "()".to_string());
                let ret = trait_segment
                    .args()
                    .bindings
                    .iter()
                    .find_map(|b| match (b.ident.name == sym::Output, &b.kind) {
                        (true, hir::TypeBindingKind::Equality { term }) => {
                            let span = match term {
                                hir::Term::Ty(ty) => ty.span,
                                hir::Term::Const(c) => self.tcx().hir().span(c.hir_id),
                            };
                            sess.source_map().span_to_snippet(span).ok()
                        }
                        _ => None,
                    })
                    .unwrap_or_else(|| "()".to_string());
                err.span_suggestion(
                    span,
                    "use parenthetical notation instead",
                    format!("{}{} -> {}", trait_segment.ident, args, ret),
                    Applicability::MaybeIncorrect,
                );
            }
            err.emit();
        }

        if is_impl {
            let trait_name = self.tcx().def_path_str(trait_def_id);
            self.tcx().sess.emit_err(ManualImplementation { span, trait_name });
        }
    }

    pub(crate) fn complain_about_assoc_type_not_found<I>(
        &self,
        all_candidates: impl Fn() -> I,
        ty_param_name: &str,
        assoc_name: Ident,
        span: Span,
    ) -> ErrorGuaranteed
    where
        I: Iterator<Item = ty::PolyTraitRef<'tcx>>,
    {
        // The fallback span is needed because `assoc_name` might be an `Fn()`'s `Output` without a
        // valid span, so we point at the whole path segment instead.
        let span = if assoc_name.span != DUMMY_SP { assoc_name.span } else { span };
        let mut err = struct_span_err!(
            self.tcx().sess,
            span,
            E0220,
            "associated type `{}` not found for `{}`",
            assoc_name,
            ty_param_name
        );

        let all_candidate_names: Vec<_> = all_candidates()
            .flat_map(|r| self.tcx().associated_items(r.def_id()).in_definition_order())
            .filter_map(
                |item| if item.kind == ty::AssocKind::Type { Some(item.name) } else { None },
            )
            .collect();

        if let (Some(suggested_name), true) = (
            find_best_match_for_name(&all_candidate_names, assoc_name.name, None),
            assoc_name.span != DUMMY_SP,
        ) {
            err.span_suggestion(
                assoc_name.span,
                "there is an associated type with a similar name",
                suggested_name,
                Applicability::MaybeIncorrect,
            );
            return err.emit();
        }

        // If we didn't find a good item in the supertraits (or couldn't get
        // the supertraits), like in ItemCtxt, then look more generally from
        // all visible traits. If there's one clear winner, just suggest that.

        let visible_traits: Vec<_> = self
            .tcx()
            .all_traits()
            .filter(|trait_def_id| {
                let viz = self.tcx().visibility(*trait_def_id);
                let def_id = self.item_def_id();
                viz.is_accessible_from(def_id, self.tcx())
            })
            .collect();

        let wider_candidate_names: Vec<_> = visible_traits
            .iter()
            .flat_map(|trait_def_id| {
                self.tcx().associated_items(*trait_def_id).in_definition_order()
            })
            .filter_map(
                |item| if item.kind == ty::AssocKind::Type { Some(item.name) } else { None },
            )
            .collect();

        if let (Some(suggested_name), true) = (
            find_best_match_for_name(&wider_candidate_names, assoc_name.name, None),
            assoc_name.span != DUMMY_SP,
        ) {
            if let [best_trait] = visible_traits
                .iter()
                .filter(|trait_def_id| {
                    self.tcx()
                        .associated_items(*trait_def_id)
                        .filter_by_name_unhygienic(suggested_name)
                        .any(|item| item.kind == ty::AssocKind::Type)
                })
                .collect::<Vec<_>>()[..]
            {
                err.span_label(
                    assoc_name.span,
                    format!(
                        "there is a similarly named associated type `{suggested_name}` in the trait `{}`",
                        self.tcx().def_path_str(*best_trait)
                    ),
                );
                return err.emit();
            }
        }

        err.span_label(span, format!("associated type `{}` not found", assoc_name));
        err.emit()
    }

    pub(crate) fn complain_about_ambiguous_inherent_assoc_type(
        &self,
        name: Ident,
        candidates: Vec<DefId>,
        span: Span,
    ) -> ErrorGuaranteed {
        let mut err = struct_span_err!(
            self.tcx().sess,
            name.span,
            E0034,
            "multiple applicable items in scope"
        );
        err.span_label(name.span, format!("multiple `{name}` found"));
        self.note_ambiguous_inherent_assoc_type(&mut err, candidates, span);
        err.emit()
    }

    // FIXME(fmease): Heavily adapted from `rustc_hir_typeck::method::suggest`. Deduplicate.
    fn note_ambiguous_inherent_assoc_type(
        &self,
        err: &mut Diagnostic,
        candidates: Vec<DefId>,
        span: Span,
    ) {
        let tcx = self.tcx();

        // Dynamic limit to avoid hiding just one candidate, which is silly.
        let limit = if candidates.len() == 5 { 5 } else { 4 };

        for (index, &item) in candidates.iter().take(limit).enumerate() {
            let impl_ = tcx.impl_of_method(item).unwrap();

            let note_span = if item.is_local() {
                Some(tcx.def_span(item))
            } else if impl_.is_local() {
                Some(tcx.def_span(impl_))
            } else {
                None
            };

            let title = if candidates.len() > 1 {
                format!("candidate #{}", index + 1)
            } else {
                "the candidate".into()
            };

            let impl_ty = tcx.at(span).type_of(impl_).subst_identity();
            let note = format!("{title} is defined in an impl for the type `{impl_ty}`");

            if let Some(span) = note_span {
                err.span_note(span, &note);
            } else {
                err.note(&note);
            }
        }
        if candidates.len() > limit {
            err.note(&format!("and {} others", candidates.len() - limit));
        }
    }

    // FIXME(inherent_associated_types): Find similarly named associated types and suggest them.
    pub(crate) fn complain_about_inherent_assoc_type_not_found(
        &self,
        name: Ident,
        self_ty: Ty<'tcx>,
        candidates: Vec<(DefId, (DefId, DefId))>,
        fulfillment_errors: Vec<FulfillmentError<'tcx>>,
        span: Span,
    ) -> ErrorGuaranteed {
        // FIXME(fmease): This was copied in parts from an old version of `rustc_hir_typeck::method::suggest`.
        // Either
        // * update this code by applying changes similar to #106702 or by taking a
        //   Vec<(DefId, (DefId, DefId), Option<Vec<FulfillmentError<'tcx>>>)> or
        // * deduplicate this code across the two crates.

        let tcx = self.tcx();

        let adt_did = self_ty.ty_adt_def().map(|def| def.did());
        let add_def_label = |err: &mut Diagnostic| {
            if let Some(did) = adt_did {
                err.span_label(
                    tcx.def_span(did),
                    format!("associated item `{name}` not found for this {}", tcx.def_descr(did)),
                );
            }
        };

        if fulfillment_errors.is_empty() {
            // FIXME(fmease): Copied from `rustc_hir_typeck::method::probe`. Deduplicate.

            let limit = if candidates.len() == 5 { 5 } else { 4 };
            let type_candidates = candidates
                .iter()
                .take(limit)
                .map(|&(impl_, _)| format!("- `{}`", tcx.at(span).type_of(impl_).subst_identity()))
                .collect::<Vec<_>>()
                .join("\n");
            let additional_types = if candidates.len() > limit {
                format!("\nand {} more types", candidates.len() - limit)
            } else {
                String::new()
            };

            let mut err = struct_span_err!(
                tcx.sess,
                name.span,
                E0220,
                "associated type `{name}` not found for `{self_ty}` in the current scope"
            );
            err.span_label(name.span, format!("associated item not found in `{self_ty}`"));
            err.note(&format!(
                "the associated type was found for\n{type_candidates}{additional_types}",
            ));
            add_def_label(&mut err);
            return err.emit();
        }

        let mut bound_spans = Vec::new();

        let mut bound_span_label = |self_ty: Ty<'_>, obligation: &str, quiet: &str| {
            let msg = format!(
                "doesn't satisfy `{}`",
                if obligation.len() > 50 { quiet } else { obligation }
            );
            match &self_ty.kind() {
                // Point at the type that couldn't satisfy the bound.
                ty::Adt(def, _) => bound_spans.push((tcx.def_span(def.did()), msg)),
                // Point at the trait object that couldn't satisfy the bound.
                ty::Dynamic(preds, _, _) => {
                    for pred in preds.iter() {
                        match pred.skip_binder() {
                            ty::ExistentialPredicate::Trait(tr) => {
                                bound_spans.push((tcx.def_span(tr.def_id), msg.clone()))
                            }
                            ty::ExistentialPredicate::Projection(_)
                            | ty::ExistentialPredicate::AutoTrait(_) => {}
                        }
                    }
                }
                // Point at the closure that couldn't satisfy the bound.
                ty::Closure(def_id, _) => {
                    bound_spans.push((tcx.def_span(*def_id), format!("doesn't satisfy `{quiet}`")))
                }
                _ => {}
            }
        };

        let format_pred = |pred: ty::Predicate<'tcx>| {
            let bound_predicate = pred.kind();
            match bound_predicate.skip_binder() {
                ty::PredicateKind::Clause(ty::Clause::Projection(pred)) => {
                    let pred = bound_predicate.rebind(pred);
                    // `<Foo as Iterator>::Item = String`.
                    let projection_ty = pred.skip_binder().projection_ty;

                    let substs_with_infer_self = tcx.mk_substs_from_iter(
                        std::iter::once(tcx.mk_ty_var(ty::TyVid::from_u32(0)).into())
                            .chain(projection_ty.substs.iter().skip(1)),
                    );

                    let quiet_projection_ty =
                        tcx.mk_alias_ty(projection_ty.def_id, substs_with_infer_self);

                    let term = pred.skip_binder().term;

                    let obligation = format!("{projection_ty} = {term}");
                    let quiet = format!("{quiet_projection_ty} = {term}");

                    bound_span_label(projection_ty.self_ty(), &obligation, &quiet);
                    Some((obligation, projection_ty.self_ty()))
                }
                ty::PredicateKind::Clause(ty::Clause::Trait(poly_trait_ref)) => {
                    let p = poly_trait_ref.trait_ref;
                    let self_ty = p.self_ty();
                    let path = p.print_only_trait_path();
                    let obligation = format!("{self_ty}: {path}");
                    let quiet = format!("_: {path}");
                    bound_span_label(self_ty, &obligation, &quiet);
                    Some((obligation, self_ty))
                }
                _ => None,
            }
        };

        // FIXME(fmease): `rustc_hir_typeck::method::suggest` uses a `skip_list` to filter out some bounds.
        // I would do the same here if it didn't mean more code duplication.
        let mut bounds: Vec<_> = fulfillment_errors
            .into_iter()
            .map(|error| error.root_obligation.predicate)
            .filter_map(format_pred)
            .map(|(p, _)| format!("`{}`", p))
            .collect();
        bounds.sort();
        bounds.dedup();

        let mut err = tcx.sess.struct_span_err(
            name.span,
            &format!("the associated type `{name}` exists for `{self_ty}`, but its trait bounds were not satisfied")
        );
        if !bounds.is_empty() {
            err.note(&format!(
                "the following trait bounds were not satisfied:\n{}",
                bounds.join("\n")
            ));
        }
        err.span_label(
            name.span,
            format!("associated type cannot be referenced on `{self_ty}` due to unsatisfied trait bounds")
        );

        bound_spans.sort();
        bound_spans.dedup();
        for (span, msg) in bound_spans {
            if !tcx.sess.source_map().is_span_accessible(span) {
                continue;
            }
            err.span_label(span, &msg);
        }
        add_def_label(&mut err);
        err.emit()
    }

    /// When there are any missing associated types, emit an E0191 error and attempt to supply a
    /// reasonable suggestion on how to write it. For the case of multiple associated types in the
    /// same trait bound have the same name (as they come from different supertraits), we instead
    /// emit a generic note suggesting using a `where` clause to constraint instead.
    pub(crate) fn complain_about_missing_associated_types(
        &self,
        associated_types: FxHashMap<Span, BTreeSet<DefId>>,
        potential_assoc_types: Vec<Span>,
        trait_bounds: &[hir::PolyTraitRef<'_>],
    ) {
        if associated_types.values().all(|v| v.is_empty()) {
            return;
        }
        let tcx = self.tcx();
        // FIXME: Marked `mut` so that we can replace the spans further below with a more
        // appropriate one, but this should be handled earlier in the span assignment.
        let mut associated_types: FxHashMap<Span, Vec<_>> = associated_types
            .into_iter()
            .map(|(span, def_ids)| {
                (span, def_ids.into_iter().map(|did| tcx.associated_item(did)).collect())
            })
            .collect();
        let mut names = vec![];

        // Account for things like `dyn Foo + 'a`, like in tests `issue-22434.rs` and
        // `issue-22560.rs`.
        let mut trait_bound_spans: Vec<Span> = vec![];
        for (span, items) in &associated_types {
            if !items.is_empty() {
                trait_bound_spans.push(*span);
            }
            for assoc_item in items {
                let trait_def_id = assoc_item.container_id(tcx);
                names.push(format!(
                    "`{}` (from trait `{}`)",
                    assoc_item.name,
                    tcx.def_path_str(trait_def_id),
                ));
            }
        }
        if let ([], [bound]) = (&potential_assoc_types[..], &trait_bounds) {
            match bound.trait_ref.path.segments {
                // FIXME: `trait_ref.path.span` can point to a full path with multiple
                // segments, even though `trait_ref.path.segments` is of length `1`. Work
                // around that bug here, even though it should be fixed elsewhere.
                // This would otherwise cause an invalid suggestion. For an example, look at
                // `tests/ui/issues/issue-28344.rs` where instead of the following:
                //
                //   error[E0191]: the value of the associated type `Output`
                //                 (from trait `std::ops::BitXor`) must be specified
                //   --> $DIR/issue-28344.rs:4:17
                //    |
                // LL |     let x: u8 = BitXor::bitor(0 as u8, 0 as u8);
                //    |                 ^^^^^^ help: specify the associated type:
                //    |                              `BitXor<Output = Type>`
                //
                // we would output:
                //
                //   error[E0191]: the value of the associated type `Output`
                //                 (from trait `std::ops::BitXor`) must be specified
                //   --> $DIR/issue-28344.rs:4:17
                //    |
                // LL |     let x: u8 = BitXor::bitor(0 as u8, 0 as u8);
                //    |                 ^^^^^^^^^^^^^ help: specify the associated type:
                //    |                                     `BitXor::bitor<Output = Type>`
                [segment] if segment.args.is_none() => {
                    trait_bound_spans = vec![segment.ident.span];
                    associated_types = associated_types
                        .into_iter()
                        .map(|(_, items)| (segment.ident.span, items))
                        .collect();
                }
                _ => {}
            }
        }
        names.sort();
        trait_bound_spans.sort();
        let mut err = struct_span_err!(
            tcx.sess,
            trait_bound_spans,
            E0191,
            "the value of the associated type{} {} must be specified",
            pluralize!(names.len()),
            names.join(", "),
        );
        let mut suggestions = vec![];
        let mut types_count = 0;
        let mut where_constraints = vec![];
        let mut already_has_generics_args_suggestion = false;
        for (span, assoc_items) in &associated_types {
            let mut names: FxHashMap<_, usize> = FxHashMap::default();
            for item in assoc_items {
                types_count += 1;
                *names.entry(item.name).or_insert(0) += 1;
            }
            let mut dupes = false;
            for item in assoc_items {
                let prefix = if names[&item.name] > 1 {
                    let trait_def_id = item.container_id(tcx);
                    dupes = true;
                    format!("{}::", tcx.def_path_str(trait_def_id))
                } else {
                    String::new()
                };
                if let Some(sp) = tcx.hir().span_if_local(item.def_id) {
                    err.span_label(sp, format!("`{}{}` defined here", prefix, item.name));
                }
            }
            if potential_assoc_types.len() == assoc_items.len() {
                // When the amount of missing associated types equals the number of
                // extra type arguments present. A suggesting to replace the generic args with
                // associated types is already emitted.
                already_has_generics_args_suggestion = true;
            } else if let (Ok(snippet), false) =
                (tcx.sess.source_map().span_to_snippet(*span), dupes)
            {
                let types: Vec<_> =
                    assoc_items.iter().map(|item| format!("{} = Type", item.name)).collect();
                let code = if snippet.ends_with('>') {
                    // The user wrote `Trait<'a>` or similar and we don't have a type we can
                    // suggest, but at least we can clue them to the correct syntax
                    // `Trait<'a, Item = Type>` while accounting for the `<'a>` in the
                    // suggestion.
                    format!("{}, {}>", &snippet[..snippet.len() - 1], types.join(", "))
                } else {
                    // The user wrote `Iterator`, so we don't have a type we can suggest, but at
                    // least we can clue them to the correct syntax `Iterator<Item = Type>`.
                    format!("{}<{}>", snippet, types.join(", "))
                };
                suggestions.push((*span, code));
            } else if dupes {
                where_constraints.push(*span);
            }
        }
        let where_msg = "consider introducing a new type parameter, adding `where` constraints \
                         using the fully-qualified path to the associated types";
        if !where_constraints.is_empty() && suggestions.is_empty() {
            // If there are duplicates associated type names and a single trait bound do not
            // use structured suggestion, it means that there are multiple supertraits with
            // the same associated type name.
            err.help(where_msg);
        }
        if suggestions.len() != 1 || already_has_generics_args_suggestion {
            // We don't need this label if there's an inline suggestion, show otherwise.
            for (span, assoc_items) in &associated_types {
                let mut names: FxHashMap<_, usize> = FxHashMap::default();
                for item in assoc_items {
                    types_count += 1;
                    *names.entry(item.name).or_insert(0) += 1;
                }
                let mut label = vec![];
                for item in assoc_items {
                    let postfix = if names[&item.name] > 1 {
                        let trait_def_id = item.container_id(tcx);
                        format!(" (from trait `{}`)", tcx.def_path_str(trait_def_id))
                    } else {
                        String::new()
                    };
                    label.push(format!("`{}`{}", item.name, postfix));
                }
                if !label.is_empty() {
                    err.span_label(
                        *span,
                        format!(
                            "associated type{} {} must be specified",
                            pluralize!(label.len()),
                            label.join(", "),
                        ),
                    );
                }
            }
        }
        if !suggestions.is_empty() {
            err.multipart_suggestion(
                &format!("specify the associated type{}", pluralize!(types_count)),
                suggestions,
                Applicability::HasPlaceholders,
            );
            if !where_constraints.is_empty() {
                err.span_help(where_constraints, where_msg);
            }
        }
        err.emit();
    }
}
