use crate::astconv::AstConv;
use crate::errors::{
    self, AssocTypeBindingNotAllowed, ManualImplementation, MissingTypeParams,
    ParenthesizedFnTraitExpansion,
};
use crate::fluent_generated as fluent;
use crate::traits::error_reporting::report_object_safety_error;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::sorted_map::SortedMap;
use rustc_data_structures::unord::UnordMap;
use rustc_errors::{
    codes::*, pluralize, struct_span_code_err, Applicability, Diag, ErrorGuaranteed,
};
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_infer::traits::FulfillmentError;
use rustc_middle::query::Key;
use rustc_middle::ty::{self, suggest_constraining_type_param, Ty, TyCtxt, TypeVisitableExt};
use rustc_session::parse::feature_err;
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::symbol::{sym, Ident};
use rustc_span::{Span, Symbol, DUMMY_SP};
use rustc_trait_selection::traits::object_safety_violations_for_assoc_item;

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

        self.tcx().dcx().emit_err(MissingTypeParams {
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
            if trait_segment.args().parenthesized == hir::GenericArgsParentheses::ParenSugar {
                // For now, require that parenthetical notation be used only with `Fn()` etc.
                feature_err(
                    &self.tcx().sess,
                    sym::unboxed_closures,
                    span,
                    "parenthetical notation is only stable when used with `Fn`-family traits",
                )
                .emit();
            }

            return;
        }

        let sess = self.tcx().sess;

        if trait_segment.args().parenthesized != hir::GenericArgsParentheses::ParenSugar {
            // For now, require that parenthetical notation be used only with `Fn()` etc.
            let mut err = feature_err(
                sess,
                sym::unboxed_closures,
                span,
                "the precise format of `Fn`-family traits' type parameters is subject to change",
            );
            // Do not suggest the other syntax if we are in trait impl:
            // the desugaring would contain an associated type constraint.
            if !is_impl {
                err.span_suggestion(
                    span,
                    "use parenthetical notation instead",
                    fn_trait_to_string(self.tcx(), trait_segment, true),
                    Applicability::MaybeIncorrect,
                );
            }
            err.emit();
        }

        if is_impl {
            let trait_name = self.tcx().def_path_str(trait_def_id);
            self.tcx().dcx().emit_err(ManualImplementation { span, trait_name });
        }
    }

    pub(super) fn complain_about_assoc_item_not_found<I>(
        &self,
        all_candidates: impl Fn() -> I,
        ty_param_name: &str,
        ty_param_def_id: Option<LocalDefId>,
        assoc_kind: ty::AssocKind,
        assoc_name: Ident,
        span: Span,
        binding: Option<&hir::TypeBinding<'tcx>>,
    ) -> ErrorGuaranteed
    where
        I: Iterator<Item = ty::PolyTraitRef<'tcx>>,
    {
        let tcx = self.tcx();

        // First and foremost, provide a more user-friendly & “intuitive” error on kind mismatches.
        if let Some(assoc_item) = all_candidates().find_map(|r| {
            tcx.associated_items(r.def_id())
                .filter_by_name_unhygienic(assoc_name.name)
                .find(|item| tcx.hygienic_eq(assoc_name, item.ident(tcx), r.def_id()))
        }) {
            return self.complain_about_assoc_kind_mismatch(
                assoc_item, assoc_kind, assoc_name, span, binding,
            );
        }

        let assoc_kind_str = super::assoc_kind_str(assoc_kind);

        // The fallback span is needed because `assoc_name` might be an `Fn()`'s `Output` without a
        // valid span, so we point at the whole path segment instead.
        let is_dummy = assoc_name.span == DUMMY_SP;

        let mut err = errors::AssocItemNotFound {
            span: if is_dummy { span } else { assoc_name.span },
            assoc_name,
            assoc_kind: assoc_kind_str,
            ty_param_name,
            label: None,
            sugg: None,
        };

        if is_dummy {
            err.label = Some(errors::AssocItemNotFoundLabel::NotFound { span });
            return tcx.dcx().emit_err(err);
        }

        let all_candidate_names: Vec<_> = all_candidates()
            .flat_map(|r| tcx.associated_items(r.def_id()).in_definition_order())
            .filter_map(|item| {
                (!item.is_impl_trait_in_trait() && item.kind == assoc_kind).then_some(item.name)
            })
            .collect();

        if let Some(suggested_name) =
            find_best_match_for_name(&all_candidate_names, assoc_name.name, None)
        {
            err.sugg = Some(errors::AssocItemNotFoundSugg::Similar {
                span: assoc_name.span,
                assoc_kind: assoc_kind_str,
                suggested_name,
            });
            return tcx.dcx().emit_err(err);
        }

        // If we didn't find a good item in the supertraits (or couldn't get
        // the supertraits), like in ItemCtxt, then look more generally from
        // all visible traits. If there's one clear winner, just suggest that.

        let visible_traits: Vec<_> = tcx
            .all_traits()
            .filter(|trait_def_id| {
                let viz = tcx.visibility(*trait_def_id);
                let def_id = self.item_def_id();
                viz.is_accessible_from(def_id, tcx)
            })
            .collect();

        let wider_candidate_names: Vec<_> = visible_traits
            .iter()
            .flat_map(|trait_def_id| tcx.associated_items(*trait_def_id).in_definition_order())
            .filter_map(|item| {
                (!item.is_impl_trait_in_trait() && item.kind == assoc_kind).then_some(item.name)
            })
            .collect();

        if let Some(suggested_name) =
            find_best_match_for_name(&wider_candidate_names, assoc_name.name, None)
        {
            if let [best_trait] = visible_traits
                .iter()
                .copied()
                .filter(|trait_def_id| {
                    tcx.associated_items(trait_def_id)
                        .filter_by_name_unhygienic(suggested_name)
                        .any(|item| item.kind == assoc_kind)
                })
                .collect::<Vec<_>>()[..]
            {
                let trait_name = tcx.def_path_str(best_trait);
                err.label = Some(errors::AssocItemNotFoundLabel::FoundInOtherTrait {
                    span: assoc_name.span,
                    assoc_kind: assoc_kind_str,
                    trait_name: &trait_name,
                    suggested_name,
                    identically_named: suggested_name == assoc_name.name,
                });
                let hir = tcx.hir();
                if let Some(def_id) = ty_param_def_id
                    && let parent = hir.get_parent_item(tcx.local_def_id_to_hir_id(def_id))
                    && let Some(generics) = hir.get_generics(parent.def_id)
                {
                    if generics.bounds_for_param(def_id).flat_map(|pred| pred.bounds.iter()).any(
                        |b| match b {
                            hir::GenericBound::Trait(t, ..) => {
                                t.trait_ref.trait_def_id() == Some(best_trait)
                            }
                            _ => false,
                        },
                    ) {
                        // The type param already has a bound for `trait_name`, we just need to
                        // change the associated item.
                        err.sugg = Some(errors::AssocItemNotFoundSugg::SimilarInOtherTrait {
                            span: assoc_name.span,
                            assoc_kind: assoc_kind_str,
                            suggested_name,
                        });
                        return tcx.dcx().emit_err(err);
                    }

                    let mut err = tcx.dcx().create_err(err);
                    if suggest_constraining_type_param(
                        tcx,
                        generics,
                        &mut err,
                        &ty_param_name,
                        &trait_name,
                        None,
                        None,
                    ) && suggested_name != assoc_name.name
                    {
                        // We suggested constraining a type parameter, but the associated item on it
                        // was also not an exact match, so we also suggest changing it.
                        err.span_suggestion_verbose(
                            assoc_name.span,
                            fluent::hir_analysis_assoc_item_not_found_similar_in_other_trait_with_bound_sugg,
                            suggested_name,
                            Applicability::MaybeIncorrect,
                        );
                    }
                    return err.emit();
                }
                return tcx.dcx().emit_err(err);
            }
        }

        // If we still couldn't find any associated item, and only one associated item exists,
        // suggests using it.
        if let [candidate_name] = all_candidate_names.as_slice() {
            // This should still compile, except on `#![feature(associated_type_defaults)]`
            // where it could suggests `type A = Self::A`, thus recursing infinitely.
            let applicability =
                if assoc_kind == ty::AssocKind::Type && tcx.features().associated_type_defaults {
                    Applicability::Unspecified
                } else {
                    Applicability::MaybeIncorrect
                };

            err.sugg = Some(errors::AssocItemNotFoundSugg::Other {
                span: assoc_name.span,
                applicability,
                ty_param_name,
                assoc_kind: assoc_kind_str,
                suggested_name: *candidate_name,
            });
        } else {
            err.label = Some(errors::AssocItemNotFoundLabel::NotFound { span: assoc_name.span });
        }

        tcx.dcx().emit_err(err)
    }

    fn complain_about_assoc_kind_mismatch(
        &self,
        assoc_item: &ty::AssocItem,
        assoc_kind: ty::AssocKind,
        ident: Ident,
        span: Span,
        binding: Option<&hir::TypeBinding<'tcx>>,
    ) -> ErrorGuaranteed {
        let tcx = self.tcx();

        let bound_on_assoc_const_label = if let ty::AssocKind::Const = assoc_item.kind
            && let Some(binding) = binding
            && let hir::TypeBindingKind::Constraint { .. } = binding.kind
        {
            let lo = if binding.gen_args.span_ext.is_dummy() {
                ident.span
            } else {
                binding.gen_args.span_ext
            };
            Some(lo.between(span.shrink_to_hi()))
        } else {
            None
        };

        // FIXME(associated_const_equality): This has quite a few false positives and negatives.
        let wrap_in_braces_sugg = if let Some(binding) = binding
            && let hir::TypeBindingKind::Equality { term: hir::Term::Ty(hir_ty) } = binding.kind
            && let ty = self.ast_ty_to_ty(hir_ty)
            && (ty.is_enum() || ty.references_error())
            && tcx.features().associated_const_equality
        {
            Some(errors::AssocKindMismatchWrapInBracesSugg {
                lo: hir_ty.span.shrink_to_lo(),
                hi: hir_ty.span.shrink_to_hi(),
            })
        } else {
            None
        };

        // For equality bounds, we want to blame the term (RHS) instead of the item (LHS) since
        // one can argue that that's more “intuitive” to the user.
        let (span, expected_because_label, expected, got) = if let Some(binding) = binding
            && let hir::TypeBindingKind::Equality { term } = binding.kind
        {
            let span = match term {
                hir::Term::Ty(ty) => ty.span,
                hir::Term::Const(ct) => tcx.def_span(ct.def_id),
            };
            (span, Some(ident.span), assoc_item.kind, assoc_kind)
        } else {
            (ident.span, None, assoc_kind, assoc_item.kind)
        };

        tcx.dcx().emit_err(errors::AssocKindMismatch {
            span,
            expected: super::assoc_kind_str(expected),
            got: super::assoc_kind_str(got),
            expected_because_label,
            assoc_kind: super::assoc_kind_str(assoc_item.kind),
            def_span: tcx.def_span(assoc_item.def_id),
            bound_on_assoc_const_label,
            wrap_in_braces_sugg,
        })
    }

    pub(crate) fn complain_about_ambiguous_inherent_assoc_type(
        &self,
        name: Ident,
        candidates: Vec<DefId>,
        span: Span,
    ) -> ErrorGuaranteed {
        let mut err = struct_span_code_err!(
            self.tcx().dcx(),
            name.span,
            E0034,
            "multiple applicable items in scope"
        );
        err.span_label(name.span, format!("multiple `{name}` found"));
        self.note_ambiguous_inherent_assoc_type(&mut err, candidates, span);
        let reported = err.emit();
        self.set_tainted_by_errors(reported);
        reported
    }

    // FIXME(fmease): Heavily adapted from `rustc_hir_typeck::method::suggest`. Deduplicate.
    fn note_ambiguous_inherent_assoc_type(
        &self,
        err: &mut Diag<'_>,
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

            let impl_ty = tcx.at(span).type_of(impl_).instantiate_identity();
            let note = format!("{title} is defined in an impl for the type `{impl_ty}`");

            if let Some(span) = note_span {
                err.span_note(span, note);
            } else {
                err.note(note);
            }
        }
        if candidates.len() > limit {
            err.note(format!("and {} others", candidates.len() - limit));
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
        let add_def_label = |err: &mut Diag<'_>| {
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
                .map(|&(impl_, _)| {
                    format!("- `{}`", tcx.at(span).type_of(impl_).instantiate_identity())
                })
                .collect::<Vec<_>>()
                .join("\n");
            let additional_types = if candidates.len() > limit {
                format!("\nand {} more types", candidates.len() - limit)
            } else {
                String::new()
            };

            let mut err = struct_span_code_err!(
                tcx.dcx(),
                name.span,
                E0220,
                "associated type `{name}` not found for `{self_ty}` in the current scope"
            );
            err.span_label(name.span, format!("associated item not found in `{self_ty}`"));
            err.note(format!(
                "the associated type was found for\n{type_candidates}{additional_types}",
            ));
            add_def_label(&mut err);
            return err.emit();
        }

        let mut bound_spans: SortedMap<Span, Vec<String>> = Default::default();

        let mut bound_span_label = |self_ty: Ty<'_>, obligation: &str, quiet: &str| {
            let msg = format!("`{}`", if obligation.len() > 50 { quiet } else { obligation });
            match &self_ty.kind() {
                // Point at the type that couldn't satisfy the bound.
                ty::Adt(def, _) => {
                    bound_spans.get_mut_or_insert_default(tcx.def_span(def.did())).push(msg)
                }
                // Point at the trait object that couldn't satisfy the bound.
                ty::Dynamic(preds, _, _) => {
                    for pred in preds.iter() {
                        match pred.skip_binder() {
                            ty::ExistentialPredicate::Trait(tr) => {
                                bound_spans
                                    .get_mut_or_insert_default(tcx.def_span(tr.def_id))
                                    .push(msg.clone());
                            }
                            ty::ExistentialPredicate::Projection(_)
                            | ty::ExistentialPredicate::AutoTrait(_) => {}
                        }
                    }
                }
                // Point at the closure that couldn't satisfy the bound.
                ty::Closure(def_id, _) => {
                    bound_spans
                        .get_mut_or_insert_default(tcx.def_span(*def_id))
                        .push(format!("`{quiet}`"));
                }
                _ => {}
            }
        };

        let format_pred = |pred: ty::Predicate<'tcx>| {
            let bound_predicate = pred.kind();
            match bound_predicate.skip_binder() {
                ty::PredicateKind::Clause(ty::ClauseKind::Projection(pred)) => {
                    let pred = bound_predicate.rebind(pred);
                    // `<Foo as Iterator>::Item = String`.
                    let projection_ty = pred.skip_binder().projection_ty;

                    let args_with_infer_self = tcx.mk_args_from_iter(
                        std::iter::once(Ty::new_var(tcx, ty::TyVid::from_u32(0)).into())
                            .chain(projection_ty.args.iter().skip(1)),
                    );

                    let quiet_projection_ty =
                        ty::AliasTy::new(tcx, projection_ty.def_id, args_with_infer_self);

                    let term = pred.skip_binder().term;

                    let obligation = format!("{projection_ty} = {term}");
                    let quiet = format!("{quiet_projection_ty} = {term}");

                    bound_span_label(projection_ty.self_ty(), &obligation, &quiet);
                    Some((obligation, projection_ty.self_ty()))
                }
                ty::PredicateKind::Clause(ty::ClauseKind::Trait(poly_trait_ref)) => {
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
            .map(|(p, _)| format!("`{p}`"))
            .collect();
        bounds.sort();
        bounds.dedup();

        let mut err = tcx.dcx().struct_span_err(
            name.span,
            format!("the associated type `{name}` exists for `{self_ty}`, but its trait bounds were not satisfied")
        );
        if !bounds.is_empty() {
            err.note(format!(
                "the following trait bounds were not satisfied:\n{}",
                bounds.join("\n")
            ));
        }
        err.span_label(
            name.span,
            format!("associated type cannot be referenced on `{self_ty}` due to unsatisfied trait bounds")
        );

        for (span, mut bounds) in bound_spans {
            if !tcx.sess.source_map().is_span_accessible(span) {
                continue;
            }
            bounds.sort();
            bounds.dedup();
            let msg = match &bounds[..] {
                [bound] => format!("doesn't satisfy {bound}"),
                bounds if bounds.len() > 4 => format!("doesn't satisfy {} bounds", bounds.len()),
                [bounds @ .., last] => format!("doesn't satisfy {} or {last}", bounds.join(", ")),
                [] => unreachable!(),
            };
            err.span_label(span, msg);
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
        associated_types: FxIndexMap<Span, FxIndexSet<DefId>>,
        potential_assoc_types: Vec<Span>,
        trait_bounds: &[hir::PolyTraitRef<'_>],
    ) {
        if associated_types.values().all(|v| v.is_empty()) {
            return;
        }

        let tcx = self.tcx();
        // FIXME: Marked `mut` so that we can replace the spans further below with a more
        // appropriate one, but this should be handled earlier in the span assignment.
        let mut associated_types: FxIndexMap<Span, Vec<_>> = associated_types
            .into_iter()
            .map(|(span, def_ids)| {
                (span, def_ids.into_iter().map(|did| tcx.associated_item(did)).collect())
            })
            .collect();
        let mut names: FxIndexMap<String, Vec<Symbol>> = Default::default();
        let mut names_len = 0;

        // Account for things like `dyn Foo + 'a`, like in tests `issue-22434.rs` and
        // `issue-22560.rs`.
        let mut trait_bound_spans: Vec<Span> = vec![];
        let mut object_safety_violations = false;
        for (span, items) in &associated_types {
            if !items.is_empty() {
                trait_bound_spans.push(*span);
            }
            for assoc_item in items {
                let trait_def_id = assoc_item.container_id(tcx);
                names.entry(tcx.def_path_str(trait_def_id)).or_default().push(assoc_item.name);
                names_len += 1;

                let violations =
                    object_safety_violations_for_assoc_item(tcx, trait_def_id, *assoc_item);
                if !violations.is_empty() {
                    report_object_safety_error(tcx, *span, None, trait_def_id, &violations).emit();
                    object_safety_violations = true;
                }
            }
        }
        if object_safety_violations {
            return;
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
                        .into_values()
                        .map(|items| (segment.ident.span, items))
                        .collect();
                }
                _ => {}
            }
        }

        // We get all the associated items that _are_ set,
        // so that we can check if any of their names match one of the ones we are missing.
        // This would mean that they are shadowing the associated type we are missing,
        // and we can then use their span to indicate this to the user.
        let bound_names = trait_bounds
            .iter()
            .filter_map(|poly_trait_ref| {
                let path = poly_trait_ref.trait_ref.path.segments.last()?;
                let args = path.args?;

                Some(args.bindings.iter().filter_map(|binding| {
                    let ident = binding.ident;
                    let trait_def = path.res.def_id();
                    let assoc_item = tcx.associated_items(trait_def).find_by_name_and_kind(
                        tcx,
                        ident,
                        ty::AssocKind::Type,
                        trait_def,
                    );

                    Some((ident.name, assoc_item?))
                }))
            })
            .flatten()
            .collect::<UnordMap<Symbol, &ty::AssocItem>>();

        let mut names = names
            .into_iter()
            .map(|(trait_, mut assocs)| {
                assocs.sort();
                format!(
                    "{} in `{trait_}`",
                    match &assocs[..] {
                        [] => String::new(),
                        [only] => format!("`{only}`"),
                        [assocs @ .., last] => format!(
                            "{} and `{last}`",
                            assocs.iter().map(|a| format!("`{a}`")).collect::<Vec<_>>().join(", ")
                        ),
                    }
                )
            })
            .collect::<Vec<String>>();
        names.sort();
        let names = names.join(", ");

        trait_bound_spans.sort();
        let mut err = struct_span_code_err!(
            tcx.dcx(),
            trait_bound_spans,
            E0191,
            "the value of the associated type{} {} must be specified",
            pluralize!(names_len),
            names,
        );
        let mut suggestions = vec![];
        let mut types_count = 0;
        let mut where_constraints = vec![];
        let mut already_has_generics_args_suggestion = false;
        for (span, assoc_items) in &associated_types {
            let mut names: UnordMap<_, usize> = Default::default();
            for item in assoc_items {
                types_count += 1;
                *names.entry(item.name).or_insert(0) += 1;
            }
            let mut dupes = false;
            let mut shadows = false;
            for item in assoc_items {
                let prefix = if names[&item.name] > 1 {
                    let trait_def_id = item.container_id(tcx);
                    dupes = true;
                    format!("{}::", tcx.def_path_str(trait_def_id))
                } else if bound_names.get(&item.name).is_some_and(|x| x != &item) {
                    let trait_def_id = item.container_id(tcx);
                    shadows = true;
                    format!("{}::", tcx.def_path_str(trait_def_id))
                } else {
                    String::new()
                };

                let mut is_shadowed = false;

                if let Some(assoc_item) = bound_names.get(&item.name)
                    && assoc_item != &item
                {
                    is_shadowed = true;

                    let rename_message =
                        if assoc_item.def_id.is_local() { ", consider renaming it" } else { "" };
                    err.span_label(
                        tcx.def_span(assoc_item.def_id),
                        format!("`{}{}` shadowed here{}", prefix, item.name, rename_message),
                    );
                }

                let rename_message = if is_shadowed { ", consider renaming it" } else { "" };

                if let Some(sp) = tcx.hir().span_if_local(item.def_id) {
                    err.span_label(
                        sp,
                        format!("`{}{}` defined here{}", prefix, item.name, rename_message),
                    );
                }
            }
            if potential_assoc_types.len() == assoc_items.len() {
                // When the amount of missing associated types equals the number of
                // extra type arguments present. A suggesting to replace the generic args with
                // associated types is already emitted.
                already_has_generics_args_suggestion = true;
            } else if let (Ok(snippet), false, false) =
                (tcx.sess.source_map().span_to_snippet(*span), dupes, shadows)
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
                let mut names: FxIndexMap<_, usize> = FxIndexMap::default();
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
        suggestions.sort_by_key(|&(span, _)| span);
        // There are cases where one bound points to a span within another bound's span, like when
        // you have code like the following (#115019), so we skip providing a suggestion in those
        // cases to avoid having a malformed suggestion.
        //
        // pub struct Flatten<I> {
        //     inner: <IntoIterator<Item: IntoIterator<Item: >>::IntoIterator as Item>::core,
        //             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        //             |                  ^^^^^^^^^^^^^^^^^^^^^
        //             |                  |
        //             |                  associated types `Item`, `IntoIter` must be specified
        //             associated types `Item`, `IntoIter` must be specified
        // }
        let overlaps = suggestions.windows(2).any(|pair| pair[0].0.overlaps(pair[1].0));
        if !suggestions.is_empty() && !overlaps {
            err.multipart_suggestion(
                format!("specify the associated type{}", pluralize!(types_count)),
                suggestions,
                Applicability::HasPlaceholders,
            );
            if !where_constraints.is_empty() {
                err.span_help(where_constraints, where_msg);
            }
        }

        self.set_tainted_by_errors(err.emit());
    }

    /// On ambiguous associated type, look for an associated function whose name matches the
    /// extended path and, if found, emit an E0223 error with a structured suggestion.
    /// e.g. for `String::from::utf8`, suggest `String::from_utf8` (#109195)
    pub(crate) fn maybe_report_similar_assoc_fn(
        &self,
        span: Span,
        qself_ty: Ty<'tcx>,
        qself: &hir::Ty<'_>,
    ) -> Result<(), ErrorGuaranteed> {
        let tcx = self.tcx();
        if let Some((_, node)) = tcx.hir().parent_iter(qself.hir_id).skip(1).next()
            && let hir::Node::Expr(hir::Expr {
                kind:
                    hir::ExprKind::Path(hir::QPath::TypeRelative(
                        hir::Ty {
                            kind:
                                hir::TyKind::Path(hir::QPath::TypeRelative(
                                    _,
                                    hir::PathSegment { ident: ident2, .. },
                                )),
                            ..
                        },
                        hir::PathSegment { ident: ident3, .. },
                    )),
                ..
            }) = node
            && let Some(ty_def_id) = qself_ty.ty_def_id()
            && let Ok([inherent_impl]) = tcx.inherent_impls(ty_def_id)
            && let name = format!("{ident2}_{ident3}")
            && let Some(ty::AssocItem { kind: ty::AssocKind::Fn, .. }) = tcx
                .associated_items(inherent_impl)
                .filter_by_name_unhygienic(Symbol::intern(&name))
                .next()
        {
            let reported =
                struct_span_code_err!(tcx.dcx(), span, E0223, "ambiguous associated type")
                    .with_span_suggestion_verbose(
                        ident2.span.to(ident3.span),
                        format!("there is an associated function with a similar name: `{name}`"),
                        name,
                        Applicability::MaybeIncorrect,
                    )
                    .emit();
            self.set_tainted_by_errors(reported);
            Err(reported)
        } else {
            Ok(())
        }
    }
}

/// Emits an error regarding forbidden type binding associations
pub fn prohibit_assoc_ty_binding(
    tcx: TyCtxt<'_>,
    span: Span,
    segment: Option<(&hir::PathSegment<'_>, Span)>,
) {
    tcx.dcx().emit_err(AssocTypeBindingNotAllowed {
        span,
        fn_trait_expansion: if let Some((segment, span)) = segment
            && segment.args().parenthesized == hir::GenericArgsParentheses::ParenSugar
        {
            Some(ParenthesizedFnTraitExpansion {
                span,
                expanded_type: fn_trait_to_string(tcx, segment, false),
            })
        } else {
            None
        },
    });
}

pub(crate) fn fn_trait_to_string(
    tcx: TyCtxt<'_>,
    trait_segment: &hir::PathSegment<'_>,
    parenthesized: bool,
) -> String {
    let args = trait_segment
        .args
        .as_ref()
        .and_then(|args| args.args.get(0))
        .and_then(|arg| match arg {
            hir::GenericArg::Type(ty) => match ty.kind {
                hir::TyKind::Tup(t) => t
                    .iter()
                    .map(|e| tcx.sess.source_map().span_to_snippet(e.span))
                    .collect::<Result<Vec<_>, _>>()
                    .map(|a| a.join(", ")),
                _ => tcx.sess.source_map().span_to_snippet(ty.span),
            }
            .map(|s| {
                // `s.empty()` checks to see if the type is the unit tuple, if so we don't want a comma
                if parenthesized || s.is_empty() { format!("({s})") } else { format!("({s},)") }
            })
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
                    hir::Term::Const(c) => tcx.hir().span(c.hir_id),
                };

                (span != tcx.hir().span(trait_segment.hir_id))
                    .then_some(tcx.sess.source_map().span_to_snippet(span).ok())
                    .flatten()
            }
            _ => None,
        })
        .unwrap_or_else(|| "()".to_string());

    if parenthesized {
        format!("{}{} -> {}", trait_segment.ident, args, ret)
    } else {
        format!("{}<{}, Output={}>", trait_segment.ident, args, ret)
    }
}
