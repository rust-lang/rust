use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_data_structures::sorted_map::SortedMap;
use rustc_data_structures::unord::UnordMap;
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, ErrorGuaranteed, MultiSpan, pluralize, struct_span_code_err,
};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_middle::bug;
use rustc_middle::ty::print::{PrintPolyTraitRefExt as _, PrintTraitRefExt as _};
use rustc_middle::ty::{
    self, AdtDef, Binder, GenericParamDefKind, TraitRef, Ty, TyCtxt, TypeVisitableExt,
    suggest_constraining_type_param,
};
use rustc_session::parse::feature_err;
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::{BytePos, DUMMY_SP, Ident, Span, Symbol, kw, sym};
use rustc_trait_selection::error_reporting::traits::report_dyn_incompatibility;
use rustc_trait_selection::traits::{
    FulfillmentError, TraitAliasExpansionInfo, dyn_compatibility_violations_for_assoc_item,
};

use crate::errors::{
    self, AssocItemConstraintsNotAllowedHere, ManualImplementation, MissingTypeParams,
    ParenthesizedFnTraitExpansion, TraitObjectDeclaredWithNoTraits,
};
use crate::fluent_generated as fluent;
use crate::hir_ty_lowering::{AssocItemQSelf, HirTyLowerer};

impl<'tcx> dyn HirTyLowerer<'tcx> + '_ {
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

        self.dcx().emit_err(MissingTypeParams {
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
        if self.tcx().features().unboxed_closures() {
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
            self.dcx().emit_err(ManualImplementation { span, trait_name });
        }
    }

    pub(super) fn complain_about_assoc_item_not_found<I>(
        &self,
        all_candidates: impl Fn() -> I,
        qself: AssocItemQSelf,
        assoc_kind: ty::AssocKind,
        assoc_name: Ident,
        span: Span,
        constraint: Option<&hir::AssocItemConstraint<'tcx>>,
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
                assoc_item, assoc_kind, assoc_name, span, constraint,
            );
        }

        let assoc_kind_str = assoc_kind_str(assoc_kind);
        let qself_str = qself.to_string(tcx);

        // The fallback span is needed because `assoc_name` might be an `Fn()`'s `Output` without a
        // valid span, so we point at the whole path segment instead.
        let is_dummy = assoc_name.span == DUMMY_SP;

        let mut err = errors::AssocItemNotFound {
            span: if is_dummy { span } else { assoc_name.span },
            assoc_name,
            assoc_kind: assoc_kind_str,
            qself: &qself_str,
            label: None,
            sugg: None,
        };

        if is_dummy {
            err.label = Some(errors::AssocItemNotFoundLabel::NotFound { span });
            return self.dcx().emit_err(err);
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
            return self.dcx().emit_err(err);
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
                if let AssocItemQSelf::TyParam(ty_param_def_id, ty_param_span) = qself
                    // Not using `self.item_def_id()` here as that would yield the opaque type itself if we're
                    // inside an opaque type while we're interested in the overarching type alias (TAIT).
                    // FIXME: However, for trait aliases, this incorrectly returns the enclosing module...
                    && let item_def_id =
                        tcx.hir().get_parent_item(tcx.local_def_id_to_hir_id(ty_param_def_id))
                    // FIXME: ...which obviously won't have any generics.
                    && let Some(generics) = tcx.hir().get_generics(item_def_id.def_id)
                {
                    // FIXME: Suggest adding supertrait bounds if we have a `Self` type param.
                    // FIXME(trait_alias): Suggest adding `Self: Trait` to
                    // `trait Alias = where Self::Proj:;` with `trait Trait { type Proj; }`.
                    if generics
                        .bounds_for_param(ty_param_def_id)
                        .flat_map(|pred| pred.bounds.iter())
                        .any(|b| match b {
                            hir::GenericBound::Trait(t, ..) => {
                                t.trait_ref.trait_def_id() == Some(best_trait)
                            }
                            _ => false,
                        })
                    {
                        // The type param already has a bound for `trait_name`, we just need to
                        // change the associated item.
                        err.sugg = Some(errors::AssocItemNotFoundSugg::SimilarInOtherTrait {
                            span: assoc_name.span,
                            assoc_kind: assoc_kind_str,
                            suggested_name,
                        });
                        return self.dcx().emit_err(err);
                    }

                    let trait_args = &ty::GenericArgs::identity_for_item(tcx, best_trait)[1..];
                    let mut trait_ref = trait_name.clone();
                    let applicability = if let [arg, args @ ..] = trait_args {
                        use std::fmt::Write;
                        write!(trait_ref, "</* {arg}").unwrap();
                        args.iter().try_for_each(|arg| write!(trait_ref, ", {arg}")).unwrap();
                        trait_ref += " */>";
                        Applicability::HasPlaceholders
                    } else {
                        Applicability::MaybeIncorrect
                    };

                    let identically_named = suggested_name == assoc_name.name;

                    if let DefKind::TyAlias = tcx.def_kind(item_def_id)
                        && !tcx.type_alias_is_lazy(item_def_id)
                    {
                        err.sugg = Some(errors::AssocItemNotFoundSugg::SimilarInOtherTraitQPath {
                            lo: ty_param_span.shrink_to_lo(),
                            mi: ty_param_span.shrink_to_hi(),
                            hi: (!identically_named).then_some(assoc_name.span),
                            trait_ref,
                            identically_named,
                            suggested_name,
                            applicability,
                        });
                    } else {
                        let mut err = self.dcx().create_err(err);
                        if suggest_constraining_type_param(
                            tcx,
                            generics,
                            &mut err,
                            &qself_str,
                            &trait_ref,
                            Some(best_trait),
                            None,
                        ) && !identically_named
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
                }
                return self.dcx().emit_err(err);
            }
        }

        // If we still couldn't find any associated item, and only one associated item exists,
        // suggest using it.
        if let [candidate_name] = all_candidate_names.as_slice() {
            err.sugg = Some(errors::AssocItemNotFoundSugg::Other {
                span: assoc_name.span,
                qself: &qself_str,
                assoc_kind: assoc_kind_str,
                suggested_name: *candidate_name,
            });
        } else {
            err.label = Some(errors::AssocItemNotFoundLabel::NotFound { span: assoc_name.span });
        }

        self.dcx().emit_err(err)
    }

    fn complain_about_assoc_kind_mismatch(
        &self,
        assoc_item: &ty::AssocItem,
        assoc_kind: ty::AssocKind,
        ident: Ident,
        span: Span,
        constraint: Option<&hir::AssocItemConstraint<'tcx>>,
    ) -> ErrorGuaranteed {
        let tcx = self.tcx();

        let bound_on_assoc_const_label = if let ty::AssocKind::Const = assoc_item.kind
            && let Some(constraint) = constraint
            && let hir::AssocItemConstraintKind::Bound { .. } = constraint.kind
        {
            let lo = if constraint.gen_args.span_ext.is_dummy() {
                ident.span
            } else {
                constraint.gen_args.span_ext
            };
            Some(lo.between(span.shrink_to_hi()))
        } else {
            None
        };

        // FIXME(associated_const_equality): This has quite a few false positives and negatives.
        let wrap_in_braces_sugg = if let Some(constraint) = constraint
            && let Some(hir_ty) = constraint.ty()
            && let ty = self.lower_ty(hir_ty)
            && (ty.is_enum() || ty.references_error())
            && tcx.features().associated_const_equality()
        {
            Some(errors::AssocKindMismatchWrapInBracesSugg {
                lo: hir_ty.span.shrink_to_lo(),
                hi: hir_ty.span.shrink_to_hi(),
            })
        } else {
            None
        };

        // For equality constraints, we want to blame the term (RHS) instead of the item (LHS) since
        // one can argue that that's more “intuitive” to the user.
        let (span, expected_because_label, expected, got) = if let Some(constraint) = constraint
            && let hir::AssocItemConstraintKind::Equality { term } = constraint.kind
        {
            let span = match term {
                hir::Term::Ty(ty) => ty.span,
                hir::Term::Const(ct) => ct.span(),
            };
            (span, Some(ident.span), assoc_item.kind, assoc_kind)
        } else {
            (ident.span, None, assoc_kind, assoc_item.kind)
        };

        self.dcx().emit_err(errors::AssocKindMismatch {
            span,
            expected: assoc_kind_str(expected),
            got: assoc_kind_str(got),
            expected_because_label,
            assoc_kind: assoc_kind_str(assoc_item.kind),
            def_span: tcx.def_span(assoc_item.def_id),
            bound_on_assoc_const_label,
            wrap_in_braces_sugg,
        })
    }

    pub(super) fn report_ambiguous_assoc_ty(
        &self,
        span: Span,
        types: &[String],
        traits: &[String],
        name: Symbol,
    ) -> ErrorGuaranteed {
        let mut err = struct_span_code_err!(self.dcx(), span, E0223, "ambiguous associated type");
        if self
            .tcx()
            .resolutions(())
            .confused_type_with_std_module
            .keys()
            .any(|full_span| full_span.contains(span))
        {
            err.span_suggestion_verbose(
                span.shrink_to_lo(),
                "you are looking for the module in `std`, not the primitive type",
                "std::",
                Applicability::MachineApplicable,
            );
        } else {
            let mut types = types.to_vec();
            types.sort();
            let mut traits = traits.to_vec();
            traits.sort();
            match (&types[..], &traits[..]) {
                ([], []) => {
                    err.span_suggestion_verbose(
                        span,
                        format!(
                            "if there were a type named `Type` that implements a trait named \
                             `Trait` with associated type `{name}`, you could use the \
                             fully-qualified path",
                        ),
                        format!("<Type as Trait>::{name}"),
                        Applicability::HasPlaceholders,
                    );
                }
                ([], [trait_str]) => {
                    err.span_suggestion_verbose(
                        span,
                        format!(
                            "if there were a type named `Example` that implemented `{trait_str}`, \
                             you could use the fully-qualified path",
                        ),
                        format!("<Example as {trait_str}>::{name}"),
                        Applicability::HasPlaceholders,
                    );
                }
                ([], traits) => {
                    err.span_suggestions(
                        span,
                        format!(
                            "if there were a type named `Example` that implemented one of the \
                             traits with associated type `{name}`, you could use the \
                             fully-qualified path",
                        ),
                        traits.iter().map(|trait_str| format!("<Example as {trait_str}>::{name}")),
                        Applicability::HasPlaceholders,
                    );
                }
                ([type_str], []) => {
                    err.span_suggestion_verbose(
                        span,
                        format!(
                            "if there were a trait named `Example` with associated type `{name}` \
                             implemented for `{type_str}`, you could use the fully-qualified path",
                        ),
                        format!("<{type_str} as Example>::{name}"),
                        Applicability::HasPlaceholders,
                    );
                }
                (types, []) => {
                    err.span_suggestions(
                        span,
                        format!(
                            "if there were a trait named `Example` with associated type `{name}` \
                             implemented for one of the types, you could use the fully-qualified \
                             path",
                        ),
                        types
                            .into_iter()
                            .map(|type_str| format!("<{type_str} as Example>::{name}")),
                        Applicability::HasPlaceholders,
                    );
                }
                (types, traits) => {
                    let mut suggestions = vec![];
                    for type_str in types {
                        for trait_str in traits {
                            suggestions.push(format!("<{type_str} as {trait_str}>::{name}"));
                        }
                    }
                    err.span_suggestions(
                        span,
                        "use fully-qualified syntax",
                        suggestions,
                        Applicability::MachineApplicable,
                    );
                }
            }
        }
        err.emit()
    }

    pub(crate) fn complain_about_ambiguous_inherent_assoc_ty(
        &self,
        name: Ident,
        candidates: Vec<DefId>,
        span: Span,
    ) -> ErrorGuaranteed {
        let mut err = struct_span_code_err!(
            self.dcx(),
            name.span,
            E0034,
            "multiple applicable items in scope"
        );
        err.span_label(name.span, format!("multiple `{name}` found"));
        self.note_ambiguous_inherent_assoc_ty(&mut err, candidates, span);
        err.emit()
    }

    // FIXME(fmease): Heavily adapted from `rustc_hir_typeck::method::suggest`. Deduplicate.
    fn note_ambiguous_inherent_assoc_ty(
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
    pub(crate) fn complain_about_inherent_assoc_ty_not_found(
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
                self.dcx(),
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
            match self_ty.kind() {
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
                    // `<Foo as Iterator>::Item = String`.
                    let projection_term = pred.projection_term;
                    let quiet_projection_term =
                        projection_term.with_self_ty(tcx, Ty::new_var(tcx, ty::TyVid::ZERO));

                    let term = pred.term;
                    let obligation = format!("{projection_term} = {term}");
                    let quiet = format!("{quiet_projection_term} = {term}");

                    bound_span_label(projection_term.self_ty(), &obligation, &quiet);
                    Some((obligation, projection_term.self_ty()))
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

        let mut err = self.dcx().struct_span_err(
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
    pub(crate) fn check_for_required_assoc_tys(
        &self,
        principal_span: Span,
        missing_assoc_types: FxIndexSet<(DefId, ty::PolyTraitRef<'tcx>)>,
        potential_assoc_types: Vec<usize>,
        trait_bounds: &[hir::PolyTraitRef<'_>],
    ) -> Result<(), ErrorGuaranteed> {
        if missing_assoc_types.is_empty() {
            return Ok(());
        }

        let tcx = self.tcx();
        // FIXME: This logic needs some more care w.r.t handling of conflicts
        let missing_assoc_types: Vec<_> = missing_assoc_types
            .into_iter()
            .map(|(def_id, trait_ref)| (tcx.associated_item(def_id), trait_ref))
            .collect();
        let mut names: FxIndexMap<_, Vec<Symbol>> = Default::default();
        let mut names_len = 0;

        // Account for things like `dyn Foo + 'a`, like in tests `issue-22434.rs` and
        // `issue-22560.rs`.
        let mut dyn_compatibility_violations = Ok(());
        for (assoc_item, trait_ref) in &missing_assoc_types {
            names.entry(trait_ref).or_default().push(assoc_item.name);
            names_len += 1;

            let violations =
                dyn_compatibility_violations_for_assoc_item(tcx, trait_ref.def_id(), *assoc_item);
            if !violations.is_empty() {
                dyn_compatibility_violations = Err(report_dyn_incompatibility(
                    tcx,
                    principal_span,
                    None,
                    trait_ref.def_id(),
                    &violations,
                )
                .emit());
            }
        }

        if let Err(guar) = dyn_compatibility_violations {
            return Err(guar);
        }

        // related to issue #91997, turbofishes added only when in an expr or pat
        let mut in_expr_or_pat = false;
        if let ([], [bound]) = (&potential_assoc_types[..], &trait_bounds) {
            let grandparent = tcx.parent_hir_node(tcx.parent_hir_id(bound.trait_ref.hir_ref_id));
            in_expr_or_pat = match grandparent {
                hir::Node::Expr(_) | hir::Node::Pat(_) => true,
                _ => false,
            };
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

                Some(args.constraints.iter().filter_map(|constraint| {
                    let ident = constraint.ident;
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
                let trait_ = trait_.print_trait_sugared();
                format!("{} in `{trait_}`", match &assocs[..] {
                    [] => String::new(),
                    [only] => format!("`{only}`"),
                    [assocs @ .., last] => format!(
                        "{} and `{last}`",
                        assocs.iter().map(|a| format!("`{a}`")).collect::<Vec<_>>().join(", ")
                    ),
                })
            })
            .collect::<Vec<String>>();
        names.sort();
        let names = names.join(", ");

        let mut err = struct_span_code_err!(
            self.dcx(),
            principal_span,
            E0191,
            "the value of the associated type{} {} must be specified",
            pluralize!(names_len),
            names,
        );
        let mut suggestions = vec![];
        let mut types_count = 0;
        let mut where_constraints = vec![];
        let mut already_has_generics_args_suggestion = false;

        let mut names: UnordMap<_, usize> = Default::default();
        for (item, _) in &missing_assoc_types {
            types_count += 1;
            *names.entry(item.name).or_insert(0) += 1;
        }
        let mut dupes = false;
        let mut shadows = false;
        for (item, trait_ref) in &missing_assoc_types {
            let prefix = if names[&item.name] > 1 {
                let trait_def_id = trait_ref.def_id();
                dupes = true;
                format!("{}::", tcx.def_path_str(trait_def_id))
            } else if bound_names.get(&item.name).is_some_and(|x| *x != item) {
                let trait_def_id = trait_ref.def_id();
                shadows = true;
                format!("{}::", tcx.def_path_str(trait_def_id))
            } else {
                String::new()
            };

            let mut is_shadowed = false;

            if let Some(assoc_item) = bound_names.get(&item.name)
                && *assoc_item != item
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
        if potential_assoc_types.len() == missing_assoc_types.len() {
            // When the amount of missing associated types equals the number of
            // extra type arguments present. A suggesting to replace the generic args with
            // associated types is already emitted.
            already_has_generics_args_suggestion = true;
        } else if let (Ok(snippet), false, false) =
            (tcx.sess.source_map().span_to_snippet(principal_span), dupes, shadows)
        {
            let types: Vec<_> = missing_assoc_types
                .iter()
                .map(|(item, _)| format!("{} = Type", item.name))
                .collect();
            let code = if snippet.ends_with('>') {
                // The user wrote `Trait<'a>` or similar and we don't have a type we can
                // suggest, but at least we can clue them to the correct syntax
                // `Trait<'a, Item = Type>` while accounting for the `<'a>` in the
                // suggestion.
                format!("{}, {}>", &snippet[..snippet.len() - 1], types.join(", "))
            } else if in_expr_or_pat {
                // The user wrote `Iterator`, so we don't have a type we can suggest, but at
                // least we can clue them to the correct syntax `Iterator::<Item = Type>`.
                format!("{}::<{}>", snippet, types.join(", "))
            } else {
                // The user wrote `Iterator`, so we don't have a type we can suggest, but at
                // least we can clue them to the correct syntax `Iterator<Item = Type>`.
                format!("{}<{}>", snippet, types.join(", "))
            };
            suggestions.push((principal_span, code));
        } else if dupes {
            where_constraints.push(principal_span);
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
            let mut names: FxIndexMap<_, usize> = FxIndexMap::default();
            for (item, _) in &missing_assoc_types {
                types_count += 1;
                *names.entry(item.name).or_insert(0) += 1;
            }
            let mut label = vec![];
            for (item, trait_ref) in &missing_assoc_types {
                let postfix = if names[&item.name] > 1 {
                    format!(" (from trait `{}`)", trait_ref.print_trait_sugared())
                } else {
                    String::new()
                };
                label.push(format!("`{}`{}", item.name, postfix));
            }
            if !label.is_empty() {
                err.span_label(
                    principal_span,
                    format!(
                        "associated type{} {} must be specified",
                        pluralize!(label.len()),
                        label.join(", "),
                    ),
                );
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

        Err(err.emit())
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
            && let Some(adt_def) = qself_ty.ty_adt_def()
            && let [inherent_impl] = tcx.inherent_impls(adt_def.did())
            && let name = format!("{ident2}_{ident3}")
            && let Some(ty::AssocItem { kind: ty::AssocKind::Fn, .. }) = tcx
                .associated_items(inherent_impl)
                .filter_by_name_unhygienic(Symbol::intern(&name))
                .next()
        {
            Err(struct_span_code_err!(self.dcx(), span, E0223, "ambiguous associated type")
                .with_span_suggestion_verbose(
                    ident2.span.to(ident3.span),
                    format!("there is an associated function with a similar name: `{name}`"),
                    name,
                    Applicability::MaybeIncorrect,
                )
                .emit())
        } else {
            Ok(())
        }
    }

    pub fn report_prohibit_generics_error<'a>(
        &self,
        segments: impl Iterator<Item = &'a hir::PathSegment<'a>> + Clone,
        args_visitors: impl Iterator<Item = &'a hir::GenericArg<'a>> + Clone,
        err_extend: GenericsArgsErrExtend<'_>,
    ) -> ErrorGuaranteed {
        #[derive(PartialEq, Eq, Hash)]
        enum ProhibitGenericsArg {
            Lifetime,
            Type,
            Const,
            Infer,
        }

        let mut prohibit_args = FxIndexSet::default();
        args_visitors.for_each(|arg| {
            match arg {
                hir::GenericArg::Lifetime(_) => prohibit_args.insert(ProhibitGenericsArg::Lifetime),
                hir::GenericArg::Type(_) => prohibit_args.insert(ProhibitGenericsArg::Type),
                hir::GenericArg::Const(_) => prohibit_args.insert(ProhibitGenericsArg::Const),
                hir::GenericArg::Infer(_) => prohibit_args.insert(ProhibitGenericsArg::Infer),
            };
        });

        let types_and_spans: Vec<_> = segments
            .clone()
            .flat_map(|segment| {
                if segment.args().args.is_empty() {
                    None
                } else {
                    Some((
                        match segment.res {
                            hir::def::Res::PrimTy(ty) => {
                                format!("{} `{}`", segment.res.descr(), ty.name())
                            }
                            hir::def::Res::Def(_, def_id)
                                if let Some(name) = self.tcx().opt_item_name(def_id) =>
                            {
                                format!("{} `{name}`", segment.res.descr())
                            }
                            hir::def::Res::Err => "this type".to_string(),
                            _ => segment.res.descr().to_string(),
                        },
                        segment.ident.span,
                    ))
                }
            })
            .collect();
        let this_type = match &types_and_spans[..] {
            [.., _, (last, _)] => format!(
                "{} and {last}",
                types_and_spans[..types_and_spans.len() - 1]
                    .iter()
                    .map(|(x, _)| x.as_str())
                    .intersperse(", ")
                    .collect::<String>()
            ),
            [(only, _)] => only.to_string(),
            [] => bug!("expected one segment to deny"),
        };

        let arg_spans: Vec<Span> = segments
            .clone()
            .flat_map(|segment| segment.args().args)
            .map(|arg| arg.span())
            .collect();

        let mut kinds = Vec::with_capacity(4);
        prohibit_args.iter().for_each(|arg| match arg {
            ProhibitGenericsArg::Lifetime => kinds.push("lifetime"),
            ProhibitGenericsArg::Type => kinds.push("type"),
            ProhibitGenericsArg::Const => kinds.push("const"),
            ProhibitGenericsArg::Infer => kinds.push("generic"),
        });

        let (kind, s) = match kinds[..] {
            [.., _, last] => (
                format!(
                    "{} and {last}",
                    kinds[..kinds.len() - 1]
                        .iter()
                        .map(|&x| x)
                        .intersperse(", ")
                        .collect::<String>()
                ),
                "s",
            ),
            [only] => (only.to_string(), ""),
            [] => bug!("expected at least one generic to prohibit"),
        };
        let last_span = *arg_spans.last().unwrap();
        let span: MultiSpan = arg_spans.into();
        let mut err = struct_span_code_err!(
            self.dcx(),
            span,
            E0109,
            "{kind} arguments are not allowed on {this_type}",
        );
        err.span_label(last_span, format!("{kind} argument{s} not allowed"));
        for (what, span) in types_and_spans {
            err.span_label(span, format!("not allowed on {what}"));
        }
        generics_args_err_extend(self.tcx(), segments, &mut err, err_extend);
        err.emit()
    }

    pub fn report_trait_object_addition_traits_error(
        &self,
        regular_traits: &Vec<TraitAliasExpansionInfo<'_>>,
    ) -> ErrorGuaranteed {
        let first_trait = &regular_traits[0];
        let additional_trait = &regular_traits[1];
        let mut err = struct_span_code_err!(
            self.dcx(),
            additional_trait.bottom().1,
            E0225,
            "only auto traits can be used as additional traits in a trait object"
        );
        additional_trait.label_with_exp_info(
            &mut err,
            "additional non-auto trait",
            "additional use",
        );
        first_trait.label_with_exp_info(&mut err, "first non-auto trait", "first use");
        err.help(format!(
            "consider creating a new trait with all of these as supertraits and using that \
             trait here instead: `trait NewTrait: {} {{}}`",
            regular_traits
                .iter()
                // FIXME: This should `print_sugared`, but also needs to integrate projection bounds...
                .map(|t| t.trait_ref().print_only_trait_path().to_string())
                .collect::<Vec<_>>()
                .join(" + "),
        ));
        err.note(
            "auto-traits like `Send` and `Sync` are traits that have special properties; \
             for more information on them, visit \
             <https://doc.rust-lang.org/reference/special-types-and-traits.html#auto-traits>",
        );
        err.emit()
    }

    pub fn report_trait_object_with_no_traits_error(
        &self,
        span: Span,
        trait_bounds: &Vec<(Binder<'tcx, TraitRef<'tcx>>, Span)>,
    ) -> ErrorGuaranteed {
        let tcx = self.tcx();
        let trait_alias_span = trait_bounds
            .iter()
            .map(|&(trait_ref, _)| trait_ref.def_id())
            .find(|&trait_ref| tcx.is_trait_alias(trait_ref))
            .map(|trait_ref| tcx.def_span(trait_ref));

        self.dcx().emit_err(TraitObjectDeclaredWithNoTraits { span, trait_alias_span })
    }
}

/// Emit an error for the given associated item constraint.
pub fn prohibit_assoc_item_constraint(
    cx: &dyn HirTyLowerer<'_>,
    constraint: &hir::AssocItemConstraint<'_>,
    segment: Option<(DefId, &hir::PathSegment<'_>, Span)>,
) -> ErrorGuaranteed {
    let tcx = cx.tcx();
    let mut err = cx.dcx().create_err(AssocItemConstraintsNotAllowedHere {
        span: constraint.span,
        fn_trait_expansion: if let Some((_, segment, span)) = segment
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

    // Emit a suggestion to turn the assoc item binding into a generic arg
    // if the relevant item has a generic param whose name matches the binding name;
    // otherwise suggest the removal of the binding.
    if let Some((def_id, segment, _)) = segment
        && segment.args().parenthesized == hir::GenericArgsParentheses::No
    {
        // Suggests removal of the offending binding
        let suggest_removal = |e: &mut Diag<'_>| {
            let constraints = segment.args().constraints;
            let args = segment.args().args;

            // Compute the span to remove based on the position
            // of the binding. We do that as follows:
            //  1. Find the index of the binding in the list of bindings
            //  2. Locate the spans preceding and following the binding.
            //     If it's the first binding the preceding span would be
            //     that of the last arg
            //  3. Using this information work out whether the span
            //     to remove will start from the end of the preceding span,
            //     the start of the next span or will simply be the
            //     span encomassing everything within the generics brackets

            let Some(index) = constraints.iter().position(|b| b.hir_id == constraint.hir_id) else {
                bug!("a type binding exists but its HIR ID not found in generics");
            };

            let preceding_span = if index > 0 {
                Some(constraints[index - 1].span)
            } else {
                args.last().map(|a| a.span())
            };

            let next_span = constraints.get(index + 1).map(|constraint| constraint.span);

            let removal_span = match (preceding_span, next_span) {
                (Some(prec), _) => constraint.span.with_lo(prec.hi()),
                (None, Some(next)) => constraint.span.with_hi(next.lo()),
                (None, None) => {
                    let Some(generics_span) = segment.args().span_ext() else {
                        bug!("a type binding exists but generic span is empty");
                    };

                    generics_span
                }
            };

            // Now emit the suggestion
            e.span_suggestion_verbose(
                removal_span,
                format!("consider removing this associated item {}", constraint.kind.descr()),
                "",
                Applicability::MaybeIncorrect,
            );
        };

        // Suggest replacing the associated item binding with a generic argument.
        // i.e., replacing `<..., T = A, ...>` with `<..., A, ...>`.
        let suggest_direct_use = |e: &mut Diag<'_>, sp: Span| {
            if let Ok(snippet) = tcx.sess.source_map().span_to_snippet(sp) {
                e.span_suggestion_verbose(
                    constraint.span,
                    format!("to use `{snippet}` as a generic argument specify it directly"),
                    snippet,
                    Applicability::MaybeIncorrect,
                );
            }
        };

        // Check if the type has a generic param with the same name
        // as the assoc type name in the associated item binding.
        let generics = tcx.generics_of(def_id);
        let matching_param = generics.own_params.iter().find(|p| p.name == constraint.ident.name);

        // Now emit the appropriate suggestion
        if let Some(matching_param) = matching_param {
            match (constraint.kind, &matching_param.kind) {
                (
                    hir::AssocItemConstraintKind::Equality { term: hir::Term::Ty(ty) },
                    GenericParamDefKind::Type { .. },
                ) => suggest_direct_use(&mut err, ty.span),
                (
                    hir::AssocItemConstraintKind::Equality { term: hir::Term::Const(c) },
                    GenericParamDefKind::Const { .. },
                ) => {
                    suggest_direct_use(&mut err, c.span());
                }
                (hir::AssocItemConstraintKind::Bound { bounds }, _) => {
                    // Suggest `impl<T: Bound> Trait<T> for Foo` when finding
                    // `impl Trait<T: Bound> for Foo`

                    // Get the parent impl block based on the binding we have
                    // and the trait DefId
                    let impl_block = tcx
                        .hir()
                        .parent_iter(constraint.hir_id)
                        .find_map(|(_, node)| node.impl_block_of_trait(def_id));

                    let type_with_constraints =
                        tcx.sess.source_map().span_to_snippet(constraint.span);

                    if let Some(impl_block) = impl_block
                        && let Ok(type_with_constraints) = type_with_constraints
                    {
                        // Filter out the lifetime parameters because
                        // they should be declared before the type parameter
                        let lifetimes: String = bounds
                            .iter()
                            .filter_map(|bound| {
                                if let hir::GenericBound::Outlives(lifetime) = bound {
                                    Some(format!("{lifetime}, "))
                                } else {
                                    None
                                }
                            })
                            .collect();
                        // Figure out a span and suggestion string based on
                        // whether there are any existing parameters
                        let param_decl = if let Some(param_span) =
                            impl_block.generics.span_for_param_suggestion()
                        {
                            (param_span, format!(", {lifetimes}{type_with_constraints}"))
                        } else {
                            (
                                impl_block.generics.span.shrink_to_lo(),
                                format!("<{lifetimes}{type_with_constraints}>"),
                            )
                        };
                        let suggestions = vec![
                            param_decl,
                            (constraint.span.with_lo(constraint.ident.span.hi()), String::new()),
                        ];

                        err.multipart_suggestion_verbose(
                            "declare the type parameter right after the `impl` keyword",
                            suggestions,
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
                _ => suggest_removal(&mut err),
            }
        } else {
            suggest_removal(&mut err);
        }
    }

    err.emit()
}

pub(crate) fn fn_trait_to_string(
    tcx: TyCtxt<'_>,
    trait_segment: &hir::PathSegment<'_>,
    parenthesized: bool,
) -> String {
    let args = trait_segment
        .args
        .and_then(|args| args.args.first())
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
                // `is_empty()` checks to see if the type is the unit tuple, if so we don't want a comma
                if parenthesized || s.is_empty() { format!("({s})") } else { format!("({s},)") }
            })
            .ok(),
            _ => None,
        })
        .unwrap_or_else(|| "()".to_string());

    let ret = trait_segment
        .args()
        .constraints
        .iter()
        .find_map(|c| {
            if c.ident.name == sym::Output
                && let Some(ty) = c.ty()
                && ty.span != tcx.hir().span(trait_segment.hir_id)
            {
                tcx.sess.source_map().span_to_snippet(ty.span).ok()
            } else {
                None
            }
        })
        .unwrap_or_else(|| "()".to_string());

    if parenthesized {
        format!("{}{} -> {}", trait_segment.ident, args, ret)
    } else {
        format!("{}<{}, Output={}>", trait_segment.ident, args, ret)
    }
}

/// Used for generics args error extend.
pub enum GenericsArgsErrExtend<'tcx> {
    EnumVariant {
        qself: &'tcx hir::Ty<'tcx>,
        assoc_segment: &'tcx hir::PathSegment<'tcx>,
        adt_def: AdtDef<'tcx>,
    },
    OpaqueTy,
    PrimTy(hir::PrimTy),
    SelfTyAlias {
        def_id: DefId,
        span: Span,
    },
    SelfTyParam(Span),
    Param(DefId),
    DefVariant,
    None,
}

fn generics_args_err_extend<'a>(
    tcx: TyCtxt<'_>,
    segments: impl Iterator<Item = &'a hir::PathSegment<'a>> + Clone,
    err: &mut Diag<'_>,
    err_extend: GenericsArgsErrExtend<'_>,
) {
    match err_extend {
        GenericsArgsErrExtend::EnumVariant { qself, assoc_segment, adt_def } => {
            err.note("enum variants can't have type parameters");
            let type_name = tcx.item_name(adt_def.did());
            let msg = format!(
                "you might have meant to specify type parameters on enum \
                `{type_name}`"
            );
            let Some(args) = assoc_segment.args else {
                return;
            };
            // Get the span of the generics args *including* the leading `::`.
            // We do so by stretching args.span_ext to the left by 2. Earlier
            // it was done based on the end of assoc segment but that sometimes
            // led to impossible spans and caused issues like #116473
            let args_span = args.span_ext.with_lo(args.span_ext.lo() - BytePos(2));
            if tcx.generics_of(adt_def.did()).is_empty() {
                // FIXME(estebank): we could also verify that the arguments being
                // work for the `enum`, instead of just looking if it takes *any*.
                err.span_suggestion_verbose(
                    args_span,
                    format!("{type_name} doesn't have generic parameters"),
                    "",
                    Applicability::MachineApplicable,
                );
                return;
            }
            let Ok(snippet) = tcx.sess.source_map().span_to_snippet(args_span) else {
                err.note(msg);
                return;
            };
            let (qself_sugg_span, is_self) =
                if let hir::TyKind::Path(hir::QPath::Resolved(_, path)) = &qself.kind {
                    // If the path segment already has type params, we want to overwrite
                    // them.
                    match &path.segments {
                        // `segment` is the previous to last element on the path,
                        // which would normally be the `enum` itself, while the last
                        // `_` `PathSegment` corresponds to the variant.
                        [
                            ..,
                            hir::PathSegment {
                                ident, args, res: Res::Def(DefKind::Enum, _), ..
                            },
                            _,
                        ] => (
                            // We need to include the `::` in `Type::Variant::<Args>`
                            // to point the span to `::<Args>`, not just `<Args>`.
                            ident
                                .span
                                .shrink_to_hi()
                                .to(args.map_or(ident.span.shrink_to_hi(), |a| a.span_ext)),
                            false,
                        ),
                        [segment] => {
                            (
                                // We need to include the `::` in `Type::Variant::<Args>`
                                // to point the span to `::<Args>`, not just `<Args>`.
                                segment.ident.span.shrink_to_hi().to(segment
                                    .args
                                    .map_or(segment.ident.span.shrink_to_hi(), |a| a.span_ext)),
                                kw::SelfUpper == segment.ident.name,
                            )
                        }
                        _ => {
                            err.note(msg);
                            return;
                        }
                    }
                } else {
                    err.note(msg);
                    return;
                };
            let suggestion = vec![
                if is_self {
                    // Account for people writing `Self::Variant::<Args>`, where
                    // `Self` is the enum, and suggest replacing `Self` with the
                    // appropriate type: `Type::<Args>::Variant`.
                    (qself.span, format!("{type_name}{snippet}"))
                } else {
                    (qself_sugg_span, snippet)
                },
                (args_span, String::new()),
            ];
            err.multipart_suggestion_verbose(msg, suggestion, Applicability::MaybeIncorrect);
        }
        GenericsArgsErrExtend::PrimTy(prim_ty) => {
            let name = prim_ty.name_str();
            for segment in segments {
                if let Some(args) = segment.args {
                    err.span_suggestion_verbose(
                        segment.ident.span.shrink_to_hi().to(args.span_ext),
                        format!("primitive type `{name}` doesn't have generic parameters"),
                        "",
                        Applicability::MaybeIncorrect,
                    );
                }
            }
        }
        GenericsArgsErrExtend::OpaqueTy => {
            err.note("`impl Trait` types can't have type parameters");
        }
        GenericsArgsErrExtend::DefVariant => {
            err.note("enum variants can't have type parameters");
        }
        GenericsArgsErrExtend::Param(def_id) => {
            let span = tcx.def_ident_span(def_id).unwrap();
            let kind = tcx.def_descr(def_id);
            let name = tcx.item_name(def_id);
            err.span_note(span, format!("{kind} `{name}` defined here"));
        }
        GenericsArgsErrExtend::SelfTyParam(span) => {
            err.span_suggestion_verbose(
                span,
                "the `Self` type doesn't accept type parameters",
                "",
                Applicability::MaybeIncorrect,
            );
        }
        GenericsArgsErrExtend::SelfTyAlias { def_id, span } => {
            let ty = tcx.at(span).type_of(def_id).instantiate_identity();
            let span_of_impl = tcx.span_of_impl(def_id);
            let def_id = match *ty.kind() {
                ty::Adt(self_def, _) => self_def.did(),
                _ => return,
            };

            let type_name = tcx.item_name(def_id);
            let span_of_ty = tcx.def_ident_span(def_id);
            let generics = tcx.generics_of(def_id).count();

            let msg = format!("`Self` is of type `{ty}`");
            if let (Ok(i_sp), Some(t_sp)) = (span_of_impl, span_of_ty) {
                let mut span: MultiSpan = vec![t_sp].into();
                span.push_span_label(
                    i_sp,
                    format!("`Self` is on type `{type_name}` in this `impl`"),
                );
                let mut postfix = "";
                if generics == 0 {
                    postfix = ", which doesn't have generic parameters";
                }
                span.push_span_label(t_sp, format!("`Self` corresponds to this type{postfix}"));
                err.span_note(span, msg);
            } else {
                err.note(msg);
            }
            for segment in segments {
                if let Some(args) = segment.args
                    && segment.ident.name == kw::SelfUpper
                {
                    if generics == 0 {
                        // FIXME(estebank): we could also verify that the arguments being
                        // work for the `enum`, instead of just looking if it takes *any*.
                        err.span_suggestion_verbose(
                            segment.ident.span.shrink_to_hi().to(args.span_ext),
                            "the `Self` type doesn't accept type parameters",
                            "",
                            Applicability::MachineApplicable,
                        );
                        return;
                    } else {
                        err.span_suggestion_verbose(
                            segment.ident.span,
                            format!(
                                "the `Self` type doesn't accept type parameters, use the \
                                concrete type's name `{type_name}` instead if you want to \
                                specify its type parameters"
                            ),
                            type_name,
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
            }
        }
        _ => {}
    }
}

pub(super) fn assoc_kind_str(kind: ty::AssocKind) -> &'static str {
    match kind {
        ty::AssocKind::Fn => "function",
        ty::AssocKind::Const => "constant",
        ty::AssocKind::Type => "type",
    }
}
