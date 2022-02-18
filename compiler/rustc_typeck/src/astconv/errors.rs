use crate::astconv::AstConv;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{pluralize, struct_span_err, Applicability};
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::ty;
use rustc_session::parse::feature_err;
use rustc_span::lev_distance::find_best_match_for_name;
use rustc_span::symbol::{sym, Ident};
use rustc_span::{Span, DUMMY_SP};

use std::collections::BTreeSet;
use std::iter;

impl<'o, 'tcx> dyn AstConv<'tcx> + 'o {
    /// On missing type parameters, emit an E0393 error and provide a structured suggestion using
    /// the type parameter's name as a placeholder.
    pub(crate) fn complain_about_missing_type_params(
        &self,
        missing_type_params: Vec<String>,
        def_id: DefId,
        span: Span,
        empty_generic_args: bool,
    ) {
        if missing_type_params.is_empty() {
            return;
        }
        let display =
            missing_type_params.iter().map(|n| format!("`{}`", n)).collect::<Vec<_>>().join(", ");
        let mut err = struct_span_err!(
            self.tcx().sess,
            span,
            E0393,
            "the type parameter{} {} must be explicitly specified",
            pluralize!(missing_type_params.len()),
            display,
        );
        err.span_label(
            self.tcx().def_span(def_id),
            &format!(
                "type parameter{} {} must be specified for this",
                pluralize!(missing_type_params.len()),
                display,
            ),
        );
        let mut suggested = false;
        if let (Ok(snippet), true) = (
            self.tcx().sess.source_map().span_to_snippet(span),
            // Don't suggest setting the type params if there are some already: the order is
            // tricky to get right and the user will already know what the syntax is.
            empty_generic_args,
        ) {
            if snippet.ends_with('>') {
                // The user wrote `Trait<'a, T>` or similar. To provide an accurate suggestion
                // we would have to preserve the right order. For now, as clearly the user is
                // aware of the syntax, we do nothing.
            } else {
                // The user wrote `Iterator`, so we don't have a type we can suggest, but at
                // least we can clue them to the correct syntax `Iterator<Type>`.
                err.span_suggestion(
                    span,
                    &format!(
                        "set the type parameter{plural} to the desired type{plural}",
                        plural = pluralize!(missing_type_params.len()),
                    ),
                    format!("{}<{}>", snippet, missing_type_params.join(", ")),
                    Applicability::HasPlaceholders,
                );
                suggested = true;
            }
        }
        if !suggested {
            err.span_label(
                span,
                format!(
                    "missing reference{} to {}",
                    pluralize!(missing_type_params.len()),
                    display,
                ),
            );
        }
        err.note(
            "because of the default `Self` reference, type parameters must be \
                  specified on object types",
        );
        err.emit();
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
            // the desugaring would contain an associated type constrait.
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
            struct_span_err!(
                self.tcx().sess,
                span,
                E0183,
                "manual implementations of `{}` are experimental",
                trait_name,
            )
            .span_label(
                span,
                format!("manual implementations of `{}` are experimental", trait_name),
            )
            .help("add `#![feature(unboxed_closures)]` to the crate attributes to enable")
            .emit();
        }
    }

    pub(crate) fn complain_about_assoc_type_not_found<I>(
        &self,
        all_candidates: impl Fn() -> I,
        ty_param_name: &str,
        assoc_name: Ident,
        span: Span,
    ) where
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
                suggested_name.to_string(),
                Applicability::MaybeIncorrect,
            );
        } else {
            err.span_label(span, format!("associated type `{}` not found", assoc_name));
        }

        err.emit();
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
                let trait_def_id = assoc_item.container.id();
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
                // `src/test/ui/issues/issue-28344.rs` where instead of the following:
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
        for (span, assoc_items) in &associated_types {
            let mut names: FxHashMap<_, usize> = FxHashMap::default();
            for item in assoc_items {
                types_count += 1;
                *names.entry(item.name).or_insert(0) += 1;
            }
            let mut dupes = false;
            for item in assoc_items {
                let prefix = if names[&item.name] > 1 {
                    let trait_def_id = item.container.id();
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
                // Only suggest when the amount of missing associated types equals the number of
                // extra type arguments present, as that gives us a relatively high confidence
                // that the user forgot to give the associtated type's name. The canonical
                // example would be trying to use `Iterator<isize>` instead of
                // `Iterator<Item = isize>`.
                for (potential, item) in iter::zip(&potential_assoc_types, assoc_items) {
                    if let Ok(snippet) = tcx.sess.source_map().span_to_snippet(*potential) {
                        suggestions.push((*potential, format!("{} = {}", item.name, snippet)));
                    }
                }
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
        if suggestions.len() != 1 {
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
                        let trait_def_id = item.container.id();
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
