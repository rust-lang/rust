use rustc_ast::TraitObjectSyntax;
use rustc_data_structures::fx::{FxHashSet, FxIndexMap, FxIndexSet};
use rustc_errors::codes::*;
use rustc_errors::{
    Applicability, Diag, EmissionGuarantee, StashKey, Suggestions, struct_span_code_err,
};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_lint_defs::builtin::{BARE_TRAIT_OBJECTS, UNUSED_ASSOCIATED_TYPE_BOUNDS};
use rustc_middle::ty::elaborate::ClauseWithSupertraitSpan;
use rustc_middle::ty::{
    self, BottomUpFolder, ExistentialPredicateStableCmpExt as _, Ty, TyCtxt, TypeFoldable,
    TypeVisitableExt, Upcast,
};
use rustc_span::edit_distance::find_best_match_for_name;
use rustc_span::{ErrorGuaranteed, Span};
use rustc_trait_selection::error_reporting::traits::report_dyn_incompatibility;
use rustc_trait_selection::error_reporting::traits::suggestions::NextTypeParamName;
use rustc_trait_selection::traits;
use smallvec::{SmallVec, smallvec};
use tracing::{debug, instrument};

use super::HirTyLowerer;
use crate::errors::SelfInTypeAlias;
use crate::hir_ty_lowering::{
    GenericArgCountMismatch, OverlappingAsssocItemConstraints, PredicateFilter, RegionInferReason,
};

impl<'tcx> dyn HirTyLowerer<'tcx> + '_ {
    /// Lower a trait object type from the HIR to our internal notion of a type.
    #[instrument(level = "debug", skip_all, ret)]
    pub(super) fn lower_trait_object_ty(
        &self,
        span: Span,
        hir_id: hir::HirId,
        hir_bounds: &[hir::PolyTraitRef<'tcx>],
        lifetime: &hir::Lifetime,
        syntax: TraitObjectSyntax,
    ) -> Ty<'tcx> {
        let tcx = self.tcx();
        let dummy_self = tcx.types.trait_object_dummy_self;

        match syntax {
            TraitObjectSyntax::Dyn => {}
            TraitObjectSyntax::None => {
                match self.prohibit_or_lint_bare_trait_object_ty(span, hir_id, hir_bounds) {
                    // Don't continue with type analysis if the `dyn` keyword is missing.
                    // It generates confusing errors, especially if the user meant to use
                    // another keyword like `impl`.
                    Some(guar) => return Ty::new_error(tcx, guar),
                    None => {}
                }
            }
        }

        let mut user_written_bounds = Vec::new();
        let mut potential_assoc_types = Vec::new();
        for poly_trait_ref in hir_bounds.iter() {
            let result = self.lower_poly_trait_ref(
                poly_trait_ref,
                dummy_self,
                &mut user_written_bounds,
                PredicateFilter::SelfOnly,
                OverlappingAsssocItemConstraints::Forbidden,
            );
            if let Err(GenericArgCountMismatch { invalid_args, .. }) = result.correct {
                potential_assoc_types.extend(invalid_args);
            }
        }

        self.add_default_traits(
            &mut user_written_bounds,
            dummy_self,
            &hir_bounds
                .iter()
                .map(|&trait_ref| hir::GenericBound::Trait(trait_ref))
                .collect::<Vec<_>>(),
            None,
            span,
        );

        let (elaborated_trait_bounds, elaborated_projection_bounds) =
            traits::expand_trait_aliases(tcx, user_written_bounds.iter().copied());
        let (regular_traits, mut auto_traits): (Vec<_>, Vec<_>) = elaborated_trait_bounds
            .into_iter()
            .partition(|(trait_ref, _)| !tcx.trait_is_auto(trait_ref.def_id()));

        // We don't support empty trait objects.
        if regular_traits.is_empty() && auto_traits.is_empty() {
            let guar =
                self.report_trait_object_with_no_traits(span, user_written_bounds.iter().copied());
            return Ty::new_error(tcx, guar);
        }
        // We don't support >1 principal
        if regular_traits.len() > 1 {
            let guar = self.report_trait_object_addition_traits(&regular_traits);
            return Ty::new_error(tcx, guar);
        }
        // Don't create a dyn trait if we have errors in the principal.
        if let Err(guar) = regular_traits.error_reported() {
            return Ty::new_error(tcx, guar);
        }

        // Check that there are no gross dyn-compatibility violations;
        // most importantly, that the supertraits don't contain `Self`,
        // to avoid ICEs.
        for (clause, span) in user_written_bounds {
            if let Some(trait_pred) = clause.as_trait_clause() {
                let violations = self.dyn_compatibility_violations(trait_pred.def_id());
                if !violations.is_empty() {
                    let reported = report_dyn_incompatibility(
                        tcx,
                        span,
                        Some(hir_id),
                        trait_pred.def_id(),
                        &violations,
                    )
                    .emit();
                    return Ty::new_error(tcx, reported);
                }
            }
        }

        // Map the projection bounds onto a key that makes it easy to remove redundant
        // bounds that are constrained by supertraits of the principal def id.
        //
        // Also make sure we detect conflicting bounds from expanding a trait alias and
        // also specifying it manually, like:
        // ```
        // type Alias = Trait<Assoc = i32>;
        // let _: &dyn Alias<Assoc = u32> = /* ... */;
        // ```
        let mut projection_bounds = FxIndexMap::default();
        for (proj, proj_span) in elaborated_projection_bounds {
            let proj = proj.map_bound(|mut b| {
                if let Some(term_ty) = &b.term.as_type() {
                    let references_self = term_ty.walk().any(|arg| arg == dummy_self.into());
                    if references_self {
                        // With trait alias and type alias combined, type resolver
                        // may not be able to catch all illegal `Self` usages (issue 139082)
                        let guar = self.dcx().emit_err(SelfInTypeAlias { span });
                        b.term = replace_dummy_self_with_error(tcx, b.term, guar);
                    }
                }
                b
            });

            let key = (
                proj.skip_binder().projection_term.def_id,
                tcx.anonymize_bound_vars(
                    proj.map_bound(|proj| proj.projection_term.trait_ref(tcx)),
                ),
            );
            if let Some((old_proj, old_proj_span)) =
                projection_bounds.insert(key, (proj, proj_span))
                && tcx.anonymize_bound_vars(proj) != tcx.anonymize_bound_vars(old_proj)
            {
                let item = tcx.item_name(proj.item_def_id());
                self.dcx()
                    .struct_span_err(
                        span,
                        format!("conflicting associated type bounds for `{item}`"),
                    )
                    .with_span_label(
                        old_proj_span,
                        format!("`{item}` is specified to be `{}` here", old_proj.term()),
                    )
                    .with_span_label(
                        proj_span,
                        format!("`{item}` is specified to be `{}` here", proj.term()),
                    )
                    .emit();
            }
        }

        let principal_trait = regular_traits.into_iter().next();

        // A stable ordering of associated types from the principal trait and all its
        // supertraits. We use this to ensure that different substitutions of a trait
        // don't result in `dyn Trait` types with different projections lists, which
        // can be unsound: <https://github.com/rust-lang/rust/pull/136458>.
        // We achieve a stable ordering by walking over the unsubstituted principal
        // trait ref.
        let mut ordered_associated_types = vec![];

        if let Some((principal_trait, ref spans)) = principal_trait {
            let principal_trait = principal_trait.map_bound(|trait_pred| {
                assert_eq!(trait_pred.polarity, ty::PredicatePolarity::Positive);
                trait_pred.trait_ref
            });

            for ClauseWithSupertraitSpan { clause, supertrait_span } in traits::elaborate(
                tcx,
                [ClauseWithSupertraitSpan::new(
                    ty::TraitRef::identity(tcx, principal_trait.def_id()).upcast(tcx),
                    *spans.last().unwrap(),
                )],
            )
            .filter_only_self()
            {
                let clause = clause.instantiate_supertrait(tcx, principal_trait);
                debug!("observing object predicate `{clause:?}`");

                let bound_predicate = clause.kind();
                match bound_predicate.skip_binder() {
                    ty::ClauseKind::Trait(pred) => {
                        // FIXME(negative_bounds): Handle this correctly...
                        let trait_ref =
                            tcx.anonymize_bound_vars(bound_predicate.rebind(pred.trait_ref));
                        ordered_associated_types.extend(
                            tcx.associated_items(pred.trait_ref.def_id)
                                .in_definition_order()
                                // We only care about associated types.
                                .filter(|item| item.is_type())
                                // No RPITITs -- they're not dyn-compatible for now.
                                .filter(|item| !item.is_impl_trait_in_trait())
                                .map(|item| (item.def_id, trait_ref)),
                        );
                    }
                    ty::ClauseKind::Projection(pred) => {
                        let pred = bound_predicate.rebind(pred);
                        // A `Self` within the original bound will be instantiated with a
                        // `trait_object_dummy_self`, so check for that.
                        let references_self = match pred.skip_binder().term.kind() {
                            ty::TermKind::Ty(ty) => ty.walk().any(|arg| arg == dummy_self.into()),
                            // FIXME(associated_const_equality): We should walk the const instead of not doing anything
                            ty::TermKind::Const(_) => false,
                        };

                        // If the projection output contains `Self`, force the user to
                        // elaborate it explicitly to avoid a lot of complexity.
                        //
                        // The "classically useful" case is the following:
                        // ```
                        //     trait MyTrait: FnMut() -> <Self as MyTrait>::MyOutput {
                        //         type MyOutput;
                        //     }
                        // ```
                        //
                        // Here, the user could theoretically write `dyn MyTrait<MyOutput = X>`,
                        // but actually supporting that would "expand" to an infinitely-long type
                        // `fix $ τ → dyn MyTrait<MyOutput = X, Output = <τ as MyTrait>::MyOutput`.
                        //
                        // Instead, we force the user to write
                        // `dyn MyTrait<MyOutput = X, Output = X>`, which is uglier but works. See
                        // the discussion in #56288 for alternatives.
                        if !references_self {
                            let key = (
                                pred.skip_binder().projection_term.def_id,
                                tcx.anonymize_bound_vars(
                                    pred.map_bound(|proj| proj.projection_term.trait_ref(tcx)),
                                ),
                            );
                            if !projection_bounds.contains_key(&key) {
                                projection_bounds.insert(key, (pred, supertrait_span));
                            }
                        }

                        self.check_elaborated_projection_mentions_input_lifetimes(
                            pred,
                            *spans.first().unwrap(),
                            supertrait_span,
                        );
                    }
                    _ => (),
                }
            }
        }

        // `dyn Trait<Assoc = Foo>` desugars to (not Rust syntax) `dyn Trait where
        // <Self as Trait>::Assoc = Foo`. So every `Projection` clause is an
        // `Assoc = Foo` bound. `needed_associated_types` contains all associated
        // types that we expect to be provided by the user, so the following loop
        // removes all the associated types that have a corresponding `Projection`
        // clause, either from expanding trait aliases or written by the user.
        for &(projection_bound, span) in projection_bounds.values() {
            let def_id = projection_bound.item_def_id();
            if tcx.generics_require_sized_self(def_id) {
                tcx.emit_node_span_lint(
                    UNUSED_ASSOCIATED_TYPE_BOUNDS,
                    hir_id,
                    span,
                    crate::errors::UnusedAssociatedTypeBounds { span },
                );
            }
        }

        // We compute the list of projection bounds taking the ordered associated types,
        // and check if there was an entry in the collected `projection_bounds`. Those
        // are computed by first taking the user-written associated types, then elaborating
        // the principal trait ref, and only using those if there was no user-written.
        // See note below about how we handle missing associated types with `Self: Sized`,
        // which are not required to be provided, but are still used if they are provided.
        let mut missing_assoc_types = FxIndexSet::default();
        let projection_bounds: Vec<_> = ordered_associated_types
            .into_iter()
            .filter_map(|key| {
                if let Some(assoc) = projection_bounds.get(&key) {
                    Some(*assoc)
                } else {
                    // If the associated type has a `where Self: Sized` bound, then
                    // we do not need to provide the associated type. This results in
                    // a `dyn Trait` type that has a different number of projection
                    // bounds, which may lead to type mismatches.
                    if !tcx.generics_require_sized_self(key.0) {
                        missing_assoc_types.insert(key);
                    }
                    None
                }
            })
            .collect();

        if let Err(guar) = self.check_for_required_assoc_tys(
            principal_trait.as_ref().map_or(smallvec![], |(_, spans)| spans.clone()),
            missing_assoc_types,
            potential_assoc_types,
            hir_bounds,
        ) {
            return Ty::new_error(tcx, guar);
        }

        // De-duplicate auto traits so that, e.g., `dyn Trait + Send + Send` is the same as
        // `dyn Trait + Send`.
        // We remove duplicates by inserting into a `FxHashSet` to avoid re-ordering
        // the bounds
        let mut duplicates = FxHashSet::default();
        auto_traits.retain(|(trait_pred, _)| duplicates.insert(trait_pred.def_id()));

        debug!(?principal_trait);
        debug!(?auto_traits);

        // Erase the `dummy_self` (`trait_object_dummy_self`) used above.
        let principal_trait_ref = principal_trait.map(|(trait_pred, spans)| {
            trait_pred.map_bound(|trait_pred| {
                let trait_ref = trait_pred.trait_ref;
                assert_eq!(trait_pred.polarity, ty::PredicatePolarity::Positive);
                assert_eq!(trait_ref.self_ty(), dummy_self);

                let span = *spans.first().unwrap();

                // Verify that `dummy_self` did not leak inside default type parameters. This
                // could not be done at path creation, since we need to see through trait aliases.
                let mut missing_type_params = vec![];
                let generics = tcx.generics_of(trait_ref.def_id);
                let args: Vec<_> = trait_ref
                    .args
                    .iter()
                    .enumerate()
                    // Skip `Self`
                    .skip(1)
                    .map(|(index, arg)| {
                        if arg.walk().any(|arg| arg == dummy_self.into()) {
                            let param = &generics.own_params[index];
                            missing_type_params.push(param.name);
                            Ty::new_misc_error(tcx).into()
                        } else {
                            arg
                        }
                    })
                    .collect();

                let empty_generic_args = hir_bounds.iter().any(|hir_bound| {
                    hir_bound.trait_ref.path.res == Res::Def(DefKind::Trait, trait_ref.def_id)
                        && hir_bound.span.contains(span)
                });
                self.report_missing_type_params(
                    missing_type_params,
                    trait_ref.def_id,
                    span,
                    empty_generic_args,
                );

                ty::ExistentialPredicate::Trait(ty::ExistentialTraitRef::new(
                    tcx,
                    trait_ref.def_id,
                    args,
                ))
            })
        });

        let existential_projections = projection_bounds.into_iter().map(|(bound, _)| {
            bound.map_bound(|mut b| {
                assert_eq!(b.projection_term.self_ty(), dummy_self);

                // Like for trait refs, verify that `dummy_self` did not leak inside default type
                // parameters.
                let references_self = b.projection_term.args.iter().skip(1).any(|arg| {
                    if arg.walk().any(|arg| arg == dummy_self.into()) {
                        return true;
                    }
                    false
                });
                if references_self {
                    let guar = tcx
                        .dcx()
                        .span_delayed_bug(span, "trait object projection bounds reference `Self`");
                    b.projection_term = replace_dummy_self_with_error(tcx, b.projection_term, guar);
                }

                ty::ExistentialPredicate::Projection(ty::ExistentialProjection::erase_self_ty(
                    tcx, b,
                ))
            })
        });

        let mut auto_trait_predicates: Vec<_> = auto_traits
            .into_iter()
            .map(|(trait_pred, _)| {
                assert_eq!(trait_pred.polarity(), ty::PredicatePolarity::Positive);
                assert_eq!(trait_pred.self_ty().skip_binder(), dummy_self);

                ty::Binder::dummy(ty::ExistentialPredicate::AutoTrait(trait_pred.def_id()))
            })
            .collect();
        auto_trait_predicates.dedup();

        // N.b. principal, projections, auto traits
        // FIXME: This is actually wrong with multiple principals in regards to symbol mangling
        let mut v = principal_trait_ref
            .into_iter()
            .chain(existential_projections)
            .chain(auto_trait_predicates)
            .collect::<SmallVec<[_; 8]>>();
        v.sort_by(|a, b| a.skip_binder().stable_cmp(tcx, &b.skip_binder()));
        let existential_predicates = tcx.mk_poly_existential_predicates(&v);

        // Use explicitly-specified region bound, unless the bound is missing.
        let region_bound = if !lifetime.is_elided() {
            self.lower_lifetime(lifetime, RegionInferReason::ExplicitObjectLifetime)
        } else {
            self.compute_object_lifetime_bound(span, existential_predicates).unwrap_or_else(|| {
                // Curiously, we prefer object lifetime default for `+ '_`...
                if tcx.named_bound_var(lifetime.hir_id).is_some() {
                    self.lower_lifetime(lifetime, RegionInferReason::ExplicitObjectLifetime)
                } else {
                    let reason =
                        if let hir::LifetimeKind::ImplicitObjectLifetimeDefault = lifetime.kind {
                            if let hir::Node::Ty(hir::Ty {
                                kind: hir::TyKind::Ref(parent_lifetime, _),
                                ..
                            }) = tcx.parent_hir_node(hir_id)
                                && tcx.named_bound_var(parent_lifetime.hir_id).is_none()
                            {
                                // Parent lifetime must have failed to resolve. Don't emit a redundant error.
                                RegionInferReason::ExplicitObjectLifetime
                            } else {
                                RegionInferReason::ObjectLifetimeDefault
                            }
                        } else {
                            RegionInferReason::ExplicitObjectLifetime
                        };
                    self.re_infer(span, reason)
                }
            })
        };
        debug!(?region_bound);

        Ty::new_dynamic(tcx, existential_predicates, region_bound)
    }

    /// Check that elaborating the principal of a trait ref doesn't lead to projections
    /// that are unconstrained. This can happen because an otherwise unconstrained
    /// *type variable* can be substituted with a type that has late-bound regions. See
    /// `elaborated-predicates-unconstrained-late-bound.rs` for a test.
    fn check_elaborated_projection_mentions_input_lifetimes(
        &self,
        pred: ty::PolyProjectionPredicate<'tcx>,
        span: Span,
        supertrait_span: Span,
    ) {
        let tcx = self.tcx();

        // Find any late-bound regions declared in `ty` that are not
        // declared in the trait-ref or assoc_item. These are not well-formed.
        //
        // Example:
        //
        //     for<'a> <T as Iterator>::Item = &'a str // <-- 'a is bad
        //     for<'a> <T as FnMut<(&'a u32,)>>::Output = &'a str // <-- 'a is ok
        let late_bound_in_projection_term =
            tcx.collect_constrained_late_bound_regions(pred.map_bound(|pred| pred.projection_term));
        let late_bound_in_term =
            tcx.collect_referenced_late_bound_regions(pred.map_bound(|pred| pred.term));
        debug!(?late_bound_in_projection_term);
        debug!(?late_bound_in_term);

        // FIXME: point at the type params that don't have appropriate lifetimes:
        // struct S1<F: for<'a> Fn(&i32, &i32) -> &'a i32>(F);
        //                         ----  ----     ^^^^^^^
        // NOTE(associated_const_equality): This error should be impossible to trigger
        //                                  with associated const equality constraints.
        self.validate_late_bound_regions(
            late_bound_in_projection_term,
            late_bound_in_term,
            |br_name| {
                let item_name = tcx.item_name(pred.item_def_id());
                struct_span_code_err!(
                    self.dcx(),
                    span,
                    E0582,
                    "binding for associated type `{}` references {}, \
                             which does not appear in the trait input types",
                    item_name,
                    br_name
                )
                .with_span_label(supertrait_span, "due to this supertrait")
            },
        );
    }

    /// Prohibit or lint against *bare* trait object types depending on the edition.
    ///
    /// *Bare* trait object types are ones that aren't preceded by the keyword `dyn`.
    /// In edition 2021 and onward we emit a hard error for them.
    fn prohibit_or_lint_bare_trait_object_ty(
        &self,
        span: Span,
        hir_id: hir::HirId,
        hir_bounds: &[hir::PolyTraitRef<'tcx>],
    ) -> Option<ErrorGuaranteed> {
        let tcx = self.tcx();
        let [poly_trait_ref, ..] = hir_bounds else { return None };

        let in_path = match tcx.parent_hir_node(hir_id) {
            hir::Node::Ty(hir::Ty {
                kind: hir::TyKind::Path(hir::QPath::TypeRelative(qself, _)),
                ..
            })
            | hir::Node::Expr(hir::Expr {
                kind: hir::ExprKind::Path(hir::QPath::TypeRelative(qself, _)),
                ..
            })
            | hir::Node::PatExpr(hir::PatExpr {
                kind: hir::PatExprKind::Path(hir::QPath::TypeRelative(qself, _)),
                ..
            }) if qself.hir_id == hir_id => true,
            _ => false,
        };
        let needs_bracket = in_path
            && !tcx
                .sess
                .source_map()
                .span_to_prev_source(span)
                .ok()
                .is_some_and(|s| s.trim_end().ends_with('<'));

        let is_global = poly_trait_ref.trait_ref.path.is_global();

        let mut sugg = vec![(
            span.shrink_to_lo(),
            format!(
                "{}dyn {}",
                if needs_bracket { "<" } else { "" },
                if is_global { "(" } else { "" },
            ),
        )];

        if is_global || needs_bracket {
            sugg.push((
                span.shrink_to_hi(),
                format!(
                    "{}{}",
                    if is_global { ")" } else { "" },
                    if needs_bracket { ">" } else { "" },
                ),
            ));
        }

        if span.edition().at_least_rust_2021() {
            let mut diag = rustc_errors::struct_span_code_err!(
                self.dcx(),
                span,
                E0782,
                "{}",
                "expected a type, found a trait"
            );
            if span.can_be_used_for_suggestions()
                && poly_trait_ref.trait_ref.trait_def_id().is_some()
                && !self.maybe_suggest_impl_trait(span, hir_id, hir_bounds, &mut diag)
                && !self.maybe_suggest_dyn_trait(hir_id, sugg, &mut diag)
            {
                self.maybe_suggest_add_generic_impl_trait(span, hir_id, &mut diag);
            }
            // Check if the impl trait that we are considering is an impl of a local trait.
            self.maybe_suggest_blanket_trait_impl(span, hir_id, &mut diag);
            self.maybe_suggest_assoc_ty_bound(hir_id, &mut diag);
            self.maybe_suggest_typoed_method(
                hir_id,
                poly_trait_ref.trait_ref.trait_def_id(),
                &mut diag,
            );
            // In case there is an associated type with the same name
            // Add the suggestion to this error
            if let Some(mut sugg) =
                self.dcx().steal_non_err(span, StashKey::AssociatedTypeSuggestion)
                && let Suggestions::Enabled(ref mut s1) = diag.suggestions
                && let Suggestions::Enabled(ref mut s2) = sugg.suggestions
            {
                s1.append(s2);
                sugg.cancel();
            }
            Some(diag.emit())
        } else {
            tcx.node_span_lint(BARE_TRAIT_OBJECTS, hir_id, span, |lint| {
                lint.primary_message("trait objects without an explicit `dyn` are deprecated");
                if span.can_be_used_for_suggestions() {
                    lint.multipart_suggestion_verbose(
                        "if this is a dyn-compatible trait, use `dyn`",
                        sugg,
                        Applicability::MachineApplicable,
                    );
                }
                self.maybe_suggest_blanket_trait_impl(span, hir_id, lint);
            });
            None
        }
    }

    /// For a struct or enum with an invalid bare trait object field, suggest turning
    /// it into a generic type bound.
    fn maybe_suggest_add_generic_impl_trait(
        &self,
        span: Span,
        hir_id: hir::HirId,
        diag: &mut Diag<'_>,
    ) -> bool {
        let tcx = self.tcx();

        let parent_hir_id = tcx.parent_hir_id(hir_id);
        let parent_item = tcx.hir_get_parent_item(hir_id).def_id;

        let generics = match tcx.hir_node_by_def_id(parent_item) {
            hir::Node::Item(hir::Item {
                kind: hir::ItemKind::Struct(_, generics, variant),
                ..
            }) => {
                if !variant.fields().iter().any(|field| field.hir_id == parent_hir_id) {
                    return false;
                }
                generics
            }
            hir::Node::Item(hir::Item { kind: hir::ItemKind::Enum(_, generics, def), .. }) => {
                if !def
                    .variants
                    .iter()
                    .flat_map(|variant| variant.data.fields().iter())
                    .any(|field| field.hir_id == parent_hir_id)
                {
                    return false;
                }
                generics
            }
            _ => return false,
        };

        let Ok(rendered_ty) = tcx.sess.source_map().span_to_snippet(span) else {
            return false;
        };

        let param = "TUV"
            .chars()
            .map(|c| c.to_string())
            .chain((0..).map(|i| format!("P{i}")))
            .find(|s| !generics.params.iter().any(|param| param.name.ident().as_str() == s))
            .expect("we definitely can find at least one param name to generate");
        let mut sugg = vec![(span, param.to_string())];
        if let Some(insertion_span) = generics.span_for_param_suggestion() {
            sugg.push((insertion_span, format!(", {param}: {}", rendered_ty)));
        } else {
            sugg.push((generics.where_clause_span, format!("<{param}: {}>", rendered_ty)));
        }
        diag.multipart_suggestion_verbose(
            "you might be missing a type parameter",
            sugg,
            Applicability::MachineApplicable,
        );
        true
    }

    /// Make sure that we are in the condition to suggest the blanket implementation.
    fn maybe_suggest_blanket_trait_impl<G: EmissionGuarantee>(
        &self,
        span: Span,
        hir_id: hir::HirId,
        diag: &mut Diag<'_, G>,
    ) {
        let tcx = self.tcx();
        let parent_id = tcx.hir_get_parent_item(hir_id).def_id;
        if let hir::Node::Item(hir::Item {
            kind: hir::ItemKind::Impl(hir::Impl { self_ty: impl_self_ty, of_trait, generics, .. }),
            ..
        }) = tcx.hir_node_by_def_id(parent_id)
            && hir_id == impl_self_ty.hir_id
        {
            let Some(of_trait) = of_trait else {
                diag.span_suggestion_verbose(
                    impl_self_ty.span.shrink_to_hi(),
                    "you might have intended to implement this trait for a given type",
                    format!(" for /* Type */"),
                    Applicability::HasPlaceholders,
                );
                return;
            };
            if !of_trait.trait_ref.trait_def_id().is_some_and(|def_id| def_id.is_local()) {
                return;
            }
            let of_trait_span = of_trait.trait_ref.path.span;
            // make sure that we are not calling unwrap to abort during the compilation
            let Ok(of_trait_name) = tcx.sess.source_map().span_to_snippet(of_trait_span) else {
                return;
            };

            let Ok(impl_trait_name) = self.tcx().sess.source_map().span_to_snippet(span) else {
                return;
            };
            let sugg = self.add_generic_param_suggestion(generics, span, &impl_trait_name);
            diag.multipart_suggestion(
                format!(
                    "alternatively use a blanket implementation to implement `{of_trait_name}` for \
                     all types that also implement `{impl_trait_name}`"
                ),
                sugg,
                Applicability::MaybeIncorrect,
            );
        }
    }

    /// Try our best to approximate when adding `dyn` would be helpful for a bare
    /// trait object.
    ///
    /// Right now, this is if the type is either directly nested in another ty,
    /// or if it's in the tail field within a struct. This approximates what the
    /// user would've gotten on edition 2015, except for the case where we have
    /// an *obvious* knock-on `Sized` error.
    fn maybe_suggest_dyn_trait(
        &self,
        hir_id: hir::HirId,
        sugg: Vec<(Span, String)>,
        diag: &mut Diag<'_>,
    ) -> bool {
        let tcx = self.tcx();

        // Look at the direct HIR parent, since we care about the relationship between
        // the type and the thing that directly encloses it.
        match tcx.parent_hir_node(hir_id) {
            // These are all generally ok. Namely, when a trait object is nested
            // into another expression or ty, it's either very certain that they
            // missed the ty (e.g. `&Trait`) or it's not really possible to tell
            // what their intention is, so let's not give confusing suggestions and
            // just mention `dyn`. The user can make up their mind what to do here.
            hir::Node::Ty(_)
            | hir::Node::Expr(_)
            | hir::Node::PatExpr(_)
            | hir::Node::PathSegment(_)
            | hir::Node::AssocItemConstraint(_)
            | hir::Node::TraitRef(_)
            | hir::Node::Item(_)
            | hir::Node::WherePredicate(_) => {}

            hir::Node::Field(field) => {
                // Enums can't have unsized fields, fields can only have an unsized tail field.
                if let hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::Struct(_, _, variant), ..
                }) = tcx.parent_hir_node(field.hir_id)
                    && variant
                        .fields()
                        .last()
                        .is_some_and(|tail_field| tail_field.hir_id == field.hir_id)
                {
                    // Ok
                } else {
                    return false;
                }
            }
            _ => return false,
        }

        // FIXME: Only emit this suggestion if the trait is dyn-compatible.
        diag.multipart_suggestion_verbose(
            "you can add the `dyn` keyword if you want a trait object",
            sugg,
            Applicability::MachineApplicable,
        );
        true
    }

    fn add_generic_param_suggestion(
        &self,
        generics: &hir::Generics<'_>,
        self_ty_span: Span,
        impl_trait_name: &str,
    ) -> Vec<(Span, String)> {
        // check if the trait has generics, to make a correct suggestion
        let param_name = generics.params.next_type_param_name(None);

        let add_generic_sugg = if let Some(span) = generics.span_for_param_suggestion() {
            (span, format!(", {param_name}: {impl_trait_name}"))
        } else {
            (generics.span, format!("<{param_name}: {impl_trait_name}>"))
        };
        vec![(self_ty_span, param_name), add_generic_sugg]
    }

    /// Make sure that we are in the condition to suggest `impl Trait`.
    fn maybe_suggest_impl_trait(
        &self,
        span: Span,
        hir_id: hir::HirId,
        hir_bounds: &[hir::PolyTraitRef<'tcx>],
        diag: &mut Diag<'_>,
    ) -> bool {
        let tcx = self.tcx();
        let parent_id = tcx.hir_get_parent_item(hir_id).def_id;
        // FIXME: If `type_alias_impl_trait` is enabled, also look for `Trait0<Ty = Trait1>`
        //        and suggest `Trait0<Ty = impl Trait1>`.
        // Functions are found in three different contexts.
        // 1. Independent functions
        // 2. Functions inside trait blocks
        // 3. Functions inside impl blocks
        let (sig, generics) = match tcx.hir_node_by_def_id(parent_id) {
            hir::Node::Item(hir::Item {
                kind: hir::ItemKind::Fn { sig, generics, .. }, ..
            }) => (sig, generics),
            hir::Node::TraitItem(hir::TraitItem {
                kind: hir::TraitItemKind::Fn(sig, _),
                generics,
                ..
            }) => (sig, generics),
            hir::Node::ImplItem(hir::ImplItem {
                kind: hir::ImplItemKind::Fn(sig, _),
                generics,
                ..
            }) => (sig, generics),
            _ => return false,
        };
        let Ok(trait_name) = tcx.sess.source_map().span_to_snippet(span) else {
            return false;
        };
        let impl_sugg = vec![(span.shrink_to_lo(), "impl ".to_string())];
        // Check if trait object is safe for suggesting dynamic dispatch.
        let is_dyn_compatible = hir_bounds.iter().all(|bound| match bound.trait_ref.path.res {
            Res::Def(DefKind::Trait, id) => tcx.is_dyn_compatible(id),
            _ => false,
        });

        let borrowed = matches!(
            tcx.parent_hir_node(hir_id),
            hir::Node::Ty(hir::Ty { kind: hir::TyKind::Ref(..), .. })
        );

        // Suggestions for function return type.
        if let hir::FnRetTy::Return(ty) = sig.decl.output
            && ty.peel_refs().hir_id == hir_id
        {
            let pre = if !is_dyn_compatible {
                format!("`{trait_name}` is dyn-incompatible, ")
            } else {
                String::new()
            };
            let msg = format!(
                "{pre}use `impl {trait_name}` to return an opaque type, as long as you return a \
                 single underlying type",
            );

            diag.multipart_suggestion_verbose(msg, impl_sugg, Applicability::MachineApplicable);

            // Suggest `Box<dyn Trait>` for return type
            if is_dyn_compatible {
                // If the return type is `&Trait`, we don't want
                // the ampersand to be displayed in the `Box<dyn Trait>`
                // suggestion.
                let suggestion = if borrowed {
                    vec![(ty.span, format!("Box<dyn {trait_name}>"))]
                } else {
                    vec![
                        (ty.span.shrink_to_lo(), "Box<dyn ".to_string()),
                        (ty.span.shrink_to_hi(), ">".to_string()),
                    ]
                };

                diag.multipart_suggestion_verbose(
                    "alternatively, you can return an owned trait object",
                    suggestion,
                    Applicability::MachineApplicable,
                );
            }
            return true;
        }

        // Suggestions for function parameters.
        for ty in sig.decl.inputs {
            if ty.peel_refs().hir_id != hir_id {
                continue;
            }
            let sugg = self.add_generic_param_suggestion(generics, span, &trait_name);
            diag.multipart_suggestion_verbose(
                format!("use a new generic type parameter, constrained by `{trait_name}`"),
                sugg,
                Applicability::MachineApplicable,
            );
            diag.multipart_suggestion_verbose(
                "you can also use an opaque type, but users won't be able to specify the type \
                 parameter when calling the `fn`, having to rely exclusively on type inference",
                impl_sugg,
                Applicability::MachineApplicable,
            );
            if !is_dyn_compatible {
                diag.note(format!(
                    "`{trait_name}` is dyn-incompatible, otherwise a trait object could be used"
                ));
            } else {
                // No ampersand in suggestion if it's borrowed already
                let (dyn_str, paren_dyn_str) =
                    if borrowed { ("dyn ", "(dyn ") } else { ("&dyn ", "&(dyn ") };

                let sugg = if let [_, _, ..] = hir_bounds {
                    // There is more than one trait bound, we need surrounding parentheses.
                    vec![
                        (span.shrink_to_lo(), paren_dyn_str.to_string()),
                        (span.shrink_to_hi(), ")".to_string()),
                    ]
                } else {
                    vec![(span.shrink_to_lo(), dyn_str.to_string())]
                };
                diag.multipart_suggestion_verbose(
                    format!(
                        "alternatively, use a trait object to accept any type that implements \
                         `{trait_name}`, accessing its methods at runtime using dynamic dispatch",
                    ),
                    sugg,
                    Applicability::MachineApplicable,
                );
            }
            return true;
        }
        false
    }

    fn maybe_suggest_assoc_ty_bound(&self, hir_id: hir::HirId, diag: &mut Diag<'_>) {
        let mut parents = self.tcx().hir_parent_iter(hir_id);

        if let Some((c_hir_id, hir::Node::AssocItemConstraint(constraint))) = parents.next()
            && let Some(obj_ty) = constraint.ty()
            && let Some((_, hir::Node::TraitRef(trait_ref))) = parents.next()
        {
            if let Some((_, hir::Node::Ty(ty))) = parents.next()
                && let hir::TyKind::TraitObject(..) = ty.kind
            {
                // Assoc ty bounds aren't permitted inside trait object types.
                return;
            }

            if trait_ref
                .path
                .segments
                .iter()
                .find_map(|seg| {
                    seg.args.filter(|args| args.constraints.iter().any(|c| c.hir_id == c_hir_id))
                })
                .is_none_or(|args| args.parenthesized != hir::GenericArgsParentheses::No)
            {
                // Only consider angle-bracketed args (where we have a `=` to replace with `:`).
                return;
            }

            let lo = if constraint.gen_args.span_ext.is_dummy() {
                constraint.ident.span
            } else {
                constraint.gen_args.span_ext
            };
            let hi = obj_ty.span;

            if !lo.eq_ctxt(hi) {
                return;
            }

            diag.span_suggestion_verbose(
                lo.between(hi),
                "you might have meant to write a bound here",
                ": ",
                Applicability::MaybeIncorrect,
            );
        }
    }

    fn maybe_suggest_typoed_method(
        &self,
        hir_id: hir::HirId,
        trait_def_id: Option<DefId>,
        diag: &mut Diag<'_>,
    ) {
        let tcx = self.tcx();
        let Some(trait_def_id) = trait_def_id else {
            return;
        };
        let hir::Node::Expr(hir::Expr {
            kind: hir::ExprKind::Path(hir::QPath::TypeRelative(path_ty, segment)),
            ..
        }) = tcx.parent_hir_node(hir_id)
        else {
            return;
        };
        if path_ty.hir_id != hir_id {
            return;
        }
        let names: Vec<_> = tcx
            .associated_items(trait_def_id)
            .in_definition_order()
            .filter(|assoc| assoc.namespace() == hir::def::Namespace::ValueNS)
            .map(|cand| cand.name())
            .collect();
        if let Some(typo) = find_best_match_for_name(&names, segment.ident.name, None) {
            diag.span_suggestion_verbose(
                segment.ident.span,
                format!(
                    "you may have misspelled this associated item, causing `{}` \
                    to be interpreted as a type rather than a trait",
                    tcx.item_name(trait_def_id),
                ),
                typo,
                Applicability::MaybeIncorrect,
            );
        }
    }
}

fn replace_dummy_self_with_error<'tcx, T: TypeFoldable<TyCtxt<'tcx>>>(
    tcx: TyCtxt<'tcx>,
    t: T,
    guar: ErrorGuaranteed,
) -> T {
    t.fold_with(&mut BottomUpFolder {
        tcx,
        ty_op: |ty| {
            if ty == tcx.types.trait_object_dummy_self { Ty::new_error(tcx, guar) } else { ty }
        },
        lt_op: |lt| lt,
        ct_op: |ct| ct,
    })
}
