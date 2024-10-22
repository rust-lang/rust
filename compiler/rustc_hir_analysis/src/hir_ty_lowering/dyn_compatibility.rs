use rustc_data_structures::fx::{FxHashSet, FxIndexMap, FxIndexSet};
use rustc_errors::codes::*;
use rustc_errors::struct_span_code_err;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_lint_defs::builtin::UNUSED_ASSOCIATED_TYPE_BOUNDS;
use rustc_middle::span_bug;
use rustc_middle::ty::fold::BottomUpFolder;
use rustc_middle::ty::{
    self, DynKind, ExistentialPredicateStableCmpExt as _, Ty, TyCtxt, TypeFoldable, Upcast,
};
use rustc_span::{ErrorGuaranteed, Span};
use rustc_trait_selection::error_reporting::traits::report_dyn_incompatibility;
use rustc_trait_selection::traits::{self, hir_ty_lowering_dyn_compatibility_violations};
use rustc_type_ir::elaborate::ClauseWithSupertraitSpan;
use smallvec::{SmallVec, smallvec};
use tracing::{debug, instrument};

use super::HirTyLowerer;
use crate::bounds::Bounds;
use crate::hir_ty_lowering::{
    GenericArgCountMismatch, GenericArgCountResult, PredicateFilter, RegionInferReason,
};

impl<'tcx> dyn HirTyLowerer<'tcx> + '_ {
    /// Lower a trait object type from the HIR to our internal notion of a type.
    #[instrument(level = "debug", skip_all, ret)]
    pub(super) fn lower_trait_object_ty(
        &self,
        span: Span,
        hir_id: hir::HirId,
        hir_trait_bounds: &[hir::PolyTraitRef<'tcx>],
        lifetime: &hir::Lifetime,
        representation: DynKind,
    ) -> Ty<'tcx> {
        let tcx = self.tcx();

        let mut bounds = Bounds::default();
        let mut potential_assoc_types = Vec::new();
        let dummy_self = self.tcx().types.trait_object_dummy_self;
        for trait_bound in hir_trait_bounds.iter().rev() {
            // FIXME: This doesn't handle `? const`.
            if trait_bound.modifiers == hir::TraitBoundModifier::Maybe {
                continue;
            }
            if let GenericArgCountResult {
                correct:
                    Err(GenericArgCountMismatch { invalid_args: cur_potential_assoc_types, .. }),
                ..
            } = self.lower_poly_trait_ref(
                &trait_bound.trait_ref,
                trait_bound.span,
                None,
                ty::PredicatePolarity::Positive,
                dummy_self,
                &mut bounds,
                PredicateFilter::SelfOnly,
            ) {
                potential_assoc_types.extend(cur_potential_assoc_types);
            }
        }

        let mut trait_bounds = vec![];
        let mut projection_bounds = vec![];
        for (pred, span) in bounds.clauses(tcx) {
            let bound_pred = pred.kind();
            match bound_pred.skip_binder() {
                ty::ClauseKind::Trait(trait_pred) => {
                    assert_eq!(trait_pred.polarity, ty::PredicatePolarity::Positive);
                    trait_bounds.push((bound_pred.rebind(trait_pred.trait_ref), span));
                }
                ty::ClauseKind::Projection(proj) => {
                    projection_bounds.push((bound_pred.rebind(proj), span));
                }
                ty::ClauseKind::TypeOutlives(_) => {
                    // Do nothing, we deal with regions separately
                }
                ty::ClauseKind::RegionOutlives(_)
                | ty::ClauseKind::ConstArgHasType(..)
                | ty::ClauseKind::WellFormed(_)
                | ty::ClauseKind::ConstEvaluatable(_) => {
                    span_bug!(span, "did not expect {pred} clause in object bounds");
                }
            }
        }

        // Expand trait aliases recursively and check that only one regular (non-auto) trait
        // is used and no 'maybe' bounds are used.
        let expanded_traits =
            traits::expand_trait_aliases(tcx, trait_bounds.iter().map(|&(a, b)| (a, b)));

        let (mut auto_traits, regular_traits): (Vec<_>, Vec<_>) =
            expanded_traits.partition(|i| tcx.trait_is_auto(i.trait_ref().def_id()));
        if regular_traits.len() > 1 {
            let _ = self.report_trait_object_addition_traits_error(&regular_traits);
        } else if regular_traits.is_empty() && auto_traits.is_empty() {
            let reported = self.report_trait_object_with_no_traits_error(span, &trait_bounds);
            return Ty::new_error(tcx, reported);
        }

        // Check that there are no gross dyn-compatibility violations;
        // most importantly, that the supertraits don't contain `Self`,
        // to avoid ICEs.
        for item in &regular_traits {
            let violations =
                hir_ty_lowering_dyn_compatibility_violations(tcx, item.trait_ref().def_id());
            if !violations.is_empty() {
                let reported = report_dyn_incompatibility(
                    tcx,
                    span,
                    Some(hir_id),
                    item.trait_ref().def_id(),
                    &violations,
                )
                .emit();
                return Ty::new_error(tcx, reported);
            }
        }

        let mut associated_types: FxIndexMap<Span, FxIndexSet<DefId>> = FxIndexMap::default();

        let regular_traits_refs_spans = trait_bounds
            .into_iter()
            .filter(|(trait_ref, _)| !tcx.trait_is_auto(trait_ref.def_id()));

        for (base_trait_ref, original_span) in regular_traits_refs_spans {
            let base_pred: ty::Predicate<'tcx> = base_trait_ref.upcast(tcx);
            for ClauseWithSupertraitSpan { pred, original_span, supertrait_span } in
                traits::elaborate(tcx, [ClauseWithSupertraitSpan::new(base_pred, original_span)])
                    .filter_only_self()
            {
                debug!("observing object predicate `{pred:?}`");

                let bound_predicate = pred.kind();
                match bound_predicate.skip_binder() {
                    ty::PredicateKind::Clause(ty::ClauseKind::Trait(pred)) => {
                        let pred = bound_predicate.rebind(pred);
                        associated_types.entry(original_span).or_default().extend(
                            tcx.associated_items(pred.def_id())
                                .in_definition_order()
                                .filter(|item| item.kind == ty::AssocKind::Type)
                                .filter(|item| {
                                    !item.is_impl_trait_in_trait() && !item.is_effects_desugaring
                                })
                                .map(|item| item.def_id),
                        );
                    }
                    ty::PredicateKind::Clause(ty::ClauseKind::Projection(pred)) => {
                        let pred = bound_predicate.rebind(pred);
                        // A `Self` within the original bound will be instantiated with a
                        // `trait_object_dummy_self`, so check for that.
                        let references_self = match pred.skip_binder().term.unpack() {
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
                        // Here, the user could theoretically write `dyn MyTrait<Output = X>`,
                        // but actually supporting that would "expand" to an infinitely-long type
                        // `fix $ τ → dyn MyTrait<MyOutput = X, Output = <τ as MyTrait>::MyOutput`.
                        //
                        // Instead, we force the user to write
                        // `dyn MyTrait<MyOutput = X, Output = X>`, which is uglier but works. See
                        // the discussion in #56288 for alternatives.
                        if !references_self {
                            // Include projections defined on supertraits.
                            projection_bounds.push((pred, original_span));
                        }

                        self.check_elaborated_projection_mentions_input_lifetimes(
                            pred,
                            original_span,
                            supertrait_span,
                        );
                    }
                    _ => (),
                }
            }
        }

        // `dyn Trait<Assoc = Foo>` desugars to (not Rust syntax) `dyn Trait where <Self as Trait>::Assoc = Foo`.
        // So every `Projection` clause is an `Assoc = Foo` bound. `associated_types` contains all associated
        // types's `DefId`, so the following loop removes all the `DefIds` of the associated types that have a
        // corresponding `Projection` clause
        for def_ids in associated_types.values_mut() {
            for (projection_bound, span) in &projection_bounds {
                let def_id = projection_bound.projection_def_id();
                // FIXME(#120456) - is `swap_remove` correct?
                def_ids.swap_remove(&def_id);
                if tcx.generics_require_sized_self(def_id) {
                    tcx.emit_node_span_lint(
                        UNUSED_ASSOCIATED_TYPE_BOUNDS,
                        hir_id,
                        *span,
                        crate::errors::UnusedAssociatedTypeBounds { span: *span },
                    );
                }
            }
            // If the associated type has a `where Self: Sized` bound, we do not need to constrain the associated
            // type in the `dyn Trait`.
            def_ids.retain(|def_id| !tcx.generics_require_sized_self(def_id));
        }

        self.complain_about_missing_assoc_tys(
            associated_types,
            potential_assoc_types,
            hir_trait_bounds,
        );

        // De-duplicate auto traits so that, e.g., `dyn Trait + Send + Send` is the same as
        // `dyn Trait + Send`.
        // We remove duplicates by inserting into a `FxHashSet` to avoid re-ordering
        // the bounds
        let mut duplicates = FxHashSet::default();
        auto_traits.retain(|i| duplicates.insert(i.trait_ref().def_id()));
        debug!(?regular_traits);
        debug!(?auto_traits);

        // Erase the `dummy_self` (`trait_object_dummy_self`) used above.
        let existential_trait_refs = regular_traits.iter().map(|i| {
            i.trait_ref().map_bound(|trait_ref: ty::TraitRef<'tcx>| {
                assert_eq!(trait_ref.self_ty(), dummy_self);

                // Verify that `dummy_self` did not leak inside default type parameters. This
                // could not be done at path creation, since we need to see through trait aliases.
                let mut missing_type_params = vec![];
                let mut references_self = false;
                let generics = tcx.generics_of(trait_ref.def_id);
                let args: Vec<_> = trait_ref
                    .args
                    .iter()
                    .enumerate()
                    .skip(1) // Remove `Self` for `ExistentialPredicate`.
                    .map(|(index, arg)| {
                        if arg == dummy_self.into() {
                            let param = &generics.own_params[index];
                            missing_type_params.push(param.name);
                            Ty::new_misc_error(tcx).into()
                        } else if arg.walk().any(|arg| arg == dummy_self.into()) {
                            references_self = true;
                            let guar = self.dcx().span_delayed_bug(
                                span,
                                "trait object trait bounds reference `Self`",
                            );
                            replace_dummy_self_with_error(tcx, arg, guar)
                        } else {
                            arg
                        }
                    })
                    .collect();
                let args = tcx.mk_args(&args);

                let span = i.bottom().1;
                let empty_generic_args = hir_trait_bounds.iter().any(|hir_bound| {
                    hir_bound.trait_ref.path.res == Res::Def(DefKind::Trait, trait_ref.def_id)
                        && hir_bound.span.contains(span)
                });
                self.complain_about_missing_type_params(
                    missing_type_params,
                    trait_ref.def_id,
                    span,
                    empty_generic_args,
                );

                if references_self {
                    let def_id = i.bottom().0.def_id();
                    struct_span_code_err!(
                        self.dcx(),
                        i.bottom().1,
                        E0038,
                        "the {} `{}` cannot be made into an object",
                        tcx.def_descr(def_id),
                        tcx.item_name(def_id),
                    )
                    .with_note(
                        rustc_middle::traits::DynCompatibilityViolation::SupertraitSelf(
                            smallvec![],
                        )
                        .error_msg(),
                    )
                    .emit();
                }

                ty::ExistentialTraitRef { def_id: trait_ref.def_id, args }
            })
        });

        let existential_projections = projection_bounds.iter().map(|(bound, _)| {
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

                ty::ExistentialProjection::erase_self_ty(tcx, b)
            })
        });

        let regular_trait_predicates = existential_trait_refs
            .map(|trait_ref| trait_ref.map_bound(ty::ExistentialPredicate::Trait));
        let auto_trait_predicates = auto_traits.into_iter().map(|trait_ref| {
            ty::Binder::dummy(ty::ExistentialPredicate::AutoTrait(trait_ref.trait_ref().def_id()))
        });
        // N.b. principal, projections, auto traits
        // FIXME: This is actually wrong with multiple principals in regards to symbol mangling
        let mut v = regular_trait_predicates
            .chain(
                existential_projections.map(|x| x.map_bound(ty::ExistentialPredicate::Projection)),
            )
            .chain(auto_trait_predicates)
            .collect::<SmallVec<[_; 8]>>();
        v.sort_by(|a, b| a.skip_binder().stable_cmp(tcx, &b.skip_binder()));
        v.dedup();
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
                        if let hir::LifetimeName::ImplicitObjectLifetimeDefault = lifetime.res {
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

        Ty::new_dynamic(tcx, existential_predicates, region_bound, representation)
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
                let item_name = tcx.item_name(pred.projection_def_id());
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
