use crate::astconv::{GenericArgCountMismatch, GenericArgCountResult, OnlySelfBounds};
use crate::bounds::Bounds;
use crate::errors::TraitObjectDeclaredWithNoTraits;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_lint_defs::builtin::UNUSED_ASSOCIATED_TYPE_BOUNDS;
use rustc_middle::ty::{self, Ty};
use rustc_middle::ty::{DynKind, ToPredicate};
use rustc_span::Span;
use rustc_trait_selection::traits::error_reporting::report_object_safety_error;
use rustc_trait_selection::traits::{self, astconv_object_safety_violations};

use smallvec::{smallvec, SmallVec};
use std::collections::BTreeSet;

use super::AstConv;

impl<'o, 'tcx> dyn AstConv<'tcx> + 'o {
    pub(super) fn conv_object_ty_poly_trait_ref(
        &self,
        span: Span,
        hir_id: hir::HirId,
        hir_trait_bounds: &[hir::PolyTraitRef<'_>],
        lifetime: &hir::Lifetime,
        borrowed: bool,
        representation: DynKind,
    ) -> Ty<'tcx> {
        let tcx = self.tcx();

        let mut bounds = Bounds::default();
        let mut potential_assoc_types = Vec::new();
        let dummy_self = self.tcx().types.trait_object_dummy_self;
        for trait_bound in hir_trait_bounds.iter().rev() {
            if let GenericArgCountResult {
                correct:
                    Err(GenericArgCountMismatch { invalid_args: cur_potential_assoc_types, .. }),
                ..
            } = self.instantiate_poly_trait_ref(
                &trait_bound.trait_ref,
                trait_bound.span,
                ty::BoundConstness::NotConst,
                ty::ImplPolarity::Positive,
                dummy_self,
                &mut bounds,
                false,
                // FIXME: This should be `true`, but we don't really handle
                // associated type bounds or type aliases in objects in a way
                // that makes this meaningful, I think.
                OnlySelfBounds(false),
            ) {
                potential_assoc_types.extend(cur_potential_assoc_types);
            }
        }

        let mut trait_bounds = vec![];
        let mut projection_bounds = vec![];
        for (pred, span) in bounds.clauses() {
            let bound_pred = pred.kind();
            match bound_pred.skip_binder() {
                ty::ClauseKind::Trait(trait_pred) => {
                    assert_eq!(trait_pred.polarity, ty::ImplPolarity::Positive);
                    trait_bounds.push((
                        bound_pred.rebind(trait_pred.trait_ref),
                        span,
                        trait_pred.constness,
                    ));
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
                    bug!()
                }
            }
        }

        // Expand trait aliases recursively and check that only one regular (non-auto) trait
        // is used and no 'maybe' bounds are used.
        let expanded_traits =
            traits::expand_trait_aliases(tcx, trait_bounds.iter().map(|&(a, b, _)| (a, b)));

        let (mut auto_traits, regular_traits): (Vec<_>, Vec<_>) = expanded_traits
            .filter(|i| i.trait_ref().self_ty().skip_binder() == dummy_self)
            .partition(|i| tcx.trait_is_auto(i.trait_ref().def_id()));
        if regular_traits.len() > 1 {
            let first_trait = &regular_traits[0];
            let additional_trait = &regular_traits[1];
            let mut err = struct_span_err!(
                tcx.sess,
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
                    .map(|t| t.trait_ref().print_only_trait_path().to_string())
                    .collect::<Vec<_>>()
                    .join(" + "),
            ));
            err.note(
                "auto-traits like `Send` and `Sync` are traits that have special properties; \
             for more information on them, visit \
             <https://doc.rust-lang.org/reference/special-types-and-traits.html#auto-traits>",
            );
            err.emit();
        }

        if regular_traits.is_empty() && auto_traits.is_empty() {
            let trait_alias_span = trait_bounds
                .iter()
                .map(|&(trait_ref, _, _)| trait_ref.def_id())
                .find(|&trait_ref| tcx.is_trait_alias(trait_ref))
                .map(|trait_ref| tcx.def_span(trait_ref));
            let reported =
                tcx.sess.emit_err(TraitObjectDeclaredWithNoTraits { span, trait_alias_span });
            return Ty::new_error(tcx, reported);
        }

        // Check that there are no gross object safety violations;
        // most importantly, that the supertraits don't contain `Self`,
        // to avoid ICEs.
        for item in &regular_traits {
            let object_safety_violations =
                astconv_object_safety_violations(tcx, item.trait_ref().def_id());
            if !object_safety_violations.is_empty() {
                let reported = report_object_safety_error(
                    tcx,
                    span,
                    item.trait_ref().def_id(),
                    &object_safety_violations,
                )
                .emit();
                return Ty::new_error(tcx, reported);
            }
        }

        // Use a `BTreeSet` to keep output in a more consistent order.
        let mut associated_types: FxHashMap<Span, BTreeSet<DefId>> = FxHashMap::default();

        let regular_traits_refs_spans = trait_bounds
            .into_iter()
            .filter(|(trait_ref, _, _)| !tcx.trait_is_auto(trait_ref.def_id()));

        for (base_trait_ref, span, constness) in regular_traits_refs_spans {
            assert_eq!(constness, ty::BoundConstness::NotConst);
            let base_pred: ty::Predicate<'tcx> = base_trait_ref.to_predicate(tcx);
            for pred in traits::elaborate(tcx, [base_pred]) {
                debug!("conv_object_ty_poly_trait_ref: observing object predicate `{:?}`", pred);

                let bound_predicate = pred.kind();
                match bound_predicate.skip_binder() {
                    ty::PredicateKind::Clause(ty::ClauseKind::Trait(pred)) => {
                        let pred = bound_predicate.rebind(pred);
                        associated_types.entry(span).or_default().extend(
                            tcx.associated_items(pred.def_id())
                                .in_definition_order()
                                .filter(|item| item.kind == ty::AssocKind::Type)
                                .filter(|item| item.opt_rpitit_info.is_none())
                                .map(|item| item.def_id),
                        );
                    }
                    ty::PredicateKind::Clause(ty::ClauseKind::Projection(pred)) => {
                        let pred = bound_predicate.rebind(pred);
                        // A `Self` within the original bound will be substituted with a
                        // `trait_object_dummy_self`, so check for that.
                        let references_self = match pred.skip_binder().term.unpack() {
                            ty::TermKind::Ty(ty) => ty.walk().any(|arg| arg == dummy_self.into()),
                            ty::TermKind::Const(c) => {
                                c.ty().walk().any(|arg| arg == dummy_self.into())
                            }
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
                            projection_bounds.push((pred, span));
                        }
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
                def_ids.remove(&def_id);
                if tcx.generics_require_sized_self(def_id) {
                    tcx.emit_spanned_lint(
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

        self.complain_about_missing_associated_types(
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
        debug!("regular_traits: {:?}", regular_traits);
        debug!("auto_traits: {:?}", auto_traits);

        // Erase the `dummy_self` (`trait_object_dummy_self`) used above.
        let existential_trait_refs = regular_traits.iter().map(|i| {
            i.trait_ref().map_bound(|trait_ref: ty::TraitRef<'tcx>| {
                assert_eq!(trait_ref.self_ty(), dummy_self);

                // Verify that `dummy_self` did not leak inside default type parameters. This
                // could not be done at path creation, since we need to see through trait aliases.
                let mut missing_type_params = vec![];
                let mut references_self = false;
                let generics = tcx.generics_of(trait_ref.def_id);
                let substs: Vec<_> = trait_ref
                    .substs
                    .iter()
                    .enumerate()
                    .skip(1) // Remove `Self` for `ExistentialPredicate`.
                    .map(|(index, arg)| {
                        if arg == dummy_self.into() {
                            let param = &generics.params[index];
                            missing_type_params.push(param.name);
                            return Ty::new_misc_error(tcx).into();
                        } else if arg.walk().any(|arg| arg == dummy_self.into()) {
                            references_self = true;
                            return Ty::new_misc_error(tcx).into();
                        }
                        arg
                    })
                    .collect();
                let substs = tcx.mk_substs(&substs);

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
                    let mut err = struct_span_err!(
                        tcx.sess,
                        i.bottom().1,
                        E0038,
                        "the {} `{}` cannot be made into an object",
                        tcx.def_descr(def_id),
                        tcx.item_name(def_id),
                    );
                    err.note(
                        rustc_middle::traits::ObjectSafetyViolation::SupertraitSelf(smallvec![])
                            .error_msg(),
                    );
                    err.emit();
                }

                ty::ExistentialTraitRef { def_id: trait_ref.def_id, substs }
            })
        });

        let existential_projections = projection_bounds
            .iter()
            // We filter out traits that don't have `Self` as their self type above,
            // we need to do the same for projections.
            .filter(|(bound, _)| bound.skip_binder().self_ty() == dummy_self)
            .map(|(bound, _)| {
                bound.map_bound(|mut b| {
                    assert_eq!(b.projection_ty.self_ty(), dummy_self);

                    // Like for trait refs, verify that `dummy_self` did not leak inside default type
                    // parameters.
                    let references_self = b.projection_ty.substs.iter().skip(1).any(|arg| {
                        if arg.walk().any(|arg| arg == dummy_self.into()) {
                            return true;
                        }
                        false
                    });
                    if references_self {
                        let guar = tcx.sess.delay_span_bug(
                            span,
                            "trait object projection bounds reference `Self`",
                        );
                        let substs: Vec<_> = b
                            .projection_ty
                            .substs
                            .iter()
                            .map(|arg| {
                                if arg.walk().any(|arg| arg == dummy_self.into()) {
                                    return Ty::new_error(tcx, guar).into();
                                }
                                arg
                            })
                            .collect();
                        b.projection_ty.substs = tcx.mk_substs(&substs);
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

        // Use explicitly-specified region bound.
        let region_bound = if !lifetime.is_elided() {
            self.ast_region_to_region(lifetime, None)
        } else {
            self.compute_object_lifetime_bound(span, existential_predicates).unwrap_or_else(|| {
                if tcx.named_bound_var(lifetime.hir_id).is_some() {
                    self.ast_region_to_region(lifetime, None)
                } else {
                    self.re_infer(None, span).unwrap_or_else(|| {
                        let mut err = struct_span_err!(
                            tcx.sess,
                            span,
                            E0228,
                            "the lifetime bound for this object type cannot be deduced \
                         from context; please supply an explicit bound"
                        );
                        let e = if borrowed {
                            // We will have already emitted an error E0106 complaining about a
                            // missing named lifetime in `&dyn Trait`, so we elide this one.
                            err.delay_as_bug()
                        } else {
                            err.emit()
                        };
                        ty::Region::new_error(tcx, e)
                    })
                }
            })
        };
        debug!("region_bound: {:?}", region_bound);

        let ty = Ty::new_dynamic(tcx, existential_predicates, region_bound, representation);
        debug!("trait_object_type: {:?}", ty);
        ty
    }
}
