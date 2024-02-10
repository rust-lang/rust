use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{codes::*, struct_span_code_err};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::ty::{self as ty, Ty};
use rustc_span::symbol::Ident;
use rustc_span::{ErrorGuaranteed, Span};
use rustc_trait_selection::traits;
use smallvec::SmallVec;

use crate::astconv::{
    AstConv, ConvertedBinding, ConvertedBindingKind, OnlySelfBounds, PredicateFilter,
};
use crate::bounds::Bounds;
use crate::errors;

impl<'tcx> dyn AstConv<'tcx> + '_ {
    /// Sets `implicitly_sized` to true on `Bounds` if necessary
    pub(crate) fn add_implicitly_sized(
        &self,
        bounds: &mut Bounds<'tcx>,
        self_ty: Ty<'tcx>,
        ast_bounds: &'tcx [hir::GenericBound<'tcx>],
        self_ty_where_predicates: Option<(LocalDefId, &'tcx [hir::WherePredicate<'tcx>])>,
        span: Span,
    ) {
        let tcx = self.tcx();
        let sized_def_id = tcx.lang_items().sized_trait();
        let mut seen_negative_sized_bound = false;
        let mut seen_positive_sized_bound = false;

        // Try to find an unbound in bounds.
        let mut unbounds: SmallVec<[_; 1]> = SmallVec::new();
        let mut search_bounds = |ast_bounds: &'tcx [hir::GenericBound<'tcx>]| {
            for ab in ast_bounds {
                let hir::GenericBound::Trait(ptr, modifier) = ab else {
                    continue;
                };
                match modifier {
                    hir::TraitBoundModifier::Maybe => unbounds.push(ptr),
                    hir::TraitBoundModifier::Negative => {
                        if let Some(sized_def_id) = sized_def_id
                            && ptr.trait_ref.path.res == Res::Def(DefKind::Trait, sized_def_id)
                        {
                            seen_negative_sized_bound = true;
                        }
                    }
                    hir::TraitBoundModifier::None => {
                        if let Some(sized_def_id) = sized_def_id
                            && ptr.trait_ref.path.res == Res::Def(DefKind::Trait, sized_def_id)
                        {
                            seen_positive_sized_bound = true;
                        }
                    }
                    _ => {}
                }
            }
        };
        search_bounds(ast_bounds);
        if let Some((self_ty, where_clause)) = self_ty_where_predicates {
            for clause in where_clause {
                if let hir::WherePredicate::BoundPredicate(pred) = clause
                    && pred.is_param_bound(self_ty.to_def_id())
                {
                    search_bounds(pred.bounds);
                }
            }
        }

        if unbounds.len() > 1 {
            tcx.dcx().emit_err(errors::MultipleRelaxedDefaultBounds {
                spans: unbounds.iter().map(|ptr| ptr.span).collect(),
            });
        }

        let mut seen_sized_unbound = false;
        for unbound in unbounds {
            if let Some(sized_def_id) = sized_def_id
                && unbound.trait_ref.path.res == Res::Def(DefKind::Trait, sized_def_id)
            {
                seen_sized_unbound = true;
                continue;
            }
            // There was a `?Trait` bound, but it was not `?Sized`; warn.
            tcx.dcx().span_warn(
                unbound.span,
                "relaxing a default bound only does something for `?Sized`; \
                all other traits are not bound by default",
            );
        }

        if seen_sized_unbound || seen_negative_sized_bound || seen_positive_sized_bound {
            // There was in fact a `?Sized`, `!Sized` or explicit `Sized` bound;
            // we don't need to do anything.
        } else if sized_def_id.is_some() {
            // There was no `?Sized`, `!Sized` or explicit `Sized` bound;
            // add `Sized` if it's available.
            bounds.push_sized(tcx, self_ty, span);
        }
    }

    /// This helper takes a *converted* parameter type (`param_ty`)
    /// and an *unconverted* list of bounds:
    ///
    /// ```text
    /// fn foo<T: Debug>
    ///        ^  ^^^^^ `ast_bounds` parameter, in HIR form
    ///        |
    ///        `param_ty`, in ty form
    /// ```
    ///
    /// It adds these `ast_bounds` into the `bounds` structure.
    ///
    /// **A note on binders:** there is an implied binder around
    /// `param_ty` and `ast_bounds`. See `instantiate_poly_trait_ref`
    /// for more details.
    #[instrument(level = "debug", skip(self, ast_bounds, bounds))]
    pub(crate) fn add_bounds<'hir, I: Iterator<Item = &'hir hir::GenericBound<'tcx>>>(
        &self,
        param_ty: Ty<'tcx>,
        ast_bounds: I,
        bounds: &mut Bounds<'tcx>,
        bound_vars: &'tcx ty::List<ty::BoundVariableKind>,
        only_self_bounds: OnlySelfBounds,
    ) where
        'tcx: 'hir,
    {
        for ast_bound in ast_bounds {
            match ast_bound {
                hir::GenericBound::Trait(poly_trait_ref, modifier) => {
                    let (constness, polarity) = match modifier {
                        hir::TraitBoundModifier::Const => {
                            (ty::BoundConstness::Const, ty::ImplPolarity::Positive)
                        }
                        hir::TraitBoundModifier::MaybeConst => {
                            (ty::BoundConstness::ConstIfConst, ty::ImplPolarity::Positive)
                        }
                        hir::TraitBoundModifier::None => {
                            (ty::BoundConstness::NotConst, ty::ImplPolarity::Positive)
                        }
                        hir::TraitBoundModifier::Negative => {
                            (ty::BoundConstness::NotConst, ty::ImplPolarity::Negative)
                        }
                        hir::TraitBoundModifier::Maybe => continue,
                    };
                    let _ = self.instantiate_poly_trait_ref(
                        &poly_trait_ref.trait_ref,
                        poly_trait_ref.span,
                        constness,
                        polarity,
                        param_ty,
                        bounds,
                        false,
                        only_self_bounds,
                    );
                }
                hir::GenericBound::Outlives(lifetime) => {
                    let region = self.ast_region_to_region(lifetime, None);
                    bounds.push_region_bound(
                        self.tcx(),
                        ty::Binder::bind_with_vars(
                            ty::OutlivesPredicate(param_ty, region),
                            bound_vars,
                        ),
                        lifetime.ident.span,
                    );
                }
            }
        }
    }

    /// Translates a list of bounds from the HIR into the `Bounds` data structure.
    /// The self-type for the bounds is given by `param_ty`.
    ///
    /// Example:
    ///
    /// ```ignore (illustrative)
    /// fn foo<T: Bar + Baz>() { }
    /// //     ^  ^^^^^^^^^ ast_bounds
    /// //     param_ty
    /// ```
    ///
    /// The `sized_by_default` parameter indicates if, in this context, the `param_ty` should be
    /// considered `Sized` unless there is an explicit `?Sized` bound. This would be true in the
    /// example above, but is not true in supertrait listings like `trait Foo: Bar + Baz`.
    ///
    /// `span` should be the declaration size of the parameter.
    pub(crate) fn compute_bounds(
        &self,
        param_ty: Ty<'tcx>,
        ast_bounds: &[hir::GenericBound<'tcx>],
        filter: PredicateFilter,
    ) -> Bounds<'tcx> {
        let mut bounds = Bounds::default();

        let only_self_bounds = match filter {
            PredicateFilter::All | PredicateFilter::SelfAndAssociatedTypeBounds => {
                OnlySelfBounds(false)
            }
            PredicateFilter::SelfOnly | PredicateFilter::SelfThatDefines(_) => OnlySelfBounds(true),
        };

        self.add_bounds(
            param_ty,
            ast_bounds.iter().filter(|bound| match filter {
                PredicateFilter::All
                | PredicateFilter::SelfOnly
                | PredicateFilter::SelfAndAssociatedTypeBounds => true,
                PredicateFilter::SelfThatDefines(assoc_name) => {
                    if let Some(trait_ref) = bound.trait_ref()
                        && let Some(trait_did) = trait_ref.trait_def_id()
                        && self.tcx().trait_may_define_assoc_item(trait_did, assoc_name)
                    {
                        true
                    } else {
                        false
                    }
                }
            }),
            &mut bounds,
            ty::List::empty(),
            only_self_bounds,
        );
        debug!(?bounds);

        bounds
    }

    /// Given an HIR binding like `Item = Foo` or `Item: Foo`, pushes the corresponding predicates
    /// onto `bounds`.
    ///
    /// **A note on binders:** given something like `T: for<'a> Iterator<Item = &'a u32>`, the
    /// `trait_ref` here will be `for<'a> T: Iterator`. The `binding` data however is from *inside*
    /// the binder (e.g., `&'a u32`) and hence may reference bound regions.
    #[instrument(level = "debug", skip(self, bounds, speculative, dup_bindings, path_span))]
    pub(super) fn add_predicates_for_ast_type_binding(
        &self,
        hir_ref_id: hir::HirId,
        trait_ref: ty::PolyTraitRef<'tcx>,
        binding: &ConvertedBinding<'_, 'tcx>,
        bounds: &mut Bounds<'tcx>,
        speculative: bool,
        dup_bindings: &mut FxHashMap<DefId, Span>,
        path_span: Span,
        only_self_bounds: OnlySelfBounds,
    ) -> Result<(), ErrorGuaranteed> {
        // Given something like `U: SomeTrait<T = X>`, we want to produce a
        // predicate like `<U as SomeTrait>::T = X`. This is somewhat
        // subtle in the event that `T` is defined in a supertrait of
        // `SomeTrait`, because in that case we need to upcast.
        //
        // That is, consider this case:
        //
        // ```
        // trait SubTrait: SuperTrait<i32> { }
        // trait SuperTrait<A> { type T; }
        //
        // ... B: SubTrait<T = foo> ...
        // ```
        //
        // We want to produce `<B as SuperTrait<i32>>::T == foo`.

        let tcx = self.tcx();

        let assoc_kind =
            if binding.gen_args.parenthesized == hir::GenericArgsParentheses::ReturnTypeNotation {
                ty::AssocKind::Fn
            } else if let ConvertedBindingKind::Equality(term) = binding.kind
                && let ty::TermKind::Const(_) = term.node.unpack()
            {
                ty::AssocKind::Const
            } else {
                ty::AssocKind::Type
            };

        let candidate = if self.trait_defines_associated_item_named(
            trait_ref.def_id(),
            assoc_kind,
            binding.item_name,
        ) {
            // Simple case: The assoc item is defined in the current trait.
            trait_ref
        } else {
            // Otherwise, we have to walk through the supertraits to find
            // one that does define it.
            self.one_bound_for_assoc_item(
                || traits::supertraits(tcx, trait_ref),
                trait_ref.skip_binder().print_only_trait_name(),
                None,
                assoc_kind,
                binding.item_name,
                path_span,
                Some(&binding),
            )?
        };

        let (assoc_ident, def_scope) =
            tcx.adjust_ident_and_get_scope(binding.item_name, candidate.def_id(), hir_ref_id);

        // We have already adjusted the item name above, so compare with `.normalize_to_macros_2_0()`
        // instead of calling `filter_by_name_and_kind` which would needlessly normalize the
        // `assoc_ident` again and again.
        let assoc_item = tcx
            .associated_items(candidate.def_id())
            .filter_by_name_unhygienic(assoc_ident.name)
            .find(|i| i.kind == assoc_kind && i.ident(tcx).normalize_to_macros_2_0() == assoc_ident)
            .expect("missing associated item");

        if !assoc_item.visibility(tcx).is_accessible_from(def_scope, tcx) {
            let reported = tcx
                .dcx()
                .struct_span_err(
                    binding.span,
                    format!("{} `{}` is private", assoc_item.kind, binding.item_name),
                )
                .with_span_label(binding.span, format!("private {}", assoc_item.kind))
                .emit();
            self.set_tainted_by_errors(reported);
        }
        tcx.check_stability(assoc_item.def_id, Some(hir_ref_id), binding.span, None);

        if !speculative {
            dup_bindings
                .entry(assoc_item.def_id)
                .and_modify(|prev_span| {
                    tcx.dcx().emit_err(errors::ValueOfAssociatedStructAlreadySpecified {
                        span: binding.span,
                        prev_span: *prev_span,
                        item_name: binding.item_name,
                        def_path: tcx.def_path_str(assoc_item.container_id(tcx)),
                    });
                })
                .or_insert(binding.span);
        }

        let projection_ty = if let ty::AssocKind::Fn = assoc_kind {
            let mut emitted_bad_param_err = None;
            // If we have an method return type bound, then we need to substitute
            // the method's early bound params with suitable late-bound params.
            let mut num_bound_vars = candidate.bound_vars().len();
            let args =
                candidate.skip_binder().args.extend_to(tcx, assoc_item.def_id, |param, _| {
                    let subst = match param.kind {
                        ty::GenericParamDefKind::Lifetime => ty::Region::new_bound(
                            tcx,
                            ty::INNERMOST,
                            ty::BoundRegion {
                                var: ty::BoundVar::from_usize(num_bound_vars),
                                kind: ty::BoundRegionKind::BrNamed(param.def_id, param.name),
                            },
                        )
                        .into(),
                        ty::GenericParamDefKind::Type { .. } => {
                            let guar = *emitted_bad_param_err.get_or_insert_with(|| {
                                tcx.dcx().emit_err(
                                    crate::errors::ReturnTypeNotationIllegalParam::Type {
                                        span: path_span,
                                        param_span: tcx.def_span(param.def_id),
                                    },
                                )
                            });
                            Ty::new_error(tcx, guar).into()
                        }
                        ty::GenericParamDefKind::Const { .. } => {
                            let guar = *emitted_bad_param_err.get_or_insert_with(|| {
                                tcx.dcx().emit_err(
                                    crate::errors::ReturnTypeNotationIllegalParam::Const {
                                        span: path_span,
                                        param_span: tcx.def_span(param.def_id),
                                    },
                                )
                            });
                            let ty = tcx
                                .type_of(param.def_id)
                                .no_bound_vars()
                                .expect("ct params cannot have early bound vars");
                            ty::Const::new_error(tcx, guar, ty).into()
                        }
                    };
                    num_bound_vars += 1;
                    subst
                });

            // Next, we need to check that the return-type notation is being used on
            // an RPITIT (return-position impl trait in trait) or AFIT (async fn in trait).
            let output = tcx.fn_sig(assoc_item.def_id).skip_binder().output();
            let output = if let ty::Alias(ty::Projection, alias_ty) = *output.skip_binder().kind()
                && tcx.is_impl_trait_in_trait(alias_ty.def_id)
            {
                alias_ty
            } else {
                return Err(self.tcx().dcx().emit_err(
                    crate::errors::ReturnTypeNotationOnNonRpitit {
                        span: binding.span,
                        ty: tcx.liberate_late_bound_regions(assoc_item.def_id, output),
                        fn_span: tcx.hir().span_if_local(assoc_item.def_id),
                        note: (),
                    },
                ));
            };

            // Finally, move the fn return type's bound vars over to account for the early bound
            // params (and trait ref's late bound params). This logic is very similar to
            // `Predicate::subst_supertrait`, and it's no coincidence why.
            let shifted_output = tcx.shift_bound_var_indices(num_bound_vars, output);
            let subst_output = ty::EarlyBinder::bind(shifted_output).instantiate(tcx, args);

            let bound_vars = tcx.late_bound_vars(binding.hir_id);
            ty::Binder::bind_with_vars(subst_output, bound_vars)
        } else {
            // Append the generic arguments of the associated type to the `trait_ref`.
            candidate.map_bound(|trait_ref| {
                let ident = Ident::new(assoc_item.name, binding.item_name.span);
                let item_segment = hir::PathSegment {
                    ident,
                    hir_id: binding.hir_id,
                    res: Res::Err,
                    args: Some(binding.gen_args),
                    infer_args: false,
                };

                let args_trait_ref_and_assoc_item = self.create_args_for_associated_item(
                    path_span,
                    assoc_item.def_id,
                    &item_segment,
                    trait_ref.args,
                );

                debug!(?args_trait_ref_and_assoc_item);

                ty::AliasTy::new(tcx, assoc_item.def_id, args_trait_ref_and_assoc_item)
            })
        };

        if !speculative {
            // Find any late-bound regions declared in `ty` that are not
            // declared in the trait-ref or assoc_item. These are not well-formed.
            //
            // Example:
            //
            //     for<'a> <T as Iterator>::Item = &'a str // <-- 'a is bad
            //     for<'a> <T as FnMut<(&'a u32,)>>::Output = &'a str // <-- 'a is ok
            if let ConvertedBindingKind::Equality(ty) = binding.kind {
                let late_bound_in_trait_ref =
                    tcx.collect_constrained_late_bound_regions(&projection_ty);
                let late_bound_in_ty =
                    tcx.collect_referenced_late_bound_regions(&trait_ref.rebind(ty.node));
                debug!(?late_bound_in_trait_ref);
                debug!(?late_bound_in_ty);

                // FIXME: point at the type params that don't have appropriate lifetimes:
                // struct S1<F: for<'a> Fn(&i32, &i32) -> &'a i32>(F);
                //                         ----  ----     ^^^^^^^
                self.validate_late_bound_regions(
                    late_bound_in_trait_ref,
                    late_bound_in_ty,
                    |br_name| {
                        struct_span_code_err!(
                            tcx.dcx(),
                            binding.span,
                            E0582,
                            "binding for associated type `{}` references {}, \
                             which does not appear in the trait input types",
                            binding.item_name,
                            br_name
                        )
                    },
                );
            }
        }

        match binding.kind {
            ConvertedBindingKind::Equality(..) if let ty::AssocKind::Fn = assoc_kind => {
                return Err(self.tcx().dcx().emit_err(
                    crate::errors::ReturnTypeNotationEqualityBound { span: binding.span },
                ));
            }
            ConvertedBindingKind::Equality(term) => {
                // "Desugar" a constraint like `T: Iterator<Item = u32>` this to
                // the "projection predicate" for:
                //
                // `<T as Iterator>::Item = u32`
                bounds.push_projection_bound(
                    tcx,
                    projection_ty.map_bound(|projection_ty| ty::ProjectionPredicate {
                        projection_ty,
                        term: term.node,
                    }),
                    binding.span,
                );
            }
            ConvertedBindingKind::Constraint(ast_bounds) => {
                // "Desugar" a constraint like `T: Iterator<Item: Debug>` to
                //
                // `<T as Iterator>::Item: Debug`
                //
                // Calling `skip_binder` is okay, because `add_bounds` expects the `param_ty`
                // parameter to have a skipped binder.
                //
                // NOTE: If `only_self_bounds` is true, do NOT expand this associated
                // type bound into a trait predicate, since we only want to add predicates
                // for the `Self` type.
                if !only_self_bounds.0 {
                    let param_ty = Ty::new_alias(tcx, ty::Projection, projection_ty.skip_binder());
                    self.add_bounds(
                        param_ty,
                        ast_bounds.iter(),
                        bounds,
                        projection_ty.bound_vars(),
                        only_self_bounds,
                    );
                }
            }
        }
        Ok(())
    }
}
