use std::ops::ControlFlow;

use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_errors::codes::*;
use rustc_errors::struct_span_code_err;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::{AmbigArg, LangItem, PolyTraitRef};
use rustc_middle::bug;
use rustc_middle::ty::{
    self as ty, IsSuggestable, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitableExt,
    TypeVisitor, Upcast,
};
use rustc_span::{ErrorGuaranteed, Ident, Span, Symbol, kw};
use rustc_trait_selection::traits;
use smallvec::SmallVec;
use tracing::{debug, instrument};

use super::errors::GenericsArgsErrExtend;
use crate::errors;
use crate::hir_ty_lowering::{
    AssocItemQSelf, FeedConstTy, HirTyLowerer, PredicateFilter, RegionInferReason,
};

#[derive(Debug, Default)]
struct CollectedBound {
    /// `Trait`
    positive: bool,
    /// `?Trait`
    maybe: bool,
    /// `!Trait`
    negative: bool,
}

impl CollectedBound {
    /// Returns `true` if any of `Trait`, `?Trait` or `!Trait` were encountered.
    fn any(&self) -> bool {
        self.positive || self.maybe || self.negative
    }
}

#[derive(Debug)]
struct CollectedSizednessBounds {
    // Collected `Sized` bounds
    sized: CollectedBound,
    // Collected `MetaSized` bounds
    meta_sized: CollectedBound,
    // Collected `PointeeSized` bounds
    pointee_sized: CollectedBound,
}

impl CollectedSizednessBounds {
    /// Returns `true` if any of `Trait`, `?Trait` or `!Trait` were encountered for `Sized`,
    /// `MetaSized` or `PointeeSized`.
    fn any(&self) -> bool {
        self.sized.any() || self.meta_sized.any() || self.pointee_sized.any()
    }
}

fn search_bounds_for<'tcx>(
    hir_bounds: &'tcx [hir::GenericBound<'tcx>],
    self_ty_where_predicates: Option<(LocalDefId, &'tcx [hir::WherePredicate<'tcx>])>,
    mut f: impl FnMut(&'tcx PolyTraitRef<'tcx>),
) {
    let mut search_bounds = |hir_bounds: &'tcx [hir::GenericBound<'tcx>]| {
        for hir_bound in hir_bounds {
            let hir::GenericBound::Trait(ptr) = hir_bound else {
                continue;
            };

            f(ptr)
        }
    };

    search_bounds(hir_bounds);
    if let Some((self_ty, where_clause)) = self_ty_where_predicates {
        for clause in where_clause {
            if let hir::WherePredicateKind::BoundPredicate(pred) = clause.kind
                && pred.is_param_bound(self_ty.to_def_id())
            {
                search_bounds(pred.bounds);
            }
        }
    }
}

fn collect_unbounds<'tcx>(
    hir_bounds: &'tcx [hir::GenericBound<'tcx>],
    self_ty_where_predicates: Option<(LocalDefId, &'tcx [hir::WherePredicate<'tcx>])>,
) -> SmallVec<[&'tcx PolyTraitRef<'tcx>; 1]> {
    let mut unbounds: SmallVec<[_; 1]> = SmallVec::new();
    search_bounds_for(hir_bounds, self_ty_where_predicates, |ptr| {
        if matches!(ptr.modifiers.polarity, hir::BoundPolarity::Maybe(_)) {
            unbounds.push(ptr);
        }
    });
    unbounds
}

fn collect_bounds<'a, 'tcx>(
    hir_bounds: &'a [hir::GenericBound<'tcx>],
    self_ty_where_predicates: Option<(LocalDefId, &'tcx [hir::WherePredicate<'tcx>])>,
    target_did: DefId,
) -> CollectedBound {
    let mut collect_into = CollectedBound::default();
    search_bounds_for(hir_bounds, self_ty_where_predicates, |ptr| {
        if !matches!(ptr.trait_ref.path.res, Res::Def(DefKind::Trait, did) if did == target_did) {
            return;
        }

        match ptr.modifiers.polarity {
            hir::BoundPolarity::Maybe(_) => collect_into.maybe = true,
            hir::BoundPolarity::Negative(_) => collect_into.negative = true,
            hir::BoundPolarity::Positive => collect_into.positive = true,
        }
    });
    collect_into
}

fn collect_sizedness_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    hir_bounds: &'tcx [hir::GenericBound<'tcx>],
    self_ty_where_predicates: Option<(LocalDefId, &'tcx [hir::WherePredicate<'tcx>])>,
    span: Span,
) -> CollectedSizednessBounds {
    let sized_did = tcx.require_lang_item(LangItem::Sized, span);
    let sized = collect_bounds(hir_bounds, self_ty_where_predicates, sized_did);

    let meta_sized_did = tcx.require_lang_item(LangItem::MetaSized, span);
    let meta_sized = collect_bounds(hir_bounds, self_ty_where_predicates, meta_sized_did);

    let pointee_sized_did = tcx.require_lang_item(LangItem::PointeeSized, span);
    let pointee_sized = collect_bounds(hir_bounds, self_ty_where_predicates, pointee_sized_did);

    CollectedSizednessBounds { sized, meta_sized, pointee_sized }
}

/// Add a trait bound for `did`.
fn add_trait_bound<'tcx>(
    tcx: TyCtxt<'tcx>,
    bounds: &mut Vec<(ty::Clause<'tcx>, Span)>,
    self_ty: Ty<'tcx>,
    did: DefId,
    span: Span,
) {
    let trait_ref = ty::TraitRef::new(tcx, did, [self_ty]);
    // Preferable to put sizedness obligations first, since we report better errors for `Sized`
    // ambiguity.
    bounds.insert(0, (trait_ref.upcast(tcx), span));
}

impl<'tcx> dyn HirTyLowerer<'tcx> + '_ {
    /// Skip `PointeeSized` bounds.
    ///
    /// `PointeeSized` is a "fake bound" insofar as anywhere a `PointeeSized` bound exists, there
    /// is actually the absence of any bounds. This avoids limitations around non-global where
    /// clauses being preferred over item bounds (where `PointeeSized` bounds would be
    /// proven) - which can result in errors when a `PointeeSized` supertrait/bound/predicate is
    /// added to some items.
    pub(crate) fn should_skip_sizedness_bound<'hir>(
        &self,
        bound: &'hir hir::GenericBound<'tcx>,
    ) -> bool {
        bound
            .trait_ref()
            .and_then(|tr| tr.trait_def_id())
            .map(|did| self.tcx().is_lang_item(did, LangItem::PointeeSized))
            .unwrap_or(false)
    }

    /// Adds sizedness bounds to a trait, trait alias, parameter, opaque type or associated type.
    ///
    /// - On parameters, opaque type and associated types, add default `Sized` bound if no explicit
    ///   sizedness bounds are present.
    /// - On traits and trait aliases, add default `MetaSized` supertrait if no explicit sizedness
    ///   bounds are present.
    /// - On parameters, opaque type, associated types and trait aliases, add a `MetaSized` bound if
    ///   a `?Sized` bound is present.
    pub(crate) fn add_sizedness_bounds(
        &self,
        bounds: &mut Vec<(ty::Clause<'tcx>, Span)>,
        self_ty: Ty<'tcx>,
        hir_bounds: &'tcx [hir::GenericBound<'tcx>],
        self_ty_where_predicates: Option<(LocalDefId, &'tcx [hir::WherePredicate<'tcx>])>,
        trait_did: Option<LocalDefId>,
        span: Span,
    ) {
        let tcx = self.tcx();

        let meta_sized_did = tcx.require_lang_item(LangItem::MetaSized, span);
        let pointee_sized_did = tcx.require_lang_item(LangItem::PointeeSized, span);

        // If adding sizedness bounds to a trait, then there are some relevant early exits
        if let Some(trait_did) = trait_did {
            let trait_did = trait_did.to_def_id();
            // Never add a default supertrait to `PointeeSized`.
            if trait_did == pointee_sized_did {
                return;
            }
            // Don't add default sizedness supertraits to auto traits because it isn't possible to
            // relax an automatically added supertrait on the defn itself.
            if tcx.trait_is_auto(trait_did) {
                return;
            }
        } else {
            // Report invalid unbounds on sizedness-bounded generic parameters.
            let unbounds = collect_unbounds(hir_bounds, self_ty_where_predicates);
            self.check_and_report_invalid_unbounds_on_param(unbounds);
        }

        let collected = collect_sizedness_bounds(tcx, hir_bounds, self_ty_where_predicates, span);
        if (collected.sized.maybe || collected.sized.negative)
            && !collected.sized.positive
            && !collected.meta_sized.any()
            && !collected.pointee_sized.any()
        {
            // `?Sized` is equivalent to `MetaSized` (but only add the bound if there aren't any
            // other explicit ones) - this can happen for trait aliases as well as bounds.
            add_trait_bound(tcx, bounds, self_ty, meta_sized_did, span);
        } else if !collected.any() {
            if trait_did.is_some() {
                // If there are no explicit sizedness bounds on a trait then add a default
                // `MetaSized` supertrait.
                add_trait_bound(tcx, bounds, self_ty, meta_sized_did, span);
            } else {
                // If there are no explicit sizedness bounds on a parameter then add a default
                // `Sized` bound.
                let sized_did = tcx.require_lang_item(LangItem::Sized, span);
                add_trait_bound(tcx, bounds, self_ty, sized_did, span);
            }
        }
    }

    /// Checks whether `Self: DefaultAutoTrait` bounds should be added on trait super bounds
    /// or associated items.
    ///
    /// To keep backward compatibility with existing code, `experimental_default_bounds` bounds
    /// should be added everywhere, including super bounds. However this causes a huge performance
    /// costs. For optimization purposes instead of adding default supertraits, bounds
    /// are added to the associated items:
    ///
    /// ```ignore(illustrative)
    /// // Default bounds are generated in the following way:
    /// trait Trait {
    ///     fn foo(&self) where Self: Leak {}
    /// }
    ///
    /// // instead of this:
    /// trait Trait: Leak {
    ///     fn foo(&self) {}
    /// }
    /// ```
    /// It is not always possible to do this because of backward compatibility:
    ///
    /// ```ignore(illustrative)
    /// pub trait Trait<Rhs = Self> {}
    /// pub trait Trait1 : Trait {}
    /// //~^ ERROR: `Rhs` requires `DefaultAutoTrait`, but `Self` is not `DefaultAutoTrait`
    /// ```
    ///
    /// or:
    ///
    /// ```ignore(illustrative)
    /// trait Trait {
    ///     type Type where Self: Sized;
    /// }
    /// trait Trait2<T> : Trait<Type = T> {}
    /// //~^ ERROR: `DefaultAutoTrait` required for `Trait2`, by implicit  `Self: DefaultAutoTrait` in `Trait::Type`
    /// ```
    ///
    /// Therefore, `experimental_default_bounds` are still being added to supertraits if
    /// the `SelfTyParam` or `AssocItemConstraint` were found in a trait header.
    fn requires_default_supertraits(
        &self,
        hir_bounds: &'tcx [hir::GenericBound<'tcx>],
        hir_generics: &'tcx hir::Generics<'tcx>,
    ) -> bool {
        struct TraitInfoCollector;

        impl<'tcx> hir::intravisit::Visitor<'tcx> for TraitInfoCollector {
            type Result = ControlFlow<()>;

            fn visit_assoc_item_constraint(
                &mut self,
                _constraint: &'tcx hir::AssocItemConstraint<'tcx>,
            ) -> Self::Result {
                ControlFlow::Break(())
            }

            fn visit_ty(&mut self, t: &'tcx hir::Ty<'tcx, AmbigArg>) -> Self::Result {
                if matches!(
                    &t.kind,
                    hir::TyKind::Path(hir::QPath::Resolved(
                        _,
                        hir::Path { res: hir::def::Res::SelfTyParam { .. }, .. },
                    ))
                ) {
                    return ControlFlow::Break(());
                }
                hir::intravisit::walk_ty(self, t)
            }
        }

        let mut found = false;
        for bound in hir_bounds {
            found |= hir::intravisit::walk_param_bound(&mut TraitInfoCollector, bound).is_break();
        }
        found |= hir::intravisit::walk_generics(&mut TraitInfoCollector, hir_generics).is_break();
        found
    }

    /// Implicitly add `Self: DefaultAutoTrait` clauses on trait associated items if
    /// they are not added as super trait bounds to the trait itself. See
    /// `requires_default_supertraits` for more information.
    pub(crate) fn add_default_trait_item_bounds(
        &self,
        trait_item: &hir::TraitItem<'tcx>,
        bounds: &mut Vec<(ty::Clause<'tcx>, Span)>,
    ) {
        let tcx = self.tcx();
        if !tcx.sess.opts.unstable_opts.experimental_default_bounds {
            return;
        }

        let parent = tcx.local_parent(trait_item.hir_id().owner.def_id);
        let hir::Node::Item(parent_trait) = tcx.hir_node_by_def_id(parent) else {
            unreachable!();
        };

        let (trait_generics, trait_bounds) = match parent_trait.kind {
            hir::ItemKind::Trait(_, _, _, generics, supertraits, _) => (generics, supertraits),
            hir::ItemKind::TraitAlias(_, generics, supertraits) => (generics, supertraits),
            _ => unreachable!(),
        };

        if !self.requires_default_supertraits(trait_bounds, trait_generics) {
            let self_ty_where_predicates = (parent, trait_item.generics.predicates);
            self.add_default_traits(
                bounds,
                tcx.types.self_param,
                &[],
                Some(self_ty_where_predicates),
                trait_item.span,
            );
        }
    }

    /// Lazily sets `experimental_default_bounds` to true on trait super bounds.
    /// See `requires_default_supertraits` for more information.
    pub(crate) fn add_default_super_traits(
        &self,
        trait_def_id: LocalDefId,
        bounds: &mut Vec<(ty::Clause<'tcx>, Span)>,
        hir_bounds: &'tcx [hir::GenericBound<'tcx>],
        hir_generics: &'tcx hir::Generics<'tcx>,
        span: Span,
    ) {
        if !self.tcx().sess.opts.unstable_opts.experimental_default_bounds {
            return;
        }

        assert!(matches!(self.tcx().def_kind(trait_def_id), DefKind::Trait | DefKind::TraitAlias));
        if self.requires_default_supertraits(hir_bounds, hir_generics) {
            let self_ty_where_predicates = (trait_def_id, hir_generics.predicates);
            self.add_default_traits(
                bounds,
                self.tcx().types.self_param,
                hir_bounds,
                Some(self_ty_where_predicates),
                span,
            );
        }
    }

    pub(crate) fn add_default_traits(
        &self,
        bounds: &mut Vec<(ty::Clause<'tcx>, Span)>,
        self_ty: Ty<'tcx>,
        hir_bounds: &[hir::GenericBound<'tcx>],
        self_ty_where_predicates: Option<(LocalDefId, &'tcx [hir::WherePredicate<'tcx>])>,
        span: Span,
    ) {
        self.tcx().default_traits().iter().for_each(|default_trait| {
            self.add_default_trait(
                *default_trait,
                bounds,
                self_ty,
                hir_bounds,
                self_ty_where_predicates,
                span,
            );
        });
    }

    /// Add a `experimental_default_bounds` bound to the `bounds` if appropriate.
    ///
    /// Doesn't add the bound if the HIR bounds contain any of `Trait`, `?Trait` or `!Trait`.
    pub(crate) fn add_default_trait(
        &self,
        trait_: hir::LangItem,
        bounds: &mut Vec<(ty::Clause<'tcx>, Span)>,
        self_ty: Ty<'tcx>,
        hir_bounds: &[hir::GenericBound<'tcx>],
        self_ty_where_predicates: Option<(LocalDefId, &'tcx [hir::WherePredicate<'tcx>])>,
        span: Span,
    ) {
        let tcx = self.tcx();
        let trait_id = tcx.lang_items().get(trait_);
        if let Some(trait_id) = trait_id
            && self.do_not_provide_default_trait_bound(
                trait_id,
                hir_bounds,
                self_ty_where_predicates,
            )
        {
            add_trait_bound(tcx, bounds, self_ty, trait_id, span);
        }
    }

    fn do_not_provide_default_trait_bound<'a>(
        &self,
        trait_def_id: DefId,
        hir_bounds: &'a [hir::GenericBound<'tcx>],
        self_ty_where_predicates: Option<(LocalDefId, &'tcx [hir::WherePredicate<'tcx>])>,
    ) -> bool {
        let collected = collect_bounds(hir_bounds, self_ty_where_predicates, trait_def_id);
        !collected.any()
    }

    /// Lower HIR bounds into `bounds` given the self type `param_ty` and the overarching late-bound vars if any.
    ///
    /// ### Examples
    ///
    /// ```ignore (illustrative)
    /// fn foo<T>() where for<'a> T: Trait<'a> + Copy {}
    /// //                ^^^^^^^ ^  ^^^^^^^^^^^^^^^^ `hir_bounds`, in HIR form
    /// //                |       |
    /// //                |       `param_ty`, in ty form
    /// //                `bound_vars`, in ty form
    ///
    /// fn bar<T>() where T: for<'a> Trait<'a> + Copy {} // no overarching `bound_vars` here!
    /// //                ^  ^^^^^^^^^^^^^^^^^^^^^^^^ `hir_bounds`, in HIR form
    /// //                |
    /// //                `param_ty`, in ty form
    /// ```
    ///
    /// ### A Note on Binders
    ///
    /// There is an implied binder around `param_ty` and `hir_bounds`.
    /// See `lower_poly_trait_ref` for more details.
    #[instrument(level = "debug", skip(self, hir_bounds, bounds))]
    pub(crate) fn lower_bounds<'hir, I: IntoIterator<Item = &'hir hir::GenericBound<'tcx>>>(
        &self,
        param_ty: Ty<'tcx>,
        hir_bounds: I,
        bounds: &mut Vec<(ty::Clause<'tcx>, Span)>,
        bound_vars: &'tcx ty::List<ty::BoundVariableKind>,
        predicate_filter: PredicateFilter,
    ) where
        'tcx: 'hir,
    {
        for hir_bound in hir_bounds {
            if self.should_skip_sizedness_bound(hir_bound) {
                continue;
            }

            // In order to avoid cycles, when we're lowering `SelfTraitThatDefines`,
            // we skip over any traits that don't define the given associated type.
            if let PredicateFilter::SelfTraitThatDefines(assoc_ident) = predicate_filter {
                if let Some(trait_ref) = hir_bound.trait_ref()
                    && let Some(trait_did) = trait_ref.trait_def_id()
                    && self.tcx().trait_may_define_assoc_item(trait_did, assoc_ident)
                {
                    // Okay
                } else {
                    continue;
                }
            }

            match hir_bound {
                hir::GenericBound::Trait(poly_trait_ref) => {
                    let hir::TraitBoundModifiers { constness, polarity } = poly_trait_ref.modifiers;
                    let _ = self.lower_poly_trait_ref(
                        &poly_trait_ref.trait_ref,
                        poly_trait_ref.span,
                        constness,
                        polarity,
                        param_ty,
                        bounds,
                        predicate_filter,
                    );
                }
                hir::GenericBound::Outlives(lifetime) => {
                    // `ConstIfConst` is only interested in `~const` bounds.
                    if matches!(
                        predicate_filter,
                        PredicateFilter::ConstIfConst | PredicateFilter::SelfConstIfConst
                    ) {
                        continue;
                    }

                    let region = self.lower_lifetime(lifetime, RegionInferReason::OutlivesBound);
                    let bound = ty::Binder::bind_with_vars(
                        ty::ClauseKind::TypeOutlives(ty::OutlivesPredicate(param_ty, region)),
                        bound_vars,
                    );
                    bounds.push((bound.upcast(self.tcx()), lifetime.ident.span));
                }
                hir::GenericBound::Use(..) => {
                    // We don't actually lower `use` into the type layer.
                }
            }
        }
    }

    /// Lower an associated item constraint from the HIR into `bounds`.
    ///
    /// ### A Note on Binders
    ///
    /// Given something like `T: for<'a> Iterator<Item = &'a u32>`,
    /// the `trait_ref` here will be `for<'a> T: Iterator`.
    /// The `constraint` data however is from *inside* the binder
    /// (e.g., `&'a u32`) and hence may reference bound regions.
    #[instrument(level = "debug", skip(self, bounds, duplicates, path_span))]
    pub(super) fn lower_assoc_item_constraint(
        &self,
        hir_ref_id: hir::HirId,
        trait_ref: ty::PolyTraitRef<'tcx>,
        constraint: &hir::AssocItemConstraint<'tcx>,
        bounds: &mut Vec<(ty::Clause<'tcx>, Span)>,
        duplicates: &mut FxIndexMap<DefId, Span>,
        path_span: Span,
        predicate_filter: PredicateFilter,
    ) -> Result<(), ErrorGuaranteed> {
        let tcx = self.tcx();

        let assoc_tag = if constraint.gen_args.parenthesized
            == hir::GenericArgsParentheses::ReturnTypeNotation
        {
            ty::AssocTag::Fn
        } else if let hir::AssocItemConstraintKind::Equality { term: hir::Term::Const(_) } =
            constraint.kind
        {
            ty::AssocTag::Const
        } else {
            ty::AssocTag::Type
        };

        // Given something like `U: Trait<T = X>`, we want to produce a predicate like
        // `<U as Trait>::T = X`.
        // This is somewhat subtle in the event that `T` is defined in a supertrait of `Trait`,
        // because in that case we need to upcast. I.e., we want to produce
        // `<B as SuperTrait<i32>>::T == X` for `B: SubTrait<T = X>` where
        //
        //     trait SubTrait: SuperTrait<i32> {}
        //     trait SuperTrait<A> { type T; }
        let candidate = if self.probe_trait_that_defines_assoc_item(
            trait_ref.def_id(),
            assoc_tag,
            constraint.ident,
        ) {
            // Simple case: The assoc item is defined in the current trait.
            trait_ref
        } else {
            // Otherwise, we have to walk through the supertraits to find
            // one that does define it.
            self.probe_single_bound_for_assoc_item(
                || traits::supertraits(tcx, trait_ref),
                AssocItemQSelf::Trait(trait_ref.def_id()),
                assoc_tag,
                constraint.ident,
                path_span,
                Some(constraint),
            )?
        };

        let assoc_item = self
            .probe_assoc_item(
                constraint.ident,
                assoc_tag,
                hir_ref_id,
                constraint.span,
                candidate.def_id(),
            )
            .expect("failed to find associated item");

        duplicates
            .entry(assoc_item.def_id)
            .and_modify(|prev_span| {
                self.dcx().emit_err(errors::ValueOfAssociatedStructAlreadySpecified {
                    span: constraint.span,
                    prev_span: *prev_span,
                    item_name: constraint.ident,
                    def_path: tcx.def_path_str(assoc_item.container_id(tcx)),
                });
            })
            .or_insert(constraint.span);

        let projection_term = if let ty::AssocTag::Fn = assoc_tag {
            let bound_vars = tcx.late_bound_vars(constraint.hir_id);
            ty::Binder::bind_with_vars(
                self.lower_return_type_notation_ty(candidate, assoc_item.def_id, path_span)?.into(),
                bound_vars,
            )
        } else {
            // Create the generic arguments for the associated type or constant by joining the
            // parent arguments (the arguments of the trait) and the own arguments (the ones of
            // the associated item itself) and construct an alias type using them.
            let alias_term = candidate.map_bound(|trait_ref| {
                let item_segment = hir::PathSegment {
                    ident: constraint.ident,
                    hir_id: constraint.hir_id,
                    res: Res::Err,
                    args: Some(constraint.gen_args),
                    infer_args: false,
                };

                let alias_args = self.lower_generic_args_of_assoc_item(
                    path_span,
                    assoc_item.def_id,
                    &item_segment,
                    trait_ref.args,
                );
                debug!(?alias_args);

                ty::AliasTerm::new_from_args(tcx, assoc_item.def_id, alias_args)
            });

            // Provide the resolved type of the associated constant to `type_of(AnonConst)`.
            if let Some(const_arg) = constraint.ct() {
                if let hir::ConstArgKind::Anon(anon_const) = const_arg.kind {
                    let ty = alias_term
                        .map_bound(|alias| tcx.type_of(alias.def_id).instantiate(tcx, alias.args));
                    let ty = check_assoc_const_binding_type(
                        self,
                        constraint.ident,
                        ty,
                        constraint.hir_id,
                    );
                    tcx.feed_anon_const_type(anon_const.def_id, ty::EarlyBinder::bind(ty));
                }
            }

            alias_term
        };

        match constraint.kind {
            hir::AssocItemConstraintKind::Equality { .. } if let ty::AssocTag::Fn = assoc_tag => {
                return Err(self.dcx().emit_err(crate::errors::ReturnTypeNotationEqualityBound {
                    span: constraint.span,
                }));
            }
            // Lower an equality constraint like `Item = u32` as found in HIR bound `T: Iterator<Item = u32>`
            // to a projection predicate: `<T as Iterator>::Item = u32`.
            hir::AssocItemConstraintKind::Equality { term } => {
                let term = match term {
                    hir::Term::Ty(ty) => self.lower_ty(ty).into(),
                    hir::Term::Const(ct) => self.lower_const_arg(ct, FeedConstTy::No).into(),
                };

                // Find any late-bound regions declared in `ty` that are not
                // declared in the trait-ref or assoc_item. These are not well-formed.
                //
                // Example:
                //
                //     for<'a> <T as Iterator>::Item = &'a str // <-- 'a is bad
                //     for<'a> <T as FnMut<(&'a u32,)>>::Output = &'a str // <-- 'a is ok
                let late_bound_in_projection_ty =
                    tcx.collect_constrained_late_bound_regions(projection_term);
                let late_bound_in_term =
                    tcx.collect_referenced_late_bound_regions(trait_ref.rebind(term));
                debug!(?late_bound_in_projection_ty);
                debug!(?late_bound_in_term);

                // FIXME: point at the type params that don't have appropriate lifetimes:
                // struct S1<F: for<'a> Fn(&i32, &i32) -> &'a i32>(F);
                //                         ----  ----     ^^^^^^^
                // NOTE(associated_const_equality): This error should be impossible to trigger
                //                                  with associated const equality constraints.
                self.validate_late_bound_regions(
                    late_bound_in_projection_ty,
                    late_bound_in_term,
                    |br_name| {
                        struct_span_code_err!(
                            self.dcx(),
                            constraint.span,
                            E0582,
                            "binding for associated type `{}` references {}, \
                             which does not appear in the trait input types",
                            constraint.ident,
                            br_name
                        )
                    },
                );

                match predicate_filter {
                    PredicateFilter::All
                    | PredicateFilter::SelfOnly
                    | PredicateFilter::SelfAndAssociatedTypeBounds => {
                        let bound = projection_term.map_bound(|projection_term| {
                            ty::ClauseKind::Projection(ty::ProjectionPredicate {
                                projection_term,
                                term,
                            })
                        });
                        bounds.push((bound.upcast(tcx), constraint.span));
                    }
                    // SelfTraitThatDefines is only interested in trait predicates.
                    PredicateFilter::SelfTraitThatDefines(_) => {}
                    // `ConstIfConst` is only interested in `~const` bounds.
                    PredicateFilter::ConstIfConst | PredicateFilter::SelfConstIfConst => {}
                }
            }
            // Lower a constraint like `Item: Debug` as found in HIR bound `T: Iterator<Item: Debug>`
            // to a bound involving a projection: `<T as Iterator>::Item: Debug`.
            hir::AssocItemConstraintKind::Bound { bounds: hir_bounds } => {
                match predicate_filter {
                    PredicateFilter::All
                    | PredicateFilter::SelfAndAssociatedTypeBounds
                    | PredicateFilter::ConstIfConst => {
                        let projection_ty = projection_term
                            .map_bound(|projection_term| projection_term.expect_ty(self.tcx()));
                        // Calling `skip_binder` is okay, because `lower_bounds` expects the `param_ty`
                        // parameter to have a skipped binder.
                        let param_ty =
                            Ty::new_alias(tcx, ty::Projection, projection_ty.skip_binder());
                        self.lower_bounds(
                            param_ty,
                            hir_bounds,
                            bounds,
                            projection_ty.bound_vars(),
                            predicate_filter,
                        );
                    }
                    PredicateFilter::SelfOnly
                    | PredicateFilter::SelfTraitThatDefines(_)
                    | PredicateFilter::SelfConstIfConst => {}
                }
            }
        }
        Ok(())
    }

    /// Lower a type, possibly specially handling the type if it's a return type notation
    /// which we otherwise deny in other positions.
    pub fn lower_ty_maybe_return_type_notation(&self, hir_ty: &hir::Ty<'tcx>) -> Ty<'tcx> {
        let hir::TyKind::Path(qpath) = hir_ty.kind else {
            return self.lower_ty(hir_ty);
        };

        let tcx = self.tcx();
        match qpath {
            hir::QPath::Resolved(opt_self_ty, path)
                if let [mod_segments @ .., trait_segment, item_segment] = &path.segments[..]
                    && item_segment.args.is_some_and(|args| {
                        matches!(
                            args.parenthesized,
                            hir::GenericArgsParentheses::ReturnTypeNotation
                        )
                    }) =>
            {
                // We don't allow generics on the module segments.
                let _ =
                    self.prohibit_generic_args(mod_segments.iter(), GenericsArgsErrExtend::None);

                let item_def_id = match path.res {
                    Res::Def(DefKind::AssocFn, item_def_id) => item_def_id,
                    Res::Err => {
                        return Ty::new_error_with_message(
                            tcx,
                            hir_ty.span,
                            "failed to resolve RTN",
                        );
                    }
                    _ => bug!("only expected method resolution for fully qualified RTN"),
                };
                let trait_def_id = tcx.parent(item_def_id);

                // Good error for `where Trait::method(..): Send`.
                let Some(self_ty) = opt_self_ty else {
                    let guar = self.report_missing_self_ty_for_resolved_path(
                        trait_def_id,
                        hir_ty.span,
                        item_segment,
                        ty::AssocTag::Type,
                    );
                    return Ty::new_error(tcx, guar);
                };
                let self_ty = self.lower_ty(self_ty);

                let trait_ref = self.lower_mono_trait_ref(
                    hir_ty.span,
                    trait_def_id,
                    self_ty,
                    trait_segment,
                    false,
                );

                // SUBTLE: As noted at the end of `try_append_return_type_notation_params`
                // in `resolve_bound_vars`, we stash the explicit bound vars of the where
                // clause onto the item segment of the RTN type. This allows us to know
                // how many bound vars are *not* coming from the signature of the function
                // from lowering RTN itself.
                //
                // For example, in `where for<'a> <T as Trait<'a>>::method(..): Other`,
                // the `late_bound_vars` of the where clause predicate (i.e. this HIR ty's
                // parent) will include `'a` AND all the early- and late-bound vars of the
                // method. But when lowering the RTN type, we just want the list of vars
                // we used to resolve the trait ref. We explicitly stored those back onto
                // the item segment, since there's no other good place to put them.
                let candidate =
                    ty::Binder::bind_with_vars(trait_ref, tcx.late_bound_vars(item_segment.hir_id));

                match self.lower_return_type_notation_ty(candidate, item_def_id, hir_ty.span) {
                    Ok(ty) => Ty::new_alias(tcx, ty::Projection, ty),
                    Err(guar) => Ty::new_error(tcx, guar),
                }
            }
            hir::QPath::TypeRelative(hir_self_ty, segment)
                if segment.args.is_some_and(|args| {
                    matches!(args.parenthesized, hir::GenericArgsParentheses::ReturnTypeNotation)
                }) =>
            {
                let self_ty = self.lower_ty(hir_self_ty);
                let (item_def_id, bound) = match self.resolve_type_relative_path(
                    self_ty,
                    hir_self_ty,
                    ty::AssocTag::Fn,
                    segment,
                    hir_ty.hir_id,
                    hir_ty.span,
                    None,
                ) {
                    Ok(result) => result,
                    Err(guar) => return Ty::new_error(tcx, guar),
                };

                // Don't let `T::method` resolve to some `for<'a> <T as Tr<'a>>::method`,
                // which may happen via a higher-ranked where clause or supertrait.
                // This is the same restrictions as associated types; even though we could
                // support it, it just makes things a lot more difficult to support in
                // `resolve_bound_vars`, since we'd need to introduce those as elided
                // bound vars on the where clause too.
                if bound.has_bound_vars() {
                    return Ty::new_error(
                        tcx,
                        self.dcx().emit_err(errors::AssociatedItemTraitUninferredGenericParams {
                            span: hir_ty.span,
                            inferred_sugg: Some(hir_ty.span.with_hi(segment.ident.span.lo())),
                            bound: format!("{}::", tcx.anonymize_bound_vars(bound).skip_binder()),
                            mpart_sugg: None,
                            what: tcx.def_descr(item_def_id),
                        }),
                    );
                }

                match self.lower_return_type_notation_ty(bound, item_def_id, hir_ty.span) {
                    Ok(ty) => Ty::new_alias(tcx, ty::Projection, ty),
                    Err(guar) => Ty::new_error(tcx, guar),
                }
            }
            _ => self.lower_ty(hir_ty),
        }
    }

    /// Do the common parts of lowering an RTN type. This involves extending the
    /// candidate binder to include all of the early- and late-bound vars that are
    /// defined on the function itself, and constructing a projection to the RPITIT
    /// return type of that function.
    fn lower_return_type_notation_ty(
        &self,
        candidate: ty::PolyTraitRef<'tcx>,
        item_def_id: DefId,
        path_span: Span,
    ) -> Result<ty::AliasTy<'tcx>, ErrorGuaranteed> {
        let tcx = self.tcx();
        let mut emitted_bad_param_err = None;
        // If we have an method return type bound, then we need to instantiate
        // the method's early bound params with suitable late-bound params.
        let mut num_bound_vars = candidate.bound_vars().len();
        let args = candidate.skip_binder().args.extend_to(tcx, item_def_id, |param, _| {
            let arg = match param.kind {
                ty::GenericParamDefKind::Lifetime => ty::Region::new_bound(
                    tcx,
                    ty::INNERMOST,
                    ty::BoundRegion {
                        var: ty::BoundVar::from_usize(num_bound_vars),
                        kind: ty::BoundRegionKind::Named(param.def_id, param.name),
                    },
                )
                .into(),
                ty::GenericParamDefKind::Type { .. } => {
                    let guar = *emitted_bad_param_err.get_or_insert_with(|| {
                        self.dcx().emit_err(crate::errors::ReturnTypeNotationIllegalParam::Type {
                            span: path_span,
                            param_span: tcx.def_span(param.def_id),
                        })
                    });
                    Ty::new_error(tcx, guar).into()
                }
                ty::GenericParamDefKind::Const { .. } => {
                    let guar = *emitted_bad_param_err.get_or_insert_with(|| {
                        self.dcx().emit_err(crate::errors::ReturnTypeNotationIllegalParam::Const {
                            span: path_span,
                            param_span: tcx.def_span(param.def_id),
                        })
                    });
                    ty::Const::new_error(tcx, guar).into()
                }
            };
            num_bound_vars += 1;
            arg
        });

        // Next, we need to check that the return-type notation is being used on
        // an RPITIT (return-position impl trait in trait) or AFIT (async fn in trait).
        let output = tcx.fn_sig(item_def_id).skip_binder().output();
        let output = if let ty::Alias(ty::Projection, alias_ty) = *output.skip_binder().kind()
            && tcx.is_impl_trait_in_trait(alias_ty.def_id)
        {
            alias_ty
        } else {
            return Err(self.dcx().emit_err(crate::errors::ReturnTypeNotationOnNonRpitit {
                span: path_span,
                ty: tcx.liberate_late_bound_regions(item_def_id, output),
                fn_span: tcx.hir_span_if_local(item_def_id),
                note: (),
            }));
        };

        // Finally, move the fn return type's bound vars over to account for the early bound
        // params (and trait ref's late bound params). This logic is very similar to
        // `rustc_middle::ty::predicate::Clause::instantiate_supertrait`
        // and it's no coincidence why.
        let shifted_output = tcx.shift_bound_var_indices(num_bound_vars, output);
        Ok(ty::EarlyBinder::bind(shifted_output).instantiate(tcx, args))
    }
}

/// Detect and reject early-bound & escaping late-bound generic params in the type of assoc const bindings.
///
/// FIXME(const_generics): This is a temporary and semi-artificial restriction until the
/// arrival of *generic const generics*[^1].
///
/// It might actually be possible that we can already support early-bound generic params
/// in such types if we just lifted some more checks in other places, too, for example
/// inside `HirTyLowerer::lower_anon_const`. However, even if that were the case, we should
/// probably gate this behind another feature flag.
///
/// [^1]: <https://github.com/rust-lang/project-const-generics/issues/28>.
fn check_assoc_const_binding_type<'tcx>(
    cx: &dyn HirTyLowerer<'tcx>,
    assoc_const: Ident,
    ty: ty::Binder<'tcx, Ty<'tcx>>,
    hir_id: hir::HirId,
) -> Ty<'tcx> {
    // We can't perform the checks for early-bound params during name resolution unlike E0770
    // because this information depends on *type* resolution.
    // We can't perform these checks in `resolve_bound_vars` either for the same reason.
    // Consider the trait ref `for<'a> Trait<'a, C = { &0 }>`. We need to know the fully
    // resolved type of `Trait::C` in order to know if it references `'a` or not.

    let ty = ty.skip_binder();
    if !ty.has_param() && !ty.has_escaping_bound_vars() {
        return ty;
    }

    let mut collector = GenericParamAndBoundVarCollector {
        cx,
        params: Default::default(),
        vars: Default::default(),
        depth: ty::INNERMOST,
    };
    let mut guar = ty.visit_with(&mut collector).break_value();

    let tcx = cx.tcx();
    let ty_note = ty
        .make_suggestable(tcx, false, None)
        .map(|ty| crate::errors::TyOfAssocConstBindingNote { assoc_const, ty });

    let enclosing_item_owner_id = tcx
        .hir_parent_owner_iter(hir_id)
        .find_map(|(owner_id, parent)| parent.generics().map(|_| owner_id))
        .unwrap();
    let generics = tcx.generics_of(enclosing_item_owner_id);
    for index in collector.params {
        let param = generics.param_at(index as _, tcx);
        let is_self_param = param.name == kw::SelfUpper;
        guar.get_or_insert(cx.dcx().emit_err(crate::errors::ParamInTyOfAssocConstBinding {
            span: assoc_const.span,
            assoc_const,
            param_name: param.name,
            param_def_kind: tcx.def_descr(param.def_id),
            param_category: if is_self_param {
                "self"
            } else if param.kind.is_synthetic() {
                "synthetic"
            } else {
                "normal"
            },
            param_defined_here_label:
                (!is_self_param).then(|| tcx.def_ident_span(param.def_id).unwrap()),
            ty_note,
        }));
    }
    for (var_def_id, var_name) in collector.vars {
        guar.get_or_insert(cx.dcx().emit_err(
            crate::errors::EscapingBoundVarInTyOfAssocConstBinding {
                span: assoc_const.span,
                assoc_const,
                var_name,
                var_def_kind: tcx.def_descr(var_def_id),
                var_defined_here_label: tcx.def_ident_span(var_def_id).unwrap(),
                ty_note,
            },
        ));
    }

    let guar = guar.unwrap_or_else(|| bug!("failed to find gen params or bound vars in ty"));
    Ty::new_error(tcx, guar)
}

struct GenericParamAndBoundVarCollector<'a, 'tcx> {
    cx: &'a dyn HirTyLowerer<'tcx>,
    params: FxIndexSet<u32>,
    vars: FxIndexSet<(DefId, Symbol)>,
    depth: ty::DebruijnIndex,
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for GenericParamAndBoundVarCollector<'_, 'tcx> {
    type Result = ControlFlow<ErrorGuaranteed>;

    fn visit_binder<T: TypeVisitable<TyCtxt<'tcx>>>(
        &mut self,
        binder: &ty::Binder<'tcx, T>,
    ) -> Self::Result {
        self.depth.shift_in(1);
        let result = binder.super_visit_with(self);
        self.depth.shift_out(1);
        result
    }

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
        match ty.kind() {
            ty::Param(param) => {
                self.params.insert(param.index);
            }
            ty::Bound(db, bt) if *db >= self.depth => {
                self.vars.insert(match bt.kind {
                    ty::BoundTyKind::Param(def_id, name) => (def_id, name),
                    ty::BoundTyKind::Anon => {
                        let reported = self
                            .cx
                            .dcx()
                            .delayed_bug(format!("unexpected anon bound ty: {:?}", bt.var));
                        return ControlFlow::Break(reported);
                    }
                });
            }
            _ if ty.has_param() || ty.has_bound_vars() => return ty.super_visit_with(self),
            _ => {}
        }
        ControlFlow::Continue(())
    }

    fn visit_region(&mut self, re: ty::Region<'tcx>) -> Self::Result {
        match re.kind() {
            ty::ReEarlyParam(param) => {
                self.params.insert(param.index);
            }
            ty::ReBound(db, br) if db >= self.depth => {
                self.vars.insert(match br.kind {
                    ty::BoundRegionKind::Named(def_id, name) => (def_id, name),
                    ty::BoundRegionKind::Anon | ty::BoundRegionKind::ClosureEnv => {
                        let guar = self
                            .cx
                            .dcx()
                            .delayed_bug(format!("unexpected bound region kind: {:?}", br.kind));
                        return ControlFlow::Break(guar);
                    }
                });
            }
            _ => {}
        }
        ControlFlow::Continue(())
    }

    fn visit_const(&mut self, ct: ty::Const<'tcx>) -> Self::Result {
        match ct.kind() {
            ty::ConstKind::Param(param) => {
                self.params.insert(param.index);
            }
            ty::ConstKind::Bound(db, ty::BoundVar { .. }) if db >= self.depth => {
                let guar = self.cx.dcx().delayed_bug("unexpected escaping late-bound const var");
                return ControlFlow::Break(guar);
            }
            _ if ct.has_param() || ct.has_bound_vars() => return ct.super_visit_with(self),
            _ => {}
        }
        ControlFlow::Continue(())
    }
}
