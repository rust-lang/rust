use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_hir as hir;
use rustc_infer::traits::util;
use rustc_middle::ty::{
    self, GenericArgs, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt,
    Upcast, shift_vars,
};
use rustc_middle::{bug, span_bug};
use rustc_span::Span;
use rustc_span::def_id::{DefId, LocalDefId};
use tracing::{debug, instrument};

use super::ItemCtxt;
use super::predicates_of::assert_only_contains_predicates_from;
use crate::hir_ty_lowering::{HirTyLowerer, PredicateFilter};

/// For associated types we include both bounds written on the type
/// (`type X: Trait`) and predicates from the trait: `where Self::X: Trait`.
///
/// Note that this filtering is done with the items identity args to
/// simplify checking that these bounds are met in impls. This means that
/// a bound such as `for<'b> <Self as X<'b>>::U: Clone` can't be used, as in
/// `hr-associated-type-bound-1.rs`.
fn associated_type_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    assoc_item_def_id: LocalDefId,
    hir_bounds: &'tcx [hir::GenericBound<'tcx>],
    span: Span,
    filter: PredicateFilter,
) -> &'tcx [(ty::Clause<'tcx>, Span)] {
    ty::print::with_reduced_queries!({
        let item_ty = Ty::new_projection_from_args(
            tcx,
            assoc_item_def_id.to_def_id(),
            GenericArgs::identity_for_item(tcx, assoc_item_def_id),
        );

        let icx = ItemCtxt::new(tcx, assoc_item_def_id);
        let mut bounds = Vec::new();
        icx.lowerer().lower_bounds(item_ty, hir_bounds, &mut bounds, ty::List::empty(), filter);
        // Implicit bounds are added to associated types unless a `?Trait` bound is found
        match filter {
            PredicateFilter::All
            | PredicateFilter::SelfOnly
            | PredicateFilter::SelfTraitThatDefines(_)
            | PredicateFilter::SelfAndAssociatedTypeBounds => {
                icx.lowerer().add_default_traits(&mut bounds, item_ty, hir_bounds, None, span);
            }
            // `ConstIfConst` is only interested in `~const` bounds.
            PredicateFilter::ConstIfConst | PredicateFilter::SelfConstIfConst => {}
        }

        let trait_def_id = tcx.local_parent(assoc_item_def_id);
        let trait_predicates = tcx.trait_explicit_predicates_and_bounds(trait_def_id);

        let item_trait_ref = ty::TraitRef::identity(tcx, tcx.parent(assoc_item_def_id.to_def_id()));
        let bounds_from_parent =
            trait_predicates.predicates.iter().copied().filter_map(|(clause, span)| {
                remap_gat_vars_and_recurse_into_nested_projections(
                    tcx,
                    filter,
                    item_trait_ref,
                    assoc_item_def_id,
                    span,
                    clause,
                )
            });

        let all_bounds = tcx.arena.alloc_from_iter(bounds.into_iter().chain(bounds_from_parent));
        debug!(
            "associated_type_bounds({}) = {:?}",
            tcx.def_path_str(assoc_item_def_id.to_def_id()),
            all_bounds
        );

        assert_only_contains_predicates_from(filter, all_bounds, item_ty);

        all_bounds
    })
}

/// The code below is quite involved, so let me explain.
///
/// We loop here, because we also want to collect vars for nested associated items as
/// well. For example, given a clause like `Self::A::B`, we want to add that to the
/// item bounds for `A`, so that we may use that bound in the case that `Self::A::B` is
/// rigid.
///
/// Secondly, regarding bound vars, when we see a where clause that mentions a GAT
/// like `for<'a, ...> Self::Assoc<'a, ...>: Bound<'b, ...>`, we want to turn that into
/// an item bound on the GAT, where all of the GAT args are substituted with the GAT's
/// param regions, and then keep all of the other late-bound vars in the bound around.
/// We need to "compress" the binder so that it doesn't mention any of those vars that
/// were mapped to params.
fn remap_gat_vars_and_recurse_into_nested_projections<'tcx>(
    tcx: TyCtxt<'tcx>,
    filter: PredicateFilter,
    item_trait_ref: ty::TraitRef<'tcx>,
    assoc_item_def_id: LocalDefId,
    span: Span,
    clause: ty::Clause<'tcx>,
) -> Option<(ty::Clause<'tcx>, Span)> {
    let mut clause_ty = match clause.kind().skip_binder() {
        ty::ClauseKind::Trait(tr) => tr.self_ty(),
        ty::ClauseKind::Projection(proj) => proj.projection_term.self_ty(),
        ty::ClauseKind::TypeOutlives(outlives) => outlives.0,
        _ => return None,
    };

    let gat_vars = loop {
        if let ty::Alias(ty::Projection, alias_ty) = *clause_ty.kind() {
            if alias_ty.trait_ref(tcx) == item_trait_ref
                && alias_ty.def_id == assoc_item_def_id.to_def_id()
            {
                // We have found the GAT in question...
                // Return the vars, since we may need to remap them.
                break &alias_ty.args[item_trait_ref.args.len()..];
            } else {
                // Only collect *self* type bounds if the filter is for self.
                match filter {
                    PredicateFilter::All => {}
                    PredicateFilter::SelfOnly => {
                        return None;
                    }
                    PredicateFilter::SelfTraitThatDefines(_)
                    | PredicateFilter::SelfConstIfConst
                    | PredicateFilter::SelfAndAssociatedTypeBounds
                    | PredicateFilter::ConstIfConst => {
                        unreachable!(
                            "invalid predicate filter for \
                            `remap_gat_vars_and_recurse_into_nested_projections`"
                        )
                    }
                }

                clause_ty = alias_ty.self_ty();
                continue;
            }
        }

        return None;
    };

    // Special-case: No GAT vars, no mapping needed.
    if gat_vars.is_empty() {
        return Some((clause, span));
    }

    // First, check that all of the GAT args are substituted with a unique late-bound arg.
    // If we find a duplicate, then it can't be mapped to the definition's params.
    let mut mapping = FxIndexMap::default();
    let generics = tcx.generics_of(assoc_item_def_id);
    for (param, var) in std::iter::zip(&generics.own_params, gat_vars) {
        let existing = match var.unpack() {
            ty::GenericArgKind::Lifetime(re) => {
                if let ty::RegionKind::ReBound(ty::INNERMOST, bv) = re.kind() {
                    mapping.insert(bv.var, tcx.mk_param_from_def(param))
                } else {
                    return None;
                }
            }
            ty::GenericArgKind::Type(ty) => {
                if let ty::Bound(ty::INNERMOST, bv) = *ty.kind() {
                    mapping.insert(bv.var, tcx.mk_param_from_def(param))
                } else {
                    return None;
                }
            }
            ty::GenericArgKind::Const(ct) => {
                if let ty::ConstKind::Bound(ty::INNERMOST, bv) = ct.kind() {
                    mapping.insert(bv, tcx.mk_param_from_def(param))
                } else {
                    return None;
                }
            }
        };

        if existing.is_some() {
            return None;
        }
    }

    // Finally, map all of the args in the GAT to the params we expect, and compress
    // the remaining late-bound vars so that they count up from var 0.
    let mut folder =
        MapAndCompressBoundVars { tcx, binder: ty::INNERMOST, still_bound_vars: vec![], mapping };
    let pred = clause.kind().skip_binder().fold_with(&mut folder);

    Some((
        ty::Binder::bind_with_vars(pred, tcx.mk_bound_variable_kinds(&folder.still_bound_vars))
            .upcast(tcx),
        span,
    ))
}

/// Given some where clause like `for<'b, 'c> <Self as Trait<'a_identity>>::Gat<'b>: Bound<'c>`,
/// the mapping will map `'b` back to the GAT's `'b_identity`. Then we need to compress the
/// remaining bound var `'c` to index 0.
///
/// This folder gives us: `for<'c> <Self as Trait<'a_identity>>::Gat<'b_identity>: Bound<'c>`,
/// which is sufficient for an item bound for `Gat`, since all of the GAT's args are identity.
struct MapAndCompressBoundVars<'tcx> {
    tcx: TyCtxt<'tcx>,
    /// How deep are we? Makes sure we don't touch the vars of nested binders.
    binder: ty::DebruijnIndex,
    /// List of bound vars that remain unsubstituted because they were not
    /// mentioned in the GAT's args.
    still_bound_vars: Vec<ty::BoundVariableKind>,
    /// Subtle invariant: If the `GenericArg` is bound, then it should be
    /// stored with the debruijn index of `INNERMOST` so it can be shifted
    /// correctly during substitution.
    mapping: FxIndexMap<ty::BoundVar, ty::GenericArg<'tcx>>,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for MapAndCompressBoundVars<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_binder<T>(&mut self, t: ty::Binder<'tcx, T>) -> ty::Binder<'tcx, T>
    where
        ty::Binder<'tcx, T>: TypeSuperFoldable<TyCtxt<'tcx>>,
    {
        self.binder.shift_in(1);
        let out = t.super_fold_with(self);
        self.binder.shift_out(1);
        out
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if !ty.has_bound_vars() {
            return ty;
        }

        if let ty::Bound(binder, old_bound) = *ty.kind()
            && self.binder == binder
        {
            let mapped = if let Some(mapped) = self.mapping.get(&old_bound.var) {
                mapped.expect_ty()
            } else {
                // If we didn't find a mapped generic, then make a new one.
                // Allocate a new var idx, and insert a new bound ty.
                let var = ty::BoundVar::from_usize(self.still_bound_vars.len());
                self.still_bound_vars.push(ty::BoundVariableKind::Ty(old_bound.kind));
                let mapped = Ty::new_bound(
                    self.tcx,
                    ty::INNERMOST,
                    ty::BoundTy { var, kind: old_bound.kind },
                );
                self.mapping.insert(old_bound.var, mapped.into());
                mapped
            };

            shift_vars(self.tcx, mapped, self.binder.as_u32())
        } else {
            ty.super_fold_with(self)
        }
    }

    fn fold_region(&mut self, re: ty::Region<'tcx>) -> ty::Region<'tcx> {
        if let ty::ReBound(binder, old_bound) = re.kind()
            && self.binder == binder
        {
            let mapped = if let Some(mapped) = self.mapping.get(&old_bound.var) {
                mapped.expect_region()
            } else {
                let var = ty::BoundVar::from_usize(self.still_bound_vars.len());
                self.still_bound_vars.push(ty::BoundVariableKind::Region(old_bound.kind));
                let mapped = ty::Region::new_bound(
                    self.tcx,
                    ty::INNERMOST,
                    ty::BoundRegion { var, kind: old_bound.kind },
                );
                self.mapping.insert(old_bound.var, mapped.into());
                mapped
            };

            shift_vars(self.tcx, mapped, self.binder.as_u32())
        } else {
            re
        }
    }

    fn fold_const(&mut self, ct: ty::Const<'tcx>) -> ty::Const<'tcx> {
        if !ct.has_bound_vars() {
            return ct;
        }

        if let ty::ConstKind::Bound(binder, old_var) = ct.kind()
            && self.binder == binder
        {
            let mapped = if let Some(mapped) = self.mapping.get(&old_var) {
                mapped.expect_const()
            } else {
                let var = ty::BoundVar::from_usize(self.still_bound_vars.len());
                self.still_bound_vars.push(ty::BoundVariableKind::Const);
                let mapped = ty::Const::new_bound(self.tcx, ty::INNERMOST, var);
                self.mapping.insert(old_var, mapped.into());
                mapped
            };

            shift_vars(self.tcx, mapped, self.binder.as_u32())
        } else {
            ct.super_fold_with(self)
        }
    }

    fn fold_predicate(&mut self, p: ty::Predicate<'tcx>) -> ty::Predicate<'tcx> {
        if !p.has_bound_vars() { p } else { p.super_fold_with(self) }
    }
}

/// Opaque types don't inherit bounds from their parent: for return position
/// impl trait it isn't possible to write a suitable predicate on the
/// containing function and for type-alias impl trait we don't have a backwards
/// compatibility issue.
#[instrument(level = "trace", skip(tcx, item_ty))]
fn opaque_type_bounds<'tcx>(
    tcx: TyCtxt<'tcx>,
    opaque_def_id: LocalDefId,
    hir_bounds: &'tcx [hir::GenericBound<'tcx>],
    item_ty: Ty<'tcx>,
    span: Span,
    filter: PredicateFilter,
) -> &'tcx [(ty::Clause<'tcx>, Span)] {
    ty::print::with_reduced_queries!({
        let icx = ItemCtxt::new(tcx, opaque_def_id);
        let mut bounds = Vec::new();
        icx.lowerer().lower_bounds(item_ty, hir_bounds, &mut bounds, ty::List::empty(), filter);
        // Implicit bounds are added to opaque types unless a `?Trait` bound is found
        match filter {
            PredicateFilter::All
            | PredicateFilter::SelfOnly
            | PredicateFilter::SelfTraitThatDefines(_)
            | PredicateFilter::SelfAndAssociatedTypeBounds => {
                icx.lowerer().add_default_traits(&mut bounds, item_ty, hir_bounds, None, span);
            }
            //`ConstIfConst` is only interested in `~const` bounds.
            PredicateFilter::ConstIfConst | PredicateFilter::SelfConstIfConst => {}
        }
        debug!(?bounds);

        tcx.arena.alloc_slice(&bounds)
    })
}

pub(super) fn explicit_item_bounds(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
) -> ty::EarlyBinder<'_, &'_ [(ty::Clause<'_>, Span)]> {
    explicit_item_bounds_with_filter(tcx, def_id, PredicateFilter::All)
}

pub(super) fn explicit_item_self_bounds(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
) -> ty::EarlyBinder<'_, &'_ [(ty::Clause<'_>, Span)]> {
    explicit_item_bounds_with_filter(tcx, def_id, PredicateFilter::SelfOnly)
}

pub(super) fn explicit_item_bounds_with_filter(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
    filter: PredicateFilter,
) -> ty::EarlyBinder<'_, &'_ [(ty::Clause<'_>, Span)]> {
    match tcx.opt_rpitit_info(def_id.to_def_id()) {
        // RPITIT's bounds are the same as opaque type bounds, but with
        // a projection self type.
        Some(ty::ImplTraitInTraitData::Trait { opaque_def_id, .. }) => {
            let opaque_ty = tcx.hir_node_by_def_id(opaque_def_id.expect_local()).expect_opaque_ty();
            let bounds =
                associated_type_bounds(tcx, def_id, opaque_ty.bounds, opaque_ty.span, filter);
            return ty::EarlyBinder::bind(bounds);
        }
        Some(ty::ImplTraitInTraitData::Impl { .. }) => {
            span_bug!(tcx.def_span(def_id), "RPITIT in impl should not have item bounds")
        }
        None => {}
    }

    let bounds = match tcx.hir_node_by_def_id(def_id) {
        hir::Node::TraitItem(hir::TraitItem {
            kind: hir::TraitItemKind::Type(bounds, _),
            span,
            ..
        }) => associated_type_bounds(tcx, def_id, bounds, *span, filter),
        hir::Node::OpaqueTy(hir::OpaqueTy { bounds, origin, span, .. }) => match origin {
            // Since RPITITs are lowered as projections in `<dyn HirTyLowerer>::lower_ty`,
            // when we're asking for the item bounds of the *opaques* in a trait's default
            // method signature, we need to map these projections back to opaques.
            rustc_hir::OpaqueTyOrigin::FnReturn {
                parent,
                in_trait_or_impl: Some(hir::RpitContext::Trait),
            }
            | rustc_hir::OpaqueTyOrigin::AsyncFn {
                parent,
                in_trait_or_impl: Some(hir::RpitContext::Trait),
            } => {
                let args = GenericArgs::identity_for_item(tcx, def_id);
                let item_ty = Ty::new_opaque(tcx, def_id.to_def_id(), args);
                let bounds = &*tcx.arena.alloc_slice(
                    &opaque_type_bounds(tcx, def_id, bounds, item_ty, *span, filter)
                        .to_vec()
                        .fold_with(&mut AssocTyToOpaque { tcx, fn_def_id: parent.to_def_id() }),
                );
                assert_only_contains_predicates_from(filter, bounds, item_ty);
                bounds
            }
            rustc_hir::OpaqueTyOrigin::FnReturn {
                parent: _,
                in_trait_or_impl: None | Some(hir::RpitContext::TraitImpl),
            }
            | rustc_hir::OpaqueTyOrigin::AsyncFn {
                parent: _,
                in_trait_or_impl: None | Some(hir::RpitContext::TraitImpl),
            }
            | rustc_hir::OpaqueTyOrigin::TyAlias { parent: _, .. } => {
                let args = GenericArgs::identity_for_item(tcx, def_id);
                let item_ty = Ty::new_opaque(tcx, def_id.to_def_id(), args);
                let bounds = opaque_type_bounds(tcx, def_id, bounds, item_ty, *span, filter);
                assert_only_contains_predicates_from(filter, bounds, item_ty);
                bounds
            }
        },
        hir::Node::Item(hir::Item { kind: hir::ItemKind::TyAlias(..), .. }) => &[],
        node => bug!("item_bounds called on {def_id:?} => {node:?}"),
    };

    ty::EarlyBinder::bind(bounds)
}

pub(super) fn item_bounds(tcx: TyCtxt<'_>, def_id: DefId) -> ty::EarlyBinder<'_, ty::Clauses<'_>> {
    tcx.explicit_item_bounds(def_id).map_bound(|bounds| {
        tcx.mk_clauses_from_iter(util::elaborate(tcx, bounds.iter().map(|&(bound, _span)| bound)))
    })
}

pub(super) fn item_self_bounds(
    tcx: TyCtxt<'_>,
    def_id: DefId,
) -> ty::EarlyBinder<'_, ty::Clauses<'_>> {
    tcx.explicit_item_self_bounds(def_id).map_bound(|bounds| {
        tcx.mk_clauses_from_iter(
            util::elaborate(tcx, bounds.iter().map(|&(bound, _span)| bound)).filter_only_self(),
        )
    })
}

/// This exists as an optimization to compute only the item bounds of the item
/// that are not `Self` bounds.
pub(super) fn item_non_self_bounds(
    tcx: TyCtxt<'_>,
    def_id: DefId,
) -> ty::EarlyBinder<'_, ty::Clauses<'_>> {
    let all_bounds: FxIndexSet<_> = tcx.item_bounds(def_id).skip_binder().iter().collect();
    let own_bounds: FxIndexSet<_> = tcx.item_self_bounds(def_id).skip_binder().iter().collect();
    if all_bounds.len() == own_bounds.len() {
        ty::EarlyBinder::bind(ty::ListWithCachedTypeInfo::empty())
    } else {
        ty::EarlyBinder::bind(tcx.mk_clauses_from_iter(all_bounds.difference(&own_bounds).copied()))
    }
}

/// This exists as an optimization to compute only the supertraits of this impl's
/// trait that are outlives bounds.
pub(super) fn impl_super_outlives(
    tcx: TyCtxt<'_>,
    def_id: DefId,
) -> ty::EarlyBinder<'_, ty::Clauses<'_>> {
    tcx.impl_trait_header(def_id).expect("expected an impl of trait").trait_ref.map_bound(
        |trait_ref| {
            let clause: ty::Clause<'_> = trait_ref.upcast(tcx);
            tcx.mk_clauses_from_iter(util::elaborate(tcx, [clause]).filter(|clause| {
                matches!(
                    clause.kind().skip_binder(),
                    ty::ClauseKind::TypeOutlives(_) | ty::ClauseKind::RegionOutlives(_)
                )
            }))
        },
    )
}

struct AssocTyToOpaque<'tcx> {
    tcx: TyCtxt<'tcx>,
    fn_def_id: DefId,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for AssocTyToOpaque<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        if let ty::Alias(ty::Projection, projection_ty) = ty.kind()
            && let Some(ty::ImplTraitInTraitData::Trait { fn_def_id, .. }) =
                self.tcx.opt_rpitit_info(projection_ty.def_id)
            && fn_def_id == self.fn_def_id
        {
            self.tcx.type_of(projection_ty.def_id).instantiate(self.tcx, projection_ty.args)
        } else {
            ty.super_fold_with(self)
        }
    }
}
