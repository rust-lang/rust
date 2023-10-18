//! Bounds are restrictions applied to some types after they've been converted into the
//! `ty` form from the HIR.

use rustc_hir::LangItem;
use rustc_middle::ty::{self, ToPredicate, Ty, TyCtxt};
use rustc_span::Span;

/// Collects together a list of type bounds. These lists of bounds occur in many places
/// in Rust's syntax:
///
/// ```text
/// trait Foo: Bar + Baz { }
///            ^^^^^^^^^ supertrait list bounding the `Self` type parameter
///
/// fn foo<T: Bar + Baz>() { }
///           ^^^^^^^^^ bounding the type parameter `T`
///
/// impl dyn Bar + Baz
///          ^^^^^^^^^ bounding the type-erased dynamic type
/// ```
///
/// Our representation is a bit mixed here -- in some cases, we
/// include the self type (e.g., `trait_bounds`) but in others we do not
#[derive(Default, PartialEq, Eq, Clone, Debug)]
pub struct Bounds<'tcx> {
    pub clauses: Vec<(ty::Clause<'tcx>, Span)>,
}

impl<'tcx> Bounds<'tcx> {
    pub fn push_region_bound(
        &mut self,
        tcx: TyCtxt<'tcx>,
        region: ty::PolyTypeOutlivesPredicate<'tcx>,
        span: Span,
    ) {
        self.clauses
            .push((region.map_bound(|p| ty::ClauseKind::TypeOutlives(p)).to_predicate(tcx), span));
    }

    pub fn push_trait_bound(
        &mut self,
        tcx: TyCtxt<'tcx>,
        trait_ref: ty::PolyTraitRef<'tcx>,
        span: Span,
        polarity: ty::ImplPolarity,
    ) {
        self.push_trait_bound_inner(tcx, trait_ref, span, polarity);

        // push a non-const (`host = true`) version of the bound if it is `~const`.
        if tcx.features().effects
            && let Some(host_effect_idx) = tcx.generics_of(trait_ref.def_id()).host_effect_index
            && trait_ref.skip_binder().args.const_at(host_effect_idx) != tcx.consts.true_
        {
            let generics = tcx.generics_of(trait_ref.def_id());
            let Some(host_index) = generics.host_effect_index else { return };
            let trait_ref = trait_ref.map_bound(|mut trait_ref| {
                trait_ref.args =
                    tcx.mk_args_from_iter(trait_ref.args.iter().enumerate().map(|(n, arg)| {
                        if host_index == n { tcx.consts.true_.into() } else { arg }
                    }));
                trait_ref
            });

            self.push_trait_bound_inner(tcx, trait_ref, span, polarity);
        }
    }

    fn push_trait_bound_inner(
        &mut self,
        tcx: TyCtxt<'tcx>,
        trait_ref: ty::PolyTraitRef<'tcx>,
        span: Span,
        polarity: ty::ImplPolarity,
    ) {
        self.clauses.push((
            trait_ref
                .map_bound(|trait_ref| {
                    ty::ClauseKind::Trait(ty::TraitPredicate { trait_ref, polarity })
                })
                .to_predicate(tcx),
            span,
        ));
    }

    pub fn push_projection_bound(
        &mut self,
        tcx: TyCtxt<'tcx>,
        projection: ty::PolyProjectionPredicate<'tcx>,
        span: Span,
    ) {
        self.clauses.push((
            projection.map_bound(|proj| ty::ClauseKind::Projection(proj)).to_predicate(tcx),
            span,
        ));
    }

    pub fn push_sized(&mut self, tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, span: Span) {
        let sized_def_id = tcx.require_lang_item(LangItem::Sized, Some(span));
        let trait_ref = ty::TraitRef::new(tcx, sized_def_id, [ty]);
        // Preferable to put this obligation first, since we report better errors for sized ambiguity.
        self.clauses.insert(0, (trait_ref.to_predicate(tcx), span));
    }

    pub fn clauses(&self) -> impl Iterator<Item = (ty::Clause<'tcx>, Span)> + '_ {
        self.clauses.iter().cloned()
    }
}
