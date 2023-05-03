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
    pub predicates: Vec<(ty::Predicate<'tcx>, Span)>,
}

impl<'tcx> Bounds<'tcx> {
    pub fn push_region_bound(
        &mut self,
        tcx: TyCtxt<'tcx>,
        region: ty::PolyTypeOutlivesPredicate<'tcx>,
        span: Span,
    ) {
        self.predicates.push((region.to_predicate(tcx), span));
    }

    pub fn push_trait_bound(
        &mut self,
        tcx: TyCtxt<'tcx>,
        trait_ref: ty::PolyTraitRef<'tcx>,
        span: Span,
        constness: ty::BoundConstness,
        polarity: ty::ImplPolarity,
    ) {
        self.predicates.push((
            trait_ref
                .map_bound(|trait_ref| ty::TraitPredicate { trait_ref, constness, polarity })
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
        self.predicates.push((projection.to_predicate(tcx), span));
    }

    pub fn push_sized(&mut self, tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, span: Span) {
        let sized_def_id = tcx.require_lang_item(LangItem::Sized, Some(span));
        let trait_ref = ty::Binder::dummy(tcx.mk_trait_ref(sized_def_id, [ty]));
        // Preferable to put this obligation first, since we report better errors for sized ambiguity.
        self.predicates.insert(0, (trait_ref.without_const().to_predicate(tcx), span));
    }

    pub fn predicates(&self) -> impl Iterator<Item = (ty::Predicate<'tcx>, Span)> + '_ {
        self.predicates.iter().cloned()
    }
}
