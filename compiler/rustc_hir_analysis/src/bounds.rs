//! Bounds are restrictions applied to some types after they've been lowered from the HIR to the
//! [`rustc_middle::ty`] form.

use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::LangItem;
use rustc_middle::ty::{self, Ty, TyCtxt, Upcast};
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
pub(crate) struct Bounds<'tcx> {
    clauses: Vec<(ty::Clause<'tcx>, Span)>,
    effects_min_tys: FxIndexMap<Ty<'tcx>, Span>,
}

impl<'tcx> Bounds<'tcx> {
    pub(crate) fn push_region_bound(
        &mut self,
        tcx: TyCtxt<'tcx>,
        region: ty::PolyTypeOutlivesPredicate<'tcx>,
        span: Span,
    ) {
        self.clauses
            .push((region.map_bound(|p| ty::ClauseKind::TypeOutlives(p)).upcast(tcx), span));
    }

    pub(crate) fn push_trait_bound(
        &mut self,
        tcx: TyCtxt<'tcx>,
        bound_trait_ref: ty::PolyTraitRef<'tcx>,
        span: Span,
        polarity: ty::PredicatePolarity,
    ) {
        let clause = (
            bound_trait_ref
                .map_bound(|trait_ref| {
                    ty::ClauseKind::Trait(ty::TraitPredicate { trait_ref, polarity })
                })
                .upcast(tcx),
            span,
        );
        // FIXME(-Znext-solver): We can likely remove this hack once the new trait solver lands.
        if tcx.is_lang_item(bound_trait_ref.def_id(), LangItem::Sized) {
            self.clauses.insert(0, clause);
        } else {
            self.clauses.push(clause);
        }
    }

    pub(crate) fn push_projection_bound(
        &mut self,
        tcx: TyCtxt<'tcx>,
        projection: ty::PolyProjectionPredicate<'tcx>,
        span: Span,
    ) {
        self.clauses.push((
            projection.map_bound(|proj| ty::ClauseKind::Projection(proj)).upcast(tcx),
            span,
        ));
    }

    pub(crate) fn push_sized(&mut self, tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, span: Span) {
        let sized_def_id = tcx.require_lang_item(LangItem::Sized, Some(span));
        let trait_ref = ty::TraitRef::new(tcx, sized_def_id, [ty]);
        // Preferable to put this obligation first, since we report better errors for sized ambiguity.
        self.clauses.insert(0, (trait_ref.upcast(tcx), span));
    }

    /// Push a `const` or `~const` bound as a `HostEffect` predicate.
    pub(crate) fn push_const_bound(
        &mut self,
        tcx: TyCtxt<'tcx>,
        bound_trait_ref: ty::PolyTraitRef<'tcx>,
        host: ty::HostPolarity,
        span: Span,
    ) {
        if tcx.is_const_trait(bound_trait_ref.def_id()) {
            self.clauses.push((bound_trait_ref.to_host_effect_clause(tcx, host), span));
        } else {
            tcx.dcx().span_delayed_bug(span, "tried to lower {host:?} bound for non-const trait");
        }
    }

    pub(crate) fn clauses(
        &self,
        // FIXME(effects): remove tcx
        _tcx: TyCtxt<'tcx>,
    ) -> impl Iterator<Item = (ty::Clause<'tcx>, Span)> + '_ {
        self.clauses.iter().cloned()
    }

    pub(crate) fn effects_min_tys(&self) -> impl Iterator<Item = Ty<'tcx>> + '_ {
        self.effects_min_tys.keys().copied()
    }
}
