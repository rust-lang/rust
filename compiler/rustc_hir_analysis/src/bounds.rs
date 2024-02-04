//! Bounds are restrictions applied to some types after they've been converted into the
//! `ty` form from the HIR.

use rustc_hir::LangItem;
use rustc_middle::ty::{self, ToPredicate, Ty, TyCtxt};
use rustc_span::{def_id::DefId, Span};

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
        defining_def_id: DefId,
        trait_ref: ty::PolyTraitRef<'tcx>,
        span: Span,
        polarity: ty::ImplPolarity,
        constness: ty::BoundConstness,
    ) {
        self.clauses.push((
            trait_ref
                .map_bound(|trait_ref| {
                    ty::ClauseKind::Trait(ty::TraitPredicate { trait_ref, polarity })
                })
                .to_predicate(tcx),
            span,
        ));
        // For `T: ~const Tr` or `T: const Tr`, we need to add an additional bound on the
        // associated type of `<T as Tr>` and make sure that the effect is compatible.
        if let Some(compat_val) = match constness {
            // TODO: do we need `T: const Trait` anymore?
            ty::BoundConstness::Const => Some(tcx.consts.false_),
            ty::BoundConstness::ConstIfConst => {
                Some(tcx.expected_host_effect_param_for_body(defining_def_id))
            }
            ty::BoundConstness::NotConst => None,
        } {
            // create a new projection type `<T as Tr>::Effects`
            let assoc = tcx.associated_type_for_effects(trait_ref.def_id()).unwrap();
            let self_ty = Ty::new_projection(tcx, assoc, trait_ref.skip_binder().args);
            // make `<T as Tr>::Effects: Compat<runtime>`
            let new_trait_ref = ty::TraitRef::new(
                tcx,
                tcx.require_lang_item(LangItem::EffectsCompat, Some(span)),
                [ty::GenericArg::from(self_ty), compat_val.into()],
            );
            self.clauses.push((trait_ref.rebind(new_trait_ref).to_predicate(tcx), span));
        }
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
