//! Bounds are restrictions applied to some types after they've been converted into the
//! `ty` form from the HIR.

use rustc_hir::{def::DefKind, LangItem};
use rustc_middle::ty::{self, ToPredicate, Ty, TyCtxt};
use rustc_span::sym;
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
        if let Some(compat_val) = match (tcx.def_kind(defining_def_id), constness) {
            // TODO: do we need `T: const Trait` anymore?
            (_, ty::BoundConstness::Const) => Some(tcx.consts.false_),
            // body owners that can have trait bounds
            (DefKind::Const | DefKind::Fn | DefKind::AssocFn, ty::BoundConstness::ConstIfConst) => {
                Some(tcx.expected_host_effect_param_for_body(defining_def_id))
            }

            (_, ty::BoundConstness::NotConst) => {
                tcx.has_attr(trait_ref.def_id(), sym::const_trait).then_some(tcx.consts.true_)
            }

            // if the defining_def_id is a trait, we wire it differently than others by equating the effects.
            (
                kind @ (DefKind::Trait | DefKind::Impl { of_trait: true } | DefKind::AssocTy),
                ty::BoundConstness::ConstIfConst,
            ) => {
                let parent_def_id = if kind == DefKind::AssocTy {
                    let did = tcx.parent(defining_def_id);
                    if !matches!(
                        tcx.def_kind(did),
                        DefKind::Trait | DefKind::Impl { of_trait: true }
                    ) {
                        tcx.dcx().span_delayed_bug(span, "invalid `~const` encountered");
                        return;
                    }
                    did
                } else {
                    defining_def_id
                };
                let trait_we_are_in = match tcx.def_kind(parent_def_id) {
                    DefKind::Trait => ty::TraitRef::identity(tcx, parent_def_id),
                    DefKind::Impl { of_trait: true } => {
                        tcx.impl_trait_ref(parent_def_id).unwrap().instantiate_identity()
                    }
                    _ => unreachable!(),
                };
                // create a new projection type `<T as TraitForBound>::Effects`
                let Some(assoc) = tcx.associated_type_for_effects(trait_ref.def_id()) else {
                    tcx.dcx().span_delayed_bug(
                        span,
                        "`~const` trait bound has no effect assoc yet no errors encountered?",
                    );
                    return;
                };
                let self_ty = Ty::new_projection(tcx, assoc, trait_ref.skip_binder().args);
                // we might have `~const Tr` where `Tr` isn't a `#[const_trait]`.
                let Some(assoc_def) = tcx.associated_type_for_effects(trait_we_are_in.def_id)
                else {
                    tcx.dcx().span_delayed_bug(
                        span,
                        "`~const` trait bound has no effect assoc yet no errors encountered?",
                    );
                    return;
                };
                let fx_ty_trait_we_are_in =
                    Ty::new_projection(tcx, assoc_def, trait_we_are_in.args);
                // make `<T as TraitForBound>::Effects: EffectsEq<<Self as TraitWeAreIn>::Effects>`
                let new_trait_ref = ty::TraitRef::new(
                    tcx,
                    tcx.require_lang_item(LangItem::EffectsEq, Some(span)),
                    [self_ty, fx_ty_trait_we_are_in],
                );
                self.clauses.push((trait_ref.rebind(new_trait_ref).to_predicate(tcx), span));
                return;
            }
            // probably illegal in this position.
            (_, ty::BoundConstness::ConstIfConst) => {
                tcx.dcx().span_delayed_bug(span, "invalid `~const` encountered");
                return;
            }
        } {
            // create a new projection type `<T as Tr>::Effects`
            let Some(assoc) = tcx.associated_type_for_effects(trait_ref.def_id()) else {
                tcx.dcx().span_delayed_bug(
                    span,
                    "`~const` trait bound has no effect assoc yet no errors encountered?",
                );
                return;
            };
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
