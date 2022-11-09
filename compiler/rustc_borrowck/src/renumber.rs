#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]
use crate::BorrowckInferCtxt;
use rustc_index::vec::IndexVec;
use rustc_infer::infer::NllRegionVariableOrigin;
use rustc_middle::mir::visit::{MutVisitor, TyContext};
use rustc_middle::mir::Constant;
use rustc_middle::mir::{Body, Location, Promoted};
use rustc_middle::ty::subst::SubstsRef;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable};
use rustc_span::Symbol;

/// Replaces all free regions appearing in the MIR with fresh
/// inference variables, returning the number of variables created.
#[instrument(skip(infcx, body, promoted), level = "debug")]
pub fn renumber_mir<'tcx>(
    infcx: &BorrowckInferCtxt<'_, 'tcx>,
    body: &mut Body<'tcx>,
    promoted: &mut IndexVec<Promoted, Body<'tcx>>,
) {
    debug!(?body.arg_count);

    let mut visitor = NllVisitor { infcx };

    for body in promoted.iter_mut() {
        visitor.visit_body(body);
    }

    visitor.visit_body(body);
}

/// Replaces all regions appearing in `value` with fresh inference
/// variables.
#[cfg(not(debug_assertions))]
#[instrument(skip(infcx), level = "debug")]
pub(crate) fn renumber_regions<'tcx, T>(infcx: &BorrowckInferCtxt<'_, 'tcx>, value: T) -> T
where
    T: TypeFoldable<'tcx>,
{
    infcx.tcx.fold_regions(value, |_region, _depth| {
        let origin = NllRegionVariableOrigin::Existential { from_forall: false };
        infcx.next_nll_region_var(origin)
    })
}

/// Replaces all regions appearing in `value` with fresh inference
/// variables.
#[cfg(debug_assertions)]
#[instrument(skip(infcx), level = "debug")]
pub(crate) fn renumber_regions<'tcx, T>(
    infcx: &BorrowckInferCtxt<'_, 'tcx>,
    value: T,
    ctxt: RegionCtxt,
) -> T
where
    T: TypeFoldable<'tcx>,
{
    infcx.tcx.fold_regions(value, |_region, _depth| {
        let origin = NllRegionVariableOrigin::Existential { from_forall: false };
        infcx.next_nll_region_var(origin, ctxt)
    })
}

#[cfg(debug_assertions)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) enum RegionCtxt {
    Location(Location),
    TyContext(TyContext),
    Free(Symbol),
    Bound(Symbol),
    LateBound(Symbol),
    Existential(Option<Symbol>),
    Placeholder(Symbol),
    Unknown,
}

#[cfg(debug_assertions)]
impl RegionCtxt {
    /// Used to determine the representative of a component in the strongly connected
    /// constraint graph
    /// FIXME: don't use underscore here. Got a 'not used' error for some reason
    pub(crate) fn _preference_value(self) -> usize {
        let _anon = Symbol::intern("anon");

        match self {
            RegionCtxt::Unknown => 1,
            RegionCtxt::Existential(None) => 2,
            RegionCtxt::Existential(Some(_anon))
            | RegionCtxt::Free(_anon)
            | RegionCtxt::Bound(_anon)
            | RegionCtxt::LateBound(_anon) => 2,
            RegionCtxt::Location(_) => 3,
            RegionCtxt::TyContext(_) => 4,
            _ => 5,
        }
    }
}

struct NllVisitor<'a, 'tcx> {
    infcx: &'a BorrowckInferCtxt<'a, 'tcx>,
}

impl<'a, 'tcx> NllVisitor<'a, 'tcx> {
    #[cfg(debug_assertions)]
    fn renumber_regions<T>(&mut self, value: T, ctxt: RegionCtxt) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        renumber_regions(self.infcx, value, ctxt)
    }

    #[cfg(not(debug_assertions))]
    fn renumber_regions<T>(&mut self, value: T) -> T
    where
        T: TypeFoldable<'tcx>,
    {
        renumber_regions(self.infcx, value)
    }
}

impl<'a, 'tcx> MutVisitor<'tcx> for NllVisitor<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    #[cfg(not(debug_assertions))]
    #[instrument(skip(self), level = "debug")]
    fn visit_ty(&mut self, ty: &mut Ty<'tcx>, _ty_context: TyContext) {
        *ty = self.renumber_regions(*ty);

        debug!(?ty);
    }

    #[cfg(debug_assertions)]
    #[instrument(skip(self), level = "debug")]
    fn visit_ty(&mut self, ty: &mut Ty<'tcx>, _ty_context: TyContext) {
        *ty = self.renumber_regions(*ty, RegionCtxt::TyContext(_ty_context));

        debug!(?ty);
    }

    #[cfg(not(debug_assertions))]
    #[instrument(skip(self), level = "debug")]
    fn visit_substs(&mut self, substs: &mut SubstsRef<'tcx>, location: Location) {
        *substs = self.renumber_regions(*substs);

        debug!(?substs);
    }

    #[cfg(debug_assertions)]
    #[instrument(skip(self), level = "debug")]
    fn visit_substs(&mut self, substs: &mut SubstsRef<'tcx>, location: Location) {
        *substs = self.renumber_regions(*substs, RegionCtxt::Location(location));

        debug!(?substs);
    }

    #[cfg(not(debug_assertions))]
    #[instrument(skip(self), level = "debug")]
    fn visit_region(&mut self, region: &mut ty::Region<'tcx>, location: Location) {
        let old_region = *region;
        *region = self.renumber_regions(old_region);

        debug!(?region);
    }

    #[cfg(debug_assertions)]
    #[instrument(skip(self), level = "debug")]
    fn visit_region(&mut self, region: &mut ty::Region<'tcx>, location: Location) {
        let old_region = *region;
        *region = self.renumber_regions(old_region, RegionCtxt::Location(location));

        debug!(?region);
    }

    #[cfg(not(debug_assertions))]
    #[instrument(skip(self), level = "debug")]
    fn visit_constant(&mut self, constant: &mut Constant<'tcx>, _location: Location) {
        let literal = constant.literal;
        constant.literal = self.renumber_regions(literal);
        debug!("constant: {:#?}", constant);
    }

    #[cfg(debug_assertions)]
    #[instrument(skip(self), level = "debug")]
    fn visit_constant(&mut self, constant: &mut Constant<'tcx>, _location: Location) {
        let literal = constant.literal;
        constant.literal = self.renumber_regions(literal, RegionCtxt::Location(_location));
        debug!("constant: {:#?}", constant);
    }
}
