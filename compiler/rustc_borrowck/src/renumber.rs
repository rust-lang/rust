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
use rustc_span::{Span, Symbol};

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
#[instrument(skip(infcx, get_ctxt_fn), level = "debug")]
pub(crate) fn renumber_regions<'tcx, T, F>(
    infcx: &BorrowckInferCtxt<'_, 'tcx>,
    value: T,
    get_ctxt_fn: F,
) -> T
where
    T: TypeFoldable<TyCtxt<'tcx>>,
    F: Fn() -> RegionCtxt,
{
    infcx.tcx.fold_regions(value, |_region, _depth| {
        let origin = NllRegionVariableOrigin::Existential { from_forall: false };
        infcx.next_nll_region_var(origin, || get_ctxt_fn())
    })
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) enum BoundRegionInfo {
    Name(Symbol),
    Span(Span),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) enum RegionCtxt {
    Location(Location),
    TyContext(TyContext),
    Free(Symbol),
    Bound(BoundRegionInfo),
    LateBound(BoundRegionInfo),
    Existential(Option<Symbol>),
    Placeholder(BoundRegionInfo),
    Unknown,
}

impl RegionCtxt {
    /// Used to determine the representative of a component in the strongly connected
    /// constraint graph
    pub(crate) fn preference_value(self) -> usize {
        let _anon = Symbol::intern("anon");

        match self {
            RegionCtxt::Unknown => 1,
            RegionCtxt::Existential(None) => 2,
            RegionCtxt::Existential(Some(_anon)) | RegionCtxt::Free(_anon) => 2,
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
    fn renumber_regions<T, F>(&mut self, value: T, region_ctxt_fn: F) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
        F: Fn() -> RegionCtxt,
    {
        renumber_regions(self.infcx, value, region_ctxt_fn)
    }
}

impl<'a, 'tcx> MutVisitor<'tcx> for NllVisitor<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_ty(&mut self, ty: &mut Ty<'tcx>, ty_context: TyContext) {
        *ty = self.renumber_regions(*ty, || RegionCtxt::TyContext(ty_context));

        debug!(?ty);
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_substs(&mut self, substs: &mut SubstsRef<'tcx>, location: Location) {
        *substs = self.renumber_regions(*substs, || RegionCtxt::Location(location));

        debug!(?substs);
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_region(&mut self, region: &mut ty::Region<'tcx>, location: Location) {
        let old_region = *region;
        *region = self.renumber_regions(old_region, || RegionCtxt::Location(location));

        debug!(?region);
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_constant(&mut self, constant: &mut Constant<'tcx>, _location: Location) {
        let literal = constant.literal;
        constant.literal = self.renumber_regions(literal, || RegionCtxt::Location(_location));
        debug!("constant: {:#?}", constant);
    }
}
