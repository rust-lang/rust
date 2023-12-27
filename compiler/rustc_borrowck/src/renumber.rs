#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]
use crate::BorrowckInferCtxt;
use rustc_index::IndexSlice;
use rustc_infer::infer::NllRegionVariableOrigin;
use rustc_middle::mir::visit::{MutVisitor, TyContext};
use rustc_middle::mir::{Body, ConstOperand, Location, Promoted};
use rustc_middle::ty::GenericArgsRef;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable};
use rustc_span::Symbol;

/// Replaces all free regions appearing in the MIR with fresh
/// inference variables, returning the number of variables created.
#[instrument(skip(infcx, body, promoted), level = "debug")]
pub fn renumber_mir<'tcx>(
    infcx: &BorrowckInferCtxt<'_, 'tcx>,
    body: &mut Body<'tcx>,
    promoted: &mut IndexSlice<Promoted, Body<'tcx>>,
) {
    debug!(?body.arg_count);

    let mut renumberer = RegionRenumberer { infcx };

    for body in promoted.iter_mut() {
        renumberer.visit_body(body);
    }

    renumberer.visit_body(body);
}

// FIXME(@lcnr): A lot of these variants overlap and it seems like
// this type is only used to decide which region should be used
// as representative. This should be cleaned up.
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

impl RegionCtxt {
    /// Used to determine the representative of a component in the strongly connected
    /// constraint graph
    pub(crate) fn preference_value(self) -> usize {
        match self {
            RegionCtxt::Unknown => 1,
            RegionCtxt::Existential(None) => 2,
            RegionCtxt::Existential(Some(_)) | RegionCtxt::Free(_) => 2,
            RegionCtxt::Location(_) => 3,
            RegionCtxt::TyContext(_) => 4,
            _ => 5,
        }
    }
}

struct RegionRenumberer<'a, 'tcx> {
    infcx: &'a BorrowckInferCtxt<'a, 'tcx>,
}

impl<'a, 'tcx> RegionRenumberer<'a, 'tcx> {
    /// Replaces all regions appearing in `value` with fresh inference
    /// variables.
    fn renumber_regions<T, F>(&mut self, value: T, region_ctxt_fn: F) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
        F: Fn() -> RegionCtxt,
    {
        let origin = NllRegionVariableOrigin::Existential { from_forall: false };
        self.infcx.tcx.fold_regions(value, |_region, _depth| {
            self.infcx.next_nll_region_var(origin, || region_ctxt_fn())
        })
    }
}

impl<'a, 'tcx> MutVisitor<'tcx> for RegionRenumberer<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_ty(&mut self, ty: &mut Ty<'tcx>, ty_context: TyContext) {
        if matches!(ty_context, TyContext::ReturnTy(_)) {
            // We will renumber the return ty when called again with `TyContext::LocalDecl`
            return;
        }
        *ty = self.renumber_regions(*ty, || RegionCtxt::TyContext(ty_context));

        debug!(?ty);
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_args(&mut self, args: &mut GenericArgsRef<'tcx>, location: Location) {
        *args = self.renumber_regions(*args, || RegionCtxt::Location(location));

        debug!(?args);
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_region(&mut self, region: &mut ty::Region<'tcx>, location: Location) {
        let old_region = *region;
        *region = self.renumber_regions(old_region, || RegionCtxt::Location(location));

        debug!(?region);
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_ty_const(&mut self, ct: &mut ty::Const<'tcx>, location: Location) {
        let old_ct = *ct;
        *ct = self.renumber_regions(old_ct, || RegionCtxt::Location(location));

        debug!(?ct);
    }

    #[instrument(skip(self), level = "debug")]
    fn visit_constant(&mut self, constant: &mut ConstOperand<'tcx>, location: Location) {
        let const_ = constant.const_;
        constant.const_ = self.renumber_regions(const_, || RegionCtxt::Location(location));
        debug!("constant: {:#?}", constant);
    }
}
