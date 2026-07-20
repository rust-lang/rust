use rustc_index::{IndexSlice, IndexVec};
use rustc_infer::infer::NllRegionVariableOrigin;
use rustc_middle::mir::visit::{MutVisitor, TyContext};
use rustc_middle::mir::*;
use rustc_middle::ty::{self, GenericArgsRef, Ty, TyCtxt, TypeFoldable, fold_regions};
use rustc_span::Symbol;
use thin_vec::ThinVec;
use tracing::{debug, instrument};

use crate::BorrowckInferCtxt;

/// Replaces all free regions appearing in the MIR with fresh
/// inference variables, returning the number of variables created.
#[instrument(skip(infcx, body, promoted), level = "debug")]
pub(crate) fn renumber_mir<'tcx>(
    infcx: &BorrowckInferCtxt<'tcx>,
    body: &mut Body<'tcx>,
    promoted: &mut IndexSlice<Promoted, Body<'tcx>>,
) {
    debug!(?body.arg_count);

    let mut renumberer = RegionRenumberer { infcx };

    for body in promoted.iter_mut() {
        split_critical_edges(body);
        renumberer.visit_body_preserves_cfg(body);
    }

    split_critical_edges(body);
    renumberer.visit_body_preserves_cfg(body);
}

#[instrument(skip(body), level = "debug")]
fn split_critical_edges(body: &mut Body<'_>) {
    let predecessors: IndexVec<BasicBlock, _> =
        body.basic_blocks.predecessors().iter().map(|preds| preds.len()).collect();
    debug!(?predecessors);

    let mut new_blocks = vec![];
    for bb in predecessors.indices() {
        let term = body.basic_blocks[bb].terminator();
        if term.successors().count() <= 1 {
            continue;
        }
        if term.successors().all(|s| predecessors[s] <= 1) {
            continue;
        }

        debug!(
            "{bb:?} has critical edges: {:?}",
            term.successors().map(|s| (s, predecessors[s])).collect::<Vec<_>>(),
        );

        let original_succ: Vec<_> = term.successors().collect();
        new_blocks.push((bb, original_succ));
    }

    if new_blocks.is_empty() {
        return;
    }

    debug!(?new_blocks);
    let basic_blocks = body.basic_blocks.as_mut();
    for (bb, successors) in new_blocks.iter_mut() {
        let source_info = basic_blocks[*bb].terminator().source_info;
        for target in successors.iter_mut() {
            if predecessors[*target] <= 1 {
                continue;
            }

            let is_cleanup = basic_blocks[*target].is_cleanup;
            let terminator = Terminator {
                source_info,
                kind: TerminatorKind::Goto { target: *target },
                attributes: ThinVec::new(),
            };
            *target = basic_blocks.push(BasicBlockData::new(Some(terminator), is_cleanup))
        }
    }

    for (bb, new_succ) in new_blocks {
        let mut new_succ = new_succ.into_iter();
        basic_blocks[bb].terminator_mut().successors_mut(|succ| *succ = new_succ.next().unwrap());
    }
}

// The fields are used only for debugging output in `sccs_info`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) enum RegionCtxt {
    Location(Location),
    TyContext(TyContext),
    Free(Symbol),
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
    infcx: &'a BorrowckInferCtxt<'tcx>,
}

impl<'a, 'tcx> RegionRenumberer<'a, 'tcx> {
    /// Replaces all regions appearing in `value` with fresh inference
    /// variables.
    fn renumber_regions<T, F>(&mut self, value: T, region_ctxt_fn: F) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
        F: Fn() -> RegionCtxt,
    {
        let origin = NllRegionVariableOrigin::Existential { name: None };
        fold_regions(self.infcx.tcx, value, |_region, _depth| {
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
    fn visit_const_operand(&mut self, constant: &mut ConstOperand<'tcx>, location: Location) {
        let const_ = constant.const_;
        constant.const_ = self.renumber_regions(const_, || RegionCtxt::Location(location));
        debug!("constant: {:#?}", constant);
    }
}
