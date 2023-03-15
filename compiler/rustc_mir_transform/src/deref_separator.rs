use crate::MirPass;
use rustc_index::vec::IndexVec;
use rustc_middle::mir::patch::MirPatch;
use rustc_middle::mir::visit::NonUseContext::VarDebugInfo;
use rustc_middle::mir::visit::{MutVisitor, PlaceContext};
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

pub struct Derefer;

pub struct DerefChecker<'tcx> {
    tcx: TyCtxt<'tcx>,
    patcher: MirPatch<'tcx>,
    local_decls: IndexVec<Local, LocalDecl<'tcx>>,
}

impl<'tcx> MutVisitor<'tcx> for DerefChecker<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_place(&mut self, place: &mut Place<'tcx>, cntxt: PlaceContext, loc: Location) {
        if !place.projection.is_empty()
            && cntxt != PlaceContext::NonUse(VarDebugInfo)
            && place.projection[1..].contains(&ProjectionElem::Deref)
        {
            let mut place_local = place.local;
            let mut last_len = 0;
            let mut last_deref_idx = 0;

            for (idx, elem) in place.projection[0..].iter().enumerate() {
                if *elem == ProjectionElem::Deref {
                    last_deref_idx = idx;
                }
            }

            for (idx, (p_ref, p_elem)) in place.iter_projections().enumerate() {
                if !p_ref.projection.is_empty() && p_elem == ProjectionElem::Deref {
                    let ty = p_ref.ty(&self.local_decls, self.tcx).ty;
                    let temp = self.patcher.new_internal_with_info(
                        ty,
                        self.local_decls[p_ref.local].source_info.span,
                        LocalInfo::DerefTemp,
                    );

                    // We are adding current p_ref's projections to our
                    // temp value, excluding projections we already covered.
                    let deref_place = Place::from(place_local)
                        .project_deeper(&p_ref.projection[last_len..], self.tcx);

                    self.patcher.add_assign(
                        loc,
                        Place::from(temp),
                        Rvalue::CopyForDeref(deref_place),
                    );
                    place_local = temp;
                    last_len = p_ref.projection.len();

                    // Change `Place` only if we are actually at the Place's last deref
                    if idx == last_deref_idx {
                        let temp_place =
                            Place::from(temp).project_deeper(&place.projection[idx..], self.tcx);
                        *place = temp_place;
                    }
                }
            }
        }
    }
}

pub fn deref_finder<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let patch = MirPatch::new(body);
    let mut checker = DerefChecker { tcx, patcher: patch, local_decls: body.local_decls.clone() };

    for (bb, data) in body.basic_blocks.as_mut_preserves_cfg().iter_enumerated_mut() {
        checker.visit_basic_block_data(bb, data);
    }

    checker.patcher.apply(body);
}

impl<'tcx> MirPass<'tcx> for Derefer {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        deref_finder(tcx, body);
    }
}
