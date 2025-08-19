use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

use crate::patch::MirPatch;

pub(super) struct Subtyper;

struct SubTypeChecker<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    patcher: MirPatch<'tcx>,
    local_decls: &'a LocalDecls<'tcx>,
}

impl<'a, 'tcx> MutVisitor<'tcx> for SubTypeChecker<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_assign(
        &mut self,
        place: &mut Place<'tcx>,
        rvalue: &mut Rvalue<'tcx>,
        location: Location,
    ) {
        // We don't need to do anything for deref temps as they are
        // not part of the source code, but used for desugaring purposes.
        if self.local_decls[place.local].is_deref_temp() {
            return;
        }
        let mut place_ty = place.ty(self.local_decls, self.tcx).ty;
        let mut rval_ty = rvalue.ty(self.local_decls, self.tcx);
        // Not erasing this causes `Free Regions` errors in validator,
        // when rval is `ReStatic`.
        rval_ty = self.tcx.erase_regions(rval_ty);
        place_ty = self.tcx.erase_regions(place_ty);
        if place_ty != rval_ty {
            let temp = self
                .patcher
                .new_temp(rval_ty, self.local_decls[place.as_ref().local].source_info.span);
            let new_place = Place::from(temp);
            self.patcher.add_assign(location, new_place, rvalue.clone());
            let subtyped = new_place.project_deeper(&[ProjectionElem::Subtype(place_ty)], self.tcx);
            *rvalue = Rvalue::Use(Operand::Move(subtyped));
        }
    }
}

// Aim here is to do this kind of transformation:
//
// let place: place_ty = rval;
// // gets transformed to
// let temp: rval_ty = rval;
// let place: place_ty = temp as place_ty;
impl<'tcx> crate::MirPass<'tcx> for Subtyper {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let patch = MirPatch::new(body);
        let mut checker = SubTypeChecker { tcx, patcher: patch, local_decls: &body.local_decls };

        for (bb, data) in body.basic_blocks.as_mut_preserves_cfg().iter_enumerated_mut() {
            checker.visit_basic_block_data(bb, data);
        }
        checker.patcher.apply(body);
    }

    fn is_required(&self) -> bool {
        true
    }
}
