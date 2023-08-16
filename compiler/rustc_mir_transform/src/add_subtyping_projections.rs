use crate::MirPass;
use rustc_index::IndexVec;
use rustc_middle::mir::patch::MirPatch;
use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

pub struct Subtyper;

pub struct SubTypeCheker<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    patcher: MirPatch<'tcx>,
    local_decls: &'a IndexVec<Local, LocalDecl<'tcx>>,
}

impl<'a, 'tcx> MutVisitor<'tcx> for SubTypeCheker<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_assign(
        &mut self,
        place: &mut Place<'tcx>,
        rvalue: &mut Rvalue<'tcx>,
        location: Location,
    ) {
        let place_ty = place.ty(self.local_decls, self.tcx);
        let rval_ty = rvalue.ty(self.local_decls, self.tcx);
        if place_ty.ty != rval_ty {
            let temp = self
                .patcher
                .new_temp(rval_ty, self.local_decls[place.as_ref().local].source_info.span);
            let new_place =
                Place::from(temp).project_deeper(&[ProjectionElem::Subtype(place_ty.ty)], self.tcx);
            self.patcher.add_assign(location, new_place, rvalue.clone());
            let new_rval = Rvalue::Use(Operand::Move(new_place));
            *rvalue = new_rval;
        }
    }
}

pub fn subtype_finder<'tcx>(tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
    let patch = MirPatch::new(body);
    let mut checker = SubTypeCheker { tcx, patcher: patch, local_decls: &body.local_decls };

    for (bb, data) in body.basic_blocks.as_mut_preserves_cfg().iter_enumerated_mut() {
        checker.visit_basic_block_data(bb, data);
    }

    checker.patcher.apply(body);
}

impl<'tcx> MirPass<'tcx> for Subtyper {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        subtype_finder(tcx, body);
    }
}
