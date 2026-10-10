//! This pass converts all `DerefTemp` locals into normal temporaries
//! and turns their `CopyForDeref` rvalues into normal copies.

use rustc_middle::mir::visit::MutVisitor;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

use crate::PassPolicy;

struct EraseDerefTempsVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> MutVisitor<'tcx> for EraseDerefTempsVisitor<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_rvalue(&mut self, rvalue: &mut Rvalue<'tcx>, _: Location) {
        if let &mut Rvalue::CopyForDeref(place) = rvalue {
            // We must *NOT* retag here! Not only can this assignment copy mutable references that
            // cannot actually be copied, given some r: &&(u8, u8), the assignment let v = (**r).0;
            // will do a CopyForDeref of the &(u8, u8) but only do a read of the first element, so
            // doing a retagging copy adds UB.
            *rvalue = Rvalue::Use(Operand::Copy(place), WithRetag::No)
        }
    }

    fn visit_local_decl(&mut self, _: Local, local_decl: &mut LocalDecl<'tcx>) {
        if local_decl.is_deref_temp() {
            let info = local_decl.local_info.as_mut().unwrap_crate_local();
            **info = LocalInfo::Boring;
        }
    }
}

pub(super) struct EraseDerefTemps;

impl<'tcx> crate::MirPass<'tcx> for EraseDerefTemps {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        EraseDerefTempsVisitor { tcx }.visit_body_preserves_cfg(body);
    }

    fn policy(&self, _ctx: &crate::PassCtx<'_>) -> PassPolicy {
        // Later MIR stages assume that CopyForDeref is gone.
        PassPolicy::Required
    }
}
