//! Removes assignments to ZST places.

use crate::transform::MirPass;
use rustc_middle::mir::{Body, StatementKind};
use rustc_middle::ty::TyCtxt;

pub struct RemoveZsts;

impl<'tcx> MirPass<'tcx> for RemoveZsts {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let param_env = tcx.param_env(body.source.def_id());
        let (basic_blocks, local_decls) = body.basic_blocks_and_local_decls_mut();
        for block in basic_blocks.iter_mut() {
            for statement in block.statements.iter_mut() {
                match statement.kind {
                    StatementKind::Assign(box (place, _)) => {
                        let place_ty = place.ty(local_decls, tcx).ty;
                        if let Ok(layout) = tcx.layout_of(param_env.and(place_ty)) {
                            if layout.is_zst() && !layout.abi.is_uninhabited() {
                                if tcx.consider_optimizing(|| {
                                    format!(
                                        "RemoveZsts - Place: {:?} SourceInfo: {:?}",
                                        place, statement.source_info
                                    )
                                }) {
                                    statement.make_nop();
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}
