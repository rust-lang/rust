//! Removes assignments to ZST places.

use crate::transform::MirPass;
use rustc_middle::mir::{Body, StatementKind};
use rustc_middle::ty::{self, Ty, TyCtxt};

pub struct RemoveZsts;

impl<'tcx> MirPass<'tcx> for RemoveZsts {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if tcx.sess.mir_opt_level() < 3 {
            return;
        }
        let param_env = tcx.param_env(body.source.def_id());
        let (basic_blocks, local_decls) = body.basic_blocks_and_local_decls_mut();
        for block in basic_blocks.iter_mut() {
            for statement in block.statements.iter_mut() {
                match statement.kind {
                    StatementKind::Assign(box (place, _)) => {
                        let place_ty = place.ty(local_decls, tcx).ty;
                        if !maybe_zst(place_ty) {
                            continue;
                        }
                        let layout = match tcx.layout_of(param_env.and(place_ty)) {
                            Ok(layout) => layout,
                            Err(_) => continue,
                        };
                        if !layout.is_zst() {
                            continue;
                        }
                        if tcx.consider_optimizing(|| {
                            format!(
                                "RemoveZsts - Place: {:?} SourceInfo: {:?}",
                                place, statement.source_info
                            )
                        }) {
                            statement.make_nop();
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

/// A cheap, approximate check to avoid unnecessary `layout_of` calls.
fn maybe_zst(ty: Ty<'_>) -> bool {
    match ty.kind() {
        // maybe ZST (could be more precise)
        ty::Adt(..) | ty::Array(..) | ty::Closure(..) | ty::Tuple(..) | ty::Opaque(..) => true,
        // definitely ZST
        ty::FnDef(..) | ty::Never => true,
        // unreachable or can't be ZST
        _ => false,
    }
}
