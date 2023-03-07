//! Removes assignments to ZST places.

use crate::MirPass;
use rustc_middle::mir::{Body, StatementKind};
use rustc_middle::ty::{self, Ty, TyCtxt};

pub struct RemoveZsts;

impl<'tcx> MirPass<'tcx> for RemoveZsts {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 0
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // Avoid query cycles (generators require optimized MIR for layout).
        if tcx.type_of(body.source.def_id()).subst_identity().is_generator() {
            return;
        }
        let param_env = tcx.param_env(body.source.def_id());
        let basic_blocks = body.basic_blocks.as_mut_preserves_cfg();
        let local_decls = &body.local_decls;
        for block in basic_blocks {
            for statement in block.statements.iter_mut() {
                if let StatementKind::Assign(box (place, _)) | StatementKind::Deinit(box place) =
                    statement.kind
                {
                    let place_ty = place.ty(local_decls, tcx).ty;
                    if !maybe_zst(place_ty) {
                        continue;
                    }
                    let Ok(layout) = tcx.layout_of(param_env.and(place_ty)) else {
                        continue;
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
            }
        }
    }
}

/// A cheap, approximate check to avoid unnecessary `layout_of` calls.
fn maybe_zst(ty: Ty<'_>) -> bool {
    match ty.kind() {
        // maybe ZST (could be more precise)
        ty::Adt(..)
        | ty::Array(..)
        | ty::Closure(..)
        | ty::Tuple(..)
        | ty::Alias(ty::Opaque, ..) => true,
        // definitely ZST
        ty::FnDef(..) | ty::Never => true,
        // unreachable or can't be ZST
        _ => false,
    }
}
