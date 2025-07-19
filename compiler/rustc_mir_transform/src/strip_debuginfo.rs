use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::debuginfo::debuginfo_locals;
use rustc_session::config::MirStripDebugInfo;

/// Conditionally remove some of the VarDebugInfo in MIR.
///
/// In particular, stripping non-parameter debug info for tiny, primitive-like
/// methods in core saves work later, and nobody ever wanted to use it anyway.
pub(super) struct StripDebugInfo;

impl<'tcx> crate::MirPass<'tcx> for StripDebugInfo {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.opts.unstable_opts.mir_strip_debuginfo != MirStripDebugInfo::None
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        match tcx.sess.opts.unstable_opts.mir_strip_debuginfo {
            MirStripDebugInfo::None => return,
            MirStripDebugInfo::AllLocals => {}
            MirStripDebugInfo::LocalsInTinyFunctions
                if let TerminatorKind::Return { .. } =
                    body.basic_blocks[START_BLOCK].terminator().kind => {}
            MirStripDebugInfo::LocalsInTinyFunctions => return,
        }

        body.var_debug_info.retain(|vdi| {
            matches!(
                vdi.value,
                VarDebugInfoContents::Place(place)
                    if place.local.as_usize() <= body.arg_count && place.local != RETURN_PLACE,
            )
        });

        let debuginfo_locals = debuginfo_locals(body);
        for data in body.basic_blocks.as_mut_preserves_cfg() {
            for stmt in data.statements.iter_mut() {
                stmt.debuginfos.retain(|debuginfo| match debuginfo {
                    StmtDebugInfo::AssignRef(local, _) | StmtDebugInfo::InvalidAssign(local) => {
                        debuginfo_locals.contains(*local)
                    }
                });
            }
            data.after_last_stmt_debuginfos.retain(|debuginfo| match debuginfo {
                StmtDebugInfo::AssignRef(local, _) | StmtDebugInfo::InvalidAssign(local) => {
                    debuginfo_locals.contains(*local)
                }
            });
        }
    }

    fn is_required(&self) -> bool {
        true
    }
}
