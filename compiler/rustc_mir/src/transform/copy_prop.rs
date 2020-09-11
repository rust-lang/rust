//! Trivial copy propagation pass.
//!
//! This uses def-use analysis to remove values that have exactly one def and one use, which must
//! be an assignment.
//!
//! To give an example, we look for patterns that look like:
//!
//!     DEST = SRC
//!     ...
//!     USE(DEST)
//!
//! where `DEST` and `SRC` are both locals of some form. We replace that with:
//!
//!     NOP
//!     ...
//!     USE(SRC)
//!
//! The assignment `DEST = SRC` must be (a) the only mutation of `DEST` and (b) the only
//! (non-mutating) use of `SRC`. These restrictions are conservative and may be relaxed in the
//! future.

use crate::transform::{MirPass, MirSource};
use crate::util::def_use::DefUseAnalysis;
use rustc_middle::mir::{
    Body, Constant, Local, LocalKind, Location, Operand, Place, Rvalue, StatementKind,
};
use rustc_middle::ty::TyCtxt;

pub struct CopyPropagation;

impl<'tcx> MirPass<'tcx> for CopyPropagation {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, source: MirSource<'tcx>, body: &mut Body<'tcx>) {
        // We only run when the MIR optimization level is > 1.
        // This avoids a slow pass, and messing up debug info.
        if tcx.sess.opts.debugging_opts.mir_opt_level <= 1 {
            return;
        }

        let mut def_use_analysis = DefUseAnalysis::new(body);
        let mut propagated_local = 0;
        let mut propagated_const = 0;

        for dest_local in def_use_analysis.body().local_decls.indices() {
            let action = match prepare_action(&mut def_use_analysis, dest_local) {
                None => continue,
                Some(action) => action,
            };
            if action.perform(tcx, &mut def_use_analysis) {
                match action {
                    Action::PropagateLocalCopy { .. } => propagated_local += 1,
                    Action::PropagateConstant { .. } => propagated_const += 1,
                }
            }
        }

        let propagated = propagated_local + propagated_const;
        info!(
            "{:?} propagated={} local={} const={}",
            source.instance.def_id(),
            propagated,
            propagated_local,
            propagated_const
        );
    }
}

fn prepare_action<'tcx, 'a>(
    def_use_analysis: &mut DefUseAnalysis<'tcx, 'a>,
    dest_local: Local,
) -> Option<Action<'tcx>> {
    debug!("considering destination local: {:?}", dest_local);

    // The destination must have exactly one def.
    let dest_use_info = def_use_analysis.local_info(dest_local);
    let dest_def_count = dest_use_info.def_count_not_including_drop();
    if dest_def_count == 0 {
        debug!("  Can't copy-propagate local: dest {:?} undefined", dest_local);
        return None;
    }
    if dest_def_count > 1 {
        debug!(
            "  Can't copy-propagate local: dest {:?} defined {} times",
            dest_local,
            dest_use_info.def_count()
        );
        return None;
    }
    if dest_use_info.use_count() == 0 {
        debug!("  Can't copy-propagate local: dest {:?} unused", dest_local);
        return None;
    }
    // Conservatively gives up if the dest is an argument,
    // because there may be uses of the original argument value.
    // Also gives up on the return place, as we cannot propagate into its implicit
    // use by `return`.
    if matches!(
        def_use_analysis.body().local_kind(dest_local),
        LocalKind::Arg | LocalKind::ReturnPointer
    ) {
        debug!("  Can't copy-propagate local: dest {:?} (argument)", dest_local);
        return None;
    }

    let mut defs = dest_use_info.defs();
    loop {
        let location = match defs.next() {
            None => return None,
            Some(def) => def,
        };
        let basic_block = &def_use_analysis.body()[location.block];
        let statement = match basic_block.statements.get(location.statement_index) {
            Some(statement) => statement,
            None => {
                debug!("  Can't copy-propagate local: used in terminator");
                continue;
            }
        };
        // That use of the source must be an assignment.
        match &statement.kind {
            StatementKind::Assign(box (place, Rvalue::Use(operand))) => {
                if let Some(local) = place.as_local() {
                    if local == dest_local {
                        let maybe_action = match operand {
                            Operand::Copy(src_place) | Operand::Move(src_place) => {
                                Action::local_copy(
                                    def_use_analysis,
                                    *src_place,
                                    dest_local,
                                    location,
                                )
                            }
                            Operand::Constant(ref src_constant) => {
                                Action::constant(src_constant, dest_local, location)
                            }
                        };
                        match maybe_action {
                            Some(action) => return Some(action),
                            None => continue,
                        }
                    } else {
                        debug!(
                            "  Can't copy-propagate local: source use is not an \
                        assignment"
                        );
                        continue;
                    }
                } else {
                    debug!(
                        "  Can't copy-propagate local: source use is not an \
                        assignment"
                    );
                    continue;
                }
            }
            _ => {
                debug!(
                    "  Can't copy-propagate local: source use is not an \
                        assignment"
                );
                continue;
            }
        }
    }
}

enum Action<'tcx> {
    PropagateLocalCopy { src_local: Local, dest_local: Local, location: Location },
    PropagateConstant { src_constant: Constant<'tcx>, dest_local: Local, location: Location },
}

impl<'tcx> Action<'tcx> {
    fn local_copy<'a>(
        def_use_analysis: &DefUseAnalysis<'tcx, 'a>,
        src_place: Place<'tcx>,
        dest_local: Local,
        location: Location,
    ) -> Option<Action<'tcx>> {
        // The source must be a local.
        let src_local = if let Some(local) = src_place.as_local() {
            local
        } else {
            debug!("  Can't copy-propagate local: source is not a local");
            return None;
        };

        // We're trying to copy propagate a local.
        // There must be exactly one use of the source used in a statement (not in a terminator).
        let src_use_info = def_use_analysis.local_info(src_local);
        let src_use_count = src_use_info.use_count();
        if src_use_count == 0 {
            debug!("  Can't copy-propagate local: no uses");
            return None;
        }
        if src_use_count != 1 {
            debug!("  Can't copy-propagate local: {} uses", src_use_info.use_count());
            return None;
        }

        // Verify that the source doesn't change in between. This is done conservatively for now,
        // by ensuring that the source has exactly one mutation. The goal is to prevent things
        // like:
        //
        //     DEST = SRC;
        //     SRC = X;
        //     USE(DEST);
        //
        // From being misoptimized into:
        //
        //     SRC = X;
        //     USE(SRC);
        let src_def_count = src_use_info.def_count_not_including_drop();
        // allow function arguments to be propagated
        let is_arg = def_use_analysis.body().local_kind(src_local) == LocalKind::Arg;
        if (is_arg && src_def_count != 0) || (!is_arg && src_def_count != 1) {
            debug!(
                "  Can't copy-propagate local: {} defs of src{}",
                src_def_count,
                if is_arg { " (argument)" } else { "" },
            );
            return None;
        }

        Some(Action::PropagateLocalCopy { src_local, dest_local, location })
    }

    fn constant(
        src_constant: &Constant<'tcx>,
        dest_local: Local,
        location: Location,
    ) -> Option<Action<'tcx>> {
        Some(Action::PropagateConstant { src_constant: *src_constant, dest_local, location })
    }

    fn perform<'a>(
        &self,
        tcx: TyCtxt<'tcx>,
        def_use_analysis: &mut DefUseAnalysis<'tcx, 'a>,
    ) -> bool {
        match *self {
            Action::PropagateLocalCopy { src_local, dest_local, location } => {
                // Eliminate the destination and the assignment.
                //
                // First, remove all markers.
                //
                // FIXME(pcwalton): Don't do this. Merge live ranges instead.
                debug!("  Replacing all uses of {:?} with {:?} (local)", dest_local, src_local);
                def_use_analysis.remove_storage_markers(dest_local);
                def_use_analysis.remove_storage_markers(src_local);

                // Replace all uses of the destination local with the source local.
                def_use_analysis.replace_with_local(tcx, dest_local, src_local);

                // Finally, zap the now-useless assignment instruction.
                debug!("  Deleting assignment");
                def_use_analysis.remove_statement(location);

                true
            }
            Action::PropagateConstant { src_constant, dest_local, location } => {
                let old_use_count = def_use_analysis.local_info(dest_local).use_count();
                // First, remove all markers.
                //
                // FIXME(pcwalton): Don't do this. Merge live ranges instead.
                debug!(
                    "  Replacing all uses of {:?} with {:?} (constant)",
                    dest_local, src_constant
                );
                def_use_analysis.remove_storage_markers(dest_local);

                // Replace all uses of the destination local with the constant.
                def_use_analysis.replace_with_constant(tcx, dest_local, src_constant);

                // Zap the assignment instruction if we eliminated all the uses. We won't have been
                // able to do that if the destination was used in a projection, because projections
                // must have places on their LHS.
                let new_use_count = def_use_analysis.local_info(dest_local).use_count();
                let uses_replaced = old_use_count - new_use_count;
                if new_use_count == 0 {
                    debug!(
                        "  {} of {} use(s) replaced; deleting assignment",
                        uses_replaced, old_use_count
                    );
                    def_use_analysis.remove_statement(location);
                    true
                } else if uses_replaced == 0 {
                    debug!("  No uses replaced; not deleting assignment");
                    false
                } else {
                    debug!(
                        "  {} of {} use(s) replaced; not deleting assignment",
                        uses_replaced, old_use_count
                    );
                    true
                }
            }
        }
    }
}
