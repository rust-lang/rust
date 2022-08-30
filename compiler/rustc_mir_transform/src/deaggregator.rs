use crate::util::expand_aggregate;
use crate::MirPass;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

pub struct Deaggregator;

impl<'tcx> MirPass<'tcx> for Deaggregator {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let basic_blocks = body.basic_blocks.as_mut_preserves_cfg();
        for bb in basic_blocks {
            bb.expand_statements(|stmt| {
                // FIXME(eddyb) don't match twice on `stmt.kind` (post-NLL).
                match stmt.kind {
                    // FIXME(#48193) Deaggregate arrays when it's cheaper to do so.
                    StatementKind::Assign(box (
                        _,
                        Rvalue::Aggregate(box AggregateKind::Array(_), _),
                    )) => {
                        return None;
                    }
                    StatementKind::Assign(box (_, Rvalue::Aggregate(_, _))) => {}
                    _ => return None,
                }

                let stmt = stmt.replace_nop();
                let source_info = stmt.source_info;
                let StatementKind::Assign(box (lhs, Rvalue::Aggregate(kind, operands))) = stmt.kind else {
                    bug!();
                };

                Some(expand_aggregate(
                    lhs,
                    operands.into_iter().map(|op| {
                        let ty = op.ty(&body.local_decls, tcx);
                        (op, ty)
                    }),
                    *kind,
                    source_info,
                    tcx,
                ))
            });
        }
    }
}
