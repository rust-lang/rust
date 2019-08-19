use rustc::mir::*;
use rustc::ty::TyCtxt;
use crate::transform::{MirPass, MirSource};
use crate::util::expand_aggregate;

pub struct Deaggregator;

impl MirPass for Deaggregator {
    fn run_pass<'tcx>(&self, tcx: TyCtxt<'tcx>, _source: MirSource<'tcx>, body: &mut Body<'tcx>) {
        let (basic_blocks, local_decls) = body.basic_blocks_and_local_decls_mut();
        let local_decls = &*local_decls;
        for bb in basic_blocks {
            bb.expand_statements(|stmt| {
                // FIXME(eddyb) don't match twice on `stmt.kind` (post-NLL).
                if let StatementKind::Assign(_, ref rhs) = stmt.kind {
                    if let Rvalue::Aggregate(ref kind, _) = **rhs {
                        // FIXME(#48193) Deaggregate arrays when it's cheaper to do so.
                        if let AggregateKind::Array(_) = **kind {
                            return None;
                        }
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }

                let stmt = stmt.replace_nop();
                let source_info = stmt.source_info;
                let (lhs, kind, operands) = match stmt.kind {
                    StatementKind::Assign(lhs, box rvalue) => {
                        match rvalue {
                            Rvalue::Aggregate(kind, operands) => (lhs, kind, operands),
                            _ => bug!()
                        }
                    }
                    _ => bug!()
                };

                Some(expand_aggregate(
                    lhs,
                    operands.into_iter().map(|op| {
                        let ty = op.ty(local_decls, tcx);
                        (op, ty)
                    }),
                    *kind,
                    source_info,
                ))
            });
        }
    }
}
