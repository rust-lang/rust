use crate::MirPass;
use rustc_middle::mir::patch::MirPatch;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

pub struct RefCmpSimplify;

impl<'tcx> MirPass<'tcx> for RefCmpSimplify {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        self.simplify_ref_cmp(tcx, body)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatchState {
    Empty,
    Deref { src_statement_idx: usize, dst: Local, src: Local },
    CopiedFrom { src_statement_idx: usize, dst: Local, real_src: Local },
    Completed { src_statement_idx: usize, dst: Local, real_src: Local },
}

impl RefCmpSimplify {
    fn simplify_ref_cmp<'tcx>(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        debug!("body: {:#?}", body);

        let n_bbs = body.basic_blocks.len() as u32;
        for bb in 0..n_bbs {
            let bb = BasicBlock::from_u32(bb);
            let mut max = Local::MAX;
            'repeat: loop {
                let mut state = MatchState::Empty;
                let bb_data = &body.basic_blocks[bb];
                for (i, stmt) in bb_data.statements.iter().enumerate().rev() {
                    state = match (state, &stmt.kind) {
                        (
                            MatchState::Empty,
                            StatementKind::Assign(box (lhs, Rvalue::Use(Operand::Copy(rhs)))),
                        ) if rhs.has_deref() && lhs.ty(body, tcx).ty.is_primitive() => {
                            let Some(dst) = lhs.as_local() else {
                                continue
                            };
                            let Some(src) = rhs.local_or_deref_local() else {
                                continue;
                            };
                            if max <= dst {
                                continue;
                            }
                            max = dst;
                            MatchState::Deref { dst, src, src_statement_idx: i }
                        }
                        (
                            MatchState::Deref { src, dst, src_statement_idx },
                            StatementKind::Assign(box (lhs, Rvalue::CopyForDeref(rhs))),
                        ) if lhs.as_local() == Some(src) && rhs.has_deref() => {
                            let Some(real_src) = rhs.local_or_deref_local() else{
                                continue;
                            };
                            MatchState::CopiedFrom { src_statement_idx, dst, real_src }
                        }
                        (
                            MatchState::CopiedFrom { src_statement_idx, dst, real_src },
                            StatementKind::Assign(box (
                                lhs,
                                Rvalue::Ref(_, BorrowKind::Shared | BorrowKind::Shallow, rhs),
                            )),
                        ) if lhs.as_local() == Some(real_src) => {
                            let Some(real_src) = rhs.as_local() else {
                                continue;
                            };
                            MatchState::Completed { dst, real_src, src_statement_idx }
                        }
                        _ => continue,
                    };
                    if let MatchState::Completed { dst, real_src, src_statement_idx } = state {
                        let mut patch = MirPatch::new(&body);
                        let src = Place::from(real_src);
                        let src = src.project_deeper(&[PlaceElem::Deref], tcx);
                        let dst = Place::from(dst);
                        let new_stmt =
                            StatementKind::Assign(Box::new((dst, Rvalue::Use(Operand::Copy(src)))));
                        patch.add_statement(
                            Location { block: bb, statement_index: src_statement_idx + 1 },
                            new_stmt,
                        );
                        patch.apply(body);
                        continue 'repeat;
                    }
                }
                break;
            }
        }
    }
}
