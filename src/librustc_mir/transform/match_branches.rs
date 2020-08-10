use crate::transform::{simplify, MirPass, MirSource};
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

pub struct MatchBranchSimplification;

// What's the intent of this pass?
// If one block is found that switches between blocks which both go to the same place
// AND both of these blocks set a similar const in their ->
// condense into 1 block based on discriminant AND goto the destination afterwards

impl<'tcx> MirPass<'tcx> for MatchBranchSimplification {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, src: MirSource<'tcx>, body: &mut Body<'tcx>) {
        let param_env = tcx.param_env(src.def_id());
        let mut did_remove_blocks = false;
        let bbs = body.basic_blocks_mut();
        'outer: for bb_idx in bbs.indices() {
            let (discr, val, switch_ty, targets) = match bbs[bb_idx].terminator().kind {
                TerminatorKind::SwitchInt {
                    discr: Operand::Move(ref place),
                    switch_ty,
                    ref targets,
                    ref values,
                    ..
                } if targets.len() == 2 && values.len() == 1 => {
                    (place.clone(), values[0], switch_ty, targets)
                }
                _ => continue,
            };
            let (first, rest) = if let ([first], rest) = targets.split_at(1) {
                (*first, rest)
            } else {
                unreachable!();
            };
            let first_dest = bbs[first].terminator().kind.clone();
            let same_destinations = rest
                .iter()
                .map(|target| &bbs[*target].terminator().kind)
                .all(|t_kind| t_kind == &first_dest);
            if !same_destinations {
                continue;
            }
            let first_stmts = &bbs[first].statements;
            for s in first_stmts.iter() {
                match &s.kind {
                    StatementKind::Assign(box (_, rhs)) => {
                        if let Rvalue::Use(Operand::Constant(_)) = rhs {
                        } else {
                            continue 'outer;
                        }
                    }
                    _ => continue 'outer,
                }
            }
            for target in rest.iter() {
                for s in bbs[*target].statements.iter() {
                    if let StatementKind::Assign(box (ref lhs, rhs)) = &s.kind {
                        if let Rvalue::Use(Operand::Constant(_)) = rhs {
                            let has_matching_assn = first_stmts
                                .iter()
                                .find(|s| {
                                    if let StatementKind::Assign(box (lhs_f, _)) = &s.kind {
                                        lhs_f == lhs
                                    } else {
                                        false
                                    }
                                })
                                .is_some();
                            if has_matching_assn {
                                continue;
                            }
                        }
                    }

                    continue 'outer;
                }
            }
            let (first_block, to_add) = bbs.pick2_mut(first, bb_idx);
            let new_stmts = first_block.statements.iter().cloned().map(|mut s| {
                if let StatementKind::Assign(box (_, ref mut rhs)) = s.kind {
                    let size = tcx.layout_of(param_env.and(switch_ty)).unwrap().size;
                    let const_cmp = Operand::const_from_scalar(
                        tcx,
                        switch_ty,
                        crate::interpret::Scalar::from_uint(val, size),
                        rustc_span::DUMMY_SP,
                    );
                    *rhs = Rvalue::BinaryOp(BinOp::Eq, Operand::Move(discr), const_cmp);
                } else {
                    unreachable!()
                }
                s
            });
            to_add.statements.extend(new_stmts);
            to_add.terminator_mut().kind = first_dest;
            did_remove_blocks = true;
        }
        if did_remove_blocks {
            simplify::remove_dead_blocks(body);
        }
    }
}
