use crate::MirPass;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use std::iter;

use super::simplify::simplify_cfg;

pub struct MatchBranchSimplification;

/// If a source block is found that switches between two blocks that are exactly
/// the same modulo const bool assignments (e.g., one assigns true another false
/// to the same place), merge a target block statements into the source block,
/// using Eq / Ne comparison with switch value where const bools value differ.
///
/// For example:
///
/// ```rust
/// bb0: {
///     switchInt(move _3) -> [42_isize: bb1, otherwise: bb2];
/// }
///
/// bb1: {
///     _2 = const true;
///     goto -> bb3;
/// }
///
/// bb2: {
///     _2 = const false;
///     goto -> bb3;
/// }
/// ```
///
/// into:
///
/// ```rust
/// bb0: {
///    _2 = Eq(move _3, const 42_isize);
///    goto -> bb3;
/// }
/// ```

impl<'tcx> MirPass<'tcx> for MatchBranchSimplification {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 3
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let def_id = body.source.def_id();
        let param_env = tcx.param_env(def_id);

        let (bbs, local_decls) = body.basic_blocks_and_local_decls_mut();
        let mut should_cleanup = false;
        'outer: for bb_idx in bbs.indices() {
            if !tcx.consider_optimizing(|| format!("MatchBranchSimplification {:?} ", def_id)) {
                continue;
            }

            let (discr, val, switch_ty, first, second) = match bbs[bb_idx].terminator().kind {
                TerminatorKind::SwitchInt {
                    discr: ref discr @ (Operand::Copy(_) | Operand::Move(_)),
                    switch_ty,
                    ref targets,
                    ..
                } if targets.iter().len() == 1 => {
                    let (value, target) = targets.iter().next().unwrap();
                    if target == targets.otherwise() {
                        continue;
                    }
                    (discr, value, switch_ty, target, targets.otherwise())
                }
                // Only optimize switch int statements
                _ => continue,
            };

            // Check that destinations are identical, and if not, then don't optimize this block
            if bbs[first].terminator().kind != bbs[second].terminator().kind {
                continue;
            }

            // Check that blocks are assignments of consts to the same place or same statement,
            // and match up 1-1, if not don't optimize this block.
            let first_stmts = &bbs[first].statements;
            let scnd_stmts = &bbs[second].statements;
            if first_stmts.len() != scnd_stmts.len() {
                continue;
            }
            for (f, s) in iter::zip(first_stmts, scnd_stmts) {
                match (&f.kind, &s.kind) {
                    // If two statements are exactly the same, we can optimize.
                    (f_s, s_s) if f_s == s_s => {}

                    // If two statements are const bool assignments to the same place, we can optimize.
                    (
                        StatementKind::Assign(box (lhs_f, Rvalue::Use(Operand::Constant(f_c)))),
                        StatementKind::Assign(box (lhs_s, Rvalue::Use(Operand::Constant(s_c)))),
                    ) if lhs_f == lhs_s
                        && f_c.literal.ty().is_bool()
                        && s_c.literal.ty().is_bool()
                        && f_c.literal.try_eval_bool(tcx, param_env).is_some()
                        && s_c.literal.try_eval_bool(tcx, param_env).is_some() => {}

                    // Otherwise we cannot optimize. Try another block.
                    _ => continue 'outer,
                }
            }
            // Take ownership of items now that we know we can optimize.
            let discr = discr.clone();

            // Introduce a temporary for the discriminant value.
            let source_info = bbs[bb_idx].terminator().source_info;
            let discr_local = local_decls.push(LocalDecl::new(switch_ty, source_info.span));

            // We already checked that first and second are different blocks,
            // and bb_idx has a different terminator from both of them.
            let (from, first, second) = bbs.pick3_mut(bb_idx, first, second);

            let new_stmts = iter::zip(&first.statements, &second.statements).map(|(f, s)| {
                match (&f.kind, &s.kind) {
                    (f_s, s_s) if f_s == s_s => (*f).clone(),

                    (
                        StatementKind::Assign(box (lhs, Rvalue::Use(Operand::Constant(f_c)))),
                        StatementKind::Assign(box (_, Rvalue::Use(Operand::Constant(s_c)))),
                    ) => {
                        // From earlier loop we know that we are dealing with bool constants only:
                        let f_b = f_c.literal.try_eval_bool(tcx, param_env).unwrap();
                        let s_b = s_c.literal.try_eval_bool(tcx, param_env).unwrap();
                        if f_b == s_b {
                            // Same value in both blocks. Use statement as is.
                            (*f).clone()
                        } else {
                            // Different value between blocks. Make value conditional on switch condition.
                            let size = tcx.layout_of(param_env.and(switch_ty)).unwrap().size;
                            let const_cmp = Operand::const_from_scalar(
                                tcx,
                                switch_ty,
                                rustc_const_eval::interpret::Scalar::from_uint(val, size),
                                rustc_span::DUMMY_SP,
                            );
                            let op = if f_b { BinOp::Eq } else { BinOp::Ne };
                            let rhs = Rvalue::BinaryOp(
                                op,
                                Box::new((Operand::Copy(Place::from(discr_local)), const_cmp)),
                            );
                            Statement {
                                source_info: f.source_info,
                                kind: StatementKind::Assign(Box::new((*lhs, rhs))),
                            }
                        }
                    }

                    _ => unreachable!(),
                }
            });

            from.statements
                .push(Statement { source_info, kind: StatementKind::StorageLive(discr_local) });
            from.statements.push(Statement {
                source_info,
                kind: StatementKind::Assign(Box::new((
                    Place::from(discr_local),
                    Rvalue::Use(discr),
                ))),
            });
            from.statements.extend(new_stmts);
            from.statements
                .push(Statement { source_info, kind: StatementKind::StorageDead(discr_local) });
            from.terminator_mut().kind = first.terminator().kind.clone();
            should_cleanup = true;
        }

        if should_cleanup {
            simplify_cfg(tcx, body);
        }
    }
}
