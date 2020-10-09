use crate::transform::MirPass;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_target::abi::VariantIdx;

pub struct MatchIdentitySimplification;

impl<'tcx> MirPass<'tcx> for MatchIdentitySimplification {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        //let param_env = tcx.param_env(body.source.def_id());
        let (bbs, local_decls) = body.basic_blocks_and_local_decls_mut();
        for bb_idx in bbs.indices() {
            let (read_discr, og_match) = match &bbs[bb_idx].statements[..] {
                &[Statement {
                    kind: StatementKind::Assign(box (dst, Rvalue::Discriminant(src))),
                    ..
                }] => (dst, src),
                _ => continue,
            };
            let (var_idx, fst, snd) = match bbs[bb_idx].terminator().kind {
                TerminatorKind::SwitchInt {
                    discr: Operand::Copy(ref place) | Operand::Move(ref place),
                    ref targets,
                    ref values,
                    ..
                } if targets.len() == 2
                    && values.len() == 1
                    && targets[0] != targets[1]
                    // check that we're switching on the read discr
                    && place == &read_discr
                    // check that this is actually
                    && place.ty(local_decls, tcx).ty.is_enum() =>
                {
                    (VariantIdx::from(values[0] as usize), targets[0], targets[1])
                }
                // Only optimize switch int statements
                _ => continue,
            };
            let stmts_ok = |stmts: &[Statement<'_>], expected_variant| match stmts {
                [Statement {
                    kind:
                        StatementKind::Assign(box (
                            dst0,
                            Rvalue::Use(Operand::Copy(from) | Operand::Move(from)),
                        )),
                    ..
                }, Statement {
                    kind: StatementKind::SetDiscriminant { place: box dst1, variant_index },
                    ..
                }] => *variant_index == expected_variant && dst0 == dst1 && og_match == *from,
                _ => false,
            };
            let bb1 = &bbs[fst];
            let bb2 = &bbs[snd];
            if bb1.terminator().kind != bb2.terminator().kind
                || stmts_ok(&bb1.statements[..], var_idx)
                || stmts_ok(&bb2.statements[..], var_idx + 1)
            {
                continue;
            }
            let dst = match (&bb1.statements[0], &bb2.statements[0]) {
                (
                    Statement { kind: StatementKind::Assign(box (dst0, _)), .. },
                    Statement { kind: StatementKind::Assign(box (dst1, _)), .. },
                ) if dst0 == dst1 => dst0.clone(),
                _ => continue,
            };
            let term_kind = bb1.terminator().kind.clone();
            // Reassign the output to just be the original
            // Replace the terminator with the terminator of the output
            bbs[bb_idx].statements[0].kind =
                StatementKind::Assign(box (dst, Rvalue::Use(Operand::Copy(og_match))));
            bbs[bb_idx].terminator_mut().kind = term_kind;
        }
    }
}
