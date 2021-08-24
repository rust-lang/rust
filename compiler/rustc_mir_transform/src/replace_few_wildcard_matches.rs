//! A pass that replaces the wildcard match with explicit matches if only few (currently: less than
//! two) variants fall under it and replaces the wildcard target with an unreachable basic block.
//! This allows the backend to potentially optimize the switch better.

// FIXME:
// * benchmark
// * determine best location to insert this pass
// * check if <= 2 matches is the best heuristic
// * tests

use crate::MirPass;
use rustc_data_structures::stable_set::FxHashSet;
use rustc_index::vec::Idx;
use rustc_middle::mir::{
    BasicBlock, BasicBlockData, Body, Local, Operand, Rvalue, SourceInfo, StatementKind,
    SwitchTargets, Terminator, TerminatorKind,
};
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_target::abi::{Layout, Variants};

pub struct ReplaceFewWildcardMatches;

fn get_discriminant_local(terminator: &TerminatorKind<'_>) -> Option<Local> {
    if let TerminatorKind::SwitchInt { discr: Operand::Move(p), .. } = terminator {
        p.as_local()
    } else {
        None
    }
}

/// If the basic block terminates by switching on a discriminant, this returns the `Ty` the
/// discriminant is read from. Otherwise, returns None.
fn get_switched_on_type<'tcx>(
    block_data: &BasicBlockData<'tcx>,
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
) -> Option<Ty<'tcx>> {
    let terminator = block_data.terminator();

    // Only bother checking blocks which terminate by switching on a local.
    if let Some(local) = get_discriminant_local(&terminator.kind) {
        let stmt_before_term = (!block_data.statements.is_empty())
            .then(|| &block_data.statements[block_data.statements.len() - 1].kind);

        if let Some(StatementKind::Assign(box (l, Rvalue::Discriminant(place)))) = stmt_before_term
        {
            if l.as_local() == Some(local) {
                let ty = place.ty(body, tcx).ty;
                if ty.is_enum() {
                    return Some(ty);
                }
            }
        }
    }

    None
}

impl<'tcx> MirPass<'tcx> for ReplaceFewWildcardMatches {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if body.source.promoted.is_some() {
            return;
        }

        trace!("ReplaceFewWildcardMatches starting for {:?}", body.source);

        let basic_block_count = body.basic_blocks().len();

        for bb in 0..basic_block_count {
            let bb = BasicBlock::from_usize(bb);
            trace!("processing block {:?}", bb);

            let discriminant_ty =
                if let Some(ty) = get_switched_on_type(&body.basic_blocks()[bb], tcx, body) {
                    ty
                } else {
                    continue;
                };

            let layout = tcx.layout_of(tcx.param_env(body.source.def_id()).and(discriminant_ty));
            let variants = if let Ok(TyAndLayout {
                layout: Layout { variants: Variants::Multiple { variants, .. }, .. },
                ty,
                ..
            }) = layout
            {
                if let ty::Adt(def, _) = *ty.kind() {
                    if def.is_variant_list_non_exhaustive() && !def.did.is_local() {
                        // cannot optimize if externally defined and non-exhaustive
                        continue;
                    }
                }
                variants
            } else {
                continue;
            };

            let mut add_unreachable_bb = false;
            let bb_count = body.basic_blocks().len();
            if let TerminatorKind::SwitchInt { targets, .. } =
                &mut body.basic_blocks_mut()[bb].terminator_mut().kind
            {
                // all_targets() includes the fallback target
                let wildcard_variant_count = variants.len() - (targets.all_targets().len() - 1);
                if 0 < wildcard_variant_count && wildcard_variant_count <= 2 {
                    let mut all_discriminants: FxHashSet<u128> = variants
                        .indices()
                        .map(|idx| discriminant_ty.discriminant_for_variant(tcx, idx).unwrap().val)
                        .collect();
                    trace!(
                        "optimization opportunity found - variants count: {:?}, target count: {:?}",
                        variants.len(),
                        targets.all_targets().len()
                    );
                    for (discriminant, _) in targets.iter() {
                        all_discriminants.remove(&discriminant);
                    }
                    let new_targets = targets.iter().chain(
                        all_discriminants
                            .iter()
                            .map(|discriminant| (*discriminant, targets.otherwise())),
                    );

                    let new_targets = SwitchTargets::new(new_targets, BasicBlock::new(bb_count));
                    *targets = new_targets;
                    add_unreachable_bb = true;
                }
            }
            if add_unreachable_bb {
                let bb = BasicBlockData {
                    statements: Vec::new(),
                    is_cleanup: body.basic_blocks()[bb].is_cleanup,
                    terminator: Some(Terminator {
                        source_info: SourceInfo::outermost(body.span),
                        kind: TerminatorKind::Unreachable,
                    }),
                };
                body.basic_blocks_mut().push(bb);
            }
        }
    }
}
