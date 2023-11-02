//! A pass that eliminates branches on uninhabited enum variants.

use crate::MirPass;
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::mir::{
    BasicBlockData, Body, Local, Operand, Rvalue, StatementKind, Terminator, TerminatorKind,
};
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{Ty, TyCtxt};
use rustc_target::abi::{Abi, Variants};

pub struct UninhabitedEnumBranching;

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
    let local = get_discriminant_local(&terminator.kind)?;

    let stmt_before_term = block_data.statements.last()?;

    if let StatementKind::Assign(box (l, Rvalue::Discriminant(place))) = stmt_before_term.kind
        && l.as_local() == Some(local)
    {
        let ty = place.ty(body, tcx).ty;
        if ty.is_enum() {
            return Some(ty);
        }
    }

    None
}

fn variant_discriminants<'tcx>(
    layout: &TyAndLayout<'tcx>,
    ty: Ty<'tcx>,
    tcx: TyCtxt<'tcx>,
) -> FxHashSet<u128> {
    match &layout.variants {
        Variants::Single { index } => {
            let mut res = FxHashSet::default();
            res.insert(
                ty.discriminant_for_variant(tcx, *index)
                    .map_or(index.as_u32() as u128, |discr| discr.val),
            );
            res
        }
        Variants::Multiple { variants, .. } => variants
            .iter_enumerated()
            .filter_map(|(idx, layout)| {
                (layout.abi != Abi::Uninhabited)
                    .then(|| ty.discriminant_for_variant(tcx, idx).unwrap().val)
            })
            .collect(),
    }
}

impl<'tcx> MirPass<'tcx> for UninhabitedEnumBranching {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 0
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        trace!("UninhabitedEnumBranching starting for {:?}", body.source);

        let mut removable_switchs = Vec::new();

        for (bb, bb_data) in body.basic_blocks.iter_enumerated() {
            trace!("processing block {:?}", bb);

            if bb_data.is_cleanup {
                continue;
            }

            let Some(discriminant_ty) = get_switched_on_type(&bb_data, tcx, body) else { continue };

            let layout = tcx.layout_of(
                tcx.param_env_reveal_all_normalized(body.source.def_id()).and(discriminant_ty),
            );

            let allowed_variants = if let Ok(layout) = layout {
                variant_discriminants(&layout, discriminant_ty, tcx)
            } else {
                continue;
            };

            trace!("allowed_variants = {:?}", allowed_variants);

            let terminator = bb_data.terminator();
            let TerminatorKind::SwitchInt { targets, .. } = &terminator.kind else { bug!() };

            let mut reachable_count = 0;
            for (index, (val, _)) in targets.iter().enumerate() {
                if allowed_variants.contains(&val) {
                    reachable_count += 1;
                } else {
                    removable_switchs.push((bb, index));
                }
            }

            if reachable_count == allowed_variants.len() {
                removable_switchs.push((bb, targets.iter().count()));
            }
        }

        if removable_switchs.is_empty() {
            return;
        }

        let new_block = BasicBlockData::new(Some(Terminator {
            source_info: body.basic_blocks[removable_switchs[0].0].terminator().source_info,
            kind: TerminatorKind::Unreachable,
        }));
        let unreachable_block = body.basic_blocks.as_mut().push(new_block);

        for (bb, index) in removable_switchs {
            let bb = &mut body.basic_blocks.as_mut()[bb];
            let terminator = bb.terminator_mut();
            let TerminatorKind::SwitchInt { targets, .. } = &mut terminator.kind else { bug!() };
            targets.all_targets_mut()[index] = unreachable_block;
        }
    }
}
