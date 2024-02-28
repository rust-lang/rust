//! A pass that eliminates branches on uninhabited enum variants.

use crate::MirPass;
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::mir::patch::MirPatch;
use rustc_middle::mir::{
    BasicBlock, BasicBlockData, BasicBlocks, Body, Local, Operand, Rvalue, StatementKind,
    TerminatorKind,
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

        let mut unreachable_targets = Vec::new();
        let mut patch = MirPatch::new(body);

        for (bb, bb_data) in body.basic_blocks.iter_enumerated() {
            trace!("processing block {:?}", bb);

            if bb_data.is_cleanup {
                continue;
            }

            let Some(discriminant_ty) = get_switched_on_type(bb_data, tcx, body) else { continue };

            let layout = tcx.layout_of(
                tcx.param_env_reveal_all_normalized(body.source.def_id()).and(discriminant_ty),
            );

            let mut allowed_variants = if let Ok(layout) = layout {
                variant_discriminants(&layout, discriminant_ty, tcx)
            } else if let Some(variant_range) = discriminant_ty.variant_range(tcx) {
                variant_range
                    .map(|variant| {
                        discriminant_ty.discriminant_for_variant(tcx, variant).unwrap().val
                    })
                    .collect()
            } else {
                continue;
            };

            trace!("allowed_variants = {:?}", allowed_variants);

            unreachable_targets.clear();
            let TerminatorKind::SwitchInt { targets, discr } = &bb_data.terminator().kind else {
                bug!()
            };

            for (index, (val, _)) in targets.iter().enumerate() {
                if !allowed_variants.remove(&val) {
                    unreachable_targets.push(index);
                }
            }
            let otherwise_is_empty_unreachable =
                body.basic_blocks[targets.otherwise()].is_empty_unreachable();
            // After resolving https://github.com/llvm/llvm-project/issues/78578,
            // we can remove the limit on the number of successors.
            fn check_successors(basic_blocks: &BasicBlocks<'_>, bb: BasicBlock) -> bool {
                let mut successors = basic_blocks[bb].terminator().successors();
                let Some(first_successor) = successors.next() else { return true };
                if successors.next().is_some() {
                    return true;
                }
                if let TerminatorKind::SwitchInt { .. } =
                    &basic_blocks[first_successor].terminator().kind
                {
                    return false;
                };
                true
            }
            let otherwise_is_last_variant = !otherwise_is_empty_unreachable
                && allowed_variants.len() == 1
                && check_successors(&body.basic_blocks, targets.otherwise());
            let replace_otherwise_to_unreachable = otherwise_is_last_variant
                || !otherwise_is_empty_unreachable && allowed_variants.is_empty();

            if unreachable_targets.is_empty() && !replace_otherwise_to_unreachable {
                continue;
            }

            let unreachable_block = patch.unreachable_no_cleanup_block();
            let mut targets = targets.clone();
            if replace_otherwise_to_unreachable {
                if otherwise_is_last_variant {
                    #[allow(rustc::potential_query_instability)]
                    let last_variant = *allowed_variants.iter().next().unwrap();
                    targets.add_target(last_variant, targets.otherwise());
                }
                unreachable_targets.push(targets.iter().count());
            }
            for index in unreachable_targets.iter() {
                targets.all_targets_mut()[*index] = unreachable_block;
            }
            patch.patch_terminator(bb, TerminatorKind::SwitchInt { targets, discr: discr.clone() });
        }

        patch.apply(body);
    }
}
