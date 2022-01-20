//! A pass that eliminates branches on uninhabited enum variants.

use crate::MirPass;
use rustc_data_structures::stable_set::FxHashSet;
use rustc_middle::mir::{
    BasicBlockData, Body, Local, Operand, Rvalue, StatementKind, SwitchTargets, TerminatorKind,
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

        for bb in body.basic_blocks().indices() {
            trace!("processing block {:?}", bb);

            let Some(discriminant_ty) = get_switched_on_type(&body.basic_blocks()[bb], tcx, body) else {
                continue;
            };

            let layout = tcx.layout_of(tcx.param_env(body.source.def_id()).and(discriminant_ty));

            let allowed_variants = if let Ok(layout) = layout {
                variant_discriminants(&layout, discriminant_ty, tcx)
            } else {
                continue;
            };

            trace!("allowed_variants = {:?}", allowed_variants);

            if let TerminatorKind::SwitchInt { targets, .. } =
                &mut body.basic_blocks_mut()[bb].terminator_mut().kind
            {
                let new_targets = SwitchTargets::new(
                    targets.iter().filter(|(val, _)| allowed_variants.contains(val)),
                    targets.otherwise(),
                );

                *targets = new_targets;
            } else {
                unreachable!()
            }
        }
    }
}
