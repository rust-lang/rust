//! A pass that eliminates branches on uninhabited enum variants.

use crate::transform::{MirPass, MirSource};
use rustc_middle::mir::{
    BasicBlock, BasicBlockData, Body, BodyAndCache, Local, Operand, Rvalue, StatementKind,
    TerminatorKind,
};
use rustc_middle::ty::layout::{Abi, TyAndLayout, Variants};
use rustc_middle::ty::{Ty, TyCtxt};

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
                if let Some(r_local) = place.as_local() {
                    let ty = body.local_decls[r_local].ty;

                    if ty.is_enum() {
                        return Some(ty);
                    }
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
) -> Vec<u128> {
    match &layout.variants {
        Variants::Single { index } => vec![index.as_u32() as u128],
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
    fn run_pass(&self, tcx: TyCtxt<'tcx>, source: MirSource<'tcx>, body: &mut BodyAndCache<'tcx>) {
        if source.promoted.is_some() {
            return;
        }

        trace!("UninhabitedEnumBranching starting for {:?}", source);

        let basic_block_count = body.basic_blocks().len();

        for bb in 0..basic_block_count {
            let bb = BasicBlock::from_usize(bb);
            trace!("processing block {:?}", bb);

            let discriminant_ty =
                if let Some(ty) = get_switched_on_type(&body.basic_blocks()[bb], body) {
                    ty
                } else {
                    continue;
                };

            let layout = tcx.layout_of(tcx.param_env(source.def_id()).and(discriminant_ty));

            let allowed_variants = if let Ok(layout) = layout {
                variant_discriminants(&layout, discriminant_ty, tcx)
            } else {
                continue;
            };

            trace!("allowed_variants = {:?}", allowed_variants);

            if let TerminatorKind::SwitchInt { values, targets, .. } =
                &mut body.basic_blocks_mut()[bb].terminator_mut().kind
            {
                let vals = &*values;
                let zipped = vals.iter().zip(targets.iter());

                let mut matched_values = Vec::with_capacity(allowed_variants.len());
                let mut matched_targets = Vec::with_capacity(allowed_variants.len() + 1);

                for (val, target) in zipped {
                    if allowed_variants.contains(val) {
                        matched_values.push(*val);
                        matched_targets.push(*target);
                    } else {
                        trace!("eliminating {:?} -> {:?}", val, target);
                    }
                }

                // handle the "otherwise" branch
                matched_targets.push(targets.pop().unwrap());

                *values = matched_values.into();
                *targets = matched_targets;
            } else {
                unreachable!()
            }
        }
    }
}
