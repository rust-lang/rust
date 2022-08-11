//! This pass inserts the same validity checks into `MaybeUninit::{uninit,zeroed}().assert_init()`
//! as in `mem::{uninitialized,zeroed}`.
//!
//! Note that this module uses `uninit` to mean `uninit` or `zeroed` unless `zeroed` is used explicitly.
//!
//! It does this by first finding a call to `MaybeUninit::uninit`, and then figuring out
//! whether the successor basic block is a call to `MaybeUninit::assume_init` on the same local.

use rustc_const_eval::interpret;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::patch::MirPatch;
use rustc_middle::mir::{
    BasicBlock, BasicBlockData, Body, Constant, ConstantKind, Operand, Place, SourceInfo,
    Terminator, TerminatorKind,
};
use rustc_middle::ty::{self, List, SubstsRef, TyCtxt};
use rustc_span::{sym, Span};

use crate::MirPass;

pub struct CheckMaybeUninit;

impl<'tcx> MirPass<'tcx> for CheckMaybeUninit {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let mut patch = MirPatch::new(body);

        for (mu_uninit_bb, _) in body.basic_blocks.iter_enumerated() {
            let terminator = body.basic_blocks[mu_uninit_bb].terminator();

            let TerminatorKind::Call {
                func: mu_uninit_func,
                target: assume_init_bb,
                destination: uninit_place,
                 ..
            } = &terminator.kind else {
                continue;
            };

            let Some((mu_method_def_id, substs)) = mu_uninit_func.const_fn_def() else {
                continue;
            };

            let Some(assume_init_bb) = assume_init_bb else {
                continue;
            };

            let Some((assume_init_operand, assume_init_call_span)) = is_block_just_assume_init(tcx, &body.basic_blocks[*assume_init_bb]) else {
                continue;
            };

            let Some(assume_init_place) = assume_init_operand.place() else {
                continue;
            };

            if assume_init_place != *uninit_place {
                // The calls here are a little sketchy, but the place that is assumed to be init is not the place that was just crated
                // as uninit, so we conservatively bail out.
                continue;
            }

            // Select the right assertion intrinsic to call depending on which MaybeUninit method we called
            let Some(init_check_def_id) = get_init_check_def_id(tcx, mu_method_def_id) else {
                continue;
            };

            let assert_valid_bb = make_assert_valid_bb(
                &mut patch,
                tcx,
                assume_init_call_span,
                init_check_def_id,
                *assume_init_bb,
                substs,
            );

            let mut new_uninit_terminator = terminator.kind.clone();
            match new_uninit_terminator {
                TerminatorKind::Call { ref mut target, .. } => {
                    *target = Some(assert_valid_bb);
                }
                _ => unreachable!("terminator must be TerminatorKind::Call as checked above"),
            }

            patch.patch_terminator(mu_uninit_bb, new_uninit_terminator);
        }

        patch.apply(body);
    }
}

fn is_block_just_assume_init<'tcx, 'blk>(
    tcx: TyCtxt<'tcx>,
    block: &'blk BasicBlockData<'tcx>,
) -> Option<(&'blk Operand<'tcx>, Span)> {
    if block.statements.is_empty()
        && let TerminatorKind::Call {
            func,
            args,
             fn_span,
             ..
        } = &block.terminator().kind
        && let Some((def_id, _)) = func.const_fn_def()
        && tcx.is_diagnostic_item(sym::assume_init, def_id)
    {
        args.get(0).map(|operand| (operand, *fn_span))
    } else {
        None
    }
}

fn get_init_check_def_id(tcx: TyCtxt<'_>, mu_method_def_id: DefId) -> Option<DefId> {
    if tcx.is_diagnostic_item(sym::maybe_uninit_uninit, mu_method_def_id) {
        tcx.lang_items().assert_uninit_valid()
    } else if tcx.is_diagnostic_item(sym::maybe_uninit_zeroed, mu_method_def_id) {
        tcx.lang_items().assert_zero_valid()
    } else {
        None
    }
}

fn make_assert_valid_bb<'tcx>(
    patch: &mut MirPatch<'tcx>,
    tcx: TyCtxt<'tcx>,
    fn_span: Span,
    init_check_def_id: DefId,
    target_bb: BasicBlock,
    substs: SubstsRef<'tcx>,
) -> BasicBlock {
    let func = make_fn_operand_for_assert_valid(tcx, init_check_def_id, fn_span, substs);

    let local = patch.new_temp(tcx.types.unit, fn_span);

    let terminator = TerminatorKind::Call {
        func,
        args: vec![],
        destination: Place { local, projection: List::empty() },
        target: Some(target_bb),
        cleanup: Some(patch.resume_block()),
        from_hir_call: true,
        fn_span,
    };

    let terminator = Terminator { source_info: SourceInfo::outermost(fn_span), kind: terminator };

    let bb_data = BasicBlockData::new(Some(terminator));

    let block = patch.new_block(bb_data);
    block
}

fn make_fn_operand_for_assert_valid<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    span: Span,
    substs: SubstsRef<'tcx>,
) -> Operand<'tcx> {
    let fn_ty = ty::FnDef(def_id, substs);
    let fn_ty = tcx.mk_ty(fn_ty);

    Operand::Constant(Box::new(Constant {
        span,
        literal: ConstantKind::Val(interpret::ConstValue::ZeroSized, fn_ty),
        user_ty: None,
    }))
}
