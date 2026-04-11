//! This pass implements a part of the MIR semantics: when invoking a function with the
//! `#[rustc_panic_entrypoint]` attribute, if the current panic strategy is immediate-abort,
//! instead of executing the function body we just abort the program.

use rustc_hir::LangItem;
use rustc_middle::mir::{Const, ConstValue, UnwindAction, *};
use rustc_middle::ty::{self, InstanceKind, Ty, TyCtxt};
use rustc_span::Span;
use rustc_target::spec::PanicStrategy;
use tracing::{debug, instrument};

pub(super) struct PanicEntrypoints;

impl<'tcx> crate::MirPass<'tcx> for PanicEntrypoints {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.panic_strategy() == PanicStrategy::ImmediateAbort
    }

    #[instrument(level = "trace", skip(self, tcx, body))]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if body.source.instance.is_panic_entrypoint(tcx) {
            debug!("Replacing body of {:?}", body.source);
            let blocks = body.basic_blocks.as_mut();
            blocks.raw.clear();
            blocks.push(BasicBlockData::new(Some(abort_terminator(tcx, body.span)), false));
            return;
        }

        for bb in body.basic_blocks.as_mut().iter_mut() {
            let terminator = bb.terminator.as_mut().expect("invalid terminator");
            let TerminatorKind::Call { func, .. } = &mut terminator.kind else {
                continue;
            };
            let Some((def_id, _args)) = func.const_fn_def() else {
                continue;
            };
            if InstanceKind::Item(def_id).is_panic_entrypoint(tcx) {
                debug!("Remapping call from {:?} to {:?}", body.source, def_id);
                *terminator = abort_terminator(tcx, terminator.source_info.span);
            }
        }
    }

    fn is_required(&self) -> bool {
        true
    }
}

fn abort_terminator<'tcx>(tcx: TyCtxt<'tcx>, span: Span) -> Terminator<'tcx> {
    let abort_intrin = tcx.require_lang_item(LangItem::AbortIntrinsic, span);
    let no_args: [ty::GenericArg<'_>; 0] = [];
    let func = Operand::Constant(Box::new(ConstOperand {
        span,
        user_ty: None,
        const_: Const::Val(ConstValue::ZeroSized, Ty::new_fn_def(tcx, abort_intrin, no_args)),
    }));
    Terminator {
        source_info: SourceInfo::outermost(span),
        kind: TerminatorKind::Call {
            func,
            args: Box::new([]),
            destination: Place::return_place(),
            target: None,
            unwind: UnwindAction::Unreachable,
            call_source: CallSource::Misc,
            fn_span: span,
        },
    }
}
