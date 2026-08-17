use rustc_middle::mir::{BasicBlock, Body, Operand, Place, TerminatorKind};
use rustc_middle::ty::TyCtxt;
use tracing::instrument;

use crate::MirPass;
use crate::pass_manager::PassPolicy;

pub(super) struct CollapseIdenticalYields;

impl<'tcx> MirPass<'tcx> for CollapseIdenticalYields {
    #[instrument(level = "debug", skip(self, tcx, body), ret)]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if body.coroutine_kind().is_none() {
            return;
        }

        tracing::debug!("running pass for {}", tcx.def_path_str(body.source.def_id()));

        let yields = body
            .basic_blocks
            .iter_enumerated()
            .filter_map(|(bb, bb_data)| match &bb_data.terminator.as_ref()?.kind {
                TerminatorKind::Yield { value, resume, resume_arg, drop } => Some(Yield {
                    basic_block: bb,
                    value: value.clone(),
                    resume: *resume,
                    resume_arg: *resume_arg,
                    drop: *drop,
                }),
                _ => None,
            })
            .collect::<Vec<_>>();

        let resume_loops =
            yields.iter().map(|yield_val| ResumeLoop::search(yield_val, body)).collect::<Vec<_>>();

        tracing::trace!("resume loops: {resume_loops:?}");
    }

    fn policy(&self, _sess: &rustc_session::Session) -> PassPolicy {
        PassPolicy::optimization(true)
    }
}

struct Yield<'tcx> {
    basic_block: BasicBlock,
    /// The value to return.
    value: Operand<'tcx>,
    /// Where to resume to.
    resume: BasicBlock,
    /// The place to store the resume argument in.
    resume_arg: Place<'tcx>,
    /// Cleanup to be done if the coroutine is dropped at this suspend point.
    drop: Option<BasicBlock>,
}

#[derive(Debug)]
struct ResumeLoop {
    inner: Vec<BasicBlock>,
}

impl ResumeLoop {
    fn search(yield_val: &Yield, body: &Body<'_>) -> Option<Self> {
        // FIXME
        None
    }
}
