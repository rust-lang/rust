use tracing::debug;

use crate::concurrency::thread::EvalContextExt as _;
use crate::{
    BlockReason, InterpResult, MachineCallback, MiriInterpCx, OpTy, UnblockKind, VisitProvenance,
    VisitWith, callback, interp_ok,
};

// Handling of code intercepted by Miri in GenMC mode, such as assume statement or `std::sync::Mutex`.

/// Other functionality not directly related to event handling
impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Handle an `assume` statement. This will tell GenMC to block the current thread if the `condition` is false.
    /// Returns `true` if the current thread should be blocked in Miri too.
    fn handle_genmc_verifier_assume(&mut self, condition: &OpTy<'tcx>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let condition_bool = this.read_scalar(condition)?.to_bool()?;
        debug!("GenMC: handle_genmc_verifier_assume, condition: {condition:?} = {condition_bool}");
        if condition_bool {
            return interp_ok(());
        }
        let genmc_ctx = this.machine.data_race.as_genmc_ref().unwrap();
        genmc_ctx.handle_assume_block(&this.machine)?;
        this.block_thread(
            BlockReason::Genmc,
            None,
            callback!(
                @capture<'tcx> {}
                |_this, unblock: UnblockKind| {
                    assert_eq!(unblock, UnblockKind::Ready);
                    unreachable!("GenMC should never unblock a thread blocked by an `assume`.");
                }
            ),
        );
        interp_ok(())
    }
}
