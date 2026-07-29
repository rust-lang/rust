use miri::{InterpResult, interp_ok};

use crate::debugger::PrirodaContext;

/// Debug Adapter Protocol frontend.
pub(crate) struct Dap;

impl Dap {
    /// Serve DAP requests on stdin/stdout.
    pub(crate) fn run_dap_loop<'tcx>(
        &self,
        _session: &mut PrirodaContext<'tcx>,
    ) -> InterpResult<'tcx> {
        // FIXME: implement DAP framing and request dispatch on top of PrirodaContext.
        interp_ok(())
    }
}
