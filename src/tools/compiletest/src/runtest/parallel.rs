use crate::runtest::{TestCx, WillExecute};

impl TestCx<'_> {
    pub(super) fn run_parallel_test(&self) {
        let pm = self.pass_mode();
        let emit_metadata = self.should_emit_metadata(pm);

        // Repeated testing due to instability in multithreaded environments.
        let iteration_count = self.props.iteration_count.unwrap_or(50);
        for _ in 0..iteration_count {
            let proc_res = self.compile_test(WillExecute::No, emit_metadata);
            // Ensure there is no ICE during parallel complication.
            self.check_no_compiler_crash(&proc_res, false);
        }
    }
}
