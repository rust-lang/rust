use super::{Emit, TestCx, WillExecute};

impl TestCx<'_> {
    pub(super) fn run_valgrind_test(&self) {
        assert!(self.revision.is_none(), "revisions not relevant here");

        // FIXME(jieyouxu): does this really make any sense? If a valgrind test isn't testing
        // valgrind, what is it even testing?
        if self.config.valgrind_path.is_none() {
            assert!(!self.config.force_valgrind);
            return self.run_rpass_test();
        }

        let should_run = self.run_if_enabled();
        let mut proc_res = self.compile_test(should_run, Emit::None);

        if !proc_res.status.success() {
            self.fatal_proc_rec("compilation failed!", &proc_res);
        }

        if let WillExecute::Disabled = should_run {
            return;
        }

        let mut new_config = self.config.clone();
        new_config.runner = new_config.valgrind_path.clone();
        let new_cx = TestCx { config: &new_config, ..*self };
        proc_res = new_cx.exec_compiled_test();

        if !proc_res.status.success() {
            self.fatal_proc_rec("test run failed!", &proc_res);
        }
    }
}
