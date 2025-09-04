use super::{FailMode, TestCx, WillExecute};
use crate::errors;

impl TestCx<'_> {
    pub(super) fn run_incremental_test(&self) {
        // Basic plan for a test incremental/foo/bar.rs:
        // - load list of revisions rpass1, cfail2, rpass3
        //   - each should begin with `cpass`, `rpass`, `cfail`, or `rfail`
        //   - if `cpass`, expect compilation to succeed, don't execute
        //   - if `rpass`, expect compilation and execution to succeed
        //   - if `cfail`, expect compilation to fail
        //   - if `rfail`, expect compilation to succeed and execution to fail
        // - create a directory build/foo/bar.incremental
        // - compile foo/bar.rs with -C incremental=.../foo/bar.incremental and -C rpass1
        //   - because name of revision starts with "rpass", expect success
        // - compile foo/bar.rs with -C incremental=.../foo/bar.incremental and -C cfail2
        //   - because name of revision starts with "cfail", expect an error
        //   - load expected errors as usual, but filter for those that end in `[rfail2]`
        // - compile foo/bar.rs with -C incremental=.../foo/bar.incremental and -C rpass3
        //   - because name of revision starts with "rpass", expect success
        // - execute build/foo/bar.exe and save output
        //
        // FIXME -- use non-incremental mode as an oracle? That doesn't apply
        // to #[rustc_dirty] and clean tests I guess

        let revision = self.revision.expect("incremental tests require a list of revisions");

        // Incremental workproduct directory should have already been created.
        let incremental_dir = self.props.incremental_dir.as_ref().unwrap();
        assert!(incremental_dir.exists(), "init_incremental_test failed to create incremental dir");

        if self.config.verbose {
            write!(self.stdout, "revision={:?} props={:#?}", revision, self.props);
        }

        if revision.starts_with("cpass") {
            if self.props.should_ice {
                self.fatal("can only use should-ice in cfail tests");
            }
            self.run_cpass_test();
        } else if revision.starts_with("rpass") {
            if self.props.should_ice {
                self.fatal("can only use should-ice in cfail tests");
            }
            self.run_rpass_test();
        } else if revision.starts_with("rfail") {
            if self.props.should_ice {
                self.fatal("can only use should-ice in cfail tests");
            }
            self.run_rfail_test();
        } else if revision.starts_with("cfail") {
            self.run_cfail_test();
        } else {
            self.fatal("revision name must begin with cpass, rpass, rfail, or cfail");
        }
    }

    fn run_cpass_test(&self) {
        let emit_metadata = self.should_emit_metadata(self.pass_mode());
        let proc_res = self.compile_test(WillExecute::No, emit_metadata);

        if !proc_res.status.success() {
            self.fatal_proc_rec("compilation failed!", &proc_res);
        }

        // FIXME(#41968): Move this check to tidy?
        if !errors::load_errors(&self.testpaths.file, self.revision).is_empty() {
            self.fatal("build-pass tests with expected warnings should be moved to ui/");
        }
    }

    fn run_rpass_test(&self) {
        let emit_metadata = self.should_emit_metadata(self.pass_mode());
        let should_run = self.run_if_enabled();
        let proc_res = self.compile_test(should_run, emit_metadata);

        if !proc_res.status.success() {
            self.fatal_proc_rec("compilation failed!", &proc_res);
        }

        // FIXME(#41968): Move this check to tidy?
        if !errors::load_errors(&self.testpaths.file, self.revision).is_empty() {
            self.fatal("run-pass tests with expected warnings should be moved to ui/");
        }

        if let WillExecute::Disabled = should_run {
            return;
        }

        let proc_res = self.exec_compiled_test();
        if !proc_res.status.success() {
            self.fatal_proc_rec("test run failed!", &proc_res);
        }
    }

    fn run_cfail_test(&self) {
        let pm = self.pass_mode();
        let proc_res = self.compile_test(WillExecute::No, self.should_emit_metadata(pm));
        self.check_if_test_should_compile(Some(FailMode::Build), pm, &proc_res);
        self.check_no_compiler_crash(&proc_res, self.props.should_ice);

        let output_to_check = self.get_output(&proc_res);
        self.check_expected_errors(&proc_res);
        self.check_all_error_patterns(&output_to_check, &proc_res);
        if self.props.should_ice {
            match proc_res.status.code() {
                Some(101) => (),
                _ => self.fatal("expected ICE"),
            }
        }

        self.check_forbid_output(&output_to_check, &proc_res);
    }

    fn run_rfail_test(&self) {
        let pm = self.pass_mode();
        let should_run = self.run_if_enabled();
        let proc_res = self.compile_test(should_run, self.should_emit_metadata(pm));

        if !proc_res.status.success() {
            self.fatal_proc_rec("compilation failed!", &proc_res);
        }

        if let WillExecute::Disabled = should_run {
            return;
        }

        let proc_res = self.exec_compiled_test();

        let output_to_check = self.get_output(&proc_res);
        self.check_correct_failure_status(&proc_res);
        self.check_all_error_patterns(&output_to_check, &proc_res);
    }
}
