use std::sync::LazyLock;

use crate::runtest::{Emit, TestCx, WillExecute};
use crate::util::string_enum;

string_enum!(
    /// How far an incremental test revision should proceed through the compile/run
    /// sequence, and whether the last step should succeed or fail, as determined
    /// from the start of the revision name.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum IncrRevKind {
        CheckPass => "cpass",
        BuildFail => "bfail",
        BuildPass => "bpass",
        RunPass => "rpass",
    }
);

impl IncrRevKind {
    fn for_revision_name(rev_name: &str) -> Result<Self, &'static str> {
        static MESSAGE: LazyLock<String> = LazyLock::new(|| {
            let values = IncrRevKind::STR_VARIANTS
                .iter()
                .map(|s| format!("`{s}`"))
                .collect::<Vec<_>>()
                .join(", ");
            format!("incremental revision name must begin with one of: {values}")
        });

        IncrRevKind::VARIANTS
            .iter()
            .copied()
            .find(|kind| rev_name.starts_with(kind.to_str()))
            .ok_or_else(|| MESSAGE.as_str())
    }
}

impl TestCx<'_> {
    /// Runs a single revision of an incremental test.
    pub(super) fn run_incremental_test(&self) {
        let revision = self.revision.expect("incremental tests require a list of revisions");

        // Incremental workproduct directory should have already been created.
        let incremental_dir = self.props.incremental_dir.as_ref().unwrap();
        assert!(incremental_dir.exists(), "init_incremental_test failed to create incremental dir");

        if self.config.verbose {
            write!(self.stdout, "revision={:?} props={:#?}", revision, self.props);
        }

        // Determine the revision kind from the revision name.
        // The revision kind should be matched exhaustively to ensure that no cases are missed.
        let rev_kind = IncrRevKind::for_revision_name(revision).unwrap_or_else(|e| self.fatal(e));

        // Compile the test for this revision.
        let emit = match rev_kind {
            IncrRevKind::CheckPass => Emit::Metadata, // Do a check build.
            IncrRevKind::BuildFail | IncrRevKind::BuildPass | IncrRevKind::RunPass => Emit::None,
        };
        let will_execute = match rev_kind {
            IncrRevKind::CheckPass | IncrRevKind::BuildFail | IncrRevKind::BuildPass => {
                WillExecute::No
            }
            IncrRevKind::RunPass => {
                // Yes, unless running test binaries is disabled.
                self.run_if_enabled()
            }
        };
        let proc_res = &self.compile_test(will_execute, emit);

        // Check the compiler's exit status.
        match rev_kind {
            IncrRevKind::CheckPass | IncrRevKind::BuildPass | IncrRevKind::RunPass => {
                // Compilation should have succeeded.
                if !proc_res.status.success() {
                    self.fatal_proc_rec("test compilation failed although it shouldn't!", proc_res);
                }
            }

            IncrRevKind::BuildFail => {
                // Compilation should have failed, with the expected status code.
                if proc_res.status.success() {
                    self.fatal_proc_rec("incremental test did not emit an error", proc_res);
                }
                if !self.props.dont_check_failure_status {
                    self.check_correct_failure_status(proc_res);
                }
            }
        }

        // Check compilation output.
        let output_to_check = self.get_output(proc_res);
        self.check_expected_errors(&proc_res);
        self.check_all_error_patterns(&output_to_check, proc_res);
        self.check_forbid_output(&output_to_check, proc_res);

        // Run the binary and check its exit status, if appropriate.
        match rev_kind {
            IncrRevKind::CheckPass | IncrRevKind::BuildFail | IncrRevKind::BuildPass => {}
            IncrRevKind::RunPass => {
                if self.config.run_enabled() {
                    let run_proc_res = self.exec_compiled_test();
                    if !run_proc_res.status.success() {
                        self.fatal_proc_rec("test run failed!", &run_proc_res);
                    }
                }
            }
        }
    }
}
