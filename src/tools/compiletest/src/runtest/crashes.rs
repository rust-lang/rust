use super::{TestCx, WillExecute};

impl TestCx<'_> {
    pub(super) fn run_crash_test(&self) {
        let pm = self.pass_mode();
        let proc_res = self.compile_test(WillExecute::No, self.should_emit_metadata(pm));

        if std::env::var("COMPILETEST_VERBOSE_CRASHES").is_ok() {
            writeln!(self.stderr, "{}", proc_res.status);
            writeln!(self.stderr, "{}", proc_res.stdout);
            writeln!(self.stderr, "{}", proc_res.stderr);
            writeln!(self.stderr, "{}", proc_res.cmdline);
        }

        // if a test does not crash, consider it an error
        if proc_res.status.success() || matches!(proc_res.status.code(), Some(1 | 0)) {
            self.fatal(&format!(
                "crashtest no longer crashes/triggers ICE, hooray! Please give it a meaningful \
                name, add a doc-comment to the start of the test explaining why it exists and \
                move it to tests/ui or wherever you see fit. Adding 'Fixes #<issueNr>' to your PR \
                description ensures that the corresponding ticket is auto-closed upon merge. \
                If you want to see verbose output, set `COMPILETEST_VERBOSE_CRASHES=1`."
            ));
        }
    }
}
