use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Write;

use rustfix::{Filter, apply_suggestions, get_suggestions_from_json};
use tracing::debug;

use super::{
    AllowUnused, Emit, FailMode, LinkToAux, PassMode, RunFailMode, RunResult, TargetLocation,
    TestCx, TestOutput, Truncated, UI_FIXED, WillExecute,
};
use crate::json;
use crate::runtest::ProcRes;

impl TestCx<'_> {
    pub(super) fn run_ui_test(&self) {
        if let Some(FailMode::Build) = self.props.fail_mode {
            // Make sure a build-fail test cannot fail due to failing analysis (e.g. typeck).
            let pm = Some(PassMode::Check);
            let proc_res =
                self.compile_test_general(WillExecute::No, Emit::Metadata, pm, Vec::new());
            self.check_if_test_should_compile(self.props.fail_mode, pm, &proc_res);
        }

        let pm = self.pass_mode();
        let should_run = self.should_run(pm);
        let emit_metadata = self.should_emit_metadata(pm);
        let proc_res = self.compile_test(should_run, emit_metadata);
        self.check_if_test_should_compile(self.props.fail_mode, pm, &proc_res);
        if matches!(proc_res.truncated, Truncated::Yes)
            && !self.props.dont_check_compiler_stdout
            && !self.props.dont_check_compiler_stderr
        {
            self.fatal_proc_rec(
                "compiler output got truncated, cannot compare with reference file",
                &proc_res,
            );
        }

        // if the user specified a format in the ui test
        // print the output to the stderr file, otherwise extract
        // the rendered error messages from json and print them
        let explicit = self.props.compile_flags.iter().any(|s| s.contains("--error-format"));

        let expected_fixed = self.load_expected_output(UI_FIXED);

        self.check_and_prune_duplicate_outputs(&proc_res, &[], &[]);

        let mut errors = self.load_compare_outputs(&proc_res, TestOutput::Compile, explicit);
        let rustfix_input = json::rustfix_diagnostics_only(&proc_res.stderr);

        if self.config.compare_mode.is_some() {
            // don't test rustfix with nll right now
        } else if self.config.rustfix_coverage {
            // Find out which tests have `MachineApplicable` suggestions but are missing
            // `run-rustfix` or `run-rustfix-only-machine-applicable` directives.
            //
            // This will return an empty `Vec` in case the executed test file has a
            // `compile-flags: --error-format=xxxx` directive with a value other than `json`.
            let suggestions = get_suggestions_from_json(
                &rustfix_input,
                &HashSet::new(),
                Filter::MachineApplicableOnly,
            )
            .unwrap_or_default();
            if !suggestions.is_empty()
                && !self.props.run_rustfix
                && !self.props.rustfix_only_machine_applicable
            {
                let mut coverage_file_path = self.config.build_test_suite_root.clone();
                coverage_file_path.push("rustfix_missing_coverage.txt");
                debug!("coverage_file_path: {}", coverage_file_path);

                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(coverage_file_path.as_path())
                    .expect("could not create or open file");

                if let Err(e) = writeln!(file, "{}", self.testpaths.file) {
                    panic!("couldn't write to {}: {e:?}", coverage_file_path);
                }
            }
        } else if self.props.run_rustfix {
            // Apply suggestions from rustc to the code itself
            let unfixed_code = self.load_expected_output_from_path(&self.testpaths.file).unwrap();
            let suggestions = get_suggestions_from_json(
                &rustfix_input,
                &HashSet::new(),
                if self.props.rustfix_only_machine_applicable {
                    Filter::MachineApplicableOnly
                } else {
                    Filter::Everything
                },
            )
            .unwrap();
            let fixed_code = apply_suggestions(&unfixed_code, &suggestions).unwrap_or_else(|e| {
                panic!(
                    "failed to apply suggestions for {:?} with rustfix: {}",
                    self.testpaths.file, e
                )
            });

            if self
                .compare_output("fixed", &fixed_code, &fixed_code, &expected_fixed)
                .should_error()
            {
                errors += 1;
            }
        } else if !expected_fixed.is_empty() {
            panic!(
                "the `//@ run-rustfix` directive wasn't found but a `*.fixed` \
                 file was found"
            );
        }

        if errors > 0 {
            writeln!(
                self.stdout,
                "To update references, rerun the tests and pass the `--bless` flag"
            );
            let relative_path_to_file =
                self.testpaths.relative_dir.join(self.testpaths.file.file_name().unwrap());
            writeln!(
                self.stdout,
                "To only update this specific test, also pass `--test-args {}`",
                relative_path_to_file,
            );
            self.fatal_proc_rec(
                &format!("{} errors occurred comparing output.", errors),
                &proc_res,
            );
        }

        // If the test is executed, capture its ProcRes separately so that
        // pattern/forbid checks can report the *runtime* stdout/stderr when they fail.
        let mut run_proc_res: Option<ProcRes> = None;
        let output_to_check = if let WillExecute::Yes = should_run {
            let proc_res = self.exec_compiled_test();
            let run_output_errors = if self.props.check_run_results {
                self.load_compare_outputs(&proc_res, TestOutput::Run, explicit)
            } else {
                0
            };
            if run_output_errors > 0 {
                self.fatal_proc_rec(
                    &format!("{} errors occurred comparing run output.", run_output_errors),
                    &proc_res,
                );
            }
            let code = proc_res.status.code();
            let run_result = if proc_res.status.success() {
                RunResult::Pass
            } else if code.is_some_and(|c| c >= 1 && c <= 127) {
                RunResult::Fail
            } else {
                RunResult::Crash
            };
            // Help users understand why the test failed by including the actual
            // exit code and actual run result in the failure message.
            let pass_hint = format!("code={code:?} so test would pass with `{run_result}`");
            if self.should_run_successfully(pm) {
                if run_result != RunResult::Pass {
                    self.fatal_proc_rec(
                        &format!("test did not exit with success! {pass_hint}"),
                        &proc_res,
                    );
                }
            } else if self.props.fail_mode == Some(FailMode::Run(RunFailMode::Fail)) {
                // If the test is marked as `run-fail` but do not support
                // unwinding we allow it to crash, since a panic will trigger an
                // abort (crash) instead of unwind (exit with code 101).
                let crash_ok = !self.config.can_unwind();
                if run_result != RunResult::Fail && !(crash_ok && run_result == RunResult::Crash) {
                    let err = if crash_ok {
                        format!(
                            "test did not exit with failure or crash (`{}` can't unwind)! {pass_hint}",
                            self.config.target
                        )
                    } else {
                        format!("test did not exit with failure! {pass_hint}")
                    };
                    self.fatal_proc_rec(&err, &proc_res);
                }
            } else if self.props.fail_mode == Some(FailMode::Run(RunFailMode::Crash)) {
                if run_result != RunResult::Crash {
                    self.fatal_proc_rec(&format!("test did not crash! {pass_hint}"), &proc_res);
                }
            } else if self.props.fail_mode == Some(FailMode::Run(RunFailMode::FailOrCrash)) {
                if run_result != RunResult::Fail && run_result != RunResult::Crash {
                    self.fatal_proc_rec(
                        &format!("test did not exit with failure or crash! {pass_hint}"),
                        &proc_res,
                    );
                }
            } else {
                unreachable!("run_ui_test() must not be called if the test should not run");
            }

            let output = self.get_output(&proc_res);
            // Move the proc_res into our option after we've extracted output.
            run_proc_res = Some(proc_res);
            output
        } else {
            self.get_output(&proc_res)
        };

        debug!(
            "run_ui_test: explicit={:?} config.compare_mode={:?} \
               proc_res.status={:?} props.error_patterns={:?}",
            explicit, self.config.compare_mode, proc_res.status, self.props.error_patterns
        );

        // Compiler diagnostics (expected errors) are always tied to the compile-time ProcRes.
        self.check_expected_errors(&proc_res);

        // For runtime pattern/forbid checks prefer the executed program's ProcRes if available
        // so that missing pattern failures include the program's stdout/stderr.
        let pattern_proc_res = run_proc_res.as_ref().unwrap_or(&proc_res);
        self.check_all_error_patterns(&output_to_check, pattern_proc_res);
        self.check_forbid_output(&output_to_check, pattern_proc_res);

        if self.props.run_rustfix && self.config.compare_mode.is_none() {
            // And finally, compile the fixed code and make sure it both
            // succeeds and has no diagnostics.
            let mut rustc = self.make_compile_args(
                &self.expected_output_path(UI_FIXED),
                TargetLocation::ThisFile(self.make_exe_name()),
                emit_metadata,
                AllowUnused::No,
                LinkToAux::Yes,
                Vec::new(),
            );

            // If a test is revisioned, it's fixed source file can be named "a.foo.fixed", which,
            // well, "a.foo" isn't a valid crate name. So we explicitly mangle the test name
            // (including the revision) here to avoid the test writer having to manually specify a
            // `#![crate_name = "..."]` as a workaround. This is okay since we're only checking if
            // the fixed code is compilable.
            if self.revision.is_some() {
                let crate_name =
                    self.testpaths.file.file_stem().expect("test must have a file stem");
                // crate name must be alphanumeric or `_`.
                // replace `a.foo` -> `a__foo` for crate name purposes.
                // replace `revision-name-with-dashes` -> `revision_name_with_underscore`
                let crate_name = crate_name.replace('.', "__");
                let crate_name = crate_name.replace('-', "_");
                rustc.arg("--crate-name");
                rustc.arg(crate_name);
            }

            let res = self.compose_and_run_compiler(rustc, None, self.testpaths);
            if !res.status.success() {
                self.fatal_proc_rec("failed to compile fixed code", &res);
            }
            if !res.stderr.is_empty()
                && !self.props.rustfix_only_machine_applicable
                && !json::rustfix_diagnostics_only(&res.stderr).is_empty()
            {
                self.fatal_proc_rec("fixed code is still producing diagnostics", &res);
            }
        }
    }
}
