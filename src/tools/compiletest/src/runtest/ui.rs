use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Write;

use rustfix::{Filter, apply_suggestions, get_suggestions_from_json};
use tracing::debug;

use super::{
    AllowUnused, Emit, FailMode, LinkToAux, PassMode, RunFailMode, TargetLocation, TestCx,
    TestOutput, Truncated, UI_FIXED, WillExecute,
};
use crate::json;

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
            // `run-rustfix` or `run-rustfix-only-machine-applicable` headers.
            //
            // This will return an empty `Vec` in case the executed test file has a
            // `compile-flags: --error-format=xxxx` header with a value other than `json`.
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
            println!("To update references, rerun the tests and pass the `--bless` flag");
            let relative_path_to_file =
                self.testpaths.relative_dir.join(self.testpaths.file.file_name().unwrap());
            println!(
                "To only update this specific test, also pass `--test-args {}`",
                relative_path_to_file,
            );
            self.fatal_proc_rec(
                &format!("{} errors occurred comparing output.", errors),
                &proc_res,
            );
        }

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
            let exit_with_success = proc_res.status.success();
            // A normal failure exit code is between 1 and 127. If we don't have
            // an exit code or if it is outside that range (and not 0) we
            // consider it a crash.
            let exit_with_failure = code.is_some_and(|c| c >= 1 && c <= 127);
            if self.should_run_successfully(pm) {
                if !exit_with_success {
                    self.fatal_proc_rec(
                        &format!("test did not exit with success! code={code:?}"),
                        &proc_res,
                    );
                }
            } else if self.props.fail_mode == Some(FailMode::Run(RunFailMode::ExitWithFailure)) {
                if !exit_with_failure {
                    self.fatal_proc_rec(
                        &format!("test did not exit with failure! code={code:?}"),
                        &proc_res,
                    );
                }
            } else {
                // If we get here it means we should not run successfully and we
                // should not exit with failure. So we should crash. And if we
                // did exit with success or did exit with failure it means we
                // did NOT crash.
                if exit_with_success || exit_with_failure {
                    self.fatal_proc_rec(&format!("test did not crash! code={code:?}"), &proc_res);
                }
            }

            self.get_output(&proc_res)
        } else {
            self.get_output(&proc_res)
        };

        debug!(
            "run_ui_test: explicit={:?} config.compare_mode={:?} \
               proc_res.status={:?} props.error_patterns={:?}",
            explicit, self.config.compare_mode, proc_res.status, self.props.error_patterns
        );

        self.check_expected_errors(&proc_res);
        self.check_all_error_patterns(&output_to_check, &proc_res);
        self.check_forbid_output(&output_to_check, &proc_res);

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
