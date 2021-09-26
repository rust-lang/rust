use super::*;

use crate::common::Debugger;
use crate::common::UI_FIXED;
use crate::common::{CompareMode, FailMode, PassMode};
use crate::errors::{self};
use crate::json;
use crate::util::logv;
use rustfix::{apply_suggestions, get_suggestions_from_json, Filter};

use std::collections::HashSet;
use std::env;
use std::fs::{self, create_dir_all, OpenOptions};
use std::path::Path;
use std::process::{Command, Stdio};

impl TestCx<'_> {
    pub(super) fn run_cfail_test(&self) {
        let pm = self.pass_mode();
        let proc_res = self.compile_test(WillExecute::No, self.should_emit_metadata(pm));
        self.check_if_test_should_compile(&proc_res, pm);
        self.check_no_compiler_crash(&proc_res, self.props.should_ice);

        let output_to_check = self.get_output(&proc_res);
        let expected_errors = errors::load_errors(&self.testpaths.file, self.revision);
        if !expected_errors.is_empty() {
            if !self.props.error_patterns.is_empty() {
                self.fatal("both error pattern and expected errors specified");
            }
            self.check_expected_errors(expected_errors, &proc_res);
        } else {
            self.check_error_patterns(&output_to_check, &proc_res, pm);
        }
        if self.props.should_ice {
            match proc_res.status.code() {
                Some(101) => (),
                _ => self.fatal("expected ICE"),
            }
        }

        self.check_forbid_output(&output_to_check, &proc_res);
    }

    pub(super) fn run_rfail_test(&self) {
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

        // The value our Makefile configures valgrind to return on failure
        const VALGRIND_ERR: i32 = 100;
        if proc_res.status.code() == Some(VALGRIND_ERR) {
            self.fatal_proc_rec("run-fail test isn't valgrind-clean!", &proc_res);
        }

        let output_to_check = self.get_output(&proc_res);
        self.check_correct_failure_status(&proc_res);
        self.check_error_patterns(&output_to_check, &proc_res, pm);
    }

    pub(super) fn run_rpass_test(&self) {
        let emit_metadata = self.should_emit_metadata(self.pass_mode());
        let should_run = self.run_if_enabled();
        let proc_res = self.compile_test(should_run, emit_metadata);

        if !proc_res.status.success() {
            self.fatal_proc_rec("compilation failed!", &proc_res);
        }

        if let WillExecute::Disabled = should_run {
            return;
        }

        // FIXME(#41968): Move this check to tidy?
        let expected_errors = errors::load_errors(&self.testpaths.file, self.revision);
        assert!(
            expected_errors.is_empty(),
            "run-pass tests with expected warnings should be moved to ui/"
        );

        let proc_res = self.exec_compiled_test();
        if !proc_res.status.success() {
            self.fatal_proc_rec("test run failed!", &proc_res);
        }
    }

    pub(super) fn run_valgrind_test(&self) {
        assert!(self.revision.is_none(), "revisions not relevant here");

        if self.config.valgrind_path.is_none() {
            assert!(!self.config.force_valgrind);
            return self.run_rpass_test();
        }

        let should_run = self.run_if_enabled();
        let mut proc_res = self.compile_test(should_run, EmitMetadata::No);

        if !proc_res.status.success() {
            self.fatal_proc_rec("compilation failed!", &proc_res);
        }

        if let WillExecute::Disabled = should_run {
            return;
        }

        let mut new_config = self.config.clone();
        new_config.runtool = new_config.valgrind_path.clone();
        let new_cx = TestCx { config: &new_config, ..*self };
        proc_res = new_cx.exec_compiled_test();

        if !proc_res.status.success() {
            self.fatal_proc_rec("test run failed!", &proc_res);
        }
    }

    pub(super) fn run_pretty_test(&self) {
        if self.props.pp_exact.is_some() {
            logv(self.config, "testing for exact pretty-printing".to_owned());
        } else {
            logv(self.config, "testing for converging pretty-printing".to_owned());
        }

        let rounds = match self.props.pp_exact {
            Some(_) => 1,
            None => 2,
        };

        let src = fs::read_to_string(&self.testpaths.file).unwrap();
        let mut srcs = vec![src];

        let mut round = 0;
        while round < rounds {
            logv(
                self.config,
                format!("pretty-printing round {} revision {:?}", round, self.revision),
            );
            let read_from =
                if round == 0 { ReadFrom::Path } else { ReadFrom::Stdin(srcs[round].to_owned()) };

            let proc_res = self.print_source(read_from, &self.props.pretty_mode);
            if !proc_res.status.success() {
                self.fatal_proc_rec(
                    &format!(
                        "pretty-printing failed in round {} revision {:?}",
                        round, self.revision
                    ),
                    &proc_res,
                );
            }

            let ProcRes { stdout, .. } = proc_res;
            srcs.push(stdout);
            round += 1;
        }

        let mut expected = match self.props.pp_exact {
            Some(ref file) => {
                let filepath = self.testpaths.file.parent().unwrap().join(file);
                fs::read_to_string(&filepath).unwrap()
            }
            None => srcs[srcs.len() - 2].clone(),
        };
        let mut actual = srcs[srcs.len() - 1].clone();

        if self.props.pp_exact.is_some() {
            // Now we have to care about line endings
            let cr = "\r".to_owned();
            actual = actual.replace(&cr, "");
            expected = expected.replace(&cr, "");
        }

        self.compare_source(&expected, &actual);

        // If we're only making sure that the output matches then just stop here
        if self.props.pretty_compare_only {
            return;
        }

        // Finally, let's make sure it actually appears to remain valid code
        let proc_res = self.typecheck_source(actual);
        if !proc_res.status.success() {
            self.fatal_proc_rec("pretty-printed source does not typecheck", &proc_res);
        }

        if !self.props.pretty_expanded {
            return;
        }

        // additionally, run `-Zunpretty=expanded` and try to build it.
        let proc_res = self.print_source(ReadFrom::Path, "expanded");
        if !proc_res.status.success() {
            self.fatal_proc_rec("pretty-printing (expanded) failed", &proc_res);
        }

        let ProcRes { stdout: expanded_src, .. } = proc_res;
        let proc_res = self.typecheck_source(expanded_src);
        if !proc_res.status.success() {
            self.fatal_proc_rec("pretty-printed source (expanded) does not typecheck", &proc_res);
        }
    }

    pub(super) fn run_debuginfo_test(&self) {
        match self.config.debugger.unwrap() {
            Debugger::Cdb => self.run_debuginfo_cdb_test(),
            Debugger::Gdb => self.run_debuginfo_gdb_test(),
            Debugger::Lldb => self.run_debuginfo_lldb_test(),
        }
    }

    pub(super) fn run_codegen_test(&self) {
        if self.config.llvm_filecheck.is_none() {
            self.fatal("missing --llvm-filecheck");
        }

        let proc_res = self.compile_test_and_save_ir();
        if !proc_res.status.success() {
            self.fatal_proc_rec("compilation failed!", &proc_res);
        }

        let output_path = self.output_base_name().with_extension("ll");
        let proc_res = self.verify_with_filecheck(&output_path);
        if !proc_res.status.success() {
            self.fatal_proc_rec("verification with 'FileCheck' failed", &proc_res);
        }
    }

    pub(super) fn run_assembly_test(&self) {
        if self.config.llvm_filecheck.is_none() {
            self.fatal("missing --llvm-filecheck");
        }

        let (proc_res, output_path) = self.compile_test_and_save_assembly();
        if !proc_res.status.success() {
            self.fatal_proc_rec("compilation failed!", &proc_res);
        }

        let proc_res = self.verify_with_filecheck(&output_path);
        if !proc_res.status.success() {
            self.fatal_proc_rec("verification with 'FileCheck' failed", &proc_res);
        }
    }

    pub(super) fn run_rustdoc_test(&self) {
        assert!(self.revision.is_none(), "revisions not relevant here");

        let out_dir = self.output_base_dir();
        let _ = fs::remove_dir_all(&out_dir);
        create_dir_all(&out_dir).unwrap();

        let proc_res = self.document(&out_dir);
        if !proc_res.status.success() {
            self.fatal_proc_rec("rustdoc failed!", &proc_res);
        }

        if self.props.check_test_line_numbers_match {
            self.check_rustdoc_test_option(proc_res);
        } else {
            let root = self.config.find_rust_src_root().unwrap();
            let res = self.cmd2procres(
                Command::new(&self.config.docck_python)
                    .arg(root.join("src/etc/htmldocck.py"))
                    .arg(&out_dir)
                    .arg(&self.testpaths.file),
            );
            if !res.status.success() {
                self.fatal_proc_rec_with_ctx("htmldocck failed!", &res, |mut this| {
                    this.compare_to_default_rustdoc(&out_dir)
                });
            }
        }
    }

    pub(super) fn run_incremental_test(&self) {
        // Basic plan for a test incremental/foo/bar.rs:
        // - load list of revisions rpass1, cfail2, rpass3
        //   - each should begin with `rpass`, `cfail`, or `rfail`
        //   - if `rpass`, expect compile and execution to succeed
        //   - if `cfail`, expect compilation to fail
        //   - if `rfail`, expect execution to fail
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
            print!("revision={:?} props={:#?}", revision, self.props);
        }

        if revision.starts_with("rpass") {
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
            self.fatal("revision name must begin with rpass, rfail, or cfail");
        }
    }

    pub(super) fn run_rmake_test(&self) {
        let cwd = env::current_dir().unwrap();
        let src_root = self.config.src_base.parent().unwrap().parent().unwrap().parent().unwrap();
        let src_root = cwd.join(&src_root);

        let tmpdir = cwd.join(self.output_base_name());
        if tmpdir.exists() {
            self.aggressive_rm_rf(&tmpdir).unwrap();
        }
        create_dir_all(&tmpdir).unwrap();

        let host = &self.config.host;
        let make = if host.contains("dragonfly")
            || host.contains("freebsd")
            || host.contains("netbsd")
            || host.contains("openbsd")
        {
            "gmake"
        } else {
            "make"
        };

        let mut cmd = Command::new(make);
        cmd.current_dir(&self.testpaths.file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("TARGET", &self.config.target)
            .env("PYTHON", &self.config.docck_python)
            .env("S", src_root)
            .env("RUST_BUILD_STAGE", &self.config.stage_id)
            .env("RUSTC", cwd.join(&self.config.rustc_path))
            .env("TMPDIR", &tmpdir)
            .env("LD_LIB_PATH_ENVVAR", dylib_env_var())
            .env("HOST_RPATH_DIR", cwd.join(&self.config.compile_lib_path))
            .env("TARGET_RPATH_DIR", cwd.join(&self.config.run_lib_path))
            .env("LLVM_COMPONENTS", &self.config.llvm_components)
            // We for sure don't want these tests to run in parallel, so make
            // sure they don't have access to these vars if we run via `make`
            // at the top level
            .env_remove("MAKEFLAGS")
            .env_remove("MFLAGS")
            .env_remove("CARGO_MAKEFLAGS");

        if let Some(ref rustdoc) = self.config.rustdoc_path {
            cmd.env("RUSTDOC", cwd.join(rustdoc));
        }

        if let Some(ref rust_demangler) = self.config.rust_demangler_path {
            cmd.env("RUST_DEMANGLER", cwd.join(rust_demangler));
        }

        if let Some(ref node) = self.config.nodejs {
            cmd.env("NODE", node);
        }

        if let Some(ref linker) = self.config.linker {
            cmd.env("RUSTC_LINKER", linker);
        }

        if let Some(ref clang) = self.config.run_clang_based_tests_with {
            cmd.env("CLANG", clang);
        }

        if let Some(ref filecheck) = self.config.llvm_filecheck {
            cmd.env("LLVM_FILECHECK", filecheck);
        }

        if let Some(ref llvm_bin_dir) = self.config.llvm_bin_dir {
            cmd.env("LLVM_BIN_DIR", llvm_bin_dir);
        }

        // We don't want RUSTFLAGS set from the outside to interfere with
        // compiler flags set in the test cases:
        cmd.env_remove("RUSTFLAGS");

        // Use dynamic musl for tests because static doesn't allow creating dylibs
        if self.config.host.contains("musl") {
            cmd.env("RUSTFLAGS", "-Ctarget-feature=-crt-static").env("IS_MUSL_HOST", "1");
        }

        if self.config.bless {
            cmd.env("RUSTC_BLESS_TEST", "--bless");
            // Assume this option is active if the environment variable is "defined", with _any_ value.
            // As an example, a `Makefile` can use this option by:
            //
            //   ifdef RUSTC_BLESS_TEST
            //       cp "$(TMPDIR)"/actual_something.ext expected_something.ext
            //   else
            //       $(DIFF) expected_something.ext "$(TMPDIR)"/actual_something.ext
            //   endif
        }

        if self.config.target.contains("msvc") && self.config.cc != "" {
            // We need to pass a path to `lib.exe`, so assume that `cc` is `cl.exe`
            // and that `lib.exe` lives next to it.
            let lib = Path::new(&self.config.cc).parent().unwrap().join("lib.exe");

            // MSYS doesn't like passing flags of the form `/foo` as it thinks it's
            // a path and instead passes `C:\msys64\foo`, so convert all
            // `/`-arguments to MSVC here to `-` arguments.
            let cflags = self
                .config
                .cflags
                .split(' ')
                .map(|s| s.replace("/", "-"))
                .collect::<Vec<_>>()
                .join(" ");

            cmd.env("IS_MSVC", "1")
                .env("IS_WINDOWS", "1")
                .env("MSVC_LIB", format!("'{}' -nologo", lib.display()))
                .env("CC", format!("'{}' {}", self.config.cc, cflags))
                .env("CXX", format!("'{}'", &self.config.cxx));
        } else {
            cmd.env("CC", format!("{} {}", self.config.cc, self.config.cflags))
                .env("CXX", format!("{} {}", self.config.cxx, self.config.cflags))
                .env("AR", &self.config.ar);

            if self.config.target.contains("windows") {
                cmd.env("IS_WINDOWS", "1");
            }
        }

        let output = cmd.spawn().and_then(read2_abbreviated).expect("failed to spawn `make`");
        if !output.status.success() {
            let res = ProcRes {
                status: output.status,
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
                cmdline: format!("{:?}", cmd),
            };
            self.fatal_proc_rec("make failed", &res);
        }
    }

    pub(super) fn run_ui_test(&self) {
        if let Some(FailMode::Build) = self.props.fail_mode {
            // Make sure a build-fail test cannot fail due to failing analysis (e.g. typeck).
            let pm = Some(PassMode::Check);
            let proc_res = self.compile_test_general(WillExecute::No, EmitMetadata::Yes, pm);
            self.check_if_test_should_compile(&proc_res, pm);
        }

        let pm = self.pass_mode();
        let should_run = self.should_run(pm);
        let emit_metadata = self.should_emit_metadata(pm);
        let proc_res = self.compile_test(should_run, emit_metadata);
        self.check_if_test_should_compile(&proc_res, pm);

        // if the user specified a format in the ui test
        // print the output to the stderr file, otherwise extract
        // the rendered error messages from json and print them
        let explicit = self.props.compile_flags.iter().any(|s| s.contains("--error-format"));

        let expected_fixed = self.load_expected_output(UI_FIXED);

        let modes_to_prune = vec![CompareMode::Nll];
        self.prune_duplicate_outputs(&modes_to_prune);

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
                let mut coverage_file_path = self.config.build_base.clone();
                coverage_file_path.push("rustfix_missing_coverage.txt");
                debug!("coverage_file_path: {}", coverage_file_path.display());

                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(coverage_file_path.as_path())
                    .expect("could not create or open file");

                if writeln!(file, "{}", self.testpaths.file.display()).is_err() {
                    panic!("couldn't write to {}", coverage_file_path.display());
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

            errors += self.compare_output("fixed", &fixed_code, &expected_fixed);
        } else if !expected_fixed.is_empty() {
            panic!(
                "the `// run-rustfix` directive wasn't found but a `*.fixed` \
                 file was found"
            );
        }

        if errors > 0 {
            println!("To update references, rerun the tests and pass the `--bless` flag");
            let relative_path_to_file =
                self.testpaths.relative_dir.join(self.testpaths.file.file_name().unwrap());
            println!(
                "To only update this specific test, also pass `--test-args {}`",
                relative_path_to_file.display(),
            );
            self.fatal_proc_rec(
                &format!("{} errors occurred comparing output.", errors),
                &proc_res,
            );
        }

        let expected_errors = errors::load_errors(&self.testpaths.file, self.revision);

        if let WillExecute::Yes = should_run {
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
            if self.should_run_successfully(pm) {
                if !proc_res.status.success() {
                    self.fatal_proc_rec("test run failed!", &proc_res);
                }
            } else {
                if proc_res.status.success() {
                    self.fatal_proc_rec("test run succeeded!", &proc_res);
                }
            }
            if !self.props.error_patterns.is_empty() {
                // "// error-pattern" comments
                self.check_error_patterns(&proc_res.stderr, &proc_res, pm);
            }
        }

        debug!(
            "run_ui_test: explicit={:?} config.compare_mode={:?} expected_errors={:?} \
               proc_res.status={:?} props.error_patterns={:?}",
            explicit,
            self.config.compare_mode,
            expected_errors,
            proc_res.status,
            self.props.error_patterns
        );
        if !explicit && self.config.compare_mode.is_none() {
            let check_patterns =
                should_run == WillExecute::No && !self.props.error_patterns.is_empty();

            let check_annotations = !check_patterns || !expected_errors.is_empty();

            if check_patterns {
                // "// error-pattern" comments
                self.check_error_patterns(&proc_res.stderr, &proc_res, pm);
            }

            if check_annotations {
                // "//~ERROR comments"
                self.check_expected_errors(expected_errors, &proc_res);
            }
        }

        if self.props.run_rustfix && self.config.compare_mode.is_none() {
            // And finally, compile the fixed code and make sure it both
            // succeeds and has no diagnostics.
            let mut rustc = self.make_compile_args(
                &self.testpaths.file.with_extension(UI_FIXED),
                TargetLocation::ThisFile(self.make_exe_name()),
                emit_metadata,
                AllowUnused::No,
            );
            rustc.arg("-L").arg(&self.aux_output_dir_name());
            let res = self.compose_and_run_compiler(rustc, None);
            if !res.status.success() {
                self.fatal_proc_rec("failed to compile fixed code", &res);
            }
            if !res.stderr.is_empty() && !self.props.rustfix_only_machine_applicable {
                if !json::rustfix_diagnostics_only(&res.stderr).is_empty() {
                    self.fatal_proc_rec("fixed code is still producing diagnostics", &res);
                }
            }
        }
    }
}
