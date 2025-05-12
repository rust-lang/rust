use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::fs::{self, File, create_dir_all};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::prelude::*;
use std::io::{self, BufReader};
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::sync::Arc;
use std::{env, iter, str};

use build_helper::fs::remove_and_create_dir_all;
use camino::{Utf8Path, Utf8PathBuf};
use colored::Colorize;
use regex::{Captures, Regex};
use tracing::*;

use crate::common::{
    Assembly, Codegen, CodegenUnits, CompareMode, Config, CoverageMap, CoverageRun, Crashes,
    DebugInfo, Debugger, FailMode, Incremental, MirOpt, PassMode, Pretty, RunMake, Rustdoc,
    RustdocJs, RustdocJson, TestPaths, UI_EXTENSIONS, UI_FIXED, UI_RUN_STDERR, UI_RUN_STDOUT,
    UI_STDERR, UI_STDOUT, UI_SVG, UI_WINDOWS_SVG, Ui, expected_output_path, incremental_dir,
    output_base_dir, output_base_name, output_testname_unique,
};
use crate::compute_diff::{DiffLine, make_diff, write_diff, write_filtered_diff};
use crate::errors::{Error, ErrorKind, load_errors};
use crate::header::TestProps;
use crate::read2::{Truncated, read2_abbreviated};
use crate::util::{Utf8PathBufExt, add_dylib_path, logv, static_regex};
use crate::{ColorConfig, json, stamp_file_path};

mod debugger;

// Helper modules that implement test running logic for each test suite.
// tidy-alphabetical-start
mod assembly;
mod codegen;
mod codegen_units;
mod coverage;
mod crashes;
mod debuginfo;
mod incremental;
mod js_doc;
mod mir_opt;
mod pretty;
mod run_make;
mod rustdoc;
mod rustdoc_json;
mod ui;
// tidy-alphabetical-end

#[cfg(test)]
mod tests;

const FAKE_SRC_BASE: &str = "fake-test-src-base";

#[cfg(windows)]
fn disable_error_reporting<F: FnOnce() -> R, R>(f: F) -> R {
    use std::sync::Mutex;

    use windows::Win32::System::Diagnostics::Debug::{
        SEM_FAILCRITICALERRORS, SEM_NOGPFAULTERRORBOX, SetErrorMode,
    };

    static LOCK: Mutex<()> = Mutex::new(());

    // Error mode is a global variable, so lock it so only one thread will change it
    let _lock = LOCK.lock().unwrap();

    // Tell Windows to not show any UI on errors (such as terminating abnormally). This is important
    // for running tests, since some of them use abnormal termination by design. This mode is
    // inherited by all child processes.
    //
    // Note that `run-make` tests require `SEM_FAILCRITICALERRORS` in addition to suppress Windows
    // Error Reporting (WER) error dialogues that come from "critical failures" such as missing
    // DLLs.
    //
    // See <https://github.com/rust-lang/rust/issues/132092> and
    // <https://learn.microsoft.com/en-us/windows/win32/api/errhandlingapi/nf-errhandlingapi-seterrormode?redirectedfrom=MSDN>.
    unsafe {
        // read inherited flags
        let old_mode = SetErrorMode(SEM_NOGPFAULTERRORBOX | SEM_FAILCRITICALERRORS);
        SetErrorMode(old_mode | SEM_NOGPFAULTERRORBOX | SEM_FAILCRITICALERRORS);
        let r = f();
        SetErrorMode(old_mode);
        r
    }
}

#[cfg(not(windows))]
fn disable_error_reporting<F: FnOnce() -> R, R>(f: F) -> R {
    f()
}

/// The platform-specific library name
fn get_lib_name(name: &str, aux_type: AuxType) -> Option<String> {
    match aux_type {
        AuxType::Bin => None,
        // In some cases (e.g. MUSL), we build a static
        // library, rather than a dynamic library.
        // In this case, the only path we can pass
        // with '--extern-meta' is the '.rlib' file
        AuxType::Lib => Some(format!("lib{name}.rlib")),
        AuxType::Dylib | AuxType::ProcMacro => Some(dylib_name(name)),
    }
}

fn dylib_name(name: &str) -> String {
    format!("{}{name}.{}", std::env::consts::DLL_PREFIX, std::env::consts::DLL_EXTENSION)
}

pub fn run(config: Arc<Config>, testpaths: &TestPaths, revision: Option<&str>) {
    match &*config.target {
        "arm-linux-androideabi"
        | "armv7-linux-androideabi"
        | "thumbv7neon-linux-androideabi"
        | "aarch64-linux-android" => {
            if !config.adb_device_status {
                panic!("android device not available");
            }
        }

        _ => {
            // android has its own gdb handling
            if config.debugger == Some(Debugger::Gdb) && config.gdb.is_none() {
                panic!("gdb not available but debuginfo gdb debuginfo test requested");
            }
        }
    }

    if config.verbose {
        // We're going to be dumping a lot of info. Start on a new line.
        print!("\n\n");
    }
    debug!("running {}", testpaths.file);
    let mut props = TestProps::from_file(&testpaths.file, revision, &config);

    // For non-incremental (i.e. regular UI) tests, the incremental directory
    // takes into account the revision name, since the revisions are independent
    // of each other and can race.
    if props.incremental {
        props.incremental_dir = Some(incremental_dir(&config, testpaths, revision));
    }

    let cx = TestCx { config: &config, props: &props, testpaths, revision };

    if let Err(e) = create_dir_all(&cx.output_base_dir()) {
        panic!("failed to create output base directory {}: {e}", cx.output_base_dir());
    }

    if props.incremental {
        cx.init_incremental_test();
    }

    if config.mode == Incremental {
        // Incremental tests are special because they cannot be run in
        // parallel.
        assert!(!props.revisions.is_empty(), "Incremental tests require revisions.");
        for revision in &props.revisions {
            let mut revision_props = TestProps::from_file(&testpaths.file, Some(revision), &config);
            revision_props.incremental_dir = props.incremental_dir.clone();
            let rev_cx = TestCx {
                config: &config,
                props: &revision_props,
                testpaths,
                revision: Some(revision),
            };
            rev_cx.run_revision();
        }
    } else {
        cx.run_revision();
    }

    cx.create_stamp();
}

pub fn compute_stamp_hash(config: &Config) -> String {
    let mut hash = DefaultHasher::new();
    config.stage_id.hash(&mut hash);
    config.run.hash(&mut hash);
    config.edition.hash(&mut hash);

    match config.debugger {
        Some(Debugger::Cdb) => {
            config.cdb.hash(&mut hash);
        }

        Some(Debugger::Gdb) => {
            config.gdb.hash(&mut hash);
            env::var_os("PATH").hash(&mut hash);
            env::var_os("PYTHONPATH").hash(&mut hash);
        }

        Some(Debugger::Lldb) => {
            config.python.hash(&mut hash);
            config.lldb_python_dir.hash(&mut hash);
            env::var_os("PATH").hash(&mut hash);
            env::var_os("PYTHONPATH").hash(&mut hash);
        }

        None => {}
    }

    if let Ui = config.mode {
        config.force_pass_mode.hash(&mut hash);
    }

    format!("{:x}", hash.finish())
}

#[derive(Copy, Clone, Debug)]
struct TestCx<'test> {
    config: &'test Config,
    props: &'test TestProps,
    testpaths: &'test TestPaths,
    revision: Option<&'test str>,
}

enum ReadFrom {
    Path,
    Stdin(String),
}

enum TestOutput {
    Compile,
    Run,
}

/// Will this test be executed? Should we use `make_exe_name`?
#[derive(Copy, Clone, PartialEq)]
enum WillExecute {
    Yes,
    No,
    Disabled,
}

/// What value should be passed to `--emit`?
#[derive(Copy, Clone)]
enum Emit {
    None,
    Metadata,
    LlvmIr,
    Mir,
    Asm,
    LinkArgsAsm,
}

impl<'test> TestCx<'test> {
    /// Code executed for each revision in turn (or, if there are no
    /// revisions, exactly once, with revision == None).
    fn run_revision(&self) {
        if self.props.should_ice && self.config.mode != Incremental && self.config.mode != Crashes {
            self.fatal("cannot use should-ice in a test that is not cfail");
        }
        match self.config.mode {
            Pretty => self.run_pretty_test(),
            DebugInfo => self.run_debuginfo_test(),
            Codegen => self.run_codegen_test(),
            Rustdoc => self.run_rustdoc_test(),
            RustdocJson => self.run_rustdoc_json_test(),
            CodegenUnits => self.run_codegen_units_test(),
            Incremental => self.run_incremental_test(),
            RunMake => self.run_rmake_test(),
            Ui => self.run_ui_test(),
            MirOpt => self.run_mir_opt_test(),
            Assembly => self.run_assembly_test(),
            RustdocJs => self.run_rustdoc_js_test(),
            CoverageMap => self.run_coverage_map_test(), // see self::coverage
            CoverageRun => self.run_coverage_run_test(), // see self::coverage
            Crashes => self.run_crash_test(),
        }
    }

    fn pass_mode(&self) -> Option<PassMode> {
        self.props.pass_mode(self.config)
    }

    fn should_run(&self, pm: Option<PassMode>) -> WillExecute {
        let test_should_run = match self.config.mode {
            Ui if pm == Some(PassMode::Run) || self.props.fail_mode == Some(FailMode::Run) => true,
            MirOpt if pm == Some(PassMode::Run) => true,
            Ui | MirOpt => false,
            mode => panic!("unimplemented for mode {:?}", mode),
        };
        if test_should_run { self.run_if_enabled() } else { WillExecute::No }
    }

    fn run_if_enabled(&self) -> WillExecute {
        if self.config.run_enabled() { WillExecute::Yes } else { WillExecute::Disabled }
    }

    fn should_run_successfully(&self, pm: Option<PassMode>) -> bool {
        match self.config.mode {
            Ui | MirOpt => pm == Some(PassMode::Run),
            mode => panic!("unimplemented for mode {:?}", mode),
        }
    }

    fn should_compile_successfully(&self, pm: Option<PassMode>) -> bool {
        match self.config.mode {
            RustdocJs => true,
            Ui => pm.is_some() || self.props.fail_mode > Some(FailMode::Build),
            Crashes => false,
            Incremental => {
                let revision =
                    self.revision.expect("incremental tests require a list of revisions");
                if revision.starts_with("cpass")
                    || revision.starts_with("rpass")
                    || revision.starts_with("rfail")
                {
                    true
                } else if revision.starts_with("cfail") {
                    pm.is_some()
                } else {
                    panic!("revision name must begin with cpass, rpass, rfail, or cfail");
                }
            }
            mode => panic!("unimplemented for mode {:?}", mode),
        }
    }

    fn check_if_test_should_compile(
        &self,
        fail_mode: Option<FailMode>,
        pass_mode: Option<PassMode>,
        proc_res: &ProcRes,
    ) {
        if self.should_compile_successfully(pass_mode) {
            if !proc_res.status.success() {
                match (fail_mode, pass_mode) {
                    (Some(FailMode::Build), Some(PassMode::Check)) => {
                        // A `build-fail` test needs to `check-pass`.
                        self.fatal_proc_rec(
                            "`build-fail` test is required to pass check build, but check build failed",
                            proc_res,
                        );
                    }
                    _ => {
                        self.fatal_proc_rec(
                            "test compilation failed although it shouldn't!",
                            proc_res,
                        );
                    }
                }
            }
        } else {
            if proc_res.status.success() {
                {
                    self.error(&format!("{} test did not emit an error", self.config.mode));
                    if self.config.mode == crate::common::Mode::Ui {
                        println!("note: by default, ui tests are expected not to compile");
                    }
                    proc_res.fatal(None, || ());
                };
            }

            if !self.props.dont_check_failure_status {
                self.check_correct_failure_status(proc_res);
            }
        }
    }

    fn get_output(&self, proc_res: &ProcRes) -> String {
        if self.props.check_stdout {
            format!("{}{}", proc_res.stdout, proc_res.stderr)
        } else {
            proc_res.stderr.clone()
        }
    }

    fn check_correct_failure_status(&self, proc_res: &ProcRes) {
        let expected_status = Some(self.props.failure_status.unwrap_or(1));
        let received_status = proc_res.status.code();

        if expected_status != received_status {
            self.fatal_proc_rec(
                &format!(
                    "Error: expected failure status ({:?}) but received status {:?}.",
                    expected_status, received_status
                ),
                proc_res,
            );
        }
    }

    /// Runs a [`Command`] and waits for it to finish, then converts its exit
    /// status and output streams into a [`ProcRes`].
    ///
    /// The command might have succeeded or failed; it is the caller's
    /// responsibility to check the exit status and take appropriate action.
    ///
    /// # Panics
    /// Panics if the command couldn't be executed at all
    /// (e.g. because the executable could not be found).
    #[must_use = "caller should check whether the command succeeded"]
    fn run_command_to_procres(&self, cmd: &mut Command) -> ProcRes {
        let output = cmd
            .output()
            .unwrap_or_else(|e| self.fatal(&format!("failed to exec `{cmd:?}` because: {e}")));

        let proc_res = ProcRes {
            status: output.status,
            stdout: String::from_utf8(output.stdout).unwrap(),
            stderr: String::from_utf8(output.stderr).unwrap(),
            truncated: Truncated::No,
            cmdline: format!("{cmd:?}"),
        };
        self.dump_output(
            self.config.verbose,
            &cmd.get_program().to_string_lossy(),
            &proc_res.stdout,
            &proc_res.stderr,
        );

        proc_res
    }

    fn print_source(&self, read_from: ReadFrom, pretty_type: &str) -> ProcRes {
        let aux_dir = self.aux_output_dir_name();
        let input: &str = match read_from {
            ReadFrom::Stdin(_) => "-",
            ReadFrom::Path => self.testpaths.file.as_str(),
        };

        let mut rustc = Command::new(&self.config.rustc_path);
        rustc
            .arg(input)
            .args(&["-Z", &format!("unpretty={}", pretty_type)])
            .args(&["--target", &self.config.target])
            .arg("-L")
            .arg(&aux_dir)
            .arg("-A")
            .arg("internal_features")
            .args(&self.props.compile_flags)
            .envs(self.props.rustc_env.clone());
        self.maybe_add_external_args(&mut rustc, &self.config.target_rustcflags);

        let src = match read_from {
            ReadFrom::Stdin(src) => Some(src),
            ReadFrom::Path => None,
        };

        self.compose_and_run(
            rustc,
            self.config.compile_lib_path.as_path(),
            Some(aux_dir.as_path()),
            src,
        )
    }

    fn compare_source(&self, expected: &str, actual: &str) {
        if expected != actual {
            self.fatal(&format!(
                "pretty-printed source does not match expected source\n\
                 expected:\n\
                 ------------------------------------------\n\
                 {}\n\
                 ------------------------------------------\n\
                 actual:\n\
                 ------------------------------------------\n\
                 {}\n\
                 ------------------------------------------\n\
                 diff:\n\
                 ------------------------------------------\n\
                 {}\n",
                expected,
                actual,
                write_diff(expected, actual, 3),
            ));
        }
    }

    fn set_revision_flags(&self, cmd: &mut Command) {
        // Normalize revisions to be lowercase and replace `-`s with `_`s.
        // Otherwise the `--cfg` flag is not valid.
        let normalize_revision = |revision: &str| revision.to_lowercase().replace("-", "_");

        if let Some(revision) = self.revision {
            let normalized_revision = normalize_revision(revision);
            let cfg_arg = ["--cfg", &normalized_revision];
            let arg = format!("--cfg={normalized_revision}");
            if self
                .props
                .compile_flags
                .windows(2)
                .any(|args| args == cfg_arg || args[0] == arg || args[1] == arg)
            {
                panic!(
                    "error: redundant cfg argument `{normalized_revision}` is already created by the revision"
                );
            }
            if self.config.builtin_cfg_names().contains(&normalized_revision) {
                panic!("error: revision `{normalized_revision}` collides with a builtin cfg");
            }
            cmd.args(cfg_arg);
        }

        if !self.props.no_auto_check_cfg {
            let mut check_cfg = String::with_capacity(25);

            // Generate `cfg(FALSE, REV1, ..., REVN)` (for all possible revisions)
            //
            // For compatibility reason we consider the `FALSE` cfg to be expected
            // since it is extensively used in the testsuite, as well as the `test`
            // cfg since we have tests that uses it.
            check_cfg.push_str("cfg(test,FALSE");
            for revision in &self.props.revisions {
                check_cfg.push(',');
                check_cfg.push_str(&normalize_revision(revision));
            }
            check_cfg.push(')');

            cmd.args(&["--check-cfg", &check_cfg]);
        }
    }

    fn typecheck_source(&self, src: String) -> ProcRes {
        let mut rustc = Command::new(&self.config.rustc_path);

        let out_dir = self.output_base_name().with_extension("pretty-out");
        remove_and_create_dir_all(&out_dir).unwrap_or_else(|e| {
            panic!("failed to remove and recreate output directory `{out_dir}`: {e}")
        });

        let target = if self.props.force_host { &*self.config.host } else { &*self.config.target };

        let aux_dir = self.aux_output_dir_name();

        rustc
            .arg("-")
            .arg("-Zno-codegen")
            .arg("--out-dir")
            .arg(&out_dir)
            .arg(&format!("--target={}", target))
            .arg("-L")
            // FIXME(jieyouxu): this search path seems questionable. Is this intended for
            // `rust_test_helpers` in ui tests?
            .arg(&self.config.build_test_suite_root)
            .arg("-L")
            .arg(aux_dir)
            .arg("-A")
            .arg("internal_features");
        self.set_revision_flags(&mut rustc);
        self.maybe_add_external_args(&mut rustc, &self.config.target_rustcflags);
        rustc.args(&self.props.compile_flags);

        self.compose_and_run_compiler(rustc, Some(src), self.testpaths)
    }

    fn maybe_add_external_args(&self, cmd: &mut Command, args: &Vec<String>) {
        // Filter out the arguments that should not be added by runtest here.
        //
        // Notable use-cases are: do not add our optimisation flag if
        // `compile-flags: -Copt-level=x` and similar for debug-info level as well.
        const OPT_FLAGS: &[&str] = &["-O", "-Copt-level=", /*-C<space>*/ "opt-level="];
        const DEBUG_FLAGS: &[&str] = &["-g", "-Cdebuginfo=", /*-C<space>*/ "debuginfo="];

        // FIXME: ideally we would "just" check the `cmd` itself, but it does not allow inspecting
        // its arguments. They need to be collected separately. For now I cannot be bothered to
        // implement this the "right" way.
        let have_opt_flag =
            self.props.compile_flags.iter().any(|arg| OPT_FLAGS.iter().any(|f| arg.starts_with(f)));
        let have_debug_flag = self
            .props
            .compile_flags
            .iter()
            .any(|arg| DEBUG_FLAGS.iter().any(|f| arg.starts_with(f)));

        for arg in args {
            if OPT_FLAGS.iter().any(|f| arg.starts_with(f)) && have_opt_flag {
                continue;
            }
            if DEBUG_FLAGS.iter().any(|f| arg.starts_with(f)) && have_debug_flag {
                continue;
            }
            cmd.arg(arg);
        }
    }

    /// Check `error-pattern` and `regex-error-pattern` directives.
    fn check_all_error_patterns(&self, output_to_check: &str, proc_res: &ProcRes) {
        let mut missing_patterns: Vec<String> = Vec::new();
        self.check_error_patterns(output_to_check, &mut missing_patterns);
        self.check_regex_error_patterns(output_to_check, proc_res, &mut missing_patterns);

        if missing_patterns.is_empty() {
            return;
        }

        if missing_patterns.len() == 1 {
            self.fatal_proc_rec(
                &format!("error pattern '{}' not found!", missing_patterns[0]),
                proc_res,
            );
        } else {
            for pattern in missing_patterns {
                self.error(&format!("error pattern '{}' not found!", pattern));
            }
            self.fatal_proc_rec("multiple error patterns not found", proc_res);
        }
    }

    fn check_error_patterns(&self, output_to_check: &str, missing_patterns: &mut Vec<String>) {
        debug!("check_error_patterns");
        for pattern in &self.props.error_patterns {
            if output_to_check.contains(pattern.trim()) {
                debug!("found error pattern {}", pattern);
            } else {
                missing_patterns.push(pattern.to_string());
            }
        }
    }

    fn check_regex_error_patterns(
        &self,
        output_to_check: &str,
        proc_res: &ProcRes,
        missing_patterns: &mut Vec<String>,
    ) {
        debug!("check_regex_error_patterns");

        for pattern in &self.props.regex_error_patterns {
            let pattern = pattern.trim();
            let re = match Regex::new(pattern) {
                Ok(re) => re,
                Err(err) => {
                    self.fatal_proc_rec(
                        &format!("invalid regex error pattern '{}': {:?}", pattern, err),
                        proc_res,
                    );
                }
            };
            if re.is_match(output_to_check) {
                debug!("found regex error pattern {}", pattern);
            } else {
                missing_patterns.push(pattern.to_string());
            }
        }
    }

    fn check_no_compiler_crash(&self, proc_res: &ProcRes, should_ice: bool) {
        match proc_res.status.code() {
            Some(101) if !should_ice => {
                self.fatal_proc_rec("compiler encountered internal error", proc_res)
            }
            None => self.fatal_proc_rec("compiler terminated by signal", proc_res),
            _ => (),
        }
    }

    fn check_forbid_output(&self, output_to_check: &str, proc_res: &ProcRes) {
        for pat in &self.props.forbid_output {
            if output_to_check.contains(pat) {
                self.fatal_proc_rec("forbidden pattern found in compiler output", proc_res);
            }
        }
    }

    /// Check `//~ KIND message` annotations.
    fn check_expected_errors(&self, proc_res: &ProcRes) {
        let expected_errors = load_errors(&self.testpaths.file, self.revision);
        debug!(
            "check_expected_errors: expected_errors={:?} proc_res.status={:?}",
            expected_errors, proc_res.status
        );
        if proc_res.status.success() && expected_errors.iter().any(|x| x.kind == ErrorKind::Error) {
            self.fatal_proc_rec("process did not return an error status", proc_res);
        }

        if self.props.known_bug {
            if !expected_errors.is_empty() {
                self.fatal_proc_rec(
                    "`known_bug` tests should not have an expected error",
                    proc_res,
                );
            }
            return;
        }

        // On Windows, translate all '\' path separators to '/'
        let file_name = self.testpaths.file.to_string().replace(r"\", "/");

        // On Windows, keep all '\' path separators to match the paths reported in the JSON output
        // from the compiler
        let diagnostic_file_name = if self.props.remap_src_base {
            let mut p = Utf8PathBuf::from(FAKE_SRC_BASE);
            p.push(&self.testpaths.relative_dir);
            p.push(self.testpaths.file.file_name().unwrap());
            p.to_string()
        } else {
            self.testpaths.file.to_string()
        };

        // Errors and warnings are always expected, other diagnostics are only expected
        // if one of them actually occurs in the test.
        let expected_kinds: HashSet<_> = [ErrorKind::Error, ErrorKind::Warning]
            .into_iter()
            .chain(expected_errors.iter().map(|e| e.kind))
            .collect();

        // Parse the JSON output from the compiler and extract out the messages.
        let actual_errors = json::parse_output(&diagnostic_file_name, &self.get_output(proc_res))
            .into_iter()
            .map(|e| Error { msg: self.normalize_output(&e.msg, &[]), ..e });

        let mut unexpected = Vec::new();
        let mut found = vec![false; expected_errors.len()];
        for actual_error in actual_errors {
            for pattern in &self.props.error_patterns {
                let pattern = pattern.trim();
                if actual_error.msg.contains(pattern) {
                    let q = if actual_error.line_num.is_none() { "?" } else { "" };
                    self.fatal(&format!(
                        "error pattern '{pattern}' is found in structured \
                         diagnostics, use `//~{q} {} {pattern}` instead",
                        actual_error.kind,
                    ));
                }
            }

            let opt_index =
                expected_errors.iter().enumerate().position(|(index, expected_error)| {
                    !found[index]
                        && actual_error.line_num == expected_error.line_num
                        && actual_error.kind == expected_error.kind
                        && actual_error.msg.contains(&expected_error.msg)
                });

            match opt_index {
                Some(index) => {
                    // found a match, everybody is happy
                    assert!(!found[index]);
                    found[index] = true;
                }

                None => {
                    if actual_error.require_annotation
                        && expected_kinds.contains(&actual_error.kind)
                        && !self.props.dont_require_annotations.contains(&actual_error.kind)
                    {
                        self.error(&format!(
                            "{}:{}: unexpected {}: '{}'",
                            file_name,
                            actual_error.line_num_str(),
                            actual_error.kind,
                            actual_error.msg
                        ));
                        unexpected.push(actual_error);
                    }
                }
            }
        }

        let mut not_found = Vec::new();
        // anything not yet found is a problem
        for (index, expected_error) in expected_errors.iter().enumerate() {
            if !found[index] {
                self.error(&format!(
                    "{}:{}: expected {} not found: {}",
                    file_name,
                    expected_error.line_num_str(),
                    expected_error.kind,
                    expected_error.msg
                ));
                not_found.push(expected_error);
            }
        }

        if !unexpected.is_empty() || !not_found.is_empty() {
            self.error(&format!(
                "{} unexpected errors found, {} expected errors not found",
                unexpected.len(),
                not_found.len()
            ));
            println!("status: {}\ncommand: {}\n", proc_res.status, proc_res.cmdline);
            if !unexpected.is_empty() {
                println!("{}", "--- unexpected errors (from JSON output) ---".green());
                for error in &unexpected {
                    println!("{}", error.render_for_expected());
                }
                println!("{}", "---".green());
            }
            if !not_found.is_empty() {
                println!("{}", "--- not found errors (from test file) ---".red());
                for error in &not_found {
                    println!("{}", error.render_for_expected());
                }
                println!("{}", "---\n".red());
            }
            panic!("errors differ from expected");
        }
    }

    fn should_emit_metadata(&self, pm: Option<PassMode>) -> Emit {
        match (pm, self.props.fail_mode, self.config.mode) {
            (Some(PassMode::Check), ..) | (_, Some(FailMode::Check), Ui) => Emit::Metadata,
            _ => Emit::None,
        }
    }

    fn compile_test(&self, will_execute: WillExecute, emit: Emit) -> ProcRes {
        self.compile_test_general(will_execute, emit, self.props.local_pass_mode(), Vec::new())
    }

    fn compile_test_with_passes(
        &self,
        will_execute: WillExecute,
        emit: Emit,
        passes: Vec<String>,
    ) -> ProcRes {
        self.compile_test_general(will_execute, emit, self.props.local_pass_mode(), passes)
    }

    fn compile_test_general(
        &self,
        will_execute: WillExecute,
        emit: Emit,
        local_pm: Option<PassMode>,
        passes: Vec<String>,
    ) -> ProcRes {
        // Only use `make_exe_name` when the test ends up being executed.
        let output_file = match will_execute {
            WillExecute::Yes => TargetLocation::ThisFile(self.make_exe_name()),
            WillExecute::No | WillExecute::Disabled => {
                TargetLocation::ThisDirectory(self.output_base_dir())
            }
        };

        let allow_unused = match self.config.mode {
            Ui => {
                // UI tests tend to have tons of unused code as
                // it's just testing various pieces of the compile, but we don't
                // want to actually assert warnings about all this code. Instead
                // let's just ignore unused code warnings by defaults and tests
                // can turn it back on if needed.
                if !self.is_rustdoc()
                    // Note that we use the local pass mode here as we don't want
                    // to set unused to allow if we've overridden the pass mode
                    // via command line flags.
                    && local_pm != Some(PassMode::Run)
                {
                    AllowUnused::Yes
                } else {
                    AllowUnused::No
                }
            }
            _ => AllowUnused::No,
        };

        let rustc = self.make_compile_args(
            &self.testpaths.file,
            output_file,
            emit,
            allow_unused,
            LinkToAux::Yes,
            passes,
        );

        self.compose_and_run_compiler(rustc, None, self.testpaths)
    }

    /// `root_out_dir` and `root_testpaths` refer to the parameters of the actual test being run.
    /// Auxiliaries, no matter how deep, have the same root_out_dir and root_testpaths.
    fn document(&self, root_out_dir: &Utf8Path, root_testpaths: &TestPaths) -> ProcRes {
        if self.props.build_aux_docs {
            for rel_ab in &self.props.aux.builds {
                let aux_testpaths = self.compute_aux_test_paths(root_testpaths, rel_ab);
                let props_for_aux =
                    self.props.from_aux_file(&aux_testpaths.file, self.revision, self.config);
                let aux_cx = TestCx {
                    config: self.config,
                    props: &props_for_aux,
                    testpaths: &aux_testpaths,
                    revision: self.revision,
                };
                // Create the directory for the stdout/stderr files.
                create_dir_all(aux_cx.output_base_dir()).unwrap();
                // use root_testpaths here, because aux-builds should have the
                // same --out-dir and auxiliary directory.
                let auxres = aux_cx.document(&root_out_dir, root_testpaths);
                if !auxres.status.success() {
                    return auxres;
                }
            }
        }

        let aux_dir = self.aux_output_dir_name();

        let rustdoc_path = self.config.rustdoc_path.as_ref().expect("--rustdoc-path not passed");

        // actual --out-dir given to the auxiliary or test, as opposed to the root out dir for the entire
        // test
        let out_dir: Cow<'_, Utf8Path> = if self.props.unique_doc_out_dir {
            let file_name = self.testpaths.file.file_stem().expect("file name should not be empty");
            let out_dir = Utf8PathBuf::from_iter([
                root_out_dir,
                Utf8Path::new("docs"),
                Utf8Path::new(file_name),
                Utf8Path::new("doc"),
            ]);
            create_dir_all(&out_dir).unwrap();
            Cow::Owned(out_dir)
        } else {
            Cow::Borrowed(root_out_dir)
        };

        let mut rustdoc = Command::new(rustdoc_path);
        let current_dir = output_base_dir(self.config, root_testpaths, self.safe_revision());
        rustdoc.current_dir(current_dir);
        rustdoc
            .arg("-L")
            .arg(self.config.run_lib_path.as_path())
            .arg("-L")
            .arg(aux_dir)
            .arg("-o")
            .arg(out_dir.as_ref())
            .arg("--deny")
            .arg("warnings")
            .arg(&self.testpaths.file)
            .arg("-A")
            .arg("internal_features")
            .args(&self.props.compile_flags)
            .args(&self.props.doc_flags);

        if self.config.mode == RustdocJson {
            rustdoc.arg("--output-format").arg("json").arg("-Zunstable-options");
        }

        if let Some(ref linker) = self.config.target_linker {
            rustdoc.arg(format!("-Clinker={}", linker));
        }

        self.compose_and_run_compiler(rustdoc, None, root_testpaths)
    }

    fn exec_compiled_test(&self) -> ProcRes {
        self.exec_compiled_test_general(&[], true)
    }

    fn exec_compiled_test_general(
        &self,
        env_extra: &[(&str, &str)],
        delete_after_success: bool,
    ) -> ProcRes {
        let prepare_env = |cmd: &mut Command| {
            for (key, val) in &self.props.exec_env {
                cmd.env(key, val);
            }
            for (key, val) in env_extra {
                cmd.env(key, val);
            }

            for key in &self.props.unset_exec_env {
                cmd.env_remove(key);
            }
        };

        let proc_res = match &*self.config.target {
            // This is pretty similar to below, we're transforming:
            //
            //      program arg1 arg2
            //
            // into
            //
            //      remote-test-client run program 2 support-lib.so support-lib2.so arg1 arg2
            //
            // The test-client program will upload `program` to the emulator
            // along with all other support libraries listed (in this case
            // `support-lib.so` and `support-lib2.so`. It will then execute
            // the program on the emulator with the arguments specified
            // (in the environment we give the process) and then report back
            // the same result.
            _ if self.config.remote_test_client.is_some() => {
                let aux_dir = self.aux_output_dir_name();
                let ProcArgs { prog, args } = self.make_run_args();
                let mut support_libs = Vec::new();
                if let Ok(entries) = aux_dir.read_dir() {
                    for entry in entries {
                        let entry = entry.unwrap();
                        if !entry.path().is_file() {
                            continue;
                        }
                        support_libs.push(entry.path());
                    }
                }
                let mut test_client =
                    Command::new(self.config.remote_test_client.as_ref().unwrap());
                test_client
                    .args(&["run", &support_libs.len().to_string()])
                    .arg(&prog)
                    .args(support_libs)
                    .args(args);

                prepare_env(&mut test_client);

                self.compose_and_run(
                    test_client,
                    self.config.run_lib_path.as_path(),
                    Some(aux_dir.as_path()),
                    None,
                )
            }
            _ if self.config.target.contains("vxworks") => {
                let aux_dir = self.aux_output_dir_name();
                let ProcArgs { prog, args } = self.make_run_args();
                let mut wr_run = Command::new("wr-run");
                wr_run.args(&[&prog]).args(args);

                prepare_env(&mut wr_run);

                self.compose_and_run(
                    wr_run,
                    self.config.run_lib_path.as_path(),
                    Some(aux_dir.as_path()),
                    None,
                )
            }
            _ => {
                let aux_dir = self.aux_output_dir_name();
                let ProcArgs { prog, args } = self.make_run_args();
                let mut program = Command::new(&prog);
                program.args(args).current_dir(&self.output_base_dir());

                prepare_env(&mut program);

                self.compose_and_run(
                    program,
                    self.config.run_lib_path.as_path(),
                    Some(aux_dir.as_path()),
                    None,
                )
            }
        };

        if delete_after_success && proc_res.status.success() {
            // delete the executable after running it to save space.
            // it is ok if the deletion failed.
            let _ = fs::remove_file(self.make_exe_name());
        }

        proc_res
    }

    /// For each `aux-build: foo/bar` annotation, we check to find the file in an `auxiliary`
    /// directory relative to the test itself (not any intermediate auxiliaries).
    fn compute_aux_test_paths(&self, of: &TestPaths, rel_ab: &str) -> TestPaths {
        let test_ab =
            of.file.parent().expect("test file path has no parent").join("auxiliary").join(rel_ab);
        if !test_ab.exists() {
            self.fatal(&format!("aux-build `{}` source not found", test_ab))
        }

        TestPaths {
            file: test_ab,
            relative_dir: of
                .relative_dir
                .join(self.output_testname_unique())
                .join("auxiliary")
                .join(rel_ab)
                .parent()
                .expect("aux-build path has no parent")
                .to_path_buf(),
        }
    }

    fn is_vxworks_pure_static(&self) -> bool {
        if self.config.target.contains("vxworks") {
            match env::var("RUST_VXWORKS_TEST_DYLINK") {
                Ok(s) => s != "1",
                _ => true,
            }
        } else {
            false
        }
    }

    fn is_vxworks_pure_dynamic(&self) -> bool {
        self.config.target.contains("vxworks") && !self.is_vxworks_pure_static()
    }

    fn has_aux_dir(&self) -> bool {
        !self.props.aux.builds.is_empty()
            || !self.props.aux.crates.is_empty()
            || !self.props.aux.proc_macros.is_empty()
    }

    fn aux_output_dir(&self) -> Utf8PathBuf {
        let aux_dir = self.aux_output_dir_name();

        if !self.props.aux.builds.is_empty() {
            remove_and_create_dir_all(&aux_dir).unwrap_or_else(|e| {
                panic!("failed to remove and recreate output directory `{aux_dir}`: {e}")
            });
        }

        if !self.props.aux.bins.is_empty() {
            let aux_bin_dir = self.aux_bin_output_dir_name();
            remove_and_create_dir_all(&aux_dir).unwrap_or_else(|e| {
                panic!("failed to remove and recreate output directory `{aux_dir}`: {e}")
            });
            remove_and_create_dir_all(&aux_bin_dir).unwrap_or_else(|e| {
                panic!("failed to remove and recreate output directory `{aux_bin_dir}`: {e}")
            });
        }

        aux_dir
    }

    fn build_all_auxiliary(&self, of: &TestPaths, aux_dir: &Utf8Path, rustc: &mut Command) {
        for rel_ab in &self.props.aux.builds {
            self.build_auxiliary(of, rel_ab, &aux_dir, None);
        }

        for rel_ab in &self.props.aux.bins {
            self.build_auxiliary(of, rel_ab, &aux_dir, Some(AuxType::Bin));
        }

        let path_to_crate_name = |path: &str| -> String {
            path.rsplit_once('/')
                .map_or(path, |(_, tail)| tail)
                .trim_end_matches(".rs")
                .replace('-', "_")
        };

        let add_extern =
            |rustc: &mut Command, aux_name: &str, aux_path: &str, aux_type: AuxType| {
                let lib_name = get_lib_name(&path_to_crate_name(aux_path), aux_type);
                if let Some(lib_name) = lib_name {
                    rustc.arg("--extern").arg(format!("{}={}/{}", aux_name, aux_dir, lib_name));
                }
            };

        for (aux_name, aux_path) in &self.props.aux.crates {
            let aux_type = self.build_auxiliary(of, &aux_path, &aux_dir, None);
            add_extern(rustc, aux_name, aux_path, aux_type);
        }

        for proc_macro in &self.props.aux.proc_macros {
            self.build_auxiliary(of, proc_macro, &aux_dir, Some(AuxType::ProcMacro));
            let crate_name = path_to_crate_name(proc_macro);
            add_extern(rustc, &crate_name, proc_macro, AuxType::ProcMacro);
        }

        // Build any `//@ aux-codegen-backend`, and pass the resulting library
        // to `-Zcodegen-backend` when compiling the test file.
        if let Some(aux_file) = &self.props.aux.codegen_backend {
            let aux_type = self.build_auxiliary(of, aux_file, aux_dir, None);
            if let Some(lib_name) = get_lib_name(aux_file.trim_end_matches(".rs"), aux_type) {
                let lib_path = aux_dir.join(&lib_name);
                rustc.arg(format!("-Zcodegen-backend={}", lib_path));
            }
        }
    }

    /// `root_testpaths` refers to the path of the original test. the auxiliary and the test with an
    /// aux-build have the same `root_testpaths`.
    fn compose_and_run_compiler(
        &self,
        mut rustc: Command,
        input: Option<String>,
        root_testpaths: &TestPaths,
    ) -> ProcRes {
        if self.props.add_core_stubs {
            let minicore_path = self.build_minicore();
            rustc.arg("--extern");
            rustc.arg(&format!("minicore={}", minicore_path));
        }

        let aux_dir = self.aux_output_dir();
        self.build_all_auxiliary(root_testpaths, &aux_dir, &mut rustc);

        rustc.envs(self.props.rustc_env.clone());
        self.props.unset_rustc_env.iter().fold(&mut rustc, Command::env_remove);
        self.compose_and_run(
            rustc,
            self.config.compile_lib_path.as_path(),
            Some(aux_dir.as_path()),
            input,
        )
    }

    /// Builds `minicore`. Returns the path to the minicore rlib within the base test output
    /// directory.
    fn build_minicore(&self) -> Utf8PathBuf {
        let output_file_path = self.output_base_dir().join("libminicore.rlib");
        let mut rustc = self.make_compile_args(
            &self.config.minicore_path,
            TargetLocation::ThisFile(output_file_path.clone()),
            Emit::None,
            AllowUnused::Yes,
            LinkToAux::No,
            vec![],
        );

        rustc.args(&["--crate-type", "rlib"]);
        rustc.arg("-Cpanic=abort");

        let res = self.compose_and_run(rustc, self.config.compile_lib_path.as_path(), None, None);
        if !res.status.success() {
            self.fatal_proc_rec(
                &format!("auxiliary build of {} failed to compile: ", self.config.minicore_path),
                &res,
            );
        }

        output_file_path
    }

    /// Builds an aux dependency.
    ///
    /// If `aux_type` is `None`, then this will determine the aux-type automatically.
    fn build_auxiliary(
        &self,
        of: &TestPaths,
        source_path: &str,
        aux_dir: &Utf8Path,
        aux_type: Option<AuxType>,
    ) -> AuxType {
        let aux_testpaths = self.compute_aux_test_paths(of, source_path);
        let mut aux_props =
            self.props.from_aux_file(&aux_testpaths.file, self.revision, self.config);
        if aux_type == Some(AuxType::ProcMacro) {
            aux_props.force_host = true;
        }
        let mut aux_dir = aux_dir.to_path_buf();
        if aux_type == Some(AuxType::Bin) {
            // On unix, the binary of `auxiliary/foo.rs` will be named
            // `auxiliary/foo` which clashes with the _dir_ `auxiliary/foo`, so
            // put bins in a `bin` subfolder.
            aux_dir.push("bin");
        }
        let aux_output = TargetLocation::ThisDirectory(aux_dir.clone());
        let aux_cx = TestCx {
            config: self.config,
            props: &aux_props,
            testpaths: &aux_testpaths,
            revision: self.revision,
        };
        // Create the directory for the stdout/stderr files.
        create_dir_all(aux_cx.output_base_dir()).unwrap();
        let input_file = &aux_testpaths.file;
        let mut aux_rustc = aux_cx.make_compile_args(
            input_file,
            aux_output,
            Emit::None,
            AllowUnused::No,
            LinkToAux::No,
            Vec::new(),
        );
        aux_cx.build_all_auxiliary(of, &aux_dir, &mut aux_rustc);

        aux_rustc.envs(aux_props.rustc_env.clone());
        for key in &aux_props.unset_rustc_env {
            aux_rustc.env_remove(key);
        }

        let (aux_type, crate_type) = if aux_type == Some(AuxType::Bin) {
            (AuxType::Bin, Some("bin"))
        } else if aux_type == Some(AuxType::ProcMacro) {
            (AuxType::ProcMacro, Some("proc-macro"))
        } else if aux_type.is_some() {
            panic!("aux_type {aux_type:?} not expected");
        } else if aux_props.no_prefer_dynamic {
            (AuxType::Dylib, None)
        } else if self.config.target.contains("emscripten")
            || (self.config.target.contains("musl")
                && !aux_props.force_host
                && !self.config.host.contains("musl"))
            || self.config.target.contains("wasm32")
            || self.config.target.contains("nvptx")
            || self.is_vxworks_pure_static()
            || self.config.target.contains("bpf")
            || !self.config.target_cfg().dynamic_linking
            || matches!(self.config.mode, CoverageMap | CoverageRun)
        {
            // We primarily compile all auxiliary libraries as dynamic libraries
            // to avoid code size bloat and large binaries as much as possible
            // for the test suite (otherwise including libstd statically in all
            // executables takes up quite a bit of space).
            //
            // For targets like MUSL or Emscripten, however, there is no support for
            // dynamic libraries so we just go back to building a normal library. Note,
            // however, that for MUSL if the library is built with `force_host` then
            // it's ok to be a dylib as the host should always support dylibs.
            //
            // Coverage tests want static linking by default so that coverage
            // mappings in auxiliary libraries can be merged into the final
            // executable.
            (AuxType::Lib, Some("lib"))
        } else {
            (AuxType::Dylib, Some("dylib"))
        };

        if let Some(crate_type) = crate_type {
            aux_rustc.args(&["--crate-type", crate_type]);
        }

        if aux_type == AuxType::ProcMacro {
            // For convenience, but this only works on 2018.
            aux_rustc.args(&["--extern", "proc_macro"]);
        }

        aux_rustc.arg("-L").arg(&aux_dir);

        let auxres = aux_cx.compose_and_run(
            aux_rustc,
            aux_cx.config.compile_lib_path.as_path(),
            Some(aux_dir.as_path()),
            None,
        );
        if !auxres.status.success() {
            self.fatal_proc_rec(
                &format!("auxiliary build of {} failed to compile: ", aux_testpaths.file),
                &auxres,
            );
        }
        aux_type
    }

    fn read2_abbreviated(&self, child: Child) -> (Output, Truncated) {
        let mut filter_paths_from_len = Vec::new();
        let mut add_path = |path: &Utf8Path| {
            let path = path.to_string();
            let windows = path.replace("\\", "\\\\");
            if windows != path {
                filter_paths_from_len.push(windows);
            }
            filter_paths_from_len.push(path);
        };

        // List of paths that will not be measured when determining whether the output is larger
        // than the output truncation threshold.
        //
        // Note: avoid adding a subdirectory of an already filtered directory here, otherwise the
        // same slice of text will be double counted and the truncation might not happen.
        add_path(&self.config.src_test_suite_root);
        add_path(&self.config.build_test_suite_root);

        read2_abbreviated(child, &filter_paths_from_len).expect("failed to read output")
    }

    fn compose_and_run(
        &self,
        mut command: Command,
        lib_path: &Utf8Path,
        aux_path: Option<&Utf8Path>,
        input: Option<String>,
    ) -> ProcRes {
        let cmdline = {
            let cmdline = self.make_cmdline(&command, lib_path);
            logv(self.config, format!("executing {}", cmdline));
            cmdline
        };

        command.stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::piped());

        // Need to be sure to put both the lib_path and the aux path in the dylib
        // search path for the child.
        add_dylib_path(&mut command, iter::once(lib_path).chain(aux_path));

        let mut child = disable_error_reporting(|| command.spawn())
            .unwrap_or_else(|e| panic!("failed to exec `{command:?}`: {e:?}"));
        if let Some(input) = input {
            child.stdin.as_mut().unwrap().write_all(input.as_bytes()).unwrap();
        }

        let (Output { status, stdout, stderr }, truncated) = self.read2_abbreviated(child);

        let result = ProcRes {
            status,
            stdout: String::from_utf8_lossy(&stdout).into_owned(),
            stderr: String::from_utf8_lossy(&stderr).into_owned(),
            truncated,
            cmdline,
        };

        self.dump_output(
            self.config.verbose,
            &command.get_program().to_string_lossy(),
            &result.stdout,
            &result.stderr,
        );

        result
    }

    fn is_rustdoc(&self) -> bool {
        matches!(self.config.suite.as_str(), "rustdoc-ui" | "rustdoc-js" | "rustdoc-json")
    }

    fn make_compile_args(
        &self,
        input_file: &Utf8Path,
        output_file: TargetLocation,
        emit: Emit,
        allow_unused: AllowUnused,
        link_to_aux: LinkToAux,
        passes: Vec<String>, // Vec of passes under mir-opt test to be dumped
    ) -> Command {
        let is_aux = input_file.components().map(|c| c.as_os_str()).any(|c| c == "auxiliary");
        let is_rustdoc = self.is_rustdoc() && !is_aux;
        let mut rustc = if !is_rustdoc {
            Command::new(&self.config.rustc_path)
        } else {
            Command::new(&self.config.rustdoc_path.clone().expect("no rustdoc built yet"))
        };
        rustc.arg(input_file);

        // Use a single thread for efficiency and a deterministic error message order
        rustc.arg("-Zthreads=1");

        // Hide libstd sources from ui tests to make sure we generate the stderr
        // output that users will see.
        // Without this, we may be producing good diagnostics in-tree but users
        // will not see half the information.
        //
        // This also has the benefit of more effectively normalizing output between different
        // compilers, so that we don't have to know the `/rustc/$sha` output to normalize after the
        // fact.
        rustc.arg("-Zsimulate-remapped-rust-src-base=/rustc/FAKE_PREFIX");
        rustc.arg("-Ztranslate-remapped-path-to-local-path=no");

        // Hide Cargo dependency sources from ui tests to make sure the error message doesn't
        // change depending on whether $CARGO_HOME is remapped or not. If this is not present,
        // when $CARGO_HOME is remapped the source won't be shown, and when it's not remapped the
        // source will be shown, causing a blessing hell.
        rustc.arg("-Z").arg(format!(
            "ignore-directory-in-diagnostics-source-blocks={}",
            home::cargo_home().expect("failed to find cargo home").to_str().unwrap()
        ));
        // Similarly, vendored sources shouldn't be shown when running from a dist tarball.
        rustc.arg("-Z").arg(format!(
            "ignore-directory-in-diagnostics-source-blocks={}",
            self.config.src_root.join("vendor"),
        ));

        // Optionally prevent default --sysroot if specified in test compile-flags.
        if !self.props.compile_flags.iter().any(|flag| flag.starts_with("--sysroot"))
            && !self.config.host_rustcflags.iter().any(|flag| flag == "--sysroot")
        {
            // In stage 0, make sure we use `stage0-sysroot` instead of the bootstrap sysroot.
            rustc.arg("--sysroot").arg(&self.config.sysroot_base);
        }

        // Optionally prevent default --target if specified in test compile-flags.
        let custom_target = self.props.compile_flags.iter().any(|x| x.starts_with("--target"));

        if !custom_target {
            let target =
                if self.props.force_host { &*self.config.host } else { &*self.config.target };

            rustc.arg(&format!("--target={}", target));
        }
        self.set_revision_flags(&mut rustc);

        if !is_rustdoc {
            if let Some(ref incremental_dir) = self.props.incremental_dir {
                rustc.args(&["-C", &format!("incremental={}", incremental_dir)]);
                rustc.args(&["-Z", "incremental-verify-ich"]);
            }

            if self.config.mode == CodegenUnits {
                rustc.args(&["-Z", "human_readable_cgu_names"]);
            }
        }

        if self.config.optimize_tests && !is_rustdoc {
            match self.config.mode {
                Ui => {
                    // If optimize-tests is true we still only want to optimize tests that actually get
                    // executed and that don't specify their own optimization levels.
                    // Note: aux libs don't have a pass-mode, so they won't get optimized
                    // unless compile-flags are set in the aux file.
                    if self.config.optimize_tests
                        && self.props.pass_mode(&self.config) == Some(PassMode::Run)
                        && !self
                            .props
                            .compile_flags
                            .iter()
                            .any(|arg| arg == "-O" || arg.contains("opt-level"))
                    {
                        rustc.arg("-O");
                    }
                }
                DebugInfo => { /* debuginfo tests must be unoptimized */ }
                CoverageMap | CoverageRun => {
                    // Coverage mappings and coverage reports are affected by
                    // optimization level, so they ignore the optimize-tests
                    // setting and set an optimization level in their mode's
                    // compile flags (below) or in per-test `compile-flags`.
                }
                _ => {
                    rustc.arg("-O");
                }
            }
        }

        let set_mir_dump_dir = |rustc: &mut Command| {
            let mir_dump_dir = self.output_base_dir();
            let mut dir_opt = "-Zdump-mir-dir=".to_string();
            dir_opt.push_str(mir_dump_dir.as_str());
            debug!("dir_opt: {:?}", dir_opt);
            rustc.arg(dir_opt);
        };

        match self.config.mode {
            Incremental => {
                // If we are extracting and matching errors in the new
                // fashion, then you want JSON mode. Old-skool error
                // patterns still match the raw compiler output.
                if self.props.error_patterns.is_empty()
                    && self.props.regex_error_patterns.is_empty()
                {
                    rustc.args(&["--error-format", "json"]);
                    rustc.args(&["--json", "future-incompat"]);
                }
                rustc.arg("-Zui-testing");
                rustc.arg("-Zdeduplicate-diagnostics=no");
            }
            Ui => {
                if !self.props.compile_flags.iter().any(|s| s.starts_with("--error-format")) {
                    rustc.args(&["--error-format", "json"]);
                    rustc.args(&["--json", "future-incompat"]);
                }
                rustc.arg("-Ccodegen-units=1");
                // Hide line numbers to reduce churn
                rustc.arg("-Zui-testing");
                rustc.arg("-Zdeduplicate-diagnostics=no");
                rustc.arg("-Zwrite-long-types-to-disk=no");
                // FIXME: use this for other modes too, for perf?
                rustc.arg("-Cstrip=debuginfo");
            }
            MirOpt => {
                // We check passes under test to minimize the mir-opt test dump
                // if files_for_miropt_test parses the passes, we dump only those passes
                // otherwise we conservatively pass -Zdump-mir=all
                let zdump_arg = if !passes.is_empty() {
                    format!("-Zdump-mir={}", passes.join(" | "))
                } else {
                    "-Zdump-mir=all".to_string()
                };

                rustc.args(&[
                    "-Copt-level=1",
                    &zdump_arg,
                    "-Zvalidate-mir",
                    "-Zlint-mir",
                    "-Zdump-mir-exclude-pass-number",
                    "-Zmir-include-spans=false", // remove span comments from NLL MIR dumps
                    "--crate-type=rlib",
                ]);
                if let Some(pass) = &self.props.mir_unit_test {
                    rustc.args(&["-Zmir-opt-level=0", &format!("-Zmir-enable-passes=+{}", pass)]);
                } else {
                    rustc.args(&[
                        "-Zmir-opt-level=4",
                        "-Zmir-enable-passes=+ReorderBasicBlocks,+ReorderLocals",
                    ]);
                }

                set_mir_dump_dir(&mut rustc);
            }
            CoverageMap => {
                rustc.arg("-Cinstrument-coverage");
                // These tests only compile to LLVM IR, so they don't need the
                // profiler runtime to be present.
                rustc.arg("-Zno-profiler-runtime");
                // Coverage mappings are sensitive to MIR optimizations, and
                // the current snapshots assume `opt-level=2` unless overridden
                // by `compile-flags`.
                rustc.arg("-Copt-level=2");
            }
            CoverageRun => {
                rustc.arg("-Cinstrument-coverage");
                // Coverage reports are sometimes sensitive to optimizations,
                // and the current snapshots assume `opt-level=2` unless
                // overridden by `compile-flags`.
                rustc.arg("-Copt-level=2");
            }
            Assembly | Codegen => {
                rustc.arg("-Cdebug-assertions=no");
            }
            Crashes => {
                set_mir_dump_dir(&mut rustc);
            }
            CodegenUnits => {
                rustc.arg("-Zprint-mono-items");
            }
            Pretty | DebugInfo | Rustdoc | RustdocJson | RunMake | RustdocJs => {
                // do not use JSON output
            }
        }

        if self.props.remap_src_base {
            rustc.arg(format!(
                "--remap-path-prefix={}={}",
                self.config.src_test_suite_root, FAKE_SRC_BASE,
            ));
        }

        match emit {
            Emit::None => {}
            Emit::Metadata if is_rustdoc => {}
            Emit::Metadata => {
                rustc.args(&["--emit", "metadata"]);
            }
            Emit::LlvmIr => {
                rustc.args(&["--emit", "llvm-ir"]);
            }
            Emit::Mir => {
                rustc.args(&["--emit", "mir"]);
            }
            Emit::Asm => {
                rustc.args(&["--emit", "asm"]);
            }
            Emit::LinkArgsAsm => {
                rustc.args(&["-Clink-args=--emit=asm"]);
            }
        }

        if !is_rustdoc {
            if self.config.target == "wasm32-unknown-unknown" || self.is_vxworks_pure_static() {
                // rustc.arg("-g"); // get any backtrace at all on errors
            } else if !self.props.no_prefer_dynamic {
                rustc.args(&["-C", "prefer-dynamic"]);
            }
        }

        match output_file {
            // If the test's compile flags specify an output path with `-o`,
            // avoid a compiler warning about `--out-dir` being ignored.
            _ if self.props.compile_flags.iter().any(|flag| flag == "-o") => {}
            TargetLocation::ThisFile(path) => {
                rustc.arg("-o").arg(path);
            }
            TargetLocation::ThisDirectory(path) => {
                if is_rustdoc {
                    // `rustdoc` uses `-o` for the output directory.
                    rustc.arg("-o").arg(path);
                } else {
                    rustc.arg("--out-dir").arg(path);
                }
            }
        }

        match self.config.compare_mode {
            Some(CompareMode::Polonius) => {
                rustc.args(&["-Zpolonius"]);
            }
            Some(CompareMode::NextSolver) => {
                rustc.args(&["-Znext-solver"]);
            }
            Some(CompareMode::NextSolverCoherence) => {
                rustc.args(&["-Znext-solver=coherence"]);
            }
            Some(CompareMode::SplitDwarf) if self.config.target.contains("windows") => {
                rustc.args(&["-Csplit-debuginfo=unpacked", "-Zunstable-options"]);
            }
            Some(CompareMode::SplitDwarf) => {
                rustc.args(&["-Csplit-debuginfo=unpacked"]);
            }
            Some(CompareMode::SplitDwarfSingle) => {
                rustc.args(&["-Csplit-debuginfo=packed"]);
            }
            None => {}
        }

        // Add `-A unused` before `config` flags and in-test (`props`) flags, so that they can
        // overwrite this.
        if let AllowUnused::Yes = allow_unused {
            rustc.args(&["-A", "unused"]);
        }

        // Allow tests to use internal features.
        rustc.args(&["-A", "internal_features"]);

        if self.props.force_host {
            self.maybe_add_external_args(&mut rustc, &self.config.host_rustcflags);
            if !is_rustdoc {
                if let Some(ref linker) = self.config.host_linker {
                    rustc.arg(format!("-Clinker={}", linker));
                }
            }
        } else {
            self.maybe_add_external_args(&mut rustc, &self.config.target_rustcflags);
            if !is_rustdoc {
                if let Some(ref linker) = self.config.target_linker {
                    rustc.arg(format!("-Clinker={}", linker));
                }
            }
        }

        // Use dynamic musl for tests because static doesn't allow creating dylibs
        if self.config.host.contains("musl") || self.is_vxworks_pure_dynamic() {
            rustc.arg("-Ctarget-feature=-crt-static");
        }

        if let LinkToAux::Yes = link_to_aux {
            // if we pass an `-L` argument to a directory that doesn't exist,
            // macOS ld emits warnings which disrupt the .stderr files
            if self.has_aux_dir() {
                rustc.arg("-L").arg(self.aux_output_dir_name());
            }
        }

        rustc.args(&self.props.compile_flags);

        // FIXME(jieyouxu): we should report a fatal error or warning if user wrote `-Cpanic=` with
        // something that's not `abort` and `-Cforce-unwind-tables` with a value that is not `yes`,
        // however, by moving this last we should override previous `-Cpanic`s and
        // `-Cforce-unwind-tables`s. Note that checking here is very fragile, because we'd have to
        // account for all possible compile flag splittings (they have some... intricacies and are
        // not yet normalized).
        //
        // `minicore` requires `#![no_std]` and `#![no_core]`, which means no unwinding panics.
        if self.props.add_core_stubs {
            rustc.arg("-Cpanic=abort");
            rustc.arg("-Cforce-unwind-tables=yes");
        }

        rustc
    }

    fn make_exe_name(&self) -> Utf8PathBuf {
        // Using a single letter here to keep the path length down for
        // Windows.  Some test names get very long.  rustc creates `rcgu`
        // files with the module name appended to it which can more than
        // double the length.
        let mut f = self.output_base_dir().join("a");
        // FIXME: This is using the host architecture exe suffix, not target!
        if self.config.target.contains("emscripten") {
            f = f.with_extra_extension("js");
        } else if self.config.target.starts_with("wasm") {
            f = f.with_extra_extension("wasm");
        } else if self.config.target.contains("spirv") {
            f = f.with_extra_extension("spv");
        } else if !env::consts::EXE_SUFFIX.is_empty() {
            f = f.with_extra_extension(env::consts::EXE_SUFFIX);
        }
        f
    }

    fn make_run_args(&self) -> ProcArgs {
        // If we've got another tool to run under (valgrind),
        // then split apart its command
        let mut args = self.split_maybe_args(&self.config.runner);

        let exe_file = self.make_exe_name();

        args.push(exe_file.into_os_string());

        // Add the arguments in the run_flags directive
        args.extend(self.props.run_flags.iter().map(OsString::from));

        let prog = args.remove(0);
        ProcArgs { prog, args }
    }

    fn split_maybe_args(&self, argstr: &Option<String>) -> Vec<OsString> {
        match *argstr {
            Some(ref s) => s
                .split(' ')
                .filter_map(|s| {
                    if s.chars().all(|c| c.is_whitespace()) {
                        None
                    } else {
                        Some(OsString::from(s))
                    }
                })
                .collect(),
            None => Vec::new(),
        }
    }

    fn make_cmdline(&self, command: &Command, libpath: &Utf8Path) -> String {
        use crate::util;

        // Linux and mac don't require adjusting the library search path
        if cfg!(unix) {
            format!("{:?}", command)
        } else {
            // Build the LD_LIBRARY_PATH variable as it would be seen on the command line
            // for diagnostic purposes
            fn lib_path_cmd_prefix(path: &str) -> String {
                format!("{}=\"{}\"", util::lib_path_env_var(), util::make_new_path(path))
            }

            format!("{} {:?}", lib_path_cmd_prefix(libpath.as_str()), command)
        }
    }

    fn dump_output(&self, print_output: bool, proc_name: &str, out: &str, err: &str) {
        let revision = if let Some(r) = self.revision { format!("{}.", r) } else { String::new() };

        self.dump_output_file(out, &format!("{}out", revision));
        self.dump_output_file(err, &format!("{}err", revision));

        if !print_output {
            return;
        }

        let path = Utf8Path::new(proc_name);
        let proc_name = if path.file_stem().is_some_and(|p| p == "rmake") {
            String::from_iter(
                path.parent()
                    .unwrap()
                    .file_name()
                    .into_iter()
                    .chain(Some("/"))
                    .chain(path.file_name()),
            )
        } else {
            path.file_name().unwrap().into()
        };
        println!("------{proc_name} stdout------------------------------");
        println!("{}", out);
        println!("------{proc_name} stderr------------------------------");
        println!("{}", err);
        println!("------------------------------------------");
    }

    fn dump_output_file(&self, out: &str, extension: &str) {
        let outfile = self.make_out_name(extension);
        fs::write(outfile.as_std_path(), out).unwrap();
    }

    /// Creates a filename for output with the given extension.
    /// E.g., `/.../testname.revision.mode/testname.extension`.
    fn make_out_name(&self, extension: &str) -> Utf8PathBuf {
        self.output_base_name().with_extension(extension)
    }

    /// Gets the directory where auxiliary files are written.
    /// E.g., `/.../testname.revision.mode/auxiliary/`.
    fn aux_output_dir_name(&self) -> Utf8PathBuf {
        self.output_base_dir()
            .join("auxiliary")
            .with_extra_extension(self.config.mode.aux_dir_disambiguator())
    }

    /// Gets the directory where auxiliary binaries are written.
    /// E.g., `/.../testname.revision.mode/auxiliary/bin`.
    fn aux_bin_output_dir_name(&self) -> Utf8PathBuf {
        self.aux_output_dir_name().join("bin")
    }

    /// Generates a unique name for the test, such as `testname.revision.mode`.
    fn output_testname_unique(&self) -> Utf8PathBuf {
        output_testname_unique(self.config, self.testpaths, self.safe_revision())
    }

    /// The revision, ignored for incremental compilation since it wants all revisions in
    /// the same directory.
    fn safe_revision(&self) -> Option<&str> {
        if self.config.mode == Incremental { None } else { self.revision }
    }

    /// Gets the absolute path to the directory where all output for the given
    /// test/revision should reside.
    /// E.g., `/path/to/build/host-tuple/test/ui/relative/testname.revision.mode/`.
    fn output_base_dir(&self) -> Utf8PathBuf {
        output_base_dir(self.config, self.testpaths, self.safe_revision())
    }

    /// Gets the absolute path to the base filename used as output for the given
    /// test/revision.
    /// E.g., `/.../relative/testname.revision.mode/testname`.
    fn output_base_name(&self) -> Utf8PathBuf {
        output_base_name(self.config, self.testpaths, self.safe_revision())
    }

    fn error(&self, err: &str) {
        match self.revision {
            Some(rev) => println!("\nerror in revision `{}`: {}", rev, err),
            None => println!("\nerror: {}", err),
        }
    }

    #[track_caller]
    fn fatal(&self, err: &str) -> ! {
        self.error(err);
        error!("fatal error, panic: {:?}", err);
        panic!("fatal error");
    }

    fn fatal_proc_rec(&self, err: &str, proc_res: &ProcRes) -> ! {
        self.error(err);
        proc_res.fatal(None, || ());
    }

    fn fatal_proc_rec_with_ctx(
        &self,
        err: &str,
        proc_res: &ProcRes,
        on_failure: impl FnOnce(Self),
    ) -> ! {
        self.error(err);
        proc_res.fatal(None, || on_failure(*self));
    }

    // codegen tests (using FileCheck)

    fn compile_test_and_save_ir(&self) -> (ProcRes, Utf8PathBuf) {
        let output_path = self.output_base_name().with_extension("ll");
        let input_file = &self.testpaths.file;
        let rustc = self.make_compile_args(
            input_file,
            TargetLocation::ThisFile(output_path.clone()),
            Emit::LlvmIr,
            AllowUnused::No,
            LinkToAux::Yes,
            Vec::new(),
        );

        let proc_res = self.compose_and_run_compiler(rustc, None, self.testpaths);
        (proc_res, output_path)
    }

    fn verify_with_filecheck(&self, output: &Utf8Path) -> ProcRes {
        let mut filecheck = Command::new(self.config.llvm_filecheck.as_ref().unwrap());
        filecheck.arg("--input-file").arg(output).arg(&self.testpaths.file);

        // Because we use custom prefixes, we also have to register the default prefix.
        filecheck.arg("--check-prefix=CHECK");

        // FIXME(#134510): auto-registering revision names as check prefix is a bit sketchy, and
        // that having to pass `--allow-unused-prefix` is an unfortunate side-effect of not knowing
        // whether the test author actually wanted revision-specific check prefixes or not.
        //
        // TL;DR We may not want to conflate `compiletest` revisions and `FileCheck` prefixes.

        // HACK: tests are allowed to use a revision name as a check prefix.
        if let Some(rev) = self.revision {
            filecheck.arg("--check-prefix").arg(rev);
        }

        // HACK: the filecheck tool normally fails if a prefix is defined but not used. However,
        // sometimes revisions are used to specify *compiletest* directives which are not FileCheck
        // concerns.
        filecheck.arg("--allow-unused-prefixes");

        // Provide more context on failures.
        filecheck.args(&["--dump-input-context", "100"]);

        // Add custom flags supplied by the `filecheck-flags:` test header.
        filecheck.args(&self.props.filecheck_flags);

        // FIXME(jieyouxu): don't pass an empty Path
        self.compose_and_run(filecheck, Utf8Path::new(""), None, None)
    }

    fn charset() -> &'static str {
        // FreeBSD 10.1 defaults to GDB 6.1.1 which doesn't support "auto" charset
        if cfg!(target_os = "freebsd") { "ISO-8859-1" } else { "UTF-8" }
    }

    fn compare_to_default_rustdoc(&mut self, out_dir: &Utf8Path) {
        if !self.config.has_html_tidy {
            return;
        }
        println!("info: generating a diff against nightly rustdoc");

        let suffix =
            self.safe_revision().map_or("nightly".into(), |path| path.to_owned() + "-nightly");
        let compare_dir = output_base_dir(self.config, self.testpaths, Some(&suffix));
        remove_and_create_dir_all(&compare_dir).unwrap_or_else(|e| {
            panic!("failed to remove and recreate output directory `{compare_dir}`: {e}")
        });

        // We need to create a new struct for the lifetimes on `config` to work.
        let new_rustdoc = TestCx {
            config: &Config {
                // FIXME: use beta or a user-specified rustdoc instead of
                // hardcoding the default toolchain
                rustdoc_path: Some("rustdoc".into()),
                // Needed for building auxiliary docs below
                rustc_path: "rustc".into(),
                ..self.config.clone()
            },
            ..*self
        };

        let output_file = TargetLocation::ThisDirectory(new_rustdoc.aux_output_dir_name());
        let mut rustc = new_rustdoc.make_compile_args(
            &new_rustdoc.testpaths.file,
            output_file,
            Emit::None,
            AllowUnused::Yes,
            LinkToAux::Yes,
            Vec::new(),
        );
        let aux_dir = new_rustdoc.aux_output_dir();
        new_rustdoc.build_all_auxiliary(&new_rustdoc.testpaths, &aux_dir, &mut rustc);

        let proc_res = new_rustdoc.document(&compare_dir, &new_rustdoc.testpaths);
        if !proc_res.status.success() {
            eprintln!("failed to run nightly rustdoc");
            return;
        }

        #[rustfmt::skip]
        let tidy_args = [
            "--new-blocklevel-tags", "rustdoc-search,rustdoc-toolbar",
            "--indent", "yes",
            "--indent-spaces", "2",
            "--wrap", "0",
            "--show-warnings", "no",
            "--markup", "yes",
            "--quiet", "yes",
            "-modify",
        ];
        let tidy_dir = |dir| {
            for entry in walkdir::WalkDir::new(dir) {
                let entry = entry.expect("failed to read file");
                if entry.file_type().is_file()
                    && entry.path().extension().and_then(|p| p.to_str()) == Some("html")
                {
                    let status =
                        Command::new("tidy").args(&tidy_args).arg(entry.path()).status().unwrap();
                    // `tidy` returns 1 if it modified the file.
                    assert!(status.success() || status.code() == Some(1));
                }
            }
        };
        tidy_dir(out_dir);
        tidy_dir(&compare_dir);

        let pager = {
            let output = Command::new("git").args(&["config", "--get", "core.pager"]).output().ok();
            output.and_then(|out| {
                if out.status.success() {
                    Some(String::from_utf8(out.stdout).expect("invalid UTF8 in git pager"))
                } else {
                    None
                }
            })
        };

        let diff_filename = format!("build/tmp/rustdoc-compare-{}.diff", std::process::id());

        if !write_filtered_diff(
            &diff_filename,
            out_dir,
            &compare_dir,
            self.config.verbose,
            |file_type, extension| {
                file_type.is_file() && (extension == Some("html") || extension == Some("js"))
            },
        ) {
            return;
        }

        match self.config.color {
            ColorConfig::AlwaysColor => colored::control::set_override(true),
            ColorConfig::NeverColor => colored::control::set_override(false),
            _ => {}
        }

        if let Some(pager) = pager {
            let pager = pager.trim();
            if self.config.verbose {
                eprintln!("using pager {}", pager);
            }
            let output = Command::new(pager)
                // disable paging; we want this to be non-interactive
                .env("PAGER", "")
                .stdin(File::open(&diff_filename).unwrap())
                // Capture output and print it explicitly so it will in turn be
                // captured by libtest.
                .output()
                .unwrap();
            assert!(output.status.success());
            println!("{}", String::from_utf8_lossy(&output.stdout));
            eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        } else {
            use colored::Colorize;
            eprintln!("warning: no pager configured, falling back to unified diff");
            eprintln!(
                "help: try configuring a git pager (e.g. `delta`) with `git config --global core.pager delta`"
            );
            let mut out = io::stdout();
            let mut diff = BufReader::new(File::open(&diff_filename).unwrap());
            let mut line = Vec::new();
            loop {
                line.truncate(0);
                match diff.read_until(b'\n', &mut line) {
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(e) => eprintln!("ERROR: {:?}", e),
                }
                match String::from_utf8(line.clone()) {
                    Ok(line) => {
                        if line.starts_with('+') {
                            write!(&mut out, "{}", line.green()).unwrap();
                        } else if line.starts_with('-') {
                            write!(&mut out, "{}", line.red()).unwrap();
                        } else if line.starts_with('@') {
                            write!(&mut out, "{}", line.blue()).unwrap();
                        } else {
                            out.write_all(line.as_bytes()).unwrap();
                        }
                    }
                    Err(_) => {
                        write!(&mut out, "{}", String::from_utf8_lossy(&line).reversed()).unwrap();
                    }
                }
            }
        };
    }

    fn get_lines(&self, path: &Utf8Path, mut other_files: Option<&mut Vec<String>>) -> Vec<usize> {
        let content = fs::read_to_string(path.as_std_path()).unwrap();
        let mut ignore = false;
        content
            .lines()
            .enumerate()
            .filter_map(|(line_nb, line)| {
                if (line.trim_start().starts_with("pub mod ")
                    || line.trim_start().starts_with("mod "))
                    && line.ends_with(';')
                {
                    if let Some(ref mut other_files) = other_files {
                        other_files.push(line.rsplit("mod ").next().unwrap().replace(';', ""));
                    }
                    None
                } else {
                    let sline = line.rsplit("///").next().unwrap();
                    let line = sline.trim_start();
                    if line.starts_with("```") {
                        if ignore {
                            ignore = false;
                            None
                        } else {
                            ignore = true;
                            Some(line_nb + 1)
                        }
                    } else {
                        None
                    }
                }
            })
            .collect()
    }

    /// This method is used for `//@ check-test-line-numbers-match`.
    ///
    /// It checks that doctests line in the displayed doctest "name" matches where they are
    /// defined in source code.
    fn check_rustdoc_test_option(&self, res: ProcRes) {
        let mut other_files = Vec::new();
        let mut files: HashMap<String, Vec<usize>> = HashMap::new();
        let normalized = fs::canonicalize(&self.testpaths.file).expect("failed to canonicalize");
        let normalized = normalized.to_str().unwrap().replace('\\', "/");
        files.insert(normalized, self.get_lines(&self.testpaths.file, Some(&mut other_files)));
        for other_file in other_files {
            let mut path = self.testpaths.file.clone();
            path.set_file_name(&format!("{}.rs", other_file));
            let path = path.canonicalize_utf8().expect("failed to canonicalize");
            let normalized = path.as_str().replace('\\', "/");
            files.insert(normalized, self.get_lines(&path, None));
        }

        let mut tested = 0;
        for _ in res.stdout.split('\n').filter(|s| s.starts_with("test ")).inspect(|s| {
            if let Some((left, right)) = s.split_once(" - ") {
                let path = left.rsplit("test ").next().unwrap();
                let path = fs::canonicalize(&path).expect("failed to canonicalize");
                let path = path.to_str().unwrap().replace('\\', "/");
                if let Some(ref mut v) = files.get_mut(&path) {
                    tested += 1;
                    let mut iter = right.split("(line ");
                    iter.next();
                    let line = iter
                        .next()
                        .unwrap_or(")")
                        .split(')')
                        .next()
                        .unwrap_or("0")
                        .parse()
                        .unwrap_or(0);
                    if let Ok(pos) = v.binary_search(&line) {
                        v.remove(pos);
                    } else {
                        self.fatal_proc_rec(
                            &format!("Not found doc test: \"{}\" in \"{}\":{:?}", s, path, v),
                            &res,
                        );
                    }
                }
            }
        }) {}
        if tested == 0 {
            self.fatal_proc_rec(&format!("No test has been found... {:?}", files), &res);
        } else {
            for (entry, v) in &files {
                if !v.is_empty() {
                    self.fatal_proc_rec(
                        &format!(
                            "Not found test at line{} \"{}\":{:?}",
                            if v.len() > 1 { "s" } else { "" },
                            entry,
                            v
                        ),
                        &res,
                    );
                }
            }
        }
    }

    fn force_color_svg(&self) -> bool {
        self.props.compile_flags.iter().any(|s| s.contains("--color=always"))
    }

    fn load_compare_outputs(
        &self,
        proc_res: &ProcRes,
        output_kind: TestOutput,
        explicit_format: bool,
    ) -> usize {
        let stderr_bits = format!("{}bit.stderr", self.config.get_pointer_width());
        let (stderr_kind, stdout_kind) = match output_kind {
            TestOutput::Compile => (
                if self.force_color_svg() {
                    if self.config.target.contains("windows") {
                        // We single out Windows here because some of the CLI coloring is
                        // specifically changed for Windows.
                        UI_WINDOWS_SVG
                    } else {
                        UI_SVG
                    }
                } else if self.props.stderr_per_bitwidth {
                    &stderr_bits
                } else {
                    UI_STDERR
                },
                UI_STDOUT,
            ),
            TestOutput::Run => (UI_RUN_STDERR, UI_RUN_STDOUT),
        };

        let expected_stderr = self.load_expected_output(stderr_kind);
        let expected_stdout = self.load_expected_output(stdout_kind);

        let mut normalized_stdout =
            self.normalize_output(&proc_res.stdout, &self.props.normalize_stdout);
        match output_kind {
            TestOutput::Run if self.config.remote_test_client.is_some() => {
                // When tests are run using the remote-test-client, the string
                // 'uploaded "$TEST_BUILD_DIR/<test_executable>, waiting for result"'
                // is printed to stdout by the client and then captured in the ProcRes,
                // so it needs to be removed when comparing the run-pass test execution output.
                normalized_stdout = static_regex!(
                    "^uploaded \"\\$TEST_BUILD_DIR(/[[:alnum:]_\\-.]+)+\", waiting for result\n"
                )
                .replace(&normalized_stdout, "")
                .to_string();
                // When there is a panic, the remote-test-client also prints "died due to signal";
                // that needs to be removed as well.
                normalized_stdout = static_regex!("^died due to signal [0-9]+\n")
                    .replace(&normalized_stdout, "")
                    .to_string();
                // FIXME: it would be much nicer if we could just tell the remote-test-client to not
                // print these things.
            }
            _ => {}
        };

        let stderr = if self.force_color_svg() {
            anstyle_svg::Term::new().render_svg(&proc_res.stderr)
        } else if explicit_format {
            proc_res.stderr.clone()
        } else {
            json::extract_rendered(&proc_res.stderr)
        };

        let normalized_stderr = self.normalize_output(&stderr, &self.props.normalize_stderr);
        let mut errors = 0;
        match output_kind {
            TestOutput::Compile => {
                if !self.props.dont_check_compiler_stdout {
                    if self
                        .compare_output(
                            stdout_kind,
                            &normalized_stdout,
                            &proc_res.stdout,
                            &expected_stdout,
                        )
                        .should_error()
                    {
                        errors += 1;
                    }
                }
                if !self.props.dont_check_compiler_stderr {
                    if self
                        .compare_output(stderr_kind, &normalized_stderr, &stderr, &expected_stderr)
                        .should_error()
                    {
                        errors += 1;
                    }
                }
            }
            TestOutput::Run => {
                if self
                    .compare_output(
                        stdout_kind,
                        &normalized_stdout,
                        &proc_res.stdout,
                        &expected_stdout,
                    )
                    .should_error()
                {
                    errors += 1;
                }

                if self
                    .compare_output(stderr_kind, &normalized_stderr, &stderr, &expected_stderr)
                    .should_error()
                {
                    errors += 1;
                }
            }
        }
        errors
    }

    fn normalize_output(&self, output: &str, custom_rules: &[(String, String)]) -> String {
        // Crude heuristic to detect when the output should have JSON-specific
        // normalization steps applied.
        let rflags = self.props.run_flags.join(" ");
        let cflags = self.props.compile_flags.join(" ");
        let json = rflags.contains("--format json")
            || rflags.contains("--format=json")
            || cflags.contains("--error-format json")
            || cflags.contains("--error-format pretty-json")
            || cflags.contains("--error-format=json")
            || cflags.contains("--error-format=pretty-json")
            || cflags.contains("--output-format json")
            || cflags.contains("--output-format=json");

        let mut normalized = output.to_string();

        let mut normalize_path = |from: &Utf8Path, to: &str| {
            let from = if json { &from.as_str().replace("\\", "\\\\") } else { from.as_str() };

            normalized = normalized.replace(from, to);
        };

        let parent_dir = self.testpaths.file.parent().unwrap();
        normalize_path(parent_dir, "$DIR");

        if self.props.remap_src_base {
            let mut remapped_parent_dir = Utf8PathBuf::from(FAKE_SRC_BASE);
            if self.testpaths.relative_dir != Utf8Path::new("") {
                remapped_parent_dir.push(&self.testpaths.relative_dir);
            }
            normalize_path(&remapped_parent_dir, "$DIR");
        }

        let base_dir = Utf8Path::new("/rustc/FAKE_PREFIX");
        // Fake paths into the libstd/libcore
        normalize_path(&base_dir.join("library"), "$SRC_DIR");
        // `ui-fulldeps` tests can show paths to the compiler source when testing macros from
        // `rustc_macros`
        // eg. /home/user/rust/compiler
        normalize_path(&base_dir.join("compiler"), "$COMPILER_DIR");

        // Real paths into the libstd/libcore
        let rust_src_dir = &self.config.sysroot_base.join("lib/rustlib/src/rust");
        rust_src_dir.try_exists().expect(&*format!("{} should exists", rust_src_dir));
        let rust_src_dir = rust_src_dir.read_link_utf8().unwrap_or(rust_src_dir.to_path_buf());
        normalize_path(&rust_src_dir.join("library"), "$SRC_DIR_REAL");

        // eg.
        // /home/user/rust/build/x86_64-unknown-linux-gnu/test/ui/<test_dir>/$name.$revision.$mode/
        normalize_path(&self.output_base_dir(), "$TEST_BUILD_DIR");
        // Same as above, but with a canonicalized path.
        // This is required because some tests print canonical paths inside test build directory,
        // so if the build directory is a symlink, normalization doesn't help.
        //
        // NOTE: There are also tests which print the non-canonical name, so we need both this and
        // the above normalizations.
        normalize_path(&self.output_base_dir().canonicalize_utf8().unwrap(), "$TEST_BUILD_DIR");
        // eg. /home/user/rust/build
        normalize_path(&self.config.build_root, "$BUILD_DIR");

        if json {
            // escaped newlines in json strings should be readable
            // in the stderr files. There's no point in being correct,
            // since only humans process the stderr files.
            // Thus we just turn escaped newlines back into newlines.
            normalized = normalized.replace("\\n", "\n");
        }

        // If there are `$SRC_DIR` normalizations with line and column numbers, then replace them
        // with placeholders as we do not want tests needing updated when compiler source code
        // changes.
        // eg. $SRC_DIR/libcore/mem.rs:323:14 becomes $SRC_DIR/libcore/mem.rs:LL:COL
        normalized = static_regex!("SRC_DIR(.+):\\d+:\\d+(: \\d+:\\d+)?")
            .replace_all(&normalized, "SRC_DIR$1:LL:COL")
            .into_owned();

        normalized = Self::normalize_platform_differences(&normalized);

        // Normalize long type name hash.
        normalized =
            static_regex!(r"\$TEST_BUILD_DIR/(?P<filename>[^\.]+).long-type-(?P<hash>\d+).txt")
                .replace_all(&normalized, |caps: &Captures<'_>| {
                    format!(
                        "$TEST_BUILD_DIR/{filename}.long-type-$LONG_TYPE_HASH.txt",
                        filename = &caps["filename"]
                    )
                })
                .into_owned();

        normalized = normalized.replace("\t", "\\t"); // makes tabs visible

        // Remove test annotations like `//~ ERROR text` from the output,
        // since they duplicate actual errors and make the output hard to read.
        // This mirrors the regex in src/tools/tidy/src/style.rs, please update
        // both if either are changed.
        normalized =
            static_regex!("\\s*//(\\[.*\\])?~.*").replace_all(&normalized, "").into_owned();

        // This code normalizes various hashes in v0 symbol mangling that is
        // emitted in the ui and mir-opt tests.
        let v0_crate_hash_prefix_re = static_regex!(r"_R.*?Cs[0-9a-zA-Z]+_");
        let v0_crate_hash_re = static_regex!(r"Cs[0-9a-zA-Z]+_");

        const V0_CRATE_HASH_PLACEHOLDER: &str = r"CsCRATE_HASH_";
        if v0_crate_hash_prefix_re.is_match(&normalized) {
            // Normalize crate hash
            normalized =
                v0_crate_hash_re.replace_all(&normalized, V0_CRATE_HASH_PLACEHOLDER).into_owned();
        }

        let v0_back_ref_prefix_re = static_regex!(r"\(_R.*?B[0-9a-zA-Z]_");
        let v0_back_ref_re = static_regex!(r"B[0-9a-zA-Z]_");

        const V0_BACK_REF_PLACEHOLDER: &str = r"B<REF>_";
        if v0_back_ref_prefix_re.is_match(&normalized) {
            // Normalize back references (see RFC 2603)
            normalized =
                v0_back_ref_re.replace_all(&normalized, V0_BACK_REF_PLACEHOLDER).into_owned();
        }

        // AllocId are numbered globally in a compilation session. This can lead to changes
        // depending on the exact compilation flags and host architecture. Meanwhile, we want
        // to keep them numbered, to see if the same id appears multiple times.
        // So we remap to deterministic numbers that only depend on the subset of allocations
        // that actually appear in the output.
        // We use uppercase ALLOC to distinguish from the non-normalized version.
        {
            let mut seen_allocs = indexmap::IndexSet::new();

            // The alloc-id appears in pretty-printed allocations.
            normalized = static_regex!(
                r"╾─*a(lloc)?([0-9]+)(\+0x[0-9]+)?(<imm>)?( \([0-9]+ ptr bytes\))?─*╼"
            )
            .replace_all(&normalized, |caps: &Captures<'_>| {
                // Renumber the captured index.
                let index = caps.get(2).unwrap().as_str().to_string();
                let (index, _) = seen_allocs.insert_full(index);
                let offset = caps.get(3).map_or("", |c| c.as_str());
                let imm = caps.get(4).map_or("", |c| c.as_str());
                // Do not bother keeping it pretty, just make it deterministic.
                format!("╾ALLOC{index}{offset}{imm}╼")
            })
            .into_owned();

            // The alloc-id appears in a sentence.
            normalized = static_regex!(r"\balloc([0-9]+)\b")
                .replace_all(&normalized, |caps: &Captures<'_>| {
                    let index = caps.get(1).unwrap().as_str().to_string();
                    let (index, _) = seen_allocs.insert_full(index);
                    format!("ALLOC{index}")
                })
                .into_owned();
        }

        // Custom normalization rules
        for rule in custom_rules {
            let re = Regex::new(&rule.0).expect("bad regex in custom normalization rule");
            normalized = re.replace_all(&normalized, &rule.1[..]).into_owned();
        }
        normalized
    }

    /// Normalize output differences across platforms. Generally changes Windows output to be more
    /// Unix-like.
    ///
    /// Replaces backslashes in paths with forward slashes, and replaces CRLF line endings
    /// with LF.
    fn normalize_platform_differences(output: &str) -> String {
        let output = output.replace(r"\\", r"\");

        // Used to find Windows paths.
        //
        // It's not possible to detect paths in the error messages generally, but this is a
        // decent enough heuristic.
        static_regex!(
                r#"(?x)
                (?:
                  # Match paths that don't include spaces.
                  (?:\\[\pL\pN\.\-_']+)+\.\pL+
                |
                  # If the path starts with a well-known root, then allow spaces and no file extension.
                  \$(?:DIR|SRC_DIR|TEST_BUILD_DIR|BUILD_DIR|LIB_DIR)(?:\\[\pL\pN\.\-_'\ ]+)+
                )"#
            )
            .replace_all(&output, |caps: &Captures<'_>| {
                println!("{}", &caps[0]);
                caps[0].replace(r"\", "/")
            })
            .replace("\r\n", "\n")
    }

    fn expected_output_path(&self, kind: &str) -> Utf8PathBuf {
        let mut path =
            expected_output_path(&self.testpaths, self.revision, &self.config.compare_mode, kind);

        if !path.exists() {
            if let Some(CompareMode::Polonius) = self.config.compare_mode {
                path = expected_output_path(&self.testpaths, self.revision, &None, kind);
            }
        }

        if !path.exists() {
            path = expected_output_path(&self.testpaths, self.revision, &None, kind);
        }

        path
    }

    fn load_expected_output(&self, kind: &str) -> String {
        let path = self.expected_output_path(kind);
        if path.exists() {
            match self.load_expected_output_from_path(&path) {
                Ok(x) => x,
                Err(x) => self.fatal(&x),
            }
        } else {
            String::new()
        }
    }

    fn load_expected_output_from_path(&self, path: &Utf8Path) -> Result<String, String> {
        fs::read_to_string(path)
            .map_err(|err| format!("failed to load expected output from `{}`: {}", path, err))
    }

    fn delete_file(&self, file: &Utf8Path) {
        if !file.exists() {
            // Deleting a nonexistent file would error.
            return;
        }
        if let Err(e) = fs::remove_file(file.as_std_path()) {
            self.fatal(&format!("failed to delete `{}`: {}", file, e,));
        }
    }

    fn compare_output(
        &self,
        stream: &str,
        actual: &str,
        actual_unnormalized: &str,
        expected: &str,
    ) -> CompareOutcome {
        let expected_path =
            expected_output_path(self.testpaths, self.revision, &self.config.compare_mode, stream);

        if self.config.bless && actual.is_empty() && expected_path.exists() {
            self.delete_file(&expected_path);
        }

        let are_different = match (self.force_color_svg(), expected.find('\n'), actual.find('\n')) {
            // FIXME: We ignore the first line of SVG files
            // because the width parameter is non-deterministic.
            (true, Some(nl_e), Some(nl_a)) => expected[nl_e..] != actual[nl_a..],
            _ => expected != actual,
        };
        if !are_different {
            return CompareOutcome::Same;
        }

        // Wrapper tools set by `runner` might provide extra output on failure,
        // for example a WebAssembly runtime might print the stack trace of an
        // `unreachable` instruction by default.
        let compare_output_by_lines = self.config.runner.is_some();

        let tmp;
        let (expected, actual): (&str, &str) = if compare_output_by_lines {
            let actual_lines: HashSet<_> = actual.lines().collect();
            let expected_lines: Vec<_> = expected.lines().collect();
            let mut used = expected_lines.clone();
            used.retain(|line| actual_lines.contains(line));
            // check if `expected` contains a subset of the lines of `actual`
            if used.len() == expected_lines.len() && (expected.is_empty() == actual.is_empty()) {
                return CompareOutcome::Same;
            }
            if expected_lines.is_empty() {
                // if we have no lines to check, force a full overwite
                ("", actual)
            } else {
                tmp = (expected_lines.join("\n"), used.join("\n"));
                (&tmp.0, &tmp.1)
            }
        } else {
            (expected, actual)
        };

        // Write the actual output to a file in build/
        let test_name = self.config.compare_mode.as_ref().map_or("", |m| m.to_str());
        let actual_path = self
            .output_base_name()
            .with_extra_extension(self.revision.unwrap_or(""))
            .with_extra_extension(test_name)
            .with_extra_extension(stream);

        if let Err(err) = fs::write(&actual_path, &actual) {
            self.fatal(&format!("failed to write {stream} to `{actual_path:?}`: {err}",));
        }
        println!("Saved the actual {stream} to {actual_path:?}");

        if !self.config.bless {
            if expected.is_empty() {
                println!("normalized {}:\n{}\n", stream, actual);
            } else {
                self.show_diff(
                    stream,
                    &expected_path,
                    &actual_path,
                    expected,
                    actual,
                    actual_unnormalized,
                );
            }
        } else {
            // Delete non-revision .stderr/.stdout file if revisions are used.
            // Without this, we'd just generate the new files and leave the old files around.
            if self.revision.is_some() {
                let old =
                    expected_output_path(self.testpaths, None, &self.config.compare_mode, stream);
                self.delete_file(&old);
            }

            if !actual.is_empty() {
                if let Err(err) = fs::write(&expected_path, &actual) {
                    self.fatal(&format!("failed to write {stream} to `{expected_path:?}`: {err}"));
                }
                println!("Blessing the {stream} of {test_name} in {expected_path:?}");
            }
        }

        println!("\nThe actual {0} differed from the expected {0}.", stream);

        if self.config.bless { CompareOutcome::Blessed } else { CompareOutcome::Differed }
    }

    /// Returns whether to show the full stderr/stdout.
    fn show_diff(
        &self,
        stream: &str,
        expected_path: &Utf8Path,
        actual_path: &Utf8Path,
        expected: &str,
        actual: &str,
        actual_unnormalized: &str,
    ) {
        eprintln!("diff of {stream}:\n");
        if let Some(diff_command) = self.config.diff_command.as_deref() {
            let mut args = diff_command.split_whitespace();
            let name = args.next().unwrap();
            match Command::new(name).args(args).args([expected_path, actual_path]).output() {
                Err(err) => {
                    self.fatal(&format!(
                        "failed to call custom diff command `{diff_command}`: {err}"
                    ));
                }
                Ok(output) => {
                    let output = String::from_utf8_lossy(&output.stdout);
                    eprint!("{output}");
                }
            }
        } else {
            eprint!("{}", write_diff(expected, actual, 3));
        }

        // NOTE: argument order is important, we need `actual` to be on the left so the line number match up when we compare it to `actual_unnormalized` below.
        let diff_results = make_diff(actual, expected, 0);

        let (mut mismatches_normalized, mut mismatch_line_nos) = (String::new(), vec![]);
        for hunk in diff_results {
            let mut line_no = hunk.line_number;
            for line in hunk.lines {
                // NOTE: `Expected` is actually correct here, the argument order is reversed so our line numbers match up
                if let DiffLine::Expected(normalized) = line {
                    mismatches_normalized += &normalized;
                    mismatches_normalized += "\n";
                    mismatch_line_nos.push(line_no);
                    line_no += 1;
                }
            }
        }
        let mut mismatches_unnormalized = String::new();
        let diff_normalized = make_diff(actual, actual_unnormalized, 0);
        for hunk in diff_normalized {
            if mismatch_line_nos.contains(&hunk.line_number) {
                for line in hunk.lines {
                    if let DiffLine::Resulting(unnormalized) = line {
                        mismatches_unnormalized += &unnormalized;
                        mismatches_unnormalized += "\n";
                    }
                }
            }
        }

        let normalized_diff = make_diff(&mismatches_normalized, &mismatches_unnormalized, 0);
        // HACK: instead of checking if each hunk is empty, this only checks if the whole input is empty. we should be smarter about this so we don't treat added or removed output as normalized.
        if !normalized_diff.is_empty()
            && !mismatches_unnormalized.is_empty()
            && !mismatches_normalized.is_empty()
        {
            eprintln!("Note: some mismatched output was normalized before being compared");
            // FIXME: respect diff_command
            eprint!("{}", write_diff(&mismatches_unnormalized, &mismatches_normalized, 0));
        }
    }

    fn check_and_prune_duplicate_outputs(
        &self,
        proc_res: &ProcRes,
        modes: &[CompareMode],
        require_same_modes: &[CompareMode],
    ) {
        for kind in UI_EXTENSIONS {
            let canon_comparison_path =
                expected_output_path(&self.testpaths, self.revision, &None, kind);

            let canon = match self.load_expected_output_from_path(&canon_comparison_path) {
                Ok(canon) => canon,
                _ => continue,
            };
            let bless = self.config.bless;
            let check_and_prune_duplicate_outputs = |mode: &CompareMode, require_same: bool| {
                let examined_path =
                    expected_output_path(&self.testpaths, self.revision, &Some(mode.clone()), kind);

                // If there is no output, there is nothing to do
                let examined_content = match self.load_expected_output_from_path(&examined_path) {
                    Ok(content) => content,
                    _ => return,
                };

                let is_duplicate = canon == examined_content;

                match (bless, require_same, is_duplicate) {
                    // If we're blessing and the output is the same, then delete the file.
                    (true, _, true) => {
                        self.delete_file(&examined_path);
                    }
                    // If we want them to be the same, but they are different, then error.
                    // We do this wether we bless or not
                    (_, true, false) => {
                        self.fatal_proc_rec(
                            &format!("`{}` should not have different output from base test!", kind),
                            proc_res,
                        );
                    }
                    _ => {}
                }
            };
            for mode in modes {
                check_and_prune_duplicate_outputs(mode, false);
            }
            for mode in require_same_modes {
                check_and_prune_duplicate_outputs(mode, true);
            }
        }
    }

    fn create_stamp(&self) {
        let stamp_file_path = stamp_file_path(&self.config, self.testpaths, self.revision);
        fs::write(&stamp_file_path, compute_stamp_hash(&self.config)).unwrap();
    }

    fn init_incremental_test(&self) {
        // (See `run_incremental_test` for an overview of how incremental tests work.)

        // Before any of the revisions have executed, create the
        // incremental workproduct directory.  Delete any old
        // incremental work products that may be there from prior
        // runs.
        let incremental_dir = self.props.incremental_dir.as_ref().unwrap();
        if incremental_dir.exists() {
            // Canonicalizing the path will convert it to the //?/ format
            // on Windows, which enables paths longer than 260 character
            let canonicalized = incremental_dir.canonicalize().unwrap();
            fs::remove_dir_all(canonicalized).unwrap();
        }
        fs::create_dir_all(&incremental_dir).unwrap();

        if self.config.verbose {
            println!("init_incremental_test: incremental_dir={incremental_dir}");
        }
    }
}

struct ProcArgs {
    prog: OsString,
    args: Vec<OsString>,
}

pub struct ProcRes {
    status: ExitStatus,
    stdout: String,
    stderr: String,
    truncated: Truncated,
    cmdline: String,
}

impl ProcRes {
    pub fn print_info(&self) {
        fn render(name: &str, contents: &str) -> String {
            let contents = json::extract_rendered(contents);
            let contents = contents.trim_end();
            if contents.is_empty() {
                format!("{name}: none")
            } else {
                format!(
                    "\
                     --- {name} -------------------------------\n\
                     {contents}\n\
                     ------------------------------------------",
                )
            }
        }

        println!(
            "status: {}\ncommand: {}\n{}\n{}\n",
            self.status,
            self.cmdline,
            render("stdout", &self.stdout),
            render("stderr", &self.stderr),
        );
    }

    pub fn fatal(&self, err: Option<&str>, on_failure: impl FnOnce()) -> ! {
        if let Some(e) = err {
            println!("\nerror: {}", e);
        }
        self.print_info();
        on_failure();
        // Use resume_unwind instead of panic!() to prevent a panic message + backtrace from
        // compiletest, which is unnecessary noise.
        std::panic::resume_unwind(Box::new(()));
    }
}

#[derive(Debug)]
enum TargetLocation {
    ThisFile(Utf8PathBuf),
    ThisDirectory(Utf8PathBuf),
}

enum AllowUnused {
    Yes,
    No,
}

enum LinkToAux {
    Yes,
    No,
}

#[derive(Debug, PartialEq)]
enum AuxType {
    Bin,
    Lib,
    Dylib,
    ProcMacro,
}

/// Outcome of comparing a stream to a blessed file,
/// e.g. `.stderr` and `.fixed`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CompareOutcome {
    /// Expected and actual outputs are the same
    Same,
    /// Outputs differed but were blessed
    Blessed,
    /// Outputs differed and an error should be emitted
    Differed,
}

impl CompareOutcome {
    fn should_error(&self) -> bool {
        matches!(self, CompareOutcome::Differed)
    }
}
