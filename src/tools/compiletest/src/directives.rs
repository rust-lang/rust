use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::io::prelude::*;
use std::process::Command;

use camino::{Utf8Path, Utf8PathBuf};
use semver::Version;
use tracing::*;

use crate::common::{CodegenBackend, Config, Debugger, FailMode, PassMode, RunFailMode, TestMode};
use crate::debuggers::{extract_cdb_version, extract_gdb_version};
use crate::directives::auxiliary::{AuxProps, parse_and_update_aux};
use crate::directives::directive_names::{
    KNOWN_DIRECTIVE_NAMES, KNOWN_HTMLDOCCK_DIRECTIVE_NAMES, KNOWN_JSONDOCCK_DIRECTIVE_NAMES,
};
use crate::directives::needs::CachedNeedsConditions;
use crate::errors::ErrorKind;
use crate::executor::{CollectedTestDesc, ShouldPanic};
use crate::help;
use crate::util::static_regex;

pub(crate) mod auxiliary;
mod cfg;
mod directive_names;
mod needs;
#[cfg(test)]
mod tests;

pub struct DirectivesCache {
    needs: CachedNeedsConditions,
}

impl DirectivesCache {
    pub fn load(config: &Config) -> Self {
        Self { needs: CachedNeedsConditions::load(config) }
    }
}

/// Properties which must be known very early, before actually running
/// the test.
#[derive(Default)]
pub struct EarlyProps {
    /// Auxiliary crates that should be built and made available to this test.
    /// Included in [`EarlyProps`] so that the indicated files can participate
    /// in up-to-date checking. Building happens via [`TestProps::aux`] instead.
    pub(crate) aux: AuxProps,
    pub revisions: Vec<String>,
}

impl EarlyProps {
    pub fn from_file(config: &Config, testfile: &Utf8Path) -> Self {
        let file = File::open(testfile.as_std_path()).expect("open test file to parse earlyprops");
        Self::from_reader(config, testfile, file)
    }

    pub fn from_reader<R: Read>(config: &Config, testfile: &Utf8Path, rdr: R) -> Self {
        let mut props = EarlyProps::default();
        let mut poisoned = false;
        iter_directives(
            config.mode,
            &mut poisoned,
            testfile,
            rdr,
            &mut |DirectiveLine { line_number, raw_directive: ln, .. }| {
                parse_and_update_aux(config, ln, testfile, line_number, &mut props.aux);
                config.parse_and_update_revisions(testfile, line_number, ln, &mut props.revisions);
            },
        );

        if poisoned {
            eprintln!("errors encountered during EarlyProps parsing: {}", testfile);
            panic!("errors encountered during EarlyProps parsing");
        }

        props
    }
}

#[derive(Clone, Debug)]
pub struct TestProps {
    // Lines that should be expected, in order, on standard out
    pub error_patterns: Vec<String>,
    // Regexes that should be expected, in order, on standard out
    pub regex_error_patterns: Vec<String>,
    // Extra flags to pass to the compiler
    pub compile_flags: Vec<String>,
    // Extra flags to pass when the compiled code is run (such as --bench)
    pub run_flags: Vec<String>,
    /// Extra flags to pass to rustdoc but not the compiler.
    pub doc_flags: Vec<String>,
    // If present, the name of a file that this test should match when
    // pretty-printed
    pub pp_exact: Option<Utf8PathBuf>,
    /// Auxiliary crates that should be built and made available to this test.
    pub(crate) aux: AuxProps,
    // Environment settings to use for compiling
    pub rustc_env: Vec<(String, String)>,
    // Environment variables to unset prior to compiling.
    // Variables are unset before applying 'rustc_env'.
    pub unset_rustc_env: Vec<String>,
    // Environment settings to use during execution
    pub exec_env: Vec<(String, String)>,
    // Environment variables to unset prior to execution.
    // Variables are unset before applying 'exec_env'
    pub unset_exec_env: Vec<String>,
    // Build documentation for all specified aux-builds as well
    pub build_aux_docs: bool,
    /// Build the documentation for each crate in a unique output directory.
    /// Uses `<root output directory>/docs/<test name>/doc`.
    pub unique_doc_out_dir: bool,
    // Flag to force a crate to be built with the host architecture
    pub force_host: bool,
    // Check stdout for error-pattern output as well as stderr
    pub check_stdout: bool,
    // Check stdout & stderr for output of run-pass test
    pub check_run_results: bool,
    // For UI tests, allows compiler to generate arbitrary output to stdout
    pub dont_check_compiler_stdout: bool,
    // For UI tests, allows compiler to generate arbitrary output to stderr
    pub dont_check_compiler_stderr: bool,
    // Don't force a --crate-type=dylib flag on the command line
    //
    // Set this for example if you have an auxiliary test file that contains
    // a proc-macro and needs `#![crate_type = "proc-macro"]`. This ensures
    // that the aux file is compiled as a `proc-macro` and not as a `dylib`.
    pub no_prefer_dynamic: bool,
    // Which pretty mode are we testing with, default to 'normal'
    pub pretty_mode: String,
    // Only compare pretty output and don't try compiling
    pub pretty_compare_only: bool,
    // Patterns which must not appear in the output of a cfail test.
    pub forbid_output: Vec<String>,
    // Revisions to test for incremental compilation.
    pub revisions: Vec<String>,
    // Directory (if any) to use for incremental compilation.  This is
    // not set by end-users; rather it is set by the incremental
    // testing harness and used when generating compilation
    // arguments. (In particular, it propagates to the aux-builds.)
    pub incremental_dir: Option<Utf8PathBuf>,
    // If `true`, this test will use incremental compilation.
    //
    // This can be set manually with the `incremental` directive, or implicitly
    // by being a part of an incremental mode test. Using the `incremental`
    // directive should be avoided if possible; using an incremental mode test is
    // preferred. Incremental mode tests support multiple passes, which can
    // verify that the incremental cache can be loaded properly after being
    // created. Just setting the directive will only verify the behavior with
    // creating an incremental cache, but doesn't check that it is created
    // correctly.
    //
    // Compiletest will create the incremental directory, and ensure it is
    // empty before the test starts. Incremental mode tests will reuse the
    // incremental directory between passes in the same test.
    pub incremental: bool,
    // If `true`, this test is a known bug.
    //
    // When set, some requirements are relaxed. Currently, this only means no
    // error annotations are needed, but this may be updated in the future to
    // include other relaxations.
    pub known_bug: bool,
    // How far should the test proceed while still passing.
    pass_mode: Option<PassMode>,
    // Ignore `--pass` overrides from the command line for this test.
    ignore_pass: bool,
    // How far this test should proceed to start failing.
    pub fail_mode: Option<FailMode>,
    // rustdoc will test the output of the `--test` option
    pub check_test_line_numbers_match: bool,
    // customized normalization rules
    pub normalize_stdout: Vec<(String, String)>,
    pub normalize_stderr: Vec<(String, String)>,
    pub failure_status: Option<i32>,
    // For UI tests, allows compiler to exit with arbitrary failure status
    pub dont_check_failure_status: bool,
    // Whether or not `rustfix` should apply the `CodeSuggestion`s of this test and compile the
    // resulting Rust code.
    pub run_rustfix: bool,
    // If true, `rustfix` will only apply `MachineApplicable` suggestions.
    pub rustfix_only_machine_applicable: bool,
    pub assembly_output: Option<String>,
    // If true, the test is expected to ICE
    pub should_ice: bool,
    // If true, the stderr is expected to be different across bit-widths.
    pub stderr_per_bitwidth: bool,
    // The MIR opt to unit test, if any
    pub mir_unit_test: Option<String>,
    // Whether to tell `rustc` to remap the "src base" directory to a fake
    // directory.
    pub remap_src_base: bool,
    /// Extra flags to pass to `llvm-cov` when producing coverage reports.
    /// Only used by the "coverage-run" test mode.
    pub llvm_cov_flags: Vec<String>,
    /// Extra flags to pass to LLVM's `filecheck` tool, in tests that use it.
    pub filecheck_flags: Vec<String>,
    /// Don't automatically insert any `--check-cfg` args
    pub no_auto_check_cfg: bool,
    /// Run tests which require enzyme being build
    pub has_enzyme: bool,
    /// Build and use `minicore` as `core` stub for `no_core` tests in cross-compilation scenarios
    /// that don't otherwise want/need `-Z build-std`.
    pub add_core_stubs: bool,
    /// Whether line annotatins are required for the given error kind.
    pub dont_require_annotations: HashSet<ErrorKind>,
    /// Whether pretty printers should be disabled in gdb.
    pub disable_gdb_pretty_printers: bool,
    /// Compare the output by lines, rather than as a single string.
    pub compare_output_by_lines: bool,
}

mod directives {
    pub const ERROR_PATTERN: &'static str = "error-pattern";
    pub const REGEX_ERROR_PATTERN: &'static str = "regex-error-pattern";
    pub const COMPILE_FLAGS: &'static str = "compile-flags";
    pub const RUN_FLAGS: &'static str = "run-flags";
    pub const DOC_FLAGS: &'static str = "doc-flags";
    pub const SHOULD_ICE: &'static str = "should-ice";
    pub const BUILD_AUX_DOCS: &'static str = "build-aux-docs";
    pub const UNIQUE_DOC_OUT_DIR: &'static str = "unique-doc-out-dir";
    pub const FORCE_HOST: &'static str = "force-host";
    pub const CHECK_STDOUT: &'static str = "check-stdout";
    pub const CHECK_RUN_RESULTS: &'static str = "check-run-results";
    pub const DONT_CHECK_COMPILER_STDOUT: &'static str = "dont-check-compiler-stdout";
    pub const DONT_CHECK_COMPILER_STDERR: &'static str = "dont-check-compiler-stderr";
    pub const DONT_REQUIRE_ANNOTATIONS: &'static str = "dont-require-annotations";
    pub const NO_PREFER_DYNAMIC: &'static str = "no-prefer-dynamic";
    pub const PRETTY_MODE: &'static str = "pretty-mode";
    pub const PRETTY_COMPARE_ONLY: &'static str = "pretty-compare-only";
    pub const AUX_BIN: &'static str = "aux-bin";
    pub const AUX_BUILD: &'static str = "aux-build";
    pub const AUX_CRATE: &'static str = "aux-crate";
    pub const PROC_MACRO: &'static str = "proc-macro";
    pub const AUX_CODEGEN_BACKEND: &'static str = "aux-codegen-backend";
    pub const EXEC_ENV: &'static str = "exec-env";
    pub const RUSTC_ENV: &'static str = "rustc-env";
    pub const UNSET_EXEC_ENV: &'static str = "unset-exec-env";
    pub const UNSET_RUSTC_ENV: &'static str = "unset-rustc-env";
    pub const FORBID_OUTPUT: &'static str = "forbid-output";
    pub const CHECK_TEST_LINE_NUMBERS_MATCH: &'static str = "check-test-line-numbers-match";
    pub const IGNORE_PASS: &'static str = "ignore-pass";
    pub const FAILURE_STATUS: &'static str = "failure-status";
    pub const DONT_CHECK_FAILURE_STATUS: &'static str = "dont-check-failure-status";
    pub const RUN_RUSTFIX: &'static str = "run-rustfix";
    pub const RUSTFIX_ONLY_MACHINE_APPLICABLE: &'static str = "rustfix-only-machine-applicable";
    pub const ASSEMBLY_OUTPUT: &'static str = "assembly-output";
    pub const STDERR_PER_BITWIDTH: &'static str = "stderr-per-bitwidth";
    pub const INCREMENTAL: &'static str = "incremental";
    pub const KNOWN_BUG: &'static str = "known-bug";
    pub const TEST_MIR_PASS: &'static str = "test-mir-pass";
    pub const REMAP_SRC_BASE: &'static str = "remap-src-base";
    pub const LLVM_COV_FLAGS: &'static str = "llvm-cov-flags";
    pub const FILECHECK_FLAGS: &'static str = "filecheck-flags";
    pub const NO_AUTO_CHECK_CFG: &'static str = "no-auto-check-cfg";
    pub const ADD_CORE_STUBS: &'static str = "add-core-stubs";
    // This isn't a real directive, just one that is probably mistyped often
    pub const INCORRECT_COMPILER_FLAGS: &'static str = "compiler-flags";
    pub const DISABLE_GDB_PRETTY_PRINTERS: &'static str = "disable-gdb-pretty-printers";
    pub const COMPARE_OUTPUT_BY_LINES: &'static str = "compare-output-by-lines";
}

impl TestProps {
    pub fn new() -> Self {
        TestProps {
            error_patterns: vec![],
            regex_error_patterns: vec![],
            compile_flags: vec![],
            run_flags: vec![],
            doc_flags: vec![],
            pp_exact: None,
            aux: Default::default(),
            revisions: vec![],
            rustc_env: vec![
                ("RUSTC_ICE".to_string(), "0".to_string()),
                ("RUST_BACKTRACE".to_string(), "short".to_string()),
            ],
            unset_rustc_env: vec![("RUSTC_LOG_COLOR".to_string())],
            exec_env: vec![],
            unset_exec_env: vec![],
            build_aux_docs: false,
            unique_doc_out_dir: false,
            force_host: false,
            check_stdout: false,
            check_run_results: false,
            dont_check_compiler_stdout: false,
            dont_check_compiler_stderr: false,
            no_prefer_dynamic: false,
            pretty_mode: "normal".to_string(),
            pretty_compare_only: false,
            forbid_output: vec![],
            incremental_dir: None,
            incremental: false,
            known_bug: false,
            pass_mode: None,
            fail_mode: None,
            ignore_pass: false,
            check_test_line_numbers_match: false,
            normalize_stdout: vec![],
            normalize_stderr: vec![],
            failure_status: None,
            dont_check_failure_status: false,
            run_rustfix: false,
            rustfix_only_machine_applicable: false,
            assembly_output: None,
            should_ice: false,
            stderr_per_bitwidth: false,
            mir_unit_test: None,
            remap_src_base: false,
            llvm_cov_flags: vec![],
            filecheck_flags: vec![],
            no_auto_check_cfg: false,
            has_enzyme: false,
            add_core_stubs: false,
            dont_require_annotations: Default::default(),
            disable_gdb_pretty_printers: false,
            compare_output_by_lines: false,
        }
    }

    pub fn from_aux_file(
        &self,
        testfile: &Utf8Path,
        revision: Option<&str>,
        config: &Config,
    ) -> Self {
        let mut props = TestProps::new();

        // copy over select properties to the aux build:
        props.incremental_dir = self.incremental_dir.clone();
        props.ignore_pass = true;
        props.load_from(testfile, revision, config);

        props
    }

    pub fn from_file(testfile: &Utf8Path, revision: Option<&str>, config: &Config) -> Self {
        let mut props = TestProps::new();
        props.load_from(testfile, revision, config);
        props.exec_env.push(("RUSTC".to_string(), config.rustc_path.to_string()));

        match (props.pass_mode, props.fail_mode) {
            (None, None) if config.mode == TestMode::Ui => props.fail_mode = Some(FailMode::Check),
            (Some(_), Some(_)) => panic!("cannot use a *-fail and *-pass mode together"),
            _ => {}
        }

        props
    }

    /// Loads properties from `testfile` into `props`. If a property is
    /// tied to a particular revision `foo` (indicated by writing
    /// `//@[foo]`), then the property is ignored unless `test_revision` is
    /// `Some("foo")`.
    fn load_from(&mut self, testfile: &Utf8Path, test_revision: Option<&str>, config: &Config) {
        let mut has_edition = false;
        if !testfile.is_dir() {
            let file = File::open(testfile.as_std_path()).unwrap();

            let mut poisoned = false;

            iter_directives(
                config.mode,
                &mut poisoned,
                testfile,
                file,
                &mut |directive @ DirectiveLine { line_number, raw_directive: ln, .. }| {
                    if !directive.applies_to_test_revision(test_revision) {
                        return;
                    }

                    use directives::*;

                    config.push_name_value_directive(
                        ln,
                        ERROR_PATTERN,
                        testfile,
                        line_number,
                        &mut self.error_patterns,
                        |r| r,
                    );
                    config.push_name_value_directive(
                        ln,
                        REGEX_ERROR_PATTERN,
                        testfile,
                        line_number,
                        &mut self.regex_error_patterns,
                        |r| r,
                    );

                    config.push_name_value_directive(
                        ln,
                        DOC_FLAGS,
                        testfile,
                        line_number,
                        &mut self.doc_flags,
                        |r| r,
                    );

                    fn split_flags(flags: &str) -> Vec<String> {
                        // Individual flags can be single-quoted to preserve spaces; see
                        // <https://github.com/rust-lang/rust/pull/115948/commits/957c5db6>.
                        flags
                            .split('\'')
                            .enumerate()
                            .flat_map(|(i, f)| {
                                if i % 2 == 1 { vec![f] } else { f.split_whitespace().collect() }
                            })
                            .map(move |s| s.to_owned())
                            .collect::<Vec<_>>()
                    }

                    if let Some(flags) =
                        config.parse_name_value_directive(ln, COMPILE_FLAGS, testfile, line_number)
                    {
                        let flags = split_flags(&flags);
                        for flag in &flags {
                            if flag == "--edition" || flag.starts_with("--edition=") {
                                panic!("you must use `//@ edition` to configure the edition");
                            }
                        }
                        self.compile_flags.extend(flags);
                    }
                    if config
                        .parse_name_value_directive(
                            ln,
                            INCORRECT_COMPILER_FLAGS,
                            testfile,
                            line_number,
                        )
                        .is_some()
                    {
                        panic!("`compiler-flags` directive should be spelled `compile-flags`");
                    }

                    if let Some(edition) = config.parse_edition(ln, testfile, line_number) {
                        // The edition is added at the start, since flags from //@compile-flags must
                        // be passed to rustc last.
                        self.compile_flags.insert(0, format!("--edition={}", edition.trim()));
                        has_edition = true;
                    }

                    config.parse_and_update_revisions(
                        testfile,
                        line_number,
                        ln,
                        &mut self.revisions,
                    );

                    if let Some(flags) =
                        config.parse_name_value_directive(ln, RUN_FLAGS, testfile, line_number)
                    {
                        self.run_flags.extend(split_flags(&flags));
                    }

                    if self.pp_exact.is_none() {
                        self.pp_exact = config.parse_pp_exact(ln, testfile, line_number);
                    }

                    config.set_name_directive(ln, SHOULD_ICE, &mut self.should_ice);
                    config.set_name_directive(ln, BUILD_AUX_DOCS, &mut self.build_aux_docs);
                    config.set_name_directive(ln, UNIQUE_DOC_OUT_DIR, &mut self.unique_doc_out_dir);

                    config.set_name_directive(ln, FORCE_HOST, &mut self.force_host);
                    config.set_name_directive(ln, CHECK_STDOUT, &mut self.check_stdout);
                    config.set_name_directive(ln, CHECK_RUN_RESULTS, &mut self.check_run_results);
                    config.set_name_directive(
                        ln,
                        DONT_CHECK_COMPILER_STDOUT,
                        &mut self.dont_check_compiler_stdout,
                    );
                    config.set_name_directive(
                        ln,
                        DONT_CHECK_COMPILER_STDERR,
                        &mut self.dont_check_compiler_stderr,
                    );
                    config.set_name_directive(ln, NO_PREFER_DYNAMIC, &mut self.no_prefer_dynamic);

                    if let Some(m) =
                        config.parse_name_value_directive(ln, PRETTY_MODE, testfile, line_number)
                    {
                        self.pretty_mode = m;
                    }

                    config.set_name_directive(
                        ln,
                        PRETTY_COMPARE_ONLY,
                        &mut self.pretty_compare_only,
                    );

                    // Call a helper method to deal with aux-related directives.
                    parse_and_update_aux(config, ln, testfile, line_number, &mut self.aux);

                    config.push_name_value_directive(
                        ln,
                        EXEC_ENV,
                        testfile,
                        line_number,
                        &mut self.exec_env,
                        Config::parse_env,
                    );
                    config.push_name_value_directive(
                        ln,
                        UNSET_EXEC_ENV,
                        testfile,
                        line_number,
                        &mut self.unset_exec_env,
                        |r| r.trim().to_owned(),
                    );
                    config.push_name_value_directive(
                        ln,
                        RUSTC_ENV,
                        testfile,
                        line_number,
                        &mut self.rustc_env,
                        Config::parse_env,
                    );
                    config.push_name_value_directive(
                        ln,
                        UNSET_RUSTC_ENV,
                        testfile,
                        line_number,
                        &mut self.unset_rustc_env,
                        |r| r.trim().to_owned(),
                    );
                    config.push_name_value_directive(
                        ln,
                        FORBID_OUTPUT,
                        testfile,
                        line_number,
                        &mut self.forbid_output,
                        |r| r,
                    );
                    config.set_name_directive(
                        ln,
                        CHECK_TEST_LINE_NUMBERS_MATCH,
                        &mut self.check_test_line_numbers_match,
                    );

                    self.update_pass_mode(ln, test_revision, config);
                    self.update_fail_mode(ln, config);

                    config.set_name_directive(ln, IGNORE_PASS, &mut self.ignore_pass);

                    if let Some(NormalizeRule { kind, regex, replacement }) =
                        config.parse_custom_normalization(ln)
                    {
                        let rule_tuple = (regex, replacement);
                        match kind {
                            NormalizeKind::Stdout => self.normalize_stdout.push(rule_tuple),
                            NormalizeKind::Stderr => self.normalize_stderr.push(rule_tuple),
                            NormalizeKind::Stderr32bit => {
                                if config.target_cfg().pointer_width == 32 {
                                    self.normalize_stderr.push(rule_tuple);
                                }
                            }
                            NormalizeKind::Stderr64bit => {
                                if config.target_cfg().pointer_width == 64 {
                                    self.normalize_stderr.push(rule_tuple);
                                }
                            }
                        }
                    }

                    if let Some(code) = config
                        .parse_name_value_directive(ln, FAILURE_STATUS, testfile, line_number)
                        .and_then(|code| code.trim().parse::<i32>().ok())
                    {
                        self.failure_status = Some(code);
                    }

                    config.set_name_directive(
                        ln,
                        DONT_CHECK_FAILURE_STATUS,
                        &mut self.dont_check_failure_status,
                    );

                    config.set_name_directive(ln, RUN_RUSTFIX, &mut self.run_rustfix);
                    config.set_name_directive(
                        ln,
                        RUSTFIX_ONLY_MACHINE_APPLICABLE,
                        &mut self.rustfix_only_machine_applicable,
                    );
                    config.set_name_value_directive(
                        ln,
                        ASSEMBLY_OUTPUT,
                        testfile,
                        line_number,
                        &mut self.assembly_output,
                        |r| r.trim().to_string(),
                    );
                    config.set_name_directive(
                        ln,
                        STDERR_PER_BITWIDTH,
                        &mut self.stderr_per_bitwidth,
                    );
                    config.set_name_directive(ln, INCREMENTAL, &mut self.incremental);

                    // Unlike the other `name_value_directive`s this needs to be handled manually,
                    // because it sets a `bool` flag.
                    if let Some(known_bug) =
                        config.parse_name_value_directive(ln, KNOWN_BUG, testfile, line_number)
                    {
                        let known_bug = known_bug.trim();
                        if known_bug == "unknown"
                            || known_bug.split(',').all(|issue_ref| {
                                issue_ref
                                    .trim()
                                    .split_once('#')
                                    .filter(|(_, number)| {
                                        number.chars().all(|digit| digit.is_numeric())
                                    })
                                    .is_some()
                            })
                        {
                            self.known_bug = true;
                        } else {
                            panic!(
                                "Invalid known-bug value: {known_bug}\nIt requires comma-separated issue references (`#000` or `chalk#000`) or `known-bug: unknown`."
                            );
                        }
                    } else if config.parse_name_directive(ln, KNOWN_BUG) {
                        panic!(
                            "Invalid known-bug attribute, requires comma-separated issue references (`#000` or `chalk#000`) or `known-bug: unknown`."
                        );
                    }

                    config.set_name_value_directive(
                        ln,
                        TEST_MIR_PASS,
                        testfile,
                        line_number,
                        &mut self.mir_unit_test,
                        |s| s.trim().to_string(),
                    );
                    config.set_name_directive(ln, REMAP_SRC_BASE, &mut self.remap_src_base);

                    if let Some(flags) =
                        config.parse_name_value_directive(ln, LLVM_COV_FLAGS, testfile, line_number)
                    {
                        self.llvm_cov_flags.extend(split_flags(&flags));
                    }

                    if let Some(flags) = config.parse_name_value_directive(
                        ln,
                        FILECHECK_FLAGS,
                        testfile,
                        line_number,
                    ) {
                        self.filecheck_flags.extend(split_flags(&flags));
                    }

                    config.set_name_directive(ln, NO_AUTO_CHECK_CFG, &mut self.no_auto_check_cfg);

                    self.update_add_core_stubs(ln, config);

                    if let Some(err_kind) = config.parse_name_value_directive(
                        ln,
                        DONT_REQUIRE_ANNOTATIONS,
                        testfile,
                        line_number,
                    ) {
                        self.dont_require_annotations
                            .insert(ErrorKind::expect_from_user_str(err_kind.trim()));
                    }

                    config.set_name_directive(
                        ln,
                        DISABLE_GDB_PRETTY_PRINTERS,
                        &mut self.disable_gdb_pretty_printers,
                    );
                    config.set_name_directive(
                        ln,
                        COMPARE_OUTPUT_BY_LINES,
                        &mut self.compare_output_by_lines,
                    );
                },
            );

            if poisoned {
                eprintln!("errors encountered during TestProps parsing: {}", testfile);
                panic!("errors encountered during TestProps parsing");
            }
        }

        if self.should_ice {
            self.failure_status = Some(101);
        }

        if config.mode == TestMode::Incremental {
            self.incremental = true;
        }

        if config.mode == TestMode::Crashes {
            // we don't want to pollute anything with backtrace-files
            // also turn off backtraces in order to save some execution
            // time on the tests; we only need to know IF it crashes
            self.rustc_env = vec![
                ("RUST_BACKTRACE".to_string(), "0".to_string()),
                ("RUSTC_ICE".to_string(), "0".to_string()),
            ];
        }

        for key in &["RUST_TEST_NOCAPTURE", "RUST_TEST_THREADS"] {
            if let Ok(val) = env::var(key) {
                if !self.exec_env.iter().any(|&(ref x, _)| x == key) {
                    self.exec_env.push(((*key).to_owned(), val))
                }
            }
        }

        if let (Some(edition), false) = (&config.edition, has_edition) {
            // The edition is added at the start, since flags from //@compile-flags must be passed
            // to rustc last.
            self.compile_flags.insert(0, format!("--edition={}", edition));
        }
    }

    fn update_fail_mode(&mut self, ln: &str, config: &Config) {
        let check_ui = |mode: &str| {
            // Mode::Crashes may need build-fail in order to trigger llvm errors or stack overflows
            if config.mode != TestMode::Ui && config.mode != TestMode::Crashes {
                panic!("`{}-fail` directive is only supported in UI tests", mode);
            }
        };
        if config.mode == TestMode::Ui && config.parse_name_directive(ln, "compile-fail") {
            panic!("`compile-fail` directive is useless in UI tests");
        }
        let fail_mode = if config.parse_name_directive(ln, "check-fail") {
            check_ui("check");
            Some(FailMode::Check)
        } else if config.parse_name_directive(ln, "build-fail") {
            check_ui("build");
            Some(FailMode::Build)
        } else if config.parse_name_directive(ln, "run-fail") {
            check_ui("run");
            Some(FailMode::Run(RunFailMode::Fail))
        } else if config.parse_name_directive(ln, "run-crash") {
            check_ui("run");
            Some(FailMode::Run(RunFailMode::Crash))
        } else if config.parse_name_directive(ln, "run-fail-or-crash") {
            check_ui("run");
            Some(FailMode::Run(RunFailMode::FailOrCrash))
        } else {
            None
        };
        match (self.fail_mode, fail_mode) {
            (None, Some(_)) => self.fail_mode = fail_mode,
            (Some(_), Some(_)) => panic!("multiple `*-fail` directives in a single test"),
            (_, None) => {}
        }
    }

    fn update_pass_mode(&mut self, ln: &str, revision: Option<&str>, config: &Config) {
        let check_no_run = |s| match (config.mode, s) {
            (TestMode::Ui, _) => (),
            (TestMode::Crashes, _) => (),
            (TestMode::Codegen, "build-pass") => (),
            (TestMode::Incremental, _) => {
                if revision.is_some() && !self.revisions.iter().all(|r| r.starts_with("cfail")) {
                    panic!("`{s}` directive is only supported in `cfail` incremental tests")
                }
            }
            (mode, _) => panic!("`{s}` directive is not supported in `{mode}` tests"),
        };
        let pass_mode = if config.parse_name_directive(ln, "check-pass") {
            check_no_run("check-pass");
            Some(PassMode::Check)
        } else if config.parse_name_directive(ln, "build-pass") {
            check_no_run("build-pass");
            Some(PassMode::Build)
        } else if config.parse_name_directive(ln, "run-pass") {
            check_no_run("run-pass");
            Some(PassMode::Run)
        } else {
            None
        };
        match (self.pass_mode, pass_mode) {
            (None, Some(_)) => self.pass_mode = pass_mode,
            (Some(_), Some(_)) => panic!("multiple `*-pass` directives in a single test"),
            (_, None) => {}
        }
    }

    pub fn pass_mode(&self, config: &Config) -> Option<PassMode> {
        if !self.ignore_pass && self.fail_mode.is_none() {
            if let mode @ Some(_) = config.force_pass_mode {
                return mode;
            }
        }
        self.pass_mode
    }

    // does not consider CLI override for pass mode
    pub fn local_pass_mode(&self) -> Option<PassMode> {
        self.pass_mode
    }

    pub fn update_add_core_stubs(&mut self, ln: &str, config: &Config) {
        let add_core_stubs = config.parse_name_directive(ln, directives::ADD_CORE_STUBS);
        if add_core_stubs {
            if !matches!(config.mode, TestMode::Ui | TestMode::Codegen | TestMode::Assembly) {
                panic!(
                    "`add-core-stubs` is currently only supported for ui, codegen and assembly test modes"
                );
            }

            // FIXME(jieyouxu): this check is currently order-dependent, but we should probably
            // collect all directives in one go then perform a validation pass after that.
            if self.local_pass_mode().is_some_and(|pm| pm == PassMode::Run) {
                // `minicore` can only be used with non-run modes, because it's `core` prelude stubs
                // and can't run.
                panic!("`add-core-stubs` cannot be used to run the test binary");
            }

            self.add_core_stubs = add_core_stubs;
        }
    }
}

/// If the given line begins with the appropriate comment prefix for a directive,
/// returns a struct containing various parts of the directive.
fn line_directive<'line>(
    line_number: usize,
    original_line: &'line str,
) -> Option<DirectiveLine<'line>> {
    // Ignore lines that don't start with the comment prefix.
    let after_comment =
        original_line.trim_start().strip_prefix(COMPILETEST_DIRECTIVE_PREFIX)?.trim_start();

    let revision;
    let raw_directive;

    if let Some(after_open_bracket) = after_comment.strip_prefix('[') {
        // A comment like `//@[foo]` only applies to revision `foo`.
        let Some((line_revision, after_close_bracket)) = after_open_bracket.split_once(']') else {
            panic!(
                "malformed condition directive: expected `{COMPILETEST_DIRECTIVE_PREFIX}[foo]`, found `{original_line}`"
            )
        };

        revision = Some(line_revision);
        raw_directive = after_close_bracket.trim_start();
    } else {
        revision = None;
        raw_directive = after_comment;
    };

    Some(DirectiveLine { line_number, revision, raw_directive })
}

/// The (partly) broken-down contents of a line containing a test directive,
/// which [`iter_directives`] passes to its callback function.
///
/// For example:
///
/// ```text
/// //@ compile-flags: -O
///     ^^^^^^^^^^^^^^^^^ raw_directive
///
/// //@ [foo] compile-flags: -O
///      ^^^                    revision
///           ^^^^^^^^^^^^^^^^^ raw_directive
/// ```
struct DirectiveLine<'ln> {
    line_number: usize,
    /// Some test directives start with a revision name in square brackets
    /// (e.g. `[foo]`), and only apply to that revision of the test.
    /// If present, this field contains the revision name (e.g. `foo`).
    revision: Option<&'ln str>,
    /// The main part of the directive, after removing the comment prefix
    /// and the optional revision specifier.
    ///
    /// This is "raw" because the directive's name and colon-separated value
    /// (if present) have not yet been extracted or checked.
    raw_directive: &'ln str,
}

impl<'ln> DirectiveLine<'ln> {
    fn applies_to_test_revision(&self, test_revision: Option<&str>) -> bool {
        self.revision.is_none() || self.revision == test_revision
    }
}

pub(crate) struct CheckDirectiveResult<'ln> {
    is_known_directive: bool,
    trailing_directive: Option<&'ln str>,
}

pub(crate) fn check_directive<'a>(
    directive_ln: &'a str,
    mode: TestMode,
) -> CheckDirectiveResult<'a> {
    let (directive_name, post) = directive_ln.split_once([':', ' ']).unwrap_or((directive_ln, ""));

    let is_known_directive = KNOWN_DIRECTIVE_NAMES.contains(&directive_name)
        || match mode {
            TestMode::Rustdoc => KNOWN_HTMLDOCCK_DIRECTIVE_NAMES.contains(&directive_name),
            TestMode::RustdocJson => KNOWN_JSONDOCCK_DIRECTIVE_NAMES.contains(&directive_name),
            _ => false,
        };

    let trailing = post.trim().split_once(' ').map(|(pre, _)| pre).unwrap_or(post);
    let trailing_directive = {
        // 1. is the directive name followed by a space? (to exclude `:`)
        directive_ln.get(directive_name.len()..).is_some_and(|s| s.starts_with(' '))
            // 2. is what is after that directive also a directive (ex: "only-x86 only-arm")
            && KNOWN_DIRECTIVE_NAMES.contains(&trailing)
    }
    .then_some(trailing);

    CheckDirectiveResult { is_known_directive, trailing_directive }
}

const COMPILETEST_DIRECTIVE_PREFIX: &str = "//@";

fn iter_directives(
    mode: TestMode,
    poisoned: &mut bool,
    testfile: &Utf8Path,
    rdr: impl Read,
    it: &mut dyn FnMut(DirectiveLine<'_>),
) {
    if testfile.is_dir() {
        return;
    }

    // Coverage tests in coverage-run mode always have these extra directives, without needing to
    // specify them manually in every test file.
    //
    // FIXME(jieyouxu): I feel like there's a better way to do this, leaving for later.
    if mode == TestMode::CoverageRun {
        let extra_directives: &[&str] = &[
            "needs-profiler-runtime",
            // FIXME(pietroalbini): this test currently does not work on cross-compiled targets
            // because remote-test is not capable of sending back the *.profraw files generated by
            // the LLVM instrumentation.
            "ignore-cross-compile",
        ];
        // Process the extra implied directives, with a dummy line number of 0.
        for raw_directive in extra_directives {
            it(DirectiveLine { line_number: 0, revision: None, raw_directive });
        }
    }

    let mut rdr = BufReader::with_capacity(1024, rdr);
    let mut ln = String::new();
    let mut line_number = 0;

    loop {
        line_number += 1;
        ln.clear();
        if rdr.read_line(&mut ln).unwrap() == 0 {
            break;
        }
        let ln = ln.trim();

        let Some(directive_line) = line_directive(line_number, ln) else {
            continue;
        };

        // Perform unknown directive check on Rust files.
        if testfile.extension() == Some("rs") {
            let CheckDirectiveResult { is_known_directive, trailing_directive } =
                check_directive(directive_line.raw_directive, mode);

            if !is_known_directive {
                *poisoned = true;

                error!(
                    "{testfile}:{line_number}: detected unknown compiletest test directive `{}`",
                    directive_line.raw_directive,
                );

                return;
            }

            if let Some(trailing_directive) = &trailing_directive {
                *poisoned = true;

                error!(
                    "{testfile}:{line_number}: detected trailing compiletest test directive `{}`",
                    trailing_directive,
                );
                help!("put the trailing directive in its own line: `//@ {}`", trailing_directive);

                return;
            }
        }

        it(directive_line);
    }
}

impl Config {
    fn parse_and_update_revisions(
        &self,
        testfile: &Utf8Path,
        line_number: usize,
        line: &str,
        existing: &mut Vec<String>,
    ) {
        const FORBIDDEN_REVISION_NAMES: [&str; 2] = [
            // `//@ revisions: true false` Implying `--cfg=true` and `--cfg=false` makes it very
            // weird for the test, since if the test writer wants a cfg of the same revision name
            // they'd have to use `cfg(r#true)` and `cfg(r#false)`.
            "true", "false",
        ];

        const FILECHECK_FORBIDDEN_REVISION_NAMES: [&str; 9] =
            ["CHECK", "COM", "NEXT", "SAME", "EMPTY", "NOT", "COUNT", "DAG", "LABEL"];

        if let Some(raw) = self.parse_name_value_directive(line, "revisions", testfile, line_number)
        {
            if self.mode == TestMode::RunMake {
                panic!("`run-make` mode tests do not support revisions: {}", testfile);
            }

            let mut duplicates: HashSet<_> = existing.iter().cloned().collect();
            for revision in raw.split_whitespace() {
                if !duplicates.insert(revision.to_string()) {
                    panic!("duplicate revision: `{}` in line `{}`: {}", revision, raw, testfile);
                }

                if FORBIDDEN_REVISION_NAMES.contains(&revision) {
                    panic!(
                        "revision name `{revision}` is not permitted: `{}` in line `{}`: {}",
                        revision, raw, testfile
                    );
                }

                if matches!(self.mode, TestMode::Assembly | TestMode::Codegen | TestMode::MirOpt)
                    && FILECHECK_FORBIDDEN_REVISION_NAMES.contains(&revision)
                {
                    panic!(
                        "revision name `{revision}` is not permitted in a test suite that uses \
                        `FileCheck` annotations as it is confusing when used as custom `FileCheck` \
                        prefix: `{revision}` in line `{}`: {}",
                        raw, testfile
                    );
                }

                existing.push(revision.to_string());
            }
        }
    }

    fn parse_env(nv: String) -> (String, String) {
        // nv is either FOO or FOO=BAR
        // FIXME(Zalathar): The form without `=` seems to be unused; should
        // we drop support for it?
        let (name, value) = nv.split_once('=').unwrap_or((&nv, ""));
        // Trim whitespace from the name, so that `//@ exec-env: FOO=BAR`
        // sees the name as `FOO` and not ` FOO`.
        let name = name.trim();
        (name.to_owned(), value.to_owned())
    }

    fn parse_pp_exact(
        &self,
        line: &str,
        testfile: &Utf8Path,
        line_number: usize,
    ) -> Option<Utf8PathBuf> {
        if let Some(s) = self.parse_name_value_directive(line, "pp-exact", testfile, line_number) {
            Some(Utf8PathBuf::from(&s))
        } else if self.parse_name_directive(line, "pp-exact") {
            testfile.file_name().map(Utf8PathBuf::from)
        } else {
            None
        }
    }

    fn parse_custom_normalization(&self, raw_directive: &str) -> Option<NormalizeRule> {
        // FIXME(Zalathar): Integrate name/value splitting into `DirectiveLine`
        // instead of doing it here.
        let (directive_name, raw_value) = raw_directive.split_once(':')?;

        let kind = match directive_name {
            "normalize-stdout" => NormalizeKind::Stdout,
            "normalize-stderr" => NormalizeKind::Stderr,
            "normalize-stderr-32bit" => NormalizeKind::Stderr32bit,
            "normalize-stderr-64bit" => NormalizeKind::Stderr64bit,
            _ => return None,
        };

        let Some((regex, replacement)) = parse_normalize_rule(raw_value) else {
            error!("couldn't parse custom normalization rule: `{raw_directive}`");
            help!("expected syntax is: `{directive_name}: \"REGEX\" -> \"REPLACEMENT\"`");
            panic!("invalid normalization rule detected");
        };
        Some(NormalizeRule { kind, regex, replacement })
    }

    fn parse_name_directive(&self, line: &str, directive: &str) -> bool {
        // Ensure the directive is a whole word. Do not match "ignore-x86" when
        // the line says "ignore-x86_64".
        line.starts_with(directive)
            && matches!(line.as_bytes().get(directive.len()), None | Some(&b' ') | Some(&b':'))
    }

    fn parse_negative_name_directive(&self, line: &str, directive: &str) -> bool {
        line.starts_with("no-") && self.parse_name_directive(&line[3..], directive)
    }

    pub fn parse_name_value_directive(
        &self,
        line: &str,
        directive: &str,
        testfile: &Utf8Path,
        line_number: usize,
    ) -> Option<String> {
        let colon = directive.len();
        if line.starts_with(directive) && line.as_bytes().get(colon) == Some(&b':') {
            let value = line[(colon + 1)..].to_owned();
            debug!("{}: {}", directive, value);
            let value = expand_variables(value, self);
            if value.is_empty() {
                error!("{testfile}:{line_number}: empty value for directive `{directive}`");
                help!("expected syntax is: `{directive}: value`");
                panic!("empty directive value detected");
            }
            Some(value)
        } else {
            None
        }
    }

    fn parse_edition(&self, line: &str, testfile: &Utf8Path, line_number: usize) -> Option<String> {
        self.parse_name_value_directive(line, "edition", testfile, line_number)
    }

    fn set_name_directive(&self, line: &str, directive: &str, value: &mut bool) {
        match value {
            true => {
                if self.parse_negative_name_directive(line, directive) {
                    *value = false;
                }
            }
            false => {
                if self.parse_name_directive(line, directive) {
                    *value = true;
                }
            }
        }
    }

    fn set_name_value_directive<T>(
        &self,
        line: &str,
        directive: &str,
        testfile: &Utf8Path,
        line_number: usize,
        value: &mut Option<T>,
        parse: impl FnOnce(String) -> T,
    ) {
        if value.is_none() {
            *value =
                self.parse_name_value_directive(line, directive, testfile, line_number).map(parse);
        }
    }

    fn push_name_value_directive<T>(
        &self,
        line: &str,
        directive: &str,
        testfile: &Utf8Path,
        line_number: usize,
        values: &mut Vec<T>,
        parse: impl FnOnce(String) -> T,
    ) {
        if let Some(value) =
            self.parse_name_value_directive(line, directive, testfile, line_number).map(parse)
        {
            values.push(value);
        }
    }
}

// FIXME(jieyouxu): fix some of these variable names to more accurately reflect what they do.
fn expand_variables(mut value: String, config: &Config) -> String {
    const CWD: &str = "{{cwd}}";
    const SRC_BASE: &str = "{{src-base}}";
    const TEST_SUITE_BUILD_BASE: &str = "{{build-base}}";
    const RUST_SRC_BASE: &str = "{{rust-src-base}}";
    const SYSROOT_BASE: &str = "{{sysroot-base}}";
    const TARGET_LINKER: &str = "{{target-linker}}";
    const TARGET: &str = "{{target}}";

    if value.contains(CWD) {
        let cwd = env::current_dir().unwrap();
        value = value.replace(CWD, &cwd.to_str().unwrap());
    }

    if value.contains(SRC_BASE) {
        value = value.replace(SRC_BASE, &config.src_test_suite_root.as_str());
    }

    if value.contains(TEST_SUITE_BUILD_BASE) {
        value = value.replace(TEST_SUITE_BUILD_BASE, &config.build_test_suite_root.as_str());
    }

    if value.contains(SYSROOT_BASE) {
        value = value.replace(SYSROOT_BASE, &config.sysroot_base.as_str());
    }

    if value.contains(TARGET_LINKER) {
        value = value.replace(TARGET_LINKER, config.target_linker.as_deref().unwrap_or(""));
    }

    if value.contains(TARGET) {
        value = value.replace(TARGET, &config.target);
    }

    if value.contains(RUST_SRC_BASE) {
        let src_base = config.sysroot_base.join("lib/rustlib/src/rust");
        src_base.try_exists().expect(&*format!("{} should exists", src_base));
        let src_base = src_base.read_link_utf8().unwrap_or(src_base);
        value = value.replace(RUST_SRC_BASE, &src_base.as_str());
    }

    value
}

struct NormalizeRule {
    kind: NormalizeKind,
    regex: String,
    replacement: String,
}

enum NormalizeKind {
    Stdout,
    Stderr,
    Stderr32bit,
    Stderr64bit,
}

/// Parses the regex and replacement values of a `//@ normalize-*` directive, in the format:
/// ```text
/// "REGEX" -> "REPLACEMENT"
/// ```
fn parse_normalize_rule(raw_value: &str) -> Option<(String, String)> {
    // FIXME: Support escaped double-quotes in strings.
    let captures = static_regex!(
        r#"(?x) # (verbose mode regex)
        ^
        \s*                     # (leading whitespace)
        "(?<regex>[^"]*)"       # "REGEX"
        \s+->\s+                # ->
        "(?<replacement>[^"]*)" # "REPLACEMENT"
        $
        "#
    )
    .captures(raw_value)?;
    let regex = captures["regex"].to_owned();
    let replacement = captures["replacement"].to_owned();
    // A `\n` sequence in the replacement becomes an actual newline.
    // FIXME: Do unescaping in a less ad-hoc way, and perhaps support escaped
    // backslashes and double-quotes.
    let replacement = replacement.replace("\\n", "\n");
    Some((regex, replacement))
}

/// Given an llvm version string that looks like `1.2.3-rc1`, extract as semver. Note that this
/// accepts more than just strict `semver` syntax (as in `major.minor.patch`); this permits omitting
/// minor and patch version components so users can write e.g. `//@ min-llvm-version: 19` instead of
/// having to write `//@ min-llvm-version: 19.0.0`.
///
/// Currently panics if the input string is malformed, though we really should not use panic as an
/// error handling strategy.
///
/// FIXME(jieyouxu): improve error handling
pub fn extract_llvm_version(version: &str) -> Version {
    // The version substring we're interested in usually looks like the `1.2.3`, without any of the
    // fancy suffix like `-rc1` or `meow`.
    let version = version.trim();
    let uninterested = |c: char| !c.is_ascii_digit() && c != '.';
    let version_without_suffix = match version.split_once(uninterested) {
        Some((prefix, _suffix)) => prefix,
        None => version,
    };

    let components: Vec<u64> = version_without_suffix
        .split('.')
        .map(|s| s.parse().expect("llvm version component should consist of only digits"))
        .collect();

    match &components[..] {
        [major] => Version::new(*major, 0, 0),
        [major, minor] => Version::new(*major, *minor, 0),
        [major, minor, patch] => Version::new(*major, *minor, *patch),
        _ => panic!("malformed llvm version string, expected only 1-3 components: {version}"),
    }
}

pub fn extract_llvm_version_from_binary(binary_path: &str) -> Option<Version> {
    let output = Command::new(binary_path).arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let version = String::from_utf8(output.stdout).ok()?;
    for line in version.lines() {
        if let Some(version) = line.split("LLVM version ").nth(1) {
            return Some(extract_llvm_version(version));
        }
    }
    None
}

/// For tests using the `needs-llvm-zstd` directive:
/// - for local LLVM builds, try to find the static zstd library in the llvm-config system libs.
/// - for `download-ci-llvm`, see if `lld` was built with zstd support.
pub fn llvm_has_libzstd(config: &Config) -> bool {
    // Strategy 1: works for local builds but not with `download-ci-llvm`.
    //
    // We check whether `llvm-config` returns the zstd library. Bootstrap's `llvm.libzstd` will only
    // ask to statically link it when building LLVM, so we only check if the list of system libs
    // contains a path to that static lib, and that it exists.
    //
    // See compiler/rustc_llvm/build.rs for more details and similar expectations.
    fn is_zstd_in_config(llvm_bin_dir: &Utf8Path) -> Option<()> {
        let llvm_config_path = llvm_bin_dir.join("llvm-config");
        let output = Command::new(llvm_config_path).arg("--system-libs").output().ok()?;
        assert!(output.status.success(), "running llvm-config --system-libs failed");

        let libs = String::from_utf8(output.stdout).ok()?;
        for lib in libs.split_whitespace() {
            if lib.ends_with("libzstd.a") && Utf8Path::new(lib).exists() {
                return Some(());
            }
        }

        None
    }

    // Strategy 2: `download-ci-llvm`'s `llvm-config --system-libs` will not return any libs to
    // use.
    //
    // The CI artifacts also don't contain the bootstrap config used to build them: otherwise we
    // could have looked at the `llvm.libzstd` config.
    //
    // We infer whether `LLVM_ENABLE_ZSTD` was used to build LLVM as a byproduct of testing whether
    // `lld` supports it. If not, an error will be emitted: "LLVM was not built with
    // LLVM_ENABLE_ZSTD or did not find zstd at build time".
    #[cfg(unix)]
    fn is_lld_built_with_zstd(llvm_bin_dir: &Utf8Path) -> Option<()> {
        let lld_path = llvm_bin_dir.join("lld");
        if lld_path.exists() {
            // We can't call `lld` as-is, it expects to be invoked by a compiler driver using a
            // different name. Prepare a temporary symlink to do that.
            let lld_symlink_path = llvm_bin_dir.join("ld.lld");
            if !lld_symlink_path.exists() {
                std::os::unix::fs::symlink(lld_path, &lld_symlink_path).ok()?;
            }

            // Run `lld` with a zstd flag. We expect this command to always error here, we don't
            // want to link actual files and don't pass any.
            let output = Command::new(&lld_symlink_path)
                .arg("--compress-debug-sections=zstd")
                .output()
                .ok()?;
            assert!(!output.status.success());

            // Look for a specific error caused by LLVM not being built with zstd support. We could
            // also look for the "no input files" message, indicating the zstd flag was accepted.
            let stderr = String::from_utf8(output.stderr).ok()?;
            let zstd_available = !stderr.contains("LLVM was not built with LLVM_ENABLE_ZSTD");

            // We don't particularly need to clean the link up (so the previous commands could fail
            // in theory but won't in practice), but we can try.
            std::fs::remove_file(lld_symlink_path).ok()?;

            if zstd_available {
                return Some(());
            }
        }

        None
    }

    #[cfg(not(unix))]
    fn is_lld_built_with_zstd(_llvm_bin_dir: &Utf8Path) -> Option<()> {
        None
    }

    if let Some(llvm_bin_dir) = &config.llvm_bin_dir {
        // Strategy 1: for local LLVM builds.
        if is_zstd_in_config(llvm_bin_dir).is_some() {
            return true;
        }

        // Strategy 2: for LLVM artifacts built on CI via `download-ci-llvm`.
        //
        // It doesn't work for cases where the artifacts don't contain the linker, but it's
        // best-effort: CI has `llvm.libzstd` and `lld` enabled on the x64 linux artifacts, so it
        // will at least work there.
        //
        // If this can be improved and expanded to less common cases in the future, it should.
        if config.target == "x86_64-unknown-linux-gnu"
            && config.host == config.target
            && is_lld_built_with_zstd(llvm_bin_dir).is_some()
        {
            return true;
        }
    }

    // Otherwise, all hope is lost.
    false
}

/// Takes a directive of the form `"<version1> [- <version2>]"`, returns the numeric representation
/// of `<version1>` and `<version2>` as tuple: `(<version1>, <version2>)`.
///
/// If the `<version2>` part is omitted, the second component of the tuple is the same as
/// `<version1>`.
fn extract_version_range<'a, F, VersionTy: Clone>(
    line: &'a str,
    parse: F,
) -> Option<(VersionTy, VersionTy)>
where
    F: Fn(&'a str) -> Option<VersionTy>,
{
    let mut splits = line.splitn(2, "- ").map(str::trim);
    let min = splits.next().unwrap();
    if min.ends_with('-') {
        return None;
    }

    let max = splits.next();

    if min.is_empty() {
        return None;
    }

    let min = parse(min)?;
    let max = match max {
        Some("") => return None,
        Some(max) => parse(max)?,
        _ => min.clone(),
    };

    Some((min, max))
}

pub(crate) fn make_test_description<R: Read>(
    config: &Config,
    cache: &DirectivesCache,
    name: String,
    path: &Utf8Path,
    filterable_path: &Utf8Path,
    src: R,
    test_revision: Option<&str>,
    poisoned: &mut bool,
) -> CollectedTestDesc {
    let mut ignore = false;
    let mut ignore_message = None;
    let mut should_fail = false;

    let mut local_poisoned = false;

    // Scan through the test file to handle `ignore-*`, `only-*`, and `needs-*` directives.
    iter_directives(
        config.mode,
        &mut local_poisoned,
        path,
        src,
        &mut |directive @ DirectiveLine { line_number, raw_directive: ln, .. }| {
            if !directive.applies_to_test_revision(test_revision) {
                return;
            }

            macro_rules! decision {
                ($e:expr) => {
                    match $e {
                        IgnoreDecision::Ignore { reason } => {
                            ignore = true;
                            ignore_message = Some(reason.into());
                        }
                        IgnoreDecision::Error { message } => {
                            error!("{path}:{line_number}: {message}");
                            *poisoned = true;
                            return;
                        }
                        IgnoreDecision::Continue => {}
                    }
                };
            }

            decision!(cfg::handle_ignore(config, ln));
            decision!(cfg::handle_only(config, ln));
            decision!(needs::handle_needs(&cache.needs, config, ln));
            decision!(ignore_llvm(config, path, ln, line_number));
            decision!(ignore_backends(config, path, ln, line_number));
            decision!(needs_backends(config, path, ln, line_number));
            decision!(ignore_cdb(config, ln));
            decision!(ignore_gdb(config, ln));
            decision!(ignore_lldb(config, ln));

            if config.target == "wasm32-unknown-unknown"
                && config.parse_name_directive(ln, directives::CHECK_RUN_RESULTS)
            {
                decision!(IgnoreDecision::Ignore {
                    reason: "ignored on WASM as the run results cannot be checked there".into(),
                });
            }

            should_fail |= config.parse_name_directive(ln, "should-fail");
        },
    );

    if local_poisoned {
        eprintln!("errors encountered when trying to make test description: {}", path);
        panic!("errors encountered when trying to make test description");
    }

    // The `should-fail` annotation doesn't apply to pretty tests,
    // since we run the pretty printer across all tests by default.
    // If desired, we could add a `should-fail-pretty` annotation.
    let should_panic = match config.mode {
        TestMode::Pretty => ShouldPanic::No,
        _ if should_fail => ShouldPanic::Yes,
        _ => ShouldPanic::No,
    };

    CollectedTestDesc {
        name,
        filterable_path: filterable_path.to_owned(),
        ignore,
        ignore_message,
        should_panic,
    }
}

fn ignore_cdb(config: &Config, line: &str) -> IgnoreDecision {
    if config.debugger != Some(Debugger::Cdb) {
        return IgnoreDecision::Continue;
    }

    if let Some(actual_version) = config.cdb_version {
        if let Some(rest) = line.strip_prefix("min-cdb-version:").map(str::trim) {
            let min_version = extract_cdb_version(rest).unwrap_or_else(|| {
                panic!("couldn't parse version range: {:?}", rest);
            });

            // Ignore if actual version is smaller than the minimum
            // required version
            if actual_version < min_version {
                return IgnoreDecision::Ignore {
                    reason: format!("ignored when the CDB version is lower than {rest}"),
                };
            }
        }
    }
    IgnoreDecision::Continue
}

fn ignore_gdb(config: &Config, line: &str) -> IgnoreDecision {
    if config.debugger != Some(Debugger::Gdb) {
        return IgnoreDecision::Continue;
    }

    if let Some(actual_version) = config.gdb_version {
        if let Some(rest) = line.strip_prefix("min-gdb-version:").map(str::trim) {
            let (start_ver, end_ver) = extract_version_range(rest, extract_gdb_version)
                .unwrap_or_else(|| {
                    panic!("couldn't parse version range: {:?}", rest);
                });

            if start_ver != end_ver {
                panic!("Expected single GDB version")
            }
            // Ignore if actual version is smaller than the minimum
            // required version
            if actual_version < start_ver {
                return IgnoreDecision::Ignore {
                    reason: format!("ignored when the GDB version is lower than {rest}"),
                };
            }
        } else if let Some(rest) = line.strip_prefix("ignore-gdb-version:").map(str::trim) {
            let (min_version, max_version) = extract_version_range(rest, extract_gdb_version)
                .unwrap_or_else(|| {
                    panic!("couldn't parse version range: {:?}", rest);
                });

            if max_version < min_version {
                panic!("Malformed GDB version range: max < min")
            }

            if actual_version >= min_version && actual_version <= max_version {
                if min_version == max_version {
                    return IgnoreDecision::Ignore {
                        reason: format!("ignored when the GDB version is {rest}"),
                    };
                } else {
                    return IgnoreDecision::Ignore {
                        reason: format!("ignored when the GDB version is between {rest}"),
                    };
                }
            }
        }
    }
    IgnoreDecision::Continue
}

fn ignore_lldb(config: &Config, line: &str) -> IgnoreDecision {
    if config.debugger != Some(Debugger::Lldb) {
        return IgnoreDecision::Continue;
    }

    if let Some(actual_version) = config.lldb_version {
        if let Some(rest) = line.strip_prefix("min-lldb-version:").map(str::trim) {
            let min_version = rest.parse().unwrap_or_else(|e| {
                panic!("Unexpected format of LLDB version string: {}\n{:?}", rest, e);
            });
            // Ignore if actual version is smaller the minimum required
            // version
            if actual_version < min_version {
                return IgnoreDecision::Ignore {
                    reason: format!("ignored when the LLDB version is {rest}"),
                };
            }
        }
    }
    IgnoreDecision::Continue
}

fn ignore_backends(
    config: &Config,
    path: &Utf8Path,
    line: &str,
    line_number: usize,
) -> IgnoreDecision {
    if let Some(backends_to_ignore) =
        config.parse_name_value_directive(line, "ignore-backends", path, line_number)
    {
        for backend in backends_to_ignore.split_whitespace().map(|backend| {
            match CodegenBackend::try_from(backend) {
                Ok(backend) => backend,
                Err(error) => {
                    panic!("Invalid ignore-backends value `{backend}` in `{path}`: {error}")
                }
            }
        }) {
            if config.default_codegen_backend == backend {
                return IgnoreDecision::Ignore {
                    reason: format!("{} backend is marked as ignore", backend.as_str()),
                };
            }
        }
    }
    IgnoreDecision::Continue
}

fn needs_backends(
    config: &Config,
    path: &Utf8Path,
    line: &str,
    line_number: usize,
) -> IgnoreDecision {
    if let Some(needed_backends) =
        config.parse_name_value_directive(line, "needs-backends", path, line_number)
    {
        if !needed_backends
            .split_whitespace()
            .map(|backend| match CodegenBackend::try_from(backend) {
                Ok(backend) => backend,
                Err(error) => {
                    panic!("Invalid needs-backends value `{backend}` in `{path}`: {error}")
                }
            })
            .any(|backend| config.default_codegen_backend == backend)
        {
            return IgnoreDecision::Ignore {
                reason: format!(
                    "{} backend is not part of required backends",
                    config.default_codegen_backend.as_str()
                ),
            };
        }
    }
    IgnoreDecision::Continue
}

fn ignore_llvm(config: &Config, path: &Utf8Path, line: &str, line_number: usize) -> IgnoreDecision {
    if let Some(needed_components) =
        config.parse_name_value_directive(line, "needs-llvm-components", path, line_number)
    {
        let components: HashSet<_> = config.llvm_components.split_whitespace().collect();
        if let Some(missing_component) = needed_components
            .split_whitespace()
            .find(|needed_component| !components.contains(needed_component))
        {
            if env::var_os("COMPILETEST_REQUIRE_ALL_LLVM_COMPONENTS").is_some() {
                panic!(
                    "missing LLVM component {}, and COMPILETEST_REQUIRE_ALL_LLVM_COMPONENTS is set: {}",
                    missing_component, path
                );
            }
            return IgnoreDecision::Ignore {
                reason: format!("ignored when the {missing_component} LLVM component is missing"),
            };
        }
    }
    if let Some(actual_version) = &config.llvm_version {
        // Note that these `min` versions will check for not just major versions.

        if let Some(version_string) =
            config.parse_name_value_directive(line, "min-llvm-version", path, line_number)
        {
            let min_version = extract_llvm_version(&version_string);
            // Ignore if actual version is smaller than the minimum required version.
            if *actual_version < min_version {
                return IgnoreDecision::Ignore {
                    reason: format!(
                        "ignored when the LLVM version {actual_version} is older than {min_version}"
                    ),
                };
            }
        } else if let Some(version_string) =
            config.parse_name_value_directive(line, "max-llvm-major-version", path, line_number)
        {
            let max_version = extract_llvm_version(&version_string);
            // Ignore if actual major version is larger than the maximum required major version.
            if actual_version.major > max_version.major {
                return IgnoreDecision::Ignore {
                    reason: format!(
                        "ignored when the LLVM version ({actual_version}) is newer than major\
                        version {}",
                        max_version.major
                    ),
                };
            }
        } else if let Some(version_string) =
            config.parse_name_value_directive(line, "min-system-llvm-version", path, line_number)
        {
            let min_version = extract_llvm_version(&version_string);
            // Ignore if using system LLVM and actual version
            // is smaller the minimum required version
            if config.system_llvm && *actual_version < min_version {
                return IgnoreDecision::Ignore {
                    reason: format!(
                        "ignored when the system LLVM version {actual_version} is older than {min_version}"
                    ),
                };
            }
        } else if let Some(version_range) =
            config.parse_name_value_directive(line, "ignore-llvm-version", path, line_number)
        {
            // Syntax is: "ignore-llvm-version: <version1> [- <version2>]"
            let (v_min, v_max) =
                extract_version_range(&version_range, |s| Some(extract_llvm_version(s)))
                    .unwrap_or_else(|| {
                        panic!("couldn't parse version range: \"{version_range}\"");
                    });
            if v_max < v_min {
                panic!("malformed LLVM version range where {v_max} < {v_min}")
            }
            // Ignore if version lies inside of range.
            if *actual_version >= v_min && *actual_version <= v_max {
                if v_min == v_max {
                    return IgnoreDecision::Ignore {
                        reason: format!("ignored when the LLVM version is {actual_version}"),
                    };
                } else {
                    return IgnoreDecision::Ignore {
                        reason: format!(
                            "ignored when the LLVM version is between {v_min} and {v_max}"
                        ),
                    };
                }
            }
        } else if let Some(version_string) =
            config.parse_name_value_directive(line, "exact-llvm-major-version", path, line_number)
        {
            // Syntax is "exact-llvm-major-version: <version>"
            let version = extract_llvm_version(&version_string);
            if actual_version.major != version.major {
                return IgnoreDecision::Ignore {
                    reason: format!(
                        "ignored when the actual LLVM major version is {}, but the test only targets major version {}",
                        actual_version.major, version.major
                    ),
                };
            }
        }
    }
    IgnoreDecision::Continue
}

enum IgnoreDecision {
    Ignore { reason: String },
    Continue,
    Error { message: String },
}
