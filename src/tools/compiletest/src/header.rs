use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use log::*;

use crate::common::{self, CompareMode, Config, Mode, PassMode};
use crate::util;

use crate::extract_gdb_version;

/// Whether to ignore the test.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Ignore {
    /// Runs it.
    Run,
    /// Ignore it totally.
    Ignore,
    /// Ignore only the gdb test, but run the lldb test.
    IgnoreGdb,
    /// Ignore only the lldb test, but run the gdb test.
    IgnoreLldb,
}

impl Ignore {
    pub fn can_run_gdb(&self) -> bool {
        *self == Ignore::Run || *self == Ignore::IgnoreLldb
    }

    pub fn can_run_lldb(&self) -> bool {
        *self == Ignore::Run || *self == Ignore::IgnoreGdb
    }

    pub fn no_gdb(&self) -> Ignore {
        match *self {
            Ignore::Run => Ignore::IgnoreGdb,
            Ignore::IgnoreGdb => Ignore::IgnoreGdb,
            _ => Ignore::Ignore,
        }
    }

    pub fn no_lldb(&self) -> Ignore {
        match *self {
            Ignore::Run => Ignore::IgnoreLldb,
            Ignore::IgnoreLldb => Ignore::IgnoreLldb,
            _ => Ignore::Ignore,
        }
    }
}

/// The result of parse_cfg_name_directive.
#[derive(Clone, Copy, PartialEq, Debug)]
enum ParsedNameDirective {
    /// No match.
    NoMatch,
    /// Match.
    Match,
    /// Mode was DebugInfoGdbLldb and this matched gdb.
    MatchGdb,
    /// Mode was DebugInfoGdbLldb and this matched lldb.
    MatchLldb,
}

/// Properties which must be known very early, before actually running
/// the test.
pub struct EarlyProps {
    pub ignore: Ignore,
    pub should_fail: bool,
    pub aux: Vec<String>,
    pub revisions: Vec<String>,
}

impl EarlyProps {
    pub fn from_file(config: &Config, testfile: &Path) -> Self {
        let mut props = EarlyProps {
            ignore: Ignore::Run,
            should_fail: false,
            aux: Vec::new(),
            revisions: vec![],
        };

        if config.mode == common::DebugInfoGdbLldb {
            if config.lldb_python_dir.is_none() {
                props.ignore = props.ignore.no_lldb();
            }
            if config.gdb_version.is_none() {
                props.ignore = props.ignore.no_gdb();
            }
        } else if config.mode == common::DebugInfoCdb {
            if config.cdb.is_none() {
                props.ignore = Ignore::Ignore;
            }
        }

        let rustc_has_profiler_support = env::var_os("RUSTC_PROFILER_SUPPORT").is_some();
        let rustc_has_sanitizer_support = env::var_os("RUSTC_SANITIZER_SUPPORT").is_some();

        iter_header(testfile, None, &mut |ln| {
            // we should check if any only-<platform> exists and if it exists
            // and does not matches the current platform, skip the test
            if props.ignore != Ignore::Ignore {
                props.ignore = match config.parse_cfg_name_directive(ln, "ignore") {
                    ParsedNameDirective::Match => Ignore::Ignore,
                    ParsedNameDirective::NoMatch => props.ignore,
                    ParsedNameDirective::MatchGdb => props.ignore.no_gdb(),
                    ParsedNameDirective::MatchLldb => props.ignore.no_lldb(),
                };

                if config.has_cfg_prefix(ln, "only") {
                    props.ignore = match config.parse_cfg_name_directive(ln, "only") {
                        ParsedNameDirective::Match => props.ignore,
                        ParsedNameDirective::NoMatch => Ignore::Ignore,
                        ParsedNameDirective::MatchLldb => props.ignore.no_gdb(),
                        ParsedNameDirective::MatchGdb => props.ignore.no_lldb(),
                    };
                }

                if ignore_llvm(config, ln) {
                    props.ignore = Ignore::Ignore;
                }

                if config.run_clang_based_tests_with.is_none() &&
                   config.parse_needs_matching_clang(ln) {
                    props.ignore = Ignore::Ignore;
                }

                if !rustc_has_profiler_support &&
                   config.parse_needs_profiler_support(ln) {
                    props.ignore = Ignore::Ignore;
                }

                if !rustc_has_sanitizer_support &&
                   config.parse_needs_sanitizer_support(ln) {
                    props.ignore = Ignore::Ignore;
                }
            }

            if (config.mode == common::DebugInfoGdb || config.mode == common::DebugInfoGdbLldb) &&
                props.ignore.can_run_gdb() && ignore_gdb(config, ln) {
                props.ignore = props.ignore.no_gdb();
            }

            if (config.mode == common::DebugInfoLldb || config.mode == common::DebugInfoGdbLldb) &&
                props.ignore.can_run_lldb() && ignore_lldb(config, ln) {
                props.ignore = props.ignore.no_lldb();
            }

            if let Some(s) = config.parse_aux_build(ln) {
                props.aux.push(s);
            }

            if let Some(r) = config.parse_revisions(ln) {
                props.revisions.extend(r);
            }

            props.should_fail = props.should_fail || config.parse_name_directive(ln, "should-fail");
        });

        return props;

        fn ignore_gdb(config: &Config, line: &str) -> bool {
            if let Some(actual_version) = config.gdb_version {
                if line.starts_with("min-gdb-version") {
                    let (start_ver, end_ver) = extract_gdb_version_range(line);

                    if start_ver != end_ver {
                        panic!("Expected single GDB version")
                    }
                    // Ignore if actual version is smaller the minimum required
                    // version
                    actual_version < start_ver
                } else if line.starts_with("ignore-gdb-version") {
                    let (min_version, max_version) = extract_gdb_version_range(line);

                    if max_version < min_version {
                        panic!("Malformed GDB version range: max < min")
                    }

                    actual_version >= min_version && actual_version <= max_version
                } else {
                    false
                }
            } else {
                false
            }
        }

        // Takes a directive of the form "ignore-gdb-version <version1> [- <version2>]",
        // returns the numeric representation of <version1> and <version2> as
        // tuple: (<version1> as u32, <version2> as u32)
        // If the <version2> part is omitted, the second component of the tuple
        // is the same as <version1>.
        fn extract_gdb_version_range(line: &str) -> (u32, u32) {
            const ERROR_MESSAGE: &'static str = "Malformed GDB version directive";

            let range_components = line.split(&[' ', '-'][..])
                                       .filter(|word| !word.is_empty())
                                       .map(extract_gdb_version)
                                       .skip_while(Option::is_none)
                                       .take(3) // 3 or more = invalid, so take at most 3.
                                       .collect::<Vec<Option<u32>>>();

            match range_components.len() {
                1 => {
                    let v = range_components[0].unwrap();
                    (v, v)
                }
                2 => {
                    let v_min = range_components[0].unwrap();
                    let v_max = range_components[1].expect(ERROR_MESSAGE);
                    (v_min, v_max)
                }
                _ => panic!(ERROR_MESSAGE),
            }
        }

        fn ignore_lldb(config: &Config, line: &str) -> bool {
            if let Some(ref actual_version) = config.lldb_version {
                if line.starts_with("min-lldb-version") {
                    let min_version = line.trim_end()
                        .rsplit(' ')
                        .next()
                        .expect("Malformed lldb version directive");
                    // Ignore if actual version is smaller the minimum required
                    // version
                    lldb_version_to_int(actual_version) < lldb_version_to_int(min_version)
                } else if line.starts_with("rust-lldb") && !config.lldb_native_rust {
                    true
                } else {
                    false
                }
            } else {
                false
            }
        }

        fn ignore_llvm(config: &Config, line: &str) -> bool {
            if config.system_llvm && line.starts_with("no-system-llvm") {
                return true;
            }
            if let Some(ref actual_version) = config.llvm_version {
                if line.starts_with("min-llvm-version") {
                    let min_version = line.trim_end()
                        .rsplit(' ')
                        .next()
                        .expect("Malformed llvm version directive");
                    // Ignore if actual version is smaller the minimum required
                    // version
                    &actual_version[..] < min_version
                } else if line.starts_with("min-system-llvm-version") {
                    let min_version = line.trim_end()
                        .rsplit(' ')
                        .next()
                        .expect("Malformed llvm version directive");
                    // Ignore if using system LLVM and actual version
                    // is smaller the minimum required version
                    config.system_llvm && &actual_version[..] < min_version
                } else if line.starts_with("ignore-llvm-version") {
                    // Syntax is: "ignore-llvm-version <version1> [- <version2>]"
                    let range_components = line.split(' ')
                        .skip(1) // Skip the directive.
                        .map(|s| s.trim())
                        .filter(|word| !word.is_empty() && word != &"-")
                        .take(3) // 3 or more = invalid, so take at most 3.
                        .collect::<Vec<&str>>();
                    match range_components.len() {
                        1 => {
                            &actual_version[..] == range_components[0]
                        }
                        2 => {
                            let v_min = range_components[0];
                            let v_max = range_components[1];
                            if v_max < v_min {
                                panic!("Malformed LLVM version range: max < min")
                            }
                            // Ignore if version lies inside of range.
                            &actual_version[..] >= v_min && &actual_version[..] <= v_max
                        }
                        _ => panic!("Malformed LLVM version directive"),
                    }
                } else {
                    false
                }
            } else {
                false
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct TestProps {
    // Lines that should be expected, in order, on standard out
    pub error_patterns: Vec<String>,
    // Extra flags to pass to the compiler
    pub compile_flags: Vec<String>,
    // Extra flags to pass when the compiled code is run (such as --bench)
    pub run_flags: Option<String>,
    // If present, the name of a file that this test should match when
    // pretty-printed
    pub pp_exact: Option<PathBuf>,
    // Other crates that should be compiled (typically from the same
    // directory as the test, but for backwards compatibility reasons
    // we also check the auxiliary directory)
    pub aux_builds: Vec<String>,
    // A list of crates to pass '--extern-private name:PATH' flags for
    // This should be a subset of 'aux_build'
    // FIXME: Replace this with a better solution: https://github.com/rust-lang/rust/pull/54020
    pub extern_private: Vec<String>,
    // Environment settings to use for compiling
    pub rustc_env: Vec<(String, String)>,
    // Environment variables to unset prior to compiling.
    // Variables are unset before applying 'rustc_env'.
    pub unset_rustc_env: Vec<String>,
    // Environment settings to use during execution
    pub exec_env: Vec<(String, String)>,
    // Lines to check if they appear in the expected debugger output
    pub check_lines: Vec<String>,
    // Build documentation for all specified aux-builds as well
    pub build_aux_docs: bool,
    // Flag to force a crate to be built with the host architecture
    pub force_host: bool,
    // Check stdout for error-pattern output as well as stderr
    pub check_stdout: bool,
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
    // Run --pretty expanded when running pretty printing tests
    pub pretty_expanded: bool,
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
    pub incremental_dir: Option<PathBuf>,
    // How far should the test proceed while still passing.
    pass_mode: Option<PassMode>,
    // Ignore `--pass` overrides from the command line for this test.
    ignore_pass: bool,
    // rustdoc will test the output of the `--test` option
    pub check_test_line_numbers_match: bool,
    // Do not pass `-Z ui-testing` to UI tests
    pub disable_ui_testing_normalization: bool,
    // customized normalization rules
    pub normalize_stdout: Vec<(String, String)>,
    pub normalize_stderr: Vec<(String, String)>,
    pub failure_status: i32,
    // Whether or not `rustfix` should apply the `CodeSuggestion`s of this test and compile the
    // resulting Rust code.
    pub run_rustfix: bool,
    // If true, `rustfix` will only apply `MachineApplicable` suggestions.
    pub rustfix_only_machine_applicable: bool,
    pub assembly_output: Option<String>,
}

impl TestProps {
    pub fn new() -> Self {
        TestProps {
            error_patterns: vec![],
            compile_flags: vec![],
            run_flags: None,
            pp_exact: None,
            aux_builds: vec![],
            extern_private: vec![],
            revisions: vec![],
            rustc_env: vec![],
            unset_rustc_env: vec![],
            exec_env: vec![],
            check_lines: vec![],
            build_aux_docs: false,
            force_host: false,
            check_stdout: false,
            dont_check_compiler_stdout: false,
            dont_check_compiler_stderr: false,
            no_prefer_dynamic: false,
            pretty_expanded: false,
            pretty_mode: "normal".to_string(),
            pretty_compare_only: false,
            forbid_output: vec![],
            incremental_dir: None,
            pass_mode: None,
            ignore_pass: false,
            check_test_line_numbers_match: false,
            disable_ui_testing_normalization: false,
            normalize_stdout: vec![],
            normalize_stderr: vec![],
            failure_status: -1,
            run_rustfix: false,
            rustfix_only_machine_applicable: false,
            assembly_output: None,
        }
    }

    pub fn from_aux_file(&self, testfile: &Path, cfg: Option<&str>, config: &Config) -> Self {
        let mut props = TestProps::new();

        // copy over select properties to the aux build:
        props.incremental_dir = self.incremental_dir.clone();
        props.load_from(testfile, cfg, config);

        props
    }

    pub fn from_file(testfile: &Path, cfg: Option<&str>, config: &Config) -> Self {
        let mut props = TestProps::new();
        props.load_from(testfile, cfg, config);
        props
    }

    /// Loads properties from `testfile` into `props`. If a property is
    /// tied to a particular revision `foo` (indicated by writing
    /// `//[foo]`), then the property is ignored unless `cfg` is
    /// `Some("foo")`.
    fn load_from(&mut self, testfile: &Path, cfg: Option<&str>, config: &Config) {
        iter_header(testfile, cfg, &mut |ln| {
            if let Some(ep) = config.parse_error_pattern(ln) {
                self.error_patterns.push(ep);
            }

            if let Some(flags) = config.parse_compile_flags(ln) {
                self.compile_flags
                    .extend(flags.split_whitespace().map(|s| s.to_owned()));
            }

            if let Some(edition) = config.parse_edition(ln) {
                self.compile_flags.push(format!("--edition={}", edition));
            }

            if let Some(r) = config.parse_revisions(ln) {
                self.revisions.extend(r);
            }

            if self.run_flags.is_none() {
                self.run_flags = config.parse_run_flags(ln);
            }

            if self.pp_exact.is_none() {
                self.pp_exact = config.parse_pp_exact(ln, testfile);
            }

            if !self.build_aux_docs {
                self.build_aux_docs = config.parse_build_aux_docs(ln);
            }

            if !self.force_host {
                self.force_host = config.parse_force_host(ln);
            }

            if !self.check_stdout {
                self.check_stdout = config.parse_check_stdout(ln);
            }

            if !self.dont_check_compiler_stdout {
                self.dont_check_compiler_stdout = config.parse_dont_check_compiler_stdout(ln);
            }

            if !self.dont_check_compiler_stderr {
                self.dont_check_compiler_stderr = config.parse_dont_check_compiler_stderr(ln);
            }

            if !self.no_prefer_dynamic {
                self.no_prefer_dynamic = config.parse_no_prefer_dynamic(ln);
            }

            if !self.pretty_expanded {
                self.pretty_expanded = config.parse_pretty_expanded(ln);
            }

            if let Some(m) = config.parse_pretty_mode(ln) {
                self.pretty_mode = m;
            }

            if !self.pretty_compare_only {
                self.pretty_compare_only = config.parse_pretty_compare_only(ln);
            }

            if let Some(ab) = config.parse_aux_build(ln) {
                self.aux_builds.push(ab);
            }

            if let Some(ep) = config.parse_extern_private(ln) {
                self.extern_private.push(ep);
            }

            if let Some(ee) = config.parse_env(ln, "exec-env") {
                self.exec_env.push(ee);
            }

            if let Some(ee) = config.parse_env(ln, "rustc-env") {
                self.rustc_env.push(ee);
            }

            if let Some(ev) = config.parse_name_value_directive(ln, "unset-rustc-env") {
                self.unset_rustc_env.push(ev);
            }

            if let Some(cl) = config.parse_check_line(ln) {
                self.check_lines.push(cl);
            }

            if let Some(of) = config.parse_forbid_output(ln) {
                self.forbid_output.push(of);
            }

            if !self.check_test_line_numbers_match {
                self.check_test_line_numbers_match = config.parse_check_test_line_numbers_match(ln);
            }

            self.update_pass_mode(ln, cfg, config);

            if !self.ignore_pass {
                self.ignore_pass = config.parse_ignore_pass(ln);
            }

            if !self.disable_ui_testing_normalization {
                self.disable_ui_testing_normalization =
                    config.parse_disable_ui_testing_normalization(ln);
            }

            if let Some(rule) = config.parse_custom_normalization(ln, "normalize-stdout") {
                self.normalize_stdout.push(rule);
            }
            if let Some(rule) = config.parse_custom_normalization(ln, "normalize-stderr") {
                self.normalize_stderr.push(rule);
            }

            if let Some(code) = config.parse_failure_status(ln) {
                self.failure_status = code;
            }

            if !self.run_rustfix {
                self.run_rustfix = config.parse_run_rustfix(ln);
            }

            if !self.rustfix_only_machine_applicable {
                self.rustfix_only_machine_applicable =
                    config.parse_rustfix_only_machine_applicable(ln);
            }

            if self.assembly_output.is_none() {
                self.assembly_output = config.parse_assembly_output(ln);
            }
        });

        if self.failure_status == -1 {
            self.failure_status = match config.mode {
                Mode::RunFail => 101,
                _ => 1,
            };
        }

        for key in &["RUST_TEST_NOCAPTURE", "RUST_TEST_THREADS"] {
            if let Ok(val) = env::var(key) {
                if self.exec_env.iter().find(|&&(ref x, _)| x == key).is_none() {
                    self.exec_env.push(((*key).to_owned(), val))
                }
            }
        }
    }

    fn update_pass_mode(&mut self, ln: &str, revision: Option<&str>, config: &Config) {
        let check_no_run = |s| {
            if config.mode != Mode::Ui && config.mode != Mode::Incremental {
                panic!("`{}` header is only supported in UI and incremental tests", s);
            }
            if config.mode == Mode::Incremental &&
                !revision.map_or(false, |r| r.starts_with("cfail")) &&
                !self.revisions.iter().all(|r| r.starts_with("cfail")) {
                panic!("`{}` header is only supported in `cfail` incremental tests", s);
            }
        };
        let pass_mode = if config.parse_name_directive(ln, "check-pass") {
            check_no_run("check-pass");
            Some(PassMode::Check)
        } else if config.parse_name_directive(ln, "build-pass") {
            check_no_run("build-pass");
            Some(PassMode::Build)
        } else if config.parse_name_directive(ln, "compile-pass") /* compatibility */ {
            check_no_run("compile-pass");
            Some(PassMode::Build)
        } else if config.parse_name_directive(ln, "run-pass") {
            if config.mode != Mode::Ui && config.mode != Mode::RunPass /* compatibility */ {
                panic!("`run-pass` header is only supported in UI tests")
            }
            Some(PassMode::Run)
        } else {
            None
        };
        match (self.pass_mode, pass_mode) {
            (None, Some(_)) => self.pass_mode = pass_mode,
            (Some(_), Some(_)) => panic!("multiple `*-pass` headers in a single test"),
            (_, None) => {}
        }
    }

    pub fn pass_mode(&self, config: &Config) -> Option<PassMode> {
        if !self.ignore_pass {
            if let (mode @ Some(_), Some(_)) = (config.force_pass_mode, self.pass_mode) {
                return mode;
            }
        }
        self.pass_mode
    }
}

fn iter_header(testfile: &Path, cfg: Option<&str>, it: &mut dyn FnMut(&str)) {
    if testfile.is_dir() {
        return;
    }

    let comment = if testfile.to_string_lossy().ends_with(".rs") {
        "//"
    } else {
        "#"
    };

    // FIXME: would be nice to allow some whitespace between comment and brace :)
    // It took me like 2 days to debug why compile-flags weren’t taken into account for my test :)
    let comment_with_brace = comment.to_string() + "[";

    let rdr = BufReader::new(File::open(testfile).unwrap());
    for ln in rdr.lines() {
        // Assume that any directives will be found before the first
        // module or function. This doesn't seem to be an optimization
        // with a warm page cache. Maybe with a cold one.
        let ln = ln.unwrap();
        let ln = ln.trim();
        if ln.starts_with("fn") || ln.starts_with("mod") {
            return;
        } else if ln.starts_with(&comment_with_brace) {
            // A comment like `//[foo]` is specific to revision `foo`
            if let Some(close_brace) = ln.find(']') {
                let open_brace = ln.find('[').unwrap();
                let lncfg = &ln[open_brace + 1 .. close_brace];
                let matches = match cfg {
                    Some(s) => s == &lncfg[..],
                    None => false,
                };
                if matches {
                    it(ln[(close_brace + 1)..].trim_start());
                }
            } else {
                panic!("malformed condition directive: expected `{}foo]`, found `{}`",
                        comment_with_brace, ln)
            }
        } else if ln.starts_with(comment) {
            it(ln[comment.len() ..].trim_start());
        }
    }
    return;
}

impl Config {
    fn parse_error_pattern(&self, line: &str) -> Option<String> {
        self.parse_name_value_directive(line, "error-pattern")
    }

    fn parse_forbid_output(&self, line: &str) -> Option<String> {
        self.parse_name_value_directive(line, "forbid-output")
    }

    fn parse_aux_build(&self, line: &str) -> Option<String> {
        self.parse_name_value_directive(line, "aux-build")
            .map(|r| r.trim().to_string())
    }

    fn parse_extern_private(&self, line: &str) -> Option<String> {
        self.parse_name_value_directive(line, "extern-private")
    }

    fn parse_compile_flags(&self, line: &str) -> Option<String> {
        self.parse_name_value_directive(line, "compile-flags")
    }

    fn parse_revisions(&self, line: &str) -> Option<Vec<String>> {
        self.parse_name_value_directive(line, "revisions")
            .map(|r| r.split_whitespace().map(|t| t.to_string()).collect())
    }

    fn parse_run_flags(&self, line: &str) -> Option<String> {
        self.parse_name_value_directive(line, "run-flags")
    }

    fn parse_check_line(&self, line: &str) -> Option<String> {
        self.parse_name_value_directive(line, "check")
    }

    fn parse_force_host(&self, line: &str) -> bool {
        self.parse_name_directive(line, "force-host")
    }

    fn parse_build_aux_docs(&self, line: &str) -> bool {
        self.parse_name_directive(line, "build-aux-docs")
    }

    fn parse_check_stdout(&self, line: &str) -> bool {
        self.parse_name_directive(line, "check-stdout")
    }

    fn parse_dont_check_compiler_stdout(&self, line: &str) -> bool {
        self.parse_name_directive(line, "dont-check-compiler-stdout")
    }

    fn parse_dont_check_compiler_stderr(&self, line: &str) -> bool {
        self.parse_name_directive(line, "dont-check-compiler-stderr")
    }

    fn parse_no_prefer_dynamic(&self, line: &str) -> bool {
        self.parse_name_directive(line, "no-prefer-dynamic")
    }

    fn parse_pretty_expanded(&self, line: &str) -> bool {
        self.parse_name_directive(line, "pretty-expanded")
    }

    fn parse_pretty_mode(&self, line: &str) -> Option<String> {
        self.parse_name_value_directive(line, "pretty-mode")
    }

    fn parse_pretty_compare_only(&self, line: &str) -> bool {
        self.parse_name_directive(line, "pretty-compare-only")
    }

    fn parse_failure_status(&self, line: &str) -> Option<i32> {
        match self.parse_name_value_directive(line, "failure-status") {
            Some(code) => code.trim().parse::<i32>().ok(),
            _ => None,
        }
    }

    fn parse_disable_ui_testing_normalization(&self, line: &str) -> bool {
        self.parse_name_directive(line, "disable-ui-testing-normalization")
    }

    fn parse_check_test_line_numbers_match(&self, line: &str) -> bool {
        self.parse_name_directive(line, "check-test-line-numbers-match")
    }

    fn parse_ignore_pass(&self, line: &str) -> bool {
        self.parse_name_directive(line, "ignore-pass")
    }

    fn parse_assembly_output(&self, line: &str) -> Option<String> {
        self.parse_name_value_directive(line, "assembly-output")
            .map(|r| r.trim().to_string())
    }

    fn parse_env(&self, line: &str, name: &str) -> Option<(String, String)> {
        self.parse_name_value_directive(line, name).map(|nv| {
            // nv is either FOO or FOO=BAR
            let mut strs: Vec<String> = nv.splitn(2, '=').map(str::to_owned).collect();

            match strs.len() {
                1 => (strs.pop().unwrap(), String::new()),
                2 => {
                    let end = strs.pop().unwrap();
                    (strs.pop().unwrap(), end)
                }
                n => panic!("Expected 1 or 2 strings, not {}", n),
            }
        })
    }

    fn parse_pp_exact(&self, line: &str, testfile: &Path) -> Option<PathBuf> {
        if let Some(s) = self.parse_name_value_directive(line, "pp-exact") {
            Some(PathBuf::from(&s))
        } else if self.parse_name_directive(line, "pp-exact") {
            testfile.file_name().map(PathBuf::from)
        } else {
            None
        }
    }

    fn parse_custom_normalization(&self, mut line: &str, prefix: &str) -> Option<(String, String)> {
        if self.parse_cfg_name_directive(line, prefix) == ParsedNameDirective::Match {
            let from = parse_normalization_string(&mut line)?;
            let to = parse_normalization_string(&mut line)?;
            Some((from, to))
        } else {
            None
        }
    }

    fn parse_needs_matching_clang(&self, line: &str) -> bool {
        self.parse_name_directive(line, "needs-matching-clang")
    }

    fn parse_needs_profiler_support(&self, line: &str) -> bool {
        self.parse_name_directive(line, "needs-profiler-support")
    }

    fn parse_needs_sanitizer_support(&self, line: &str) -> bool {
        self.parse_name_directive(line, "needs-sanitizer-support")
    }

    /// Parses a name-value directive which contains config-specific information, e.g., `ignore-x86`
    /// or `normalize-stderr-32bit`.
    fn parse_cfg_name_directive(&self, line: &str, prefix: &str) -> ParsedNameDirective {
        if line.starts_with(prefix) && line.as_bytes().get(prefix.len()) == Some(&b'-') {
            let name = line[prefix.len() + 1..]
                .split(&[':', ' '][..])
                .next()
                .unwrap();

            if name == "test" ||
                util::matches_os(&self.target, name) ||             // target
                name == util::get_arch(&self.target) ||             // architecture
                name == util::get_pointer_width(&self.target) ||    // pointer width
                name == self.stage_id.split('-').next().unwrap() || // stage
                Some(name) == util::get_env(&self.target) ||        // env
                (self.target != self.host && name == "cross-compile") ||
                match self.compare_mode {
                    Some(CompareMode::Nll) => name == "compare-mode-nll",
                    Some(CompareMode::Polonius) => name == "compare-mode-polonius",
                    None => false,
                } ||
                (cfg!(debug_assertions) && name == "debug") {
                ParsedNameDirective::Match
            } else {
                match self.mode {
                    common::DebugInfoGdbLldb => {
                        if name == "gdb" {
                            ParsedNameDirective::MatchGdb
                        } else if name == "lldb" {
                            ParsedNameDirective::MatchLldb
                        } else {
                            ParsedNameDirective::NoMatch
                        }
                    },
                    common::DebugInfoCdb => if name == "cdb" {
                        ParsedNameDirective::Match
                    } else {
                        ParsedNameDirective::NoMatch
                    },
                    common::DebugInfoGdb => if name == "gdb" {
                        ParsedNameDirective::Match
                    } else {
                        ParsedNameDirective::NoMatch
                    },
                    common::DebugInfoLldb => if name == "lldb" {
                        ParsedNameDirective::Match
                    } else {
                        ParsedNameDirective::NoMatch
                    },
                    common::Pretty => if name == "pretty" {
                        ParsedNameDirective::Match
                    } else {
                        ParsedNameDirective::NoMatch
                    },
                    _ => ParsedNameDirective::NoMatch,
                }
            }
        } else {
            ParsedNameDirective::NoMatch
        }
    }

    fn has_cfg_prefix(&self, line: &str, prefix: &str) -> bool {
        // returns whether this line contains this prefix or not. For prefix
        // "ignore", returns true if line says "ignore-x86_64", "ignore-arch",
        // "ignore-android" etc.
        line.starts_with(prefix) && line.as_bytes().get(prefix.len()) == Some(&b'-')
    }

    fn parse_name_directive(&self, line: &str, directive: &str) -> bool {
        // Ensure the directive is a whole word. Do not match "ignore-x86" when
        // the line says "ignore-x86_64".
        line.starts_with(directive) && match line.as_bytes().get(directive.len()) {
            None | Some(&b' ') | Some(&b':') => true,
            _ => false,
        }
    }

    pub fn parse_name_value_directive(&self, line: &str, directive: &str) -> Option<String> {
        let colon = directive.len();
        if line.starts_with(directive) && line.as_bytes().get(colon) == Some(&b':') {
            let value = line[(colon + 1)..].to_owned();
            debug!("{}: {}", directive, value);
            Some(expand_variables(value, self))
        } else {
            None
        }
    }

    pub fn find_rust_src_root(&self) -> Option<PathBuf> {
        let mut path = self.src_base.clone();
        let path_postfix = Path::new("src/etc/lldb_batchmode.py");

        while path.pop() {
            if path.join(&path_postfix).is_file() {
                return Some(path);
            }
        }

        None
    }

    fn parse_run_rustfix(&self, line: &str) -> bool {
        self.parse_name_directive(line, "run-rustfix")
    }

    fn parse_rustfix_only_machine_applicable(&self, line: &str) -> bool {
        self.parse_name_directive(line, "rustfix-only-machine-applicable")
    }

    fn parse_edition(&self, line: &str) -> Option<String> {
        self.parse_name_value_directive(line, "edition")
    }
}

pub fn lldb_version_to_int(version_string: &str) -> isize {
    let error_string = format!(
        "Encountered LLDB version string with unexpected format: {}",
        version_string
    );
    version_string.parse().expect(&error_string)
}

fn expand_variables(mut value: String, config: &Config) -> String {
    const CWD: &'static str = "{{cwd}}";
    const SRC_BASE: &'static str = "{{src-base}}";
    const BUILD_BASE: &'static str = "{{build-base}}";

    if value.contains(CWD) {
        let cwd = env::current_dir().unwrap();
        value = value.replace(CWD, &cwd.to_string_lossy());
    }

    if value.contains(SRC_BASE) {
        value = value.replace(SRC_BASE, &config.src_base.to_string_lossy());
    }

    if value.contains(BUILD_BASE) {
        value = value.replace(BUILD_BASE, &config.build_base.to_string_lossy());
    }

    value
}

/// Finds the next quoted string `"..."` in `line`, and extract the content from it. Move the `line`
/// variable after the end of the quoted string.
///
/// # Examples
///
/// ```
/// let mut s = "normalize-stderr-32bit: \"something (32 bits)\" -> \"something ($WORD bits)\".";
/// let first = parse_normalization_string(&mut s);
/// assert_eq!(first, Some("something (32 bits)".to_owned()));
/// assert_eq!(s, " -> \"something ($WORD bits)\".");
/// ```
fn parse_normalization_string(line: &mut &str) -> Option<String> {
    // FIXME support escapes in strings.
    let begin = line.find('"')? + 1;
    let end = line[begin..].find('"')? + begin;
    let result = line[begin..end].to_owned();
    *line = &line[end + 1..];
    Some(result)
}

#[test]
fn test_parse_normalization_string() {
    let mut s = "normalize-stderr-32bit: \"something (32 bits)\" -> \"something ($WORD bits)\".";
    let first = parse_normalization_string(&mut s);
    assert_eq!(first, Some("something (32 bits)".to_owned()));
    assert_eq!(s, " -> \"something ($WORD bits)\".");

    // Nothing to normalize (No quotes)
    let mut s = "normalize-stderr-32bit: something (32 bits) -> something ($WORD bits).";
    let first = parse_normalization_string(&mut s);
    assert_eq!(first, None);
    assert_eq!(s, r#"normalize-stderr-32bit: something (32 bits) -> something ($WORD bits)."#);

    // Nothing to normalize (Only a single quote)
    let mut s = "normalize-stderr-32bit: \"something (32 bits) -> something ($WORD bits).";
    let first = parse_normalization_string(&mut s);
    assert_eq!(first, None);
    assert_eq!(s, "normalize-stderr-32bit: \"something (32 bits) -> something ($WORD bits).");

    // Nothing to normalize (Three quotes)
    let mut s = "normalize-stderr-32bit: \"something (32 bits)\" -> \"something ($WORD bits).";
    let first = parse_normalization_string(&mut s);
    assert_eq!(first, Some("something (32 bits)".to_owned()));
    assert_eq!(s, " -> \"something ($WORD bits).");
}
