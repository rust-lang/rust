use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use tracing::*;

use crate::common::{CompareMode, Config, Debugger, FailMode, Mode, PanicStrategy, PassMode};
use crate::util;
use crate::{extract_cdb_version, extract_gdb_version};

#[cfg(test)]
mod tests;

/// The result of parse_cfg_name_directive.
#[derive(Clone, Copy, PartialEq, Debug)]
enum ParsedNameDirective {
    /// No match.
    NoMatch,
    /// Match.
    Match,
}

/// Properties which must be known very early, before actually running
/// the test.
#[derive(Default)]
pub struct EarlyProps {
    pub aux: Vec<String>,
    pub aux_crate: Vec<(String, String)>,
    pub revisions: Vec<String>,
}

impl EarlyProps {
    pub fn from_file(config: &Config, testfile: &Path) -> Self {
        let file = File::open(testfile).expect("open test file to parse earlyprops");
        Self::from_reader(config, testfile, file)
    }

    pub fn from_reader<R: Read>(config: &Config, testfile: &Path, rdr: R) -> Self {
        let mut props = EarlyProps::default();
        iter_header(testfile, rdr, &mut |_, ln| {
            if let Some(s) = config.parse_aux_build(ln) {
                props.aux.push(s);
            }
            if let Some(ac) = config.parse_aux_crate(ln) {
                props.aux_crate.push(ac);
            }
            config.parse_and_update_revisions(ln, &mut props.revisions);
        });
        return props;
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
    // Similar to `aux_builds`, but a list of NAME=somelib.rs of dependencies
    // to build and pass with the `--extern` flag.
    pub aux_crates: Vec<(String, String)>,
    // Environment settings to use for compiling
    pub rustc_env: Vec<(String, String)>,
    // Environment variables to unset prior to compiling.
    // Variables are unset before applying 'rustc_env'.
    pub unset_rustc_env: Vec<String>,
    // Environment settings to use during execution
    pub exec_env: Vec<(String, String)>,
    // Build documentation for all specified aux-builds as well
    pub build_aux_docs: bool,
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
    // Run -Zunpretty expanded when running pretty printing tests
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
    // If `true`, this test will use incremental compilation.
    //
    // This can be set manually with the `incremental` header, or implicitly
    // by being a part of an incremental mode test. Using the `incremental`
    // header should be avoided if possible; using an incremental mode test is
    // preferred. Incremental mode tests support multiple passes, which can
    // verify that the incremental cache can be loaded properly after being
    // created. Just setting the header will only verify the behavior with
    // creating an incremental cache, but doesn't check that it is created
    // correctly.
    //
    // Compiletest will create the incremental directory, and ensure it is
    // empty before the test starts. Incremental mode tests will reuse the
    // incremental directory between passes in the same test.
    pub incremental: bool,
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
    pub failure_status: i32,
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
}

impl TestProps {
    pub fn new() -> Self {
        TestProps {
            error_patterns: vec![],
            compile_flags: vec![],
            run_flags: None,
            pp_exact: None,
            aux_builds: vec![],
            aux_crates: vec![],
            revisions: vec![],
            rustc_env: vec![],
            unset_rustc_env: vec![],
            exec_env: vec![],
            build_aux_docs: false,
            force_host: false,
            check_stdout: false,
            check_run_results: false,
            dont_check_compiler_stdout: false,
            dont_check_compiler_stderr: false,
            no_prefer_dynamic: false,
            pretty_expanded: false,
            pretty_mode: "normal".to_string(),
            pretty_compare_only: false,
            forbid_output: vec![],
            incremental_dir: None,
            incremental: false,
            pass_mode: None,
            fail_mode: None,
            ignore_pass: false,
            check_test_line_numbers_match: false,
            normalize_stdout: vec![],
            normalize_stderr: vec![],
            failure_status: -1,
            run_rustfix: false,
            rustfix_only_machine_applicable: false,
            assembly_output: None,
            should_ice: false,
            stderr_per_bitwidth: false,
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

        match (props.pass_mode, props.fail_mode) {
            (None, None) => props.fail_mode = Some(FailMode::Check),
            (Some(_), None) | (None, Some(_)) => {}
            (Some(_), Some(_)) => panic!("cannot use a *-fail and *-pass mode together"),
        }

        props
    }

    /// Loads properties from `testfile` into `props`. If a property is
    /// tied to a particular revision `foo` (indicated by writing
    /// `//[foo]`), then the property is ignored unless `cfg` is
    /// `Some("foo")`.
    fn load_from(&mut self, testfile: &Path, cfg: Option<&str>, config: &Config) {
        let mut has_edition = false;
        if !testfile.is_dir() {
            let file = File::open(testfile).unwrap();

            iter_header(testfile, file, &mut |revision, ln| {
                if revision.is_some() && revision != cfg {
                    return;
                }

                if let Some(ep) = config.parse_error_pattern(ln) {
                    self.error_patterns.push(ep);
                }

                if let Some(flags) = config.parse_compile_flags(ln) {
                    self.compile_flags.extend(flags.split_whitespace().map(|s| s.to_owned()));
                }

                if let Some(edition) = config.parse_edition(ln) {
                    self.compile_flags.push(format!("--edition={}", edition));
                    has_edition = true;
                }

                config.parse_and_update_revisions(ln, &mut self.revisions);

                if self.run_flags.is_none() {
                    self.run_flags = config.parse_run_flags(ln);
                }

                if self.pp_exact.is_none() {
                    self.pp_exact = config.parse_pp_exact(ln, testfile);
                }

                if !self.should_ice {
                    self.should_ice = config.parse_should_ice(ln);
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

                if !self.check_run_results {
                    self.check_run_results = config.parse_check_run_results(ln);
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

                if let Some(ac) = config.parse_aux_crate(ln) {
                    self.aux_crates.push(ac);
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

                if let Some(of) = config.parse_forbid_output(ln) {
                    self.forbid_output.push(of);
                }

                if !self.check_test_line_numbers_match {
                    self.check_test_line_numbers_match =
                        config.parse_check_test_line_numbers_match(ln);
                }

                self.update_pass_mode(ln, cfg, config);
                self.update_fail_mode(ln, config);

                if !self.ignore_pass {
                    self.ignore_pass = config.parse_ignore_pass(ln);
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

                if !self.stderr_per_bitwidth {
                    self.stderr_per_bitwidth = config.parse_stderr_per_bitwidth(ln);
                }

                if !self.incremental {
                    self.incremental = config.parse_incremental(ln);
                }
            });
        }

        if self.failure_status == -1 {
            self.failure_status = 1;
        }
        if self.should_ice {
            self.failure_status = 101;
        }

        if config.mode == Mode::Incremental {
            self.incremental = true;
        }

        for key in &["RUST_TEST_NOCAPTURE", "RUST_TEST_THREADS"] {
            if let Ok(val) = env::var(key) {
                if self.exec_env.iter().find(|&&(ref x, _)| x == key).is_none() {
                    self.exec_env.push(((*key).to_owned(), val))
                }
            }
        }

        if let (Some(edition), false) = (&config.edition, has_edition) {
            self.compile_flags.push(format!("--edition={}", edition));
        }
    }

    fn update_fail_mode(&mut self, ln: &str, config: &Config) {
        let check_ui = |mode: &str| {
            if config.mode != Mode::Ui {
                panic!("`{}-fail` header is only supported in UI tests", mode);
            }
        };
        if config.mode == Mode::Ui && config.parse_name_directive(ln, "compile-fail") {
            panic!("`compile-fail` header is useless in UI tests");
        }
        let fail_mode = if config.parse_name_directive(ln, "check-fail") {
            check_ui("check");
            Some(FailMode::Check)
        } else if config.parse_name_directive(ln, "build-fail") {
            check_ui("build");
            Some(FailMode::Build)
        } else if config.parse_name_directive(ln, "run-fail") {
            check_ui("run");
            Some(FailMode::Run)
        } else {
            None
        };
        match (self.fail_mode, fail_mode) {
            (None, Some(_)) => self.fail_mode = fail_mode,
            (Some(_), Some(_)) => panic!("multiple `*-fail` headers in a single test"),
            (_, None) => {}
        }
    }

    fn update_pass_mode(&mut self, ln: &str, revision: Option<&str>, config: &Config) {
        let check_no_run = |s| {
            if config.mode != Mode::Ui && config.mode != Mode::Incremental {
                panic!("`{}` header is only supported in UI and incremental tests", s);
            }
            if config.mode == Mode::Incremental
                && !revision.map_or(false, |r| r.starts_with("cfail"))
                && !self.revisions.iter().all(|r| r.starts_with("cfail"))
            {
                panic!("`{}` header is only supported in `cfail` incremental tests", s);
            }
        };
        let pass_mode = if config.parse_name_directive(ln, "check-pass") {
            check_no_run("check-pass");
            Some(PassMode::Check)
        } else if config.parse_name_directive(ln, "build-pass") {
            check_no_run("build-pass");
            Some(PassMode::Build)
        } else if config.parse_name_directive(ln, "run-pass") {
            if config.mode != Mode::Ui {
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
        if !self.ignore_pass && self.fail_mode.is_none() && config.mode == Mode::Ui {
            if let (mode @ Some(_), Some(_)) = (config.force_pass_mode, self.pass_mode) {
                return mode;
            }
        }
        self.pass_mode
    }

    // does not consider CLI override for pass mode
    pub fn local_pass_mode(&self) -> Option<PassMode> {
        self.pass_mode
    }
}

fn iter_header<R: Read>(testfile: &Path, rdr: R, it: &mut dyn FnMut(Option<&str>, &str)) {
    if testfile.is_dir() {
        return;
    }

    let comment = if testfile.extension().map(|e| e == "rs") == Some(true) { "//" } else { "#" };

    let mut rdr = BufReader::new(rdr);
    let mut ln = String::new();

    loop {
        ln.clear();
        if rdr.read_line(&mut ln).unwrap() == 0 {
            break;
        }

        // Assume that any directives will be found before the first
        // module or function. This doesn't seem to be an optimization
        // with a warm page cache. Maybe with a cold one.
        let ln = ln.trim();
        if ln.starts_with("fn") || ln.starts_with("mod") {
            return;
        } else if ln.starts_with(comment) && ln[comment.len()..].trim_start().starts_with('[') {
            // A comment like `//[foo]` is specific to revision `foo`
            if let Some(close_brace) = ln.find(']') {
                let open_brace = ln.find('[').unwrap();
                let lncfg = &ln[open_brace + 1..close_brace];
                it(Some(lncfg), ln[(close_brace + 1)..].trim_start());
            } else {
                panic!("malformed condition directive: expected `{}[foo]`, found `{}`", comment, ln)
            }
        } else if ln.starts_with(comment) {
            it(None, ln[comment.len()..].trim_start());
        }
    }
}

impl Config {
    fn parse_should_ice(&self, line: &str) -> bool {
        self.parse_name_directive(line, "should-ice")
    }
    fn parse_error_pattern(&self, line: &str) -> Option<String> {
        self.parse_name_value_directive(line, "error-pattern")
    }

    fn parse_forbid_output(&self, line: &str) -> Option<String> {
        self.parse_name_value_directive(line, "forbid-output")
    }

    fn parse_aux_build(&self, line: &str) -> Option<String> {
        self.parse_name_value_directive(line, "aux-build").map(|r| r.trim().to_string())
    }

    fn parse_aux_crate(&self, line: &str) -> Option<(String, String)> {
        self.parse_name_value_directive(line, "aux-crate").map(|r| {
            let mut parts = r.trim().splitn(2, '=');
            (
                parts.next().expect("missing aux-crate name (e.g. log=log.rs)").to_string(),
                parts.next().expect("missing aux-crate value (e.g. log=log.rs)").to_string(),
            )
        })
    }

    fn parse_compile_flags(&self, line: &str) -> Option<String> {
        self.parse_name_value_directive(line, "compile-flags")
    }

    fn parse_and_update_revisions(&self, line: &str, existing: &mut Vec<String>) {
        if let Some(raw) = self.parse_name_value_directive(line, "revisions") {
            let mut duplicates: HashSet<_> = existing.iter().cloned().collect();
            for revision in raw.split_whitespace().map(|r| r.to_string()) {
                if !duplicates.insert(revision.clone()) {
                    panic!("Duplicate revision: `{}` in line `{}`", revision, raw);
                }
                existing.push(revision);
            }
        }
    }

    fn parse_run_flags(&self, line: &str) -> Option<String> {
        self.parse_name_value_directive(line, "run-flags")
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

    fn parse_check_run_results(&self, line: &str) -> bool {
        self.parse_name_directive(line, "check-run-results")
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

    fn parse_check_test_line_numbers_match(&self, line: &str) -> bool {
        self.parse_name_directive(line, "check-test-line-numbers-match")
    }

    fn parse_ignore_pass(&self, line: &str) -> bool {
        self.parse_name_directive(line, "ignore-pass")
    }

    fn parse_stderr_per_bitwidth(&self, line: &str) -> bool {
        self.parse_name_directive(line, "stderr-per-bitwidth")
    }

    fn parse_assembly_output(&self, line: &str) -> Option<String> {
        self.parse_name_value_directive(line, "assembly-output").map(|r| r.trim().to_string())
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

    /// Parses a name-value directive which contains config-specific information, e.g., `ignore-x86`
    /// or `normalize-stderr-32bit`.
    fn parse_cfg_name_directive(&self, line: &str, prefix: &str) -> ParsedNameDirective {
        if !line.as_bytes().starts_with(prefix.as_bytes()) {
            return ParsedNameDirective::NoMatch;
        }
        if line.as_bytes().get(prefix.len()) != Some(&b'-') {
            return ParsedNameDirective::NoMatch;
        }

        let name = line[prefix.len() + 1..].split(&[':', ' '][..]).next().unwrap();

        let is_match = name == "test" ||
            self.target == name ||                              // triple
            util::matches_os(&self.target, name) ||             // target
            util::matches_env(&self.target, name) ||            // env
            self.target.ends_with(name) ||                      // target and env
            name == util::get_arch(&self.target) ||             // architecture
            name == util::get_pointer_width(&self.target) ||    // pointer width
            name == self.stage_id.split('-').next().unwrap() || // stage
            name == self.channel ||                             // channel
            (self.target != self.host && name == "cross-compile") ||
            (name == "endian-big" && util::is_big_endian(&self.target)) ||
            (self.remote_test_client.is_some() && name == "remote") ||
            match self.compare_mode {
                Some(CompareMode::Nll) => name == "compare-mode-nll",
                Some(CompareMode::Polonius) => name == "compare-mode-polonius",
                Some(CompareMode::Chalk) => name == "compare-mode-chalk",
                Some(CompareMode::SplitDwarf) => name == "compare-mode-split-dwarf",
                Some(CompareMode::SplitDwarfSingle) => name == "compare-mode-split-dwarf-single",
                None => false,
            } ||
            (cfg!(debug_assertions) && name == "debug") ||
            match self.debugger {
                Some(Debugger::Cdb) => name == "cdb",
                Some(Debugger::Gdb) => name == "gdb",
                Some(Debugger::Lldb) => name == "lldb",
                None => false,
            };

        if is_match { ParsedNameDirective::Match } else { ParsedNameDirective::NoMatch }
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
        line.starts_with(directive)
            && matches!(line.as_bytes().get(directive.len()), None | Some(&b' ') | Some(&b':'))
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

    fn parse_incremental(&self, line: &str) -> bool {
        self.parse_name_directive(line, "incremental")
    }
}

fn expand_variables(mut value: String, config: &Config) -> String {
    const CWD: &str = "{{cwd}}";
    const SRC_BASE: &str = "{{src-base}}";
    const BUILD_BASE: &str = "{{build-base}}";

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

pub fn extract_llvm_version(version: &str) -> Option<u32> {
    let pat = |c: char| !c.is_ascii_digit() && c != '.';
    let version_without_suffix = match version.find(pat) {
        Some(pos) => &version[..pos],
        None => version,
    };
    let components: Vec<u32> = version_without_suffix
        .split('.')
        .map(|s| s.parse().expect("Malformed version component"))
        .collect();
    let version = match *components {
        [a] => a * 10_000,
        [a, b] => a * 10_000 + b * 100,
        [a, b, c] => a * 10_000 + b * 100 + c,
        _ => panic!("Malformed version"),
    };
    Some(version)
}

/// Takes a directive of the form "<version1> [- <version2>]",
/// returns the numeric representation of <version1> and <version2> as
/// tuple: (<version1> as u32, <version2> as u32)
///
/// If the <version2> part is omitted, the second component of the tuple
/// is the same as <version1>.
fn extract_version_range<F>(line: &str, parse: F) -> Option<(u32, u32)>
where
    F: Fn(&str) -> Option<u32>,
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
        Some(max) if max.is_empty() => return None,
        Some(max) => parse(max)?,
        _ => min,
    };

    Some((min, max))
}

pub fn make_test_description<R: Read>(
    config: &Config,
    name: test::TestName,
    path: &Path,
    src: R,
    cfg: Option<&str>,
) -> test::TestDesc {
    let mut ignore = false;
    let mut should_fail = false;

    let rustc_has_profiler_support = env::var_os("RUSTC_PROFILER_SUPPORT").is_some();
    let rustc_has_sanitizer_support = env::var_os("RUSTC_SANITIZER_SUPPORT").is_some();
    let has_asm_support = util::has_asm_support(&config.target);
    let has_asan = util::ASAN_SUPPORTED_TARGETS.contains(&&*config.target);
    let has_lsan = util::LSAN_SUPPORTED_TARGETS.contains(&&*config.target);
    let has_msan = util::MSAN_SUPPORTED_TARGETS.contains(&&*config.target);
    let has_tsan = util::TSAN_SUPPORTED_TARGETS.contains(&&*config.target);
    let has_hwasan = util::HWASAN_SUPPORTED_TARGETS.contains(&&*config.target);
    // for `-Z gcc-ld=lld`
    let has_rust_lld = config
        .compile_lib_path
        .join("rustlib")
        .join(&config.target)
        .join("bin")
        .join("gcc-ld")
        .join(if config.host.contains("windows") { "ld.exe" } else { "ld" })
        .exists();
    iter_header(path, src, &mut |revision, ln| {
        if revision.is_some() && revision != cfg {
            return;
        }
        ignore = match config.parse_cfg_name_directive(ln, "ignore") {
            ParsedNameDirective::Match => true,
            ParsedNameDirective::NoMatch => ignore,
        };
        if config.has_cfg_prefix(ln, "only") {
            ignore = match config.parse_cfg_name_directive(ln, "only") {
                ParsedNameDirective::Match => ignore,
                ParsedNameDirective::NoMatch => true,
            };
        }
        ignore |= ignore_llvm(config, ln);
        ignore |=
            config.run_clang_based_tests_with.is_none() && config.parse_needs_matching_clang(ln);
        ignore |= !has_asm_support && config.parse_name_directive(ln, "needs-asm-support");
        ignore |= !rustc_has_profiler_support && config.parse_needs_profiler_support(ln);
        ignore |= !config.run_enabled() && config.parse_name_directive(ln, "needs-run-enabled");
        ignore |= !rustc_has_sanitizer_support
            && config.parse_name_directive(ln, "needs-sanitizer-support");
        ignore |= !has_asan && config.parse_name_directive(ln, "needs-sanitizer-address");
        ignore |= !has_lsan && config.parse_name_directive(ln, "needs-sanitizer-leak");
        ignore |= !has_msan && config.parse_name_directive(ln, "needs-sanitizer-memory");
        ignore |= !has_tsan && config.parse_name_directive(ln, "needs-sanitizer-thread");
        ignore |= !has_hwasan && config.parse_name_directive(ln, "needs-sanitizer-hwaddress");
        ignore |= config.target_panic == PanicStrategy::Abort
            && config.parse_name_directive(ln, "needs-unwind");
        ignore |= config.target == "wasm32-unknown-unknown" && config.parse_check_run_results(ln);
        ignore |= config.debugger == Some(Debugger::Cdb) && ignore_cdb(config, ln);
        ignore |= config.debugger == Some(Debugger::Gdb) && ignore_gdb(config, ln);
        ignore |= config.debugger == Some(Debugger::Lldb) && ignore_lldb(config, ln);
        ignore |= !has_rust_lld && config.parse_name_directive(ln, "needs-rust-lld");
        should_fail |= config.parse_name_directive(ln, "should-fail");
    });

    // The `should-fail` annotation doesn't apply to pretty tests,
    // since we run the pretty printer across all tests by default.
    // If desired, we could add a `should-fail-pretty` annotation.
    let should_panic = match config.mode {
        crate::common::Pretty => test::ShouldPanic::No,
        _ if should_fail => test::ShouldPanic::Yes,
        _ => test::ShouldPanic::No,
    };

    test::TestDesc {
        name,
        ignore,
        should_panic,
        allow_fail: false,
        compile_fail: false,
        no_run: false,
        test_type: test::TestType::Unknown,
    }
}

fn ignore_cdb(config: &Config, line: &str) -> bool {
    if let Some(actual_version) = config.cdb_version {
        if let Some(min_version) = line.strip_prefix("min-cdb-version:").map(str::trim) {
            let min_version = extract_cdb_version(min_version).unwrap_or_else(|| {
                panic!("couldn't parse version range: {:?}", min_version);
            });

            // Ignore if actual version is smaller than the minimum
            // required version
            return actual_version < min_version;
        }
    }
    false
}

fn ignore_gdb(config: &Config, line: &str) -> bool {
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
            return actual_version < start_ver;
        } else if let Some(rest) = line.strip_prefix("ignore-gdb-version:").map(str::trim) {
            let (min_version, max_version) = extract_version_range(rest, extract_gdb_version)
                .unwrap_or_else(|| {
                    panic!("couldn't parse version range: {:?}", rest);
                });

            if max_version < min_version {
                panic!("Malformed GDB version range: max < min")
            }

            return actual_version >= min_version && actual_version <= max_version;
        }
    }
    false
}

fn ignore_lldb(config: &Config, line: &str) -> bool {
    if let Some(actual_version) = config.lldb_version {
        if let Some(min_version) = line.strip_prefix("min-lldb-version:").map(str::trim) {
            let min_version = min_version.parse().unwrap_or_else(|e| {
                panic!("Unexpected format of LLDB version string: {}\n{:?}", min_version, e);
            });
            // Ignore if actual version is smaller the minimum required
            // version
            actual_version < min_version
        } else {
            line.starts_with("rust-lldb") && !config.lldb_native_rust
        }
    } else {
        false
    }
}

fn ignore_llvm(config: &Config, line: &str) -> bool {
    if config.system_llvm && line.starts_with("no-system-llvm") {
        return true;
    }
    if let Some(needed_components) =
        config.parse_name_value_directive(line, "needs-llvm-components")
    {
        let components: HashSet<_> = config.llvm_components.split_whitespace().collect();
        if let Some(missing_component) = needed_components
            .split_whitespace()
            .find(|needed_component| !components.contains(needed_component))
        {
            if env::var_os("COMPILETEST_NEEDS_ALL_LLVM_COMPONENTS").is_some() {
                panic!("missing LLVM component: {}", missing_component);
            }
            return true;
        }
    }
    if let Some(actual_version) = config.llvm_version {
        if let Some(rest) = line.strip_prefix("min-llvm-version:").map(str::trim) {
            let min_version = extract_llvm_version(rest).unwrap();
            // Ignore if actual version is smaller the minimum required
            // version
            actual_version < min_version
        } else if let Some(rest) = line.strip_prefix("min-system-llvm-version:").map(str::trim) {
            let min_version = extract_llvm_version(rest).unwrap();
            // Ignore if using system LLVM and actual version
            // is smaller the minimum required version
            config.system_llvm && actual_version < min_version
        } else if let Some(rest) = line.strip_prefix("ignore-llvm-version:").map(str::trim) {
            // Syntax is: "ignore-llvm-version: <version1> [- <version2>]"
            let (v_min, v_max) =
                extract_version_range(rest, extract_llvm_version).unwrap_or_else(|| {
                    panic!("couldn't parse version range: {:?}", rest);
                });
            if v_max < v_min {
                panic!("Malformed LLVM version range: max < min")
            }
            // Ignore if version lies inside of range.
            actual_version >= v_min && actual_version <= v_max
        } else {
            false
        }
    } else {
        false
    }
}
