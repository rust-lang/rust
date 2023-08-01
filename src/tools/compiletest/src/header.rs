use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::process::Command;

use tracing::*;

use test_common::directives::{self, NameDirective, NameValueDirective};
use test_common::TestComment;

use crate::common::{Config, Debugger, FailMode, Mode, PassMode};
use crate::header::cfg::parse_cfg_name_directive;
use crate::header::cfg::MatchOutcome;
use crate::header::needs::CachedNeedsConditions;
use crate::{extract_cdb_version, extract_gdb_version};

mod cfg;
mod needs;
#[cfg(test)]
mod tests;

pub struct HeadersCache {
    needs: CachedNeedsConditions,
}

impl HeadersCache {
    pub fn load(config: &Config) -> Self {
        Self { needs: CachedNeedsConditions::load(config) }
    }
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
        test_common::iter_header(testfile, rdr, &mut |comment| {
            use directives::*;

            config.push_name_value_directive(&comment, AuxBuildDirective, &mut props.aux, |r| {
                r.trim().to_string()
            });
            config.push_name_value_directive(
                &comment,
                AuxCrateDirective,
                &mut props.aux_crate,
                Config::parse_aux_crate,
            );

            config.parse_and_update_revisions(&comment, &mut props.revisions);
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
    // Environment variables to unset prior to execution.
    // Variables are unset before applying 'exec_env'
    pub unset_exec_env: Vec<String>,
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
    // When checking the output of stdout or stderr check
    // that the lines of expected output are a subset of the actual output.
    pub compare_output_lines_by_subset: bool,
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
            rustc_env: vec![("RUSTC_ICE".to_string(), "0".to_string())],
            unset_rustc_env: vec![],
            exec_env: vec![],
            unset_exec_env: vec![],
            build_aux_docs: false,
            force_host: false,
            check_stdout: false,
            check_run_results: false,
            dont_check_compiler_stdout: false,
            dont_check_compiler_stderr: false,
            compare_output_lines_by_subset: false,
            no_prefer_dynamic: false,
            pretty_expanded: false,
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
        }
    }

    pub fn from_aux_file(&self, testfile: &Path, cfg: Option<&str>, config: &Config) -> Self {
        let mut props = TestProps::new();

        // copy over select properties to the aux build:
        props.incremental_dir = self.incremental_dir.clone();
        props.ignore_pass = true;
        props.load_from(testfile, cfg, config);

        props
    }

    pub fn from_file(testfile: &Path, cfg: Option<&str>, config: &Config) -> Self {
        let mut props = TestProps::new();
        props.load_from(testfile, cfg, config);

        match (props.pass_mode, props.fail_mode) {
            (None, None) if config.mode == Mode::Ui => props.fail_mode = Some(FailMode::Check),
            (Some(_), Some(_)) => panic!("cannot use a *-fail and *-pass mode together"),
            _ => {}
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

            test_common::iter_header(testfile, file, &mut |comment| {
                if comment.revision().is_some() && comment.revision() != cfg {
                    return;
                }

                use directives::*;

                config.push_name_value_directive(
                    &comment,
                    ErrorPatternDirective,
                    &mut self.error_patterns,
                    |r| r,
                );

                if let Some(flags) = config.parse_expand_name_value(CompileFlagsDirective, &comment)
                {
                    self.compile_flags.extend(flags.split_whitespace().map(|s| s.to_owned()));
                }

                if IncorrectCompileFlagsDirective.parse_name_value(&comment).is_some() {
                    panic!("`compiler-flags` directive should be spelled `compile-flags`");
                }

                if let Some(edition) = config.parse_edition(&comment) {
                    self.compile_flags.push(format!("--edition={}", edition.trim()));
                    has_edition = true;
                }

                config.parse_and_update_revisions(&comment, &mut self.revisions);

                config.set_name_value_directive(
                    &comment,
                    RunFlagsDirective,
                    &mut self.run_flags,
                    |r| r,
                );

                if self.pp_exact.is_none() {
                    self.pp_exact = config.parse_pp_exact(&comment, testfile);
                }

                config.set_name_directive(&comment, ShouldIceDirective, &mut self.should_ice);
                config.set_name_directive(
                    &comment,
                    BuildAuxDocsDirective,
                    &mut self.build_aux_docs,
                );
                config.set_name_directive(&comment, ForceHostDirective, &mut self.force_host);
                config.set_name_directive(&comment, CheckStdoutDirective, &mut self.check_stdout);
                config.set_name_directive(
                    &comment,
                    CheckRunResultsDirective,
                    &mut self.check_run_results,
                );
                config.set_name_directive(
                    &comment,
                    DontCheckCompilerStdoutDirective,
                    &mut self.dont_check_compiler_stdout,
                );
                config.set_name_directive(
                    &comment,
                    DontCheckCompilerStderrDirective,
                    &mut self.dont_check_compiler_stderr,
                );
                config.set_name_directive(
                    &comment,
                    NoPreferDynamicDirective,
                    &mut self.no_prefer_dynamic,
                );
                config.set_name_directive(
                    &comment,
                    PrettyExpandedDirective,
                    &mut self.pretty_expanded,
                );

                if let Some(m) = config.parse_expand_name_value(PrettyModeDirective, &comment) {
                    self.pretty_mode = m;
                }

                config.set_name_directive(
                    &comment,
                    PrettyCompareOnlyDirective,
                    &mut self.pretty_compare_only,
                );

                config.push_name_value_directive(
                    &comment,
                    AuxBuildDirective,
                    &mut self.aux_builds,
                    |r| r.trim().to_string(),
                );
                config.push_name_value_directive(
                    &comment,
                    AuxCrateDirective,
                    &mut self.aux_crates,
                    Config::parse_aux_crate,
                );
                config.push_name_value_directive(
                    &comment,
                    ExecEnvDirective,
                    &mut self.exec_env,
                    Config::parse_env,
                );
                config.push_name_value_directive(
                    &comment,
                    UnsetExecEnvDirective,
                    &mut self.unset_exec_env,
                    |r| r,
                );
                config.push_name_value_directive(
                    &comment,
                    RustcEnvDirective,
                    &mut self.rustc_env,
                    Config::parse_env,
                );
                config.push_name_value_directive(
                    &comment,
                    UnsetRustcEnvDirective,
                    &mut self.unset_rustc_env,
                    |r| r,
                );
                config.push_name_value_directive(
                    &comment,
                    ForbidOutputDirective,
                    &mut self.forbid_output,
                    |r| r,
                );

                config.set_name_directive(
                    &comment,
                    CheckTestLineNumbersMatchDirective,
                    &mut self.check_test_line_numbers_match,
                );

                self.update_pass_mode(&comment, cfg, config);
                self.update_fail_mode(&comment, config);

                config.set_name_directive(&comment, IgnorePassDirective, &mut self.ignore_pass);

                if let Some(rule) =
                    config.parse_custom_normalization(comment.comment_str(), "normalize-stdout")
                {
                    self.normalize_stdout.push(rule);
                }
                if let Some(rule) =
                    config.parse_custom_normalization(comment.comment_str(), "normalize-stderr")
                {
                    self.normalize_stderr.push(rule);
                }

                if let Some(code) = config
                    .parse_expand_name_value(FailureStatusDirective, &comment)
                    .and_then(|code| code.trim().parse::<i32>().ok())
                {
                    self.failure_status = Some(code);
                }

                config.set_name_directive(
                    &comment,
                    DontCheckFailureStatusDirective,
                    &mut self.dont_check_failure_status,
                );
                config.set_name_directive(&comment, RunRustfixDirective, &mut self.run_rustfix);
                config.set_name_directive(
                    &comment,
                    RustfixOnlyMachineApplicableDirective,
                    &mut self.rustfix_only_machine_applicable,
                );
                config.set_name_value_directive(
                    &comment,
                    AssemblyOutputDirective,
                    &mut self.assembly_output,
                    |r| r.trim().to_string(),
                );
                config.set_name_directive(
                    &comment,
                    StderrPerBitwidthDirective,
                    &mut self.stderr_per_bitwidth,
                );
                config.set_name_directive(&comment, IncrementalDirective, &mut self.incremental);

                // Unlike the other `name_value_directive`s this needs to be handled manually,
                // because it sets a `bool` flag.
                if let Some(known_bug) = KnownBugDirective.parse_name_value(&comment) {
                    // Because known-bug has such a strict value format,
                    // don't expand variables inside it, they'll never match.
                    let known_bug = known_bug.unexpanded_value().trim();
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
                } else if KnownBugDirective.parse_name(&comment) {
                    panic!(
                        "Invalid known-bug attribute, requires comma-separated issue references (`#000` or `chalk#000`) or `known-bug: unknown`."
                    );
                }

                config.set_name_value_directive(
                    &comment,
                    MirUnitTestDirective,
                    &mut self.mir_unit_test,
                    |s| s.trim().to_string(),
                );
                config.set_name_directive(
                    &comment,
                    RemapSrcBaseDirective,
                    &mut self.remap_src_base,
                );
                config.set_name_directive(
                    &comment,
                    CompareOutputLinesBySubsetDirective,
                    &mut self.compare_output_lines_by_subset,
                );
            });
        }

        if self.should_ice {
            self.failure_status = Some(101);
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

    fn update_fail_mode(&mut self, comment: &TestComment<'_>, config: &Config) {
        let check_ui = |mode: &str| {
            if config.mode != Mode::Ui {
                panic!("`{}-fail` header is only supported in UI tests", mode);
            }
        };
        use directives::*;
        if config.mode == Mode::Ui && CompileFailDirective.parse_name(comment) {
            panic!("`compile-fail` header is useless in UI tests");
        }
        let fail_mode = if CheckFailDirective.parse_name(comment) {
            check_ui("check");
            Some(FailMode::Check)
        } else if BuildFailDirective.parse_name(comment) {
            check_ui("build");
            Some(FailMode::Build)
        } else if RunFailDirective.parse_name(comment) {
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

    fn update_pass_mode(
        &mut self,
        comment: &TestComment<'_>,
        revision: Option<&str>,
        config: &Config,
    ) {
        use directives::*;
        let check_no_run = |s| match (config.mode, s) {
            (Mode::Ui, _) => (),
            (Mode::Codegen, "build-pass") => (),
            (Mode::Incremental, _) => {
                if revision.is_some() && !self.revisions.iter().all(|r| r.starts_with("cfail")) {
                    panic!("`{s}` header is only supported in `cfail` incremental tests")
                }
            }
            (mode, _) => panic!("`{s}` header is not supported in `{mode}` tests"),
        };
        let pass_mode = if CheckPassDirective.parse_name(comment) {
            check_no_run("check-pass");
            Some(PassMode::Check)
        } else if BuildPassDirective.parse_name(comment) {
            check_no_run("build-pass");
            Some(PassMode::Build)
        } else if RunPassDirective.parse_name(comment) {
            check_no_run("run-pass");
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
}

fn iter_header_extra(
    testfile: &Path,
    rdr: impl Read,
    extra_directives: &[&str],
    it: &mut dyn FnMut(TestComment<'_>),
) {
    if testfile.is_dir() {
        return;
    }

    // Process any extra directives supplied by the caller (e.g. because they
    // are implied by the test mode), with a dummy line number of 0.
    for directive in extra_directives {
        it(TestComment::new(directive, None, test_common::CommentKind::Compiletest(directive), 0));
    }

    test_common::iter_header(testfile, rdr, it);
}

impl Config {
    fn parse_aux_crate(r: String) -> (String, String) {
        let mut parts = r.trim().splitn(2, '=');
        (
            parts.next().expect("missing aux-crate name (e.g. log=log.rs)").to_string(),
            parts.next().expect("missing aux-crate value (e.g. log=log.rs)").to_string(),
        )
    }

    fn parse_and_update_revisions(&self, comment: &TestComment<'_>, existing: &mut Vec<String>) {
        // FIXME: (ui_test) this probably shouldn't expand actually
        if let Some(raw) = self.parse_expand_name_value(directives::RevisionsDirective, comment) {
            let mut duplicates: HashSet<_> = existing.iter().cloned().collect();
            for revision in raw.split_whitespace().map(|r| r.to_string()) {
                if !duplicates.insert(revision.clone()) {
                    panic!("Duplicate revision: `{}` in line `{}`", revision, raw);
                }
                existing.push(revision);
            }
        }
    }

    fn parse_env(nv: String) -> (String, String) {
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
    }

    fn parse_pp_exact(&self, comment: &TestComment<'_>, testfile: &Path) -> Option<PathBuf> {
        use directives::PPExactDirective;
        if let Some(s) = self.parse_expand_name_value(PPExactDirective, comment) {
            Some(PathBuf::from(&s))
        } else if PPExactDirective.parse_name(comment) {
            testfile.file_name().map(PathBuf::from)
        } else {
            None
        }
    }

    fn parse_custom_normalization(&self, mut line: &str, prefix: &str) -> Option<(String, String)> {
        if parse_cfg_name_directive(self, line, prefix).outcome == MatchOutcome::Match {
            let from = parse_normalization_string(&mut line)?;
            let to = parse_normalization_string(&mut line)?;
            Some((from, to))
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

    fn parse_edition(&self, comment: &TestComment<'_>) -> Option<String> {
        self.parse_expand_name_value(directives::EditionDirective, comment)
    }

    fn set_name_directive(
        &self,
        comment: &TestComment<'_>,
        directive: impl NameDirective,
        value: &mut bool,
    ) {
        match value {
            true => {
                if directive.parse_name_negative(comment) {
                    *value = false;
                }
            }
            false => {
                if directive.parse_name(comment) {
                    *value = true;
                }
            }
        }
    }

    /// This function should be avoided. It cannot accept ui_test style comments.
    // FIXME: (ui_test) can this be removed?
    pub fn parse_compiletest_arbitrary_name_value(
        &self,
        line: &str,
        directive: &str,
    ) -> Option<String> {
        if let Some(args) =
            line.strip_prefix(directive).and_then(|with_colon| with_colon.strip_prefix(':'))
        {
            debug!("{}: {}", directive, args);
            Some(expand_variables(args.to_string(), self))
        } else {
            None
        }
    }

    fn set_name_value_directive<T>(
        &self,
        comment: &TestComment<'_>,
        directive: impl NameValueDirective,
        value: &mut Option<T>,
        parse: impl FnOnce(String) -> T,
    ) {
        if value.is_none() {
            *value = self.parse_expand_name_value(directive, comment).map(parse);
        }
    }

    fn push_name_value_directive<T>(
        &self,
        comment: &TestComment<'_>,
        directive: impl NameValueDirective,
        values: &mut Vec<T>,
        parse: impl FnOnce(String) -> T,
    ) {
        if let Some(value) = self.parse_expand_name_value(directive, comment).map(parse) {
            values.push(value);
        }
    }

    fn parse_expand_name_value<'line>(
        &self,
        directive: impl NameValueDirective,
        comment: &'line TestComment<'_>,
    ) -> Option<String> {
        directive
            .parse_name_value(comment)
            .map(|unexpanded| expand_variables(unexpanded.unexpanded_value(), self))
    }
}

fn expand_variables(value: impl Into<String>, config: &Config) -> String {
    const CWD: &str = "{{cwd}}";
    const SRC_BASE: &str = "{{src-base}}";
    const BUILD_BASE: &str = "{{build-base}}";

    let mut value = value.into();

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

pub fn extract_llvm_version_from_binary(binary_path: &str) -> Option<u32> {
    let output = Command::new(binary_path).arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let version = String::from_utf8(output.stdout).ok()?;
    for line in version.lines() {
        if let Some(version) = line.split("LLVM version ").skip(1).next() {
            return extract_llvm_version(version);
        }
    }
    None
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
    cache: &HeadersCache,
    name: test::TestName,
    path: &Path,
    src: R,
    cfg: Option<&str>,
    poisoned: &mut bool,
) -> test::TestDesc {
    let mut ignore = false;
    let mut ignore_message = None;
    let mut should_fail = false;

    let extra_directives: &[&str] = match config.mode {
        // The run-coverage tests are treated as having these extra directives,
        // without needing to specify them manually in every test file.
        // (Some of the comments below have been copied over from
        // `tests/run-make/coverage-reports/Makefile`, which no longer exists.)
        Mode::RunCoverage => {
            &[
                "needs-profiler-support",
                // FIXME(mati865): MinGW GCC miscompiles compiler-rt profiling library but with Clang it works
                // properly. Since we only have GCC on the CI ignore the test for now.
                "ignore-windows-gnu",
                // FIXME(pietroalbini): this test currently does not work on cross-compiled
                // targets because remote-test is not capable of sending back the *.profraw
                // files generated by the LLVM instrumentation.
                "ignore-cross-compile",
            ]
        }
        _ => &[],
    };

    iter_header_extra(path, src, extra_directives, &mut |comment| {
        if comment.revision().is_some() && comment.revision() != cfg {
            return;
        }

        let line_number = comment.line_num();

        macro_rules! decision {
            ($e:expr) => {
                match $e {
                    IgnoreDecision::Ignore { reason } => {
                        ignore = true;
                        // The ignore reason must be a &'static str, so we have to leak memory to
                        // create it. This is fine, as the header is parsed only at the start of
                        // compiletest so it won't grow indefinitely.
                        ignore_message = Some(&*Box::leak(Box::<str>::from(reason)));
                    }
                    IgnoreDecision::Error { message } => {
                        eprintln!("error: {}:{line_number}: {message}", path.display());
                        *poisoned = true;
                        return;
                    }
                    IgnoreDecision::Continue => {}
                }
            };
        }

        let ln = comment.comment_str();
        decision!(cfg::handle_ignore(config, ln));
        decision!(cfg::handle_only(config, ln));
        decision!(needs::handle_needs(&cache.needs, config, ln));
        decision!(ignore_llvm(config, &comment));
        decision!(ignore_cdb(config, ln));
        decision!(ignore_gdb(config, ln));
        decision!(ignore_lldb(config, ln));

        if config.target == "wasm32-unknown-unknown" {
            if directives::CheckRunResultsDirective.parse_name(&comment) {
                decision!(IgnoreDecision::Ignore {
                    reason: "ignored when checking the run results on WASM".into(),
                });
            }
        }

        should_fail |= directives::ShouldFailDirective.parse_name(&comment);
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
        ignore_message,
        source_file: "",
        start_line: 0,
        start_col: 0,
        end_line: 0,
        end_col: 0,
        should_panic,
        compile_fail: false,
        no_run: false,
        test_type: test::TestType::Unknown,
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

fn ignore_llvm(config: &Config, comment: &TestComment<'_>) -> IgnoreDecision {
    let line = comment.comment_str();
    if config.system_llvm && line.starts_with("no-system-llvm") {
        return IgnoreDecision::Ignore { reason: "ignored when the system LLVM is used".into() };
    }
    if let Some(needed_components) =
        config.parse_expand_name_value(directives::NeedsLlvmComponentsDirective, comment)
    {
        let components: HashSet<_> = config.llvm_components.split_whitespace().collect();
        if let Some(missing_component) = needed_components
            .split_whitespace()
            .find(|needed_component| !components.contains(needed_component))
        {
            if env::var_os("COMPILETEST_NEEDS_ALL_LLVM_COMPONENTS").is_some() {
                panic!("missing LLVM component: {}", missing_component);
            }
            return IgnoreDecision::Ignore {
                reason: format!("ignored when the {missing_component} LLVM component is missing"),
            };
        }
    }
    if let Some(actual_version) = config.llvm_version {
        if let Some(rest) = line.strip_prefix("min-llvm-version:").map(str::trim) {
            let min_version = extract_llvm_version(rest).unwrap();
            // Ignore if actual version is smaller the minimum required
            // version
            if actual_version < min_version {
                return IgnoreDecision::Ignore {
                    reason: format!("ignored when the LLVM version is older than {rest}"),
                };
            }
        } else if let Some(rest) = line.strip_prefix("min-system-llvm-version:").map(str::trim) {
            let min_version = extract_llvm_version(rest).unwrap();
            // Ignore if using system LLVM and actual version
            // is smaller the minimum required version
            if config.system_llvm && actual_version < min_version {
                return IgnoreDecision::Ignore {
                    reason: format!("ignored when the system LLVM version is older than {rest}"),
                };
            }
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
            if actual_version >= v_min && actual_version <= v_max {
                if v_min == v_max {
                    return IgnoreDecision::Ignore {
                        reason: format!("ignored when the LLVM version is {rest}"),
                    };
                } else {
                    return IgnoreDecision::Ignore {
                        reason: format!("ignored when the LLVM version is between {rest}"),
                    };
                }
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
