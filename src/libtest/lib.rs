//! Support code for rustc's built in unit-test and micro-benchmarking
//! framework.
//!
//! Almost all user code will only be interested in `Bencher` and
//! `black_box`. All other interactions (such as writing tests and
//! benchmarks themselves) should be done via the `#[test]` and
//! `#[bench]` attributes.
//!
//! See the [Testing Chapter](../book/ch11-00-testing.html) of the book for more details.

// Currently, not much of this is meant for users. It is intended to
// support the simplest interface possible for representing and
// running tests while providing a base that other test frameworks may
// build off of.

// N.B., this is also specified in this crate's Cargo.toml, but libsyntax contains logic specific to
// this crate, which relies on this attribute (rather than the value of `--crate-name` passed by
// cargo) to detect this crate.

#![deny(rust_2018_idioms)]
#![crate_name = "test"]
#![unstable(feature = "test", issue = "27812")]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/", test(attr(deny(warnings))))]
#![feature(asm)]
#![cfg_attr(any(unix, target_os = "cloudabi"), feature(libc, rustc_private))]
#![feature(nll)]
#![feature(set_stdio)]
#![feature(panic_unwind)]
#![feature(staged_api)]
#![feature(termination_trait_lib)]
#![feature(test)]

use getopts;
#[cfg(any(unix, target_os = "cloudabi"))]
extern crate libc;
use term;

// FIXME(#54291): rustc and/or LLVM don't yet support building with panic-unwind
//                on aarch64-pc-windows-msvc, or thumbv7a-pc-windows-msvc
//                so we don't link libtest against libunwind (for the time being)
//                even though it means that libtest won't be fully functional on
//                these platforms.
//
// See also: https://github.com/rust-lang/rust/issues/54190#issuecomment-422904437
#[cfg(not(all(windows, any(target_arch = "aarch64", target_arch = "arm"))))]
extern crate panic_unwind;

pub use self::ColorConfig::*;
use self::NamePadding::*;
use self::OutputLocation::*;
use self::TestEvent::*;
pub use self::TestFn::*;
pub use self::TestName::*;
pub use self::TestResult::*;

use std::any::Any;
use std::borrow::Cow;
use std::cmp;
use std::collections::BTreeMap;
use std::env;
use std::fmt;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::process;
use std::process::Termination;
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const TEST_WARN_TIMEOUT_S: u64 = 60;
const QUIET_MODE_MAX_COLUMN: usize = 100; // insert a '\n' after 100 tests in quiet mode

// to be used by rustc to compile tests in libtest
pub mod test {
    pub use crate::{
        assert_test_result, filter_tests, parse_opts, run_test, test_main, test_main_static,
        Bencher, DynTestFn, DynTestName, Metric, MetricMap, Options, RunIgnored, ShouldPanic,
        StaticBenchFn, StaticTestFn, StaticTestName, TestDesc, TestDescAndFn, TestName, TestOpts,
        TestResult, TrFailed, TrFailedMsg, TrIgnored, TrOk,
    };
}

mod formatters;
pub mod stats;

use crate::formatters::{JsonFormatter, OutputFormatter, PrettyFormatter, TerseFormatter};

/// Whether to execute tests concurrently or not
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Concurrent {
    Yes,
    No,
}

// The name of a test. By convention this follows the rules for rust
// paths; i.e., it should be a series of identifiers separated by double
// colons. This way if some test runner wants to arrange the tests
// hierarchically it may.

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TestName {
    StaticTestName(&'static str),
    DynTestName(String),
    AlignedTestName(Cow<'static, str>, NamePadding),
}
impl TestName {
    fn as_slice(&self) -> &str {
        match *self {
            StaticTestName(s) => s,
            DynTestName(ref s) => s,
            AlignedTestName(ref s, _) => &*s,
        }
    }

    fn padding(&self) -> NamePadding {
        match self {
            &AlignedTestName(_, p) => p,
            _ => PadNone,
        }
    }

    fn with_padding(&self, padding: NamePadding) -> TestName {
        let name = match self {
            &TestName::StaticTestName(name) => Cow::Borrowed(name),
            &TestName::DynTestName(ref name) => Cow::Owned(name.clone()),
            &TestName::AlignedTestName(ref name, _) => name.clone(),
        };

        TestName::AlignedTestName(name, padding)
    }
}
impl fmt::Display for TestName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_slice(), f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum NamePadding {
    PadNone,
    PadOnRight,
}

impl TestDesc {
    fn padded_name(&self, column_count: usize, align: NamePadding) -> String {
        let mut name = String::from(self.name.as_slice());
        let fill = column_count.saturating_sub(name.len());
        let pad = " ".repeat(fill);
        match align {
            PadNone => name,
            PadOnRight => {
                name.push_str(&pad);
                name
            }
        }
    }
}

/// Represents a benchmark function.
pub trait TDynBenchFn: Send {
    fn run(&self, harness: &mut Bencher);
}

// A function that runs a test. If the function returns successfully,
// the test succeeds; if the function panics then the test fails. We
// may need to come up with a more clever definition of test in order
// to support isolation of tests into threads.
pub enum TestFn {
    StaticTestFn(fn()),
    StaticBenchFn(fn(&mut Bencher)),
    DynTestFn(Box<dyn FnOnce() + Send>),
    DynBenchFn(Box<dyn TDynBenchFn + 'static>),
}

impl TestFn {
    fn padding(&self) -> NamePadding {
        match *self {
            StaticTestFn(..) => PadNone,
            StaticBenchFn(..) => PadOnRight,
            DynTestFn(..) => PadNone,
            DynBenchFn(..) => PadOnRight,
        }
    }
}

impl fmt::Debug for TestFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            StaticTestFn(..) => "StaticTestFn(..)",
            StaticBenchFn(..) => "StaticBenchFn(..)",
            DynTestFn(..) => "DynTestFn(..)",
            DynBenchFn(..) => "DynBenchFn(..)",
        })
    }
}

/// Manager of the benchmarking runs.
///
/// This is fed into functions marked with `#[bench]` to allow for
/// set-up & tear-down before running a piece of code repeatedly via a
/// call to `iter`.
#[derive(Clone)]
pub struct Bencher {
    mode: BenchMode,
    summary: Option<stats::Summary>,
    pub bytes: u64,
}

#[derive(Clone, PartialEq, Eq)]
pub enum BenchMode {
    Auto,
    Single,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShouldPanic {
    No,
    Yes,
    YesWithMessage(&'static str),
}

// The definition of a single test. A test runner will run a list of
// these.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TestDesc {
    pub name: TestName,
    pub ignore: bool,
    pub should_panic: ShouldPanic,
    pub allow_fail: bool,
}

#[derive(Debug)]
pub struct TestDescAndFn {
    pub desc: TestDesc,
    pub testfn: TestFn,
}

#[derive(Clone, PartialEq, Debug, Copy)]
pub struct Metric {
    value: f64,
    noise: f64,
}

impl Metric {
    pub fn new(value: f64, noise: f64) -> Metric {
        Metric { value, noise }
    }
}

/// In case we want to add other options as well, just add them in this struct.
#[derive(Copy, Clone, Debug)]
pub struct Options {
    display_output: bool,
}

impl Options {
    pub fn new() -> Options {
        Options {
            display_output: false,
        }
    }

    pub fn display_output(mut self, display_output: bool) -> Options {
        self.display_output = display_output;
        self
    }
}

// The default console test runner. It accepts the command line
// arguments and a vector of test_descs.
pub fn test_main(args: &[String], tests: Vec<TestDescAndFn>, options: Options) {
    let mut opts = match parse_opts(args) {
        Some(Ok(o)) => o,
        Some(Err(msg)) => {
            eprintln!("error: {}", msg);
            process::exit(101);
        }
        None => return,
    };

    opts.options = options;
    if opts.list {
        if let Err(e) = list_tests_console(&opts, tests) {
            eprintln!("error: io error when listing tests: {:?}", e);
            process::exit(101);
        }
    } else {
        match run_tests_console(&opts, tests) {
            Ok(true) => {}
            Ok(false) => process::exit(101),
            Err(e) => {
                eprintln!("error: io error when listing tests: {:?}", e);
                process::exit(101);
            }
        }
    }
}

// A variant optimized for invocation with a static test vector.
// This will panic (intentionally) when fed any dynamic tests, because
// it is copying the static values out into a dynamic vector and cannot
// copy dynamic values. It is doing this because from this point on
// a Vec<TestDescAndFn> is used in order to effect ownership-transfer
// semantics into parallel test runners, which in turn requires a Vec<>
// rather than a &[].
pub fn test_main_static(tests: &[&TestDescAndFn]) {
    let args = env::args().collect::<Vec<_>>();
    let owned_tests = tests
        .iter()
        .map(|t| match t.testfn {
            StaticTestFn(f) => TestDescAndFn {
                testfn: StaticTestFn(f),
                desc: t.desc.clone(),
            },
            StaticBenchFn(f) => TestDescAndFn {
                testfn: StaticBenchFn(f),
                desc: t.desc.clone(),
            },
            _ => panic!("non-static tests passed to test::test_main_static"),
        })
        .collect();
    test_main(&args, owned_tests, Options::new())
}

/// Invoked when unit tests terminate. Should panic if the unit
/// Tests is considered a failure. By default, invokes `report()`
/// and checks for a `0` result.
pub fn assert_test_result<T: Termination>(result: T) {
    let code = result.report();
    assert_eq!(
        code, 0,
        "the test returned a termination value with a non-zero status code ({}) \
         which indicates a failure",
        code
    );
}

#[derive(Copy, Clone, Debug)]
pub enum ColorConfig {
    AutoColor,
    AlwaysColor,
    NeverColor,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    Pretty,
    Terse,
    Json,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RunIgnored {
    Yes,
    No,
    Only,
}

#[derive(Debug)]
pub struct TestOpts {
    pub list: bool,
    pub filter: Option<String>,
    pub filter_exact: bool,
    pub exclude_should_panic: bool,
    pub run_ignored: RunIgnored,
    pub run_tests: bool,
    pub bench_benchmarks: bool,
    pub logfile: Option<PathBuf>,
    pub nocapture: bool,
    pub color: ColorConfig,
    pub format: OutputFormat,
    pub test_threads: Option<usize>,
    pub skip: Vec<String>,
    pub options: Options,
}

impl TestOpts {
    #[cfg(test)]
    fn new() -> TestOpts {
        TestOpts {
            list: false,
            filter: None,
            filter_exact: false,
            exclude_should_panic: false,
            run_ignored: RunIgnored::No,
            run_tests: false,
            bench_benchmarks: false,
            logfile: None,
            nocapture: false,
            color: AutoColor,
            format: OutputFormat::Pretty,
            test_threads: None,
            skip: vec![],
            options: Options::new(),
        }
    }
}

/// Result of parsing the options.
pub type OptRes = Result<TestOpts, String>;

fn optgroups() -> getopts::Options {
    let mut opts = getopts::Options::new();
    opts.optflag("", "include-ignored", "Run ignored and not ignored tests")
        .optflag("", "ignored", "Run only ignored tests")
        .optflag("", "exclude-should-panic", "Excludes tests marked as should_panic")
        .optflag("", "test", "Run tests and not benchmarks")
        .optflag("", "bench", "Run benchmarks instead of tests")
        .optflag("", "list", "List all tests and benchmarks")
        .optflag("h", "help", "Display this message (longer with --help)")
        .optopt(
            "",
            "logfile",
            "Write logs to the specified file instead \
             of stdout",
            "PATH",
        )
        .optflag(
            "",
            "nocapture",
            "don't capture stdout/stderr of each \
             task, allow printing directly",
        )
        .optopt(
            "",
            "test-threads",
            "Number of threads used for running tests \
             in parallel",
            "n_threads",
        )
        .optmulti(
            "",
            "skip",
            "Skip tests whose names contain FILTER (this flag can \
             be used multiple times)",
            "FILTER",
        )
        .optflag(
            "q",
            "quiet",
            "Display one character per test instead of one line. \
             Alias to --format=terse",
        )
        .optflag(
            "",
            "exact",
            "Exactly match filters rather than by substring",
        )
        .optopt(
            "",
            "color",
            "Configure coloring of output:
            auto   = colorize if stdout is a tty and tests are run on serially (default);
            always = always colorize output;
            never  = never colorize output;",
            "auto|always|never",
        )
        .optopt(
            "",
            "format",
            "Configure formatting of output:
            pretty = Print verbose output;
            terse  = Display one character per test;
            json   = Output a json document",
            "pretty|terse|json",
        )
        .optopt(
            "Z",
            "",
            "Enable nightly-only flags:
            unstable-options = Allow use of experimental features",
            "unstable-options",
        );
    return opts;
}

fn usage(binary: &str, options: &getopts::Options) {
    let message = format!("Usage: {} [OPTIONS] [FILTER]", binary);
    println!(
        r#"{usage}

The FILTER string is tested against the name of all tests, and only those
tests whose names contain the filter are run.

By default, all tests are run in parallel. This can be altered with the
--test-threads flag or the RUST_TEST_THREADS environment variable when running
tests (set it to 1).

All tests have their standard output and standard error captured by default.
This can be overridden with the --nocapture flag or setting RUST_TEST_NOCAPTURE
environment variable to a value other than "0". Logging is not captured by default.

Test Attributes:

    #[test]        - Indicates a function is a test to be run. This function
                     takes no arguments.
    #[bench]       - Indicates a function is a benchmark to be run. This
                     function takes one argument (test::Bencher).
    #[should_panic] - This function (also labeled with #[test]) will only pass if
                     the code causes a panic (an assertion failure or panic!)
                     A message may be provided, which the failure string must
                     contain: #[should_panic(expected = "foo")].
    #[ignore]      - When applied to a function which is already attributed as a
                     test, then the test runner will ignore these tests during
                     normal test runs. Running with --ignored or --include-ignored will run
                     these tests."#,
        usage = options.usage(&message)
    );
}

// FIXME: Copied from libsyntax until linkage errors are resolved. Issue #47566
fn is_nightly() -> bool {
    // Whether this is a feature-staged build, i.e., on the beta or stable channel
    let disable_unstable_features = option_env!("CFG_DISABLE_UNSTABLE_FEATURES").is_some();
    // Whether we should enable unstable features for bootstrapping
    let bootstrap = env::var("RUSTC_BOOTSTRAP").is_ok();

    bootstrap || !disable_unstable_features
}

// Parses command line arguments into test options
pub fn parse_opts(args: &[String]) -> Option<OptRes> {
    let mut allow_unstable = false;
    let opts = optgroups();
    let args = args.get(1..).unwrap_or(args);
    let matches = match opts.parse(args) {
        Ok(m) => m,
        Err(f) => return Some(Err(f.to_string())),
    };

    if let Some(opt) = matches.opt_str("Z") {
        if !is_nightly() {
            return Some(Err(
                "the option `Z` is only accepted on the nightly compiler".into(),
            ));
        }

        match &*opt {
            "unstable-options" => {
                allow_unstable = true;
            }
            _ => {
                return Some(Err("Unrecognized option to `Z`".into()));
            }
        }
    };

    if matches.opt_present("h") {
        usage(&args[0], &opts);
        return None;
    }

    let filter = if !matches.free.is_empty() {
        Some(matches.free[0].clone())
    } else {
        None
    };

    let exclude_should_panic = matches.opt_present("exclude-should-panic");
    if !allow_unstable && exclude_should_panic {
        return Some(Err(
            "The \"exclude-should-panic\" flag is only accepted on the nightly compiler".into(),
        ));
    }

    let include_ignored = matches.opt_present("include-ignored");
    if !allow_unstable && include_ignored {
        return Some(Err(
            "The \"include-ignored\" flag is only accepted on the nightly compiler".into(),
        ));
    }

    let run_ignored = match (include_ignored, matches.opt_present("ignored")) {
        (true, true) => {
            return Some(Err(
                "the options --include-ignored and --ignored are mutually exclusive".into(),
            ));
        }
        (true, false) => RunIgnored::Yes,
        (false, true) => RunIgnored::Only,
        (false, false) => RunIgnored::No,
    };
    let quiet = matches.opt_present("quiet");
    let exact = matches.opt_present("exact");
    let list = matches.opt_present("list");

    let logfile = matches.opt_str("logfile");
    let logfile = logfile.map(|s| PathBuf::from(&s));

    let bench_benchmarks = matches.opt_present("bench");
    let run_tests = !bench_benchmarks || matches.opt_present("test");

    let mut nocapture = matches.opt_present("nocapture");
    if !nocapture {
        nocapture = match env::var("RUST_TEST_NOCAPTURE") {
            Ok(val) => &val != "0",
            Err(_) => false,
        };
    }

    let test_threads = match matches.opt_str("test-threads") {
        Some(n_str) => match n_str.parse::<usize>() {
            Ok(0) => return Some(Err("argument for --test-threads must not be 0".to_string())),
            Ok(n) => Some(n),
            Err(e) => {
                return Some(Err(format!(
                    "argument for --test-threads must be a number > 0 \
                     (error: {})",
                    e
                )));
            }
        },
        None => None,
    };

    let color = match matches.opt_str("color").as_ref().map(|s| &**s) {
        Some("auto") | None => AutoColor,
        Some("always") => AlwaysColor,
        Some("never") => NeverColor,

        Some(v) => {
            return Some(Err(format!(
                "argument for --color must be auto, always, or never (was \
                 {})",
                v
            )));
        }
    };

    let format = match matches.opt_str("format").as_ref().map(|s| &**s) {
        None if quiet => OutputFormat::Terse,
        Some("pretty") | None => OutputFormat::Pretty,
        Some("terse") => OutputFormat::Terse,
        Some("json") => {
            if !allow_unstable {
                return Some(Err(
                    "The \"json\" format is only accepted on the nightly compiler".into(),
                ));
            }
            OutputFormat::Json
        }

        Some(v) => {
            return Some(Err(format!(
                "argument for --format must be pretty, terse, or json (was \
                 {})",
                v
            )));
        }
    };

    let test_opts = TestOpts {
        list,
        filter,
        filter_exact: exact,
        exclude_should_panic,
        run_ignored,
        run_tests,
        bench_benchmarks,
        logfile,
        nocapture,
        color,
        format,
        test_threads,
        skip: matches.opt_strs("skip"),
        options: Options::new(),
    };

    Some(Ok(test_opts))
}

#[derive(Clone, PartialEq)]
pub struct BenchSamples {
    ns_iter_summ: stats::Summary,
    mb_s: usize,
}

#[derive(Clone, PartialEq)]
pub enum TestResult {
    TrOk,
    TrFailed,
    TrFailedMsg(String),
    TrIgnored,
    TrAllowedFail,
    TrBench(BenchSamples),
}

unsafe impl Send for TestResult {}

enum OutputLocation<T> {
    Pretty(Box<term::StdoutTerminal>),
    Raw(T),
}

impl<T: Write> Write for OutputLocation<T> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match *self {
            Pretty(ref mut term) => term.write(buf),
            Raw(ref mut stdout) => stdout.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match *self {
            Pretty(ref mut term) => term.flush(),
            Raw(ref mut stdout) => stdout.flush(),
        }
    }
}

struct ConsoleTestState {
    log_out: Option<File>,
    total: usize,
    passed: usize,
    failed: usize,
    ignored: usize,
    allowed_fail: usize,
    filtered_out: usize,
    measured: usize,
    metrics: MetricMap,
    failures: Vec<(TestDesc, Vec<u8>)>,
    not_failures: Vec<(TestDesc, Vec<u8>)>,
    options: Options,
}

impl ConsoleTestState {
    pub fn new(opts: &TestOpts) -> io::Result<ConsoleTestState> {
        let log_out = match opts.logfile {
            Some(ref path) => Some(File::create(path)?),
            None => None,
        };

        Ok(ConsoleTestState {
            log_out,
            total: 0,
            passed: 0,
            failed: 0,
            ignored: 0,
            allowed_fail: 0,
            filtered_out: 0,
            measured: 0,
            metrics: MetricMap::new(),
            failures: Vec::new(),
            not_failures: Vec::new(),
            options: opts.options,
        })
    }

    pub fn write_log<S: AsRef<str>>(&mut self, msg: S) -> io::Result<()> {
        let msg = msg.as_ref();
        match self.log_out {
            None => Ok(()),
            Some(ref mut o) => o.write_all(msg.as_bytes()),
        }
    }

    pub fn write_log_result(&mut self, test: &TestDesc, result: &TestResult) -> io::Result<()> {
        self.write_log(format!(
            "{} {}\n",
            match *result {
                TrOk => "ok".to_owned(),
                TrFailed => "failed".to_owned(),
                TrFailedMsg(ref msg) => format!("failed: {}", msg),
                TrIgnored => "ignored".to_owned(),
                TrAllowedFail => "failed (allowed)".to_owned(),
                TrBench(ref bs) => fmt_bench_samples(bs),
            },
            test.name
        ))
    }

    fn current_test_count(&self) -> usize {
        self.passed + self.failed + self.ignored + self.measured + self.allowed_fail
    }
}

// Format a number with thousands separators
fn fmt_thousands_sep(mut n: usize, sep: char) -> String {
    use std::fmt::Write;
    let mut output = String::new();
    let mut trailing = false;
    for &pow in &[9, 6, 3, 0] {
        let base = 10_usize.pow(pow);
        if pow == 0 || trailing || n / base != 0 {
            if !trailing {
                output.write_fmt(format_args!("{}", n / base)).unwrap();
            } else {
                output.write_fmt(format_args!("{:03}", n / base)).unwrap();
            }
            if pow != 0 {
                output.push(sep);
            }
            trailing = true;
        }
        n %= base;
    }

    output
}

pub fn fmt_bench_samples(bs: &BenchSamples) -> String {
    use std::fmt::Write;
    let mut output = String::new();

    let median = bs.ns_iter_summ.median as usize;
    let deviation = (bs.ns_iter_summ.max - bs.ns_iter_summ.min) as usize;

    output
        .write_fmt(format_args!(
            "{:>11} ns/iter (+/- {})",
            fmt_thousands_sep(median, ','),
            fmt_thousands_sep(deviation, ',')
        ))
        .unwrap();
    if bs.mb_s != 0 {
        output
            .write_fmt(format_args!(" = {} MB/s", bs.mb_s))
            .unwrap();
    }
    output
}

// List the tests to console, and optionally to logfile. Filters are honored.
pub fn list_tests_console(opts: &TestOpts, tests: Vec<TestDescAndFn>) -> io::Result<()> {
    let mut output = match term::stdout() {
        None => Raw(io::stdout()),
        Some(t) => Pretty(t),
    };

    let quiet = opts.format == OutputFormat::Terse;
    let mut st = ConsoleTestState::new(opts)?;

    let mut ntest = 0;
    let mut nbench = 0;

    for test in filter_tests(&opts, tests) {
        use crate::TestFn::*;

        let TestDescAndFn {
            desc: TestDesc { name, .. },
            testfn,
        } = test;

        let fntype = match testfn {
            StaticTestFn(..) | DynTestFn(..) => {
                ntest += 1;
                "test"
            }
            StaticBenchFn(..) | DynBenchFn(..) => {
                nbench += 1;
                "benchmark"
            }
        };

        writeln!(output, "{}: {}", name, fntype)?;
        st.write_log(format!("{} {}\n", fntype, name))?;
    }

    fn plural(count: u32, s: &str) -> String {
        match count {
            1 => format!("{} {}", 1, s),
            n => format!("{} {}s", n, s),
        }
    }

    if !quiet {
        if ntest != 0 || nbench != 0 {
            writeln!(output, "")?;
        }

        writeln!(
            output,
            "{}, {}",
            plural(ntest, "test"),
            plural(nbench, "benchmark")
        )?;
    }

    Ok(())
}

// A simple console test runner
pub fn run_tests_console(opts: &TestOpts, tests: Vec<TestDescAndFn>) -> io::Result<bool> {
    fn callback(
        event: &TestEvent,
        st: &mut ConsoleTestState,
        out: &mut dyn OutputFormatter,
    ) -> io::Result<()> {
        match (*event).clone() {
            TeFiltered(ref filtered_tests) => {
                st.total = filtered_tests.len();
                out.write_run_start(filtered_tests.len())
            }
            TeFilteredOut(filtered_out) => Ok(st.filtered_out = filtered_out),
            TeWait(ref test) => out.write_test_start(test),
            TeTimeout(ref test) => out.write_timeout(test),
            TeResult(test, result, stdout) => {
                st.write_log_result(&test, &result)?;
                out.write_result(&test, &result, &*stdout)?;
                match result {
                    TrOk => {
                        st.passed += 1;
                        st.not_failures.push((test, stdout));
                    }
                    TrIgnored => st.ignored += 1,
                    TrAllowedFail => st.allowed_fail += 1,
                    TrBench(bs) => {
                        st.metrics.insert_metric(
                            test.name.as_slice(),
                            bs.ns_iter_summ.median,
                            bs.ns_iter_summ.max - bs.ns_iter_summ.min,
                        );
                        st.measured += 1
                    }
                    TrFailed => {
                        st.failed += 1;
                        st.failures.push((test, stdout));
                    }
                    TrFailedMsg(msg) => {
                        st.failed += 1;
                        let mut stdout = stdout;
                        stdout.extend_from_slice(format!("note: {}", msg).as_bytes());
                        st.failures.push((test, stdout));
                    }
                }
                Ok(())
            }
        }
    }

    let output = match term::stdout() {
        None => Raw(io::stdout()),
        Some(t) => Pretty(t),
    };

    let max_name_len = tests
        .iter()
        .max_by_key(|t| len_if_padded(*t))
        .map(|t| t.desc.name.as_slice().len())
        .unwrap_or(0);

    let is_multithreaded = opts.test_threads.unwrap_or_else(get_concurrency) > 1;

    let mut out: Box<dyn OutputFormatter> = match opts.format {
        OutputFormat::Pretty => Box::new(PrettyFormatter::new(
            output,
            use_color(opts),
            max_name_len,
            is_multithreaded,
        )),
        OutputFormat::Terse => Box::new(TerseFormatter::new(
            output,
            use_color(opts),
            max_name_len,
            is_multithreaded,
        )),
        OutputFormat::Json => Box::new(JsonFormatter::new(output)),
    };
    let mut st = ConsoleTestState::new(opts)?;
    fn len_if_padded(t: &TestDescAndFn) -> usize {
        match t.testfn.padding() {
            PadNone => 0,
            PadOnRight => t.desc.name.as_slice().len(),
        }
    }

    run_tests(opts, tests, |x| callback(&x, &mut st, &mut *out))?;

    assert!(st.current_test_count() == st.total);

    return out.write_run_finish(&st);
}

#[test]
fn should_sort_failures_before_printing_them() {
    let test_a = TestDesc {
        name: StaticTestName("a"),
        ignore: false,
        should_panic: ShouldPanic::No,
        allow_fail: false,
    };

    let test_b = TestDesc {
        name: StaticTestName("b"),
        ignore: false,
        should_panic: ShouldPanic::No,
        allow_fail: false,
    };

    let mut out = PrettyFormatter::new(Raw(Vec::new()), false, 10, false);

    let st = ConsoleTestState {
        log_out: None,
        total: 0,
        passed: 0,
        failed: 0,
        ignored: 0,
        allowed_fail: 0,
        filtered_out: 0,
        measured: 0,
        metrics: MetricMap::new(),
        failures: vec![(test_b, Vec::new()), (test_a, Vec::new())],
        options: Options::new(),
        not_failures: Vec::new(),
    };

    out.write_failures(&st).unwrap();
    let s = match out.output_location() {
        &Raw(ref m) => String::from_utf8_lossy(&m[..]),
        &Pretty(_) => unreachable!(),
    };

    let apos = s.find("a").unwrap();
    let bpos = s.find("b").unwrap();
    assert!(apos < bpos);
}

fn use_color(opts: &TestOpts) -> bool {
    match opts.color {
        AutoColor => !opts.nocapture && stdout_isatty(),
        AlwaysColor => true,
        NeverColor => false,
    }
}

#[cfg(any(
    target_os = "cloudabi",
    target_os = "redox",
    all(target_arch = "wasm32", not(target_os = "emscripten")),
    all(target_vendor = "fortanix", target_env = "sgx")
))]
fn stdout_isatty() -> bool {
    // FIXME: Implement isatty on Redox and SGX
    false
}
#[cfg(unix)]
fn stdout_isatty() -> bool {
    unsafe { libc::isatty(libc::STDOUT_FILENO) != 0 }
}
#[cfg(windows)]
fn stdout_isatty() -> bool {
    type DWORD = u32;
    type BOOL = i32;
    type HANDLE = *mut u8;
    type LPDWORD = *mut u32;
    const STD_OUTPUT_HANDLE: DWORD = -11i32 as DWORD;
    extern "system" {
        fn GetStdHandle(which: DWORD) -> HANDLE;
        fn GetConsoleMode(hConsoleHandle: HANDLE, lpMode: LPDWORD) -> BOOL;
    }
    unsafe {
        let handle = GetStdHandle(STD_OUTPUT_HANDLE);
        let mut out = 0;
        GetConsoleMode(handle, &mut out) != 0
    }
}

#[derive(Clone)]
pub enum TestEvent {
    TeFiltered(Vec<TestDesc>),
    TeWait(TestDesc),
    TeResult(TestDesc, TestResult, Vec<u8>),
    TeTimeout(TestDesc),
    TeFilteredOut(usize),
}

pub type MonitorMsg = (TestDesc, TestResult, Vec<u8>);

struct Sink(Arc<Mutex<Vec<u8>>>);
impl Write for Sink {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        Write::write(&mut *self.0.lock().unwrap(), data)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub fn run_tests<F>(opts: &TestOpts, tests: Vec<TestDescAndFn>, mut callback: F) -> io::Result<()>
where
    F: FnMut(TestEvent) -> io::Result<()>,
{
    use std::collections::{self, HashMap};
    use std::hash::BuildHasherDefault;
    use std::sync::mpsc::RecvTimeoutError;
    // Use a deterministic hasher
    type TestMap =
        HashMap<TestDesc, Instant, BuildHasherDefault<collections::hash_map::DefaultHasher>>;

    let tests_len = tests.len();

    let mut filtered_tests = filter_tests(opts, tests);
    if !opts.bench_benchmarks {
        filtered_tests = convert_benchmarks_to_tests(filtered_tests);
    }

    let filtered_tests = {
        let mut filtered_tests = filtered_tests;
        for test in filtered_tests.iter_mut() {
            test.desc.name = test.desc.name.with_padding(test.testfn.padding());
        }

        filtered_tests
    };

    let filtered_out = tests_len - filtered_tests.len();
    callback(TeFilteredOut(filtered_out))?;

    let filtered_descs = filtered_tests.iter().map(|t| t.desc.clone()).collect();

    callback(TeFiltered(filtered_descs))?;

    let (filtered_tests, filtered_benchs): (Vec<_>, _) =
        filtered_tests.into_iter().partition(|e| match e.testfn {
            StaticTestFn(_) | DynTestFn(_) => true,
            _ => false,
        });

    let concurrency = opts.test_threads.unwrap_or_else(get_concurrency);

    let mut remaining = filtered_tests;
    remaining.reverse();
    let mut pending = 0;

    let (tx, rx) = channel::<MonitorMsg>();

    let mut running_tests: TestMap = HashMap::default();

    fn get_timed_out_tests(running_tests: &mut TestMap) -> Vec<TestDesc> {
        let now = Instant::now();
        let timed_out = running_tests
            .iter()
            .filter_map(|(desc, timeout)| {
                if &now >= timeout {
                    Some(desc.clone())
                } else {
                    None
                }
            })
            .collect();
        for test in &timed_out {
            running_tests.remove(test);
        }
        timed_out
    };

    fn calc_timeout(running_tests: &TestMap) -> Option<Duration> {
        running_tests.values().min().map(|next_timeout| {
            let now = Instant::now();
            if *next_timeout >= now {
                *next_timeout - now
            } else {
                Duration::new(0, 0)
            }
        })
    };

    if concurrency == 1 {
        while !remaining.is_empty() {
            let test = remaining.pop().unwrap();
            callback(TeWait(test.desc.clone()))?;
            run_test(opts, !opts.run_tests, test, tx.clone(), Concurrent::No);
            let (test, result, stdout) = rx.recv().unwrap();
            callback(TeResult(test, result, stdout))?;
        }
    } else {
        while pending > 0 || !remaining.is_empty() {
            while pending < concurrency && !remaining.is_empty() {
                let test = remaining.pop().unwrap();
                let timeout = Instant::now() + Duration::from_secs(TEST_WARN_TIMEOUT_S);
                running_tests.insert(test.desc.clone(), timeout);
                callback(TeWait(test.desc.clone()))?; //here no pad
                run_test(opts, !opts.run_tests, test, tx.clone(), Concurrent::Yes);
                pending += 1;
            }

            let mut res;
            loop {
                if let Some(timeout) = calc_timeout(&running_tests) {
                    res = rx.recv_timeout(timeout);
                    for test in get_timed_out_tests(&mut running_tests) {
                        callback(TeTimeout(test))?;
                    }
                    if res != Err(RecvTimeoutError::Timeout) {
                        break;
                    }
                } else {
                    res = rx.recv().map_err(|_| RecvTimeoutError::Disconnected);
                    break;
                }
            }

            let (desc, result, stdout) = res.unwrap();
            running_tests.remove(&desc);

            callback(TeResult(desc, result, stdout))?;
            pending -= 1;
        }
    }

    if opts.bench_benchmarks {
        // All benchmarks run at the end, in serial.
        for b in filtered_benchs {
            callback(TeWait(b.desc.clone()))?;
            run_test(opts, false, b, tx.clone(), Concurrent::No);
            let (test, result, stdout) = rx.recv().unwrap();
            callback(TeResult(test, result, stdout))?;
        }
    }
    Ok(())
}

#[allow(deprecated)]
fn get_concurrency() -> usize {
    return match env::var("RUST_TEST_THREADS") {
        Ok(s) => {
            let opt_n: Option<usize> = s.parse().ok();
            match opt_n {
                Some(n) if n > 0 => n,
                _ => panic!(
                    "RUST_TEST_THREADS is `{}`, should be a positive integer.",
                    s
                ),
            }
        }
        Err(..) => num_cpus(),
    };

    #[cfg(windows)]
    #[allow(nonstandard_style)]
    fn num_cpus() -> usize {
        #[repr(C)]
        struct SYSTEM_INFO {
            wProcessorArchitecture: u16,
            wReserved: u16,
            dwPageSize: u32,
            lpMinimumApplicationAddress: *mut u8,
            lpMaximumApplicationAddress: *mut u8,
            dwActiveProcessorMask: *mut u8,
            dwNumberOfProcessors: u32,
            dwProcessorType: u32,
            dwAllocationGranularity: u32,
            wProcessorLevel: u16,
            wProcessorRevision: u16,
        }
        extern "system" {
            fn GetSystemInfo(info: *mut SYSTEM_INFO) -> i32;
        }
        unsafe {
            let mut sysinfo = std::mem::zeroed();
            GetSystemInfo(&mut sysinfo);
            sysinfo.dwNumberOfProcessors as usize
        }
    }

    #[cfg(target_os = "redox")]
    fn num_cpus() -> usize {
        // FIXME: Implement num_cpus on Redox
        1
    }

    #[cfg(any(
        all(target_arch = "wasm32", not(target_os = "emscripten")),
        all(target_vendor = "fortanix", target_env = "sgx")
    ))]
    fn num_cpus() -> usize {
        1
    }

    #[cfg(any(
        target_os = "android",
        target_os = "cloudabi",
        target_os = "emscripten",
        target_os = "fuchsia",
        target_os = "ios",
        target_os = "linux",
        target_os = "macos",
        target_os = "solaris"
    ))]
    fn num_cpus() -> usize {
        unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) as usize }
    }

    #[cfg(any(
        target_os = "freebsd",
        target_os = "dragonfly",
        target_os = "netbsd"
    ))]
    fn num_cpus() -> usize {
        use std::ptr;

        let mut cpus: libc::c_uint = 0;
        let mut cpus_size = std::mem::size_of_val(&cpus);

        unsafe {
            cpus = libc::sysconf(libc::_SC_NPROCESSORS_ONLN) as libc::c_uint;
        }
        if cpus < 1 {
            let mut mib = [libc::CTL_HW, libc::HW_NCPU, 0, 0];
            unsafe {
                libc::sysctl(
                    mib.as_mut_ptr(),
                    2,
                    &mut cpus as *mut _ as *mut _,
                    &mut cpus_size as *mut _ as *mut _,
                    ptr::null_mut(),
                    0,
                );
            }
            if cpus < 1 {
                cpus = 1;
            }
        }
        cpus as usize
    }

    #[cfg(target_os = "openbsd")]
    fn num_cpus() -> usize {
        use std::ptr;

        let mut cpus: libc::c_uint = 0;
        let mut cpus_size = std::mem::size_of_val(&cpus);
        let mut mib = [libc::CTL_HW, libc::HW_NCPU, 0, 0];

        unsafe {
            libc::sysctl(
                mib.as_mut_ptr(),
                2,
                &mut cpus as *mut _ as *mut _,
                &mut cpus_size as *mut _ as *mut _,
                ptr::null_mut(),
                0,
            );
        }
        if cpus < 1 {
            cpus = 1;
        }
        cpus as usize
    }

    #[cfg(target_os = "haiku")]
    fn num_cpus() -> usize {
        // FIXME: implement
        1
    }

    #[cfg(target_os = "l4re")]
    fn num_cpus() -> usize {
        // FIXME: implement
        1
    }
}

pub fn filter_tests(opts: &TestOpts, tests: Vec<TestDescAndFn>) -> Vec<TestDescAndFn> {
    let mut filtered = tests;
    let matches_filter = |test: &TestDescAndFn, filter: &str| {
        let test_name = test.desc.name.as_slice();

        match opts.filter_exact {
            true => test_name == filter,
            false => test_name.contains(filter),
        }
    };

    // Remove tests that don't match the test filter
    if let Some(ref filter) = opts.filter {
        filtered.retain(|test| matches_filter(test, filter));
    }

    // Skip tests that match any of the skip filters
    filtered.retain(|test| !opts.skip.iter().any(|sf| matches_filter(test, sf)));

    // Excludes #[should_panic] tests
    if opts.exclude_should_panic {
        filtered.retain(|test| test.desc.should_panic == ShouldPanic::No);
    }

    // maybe unignore tests
    match opts.run_ignored {
        RunIgnored::Yes => {
            filtered
                .iter_mut()
                .for_each(|test| test.desc.ignore = false);
        }
        RunIgnored::Only => {
            filtered.retain(|test| test.desc.ignore);
            filtered
                .iter_mut()
                .for_each(|test| test.desc.ignore = false);
        }
        RunIgnored::No => {}
    }

    // Sort the tests alphabetically
    filtered.sort_by(|t1, t2| t1.desc.name.as_slice().cmp(t2.desc.name.as_slice()));

    filtered
}

pub fn convert_benchmarks_to_tests(tests: Vec<TestDescAndFn>) -> Vec<TestDescAndFn> {
    // convert benchmarks to tests, if we're not benchmarking them
    tests
        .into_iter()
        .map(|x| {
            let testfn = match x.testfn {
                DynBenchFn(bench) => DynTestFn(Box::new(move || {
                    bench::run_once(|b| __rust_begin_short_backtrace(|| bench.run(b)))
                })),
                StaticBenchFn(benchfn) => DynTestFn(Box::new(move || {
                    bench::run_once(|b| __rust_begin_short_backtrace(|| benchfn(b)))
                })),
                f => f,
            };
            TestDescAndFn {
                desc: x.desc,
                testfn,
            }
        })
        .collect()
}

pub fn run_test(
    opts: &TestOpts,
    force_ignore: bool,
    test: TestDescAndFn,
    monitor_ch: Sender<MonitorMsg>,
    concurrency: Concurrent,
) {
    let TestDescAndFn { desc, testfn } = test;

    let ignore_because_panic_abort = cfg!(target_arch = "wasm32")
        && !cfg!(target_os = "emscripten")
        && desc.should_panic != ShouldPanic::No;

    if force_ignore || desc.ignore || ignore_because_panic_abort {
        monitor_ch.send((desc, TrIgnored, Vec::new())).unwrap();
        return;
    }

    fn run_test_inner(
        desc: TestDesc,
        monitor_ch: Sender<MonitorMsg>,
        nocapture: bool,
        testfn: Box<dyn FnOnce() + Send>,
        concurrency: Concurrent,
    ) {
        // Buffer for capturing standard I/O
        let data = Arc::new(Mutex::new(Vec::new()));
        let data2 = data.clone();

        let name = desc.name.clone();
        let runtest = move || {
            let oldio = if !nocapture {
                Some((
                    io::set_print(Some(Box::new(Sink(data2.clone())))),
                    io::set_panic(Some(Box::new(Sink(data2)))),
                ))
            } else {
                None
            };

            let result = catch_unwind(AssertUnwindSafe(testfn));

            if let Some((printio, panicio)) = oldio {
                io::set_print(printio);
                io::set_panic(panicio);
            };

            let test_result = calc_result(&desc, result);
            let stdout = data.lock().unwrap().to_vec();
            monitor_ch
                .send((desc.clone(), test_result, stdout))
                .unwrap();
        };

        // If the platform is single-threaded we're just going to run
        // the test synchronously, regardless of the concurrency
        // level.
        let supports_threads = !cfg!(target_os = "emscripten") && !cfg!(target_arch = "wasm32");
        if concurrency == Concurrent::Yes && supports_threads {
            let cfg = thread::Builder::new().name(name.as_slice().to_owned());
            cfg.spawn(runtest).unwrap();
        } else {
            runtest();
        }
    }

    match testfn {
        DynBenchFn(bencher) => {
            crate::bench::benchmark(desc, monitor_ch, opts.nocapture, |harness| {
                bencher.run(harness)
            });
        }
        StaticBenchFn(benchfn) => {
            crate::bench::benchmark(desc, monitor_ch, opts.nocapture, |harness| {
                (benchfn.clone())(harness)
            });
        }
        DynTestFn(f) => {
            let cb = move || __rust_begin_short_backtrace(f);
            run_test_inner(desc, monitor_ch, opts.nocapture, Box::new(cb), concurrency)
        }
        StaticTestFn(f) => run_test_inner(
            desc,
            monitor_ch,
            opts.nocapture,
            Box::new(move || __rust_begin_short_backtrace(f)),
            concurrency,
        ),
    }
}

/// Fixed frame used to clean the backtrace with `RUST_BACKTRACE=1`.
#[inline(never)]
fn __rust_begin_short_backtrace<F: FnOnce()>(f: F) {
    f()
}

fn calc_result(desc: &TestDesc, task_result: Result<(), Box<dyn Any + Send>>) -> TestResult {
    match (&desc.should_panic, task_result) {
        (&ShouldPanic::No, Ok(())) | (&ShouldPanic::Yes, Err(_)) => TrOk,
        (&ShouldPanic::YesWithMessage(msg), Err(ref err)) => {
            if err
                .downcast_ref::<String>()
                .map(|e| &**e)
                .or_else(|| err.downcast_ref::<&'static str>().map(|e| *e))
                .map(|e| e.contains(msg))
                .unwrap_or(false)
            {
                TrOk
            } else {
                if desc.allow_fail {
                    TrAllowedFail
                } else {
                    TrFailedMsg(format!("panic did not include expected string '{}'", msg))
                }
            }
        }
        _ if desc.allow_fail => TrAllowedFail,
        _ => TrFailed,
    }
}

#[derive(Clone, PartialEq)]
pub struct MetricMap(BTreeMap<String, Metric>);

impl MetricMap {
    pub fn new() -> MetricMap {
        MetricMap(BTreeMap::new())
    }

    /// Insert a named `value` (+/- `noise`) metric into the map. The value
    /// must be non-negative. The `noise` indicates the uncertainty of the
    /// metric, which doubles as the "noise range" of acceptable
    /// pairwise-regressions on this named value, when comparing from one
    /// metric to the next using `compare_to_old`.
    ///
    /// If `noise` is positive, then it means this metric is of a value
    /// you want to see grow smaller, so a change larger than `noise` in the
    /// positive direction represents a regression.
    ///
    /// If `noise` is negative, then it means this metric is of a value
    /// you want to see grow larger, so a change larger than `noise` in the
    /// negative direction represents a regression.
    pub fn insert_metric(&mut self, name: &str, value: f64, noise: f64) {
        let m = Metric { value, noise };
        self.0.insert(name.to_owned(), m);
    }

    pub fn fmt_metrics(&self) -> String {
        let v = self
            .0
            .iter()
            .map(|(k, v)| format!("{}: {} (+/- {})", *k, v.value, v.noise))
            .collect::<Vec<_>>();
        v.join(", ")
    }
}

// Benchmarking

pub use std::hint::black_box;

impl Bencher {
    /// Callback for benchmark functions to run in their body.
    pub fn iter<T, F>(&mut self, mut inner: F)
    where
        F: FnMut() -> T,
    {
        if self.mode == BenchMode::Single {
            ns_iter_inner(&mut inner, 1);
            return;
        }

        self.summary = Some(iter(&mut inner));
    }

    pub fn bench<F>(&mut self, mut f: F) -> Option<stats::Summary>
    where
        F: FnMut(&mut Bencher),
    {
        f(self);
        return self.summary;
    }
}

fn ns_from_dur(dur: Duration) -> u64 {
    dur.as_secs() * 1_000_000_000 + (dur.subsec_nanos() as u64)
}

fn ns_iter_inner<T, F>(inner: &mut F, k: u64) -> u64
where
    F: FnMut() -> T,
{
    let start = Instant::now();
    for _ in 0..k {
        black_box(inner());
    }
    return ns_from_dur(start.elapsed());
}

pub fn iter<T, F>(inner: &mut F) -> stats::Summary
where
    F: FnMut() -> T,
{
    // Initial bench run to get ballpark figure.
    let ns_single = ns_iter_inner(inner, 1);

    // Try to estimate iter count for 1ms falling back to 1m
    // iterations if first run took < 1ns.
    let ns_target_total = 1_000_000; // 1ms
    let mut n = ns_target_total / cmp::max(1, ns_single);

    // if the first run took more than 1ms we don't want to just
    // be left doing 0 iterations on every loop. The unfortunate
    // side effect of not being able to do as many runs is
    // automatically handled by the statistical analysis below
    // (i.e., larger error bars).
    n = cmp::max(1, n);

    let mut total_run = Duration::new(0, 0);
    let samples: &mut [f64] = &mut [0.0_f64; 50];
    loop {
        let loop_start = Instant::now();

        for p in &mut *samples {
            *p = ns_iter_inner(inner, n) as f64 / n as f64;
        }

        stats::winsorize(samples, 5.0);
        let summ = stats::Summary::new(samples);

        for p in &mut *samples {
            let ns = ns_iter_inner(inner, 5 * n);
            *p = ns as f64 / (5 * n) as f64;
        }

        stats::winsorize(samples, 5.0);
        let summ5 = stats::Summary::new(samples);

        let loop_run = loop_start.elapsed();

        // If we've run for 100ms and seem to have converged to a
        // stable median.
        if loop_run > Duration::from_millis(100)
            && summ.median_abs_dev_pct < 1.0
            && summ.median - summ5.median < summ5.median_abs_dev
        {
            return summ5;
        }

        total_run = total_run + loop_run;
        // Longest we ever run for is 3s.
        if total_run > Duration::from_secs(3) {
            return summ5;
        }

        // If we overflow here just return the results so far. We check a
        // multiplier of 10 because we're about to multiply by 2 and the
        // next iteration of the loop will also multiply by 5 (to calculate
        // the summ5 result)
        n = match n.checked_mul(10) {
            Some(_) => n * 2,
            None => {
                return summ5;
            }
        };
    }
}

pub mod bench {
    use super::{BenchMode, BenchSamples, Bencher, MonitorMsg, Sender, Sink, TestDesc, TestResult};
    use crate::stats;
    use std::cmp;
    use std::io;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::{Arc, Mutex};

    pub fn benchmark<F>(desc: TestDesc, monitor_ch: Sender<MonitorMsg>, nocapture: bool, f: F)
    where
        F: FnMut(&mut Bencher),
    {
        let mut bs = Bencher {
            mode: BenchMode::Auto,
            summary: None,
            bytes: 0,
        };

        let data = Arc::new(Mutex::new(Vec::new()));
        let data2 = data.clone();

        let oldio = if !nocapture {
            Some((
                io::set_print(Some(Box::new(Sink(data2.clone())))),
                io::set_panic(Some(Box::new(Sink(data2)))),
            ))
        } else {
            None
        };

        let result = catch_unwind(AssertUnwindSafe(|| bs.bench(f)));

        if let Some((printio, panicio)) = oldio {
            io::set_print(printio);
            io::set_panic(panicio);
        };

        let test_result = match result {
            //bs.bench(f) {
            Ok(Some(ns_iter_summ)) => {
                let ns_iter = cmp::max(ns_iter_summ.median as u64, 1);
                let mb_s = bs.bytes * 1000 / ns_iter;

                let bs = BenchSamples {
                    ns_iter_summ,
                    mb_s: mb_s as usize,
                };
                TestResult::TrBench(bs)
            }
            Ok(None) => {
                // iter not called, so no data.
                // FIXME: error in this case?
                let samples: &mut [f64] = &mut [0.0_f64; 1];
                let bs = BenchSamples {
                    ns_iter_summ: stats::Summary::new(samples),
                    mb_s: 0,
                };
                TestResult::TrBench(bs)
            }
            Err(_) => TestResult::TrFailed,
        };

        let stdout = data.lock().unwrap().to_vec();
        monitor_ch.send((desc, test_result, stdout)).unwrap();
    }

    pub fn run_once<F>(f: F)
    where
        F: FnMut(&mut Bencher),
    {
        let mut bs = Bencher {
            mode: BenchMode::Single,
            summary: None,
            bytes: 0,
        };
        bs.bench(f);
    }
}

#[cfg(test)]
mod tests;
