// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Support code for rustc's built in unit-test and micro-benchmarking
//! framework.
//!
//! Almost all user code will only be interested in `Bencher` and
//! `black_box`. All other interactions (such as writing tests and
//! benchmarks themselves) should be done via the `#[test]` and
//! `#[bench]` attributes.
//!
//! See the [Testing Chapter](../book/first-edition/testing.html) of the book for more details.

// Currently, not much of this is meant for users. It is intended to
// support the simplest interface possible for representing and
// running tests while providing a base that other test frameworks may
// build off of.

// NB: this is also specified in this crate's Cargo.toml, but libsyntax contains logic specific to
// this crate, which relies on this attribute (rather than the value of `--crate-name` passed by
// cargo) to detect this crate.
#![crate_name = "test"]
#![unstable(feature = "test", issue = "27812")]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       test(attr(deny(warnings))))]
#![deny(warnings)]

#![feature(asm)]
#![feature(fnbox)]
#![cfg_attr(any(unix, target_os = "cloudabi"), feature(libc))]
#![feature(set_stdio)]
#![feature(panic_unwind)]
#![feature(staged_api)]

extern crate getopts;
extern crate term;
#[cfg(any(unix, target_os = "cloudabi"))]
extern crate libc;
extern crate panic_unwind;

pub use self::TestFn::*;
pub use self::ColorConfig::*;
pub use self::TestResult::*;
pub use self::TestName::*;
use self::TestEvent::*;
use self::NamePadding::*;
use self::OutputLocation::*;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::any::Any;
use std::boxed::FnBox;
use std::cmp;
use std::collections::BTreeMap;
use std::env;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use std::io;
use std::iter::repeat;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Instant, Duration};

const TEST_WARN_TIMEOUT_S: u64 = 60;
const QUIET_MODE_MAX_COLUMN: usize = 100; // insert a '\n' after 100 tests in quiet mode

// to be used by rustc to compile tests in libtest
pub mod test {
    pub use {Bencher, TestName, TestResult, TestDesc, TestDescAndFn, TestOpts, TrFailed,
             TrFailedMsg, TrIgnored, TrOk, Metric, MetricMap, StaticTestFn, StaticTestName,
             DynTestName, DynTestFn, run_test, test_main, test_main_static, filter_tests,
             parse_opts, StaticBenchFn, ShouldPanic, Options};
}

pub mod stats;

// The name of a test. By convention this follows the rules for rust
// paths; i.e. it should be a series of identifiers separated by double
// colons. This way if some test runner wants to arrange the tests
// hierarchically it may.

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TestName {
    StaticTestName(&'static str),
    DynTestName(String),
}
impl TestName {
    fn as_slice(&self) -> &str {
        match *self {
            StaticTestName(s) => s,
            DynTestName(ref s) => s,
        }
    }
}
impl fmt::Display for TestName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self.as_slice(), f)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NamePadding {
    PadNone,
    PadOnRight,
}

impl TestDesc {
    fn padded_name(&self, column_count: usize, align: NamePadding) -> String {
        let mut name = String::from(self.name.as_slice());
        let fill = column_count.saturating_sub(name.len());
        let pad = repeat(" ").take(fill).collect::<String>();
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
    DynTestFn(Box<FnBox() + Send>),
    DynBenchFn(Box<TDynBenchFn + 'static>),
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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
        Metric {
            value,
            noise,
        }
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
        Some(Err(msg)) => panic!("{:?}", msg),
        None => return,
    };
    opts.options = options;
    if opts.list {
        if let Err(e) = list_tests_console(&opts, tests) {
            panic!("io error when listing tests: {:?}", e);
        }
    } else {
        match run_tests_console(&opts, tests) {
            Ok(true) => {}
            Ok(false) => std::process::exit(101),
            Err(e) => panic!("io error when running tests: {:?}", e),
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
pub fn test_main_static(tests: &[TestDescAndFn]) {
    let args = env::args().collect::<Vec<_>>();
    let owned_tests = tests.iter()
                           .map(|t| {
                               match t.testfn {
                                   StaticTestFn(f) => {
                                       TestDescAndFn {
                                           testfn: StaticTestFn(f),
                                           desc: t.desc.clone(),
                                       }
                                   }
                                   StaticBenchFn(f) => {
                                       TestDescAndFn {
                                           testfn: StaticBenchFn(f),
                                           desc: t.desc.clone(),
                                       }
                                   }
                                   _ => panic!("non-static tests passed to test::test_main_static"),
                               }
                           })
                           .collect();
    test_main(&args, owned_tests, Options::new())
}

#[derive(Copy, Clone, Debug)]
pub enum ColorConfig {
    AutoColor,
    AlwaysColor,
    NeverColor,
}

#[derive(Debug)]
pub struct TestOpts {
    pub list: bool,
    pub filter: Option<String>,
    pub filter_exact: bool,
    pub run_ignored: bool,
    pub run_tests: bool,
    pub bench_benchmarks: bool,
    pub logfile: Option<PathBuf>,
    pub nocapture: bool,
    pub color: ColorConfig,
    pub quiet: bool,
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
            run_ignored: false,
            run_tests: false,
            bench_benchmarks: false,
            logfile: None,
            nocapture: false,
            color: AutoColor,
            quiet: false,
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
    opts.optflag("", "ignored", "Run ignored tests")
        .optflag("", "test", "Run tests and not benchmarks")
        .optflag("", "bench", "Run benchmarks instead of tests")
        .optflag("", "list", "List all tests and benchmarks")
        .optflag("h", "help", "Display this message (longer with --help)")
        .optopt("", "logfile", "Write logs to the specified file instead \
                                of stdout", "PATH")
        .optflag("", "nocapture", "don't capture stdout/stderr of each \
                                   task, allow printing directly")
        .optopt("", "test-threads", "Number of threads used for running tests \
                                     in parallel", "n_threads")
        .optmulti("", "skip", "Skip tests whose names contain FILTER (this flag can \
                               be used multiple times)","FILTER")
        .optflag("q", "quiet", "Display one character per test instead of one line")
        .optflag("", "exact", "Exactly match filters rather than by substring")
        .optopt("", "color", "Configure coloring of output:
            auto   = colorize if stdout is a tty and tests are run on serially (default);
            always = always colorize output;
            never  = never colorize output;", "auto|always|never");
    return opts
}

fn usage(binary: &str, options: &getopts::Options) {
    let message = format!("Usage: {} [OPTIONS] [FILTER]", binary);
    println!(r#"{usage}

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
                     normal test runs. Running with --ignored will run these
                     tests."#,
             usage = options.usage(&message));
}

// Parses command line arguments into test options
pub fn parse_opts(args: &[String]) -> Option<OptRes> {
    let opts = optgroups();
    let args = args.get(1..).unwrap_or(args);
    let matches = match opts.parse(args) {
        Ok(m) => m,
        Err(f) => return Some(Err(f.to_string())),
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

    let run_ignored = matches.opt_present("ignored");
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
            Err(_) => false
        };
    }

    let test_threads = match matches.opt_str("test-threads") {
        Some(n_str) =>
            match n_str.parse::<usize>() {
                Ok(0) =>
                    return Some(Err(format!("argument for --test-threads must not be 0"))),
                Ok(n) => Some(n),
                Err(e) =>
                    return Some(Err(format!("argument for --test-threads must be a number > 0 \
                                             (error: {})", e)))
            },
        None =>
            None,
    };

    let color = match matches.opt_str("color").as_ref().map(|s| &**s) {
        Some("auto") | None => AutoColor,
        Some("always") => AlwaysColor,
        Some("never") => NeverColor,

        Some(v) => {
            return Some(Err(format!("argument for --color must be auto, always, or never (was \
                                     {})",
                                    v)))
        }
    };

    let test_opts = TestOpts {
        list,
        filter,
        filter_exact: exact,
        run_ignored,
        run_tests,
        bench_benchmarks,
        logfile,
        nocapture,
        color,
        quiet,
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

struct ConsoleTestState<T> {
    log_out: Option<File>,
    out: OutputLocation<T>,
    use_color: bool,
    quiet: bool,
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
    max_name_len: usize, // number of columns to fill when aligning names
    options: Options,
}

impl<T: Write> ConsoleTestState<T> {
    pub fn new(opts: &TestOpts, _: Option<T>) -> io::Result<ConsoleTestState<io::Stdout>> {
        let log_out = match opts.logfile {
            Some(ref path) => Some(File::create(path)?),
            None => None,
        };
        let out = match term::stdout() {
            None => Raw(io::stdout()),
            Some(t) => Pretty(t),
        };

        Ok(ConsoleTestState {
            out,
            log_out,
            use_color: use_color(opts),
            quiet: opts.quiet,
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
            max_name_len: 0,
            options: opts.options,
        })
    }

    pub fn write_ok(&mut self) -> io::Result<()> {
        self.write_short_result("ok", ".", term::color::GREEN)
    }

    pub fn write_failed(&mut self) -> io::Result<()> {
        self.write_short_result("FAILED", "F", term::color::RED)
    }

    pub fn write_ignored(&mut self) -> io::Result<()> {
        self.write_short_result("ignored", "i", term::color::YELLOW)
    }

    pub fn write_allowed_fail(&mut self) -> io::Result<()> {
        self.write_short_result("FAILED (allowed)", "a", term::color::YELLOW)
    }

    pub fn write_bench(&mut self) -> io::Result<()> {
        self.write_pretty("bench", term::color::CYAN)
    }

    pub fn write_short_result(&mut self, verbose: &str, quiet: &str, color: term::color::Color)
                              -> io::Result<()> {
        if self.quiet {
            self.write_pretty(quiet, color)?;
            if self.current_test_count() % QUIET_MODE_MAX_COLUMN == QUIET_MODE_MAX_COLUMN - 1 {
                // we insert a new line every 100 dots in order to flush the
                // screen when dealing with line-buffered output (e.g. piping to
                // `stamp` in the rust CI).
                self.write_plain("\n")?;
            }
            Ok(())
        } else {
            self.write_pretty(verbose, color)?;
            self.write_plain("\n")
        }
    }

    pub fn write_pretty(&mut self, word: &str, color: term::color::Color) -> io::Result<()> {
        match self.out {
            Pretty(ref mut term) => {
                if self.use_color {
                    term.fg(color)?;
                }
                term.write_all(word.as_bytes())?;
                if self.use_color {
                    term.reset()?;
                }
                term.flush()
            }
            Raw(ref mut stdout) => {
                stdout.write_all(word.as_bytes())?;
                stdout.flush()
            }
        }
    }

    pub fn write_plain<S: AsRef<str>>(&mut self, s: S) -> io::Result<()> {
        let s = s.as_ref();
        match self.out {
            Pretty(ref mut term) => {
                term.write_all(s.as_bytes())?;
                term.flush()
            }
            Raw(ref mut stdout) => {
                stdout.write_all(s.as_bytes())?;
                stdout.flush()
            }
        }
    }

    pub fn write_run_start(&mut self, len: usize) -> io::Result<()> {
        self.total = len;
        let noun = if len != 1 {
            "tests"
        } else {
            "test"
        };
        self.write_plain(&format!("\nrunning {} {}\n", len, noun))
    }

    pub fn write_test_start(&mut self, test: &TestDesc, align: NamePadding) -> io::Result<()> {
        if self.quiet && align != PadOnRight {
            Ok(())
        } else {
            let name = test.padded_name(self.max_name_len, align);
            self.write_plain(&format!("test {} ... ", name))
        }
    }

    pub fn write_result(&mut self, result: &TestResult) -> io::Result<()> {
        match *result {
            TrOk => self.write_ok(),
            TrFailed | TrFailedMsg(_) => self.write_failed(),
            TrIgnored => self.write_ignored(),
            TrAllowedFail => self.write_allowed_fail(),
            TrBench(ref bs) => {
                self.write_bench()?;
                self.write_plain(&format!(": {}\n", fmt_bench_samples(bs)))
            }
        }
    }

    pub fn write_timeout(&mut self, desc: &TestDesc) -> io::Result<()> {
        self.write_plain(&format!("test {} has been running for over {} seconds\n",
                                  desc.name,
                                  TEST_WARN_TIMEOUT_S))
    }

    pub fn write_log<S: AsRef<str>>(&mut self, msg: S) -> io::Result<()> {
        let msg = msg.as_ref();
        match self.log_out {
            None => Ok(()),
            Some(ref mut o) => o.write_all(msg.as_bytes()),
        }
    }

    pub fn write_log_result(&mut self, test: &TestDesc, result: &TestResult) -> io::Result<()> {
        self.write_log(
            format!("{} {}\n",
                    match *result {
                        TrOk => "ok".to_owned(),
                        TrFailed => "failed".to_owned(),
                        TrFailedMsg(ref msg) => format!("failed: {}", msg),
                        TrIgnored => "ignored".to_owned(),
                        TrAllowedFail => "failed (allowed)".to_owned(),
                        TrBench(ref bs) => fmt_bench_samples(bs),
                    },
                    test.name))
    }

    pub fn write_failures(&mut self) -> io::Result<()> {
        self.write_plain("\nfailures:\n")?;
        let mut failures = Vec::new();
        let mut fail_out = String::new();
        for &(ref f, ref stdout) in &self.failures {
            failures.push(f.name.to_string());
            if !stdout.is_empty() {
                fail_out.push_str(&format!("---- {} stdout ----\n\t", f.name));
                let output = String::from_utf8_lossy(stdout);
                fail_out.push_str(&output);
                fail_out.push_str("\n");
            }
        }
        if !fail_out.is_empty() {
            self.write_plain("\n")?;
            self.write_plain(&fail_out)?;
        }

        self.write_plain("\nfailures:\n")?;
        failures.sort();
        for name in &failures {
            self.write_plain(&format!("    {}\n", name))?;
        }
        Ok(())
    }

    pub fn write_outputs(&mut self) -> io::Result<()> {
        self.write_plain("\nsuccesses:\n")?;
        let mut successes = Vec::new();
        let mut stdouts = String::new();
        for &(ref f, ref stdout) in &self.not_failures {
            successes.push(f.name.to_string());
            if !stdout.is_empty() {
                stdouts.push_str(&format!("---- {} stdout ----\n\t", f.name));
                let output = String::from_utf8_lossy(stdout);
                stdouts.push_str(&output);
                stdouts.push_str("\n");
            }
        }
        if !stdouts.is_empty() {
            self.write_plain("\n")?;
            self.write_plain(&stdouts)?;
        }

        self.write_plain("\nsuccesses:\n")?;
        successes.sort();
        for name in &successes {
            self.write_plain(&format!("    {}\n", name))?;
        }
        Ok(())
    }

    fn current_test_count(&self) -> usize {
        self.passed + self.failed + self.ignored + self.measured + self.allowed_fail
    }

    pub fn write_run_finish(&mut self) -> io::Result<bool> {
        assert!(self.current_test_count() == self.total);

        if self.options.display_output {
            self.write_outputs()?;
        }
        let success = self.failed == 0;
        if !success {
            self.write_failures()?;
        }

        self.write_plain("\ntest result: ")?;
        if success {
            // There's no parallelism at this point so it's safe to use color
            self.write_pretty("ok", term::color::GREEN)?;
        } else {
            self.write_pretty("FAILED", term::color::RED)?;
        }
        let s = if self.allowed_fail > 0 {
            format!(
                ". {} passed; {} failed ({} allowed); {} ignored; {} measured; {} filtered out\n\n",
                self.passed,
                self.failed + self.allowed_fail,
                self.allowed_fail,
                self.ignored,
                self.measured,
                self.filtered_out)
        } else {
            format!(
                ". {} passed; {} failed; {} ignored; {} measured; {} filtered out\n\n",
                self.passed,
                self.failed,
                self.ignored,
                self.measured,
                self.filtered_out)
        };
        self.write_plain(&s)?;
        return Ok(success);
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

    output.write_fmt(format_args!("{:>11} ns/iter (+/- {})",
                                  fmt_thousands_sep(median, ','),
                                  fmt_thousands_sep(deviation, ',')))
          .unwrap();
    if bs.mb_s != 0 {
        output.write_fmt(format_args!(" = {} MB/s", bs.mb_s)).unwrap();
    }
    output
}

// List the tests to console, and optionally to logfile. Filters are honored.
pub fn list_tests_console(opts: &TestOpts, tests: Vec<TestDescAndFn>) -> io::Result<()> {
    let mut st = ConsoleTestState::new(opts, None::<io::Stdout>)?;

    let mut ntest = 0;
    let mut nbench = 0;

    for test in filter_tests(&opts, tests) {
        use TestFn::*;

        let TestDescAndFn { desc: TestDesc { name, .. }, testfn } = test;

        let fntype = match testfn {
            StaticTestFn(..) | DynTestFn(..) => { ntest += 1; "test" },
            StaticBenchFn(..) | DynBenchFn(..) => { nbench += 1; "benchmark" },
        };

        st.write_plain(format!("{}: {}\n", name, fntype))?;
        st.write_log(format!("{} {}\n", fntype, name))?;
    }

    fn plural(count: u32, s: &str) -> String {
        match count {
            1 => format!("{} {}", 1, s),
            n => format!("{} {}s", n, s),
        }
    }

    if !opts.quiet {
        if ntest != 0 || nbench != 0 {
            st.write_plain("\n")?;
        }
        st.write_plain(format!("{}, {}\n",
            plural(ntest, "test"),
            plural(nbench, "benchmark")))?;
    }

    Ok(())
}

// A simple console test runner
pub fn run_tests_console(opts: &TestOpts, tests: Vec<TestDescAndFn>) -> io::Result<bool> {

    fn callback<T: Write>(event: &TestEvent, st: &mut ConsoleTestState<T>) -> io::Result<()> {
        match (*event).clone() {
            TeFiltered(ref filtered_tests) => st.write_run_start(filtered_tests.len()),
            TeFilteredOut(filtered_out) => Ok(st.filtered_out = filtered_out),
            TeWait(ref test, padding) => st.write_test_start(test, padding),
            TeTimeout(ref test) => st.write_timeout(test),
            TeResult(test, result, stdout) => {
                st.write_log_result(&test, &result)?;
                st.write_result(&result)?;
                match result {
                    TrOk => {
                        st.passed += 1;
                        st.not_failures.push((test, stdout));
                    }
                    TrIgnored => st.ignored += 1,
                    TrAllowedFail => st.allowed_fail += 1,
                    TrBench(bs) => {
                        st.metrics.insert_metric(test.name.as_slice(),
                                                 bs.ns_iter_summ.median,
                                                 bs.ns_iter_summ.max - bs.ns_iter_summ.min);
                        st.measured += 1
                    }
                    TrFailed => {
                        st.failed += 1;
                        st.failures.push((test, stdout));
                    }
                    TrFailedMsg(msg) => {
                        st.failed += 1;
                        let mut stdout = stdout;
                        stdout.extend_from_slice(
                            format!("note: {}", msg).as_bytes()
                        );
                        st.failures.push((test, stdout));
                    }
                }
                Ok(())
            }
        }
    }

    let mut st = ConsoleTestState::new(opts, None::<io::Stdout>)?;
    fn len_if_padded(t: &TestDescAndFn) -> usize {
        match t.testfn.padding() {
            PadNone => 0,
            PadOnRight => t.desc.name.as_slice().len(),
        }
    }
    if let Some(t) = tests.iter().max_by_key(|t| len_if_padded(*t)) {
        let n = t.desc.name.as_slice();
        st.max_name_len = n.len();
    }
    run_tests(opts, tests, |x| callback(&x, &mut st))?;
    return st.write_run_finish();
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

    let mut st = ConsoleTestState {
        log_out: None,
        out: Raw(Vec::new()),
        use_color: false,
        quiet: false,
        total: 0,
        passed: 0,
        failed: 0,
        ignored: 0,
        allowed_fail: 0,
        filtered_out: 0,
        measured: 0,
        max_name_len: 10,
        metrics: MetricMap::new(),
        failures: vec![(test_b, Vec::new()), (test_a, Vec::new())],
        options: Options::new(),
        not_failures: Vec::new(),
    };

    st.write_failures().unwrap();
    let s = match st.out {
        Raw(ref m) => String::from_utf8_lossy(&m[..]),
        Pretty(_) => unreachable!(),
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

#[cfg(any(target_os = "cloudabi",
          target_os = "redox",
          all(target_arch = "wasm32", not(target_os = "emscripten"))))]
fn stdout_isatty() -> bool {
    // FIXME: Implement isatty on Redox
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
    TeWait(TestDesc, NamePadding),
    TeResult(TestDesc, TestResult, Vec<u8>),
    TeTimeout(TestDesc),
    TeFilteredOut(usize),
}

pub type MonitorMsg = (TestDesc, TestResult, Vec<u8>);


pub fn run_tests<F>(opts: &TestOpts, tests: Vec<TestDescAndFn>, mut callback: F) -> io::Result<()>
    where F: FnMut(TestEvent) -> io::Result<()>
{
    use std::collections::HashMap;
    use std::sync::mpsc::RecvTimeoutError;

    let tests_len = tests.len();

    let mut filtered_tests = filter_tests(opts, tests);
    if !opts.bench_benchmarks {
        filtered_tests = convert_benchmarks_to_tests(filtered_tests);
    }

    let filtered_out = tests_len - filtered_tests.len();
    callback(TeFilteredOut(filtered_out))?;

    let filtered_descs = filtered_tests.iter()
                                       .map(|t| t.desc.clone())
                                       .collect();

    callback(TeFiltered(filtered_descs))?;

    let (filtered_tests, filtered_benchs): (Vec<_>, _) =
        filtered_tests.into_iter().partition(|e| {
            match e.testfn {
                StaticTestFn(_) | DynTestFn(_) => true,
                _ => false,
            }
        });

    let concurrency = match opts.test_threads {
        Some(n) => n,
        None => get_concurrency(),
    };

    let mut remaining = filtered_tests;
    remaining.reverse();
    let mut pending = 0;

    let (tx, rx) = channel::<MonitorMsg>();

    let mut running_tests: HashMap<TestDesc, Instant> = HashMap::new();

    fn get_timed_out_tests(running_tests: &mut HashMap<TestDesc, Instant>) -> Vec<TestDesc> {
        let now = Instant::now();
        let timed_out = running_tests.iter()
            .filter_map(|(desc, timeout)| if &now >= timeout { Some(desc.clone())} else { None })
            .collect();
        for test in &timed_out {
            running_tests.remove(test);
        }
        timed_out
    };

    fn calc_timeout(running_tests: &HashMap<TestDesc, Instant>) -> Option<Duration> {
        running_tests.values().min().map(|next_timeout| {
            let now = Instant::now();
            if *next_timeout >= now {
                *next_timeout - now
            } else {
                Duration::new(0, 0)
            }})
    };

    if concurrency == 1 {
        while !remaining.is_empty() {
            let test = remaining.pop().unwrap();
            callback(TeWait(test.desc.clone(), test.testfn.padding()))?;
            run_test(opts, !opts.run_tests, test, tx.clone());
            let (test, result, stdout) = rx.recv().unwrap();
            callback(TeResult(test, result, stdout))?;
        }
    } else {
        while pending > 0 || !remaining.is_empty() {
            while pending < concurrency && !remaining.is_empty() {
                let test = remaining.pop().unwrap();
                let timeout = Instant::now() + Duration::from_secs(TEST_WARN_TIMEOUT_S);
                running_tests.insert(test.desc.clone(), timeout);
                run_test(opts, !opts.run_tests, test, tx.clone());
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

            callback(TeWait(desc.clone(), PadNone))?;
            callback(TeResult(desc, result, stdout))?;
            pending -= 1;
        }
    }

    if opts.bench_benchmarks {
        // All benchmarks run at the end, in serial.
        for b in filtered_benchs {
            callback(TeWait(b.desc.clone(), b.testfn.padding()))?;
            run_test(opts, false, b, tx.clone());
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
                _ => {
                    panic!("RUST_TEST_THREADS is `{}`, should be a positive integer.",
                           s)
                }
            }
        }
        Err(..) => num_cpus(),
    };

    #[cfg(windows)]
    #[allow(bad_style)]
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

    #[cfg(all(target_arch = "wasm32", not(target_os = "emscripten")))]
    fn num_cpus() -> usize {
        1
    }

    #[cfg(any(target_os = "android",
              target_os = "cloudabi",
              target_os = "emscripten",
              target_os = "fuchsia",
              target_os = "ios",
              target_os = "linux",
              target_os = "macos",
              target_os = "solaris"))]
    fn num_cpus() -> usize {
        unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) as usize }
    }

    #[cfg(any(target_os = "freebsd",
              target_os = "dragonfly",
              target_os = "bitrig",
              target_os = "netbsd"))]
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
                libc::sysctl(mib.as_mut_ptr(),
                             2,
                             &mut cpus as *mut _ as *mut _,
                             &mut cpus_size as *mut _ as *mut _,
                             ptr::null_mut(),
                             0);
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
            libc::sysctl(mib.as_mut_ptr(),
                         2,
                         &mut cpus as *mut _ as *mut _,
                         &mut cpus_size as *mut _ as *mut _,
                         ptr::null_mut(),
                         0);
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
}

pub fn filter_tests(opts: &TestOpts, tests: Vec<TestDescAndFn>) -> Vec<TestDescAndFn> {
    let mut filtered = tests;

    // Remove tests that don't match the test filter
    filtered = match opts.filter {
        None => filtered,
        Some(ref filter) => {
            filtered.into_iter()
                    .filter(|test| {
                        if opts.filter_exact {
                            test.desc.name.as_slice() == &filter[..]
                        } else {
                            test.desc.name.as_slice().contains(&filter[..])
                        }
                    })
                    .collect()
        }
    };

    // Skip tests that match any of the skip filters
    filtered = filtered.into_iter()
        .filter(|t| !opts.skip.iter().any(|sf| {
                if opts.filter_exact {
                    t.desc.name.as_slice() == &sf[..]
                } else {
                    t.desc.name.as_slice().contains(&sf[..])
                }
            }))
        .collect();

    // Maybe pull out the ignored test and unignore them
    filtered = if !opts.run_ignored {
        filtered
    } else {
        fn filter(test: TestDescAndFn) -> Option<TestDescAndFn> {
            if test.desc.ignore {
                let TestDescAndFn {desc, testfn} = test;
                Some(TestDescAndFn {
                    desc: TestDesc { ignore: false, ..desc },
                    testfn,
                })
            } else {
                None
            }
        }
        filtered.into_iter().filter_map(filter).collect()
    };

    // Sort the tests alphabetically
    filtered.sort_by(|t1, t2| t1.desc.name.as_slice().cmp(t2.desc.name.as_slice()));

    filtered
}

pub fn convert_benchmarks_to_tests(tests: Vec<TestDescAndFn>) -> Vec<TestDescAndFn> {
    // convert benchmarks to tests, if we're not benchmarking them
    tests.into_iter().map(|x| {
        let testfn = match x.testfn {
            DynBenchFn(bench) => {
                DynTestFn(Box::new(move || {
                    bench::run_once(|b| {
                        __rust_begin_short_backtrace(|| bench.run(b))
                    })
                }))
            }
            StaticBenchFn(benchfn) => {
                DynTestFn(Box::new(move || {
                    bench::run_once(|b| {
                        __rust_begin_short_backtrace(|| benchfn(b))
                    })
                }))
            }
            f => f,
        };
        TestDescAndFn {
            desc: x.desc,
            testfn,
        }
    }).collect()
}

pub fn run_test(opts: &TestOpts,
                force_ignore: bool,
                test: TestDescAndFn,
                monitor_ch: Sender<MonitorMsg>) {

    let TestDescAndFn {desc, testfn} = test;

    let ignore_because_panic_abort =
        cfg!(target_arch = "wasm32") &&
        !cfg!(target_os = "emscripten") &&
        desc.should_panic != ShouldPanic::No;

    if force_ignore || desc.ignore || ignore_because_panic_abort {
        monitor_ch.send((desc, TrIgnored, Vec::new())).unwrap();
        return;
    }

    fn run_test_inner(desc: TestDesc,
                      monitor_ch: Sender<MonitorMsg>,
                      nocapture: bool,
                      testfn: Box<FnBox() + Send>) {
        struct Sink(Arc<Mutex<Vec<u8>>>);
        impl Write for Sink {
            fn write(&mut self, data: &[u8]) -> io::Result<usize> {
                Write::write(&mut *self.0.lock().unwrap(), data)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        // Buffer for capturing standard I/O
        let data = Arc::new(Mutex::new(Vec::new()));
        let data2 = data.clone();

        let name = desc.name.clone();
        let runtest = move || {
            let oldio = if !nocapture {
                Some((
                    io::set_print(Some(Box::new(Sink(data2.clone())))),
                    io::set_panic(Some(Box::new(Sink(data2))))
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
            monitor_ch.send((desc.clone(), test_result, stdout)).unwrap();
        };


        // If the platform is single-threaded we're just going to run
        // the test synchronously, regardless of the concurrency
        // level.
        let supports_threads =
            !cfg!(target_os = "emscripten") &&
            !cfg!(target_arch = "wasm32");
        if supports_threads {
            let cfg = thread::Builder::new().name(match name {
                DynTestName(ref name) => name.clone(),
                StaticTestName(name) => name.to_owned(),
            });
            cfg.spawn(runtest).unwrap();
        } else {
            runtest();
        }
    }

    match testfn {
        DynBenchFn(bencher) => {
            let bs = ::bench::benchmark(|harness| bencher.run(harness));
            monitor_ch.send((desc, TrBench(bs), Vec::new())).unwrap();
            return;
        }
        StaticBenchFn(benchfn) => {
            let bs = ::bench::benchmark(|harness| (benchfn.clone())(harness));
            monitor_ch.send((desc, TrBench(bs), Vec::new())).unwrap();
            return;
        }
        DynTestFn(f) => {
            let cb = move || {
                __rust_begin_short_backtrace(f)
            };
            run_test_inner(desc, monitor_ch, opts.nocapture, Box::new(cb))
        }
        StaticTestFn(f) =>
            run_test_inner(desc, monitor_ch, opts.nocapture,
                           Box::new(move || __rust_begin_short_backtrace(f))),
    }
}

/// Fixed frame used to clean the backtrace with `RUST_BACKTRACE=1`.
#[inline(never)]
fn __rust_begin_short_backtrace<F: FnOnce()>(f: F) {
    f()
}

fn calc_result(desc: &TestDesc, task_result: Result<(), Box<Any + Send>>) -> TestResult {
    match (&desc.should_panic, task_result) {
        (&ShouldPanic::No, Ok(())) |
        (&ShouldPanic::Yes, Err(_)) => TrOk,
        (&ShouldPanic::YesWithMessage(msg), Err(ref err)) =>
            if err.downcast_ref::<String>()
                  .map(|e| &**e)
                  .or_else(|| err.downcast_ref::<&'static str>().map(|e| *e))
                  .map(|e| e.contains(msg))
                  .unwrap_or(false) {
                TrOk
            } else {
                if desc.allow_fail {
                    TrAllowedFail
                } else {
                    TrFailedMsg(format!("Panic did not include expected string '{}'", msg))
                }
            },
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
        let m = Metric {
            value,
            noise,
        };
        self.0.insert(name.to_owned(), m);
    }

    pub fn fmt_metrics(&self) -> String {
        let v = self.0
                   .iter()
                   .map(|(k, v)| format!("{}: {} (+/- {})", *k, v.value, v.noise))
                   .collect::<Vec<_>>();
        v.join(", ")
    }
}


// Benchmarking

/// A function that is opaque to the optimizer, to allow benchmarks to
/// pretend to use outputs to assist in avoiding dead-code
/// elimination.
///
/// This function is a no-op, and does not even read from `dummy`.
#[cfg(not(any(target_arch = "asmjs", target_arch = "wasm32")))]
pub fn black_box<T>(dummy: T) -> T {
    // we need to "use" the argument in some way LLVM can't
    // introspect.
    unsafe { asm!("" : : "r"(&dummy)) }
    dummy
}
#[cfg(any(target_arch = "asmjs", target_arch = "wasm32"))]
#[inline(never)]
pub fn black_box<T>(dummy: T) -> T {
    dummy
}


impl Bencher {
    /// Callback for benchmark functions to run in their body.
    pub fn iter<T, F>(&mut self, mut inner: F)
        where F: FnMut() -> T
    {
        if self.mode == BenchMode::Single {
            ns_iter_inner(&mut inner, 1);
            return;
        }

        self.summary = Some(iter(&mut inner));
    }

    pub fn bench<F>(&mut self, mut f: F) -> Option<stats::Summary>
        where F: FnMut(&mut Bencher)
    {
        f(self);
        return self.summary;
    }
}

fn ns_from_dur(dur: Duration) -> u64 {
    dur.as_secs() * 1_000_000_000 + (dur.subsec_nanos() as u64)
}

fn ns_iter_inner<T, F>(inner: &mut F, k: u64) -> u64
    where F: FnMut() -> T
{
    let start = Instant::now();
    for _ in 0..k {
        black_box(inner());
    }
    return ns_from_dur(start.elapsed());
}


pub fn iter<T, F>(inner: &mut F) -> stats::Summary
    where F: FnMut() -> T
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
    // (i.e. larger error bars).
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
        if loop_run > Duration::from_millis(100) && summ.median_abs_dev_pct < 1.0 &&
           summ.median - summ5.median < summ5.median_abs_dev {
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
    use std::cmp;
    use stats;
    use super::{Bencher, BenchSamples, BenchMode};

    pub fn benchmark<F>(f: F) -> BenchSamples
        where F: FnMut(&mut Bencher)
    {
        let mut bs = Bencher {
            mode: BenchMode::Auto,
            summary: None,
            bytes: 0,
        };

        return match bs.bench(f) {
            Some(ns_iter_summ) => {
                let ns_iter = cmp::max(ns_iter_summ.median as u64, 1);
                let mb_s = bs.bytes * 1000 / ns_iter;

                BenchSamples {
                    ns_iter_summ,
                    mb_s: mb_s as usize,
                }
            }
            None => {
                // iter not called, so no data.
                // FIXME: error in this case?
                let samples: &mut [f64] = &mut [0.0_f64; 1];
                BenchSamples {
                    ns_iter_summ: stats::Summary::new(samples),
                    mb_s: 0,
                }
            }
        };
    }

    pub fn run_once<F>(f: F)
        where F: FnMut(&mut Bencher)
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
mod tests {
    use test::{TrFailed, TrFailedMsg, TrIgnored, TrOk, filter_tests, parse_opts, TestDesc,
               TestDescAndFn, TestOpts, run_test, MetricMap, StaticTestName, DynTestName,
               DynTestFn, ShouldPanic};
    use std::sync::mpsc::channel;
    use bench;
    use Bencher;

    #[test]
    pub fn do_not_run_ignored_tests() {
        fn f() {
            panic!();
        }
        let desc = TestDescAndFn {
            desc: TestDesc {
                name: StaticTestName("whatever"),
                ignore: true,
                should_panic: ShouldPanic::No,
                allow_fail: false,
            },
            testfn: DynTestFn(Box::new(f)),
        };
        let (tx, rx) = channel();
        run_test(&TestOpts::new(), false, desc, tx);
        let (_, res, _) = rx.recv().unwrap();
        assert!(res != TrOk);
    }

    #[test]
    pub fn ignored_tests_result_in_ignored() {
        fn f() {}
        let desc = TestDescAndFn {
            desc: TestDesc {
                name: StaticTestName("whatever"),
                ignore: true,
                should_panic: ShouldPanic::No,
                allow_fail: false,
            },
            testfn: DynTestFn(Box::new(f)),
        };
        let (tx, rx) = channel();
        run_test(&TestOpts::new(), false, desc, tx);
        let (_, res, _) = rx.recv().unwrap();
        assert!(res == TrIgnored);
    }

    #[test]
    fn test_should_panic() {
        fn f() {
            panic!();
        }
        let desc = TestDescAndFn {
            desc: TestDesc {
                name: StaticTestName("whatever"),
                ignore: false,
                should_panic: ShouldPanic::Yes,
                allow_fail: false,
            },
            testfn: DynTestFn(Box::new(f)),
        };
        let (tx, rx) = channel();
        run_test(&TestOpts::new(), false, desc, tx);
        let (_, res, _) = rx.recv().unwrap();
        assert!(res == TrOk);
    }

    #[test]
    fn test_should_panic_good_message() {
        fn f() {
            panic!("an error message");
        }
        let desc = TestDescAndFn {
            desc: TestDesc {
                name: StaticTestName("whatever"),
                ignore: false,
                should_panic: ShouldPanic::YesWithMessage("error message"),
                allow_fail: false,
            },
            testfn: DynTestFn(Box::new(f)),
        };
        let (tx, rx) = channel();
        run_test(&TestOpts::new(), false, desc, tx);
        let (_, res, _) = rx.recv().unwrap();
        assert!(res == TrOk);
    }

    #[test]
    fn test_should_panic_bad_message() {
        fn f() {
            panic!("an error message");
        }
        let expected = "foobar";
        let failed_msg = "Panic did not include expected string";
        let desc = TestDescAndFn {
            desc: TestDesc {
                name: StaticTestName("whatever"),
                ignore: false,
                should_panic: ShouldPanic::YesWithMessage(expected),
                allow_fail: false,
            },
            testfn: DynTestFn(Box::new(f)),
        };
        let (tx, rx) = channel();
        run_test(&TestOpts::new(), false, desc, tx);
        let (_, res, _) = rx.recv().unwrap();
        assert!(res == TrFailedMsg(format!("{} '{}'", failed_msg, expected)));
    }

    #[test]
    fn test_should_panic_but_succeeds() {
        fn f() {}
        let desc = TestDescAndFn {
            desc: TestDesc {
                name: StaticTestName("whatever"),
                ignore: false,
                should_panic: ShouldPanic::Yes,
                allow_fail: false,
            },
            testfn: DynTestFn(Box::new(f)),
        };
        let (tx, rx) = channel();
        run_test(&TestOpts::new(), false, desc, tx);
        let (_, res, _) = rx.recv().unwrap();
        assert!(res == TrFailed);
    }

    #[test]
    fn parse_ignored_flag() {
        let args = vec!["progname".to_string(), "filter".to_string(), "--ignored".to_string()];
        let opts = match parse_opts(&args) {
            Some(Ok(o)) => o,
            _ => panic!("Malformed arg in parse_ignored_flag"),
        };
        assert!((opts.run_ignored));
    }

    #[test]
    pub fn filter_for_ignored_option() {
        // When we run ignored tests the test filter should filter out all the
        // unignored tests and flip the ignore flag on the rest to false

        let mut opts = TestOpts::new();
        opts.run_tests = true;
        opts.run_ignored = true;

        let tests = vec![TestDescAndFn {
                             desc: TestDesc {
                                 name: StaticTestName("1"),
                                 ignore: true,
                                 should_panic: ShouldPanic::No,
                                 allow_fail: false,
                             },
                             testfn: DynTestFn(Box::new(move || {})),
                         },
                         TestDescAndFn {
                             desc: TestDesc {
                                 name: StaticTestName("2"),
                                 ignore: false,
                                 should_panic: ShouldPanic::No,
                                 allow_fail: false,
                             },
                             testfn: DynTestFn(Box::new(move || {})),
                         }];
        let filtered = filter_tests(&opts, tests);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].desc.name.to_string(), "1");
        assert!(!filtered[0].desc.ignore);
    }

    #[test]
    pub fn exact_filter_match() {
        fn tests() -> Vec<TestDescAndFn> {
            vec!["base",
                 "base::test",
                 "base::test1",
                 "base::test2",
            ].into_iter()
            .map(|name| TestDescAndFn {
                desc: TestDesc {
                    name: StaticTestName(name),
                    ignore: false,
                    should_panic: ShouldPanic::No,
                    allow_fail: false,
                },
                testfn: DynTestFn(Box::new(move || {}))
            })
            .collect()
        }

        let substr = filter_tests(&TestOpts {
                filter: Some("base".into()),
                ..TestOpts::new()
            }, tests());
        assert_eq!(substr.len(), 4);

        let substr = filter_tests(&TestOpts {
                filter: Some("bas".into()),
                ..TestOpts::new()
            }, tests());
        assert_eq!(substr.len(), 4);

        let substr = filter_tests(&TestOpts {
                filter: Some("::test".into()),
                ..TestOpts::new()
            }, tests());
        assert_eq!(substr.len(), 3);

        let substr = filter_tests(&TestOpts {
                filter: Some("base::test".into()),
                ..TestOpts::new()
            }, tests());
        assert_eq!(substr.len(), 3);

        let exact = filter_tests(&TestOpts {
                filter: Some("base".into()),
                filter_exact: true, ..TestOpts::new()
            }, tests());
        assert_eq!(exact.len(), 1);

        let exact = filter_tests(&TestOpts {
                filter: Some("bas".into()),
                filter_exact: true,
                ..TestOpts::new()
            }, tests());
        assert_eq!(exact.len(), 0);

        let exact = filter_tests(&TestOpts {
                filter: Some("::test".into()),
                filter_exact: true,
                ..TestOpts::new()
            }, tests());
        assert_eq!(exact.len(), 0);

        let exact = filter_tests(&TestOpts {
                filter: Some("base::test".into()),
                filter_exact: true,
                ..TestOpts::new()
            }, tests());
        assert_eq!(exact.len(), 1);
    }

    #[test]
    pub fn sort_tests() {
        let mut opts = TestOpts::new();
        opts.run_tests = true;

        let names = vec!["sha1::test".to_string(),
                         "isize::test_to_str".to_string(),
                         "isize::test_pow".to_string(),
                         "test::do_not_run_ignored_tests".to_string(),
                         "test::ignored_tests_result_in_ignored".to_string(),
                         "test::first_free_arg_should_be_a_filter".to_string(),
                         "test::parse_ignored_flag".to_string(),
                         "test::filter_for_ignored_option".to_string(),
                         "test::sort_tests".to_string()];
        let tests = {
            fn testfn() {}
            let mut tests = Vec::new();
            for name in &names {
                let test = TestDescAndFn {
                    desc: TestDesc {
                        name: DynTestName((*name).clone()),
                        ignore: false,
                        should_panic: ShouldPanic::No,
                        allow_fail: false,
                    },
                    testfn: DynTestFn(Box::new(testfn)),
                };
                tests.push(test);
            }
            tests
        };
        let filtered = filter_tests(&opts, tests);

        let expected = vec!["isize::test_pow".to_string(),
                            "isize::test_to_str".to_string(),
                            "sha1::test".to_string(),
                            "test::do_not_run_ignored_tests".to_string(),
                            "test::filter_for_ignored_option".to_string(),
                            "test::first_free_arg_should_be_a_filter".to_string(),
                            "test::ignored_tests_result_in_ignored".to_string(),
                            "test::parse_ignored_flag".to_string(),
                            "test::sort_tests".to_string()];

        for (a, b) in expected.iter().zip(filtered) {
            assert!(*a == b.desc.name.to_string());
        }
    }

    #[test]
    pub fn test_metricmap_compare() {
        let mut m1 = MetricMap::new();
        let mut m2 = MetricMap::new();
        m1.insert_metric("in-both-noise", 1000.0, 200.0);
        m2.insert_metric("in-both-noise", 1100.0, 200.0);

        m1.insert_metric("in-first-noise", 1000.0, 2.0);
        m2.insert_metric("in-second-noise", 1000.0, 2.0);

        m1.insert_metric("in-both-want-downwards-but-regressed", 1000.0, 10.0);
        m2.insert_metric("in-both-want-downwards-but-regressed", 2000.0, 10.0);

        m1.insert_metric("in-both-want-downwards-and-improved", 2000.0, 10.0);
        m2.insert_metric("in-both-want-downwards-and-improved", 1000.0, 10.0);

        m1.insert_metric("in-both-want-upwards-but-regressed", 2000.0, -10.0);
        m2.insert_metric("in-both-want-upwards-but-regressed", 1000.0, -10.0);

        m1.insert_metric("in-both-want-upwards-and-improved", 1000.0, -10.0);
        m2.insert_metric("in-both-want-upwards-and-improved", 2000.0, -10.0);
    }

    #[test]
    pub fn test_bench_once_no_iter() {
        fn f(_: &mut Bencher) {}
        bench::run_once(f);
    }

    #[test]
    pub fn test_bench_once_iter() {
        fn f(b: &mut Bencher) {
            b.iter(|| {
            })
        }
        bench::run_once(f);
    }

    #[test]
    pub fn test_bench_no_iter() {
        fn f(_: &mut Bencher) {}
        bench::benchmark(f);
    }

    #[test]
    pub fn test_bench_iter() {
        fn f(b: &mut Bencher) {
            b.iter(|| {
            })
        }
        bench::benchmark(f);
    }
}
