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
//! See the [Testing Chapter](../book/testing.html) of the book for more details.

// Currently, not much of this is meant for users. It is intended to
// support the simplest interface possible for representing and
// running tests while providing a base that other test frameworks may
// build off of.

// Do not remove on snapshot creation. Needed for bootstrap. (Issue #22364)
#![cfg_attr(stage0, feature(custom_attribute))]
#![crate_name = "test"]
#![unstable(feature = "test", issue = "27812")]
#![cfg_attr(stage0, staged_api)]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "https://doc.rust-lang.org/favicon.ico",
       html_root_url = "https://doc.rust-lang.org/nightly/",
       test(attr(deny(warnings))))]

#![feature(asm)]
#![feature(box_syntax)]
#![feature(fnbox)]
#![feature(libc)]
#![feature(rustc_private)]
#![feature(set_stdio)]
#![feature(staged_api)]
#![feature(time2)]

extern crate getopts;
extern crate serialize;
extern crate serialize as rustc_serialize;
extern crate term;
extern crate libc;

pub use self::TestFn::*;
pub use self::ColorConfig::*;
pub use self::TestResult::*;
pub use self::TestName::*;
use self::TestEvent::*;
use self::NamePadding::*;
use self::OutputLocation::*;

use getopts::{OptGroup, optflag, optopt};
use std::boxed::FnBox;
use term::Terminal;
use term::color::{Color, RED, YELLOW, GREEN, CYAN};

use std::any::Any;
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

// to be used by rustc to compile tests in libtest
pub mod test {
    pub use {Bencher, TestName, TestResult, TestDesc,
             TestDescAndFn, TestOpts, TrFailed, TrIgnored, TrOk,
             Metric, MetricMap,
             StaticTestFn, StaticTestName, DynTestName, DynTestFn,
             run_test, test_main, test_main_static, filter_tests,
             parse_opts, StaticBenchFn, ShouldPanic};
}

pub mod stats;

// The name of a test. By convention this follows the rules for rust
// paths; i.e. it should be a series of identifiers separated by double
// colons. This way if some test runner wants to arrange the tests
// hierarchically it may.

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum TestName {
    StaticTestName(&'static str),
    DynTestName(String)
}
impl TestName {
    fn as_slice(&self) -> &str {
        match *self {
            StaticTestName(s) => s,
            DynTestName(ref s) => s
        }
    }
}
impl fmt::Display for TestName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self.as_slice(), f)
    }
}

#[derive(Clone, Copy)]
enum NamePadding {
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
    StaticMetricFn(fn(&mut MetricMap)),
    DynTestFn(Box<FnBox() + Send>),
    DynMetricFn(Box<FnBox(&mut MetricMap)+Send>),
    DynBenchFn(Box<TDynBenchFn+'static>)
}

impl TestFn {
    fn padding(&self) -> NamePadding {
        match *self {
            StaticTestFn(..)   => PadNone,
            StaticBenchFn(..)  => PadOnRight,
            StaticMetricFn(..) => PadOnRight,
            DynTestFn(..)      => PadNone,
            DynMetricFn(..)    => PadOnRight,
            DynBenchFn(..)     => PadOnRight,
        }
    }
}

impl fmt::Debug for TestFn {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match *self {
            StaticTestFn(..) => "StaticTestFn(..)",
            StaticBenchFn(..) => "StaticBenchFn(..)",
            StaticMetricFn(..) => "StaticMetricFn(..)",
            DynTestFn(..) => "DynTestFn(..)",
            DynMetricFn(..) => "DynMetricFn(..)",
            DynBenchFn(..) => "DynBenchFn(..)"
        })
    }
}

/// Manager of the benchmarking runs.
///
/// This is fed into functions marked with `#[bench]` to allow for
/// set-up & tear-down before running a piece of code repeatedly via a
/// call to `iter`.
#[derive(Copy, Clone)]
pub struct Bencher {
    iterations: u64,
    dur: Duration,
    pub bytes: u64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShouldPanic {
    No,
    Yes,
    YesWithMessage(&'static str)
}

// The definition of a single test. A test runner will run a list of
// these.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TestDesc {
    pub name: TestName,
    pub ignore: bool,
    pub should_panic: ShouldPanic,
}

unsafe impl Send for TestDesc {}

#[derive(Debug)]
pub struct TestDescAndFn {
    pub desc: TestDesc,
    pub testfn: TestFn,
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq, Debug, Copy)]
pub struct Metric {
    value: f64,
    noise: f64
}

impl Metric {
    pub fn new(value: f64, noise: f64) -> Metric {
        Metric {value: value, noise: noise}
    }
}

#[derive(PartialEq)]
pub struct MetricMap(BTreeMap<String,Metric>);

impl Clone for MetricMap {
    fn clone(&self) -> MetricMap {
        let MetricMap(ref map) = *self;
        MetricMap(map.clone())
    }
}

// The default console test runner. It accepts the command line
// arguments and a vector of test_descs.
pub fn test_main(args: &[String], tests: Vec<TestDescAndFn> ) {
    let opts =
        match parse_opts(args) {
            Some(Ok(o)) => o,
            Some(Err(msg)) => panic!("{:?}", msg),
            None => return
        };
    match run_tests_console(&opts, tests) {
        Ok(true) => {}
        Ok(false) => std::process::exit(101),
        Err(e) => panic!("io error when running tests: {:?}", e),
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
    let owned_tests = tests.iter().map(|t| {
        match t.testfn {
            StaticTestFn(f) => TestDescAndFn { testfn: StaticTestFn(f), desc: t.desc.clone() },
            StaticBenchFn(f) => TestDescAndFn { testfn: StaticBenchFn(f), desc: t.desc.clone() },
            _ => panic!("non-static tests passed to test::test_main_static")
        }
    }).collect();
    test_main(&args, owned_tests)
}

#[derive(Copy, Clone)]
pub enum ColorConfig {
    AutoColor,
    AlwaysColor,
    NeverColor,
}

pub struct TestOpts {
    pub filter: Option<String>,
    pub run_ignored: bool,
    pub run_tests: bool,
    pub bench_benchmarks: bool,
    pub logfile: Option<PathBuf>,
    pub nocapture: bool,
    pub color: ColorConfig,
}

impl TestOpts {
    #[cfg(test)]
    fn new() -> TestOpts {
        TestOpts {
            filter: None,
            run_ignored: false,
            run_tests: false,
            bench_benchmarks: false,
            logfile: None,
            nocapture: false,
            color: AutoColor,
        }
    }
}

/// Result of parsing the options.
pub type OptRes = Result<TestOpts, String>;

fn optgroups() -> Vec<getopts::OptGroup> {
    vec!(getopts::optflag("", "ignored", "Run ignored tests"),
      getopts::optflag("", "test", "Run tests and not benchmarks"),
      getopts::optflag("", "bench", "Run benchmarks instead of tests"),
      getopts::optflag("h", "help", "Display this message (longer with --help)"),
      getopts::optopt("", "logfile", "Write logs to the specified file instead \
                          of stdout", "PATH"),
      getopts::optflag("", "nocapture", "don't capture stdout/stderr of each \
                                         task, allow printing directly"),
      getopts::optopt("", "color", "Configure coloring of output:
            auto   = colorize if stdout is a tty and tests are run on serially (default);
            always = always colorize output;
            never  = never colorize output;", "auto|always|never"))
}

fn usage(binary: &str) {
    let message = format!("Usage: {} [OPTIONS] [FILTER]", binary);
    println!(r#"{usage}

The FILTER regex is tested against the name of all tests to run, and
only those tests that match are run.

By default, all tests are run in parallel. This can be altered with the
RUST_TEST_THREADS environment variable when running tests (set it to 1).

All tests have their standard output and standard error captured by default.
This can be overridden with the --nocapture flag or the RUST_TEST_NOCAPTURE=1
environment variable. Logging is not captured by default.

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
             usage = getopts::usage(&message, &optgroups()));
}

// Parses command line arguments into test options
pub fn parse_opts(args: &[String]) -> Option<OptRes> {
    let args_ = &args[1..];
    let matches =
        match getopts::getopts(args_, &optgroups()) {
          Ok(m) => m,
          Err(f) => return Some(Err(f.to_string()))
        };

    if matches.opt_present("h") { usage(&args[0]); return None; }

    let filter = if !matches.free.is_empty() {
        Some(matches.free[0].clone())
    } else {
        None
    };

    let run_ignored = matches.opt_present("ignored");

    let logfile = matches.opt_str("logfile");
    let logfile = logfile.map(|s| PathBuf::from(&s));

    let bench_benchmarks = matches.opt_present("bench");
    let run_tests = ! bench_benchmarks ||
        matches.opt_present("test");

    let mut nocapture = matches.opt_present("nocapture");
    if !nocapture {
        nocapture = env::var("RUST_TEST_NOCAPTURE").is_ok();
    }

    let color = match matches.opt_str("color").as_ref().map(|s| &**s) {
        Some("auto") | None => AutoColor,
        Some("always") => AlwaysColor,
        Some("never") => NeverColor,

        Some(v) => return Some(Err(format!("argument for --color must be \
                                            auto, always, or never (was {})",
                                            v))),
    };

    let test_opts = TestOpts {
        filter: filter,
        run_ignored: run_ignored,
        run_tests: run_tests,
        bench_benchmarks: bench_benchmarks,
        logfile: logfile,
        nocapture: nocapture,
        color: color,
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
    TrIgnored,
    TrMetrics(MetricMap),
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
    total: usize,
    passed: usize,
    failed: usize,
    ignored: usize,
    measured: usize,
    metrics: MetricMap,
    failures: Vec<(TestDesc, Vec<u8> )> ,
    max_name_len: usize, // number of columns to fill when aligning names
}

impl<T: Write> ConsoleTestState<T> {
    pub fn new(opts: &TestOpts,
               _: Option<T>) -> io::Result<ConsoleTestState<io::Stdout>> {
        let log_out = match opts.logfile {
            Some(ref path) => Some(try!(File::create(path))),
            None => None
        };
        let out = match term::stdout() {
            None => Raw(io::stdout()),
            Some(t) => Pretty(t)
        };

        Ok(ConsoleTestState {
            out: out,
            log_out: log_out,
            use_color: use_color(opts),
            total: 0,
            passed: 0,
            failed: 0,
            ignored: 0,
            measured: 0,
            metrics: MetricMap::new(),
            failures: Vec::new(),
            max_name_len: 0,
        })
    }

    pub fn write_ok(&mut self) -> io::Result<()> {
        self.write_pretty("ok", term::color::GREEN)
    }

    pub fn write_failed(&mut self) -> io::Result<()> {
        self.write_pretty("FAILED", term::color::RED)
    }

    pub fn write_ignored(&mut self) -> io::Result<()> {
        self.write_pretty("ignored", term::color::YELLOW)
    }

    pub fn write_metric(&mut self) -> io::Result<()> {
        self.write_pretty("metric", term::color::CYAN)
    }

    pub fn write_bench(&mut self) -> io::Result<()> {
        self.write_pretty("bench", term::color::CYAN)
    }

    pub fn write_pretty(&mut self,
                        word: &str,
                        color: term::color::Color) -> io::Result<()> {
        match self.out {
            Pretty(ref mut term) => {
                if self.use_color {
                    try!(term.fg(color));
                }
                try!(term.write_all(word.as_bytes()));
                if self.use_color {
                    try!(term.reset());
                }
                term.flush()
            }
            Raw(ref mut stdout) => {
                try!(stdout.write_all(word.as_bytes()));
                stdout.flush()
            }
        }
    }

    pub fn write_plain(&mut self, s: &str) -> io::Result<()> {
        match self.out {
            Pretty(ref mut term) => {
                try!(term.write_all(s.as_bytes()));
                term.flush()
            },
            Raw(ref mut stdout) => {
                try!(stdout.write_all(s.as_bytes()));
                stdout.flush()
            },
        }
    }

    pub fn write_run_start(&mut self, len: usize) -> io::Result<()> {
        self.total = len;
        let noun = if len != 1 { "tests" } else { "test" };
        self.write_plain(&format!("\nrunning {} {}\n", len, noun))
    }

    pub fn write_test_start(&mut self, test: &TestDesc,
                            align: NamePadding) -> io::Result<()> {
        let name = test.padded_name(self.max_name_len, align);
        self.write_plain(&format!("test {} ... ", name))
    }

    pub fn write_result(&mut self, result: &TestResult) -> io::Result<()> {
        try!(match *result {
            TrOk => self.write_ok(),
            TrFailed => self.write_failed(),
            TrIgnored => self.write_ignored(),
            TrMetrics(ref mm) => {
                try!(self.write_metric());
                self.write_plain(&format!(": {}", mm.fmt_metrics()))
            }
            TrBench(ref bs) => {
                try!(self.write_bench());

                try!(self.write_plain(&format!(": {}", fmt_bench_samples(bs))));

                Ok(())
            }
        });
        self.write_plain("\n")
    }

    pub fn write_log(&mut self, test: &TestDesc,
                     result: &TestResult) -> io::Result<()> {
        match self.log_out {
            None => Ok(()),
            Some(ref mut o) => {
                let s = format!("{} {}\n", match *result {
                        TrOk => "ok".to_owned(),
                        TrFailed => "failed".to_owned(),
                        TrIgnored => "ignored".to_owned(),
                        TrMetrics(ref mm) => mm.fmt_metrics(),
                        TrBench(ref bs) => fmt_bench_samples(bs)
                    }, test.name);
                o.write_all(s.as_bytes())
            }
        }
    }

    pub fn write_failures(&mut self) -> io::Result<()> {
        try!(self.write_plain("\nfailures:\n"));
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
            try!(self.write_plain("\n"));
            try!(self.write_plain(&fail_out));
        }

        try!(self.write_plain("\nfailures:\n"));
        failures.sort();
        for name in &failures {
            try!(self.write_plain(&format!("    {}\n", name)));
        }
        Ok(())
    }

    pub fn write_run_finish(&mut self) -> io::Result<bool> {
        assert!(self.passed + self.failed + self.ignored + self.measured == self.total);

        let success = self.failed == 0;
        if !success {
            try!(self.write_failures());
        }

        try!(self.write_plain("\ntest result: "));
        if success {
            // There's no parallelism at this point so it's safe to use color
            try!(self.write_ok());
        } else {
            try!(self.write_failed());
        }
        let s = format!(". {} passed; {} failed; {} ignored; {} measured\n\n",
                        self.passed, self.failed, self.ignored, self.measured);
        try!(self.write_plain(&s));
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
                     fmt_thousands_sep(deviation, ','))).unwrap();
    if bs.mb_s != 0 {
        output.write_fmt(format_args!(" = {} MB/s", bs.mb_s)).unwrap();
    }
    output
}

// A simple console test runner
pub fn run_tests_console(opts: &TestOpts, tests: Vec<TestDescAndFn> ) -> io::Result<bool> {

    fn callback<T: Write>(event: &TestEvent,
                          st: &mut ConsoleTestState<T>) -> io::Result<()> {
        match (*event).clone() {
            TeFiltered(ref filtered_tests) => st.write_run_start(filtered_tests.len()),
            TeWait(ref test, padding) => st.write_test_start(test, padding),
            TeResult(test, result, stdout) => {
                try!(st.write_log(&test, &result));
                try!(st.write_result(&result));
                match result {
                    TrOk => st.passed += 1,
                    TrIgnored => st.ignored += 1,
                    TrMetrics(mm) => {
                        let tname = test.name;
                        let MetricMap(mm) = mm;
                        for (k,v) in &mm {
                            st.metrics
                              .insert_metric(&format!("{}.{}",
                                                      tname,
                                                      k),
                                             v.value,
                                             v.noise);
                        }
                        st.measured += 1
                    }
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
                }
                Ok(())
            }
        }
    }

    let mut st = try!(ConsoleTestState::new(opts, None::<io::Stdout>));
    fn len_if_padded(t: &TestDescAndFn) -> usize {
        match t.testfn.padding() {
            PadNone => 0,
            PadOnRight => t.desc.name.as_slice().len(),
        }
    }
    match tests.iter().max_by_key(|t|len_if_padded(*t)) {
        Some(t) => {
            let n = t.desc.name.as_slice();
            st.max_name_len = n.len();
        },
        None => {}
    }
    try!(run_tests(opts, tests, |x| callback(&x, &mut st)));
    return st.write_run_finish();
}

#[test]
fn should_sort_failures_before_printing_them() {
    let test_a = TestDesc {
        name: StaticTestName("a"),
        ignore: false,
        should_panic: ShouldPanic::No
    };

    let test_b = TestDesc {
        name: StaticTestName("b"),
        ignore: false,
        should_panic: ShouldPanic::No
    };

    let mut st = ConsoleTestState {
        log_out: None,
        out: Raw(Vec::new()),
        use_color: false,
        total: 0,
        passed: 0,
        failed: 0,
        ignored: 0,
        measured: 0,
        max_name_len: 10,
        metrics: MetricMap::new(),
        failures: vec!((test_b, Vec::new()), (test_a, Vec::new()))
    };

    st.write_failures().unwrap();
    let s = match st.out {
        Raw(ref m) => String::from_utf8_lossy(&m[..]),
        Pretty(_) => unreachable!()
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
enum TestEvent {
    TeFiltered(Vec<TestDesc> ),
    TeWait(TestDesc, NamePadding),
    TeResult(TestDesc, TestResult, Vec<u8> ),
}

pub type MonitorMsg = (TestDesc, TestResult, Vec<u8> );


fn run_tests<F>(opts: &TestOpts,
                tests: Vec<TestDescAndFn> ,
                mut callback: F) -> io::Result<()> where
    F: FnMut(TestEvent) -> io::Result<()>,
{
    let mut filtered_tests = filter_tests(opts, tests);
    if !opts.bench_benchmarks {
        filtered_tests = convert_benchmarks_to_tests(filtered_tests);
    }

    let filtered_descs = filtered_tests.iter()
                                       .map(|t| t.desc.clone())
                                       .collect();

    try!(callback(TeFiltered(filtered_descs)));

    let (filtered_tests, filtered_benchs_and_metrics): (Vec<_>, _) =
        filtered_tests.into_iter().partition(|e| {
            match e.testfn {
                StaticTestFn(_) | DynTestFn(_) => true,
                _ => false
            }
        });

    // It's tempting to just spawn all the tests at once, but since we have
    // many tests that run in other processes we would be making a big mess.
    let concurrency = get_concurrency();

    let mut remaining = filtered_tests;
    remaining.reverse();
    let mut pending = 0;

    let (tx, rx) = channel::<MonitorMsg>();

    while pending > 0 || !remaining.is_empty() {
        while pending < concurrency && !remaining.is_empty() {
            let test = remaining.pop().unwrap();
            if concurrency == 1 {
                // We are doing one test at a time so we can print the name
                // of the test before we run it. Useful for debugging tests
                // that hang forever.
                try!(callback(TeWait(test.desc.clone(), test.testfn.padding())));
            }
            run_test(opts, !opts.run_tests, test, tx.clone());
            pending += 1;
        }

        let (desc, result, stdout) = rx.recv().unwrap();
        if concurrency != 1 {
            try!(callback(TeWait(desc.clone(), PadNone)));
        }
        try!(callback(TeResult(desc, result, stdout)));
        pending -= 1;
    }

    if opts.bench_benchmarks {
        // All benchmarks run at the end, in serial.
        // (this includes metric fns)
        for b in filtered_benchs_and_metrics {
            try!(callback(TeWait(b.desc.clone(), b.testfn.padding())));
            run_test(opts, false, b, tx.clone());
            let (test, result, stdout) = rx.recv().unwrap();
            try!(callback(TeResult(test, result, stdout)));
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
                _ => panic!("RUST_TEST_THREADS is `{}`, should be a positive integer.", s)
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

    #[cfg(unix)]
    fn num_cpus() -> usize {
        extern { fn rust_get_num_cpus() -> libc::uintptr_t; }
        unsafe { rust_get_num_cpus() as usize }
    }
}

pub fn filter_tests(opts: &TestOpts, tests: Vec<TestDescAndFn>) -> Vec<TestDescAndFn> {
    let mut filtered = tests;

    // Remove tests that don't match the test filter
    filtered = match opts.filter {
        None => filtered,
        Some(ref filter) => {
            filtered.into_iter().filter(|test| {
                test.desc.name.as_slice().contains(&filter[..])
            }).collect()
        }
    };

    // Maybe pull out the ignored test and unignore them
    filtered = if !opts.run_ignored {
        filtered
    } else {
        fn filter(test: TestDescAndFn) -> Option<TestDescAndFn> {
            if test.desc.ignore {
                let TestDescAndFn {desc, testfn} = test;
                Some(TestDescAndFn {
                    desc: TestDesc {ignore: false, ..desc},
                    testfn: testfn
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
                DynTestFn(Box::new(move || bench::run_once(|b| bench.run(b))))
            }
            StaticBenchFn(benchfn) => {
                DynTestFn(Box::new(move || bench::run_once(|b| benchfn(b))))
            }
            f => f
        };
        TestDescAndFn { desc: x.desc, testfn: testfn }
    }).collect()
}

pub fn run_test(opts: &TestOpts,
                force_ignore: bool,
                test: TestDescAndFn,
                monitor_ch: Sender<MonitorMsg>) {

    let TestDescAndFn {desc, testfn} = test;

    if force_ignore || desc.ignore {
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
            fn flush(&mut self) -> io::Result<()> { Ok(()) }
        }

        thread::spawn(move || {
            let data = Arc::new(Mutex::new(Vec::new()));
            let data2 = data.clone();
            let cfg = thread::Builder::new().name(match desc.name {
                DynTestName(ref name) => name.clone(),
                StaticTestName(name) => name.to_owned(),
            });

            let result_guard = cfg.spawn(move || {
                if !nocapture {
                    io::set_print(box Sink(data2.clone()));
                    io::set_panic(box Sink(data2));
                }
                testfn()
            }).unwrap();
            let test_result = calc_result(&desc, result_guard.join());
            let stdout = data.lock().unwrap().to_vec();
            monitor_ch.send((desc.clone(), test_result, stdout)).unwrap();
        });
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
        DynMetricFn(f) => {
            let mut mm = MetricMap::new();
            f.call_box((&mut mm,));
            monitor_ch.send((desc, TrMetrics(mm), Vec::new())).unwrap();
            return;
        }
        StaticMetricFn(f) => {
            let mut mm = MetricMap::new();
            f(&mut mm);
            monitor_ch.send((desc, TrMetrics(mm), Vec::new())).unwrap();
            return;
        }
        DynTestFn(f) => run_test_inner(desc, monitor_ch, opts.nocapture, f),
        StaticTestFn(f) => run_test_inner(desc, monitor_ch, opts.nocapture,
                                          Box::new(f))
    }
}

fn calc_result(desc: &TestDesc, task_result: Result<(), Box<Any+Send>>) -> TestResult {
    match (&desc.should_panic, task_result) {
        (&ShouldPanic::No, Ok(())) |
        (&ShouldPanic::Yes, Err(_)) => TrOk,
        (&ShouldPanic::YesWithMessage(msg), Err(ref err))
            if err.downcast_ref::<String>()
                .map(|e| &**e)
                .or_else(|| err.downcast_ref::<&'static str>().map(|e| *e))
                .map(|e| e.contains(msg))
                .unwrap_or(false) => TrOk,
        _ => TrFailed,
    }
}

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
            value: value,
            noise: noise
        };
        let MetricMap(ref mut map) = *self;
        map.insert(name.to_owned(), m);
    }

    pub fn fmt_metrics(&self) -> String {
        let MetricMap(ref mm) = *self;
        let v : Vec<String> = mm.iter()
            .map(|(k,v)| format!("{}: {} (+/- {})", *k,
                                 v.value, v.noise))
            .collect();
        v.join(", ")
    }
}


// Benchmarking

/// A function that is opaque to the optimizer, to allow benchmarks to
/// pretend to use outputs to assist in avoiding dead-code
/// elimination.
///
/// This function is a no-op, and does not even read from `dummy`.
#[cfg(not(all(target_os = "nacl", target_arch = "le32")))]
pub fn black_box<T>(dummy: T) -> T {
    // we need to "use" the argument in some way LLVM can't
    // introspect.
    unsafe {asm!("" : : "r"(&dummy))}
    dummy
}
#[cfg(all(target_os = "nacl", target_arch = "le32"))]
#[inline(never)]
pub fn black_box<T>(dummy: T) -> T { dummy }


impl Bencher {
    /// Callback for benchmark functions to run in their body.
    pub fn iter<T, F>(&mut self, mut inner: F) where F: FnMut() -> T {
        let start = Instant::now();
        let k = self.iterations;
        for _ in 0..k {
            black_box(inner());
        }
        self.dur = start.elapsed();
    }

    pub fn ns_elapsed(&mut self) -> u64 {
        self.dur.as_secs() * 1_000_000_000 + (self.dur.subsec_nanos() as u64)
    }

    pub fn ns_per_iter(&mut self) -> u64 {
        if self.iterations == 0 {
            0
        } else {
            self.ns_elapsed() / cmp::max(self.iterations, 1)
        }
    }

    pub fn bench_n<F>(&mut self, n: u64, f: F) where F: FnOnce(&mut Bencher) {
        self.iterations = n;
        f(self);
    }

    // This is a more statistics-driven benchmark algorithm
    pub fn auto_bench<F>(&mut self, mut f: F) -> stats::Summary where F: FnMut(&mut Bencher) {
        // Initial bench run to get ballpark figure.
        let mut n = 1;
        self.bench_n(n, |x| f(x));

        // Try to estimate iter count for 1ms falling back to 1m
        // iterations if first run took < 1ns.
        if self.ns_per_iter() == 0 {
            n = 1_000_000;
        } else {
            n = 1_000_000 / cmp::max(self.ns_per_iter(), 1);
        }
        // if the first run took more than 1ms we don't want to just
        // be left doing 0 iterations on every loop. The unfortunate
        // side effect of not being able to do as many runs is
        // automatically handled by the statistical analysis below
        // (i.e. larger error bars).
        if n == 0 { n = 1; }

        let mut total_run = Duration::new(0, 0);
        let samples : &mut [f64] = &mut [0.0_f64; 50];
        loop {
            let loop_start = Instant::now();

            for p in &mut *samples {
                self.bench_n(n, |x| f(x));
                *p = self.ns_per_iter() as f64;
            };

            stats::winsorize(samples, 5.0);
            let summ = stats::Summary::new(samples);

            for p in &mut *samples {
                self.bench_n(5 * n, |x| f(x));
                *p = self.ns_per_iter() as f64;
            };

            stats::winsorize(samples, 5.0);
            let summ5 = stats::Summary::new(samples);
            let loop_run = loop_start.elapsed();

            // If we've run for 100ms and seem to have converged to a
            // stable median.
            if loop_run > Duration::from_millis(100) &&
                summ.median_abs_dev_pct < 1.0 &&
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
                None => return summ5,
            };
        }
    }
}

pub mod bench {
    use std::cmp;
    use std::time::Duration;
    use super::{Bencher, BenchSamples};

    pub fn benchmark<F>(f: F) -> BenchSamples where F: FnMut(&mut Bencher) {
        let mut bs = Bencher {
            iterations: 0,
            dur: Duration::new(0, 0),
            bytes: 0
        };

        let ns_iter_summ = bs.auto_bench(f);

        let ns_iter = cmp::max(ns_iter_summ.median as u64, 1);
        let iter_s = 1_000_000_000 / ns_iter;
        let mb_s = (bs.bytes * iter_s) / 1_000_000;

        BenchSamples {
            ns_iter_summ: ns_iter_summ,
            mb_s: mb_s as usize
        }
    }

    pub fn run_once<F>(f: F) where F: FnOnce(&mut Bencher) {
        let mut bs = Bencher {
            iterations: 0,
            dur: Duration::new(0, 0),
            bytes: 0
        };
        bs.bench_n(1, f);
    }
}

#[cfg(test)]
mod tests {
    use test::{TrFailed, TrIgnored, TrOk, filter_tests, parse_opts,
               TestDesc, TestDescAndFn, TestOpts, run_test,
               MetricMap,
               StaticTestName, DynTestName, DynTestFn, ShouldPanic};
    use std::sync::mpsc::channel;

    #[test]
    pub fn do_not_run_ignored_tests() {
        fn f() { panic!(); }
        let desc = TestDescAndFn {
            desc: TestDesc {
                name: StaticTestName("whatever"),
                ignore: true,
                should_panic: ShouldPanic::No,
            },
            testfn: DynTestFn(Box::new(move|| f())),
        };
        let (tx, rx) = channel();
        run_test(&TestOpts::new(), false, desc, tx);
        let (_, res, _) = rx.recv().unwrap();
        assert!(res != TrOk);
    }

    #[test]
    pub fn ignored_tests_result_in_ignored() {
        fn f() { }
        let desc = TestDescAndFn {
            desc: TestDesc {
                name: StaticTestName("whatever"),
                ignore: true,
                should_panic: ShouldPanic::No,
            },
            testfn: DynTestFn(Box::new(move|| f())),
        };
        let (tx, rx) = channel();
        run_test(&TestOpts::new(), false, desc, tx);
        let (_, res, _) = rx.recv().unwrap();
        assert!(res == TrIgnored);
    }

    #[test]
    fn test_should_panic() {
        fn f() { panic!(); }
        let desc = TestDescAndFn {
            desc: TestDesc {
                name: StaticTestName("whatever"),
                ignore: false,
                should_panic: ShouldPanic::Yes,
            },
            testfn: DynTestFn(Box::new(move|| f())),
        };
        let (tx, rx) = channel();
        run_test(&TestOpts::new(), false, desc, tx);
        let (_, res, _) = rx.recv().unwrap();
        assert!(res == TrOk);
    }

    #[test]
    fn test_should_panic_good_message() {
        fn f() { panic!("an error message"); }
        let desc = TestDescAndFn {
            desc: TestDesc {
                name: StaticTestName("whatever"),
                ignore: false,
                should_panic: ShouldPanic::YesWithMessage("error message"),
            },
            testfn: DynTestFn(Box::new(move|| f())),
        };
        let (tx, rx) = channel();
        run_test(&TestOpts::new(), false, desc, tx);
        let (_, res, _) = rx.recv().unwrap();
        assert!(res == TrOk);
    }

    #[test]
    fn test_should_panic_bad_message() {
        fn f() { panic!("an error message"); }
        let desc = TestDescAndFn {
            desc: TestDesc {
                name: StaticTestName("whatever"),
                ignore: false,
                should_panic: ShouldPanic::YesWithMessage("foobar"),
            },
            testfn: DynTestFn(Box::new(move|| f())),
        };
        let (tx, rx) = channel();
        run_test(&TestOpts::new(), false, desc, tx);
        let (_, res, _) = rx.recv().unwrap();
        assert!(res == TrFailed);
    }

    #[test]
    fn test_should_panic_but_succeeds() {
        fn f() { }
        let desc = TestDescAndFn {
            desc: TestDesc {
                name: StaticTestName("whatever"),
                ignore: false,
                should_panic: ShouldPanic::Yes,
            },
            testfn: DynTestFn(Box::new(move|| f())),
        };
        let (tx, rx) = channel();
        run_test(&TestOpts::new(), false, desc, tx);
        let (_, res, _) = rx.recv().unwrap();
        assert!(res == TrFailed);
    }

    #[test]
    fn parse_ignored_flag() {
        let args = vec!("progname".to_string(),
                        "filter".to_string(),
                        "--ignored".to_string());
        let opts = match parse_opts(&args) {
            Some(Ok(o)) => o,
            _ => panic!("Malformed arg in parse_ignored_flag")
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

        let tests = vec!(
            TestDescAndFn {
                desc: TestDesc {
                    name: StaticTestName("1"),
                    ignore: true,
                    should_panic: ShouldPanic::No,
                },
                testfn: DynTestFn(Box::new(move|| {})),
            },
            TestDescAndFn {
                desc: TestDesc {
                    name: StaticTestName("2"),
                    ignore: false,
                    should_panic: ShouldPanic::No,
                },
                testfn: DynTestFn(Box::new(move|| {})),
            });
        let filtered = filter_tests(&opts, tests);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].desc.name.to_string(),
                   "1");
        assert!(filtered[0].desc.ignore == false);
    }

    #[test]
    pub fn sort_tests() {
        let mut opts = TestOpts::new();
        opts.run_tests = true;

        let names =
            vec!("sha1::test".to_string(),
                 "isize::test_to_str".to_string(),
                 "isize::test_pow".to_string(),
                 "test::do_not_run_ignored_tests".to_string(),
                 "test::ignored_tests_result_in_ignored".to_string(),
                 "test::first_free_arg_should_be_a_filter".to_string(),
                 "test::parse_ignored_flag".to_string(),
                 "test::filter_for_ignored_option".to_string(),
                 "test::sort_tests".to_string());
        let tests =
        {
            fn testfn() { }
            let mut tests = Vec::new();
            for name in &names {
                let test = TestDescAndFn {
                    desc: TestDesc {
                        name: DynTestName((*name).clone()),
                        ignore: false,
                        should_panic: ShouldPanic::No,
                    },
                    testfn: DynTestFn(Box::new(testfn)),
                };
                tests.push(test);
            }
            tests
        };
        let filtered = filter_tests(&opts, tests);

        let expected =
            vec!("isize::test_pow".to_string(),
                 "isize::test_to_str".to_string(),
                 "sha1::test".to_string(),
                 "test::do_not_run_ignored_tests".to_string(),
                 "test::filter_for_ignored_option".to_string(),
                 "test::first_free_arg_should_be_a_filter".to_string(),
                 "test::ignored_tests_result_in_ignored".to_string(),
                 "test::parse_ignored_flag".to_string(),
                 "test::sort_tests".to_string());

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
}
