//! Module providing interface for running tests in the console.

use std::fs::File;
use std::io;
use std::io::prelude::Write;
use std::time::Instant;

use super::{
    bench::fmt_bench_samples,
    cli::TestOpts,
    event::{CompletedTest, TestEvent},
    filter_tests,
    formatters::{JsonFormatter, JunitFormatter, OutputFormatter, PrettyFormatter, TerseFormatter},
    helpers::{concurrency::get_concurrency, metrics::MetricMap},
    options::{Options, OutputFormat},
    run_tests, term,
    test_result::TestResult,
    time::{TestExecTime, TestSuiteExecTime},
    types::{NamePadding, TestDesc, TestDescAndFn},
};

/// Generic wrapper over stdout.
pub enum OutputLocation<T> {
    Pretty(Box<term::StdoutTerminal>),
    Raw(T),
}

impl<T: Write> Write for OutputLocation<T> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match *self {
            OutputLocation::Pretty(ref mut term) => term.write(buf),
            OutputLocation::Raw(ref mut stdout) => stdout.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match *self {
            OutputLocation::Pretty(ref mut term) => term.flush(),
            OutputLocation::Raw(ref mut stdout) => stdout.flush(),
        }
    }
}

pub struct ConsoleTestDiscoveryState {
    pub log_out: Option<File>,
    pub tests: usize,
    pub benchmarks: usize,
    pub ignored: usize,
    pub options: Options,
}

impl ConsoleTestDiscoveryState {
    pub fn new(opts: &TestOpts) -> io::Result<ConsoleTestDiscoveryState> {
        let log_out = match opts.logfile {
            Some(ref path) => Some(File::create(path)?),
            None => None,
        };

        Ok(ConsoleTestDiscoveryState {
            log_out,
            tests: 0,
            benchmarks: 0,
            ignored: 0,
            options: opts.options,
        })
    }

    pub fn write_log<F, S>(&mut self, msg: F) -> io::Result<()>
    where
        S: AsRef<str>,
        F: FnOnce() -> S,
    {
        match self.log_out {
            None => Ok(()),
            Some(ref mut o) => {
                let msg = msg();
                let msg = msg.as_ref();
                o.write_all(msg.as_bytes())
            }
        }
    }
}

pub struct ConsoleTestState {
    pub log_out: Option<File>,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub ignored: usize,
    pub filtered_out: usize,
    pub measured: usize,
    pub exec_time: Option<TestSuiteExecTime>,
    pub metrics: MetricMap,
    pub failures: Vec<(TestDesc, Vec<u8>)>,
    pub not_failures: Vec<(TestDesc, Vec<u8>)>,
    pub ignores: Vec<(TestDesc, Vec<u8>)>,
    pub time_failures: Vec<(TestDesc, Vec<u8>)>,
    pub options: Options,
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
            filtered_out: 0,
            measured: 0,
            exec_time: None,
            metrics: MetricMap::new(),
            failures: Vec::new(),
            not_failures: Vec::new(),
            ignores: Vec::new(),
            time_failures: Vec::new(),
            options: opts.options,
        })
    }

    pub fn write_log<F, S>(&mut self, msg: F) -> io::Result<()>
    where
        S: AsRef<str>,
        F: FnOnce() -> S,
    {
        match self.log_out {
            None => Ok(()),
            Some(ref mut o) => {
                let msg = msg();
                let msg = msg.as_ref();
                o.write_all(msg.as_bytes())
            }
        }
    }

    pub fn write_log_result(
        &mut self,
        test: &TestDesc,
        result: &TestResult,
        exec_time: Option<&TestExecTime>,
    ) -> io::Result<()> {
        self.write_log(|| {
            let TestDesc { name, ignore_message, .. } = test;
            format!(
                "{} {}",
                match *result {
                    TestResult::TrOk => "ok".to_owned(),
                    TestResult::TrFailed => "failed".to_owned(),
                    TestResult::TrFailedMsg(ref msg) => format!("failed: {msg}"),
                    TestResult::TrIgnored => {
                        if let Some(msg) = ignore_message {
                            format!("ignored: {msg}")
                        } else {
                            "ignored".to_owned()
                        }
                    }
                    TestResult::TrBench(ref bs) => fmt_bench_samples(bs),
                    TestResult::TrTimedFail => "failed (time limit exceeded)".to_owned(),
                },
                name,
            )
        })?;
        if let Some(exec_time) = exec_time {
            self.write_log(|| format!(" <{exec_time}>"))?;
        }
        self.write_log(|| "\n")
    }

    fn current_test_count(&self) -> usize {
        self.passed + self.failed + self.ignored + self.measured
    }
}

// List the tests to console, and optionally to logfile. Filters are honored.
pub fn list_tests_console(opts: &TestOpts, tests: Vec<TestDescAndFn>) -> io::Result<()> {
    let output = match term::stdout() {
        None => OutputLocation::Raw(io::stdout().lock()),
        Some(t) => OutputLocation::Pretty(t),
    };

    let mut out: Box<dyn OutputFormatter> = match opts.format {
        OutputFormat::Pretty | OutputFormat::Junit => {
            Box::new(PrettyFormatter::new(output, false, 0, false, None))
        }
        OutputFormat::Terse => Box::new(TerseFormatter::new(output, false, 0, false)),
        OutputFormat::Json => Box::new(JsonFormatter::new(output)),
    };
    let mut st = ConsoleTestDiscoveryState::new(opts)?;

    out.write_discovery_start()?;
    for test in filter_tests(opts, tests).into_iter() {
        use crate::TestFn::*;

        let TestDescAndFn { desc, testfn } = test;

        let fntype = match testfn {
            StaticTestFn(..) | DynTestFn(..) | StaticBenchAsTestFn(..) | DynBenchAsTestFn(..) => {
                st.tests += 1;
                "test"
            }
            StaticBenchFn(..) | DynBenchFn(..) => {
                st.benchmarks += 1;
                "benchmark"
            }
        };

        st.ignored += if desc.ignore { 1 } else { 0 };

        out.write_test_discovered(&desc, fntype)?;
        st.write_log(|| format!("{fntype} {}\n", desc.name))?;
    }

    out.write_discovery_finish(&st)
}

// Updates `ConsoleTestState` depending on result of the test execution.
fn handle_test_result(st: &mut ConsoleTestState, completed_test: CompletedTest) {
    let test = completed_test.desc;
    let stdout = completed_test.stdout;
    match completed_test.result {
        TestResult::TrOk => {
            st.passed += 1;
            st.not_failures.push((test, stdout));
        }
        TestResult::TrIgnored => {
            st.ignored += 1;
            st.ignores.push((test, stdout));
        }
        TestResult::TrBench(bs) => {
            st.metrics.insert_metric(
                test.name.as_slice(),
                bs.ns_iter_summ.median,
                bs.ns_iter_summ.max - bs.ns_iter_summ.min,
            );
            st.measured += 1
        }
        TestResult::TrFailed => {
            st.failed += 1;
            st.failures.push((test, stdout));
        }
        TestResult::TrFailedMsg(msg) => {
            st.failed += 1;
            let mut stdout = stdout;
            stdout.extend_from_slice(format!("note: {msg}").as_bytes());
            st.failures.push((test, stdout));
        }
        TestResult::TrTimedFail => {
            st.failed += 1;
            st.time_failures.push((test, stdout));
        }
    }
}

// Handler for events that occur during test execution.
// It is provided as a callback to the `run_tests` function.
fn on_test_event(
    event: &TestEvent,
    st: &mut ConsoleTestState,
    out: &mut dyn OutputFormatter,
) -> io::Result<()> {
    match (*event).clone() {
        TestEvent::TeFiltered(filtered_tests, shuffle_seed) => {
            st.total = filtered_tests;
            out.write_run_start(filtered_tests, shuffle_seed)?;
        }
        TestEvent::TeFilteredOut(filtered_out) => {
            st.filtered_out = filtered_out;
        }
        TestEvent::TeWait(ref test) => out.write_test_start(test)?,
        TestEvent::TeTimeout(ref test) => out.write_timeout(test)?,
        TestEvent::TeResult(completed_test) => {
            let test = &completed_test.desc;
            let result = &completed_test.result;
            let exec_time = &completed_test.exec_time;
            let stdout = &completed_test.stdout;

            st.write_log_result(test, result, exec_time.as_ref())?;
            out.write_result(test, result, exec_time.as_ref(), stdout, st)?;
            handle_test_result(st, completed_test);
        }
    }

    Ok(())
}

/// A simple console test runner.
/// Runs provided tests reporting process and results to the stdout.
pub fn run_tests_console(opts: &TestOpts, tests: Vec<TestDescAndFn>) -> io::Result<bool> {
    let output = match term::stdout() {
        None => OutputLocation::Raw(io::stdout()),
        Some(t) => OutputLocation::Pretty(t),
    };

    let max_name_len = tests
        .iter()
        .max_by_key(|t| len_if_padded(t))
        .map(|t| t.desc.name.as_slice().len())
        .unwrap_or(0);

    let is_multithreaded = opts.test_threads.unwrap_or_else(get_concurrency) > 1;

    let mut out: Box<dyn OutputFormatter> = match opts.format {
        OutputFormat::Pretty => Box::new(PrettyFormatter::new(
            output,
            opts.use_color(),
            max_name_len,
            is_multithreaded,
            opts.time_options,
        )),
        OutputFormat::Terse => {
            Box::new(TerseFormatter::new(output, opts.use_color(), max_name_len, is_multithreaded))
        }
        OutputFormat::Json => Box::new(JsonFormatter::new(output)),
        OutputFormat::Junit => Box::new(JunitFormatter::new(output)),
    };
    let mut st = ConsoleTestState::new(opts)?;

    // Prevent the usage of `Instant` in some cases:
    // - It's currently not supported for wasm targets.
    // - We disable it for miri because it's not available when isolation is enabled.
    let is_instant_supported = !cfg!(target_family = "wasm") && !cfg!(miri);

    let start_time = is_instant_supported.then(Instant::now);
    run_tests(opts, tests, |x| on_test_event(&x, &mut st, &mut *out))?;
    st.exec_time = start_time.map(|t| TestSuiteExecTime(t.elapsed()));

    assert!(opts.fail_fast || st.current_test_count() == st.total);

    out.write_run_finish(&st)
}

// Calculates padding for given test description.
fn len_if_padded(t: &TestDescAndFn) -> usize {
    match t.testfn.padding() {
        NamePadding::PadNone => 0,
        NamePadding::PadOnRight => t.desc.name.as_slice().len(),
    }
}
