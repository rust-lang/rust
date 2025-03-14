//! Support code for rustc's built in unit-test and micro-benchmarking
//! framework.
//!
//! Almost all user code will only be interested in `Bencher` and
//! `black_box`. All other interactions (such as writing tests and
//! benchmarks themselves) should be done via the `#[test]` and
//! `#[bench]` attributes.
//!
//! See the [Testing Chapter](../book/ch11-00-testing.html) of the book for more
//! details.

// Currently, not much of this is meant for users. It is intended to
// support the simplest interface possible for representing and
// running tests while providing a base that other test frameworks may
// build off of.

#![unstable(feature = "test", issue = "50297")]
#![doc(test(attr(deny(warnings))))]
#![doc(rust_logo)]
#![feature(rustdoc_internals)]
#![feature(file_buffered)]
#![feature(internal_output_capture)]
#![feature(io_const_error)]
#![feature(staged_api)]
#![feature(process_exitcode_internals)]
#![feature(panic_can_unwind)]
#![feature(test)]
#![feature(thread_spawn_hook)]
#![allow(internal_features)]
#![warn(rustdoc::unescaped_backticks)]
#![warn(unreachable_pub)]

pub use cli::TestOpts;

pub use self::ColorConfig::*;
pub use self::bench::{Bencher, black_box};
pub use self::console::run_tests_console;
pub use self::options::{ColorConfig, Options, OutputFormat, RunIgnored, ShouldPanic};
pub use self::types::TestName::*;
pub use self::types::*;

// Module to be used by rustc to compile tests in libtest
pub mod test {
    pub use crate::bench::Bencher;
    pub use crate::cli::{TestOpts, parse_opts};
    pub use crate::helpers::metrics::{Metric, MetricMap};
    pub use crate::options::{Options, RunIgnored, RunStrategy, ShouldPanic};
    pub use crate::test_result::{TestResult, TrFailed, TrFailedMsg, TrIgnored, TrOk};
    pub use crate::time::{TestExecTime, TestTimeOptions};
    pub use crate::types::{
        DynTestFn, DynTestName, StaticBenchFn, StaticTestFn, StaticTestName, TestDesc,
        TestDescAndFn, TestId, TestName, TestType,
    };
    pub use crate::{assert_test_result, filter_tests, run_test, test_main, test_main_static};
}

use std::collections::VecDeque;
use std::io::prelude::Write;
use std::mem::ManuallyDrop;
use std::panic::{self, AssertUnwindSafe, PanicHookInfo, catch_unwind};
use std::process::{self, Command, Termination};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::{env, io, thread};

pub mod bench;
mod cli;
mod console;
mod event;
mod formatters;
mod helpers;
mod options;
pub mod stats;
mod term;
mod test_result;
mod time;
mod types;

#[cfg(test)]
mod tests;

use core::any::Any;

use event::{CompletedTest, TestEvent};
use helpers::concurrency::get_concurrency;
use helpers::shuffle::{get_shuffle_seed, shuffle_tests};
use options::RunStrategy;
use test_result::*;
use time::TestExecTime;

// Process exit code to be used to indicate test failures.
const ERROR_EXIT_CODE: i32 = 101;

const SECONDARY_TEST_INVOKER_VAR: &str = "__RUST_TEST_INVOKE";
const SECONDARY_TEST_BENCH_BENCHMARKS_VAR: &str = "__RUST_TEST_BENCH_BENCHMARKS";

// The default console test runner. It accepts the command line
// arguments and a vector of test_descs.
pub fn test_main(args: &[String], tests: Vec<TestDescAndFn>, options: Option<Options>) {
    let mut opts = match cli::parse_opts(args) {
        Some(Ok(o)) => o,
        Some(Err(msg)) => {
            eprintln!("error: {msg}");
            process::exit(ERROR_EXIT_CODE);
        }
        None => return,
    };
    if let Some(options) = options {
        opts.options = options;
    }
    if opts.list {
        if let Err(e) = console::list_tests_console(&opts, tests) {
            eprintln!("error: io error when listing tests: {e:?}");
            process::exit(ERROR_EXIT_CODE);
        }
    } else {
        if !opts.nocapture {
            // If we encounter a non-unwinding panic, flush any captured output from the current test,
            // and stop capturing output to ensure that the non-unwinding panic message is visible.
            // We also acquire the locks for both output streams to prevent output from other threads
            // from interleaving with the panic message or appearing after it.
            let builtin_panic_hook = panic::take_hook();
            let hook = Box::new({
                move |info: &'_ PanicHookInfo<'_>| {
                    if !info.can_unwind() {
                        std::mem::forget(std::io::stderr().lock());
                        let mut stdout = ManuallyDrop::new(std::io::stdout().lock());
                        if let Some(captured) = io::set_output_capture(None) {
                            if let Ok(data) = captured.lock() {
                                let _ = stdout.write_all(&data);
                                let _ = stdout.flush();
                            }
                        }
                    }
                    builtin_panic_hook(info);
                }
            });
            panic::set_hook(hook);
            // Use a thread spawning hook to make new threads inherit output capturing.
            std::thread::add_spawn_hook(|_| {
                // Get and clone the output capture of the current thread.
                let output_capture = io::set_output_capture(None);
                io::set_output_capture(output_capture.clone());
                // Set the output capture of the new thread.
                || {
                    io::set_output_capture(output_capture);
                }
            });
        }
        let res = console::run_tests_console(&opts, tests);
        // Prevent Valgrind from reporting reachable blocks in users' unit tests.
        drop(panic::take_hook());
        match res {
            Ok(true) => {}
            Ok(false) => process::exit(ERROR_EXIT_CODE),
            Err(e) => {
                eprintln!("error: io error when listing tests: {e:?}");
                process::exit(ERROR_EXIT_CODE);
            }
        }
    }
}

/// A variant optimized for invocation with a static test vector.
/// This will panic (intentionally) when fed any dynamic tests.
///
/// This is the entry point for the main function generated by `rustc --test`
/// when panic=unwind.
pub fn test_main_static(tests: &[&TestDescAndFn]) {
    let args = env::args().collect::<Vec<_>>();
    let owned_tests: Vec<_> = tests.iter().map(make_owned_test).collect();
    test_main(&args, owned_tests, None)
}

/// A variant optimized for invocation with a static test vector.
/// This will panic (intentionally) when fed any dynamic tests.
///
/// Runs tests in panic=abort mode, which involves spawning subprocesses for
/// tests.
///
/// This is the entry point for the main function generated by `rustc --test`
/// when panic=abort.
pub fn test_main_static_abort(tests: &[&TestDescAndFn]) {
    // If we're being run in SpawnedSecondary mode, run the test here. run_test
    // will then exit the process.
    if let Ok(name) = env::var(SECONDARY_TEST_INVOKER_VAR) {
        unsafe {
            env::remove_var(SECONDARY_TEST_INVOKER_VAR);
        }

        // Convert benchmarks to tests if we're not benchmarking.
        let mut tests = tests.iter().map(make_owned_test).collect::<Vec<_>>();
        if env::var(SECONDARY_TEST_BENCH_BENCHMARKS_VAR).is_ok() {
            unsafe {
                env::remove_var(SECONDARY_TEST_BENCH_BENCHMARKS_VAR);
            }
        } else {
            tests = convert_benchmarks_to_tests(tests);
        };

        let test = tests
            .into_iter()
            .find(|test| test.desc.name.as_slice() == name)
            .unwrap_or_else(|| panic!("couldn't find a test with the provided name '{name}'"));
        let TestDescAndFn { desc, testfn } = test;
        match testfn.into_runnable() {
            Runnable::Test(runnable_test) => {
                if runnable_test.is_dynamic() {
                    panic!("only static tests are supported");
                }
                run_test_in_spawned_subprocess(desc, runnable_test);
            }
            Runnable::Bench(_) => {
                panic!("benchmarks should not be executed into child processes")
            }
        }
    }

    let args = env::args().collect::<Vec<_>>();
    let owned_tests: Vec<_> = tests.iter().map(make_owned_test).collect();
    test_main(&args, owned_tests, Some(Options::new().panic_abort(true)))
}

/// Clones static values for putting into a dynamic vector, which test_main()
/// needs to hand out ownership of tests to parallel test runners.
///
/// This will panic when fed any dynamic tests, because they cannot be cloned.
fn make_owned_test(test: &&TestDescAndFn) -> TestDescAndFn {
    match test.testfn {
        StaticTestFn(f) => TestDescAndFn { testfn: StaticTestFn(f), desc: test.desc.clone() },
        StaticBenchFn(f) => TestDescAndFn { testfn: StaticBenchFn(f), desc: test.desc.clone() },
        _ => panic!("non-static tests passed to test::test_main_static"),
    }
}

/// Invoked when unit tests terminate. Returns `Result::Err` if the test is
/// considered a failure. By default, invokes `report()` and checks for a `0`
/// result.
pub fn assert_test_result<T: Termination>(result: T) -> Result<(), String> {
    let code = result.report().to_i32();
    if code == 0 {
        Ok(())
    } else {
        Err(format!(
            "the test returned a termination value with a non-zero status code \
             ({code}) which indicates a failure"
        ))
    }
}

struct FilteredTests {
    tests: Vec<(TestId, TestDescAndFn)>,
    benches: Vec<(TestId, TestDescAndFn)>,
    next_id: usize,
}

impl FilteredTests {
    fn add_bench(&mut self, desc: TestDesc, testfn: TestFn) {
        let test = TestDescAndFn { desc, testfn };
        self.benches.push((TestId(self.next_id), test));
        self.next_id += 1;
    }
    fn add_test(&mut self, desc: TestDesc, testfn: TestFn) {
        let test = TestDescAndFn { desc, testfn };
        self.tests.push((TestId(self.next_id), test));
        self.next_id += 1;
    }
    fn total_len(&self) -> usize {
        self.tests.len() + self.benches.len()
    }
}

pub fn run_tests<F>(
    opts: &TestOpts,
    tests: Vec<TestDescAndFn>,
    mut notify_about_test_event: F,
) -> io::Result<()>
where
    F: FnMut(TestEvent) -> io::Result<()>,
{
    use std::collections::HashMap;
    use std::hash::{BuildHasherDefault, DefaultHasher};
    use std::sync::mpsc::RecvTimeoutError;

    struct RunningTest {
        join_handle: Option<thread::JoinHandle<()>>,
    }

    impl RunningTest {
        fn join(self, completed_test: &mut CompletedTest) {
            if let Some(join_handle) = self.join_handle {
                if let Err(_) = join_handle.join() {
                    if let TrOk = completed_test.result {
                        completed_test.result =
                            TrFailedMsg("panicked after reporting success".to_string());
                    }
                }
            }
        }
    }

    // Use a deterministic hasher
    type TestMap = HashMap<TestId, RunningTest, BuildHasherDefault<DefaultHasher>>;

    struct TimeoutEntry {
        id: TestId,
        desc: TestDesc,
        timeout: Instant,
    }

    let tests_len = tests.len();

    let mut filtered = FilteredTests { tests: Vec::new(), benches: Vec::new(), next_id: 0 };

    let mut filtered_tests = filter_tests(opts, tests);
    if !opts.bench_benchmarks {
        filtered_tests = convert_benchmarks_to_tests(filtered_tests);
    }

    for test in filtered_tests {
        let mut desc = test.desc;
        desc.name = desc.name.with_padding(test.testfn.padding());

        match test.testfn {
            DynBenchFn(_) | StaticBenchFn(_) => {
                filtered.add_bench(desc, test.testfn);
            }
            testfn => {
                filtered.add_test(desc, testfn);
            }
        };
    }

    let filtered_out = tests_len - filtered.total_len();
    let event = TestEvent::TeFilteredOut(filtered_out);
    notify_about_test_event(event)?;

    let shuffle_seed = get_shuffle_seed(opts);

    let event = TestEvent::TeFiltered(filtered.total_len(), shuffle_seed);
    notify_about_test_event(event)?;

    let concurrency = opts.test_threads.unwrap_or_else(get_concurrency);

    let mut remaining = filtered.tests;
    if let Some(shuffle_seed) = shuffle_seed {
        shuffle_tests(shuffle_seed, &mut remaining);
    }
    // Store the tests in a VecDeque so we can efficiently remove the first element to run the
    // tests in the order they were passed (unless shuffled).
    let mut remaining = VecDeque::from(remaining);
    let mut pending = 0;

    let (tx, rx) = channel::<CompletedTest>();
    let run_strategy = if opts.options.panic_abort && !opts.force_run_in_process {
        RunStrategy::SpawnPrimary
    } else {
        RunStrategy::InProcess
    };

    let mut running_tests: TestMap = HashMap::default();
    let mut timeout_queue: VecDeque<TimeoutEntry> = VecDeque::new();

    fn get_timed_out_tests(
        running_tests: &TestMap,
        timeout_queue: &mut VecDeque<TimeoutEntry>,
    ) -> Vec<TestDesc> {
        let now = Instant::now();
        let mut timed_out = Vec::new();
        while let Some(timeout_entry) = timeout_queue.front() {
            if now < timeout_entry.timeout {
                break;
            }
            let timeout_entry = timeout_queue.pop_front().unwrap();
            if running_tests.contains_key(&timeout_entry.id) {
                timed_out.push(timeout_entry.desc);
            }
        }
        timed_out
    }

    fn calc_timeout(timeout_queue: &VecDeque<TimeoutEntry>) -> Option<Duration> {
        timeout_queue.front().map(|&TimeoutEntry { timeout: next_timeout, .. }| {
            let now = Instant::now();
            if next_timeout >= now { next_timeout - now } else { Duration::new(0, 0) }
        })
    }

    if concurrency == 1 {
        while !remaining.is_empty() {
            let (id, test) = remaining.pop_front().unwrap();
            let event = TestEvent::TeWait(test.desc.clone());
            notify_about_test_event(event)?;
            let join_handle = run_test(opts, !opts.run_tests, id, test, run_strategy, tx.clone());
            // Wait for the test to complete.
            let mut completed_test = rx.recv().unwrap();
            RunningTest { join_handle }.join(&mut completed_test);

            let fail_fast = match completed_test.result {
                TrIgnored | TrOk | TrBench(_) => false,
                TrFailed | TrFailedMsg(_) | TrTimedFail => opts.fail_fast,
            };

            let event = TestEvent::TeResult(completed_test);
            notify_about_test_event(event)?;

            if fail_fast {
                return Ok(());
            }
        }
    } else {
        while pending > 0 || !remaining.is_empty() {
            while pending < concurrency && !remaining.is_empty() {
                let (id, test) = remaining.pop_front().unwrap();
                let timeout = time::get_default_test_timeout();
                let desc = test.desc.clone();

                let event = TestEvent::TeWait(desc.clone());
                notify_about_test_event(event)?; //here no pad
                let join_handle =
                    run_test(opts, !opts.run_tests, id, test, run_strategy, tx.clone());
                running_tests.insert(id, RunningTest { join_handle });
                timeout_queue.push_back(TimeoutEntry { id, desc, timeout });
                pending += 1;
            }

            let mut res;
            loop {
                if let Some(timeout) = calc_timeout(&timeout_queue) {
                    res = rx.recv_timeout(timeout);
                    for test in get_timed_out_tests(&running_tests, &mut timeout_queue) {
                        let event = TestEvent::TeTimeout(test);
                        notify_about_test_event(event)?;
                    }

                    match res {
                        Err(RecvTimeoutError::Timeout) => {
                            // Result is not yet ready, continue waiting.
                        }
                        _ => {
                            // We've got a result, stop the loop.
                            break;
                        }
                    }
                } else {
                    res = rx.recv().map_err(|_| RecvTimeoutError::Disconnected);
                    break;
                }
            }

            let mut completed_test = res.unwrap();
            let running_test = running_tests.remove(&completed_test.id).unwrap();
            running_test.join(&mut completed_test);

            let fail_fast = match completed_test.result {
                TrIgnored | TrOk | TrBench(_) => false,
                TrFailed | TrFailedMsg(_) | TrTimedFail => opts.fail_fast,
            };

            let event = TestEvent::TeResult(completed_test);
            notify_about_test_event(event)?;
            pending -= 1;

            if fail_fast {
                // Prevent remaining test threads from panicking
                std::mem::forget(rx);
                return Ok(());
            }
        }
    }

    if opts.bench_benchmarks {
        // All benchmarks run at the end, in serial.
        for (id, b) in filtered.benches {
            let event = TestEvent::TeWait(b.desc.clone());
            notify_about_test_event(event)?;
            let join_handle = run_test(opts, false, id, b, run_strategy, tx.clone());
            // Wait for the test to complete.
            let mut completed_test = rx.recv().unwrap();
            RunningTest { join_handle }.join(&mut completed_test);

            let event = TestEvent::TeResult(completed_test);
            notify_about_test_event(event)?;
        }
    }
    Ok(())
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
    if !opts.filters.is_empty() {
        filtered.retain(|test| opts.filters.iter().any(|filter| matches_filter(test, filter)));
    }

    // Skip tests that match any of the skip filters
    if !opts.skip.is_empty() {
        filtered.retain(|test| !opts.skip.iter().any(|sf| matches_filter(test, sf)));
    }

    // Excludes #[should_panic] tests
    if opts.exclude_should_panic {
        filtered.retain(|test| test.desc.should_panic == ShouldPanic::No);
    }

    // maybe unignore tests
    match opts.run_ignored {
        RunIgnored::Yes => {
            filtered.iter_mut().for_each(|test| test.desc.ignore = false);
        }
        RunIgnored::Only => {
            filtered.retain(|test| test.desc.ignore);
            filtered.iter_mut().for_each(|test| test.desc.ignore = false);
        }
        RunIgnored::No => {}
    }

    filtered
}

pub fn convert_benchmarks_to_tests(tests: Vec<TestDescAndFn>) -> Vec<TestDescAndFn> {
    // convert benchmarks to tests, if we're not benchmarking them
    tests
        .into_iter()
        .map(|x| {
            let testfn = match x.testfn {
                DynBenchFn(benchfn) => DynBenchAsTestFn(benchfn),
                StaticBenchFn(benchfn) => StaticBenchAsTestFn(benchfn),
                f => f,
            };
            TestDescAndFn { desc: x.desc, testfn }
        })
        .collect()
}

pub fn run_test(
    opts: &TestOpts,
    force_ignore: bool,
    id: TestId,
    test: TestDescAndFn,
    strategy: RunStrategy,
    monitor_ch: Sender<CompletedTest>,
) -> Option<thread::JoinHandle<()>> {
    let TestDescAndFn { desc, testfn } = test;

    // Emscripten can catch panics but other wasm targets cannot
    let ignore_because_no_process_support = desc.should_panic != ShouldPanic::No
        && (cfg!(target_family = "wasm") || cfg!(target_os = "zkvm"))
        && !cfg!(target_os = "emscripten");

    if force_ignore || desc.ignore || ignore_because_no_process_support {
        let message = CompletedTest::new(id, desc, TrIgnored, None, Vec::new());
        monitor_ch.send(message).unwrap();
        return None;
    }

    match testfn.into_runnable() {
        Runnable::Test(runnable_test) => {
            if runnable_test.is_dynamic() {
                match strategy {
                    RunStrategy::InProcess => (),
                    _ => panic!("Cannot run dynamic test fn out-of-process"),
                };
            }

            let name = desc.name.clone();
            let nocapture = opts.nocapture;
            let time_options = opts.time_options;
            let bench_benchmarks = opts.bench_benchmarks;

            let runtest = move || match strategy {
                RunStrategy::InProcess => run_test_in_process(
                    id,
                    desc,
                    nocapture,
                    time_options.is_some(),
                    runnable_test,
                    monitor_ch,
                    time_options,
                ),
                RunStrategy::SpawnPrimary => spawn_test_subprocess(
                    id,
                    desc,
                    nocapture,
                    time_options.is_some(),
                    monitor_ch,
                    time_options,
                    bench_benchmarks,
                ),
            };

            // If the platform is single-threaded we're just going to run
            // the test synchronously, regardless of the concurrency
            // level.
            let supports_threads = !cfg!(target_os = "emscripten")
                && !cfg!(target_family = "wasm")
                && !cfg!(target_os = "zkvm");
            if supports_threads {
                let cfg = thread::Builder::new().name(name.as_slice().to_owned());
                let mut runtest = Arc::new(Mutex::new(Some(runtest)));
                let runtest2 = runtest.clone();
                match cfg.spawn(move || runtest2.lock().unwrap().take().unwrap()()) {
                    Ok(handle) => Some(handle),
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                        // `ErrorKind::WouldBlock` means hitting the thread limit on some
                        // platforms, so run the test synchronously here instead.
                        Arc::get_mut(&mut runtest).unwrap().get_mut().unwrap().take().unwrap()();
                        None
                    }
                    Err(e) => panic!("failed to spawn thread to run test: {e}"),
                }
            } else {
                runtest();
                None
            }
        }
        Runnable::Bench(runnable_bench) => {
            // Benchmarks aren't expected to panic, so we run them all in-process.
            runnable_bench.run(id, &desc, &monitor_ch, opts.nocapture);
            None
        }
    }
}

/// Fixed frame used to clean the backtrace with `RUST_BACKTRACE=1`.
#[inline(never)]
fn __rust_begin_short_backtrace<T, F: FnOnce() -> T>(f: F) -> T {
    let result = f();

    // prevent this frame from being tail-call optimised away
    black_box(result)
}

fn run_test_in_process(
    id: TestId,
    desc: TestDesc,
    nocapture: bool,
    report_time: bool,
    runnable_test: RunnableTest,
    monitor_ch: Sender<CompletedTest>,
    time_opts: Option<time::TestTimeOptions>,
) {
    // Buffer for capturing standard I/O
    let data = Arc::new(Mutex::new(Vec::new()));

    if !nocapture {
        io::set_output_capture(Some(data.clone()));
    }

    let start = report_time.then(Instant::now);
    let result = fold_err(catch_unwind(AssertUnwindSafe(|| runnable_test.run())));
    let exec_time = start.map(|start| {
        let duration = start.elapsed();
        TestExecTime(duration)
    });

    io::set_output_capture(None);

    let test_result = match result {
        Ok(()) => calc_result(&desc, Ok(()), time_opts.as_ref(), exec_time.as_ref()),
        Err(e) => calc_result(&desc, Err(e.as_ref()), time_opts.as_ref(), exec_time.as_ref()),
    };
    let stdout = data.lock().unwrap_or_else(|e| e.into_inner()).to_vec();
    let message = CompletedTest::new(id, desc, test_result, exec_time, stdout);
    monitor_ch.send(message).unwrap();
}

fn fold_err<T, E>(
    result: Result<Result<T, E>, Box<dyn Any + Send>>,
) -> Result<T, Box<dyn Any + Send>>
where
    E: Send + 'static,
{
    match result {
        Ok(Err(e)) => Err(Box::new(e)),
        Ok(Ok(v)) => Ok(v),
        Err(e) => Err(e),
    }
}

fn spawn_test_subprocess(
    id: TestId,
    desc: TestDesc,
    nocapture: bool,
    report_time: bool,
    monitor_ch: Sender<CompletedTest>,
    time_opts: Option<time::TestTimeOptions>,
    bench_benchmarks: bool,
) {
    let (result, test_output, exec_time) = (|| {
        let args = env::args().collect::<Vec<_>>();
        let current_exe = &args[0];

        let mut command = Command::new(current_exe);
        command.env(SECONDARY_TEST_INVOKER_VAR, desc.name.as_slice());
        if bench_benchmarks {
            command.env(SECONDARY_TEST_BENCH_BENCHMARKS_VAR, "1");
        }
        if nocapture {
            command.stdout(process::Stdio::inherit());
            command.stderr(process::Stdio::inherit());
        }

        let start = report_time.then(Instant::now);
        let output = match command.output() {
            Ok(out) => out,
            Err(e) => {
                let err = format!("Failed to spawn {} as child for test: {:?}", args[0], e);
                return (TrFailed, err.into_bytes(), None);
            }
        };
        let exec_time = start.map(|start| {
            let duration = start.elapsed();
            TestExecTime(duration)
        });

        let std::process::Output { stdout, stderr, status } = output;
        let mut test_output = stdout;
        formatters::write_stderr_delimiter(&mut test_output, &desc.name);
        test_output.extend_from_slice(&stderr);

        let result =
            get_result_from_exit_code(&desc, status, time_opts.as_ref(), exec_time.as_ref());
        (result, test_output, exec_time)
    })();

    let message = CompletedTest::new(id, desc, result, exec_time, test_output);
    monitor_ch.send(message).unwrap();
}

fn run_test_in_spawned_subprocess(desc: TestDesc, runnable_test: RunnableTest) -> ! {
    let builtin_panic_hook = panic::take_hook();
    let record_result = Arc::new(move |panic_info: Option<&'_ PanicHookInfo<'_>>| {
        let test_result = match panic_info {
            Some(info) => calc_result(&desc, Err(info.payload()), None, None),
            None => calc_result(&desc, Ok(()), None, None),
        };

        // We don't support serializing TrFailedMsg, so just
        // print the message out to stderr.
        if let TrFailedMsg(msg) = &test_result {
            eprintln!("{msg}");
        }

        if let Some(info) = panic_info {
            builtin_panic_hook(info);
        }

        if let TrOk = test_result {
            process::exit(test_result::TR_OK);
        } else {
            process::abort();
        }
    });
    let record_result2 = record_result.clone();
    panic::set_hook(Box::new(move |info| record_result2(Some(info))));
    if let Err(message) = runnable_test.run() {
        panic!("{}", message);
    }
    record_result(None);
    unreachable!("panic=abort callback should have exited the process")
}
