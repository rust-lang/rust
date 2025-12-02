//! This module contains a reimplementation of the subset of libtest
//! functionality needed by compiletest.
//!
//! FIXME(Zalathar): Much of this code was originally designed to mimic libtest
//! as closely as possible, for ease of migration. Now that libtest is no longer
//! used, we can potentially redesign things to be a better fit for compiletest.

use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, DefaultHasher};
use std::num::NonZero;
use std::sync::{Arc, mpsc};
use std::{env, hint, mem, panic, thread};

use camino::Utf8PathBuf;

use crate::common::{Config, TestPaths};
use crate::output_capture::{self, ConsoleOut};
use crate::panic_hook;

mod deadline;
mod json;

pub(crate) fn run_tests(config: &Config, tests: Vec<CollectedTest>) -> bool {
    let tests_len = tests.len();
    let filtered = filter_tests(config, tests);
    // Iterator yielding tests that haven't been started yet.
    let mut fresh_tests = (0..).map(TestId).zip(&filtered);

    let concurrency = get_concurrency();
    assert!(concurrency > 0);
    let concurrent_capacity = concurrency.min(filtered.len());

    let mut listener = json::Listener::new();
    let mut running_tests = HashMap::with_capacity_and_hasher(
        concurrent_capacity,
        BuildHasherDefault::<DefaultHasher>::new(),
    );
    let mut deadline_queue = deadline::DeadlineQueue::with_capacity(concurrent_capacity);

    let num_filtered_out = tests_len - filtered.len();
    listener.suite_started(filtered.len(), num_filtered_out);

    // Channel used by test threads to report the test outcome when done.
    let (completion_tx, completion_rx) = mpsc::channel::<TestCompletion>();

    // Unlike libtest, we don't have a separate code path for concurrency=1.
    // In that case, the tests will effectively be run serially anyway.
    loop {
        // Spawn new test threads, up to the concurrency limit.
        while running_tests.len() < concurrency
            && let Some((id, test)) = fresh_tests.next()
        {
            listener.test_started(test);
            deadline_queue.push(id, test);
            let join_handle = spawn_test_thread(id, test, completion_tx.clone());
            running_tests.insert(id, RunningTest { test, join_handle });
        }

        // If all running tests have finished, and there weren't any unstarted
        // tests to spawn, then we're done.
        if running_tests.is_empty() {
            break;
        }

        let completion = deadline_queue
            .read_channel_while_checking_deadlines(
                &completion_rx,
                |id| running_tests.contains_key(&id),
                |_id, test| listener.test_timed_out(test),
            )
            .expect("receive channel should never be closed early");

        let RunningTest { test, join_handle } = running_tests.remove(&completion.id).unwrap();
        if let Some(join_handle) = join_handle {
            join_handle.join().unwrap_or_else(|_| {
                panic!("thread for `{}` panicked after reporting completion", test.desc.name)
            });
        }

        listener.test_finished(test, &completion);

        if completion.outcome.is_failed() && config.fail_fast {
            // Prevent any other in-flight threads from panicking when they
            // write to the completion channel.
            mem::forget(completion_rx);
            break;
        }
    }

    let suite_passed = listener.suite_finished();
    suite_passed
}

/// Spawns a thread to run a single test, and returns the thread's join handle.
///
/// Returns `None` if the test was ignored, so no thread was spawned.
fn spawn_test_thread(
    id: TestId,
    test: &CollectedTest,
    completion_sender: mpsc::Sender<TestCompletion>,
) -> Option<thread::JoinHandle<()>> {
    if test.desc.ignore && !test.config.run_ignored {
        completion_sender
            .send(TestCompletion { id, outcome: TestOutcome::Ignored, stdout: None })
            .unwrap();
        return None;
    }

    let args = TestThreadArgs {
        id,
        config: Arc::clone(&test.config),
        testpaths: test.testpaths.clone(),
        revision: test.revision.clone(),
        should_fail: test.desc.should_fail,
        completion_sender,
    };
    let thread_builder = thread::Builder::new().name(test.desc.name.clone());
    let join_handle = thread_builder.spawn(move || test_thread_main(args)).unwrap();
    Some(join_handle)
}

/// All of the owned data needed by `test_thread_main`.
struct TestThreadArgs {
    id: TestId,

    config: Arc<Config>,
    testpaths: TestPaths,
    revision: Option<String>,
    should_fail: ShouldFail,

    completion_sender: mpsc::Sender<TestCompletion>,
}

/// Runs a single test, within the dedicated thread spawned by the caller.
fn test_thread_main(args: TestThreadArgs) {
    let capture = CaptureKind::for_config(&args.config);

    // Install a panic-capture buffer for use by the custom panic hook.
    if capture.should_set_panic_hook() {
        panic_hook::set_capture_buf(Default::default());
    }

    let stdout = capture.stdout();
    let stderr = capture.stderr();

    // Run the test, catching any panics so that we can gracefully report
    // failure (or success).
    //
    // FIXME(Zalathar): Ideally we would report test failures with `Result`,
    // and use panics only for bugs within compiletest itself, but that would
    // require a major overhaul of error handling in the test runners.
    let panic_payload = panic::catch_unwind(|| {
        __rust_begin_short_backtrace(|| {
            crate::runtest::run(
                &args.config,
                stdout,
                stderr,
                &args.testpaths,
                args.revision.as_deref(),
            );
        });
    })
    .err();

    if let Some(panic_buf) = panic_hook::take_capture_buf() {
        let panic_buf = panic_buf.lock().unwrap_or_else(|e| e.into_inner());
        // Forward any captured panic message to (captured) stderr.
        write!(stderr, "{panic_buf}");
    }

    // Interpret the presence/absence of a panic as test failure/success.
    let outcome = match (args.should_fail, panic_payload) {
        (ShouldFail::No, None) | (ShouldFail::Yes, Some(_)) => TestOutcome::Succeeded,
        (ShouldFail::No, Some(_)) => TestOutcome::Failed { message: None },
        (ShouldFail::Yes, None) => {
            TestOutcome::Failed { message: Some("`//@ should-fail` test did not fail as expected") }
        }
    };

    let stdout = capture.into_inner();
    args.completion_sender.send(TestCompletion { id: args.id, outcome, stdout }).unwrap();
}

enum CaptureKind {
    /// Do not capture test-runner output, for `--no-capture`.
    ///
    /// (This does not affect `rustc` and other subprocesses spawned by test
    /// runners, whose output is always captured.)
    None,

    /// Capture all console output that would be printed by test runners via
    /// their `stdout` and `stderr` trait objects, or via the custom panic hook.
    Capture { buf: output_capture::CaptureBuf },
}

impl CaptureKind {
    fn for_config(config: &Config) -> Self {
        if config.nocapture {
            Self::None
        } else {
            Self::Capture { buf: output_capture::CaptureBuf::new() }
        }
    }

    fn should_set_panic_hook(&self) -> bool {
        match self {
            Self::None => false,
            Self::Capture { .. } => true,
        }
    }

    fn stdout(&self) -> &dyn ConsoleOut {
        self.capture_buf_or(&output_capture::Stdout)
    }

    fn stderr(&self) -> &dyn ConsoleOut {
        self.capture_buf_or(&output_capture::Stderr)
    }

    fn capture_buf_or<'a>(&'a self, fallback: &'a dyn ConsoleOut) -> &'a dyn ConsoleOut {
        match self {
            Self::None => fallback,
            Self::Capture { buf } => buf,
        }
    }

    fn into_inner(self) -> Option<Vec<u8>> {
        match self {
            Self::None => None,
            Self::Capture { buf } => Some(buf.into_inner().into()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TestId(usize);

/// Fixed frame used to clean the backtrace with `RUST_BACKTRACE=1`.
#[inline(never)]
fn __rust_begin_short_backtrace<T, F: FnOnce() -> T>(f: F) -> T {
    let result = f();

    // prevent this frame from being tail-call optimised away
    hint::black_box(result)
}

struct RunningTest<'a> {
    test: &'a CollectedTest,
    join_handle: Option<thread::JoinHandle<()>>,
}

/// Test completion message sent by individual test threads when their test
/// finishes (successfully or unsuccessfully).
struct TestCompletion {
    id: TestId,
    outcome: TestOutcome,
    stdout: Option<Vec<u8>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum TestOutcome {
    Succeeded,
    Failed { message: Option<&'static str> },
    Ignored,
}

impl TestOutcome {
    fn is_failed(&self) -> bool {
        matches!(self, Self::Failed { .. })
    }
}

/// Applies command-line arguments for filtering/skipping tests by name.
///
/// Adapted from `filter_tests` in libtest.
///
/// FIXME(#139660): Now that libtest has been removed, redesign the whole filtering system to
/// do a better job of understanding and filtering _paths_, instead of being tied to libtest's
/// substring/exact matching behaviour.
fn filter_tests(opts: &Config, tests: Vec<CollectedTest>) -> Vec<CollectedTest> {
    let mut filtered = tests;

    let matches_filter = |test: &CollectedTest, filter_str: &str| {
        if opts.filter_exact {
            // When `--exact` is used we must use `filterable_path` to get
            // reasonable filtering behavior.
            test.desc.filterable_path.as_str() == filter_str
        } else {
            // For compatibility we use the name (which includes the full path)
            // if `--exact` is not used.
            test.desc.name.contains(filter_str)
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

    filtered
}

/// Determines the number of tests to run concurrently.
///
/// Copied from `get_concurrency` in libtest.
///
/// FIXME(#139660): After the libtest dependency is removed, consider making bootstrap specify the
/// number of threads on the command-line, instead of propagating the `RUST_TEST_THREADS`
/// environment variable.
fn get_concurrency() -> usize {
    if let Ok(value) = env::var("RUST_TEST_THREADS") {
        match value.parse::<NonZero<usize>>().ok() {
            Some(n) => n.get(),
            _ => panic!("RUST_TEST_THREADS is `{value}`, should be a positive integer."),
        }
    } else {
        thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    }
}

/// Information that was historically needed to create a libtest `TestDescAndFn`.
pub(crate) struct CollectedTest {
    pub(crate) desc: CollectedTestDesc,
    pub(crate) config: Arc<Config>,
    pub(crate) testpaths: TestPaths,
    pub(crate) revision: Option<String>,
}

/// Information that was historically needed to create a libtest `TestDesc`.
pub(crate) struct CollectedTestDesc {
    pub(crate) name: String,
    pub(crate) filterable_path: Utf8PathBuf,
    pub(crate) ignore: bool,
    pub(crate) ignore_message: Option<Cow<'static, str>>,
    pub(crate) should_fail: ShouldFail,
}

/// Whether console output should be colored or not.
#[derive(Copy, Clone, Default, Debug)]
pub enum ColorConfig {
    #[default]
    AutoColor,
    AlwaysColor,
    NeverColor,
}

/// Tests with `//@ should-fail` are tests of compiletest itself, and should
/// be reported as successful if and only if they would have _failed_.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ShouldFail {
    No,
    Yes,
}
