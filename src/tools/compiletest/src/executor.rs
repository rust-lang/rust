//! This module contains a reimplementation of the subset of libtest
//! functionality needed by compiletest.

use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, DefaultHasher};
use std::num::NonZero;
use std::sync::{Arc, Mutex, mpsc};
use std::{env, hint, io, mem, panic, thread};

use crate::common::{Config, TestPaths};

mod deadline;
mod json;
pub(crate) mod libtest;

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
        // FIXME(let_chains): Use a let-chain here when stable in bootstrap.
        'spawn: while running_tests.len() < concurrency {
            let Some((id, test)) = fresh_tests.next() else { break 'spawn };
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
    completion_tx: mpsc::Sender<TestCompletion>,
) -> Option<thread::JoinHandle<()>> {
    if test.desc.ignore && !test.config.run_ignored {
        completion_tx
            .send(TestCompletion { id, outcome: TestOutcome::Ignored, stdout: None })
            .unwrap();
        return None;
    }

    let runnable_test = RunnableTest::new(test);
    let should_panic = test.desc.should_panic;
    let run_test = move || run_test_inner(id, should_panic, runnable_test, completion_tx);

    let thread_builder = thread::Builder::new().name(test.desc.name.clone());
    let join_handle = thread_builder.spawn(run_test).unwrap();
    Some(join_handle)
}

/// Runs a single test, within the dedicated thread spawned by the caller.
fn run_test_inner(
    id: TestId,
    should_panic: ShouldPanic,
    runnable_test: RunnableTest,
    completion_sender: mpsc::Sender<TestCompletion>,
) {
    let is_capture = !runnable_test.config.nocapture;
    let capture_buf = is_capture.then(|| Arc::new(Mutex::new(vec![])));

    if let Some(capture_buf) = &capture_buf {
        io::set_output_capture(Some(Arc::clone(capture_buf)));
    }

    let panic_payload = panic::catch_unwind(move || runnable_test.run()).err();

    if is_capture {
        io::set_output_capture(None);
    }

    let outcome = match (should_panic, panic_payload) {
        (ShouldPanic::No, None) | (ShouldPanic::Yes, Some(_)) => TestOutcome::Succeeded,
        (ShouldPanic::No, Some(_)) => TestOutcome::Failed { message: None },
        (ShouldPanic::Yes, None) => {
            TestOutcome::Failed { message: Some("test did not panic as expected") }
        }
    };
    let stdout = capture_buf.map(|mutex| mutex.lock().unwrap_or_else(|e| e.into_inner()).to_vec());

    completion_sender.send(TestCompletion { id, outcome, stdout }).unwrap();
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TestId(usize);

struct RunnableTest {
    config: Arc<Config>,
    testpaths: TestPaths,
    revision: Option<String>,
}

impl RunnableTest {
    fn new(test: &CollectedTest) -> Self {
        let config = Arc::clone(&test.config);
        let testpaths = test.testpaths.clone();
        let revision = test.revision.clone();
        Self { config, testpaths, revision }
    }

    fn run(&self) {
        __rust_begin_short_backtrace(|| {
            crate::runtest::run(
                Arc::clone(&self.config),
                &self.testpaths,
                self.revision.as_deref(),
            );
        });
    }
}

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
/// FIXME(#139660): After the libtest dependency is removed, redesign the whole
/// filtering system to do a better job of understanding and filtering _paths_,
/// instead of being tied to libtest's substring/exact matching behaviour.
fn filter_tests(opts: &Config, tests: Vec<CollectedTest>) -> Vec<CollectedTest> {
    let mut filtered = tests;

    let matches_filter = |test: &CollectedTest, filter_str: &str| {
        let test_name = &test.desc.name;
        if opts.filter_exact { test_name == filter_str } else { test_name.contains(filter_str) }
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
/// FIXME(#139660): After the libtest dependency is removed, consider making
/// bootstrap specify the number of threads on the command-line, instead of
/// propagating the `RUST_TEST_THREADS` environment variable.
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

/// Information needed to create a `test::TestDescAndFn`.
pub(crate) struct CollectedTest {
    pub(crate) desc: CollectedTestDesc,
    pub(crate) config: Arc<Config>,
    pub(crate) testpaths: TestPaths,
    pub(crate) revision: Option<String>,
}

/// Information needed to create a `test::TestDesc`.
pub(crate) struct CollectedTestDesc {
    pub(crate) name: String,
    pub(crate) ignore: bool,
    pub(crate) ignore_message: Option<Cow<'static, str>>,
    pub(crate) should_panic: ShouldPanic,
}

/// Whether console output should be colored or not.
#[derive(Copy, Clone, Default, Debug)]
pub enum ColorConfig {
    #[default]
    AutoColor,
    AlwaysColor,
    NeverColor,
}

/// Format of the test results output.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum OutputFormat {
    /// Verbose output
    Pretty,
    /// Quiet output
    #[default]
    Terse,
    /// JSON output
    Json,
}

/// Whether test is expected to panic or not.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ShouldPanic {
    No,
    Yes,
}
