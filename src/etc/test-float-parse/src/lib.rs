mod traits;
mod ui;
mod validate;

use std::any::{TypeId, type_name};
use std::cmp::min;
use std::ops::RangeInclusive;
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{OnceLock, mpsc};
use std::{fmt, time};

use indicatif::{MultiProgress, ProgressBar};
use rand::distributions::{Distribution, Standard};
use rayon::prelude::*;
use time::{Duration, Instant};
use traits::{Float, Generator, Int};

/// Test generators.
mod gen {
    pub mod exhaustive;
    pub mod exponents;
    pub mod fuzz;
    pub mod integers;
    pub mod long_fractions;
    pub mod many_digits;
    pub mod sparse;
    pub mod spot_checks;
    pub mod subnorm;
}

/// How many failures to exit after if unspecified.
const DEFAULT_MAX_FAILURES: u64 = 20;

/// Register exhaustive tests only for <= 32 bits. No more because it would take years.
const MAX_BITS_FOR_EXHAUUSTIVE: u32 = 32;

/// If there are more tests than this threashold, the test will be defered until after all
/// others run (so as to avoid thread pool starvation). They also can be excluded with
/// `--skip-huge`.
const HUGE_TEST_CUTOFF: u64 = 5_000_000;

/// Seed for tests that use a deterministic RNG.
const SEED: [u8; 32] = *b"3.141592653589793238462643383279";

/// Global configuration
#[derive(Debug)]
pub struct Config {
    pub timeout: Duration,
    /// Failures per test
    pub max_failures: u64,
    pub disable_max_failures: bool,
    /// If `None`, the default will be used
    pub fuzz_count: Option<u64>,
    pub skip_huge: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(60 * 60 * 3),
            max_failures: DEFAULT_MAX_FAILURES,
            disable_max_failures: false,
            fuzz_count: None,
            skip_huge: false,
        }
    }
}

/// Collect, filter, and launch all tests.
pub fn run(cfg: Config, include: &[String], exclude: &[String]) -> ExitCode {
    // With default parallelism, the CPU doesn't saturate. We don't need to be nice to
    // other processes, so do 1.5x to make sure we use all available resources.
    let threads = std::thread::available_parallelism().map(Into::into).unwrap_or(0) * 3 / 2;
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();

    let mut tests = register_tests(&cfg);
    println!("registered");
    let initial_tests: Vec<_> = tests.iter().map(|t| t.name.clone()).collect();

    let unmatched: Vec<_> = include
        .iter()
        .chain(exclude.iter())
        .filter(|filt| !tests.iter().any(|t| t.matches(filt)))
        .collect();

    assert!(
        unmatched.is_empty(),
        "filters were provided that have no matching tests: {unmatched:#?}"
    );

    tests.retain(|test| !exclude.iter().any(|exc| test.matches(exc)));

    if cfg.skip_huge {
        tests.retain(|test| !test.is_huge_test());
    }

    if !include.is_empty() {
        tests.retain(|test| include.iter().any(|inc| test.matches(inc)));
    }

    for exc in initial_tests.iter().filter(|orig_name| !tests.iter().any(|t| t.name == **orig_name))
    {
        println!("Skipping test '{exc}'");
    }

    println!("launching");
    let elapsed = launch_tests(&mut tests, &cfg);
    ui::finish(&tests, elapsed, &cfg)
}

/// Enumerate tests to run but don't actaully run them.
pub fn register_tests(cfg: &Config) -> Vec<TestInfo> {
    let mut tests = Vec::new();

    // Register normal generators for all floats.
    register_float::<f32>(&mut tests, cfg);
    register_float::<f64>(&mut tests, cfg);

    tests.sort_unstable_by_key(|t| (t.float_name, t.gen_name));
    for i in 0..(tests.len() - 1) {
        if tests[i].gen_name == tests[i + 1].gen_name {
            panic!("dupliate test name {}", tests[i].gen_name);
        }
    }

    tests
}

/// Register all generators for a single float.
fn register_float<F: Float>(tests: &mut Vec<TestInfo>, cfg: &Config)
where
    RangeInclusive<F::Int>: Iterator<Item = F::Int>,
    <F::Int as TryFrom<u128>>::Error: std::fmt::Debug,
    Standard: Distribution<<F as traits::Float>::Int>,
{
    if F::BITS <= MAX_BITS_FOR_EXHAUUSTIVE {
        // Only run exhaustive tests if there is a chance of completion.
        TestInfo::register::<F, gen::exhaustive::Exhaustive<F>>(tests);
    }

    gen::fuzz::Fuzz::<F>::set_iterations(cfg.fuzz_count);

    TestInfo::register::<F, gen::exponents::LargeExponents<F>>(tests);
    TestInfo::register::<F, gen::exponents::SmallExponents<F>>(tests);
    TestInfo::register::<F, gen::fuzz::Fuzz<F>>(tests);
    TestInfo::register::<F, gen::integers::LargeInt<F>>(tests);
    TestInfo::register::<F, gen::integers::SmallInt>(tests);
    TestInfo::register::<F, gen::long_fractions::RepeatingDecimal>(tests);
    TestInfo::register::<F, gen::many_digits::RandDigits<F>>(tests);
    TestInfo::register::<F, gen::sparse::FewOnesFloat<F>>(tests);
    TestInfo::register::<F, gen::sparse::FewOnesInt<F>>(tests);
    TestInfo::register::<F, gen::spot_checks::RegressionCheck>(tests);
    TestInfo::register::<F, gen::spot_checks::Special>(tests);
    TestInfo::register::<F, gen::subnorm::SubnormComplete<F>>(tests);
    TestInfo::register::<F, gen::subnorm::SubnormEdgeCases<F>>(tests);
}

/// Configuration for a single test.
#[derive(Debug)]
pub struct TestInfo {
    pub name: String,
    /// Tests are identified by the type ID of `(F, G)` (tuple of the float and generator type).
    /// This gives an easy way to associate messages with tests.
    id: TypeId,
    float_name: &'static str,
    gen_name: &'static str,
    /// Name for display in the progress bar.
    short_name: String,
    total_tests: u64,
    /// Function to launch this test.
    launch: fn(&mpsc::Sender<Msg>, &TestInfo, &Config),
    /// Progress bar to be updated.
    pb: Option<ProgressBar>,
    /// Once completed, this will be set.
    completed: OnceLock<Completed>,
}

impl TestInfo {
    /// Check if either the name or short name is a match, for filtering.
    fn matches(&self, pat: &str) -> bool {
        self.short_name.contains(pat) || self.name.contains(pat)
    }

    /// Create a `TestInfo` for a given float and generator, then add it to a list.
    fn register<F: Float, G: Generator<F>>(v: &mut Vec<Self>) {
        let f_name = type_name::<F>();
        let gen_name = G::NAME;
        let gen_short_name = G::SHORT_NAME;

        let info = TestInfo {
            id: TypeId::of::<(F, G)>(),
            float_name: f_name,
            gen_name,
            pb: None,
            name: format!("{f_name} {gen_name}"),
            short_name: format!("{f_name} {gen_short_name}"),
            launch: test_runner::<F, G>,
            total_tests: G::total_tests(),
            completed: OnceLock::new(),
        };
        v.push(info);
    }

    /// Pad the short name to a common width for progress bar use.
    fn short_name_padded(&self) -> String {
        format!("{:18}", self.short_name)
    }

    /// Create a progress bar for this test within a multiprogress bar.
    fn register_pb(&mut self, mp: &MultiProgress, drop_bars: &mut Vec<ProgressBar>) {
        self.pb = Some(ui::create_pb(mp, self.total_tests, &self.short_name_padded(), drop_bars));
    }

    /// When the test is finished, update progress bar messages and finalize.
    fn finalize_pb(&self, c: &Completed) {
        let pb = self.pb.as_ref().unwrap();
        ui::finalize_pb(pb, &self.short_name_padded(), c);
    }

    /// True if this should be run after all others.
    fn is_huge_test(&self) -> bool {
        self.total_tests >= HUGE_TEST_CUTOFF
    }
}

/// A message sent from test runner threads to the UI/log thread.
#[derive(Clone, Debug)]
struct Msg {
    id: TypeId,
    update: Update,
}

impl Msg {
    /// Wrap an `Update` into a message for the specified type. We use the `TypeId` of `(F, G)` to
    /// identify which test a message in the channel came from.
    fn new<F: Float, G: Generator<F>>(u: Update) -> Self {
        Self { id: TypeId::of::<(F, G)>(), update: u }
    }

    /// Get the matching test from a list. Panics if not found.
    fn find_test<'a>(&self, tests: &'a [TestInfo]) -> &'a TestInfo {
        tests.iter().find(|t| t.id == self.id).unwrap()
    }

    /// Update UI as needed for a single message received from the test runners.
    fn handle(self, tests: &[TestInfo], mp: &MultiProgress) {
        let test = self.find_test(tests);
        let pb = test.pb.as_ref().unwrap();

        match self.update {
            Update::Started => {
                mp.println(format!("Testing '{}'", test.name)).unwrap();
            }
            Update::Progress { executed, failures } => {
                pb.set_message(format! {"{failures}"});
                pb.set_position(executed);
            }
            Update::Failure { fail, input, float_res } => {
                mp.println(format!(
                    "Failure in '{}': {fail}. parsing '{input}'. Parsed as: {float_res}",
                    test.name
                ))
                .unwrap();
            }
            Update::Completed(c) => {
                test.finalize_pb(&c);

                let prefix = match c.result {
                    Ok(FinishedAll) => "Completed tests for",
                    Err(EarlyExit::Timeout) => "Timed out",
                    Err(EarlyExit::MaxFailures) => "Max failures reached for",
                };

                mp.println(format!(
                    "{prefix} generator '{}' in {:?}. {} tests run, {} failures",
                    test.name, c.elapsed, c.executed, c.failures
                ))
                .unwrap();
                test.completed.set(c).unwrap();
            }
        };
    }
}

/// Status sent with a message.
#[derive(Clone, Debug)]
enum Update {
    /// Starting a new test runner.
    Started,
    /// Completed a out of b tests.
    Progress { executed: u64, failures: u64 },
    /// Received a failed test.
    Failure {
        fail: CheckFailure,
        /// String for which parsing was attempted.
        input: Box<str>,
        /// The parsed & decomposed `FloatRes`, aleady stringified so we don't need generics here.
        float_res: Box<str>,
    },
    /// Exited with an unexpected condition.
    Completed(Completed),
}

/// Result of an input did not parsing successfully.
#[derive(Clone, Debug)]
enum CheckFailure {
    /// Above the zero cutoff but got rounded to zero.
    UnexpectedZero,
    /// Below the infinity cutoff but got rounded to infinity.
    UnexpectedInf,
    /// Above the negative infinity cutoff but got rounded to negative infinity.
    UnexpectedNegInf,
    /// Got a `NaN` when none was expected.
    UnexpectedNan,
    /// Expected `NaN`, got none.
    ExpectedNan,
    /// Expected infinity, got finite.
    ExpectedInf,
    /// Expected negative infinity, got finite.
    ExpectedNegInf,
    /// The value exceeded its error tolerance.
    InvalidReal {
        /// Error from the expected value, as a float.
        error_float: Option<f64>,
        /// Error as a rational string (since it can't always be represented as a float).
        error_str: Box<str>,
        /// True if the error was caused by not rounding to even at the midpoint between
        /// two representable values.
        incorrect_midpoint_rounding: bool,
    },
}

impl fmt::Display for CheckFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CheckFailure::UnexpectedZero => {
                write!(f, "incorrectly rounded to 0 (expected nonzero)")
            }
            CheckFailure::UnexpectedInf => {
                write!(f, "incorrectly rounded to +inf (expected finite)")
            }
            CheckFailure::UnexpectedNegInf => {
                write!(f, "incorrectly rounded to -inf (expected finite)")
            }
            CheckFailure::UnexpectedNan => write!(f, "got a NaN where none was expected"),
            CheckFailure::ExpectedNan => write!(f, "expected a NaN but did not get it"),
            CheckFailure::ExpectedInf => write!(f, "expected +inf but did not get it"),
            CheckFailure::ExpectedNegInf => write!(f, "expected -inf but did not get it"),
            CheckFailure::InvalidReal { error_float, error_str, incorrect_midpoint_rounding } => {
                if *incorrect_midpoint_rounding {
                    write!(
                        f,
                        "midpoint between two representable values did not correctly \
                        round to even; error: {error_str}"
                    )?;
                } else {
                    write!(f, "real number did not parse correctly; error: {error_str}")?;
                }

                if let Some(float) = error_float {
                    write!(f, " ({float})")?;
                }
                Ok(())
            }
        }
    }
}

/// Information about a completed test generator.
#[derive(Clone, Debug)]
struct Completed {
    /// Finished tests (both successful and failed).
    executed: u64,
    /// Failed tests.
    failures: u64,
    /// Extra exit information if unsuccessful.
    result: Result<FinishedAll, EarlyExit>,
    /// If there is something to warn about (e.g bad estimate), leave it here.
    warning: Option<Box<str>>,
    /// Total time to run the test.
    elapsed: Duration,
}

/// Marker for completing all tests (used in `Result` types).
#[derive(Clone, Debug)]
struct FinishedAll;

/// Reasons for exiting early.
#[derive(Clone, Debug)]
enum EarlyExit {
    Timeout,
    MaxFailures,
}

/// Run all tests in `tests`.
///
/// This launches a main thread that receives messages and handlees UI updates, and uses the
/// rest of the thread pool to execute the tests.
fn launch_tests(tests: &mut [TestInfo], cfg: &Config) -> Duration {
    // Run shorter tests first
    tests.sort_unstable_by_key(|test| test.total_tests);

    for test in tests.iter() {
        println!("Launching test '{}'", test.name);
    }

    // Configure progress bars
    let mut all_progress_bars = Vec::new();
    let mp = MultiProgress::new();
    mp.set_move_cursor(true);
    for test in tests.iter_mut() {
        test.register_pb(&mp, &mut all_progress_bars);
    }

    ui::set_panic_hook(all_progress_bars);

    let (tx, rx) = mpsc::channel::<Msg>();
    let start = Instant::now();

    rayon::scope(|scope| {
        // Thread that updates the UI
        scope.spawn(|_scope| {
            let rx = rx; // move rx

            loop {
                if tests.iter().all(|t| t.completed.get().is_some()) {
                    break;
                }

                let msg = rx.recv().unwrap();
                msg.handle(tests, &mp);
            }

            // All tests completed; finish things up
            drop(mp);
            assert_eq!(rx.try_recv().unwrap_err(), mpsc::TryRecvError::Empty);
        });

        // Don't let the thread pool be starved by huge tests. Run faster tests first in parallel,
        // then parallelize only within the rest of the tests.
        let (huge_tests, normal_tests): (Vec<_>, Vec<_>) =
            tests.iter().partition(|t| t.is_huge_test());

        // Run the actual tests
        normal_tests.par_iter().for_each(|test| ((test.launch)(&tx, test, cfg)));

        huge_tests.par_iter().for_each(|test| ((test.launch)(&tx, test, cfg)));
    });

    start.elapsed()
}

/// Test runer for a single generator.
///
/// This calls the generator's iterator multiple times (in parallel) and validates each output.
fn test_runner<F: Float, G: Generator<F>>(tx: &mpsc::Sender<Msg>, _info: &TestInfo, cfg: &Config) {
    tx.send(Msg::new::<F, G>(Update::Started)).unwrap();

    let total = G::total_tests();
    let gen = G::new();
    let executed = AtomicU64::new(0);
    let failures = AtomicU64::new(0);

    let checks_per_update = min(total, 1000);
    let started = Instant::now();

    // Function to execute for a single test iteration.
    let check_one = |buf: &mut String, ctx: G::WriteCtx| {
        let executed = executed.fetch_add(1, Ordering::Relaxed);
        buf.clear();
        G::write_string(buf, ctx);

        match validate::validate::<F>(buf) {
            Ok(()) => (),
            Err(e) => {
                tx.send(Msg::new::<F, G>(e)).unwrap();
                let f = failures.fetch_add(1, Ordering::Relaxed);
                // End early if the limit is exceeded.
                if f >= cfg.max_failures {
                    return Err(EarlyExit::MaxFailures);
                }
            }
        };

        // Send periodic updates
        if executed % checks_per_update == 0 {
            let failures = failures.load(Ordering::Relaxed);

            tx.send(Msg::new::<F, G>(Update::Progress { executed, failures })).unwrap();

            if started.elapsed() > cfg.timeout {
                return Err(EarlyExit::Timeout);
            }
        }

        Ok(())
    };

    // Run the test iterations in parallel. Each thread gets a string buffer to write
    // its check values to.
    let res = gen.par_bridge().try_for_each_init(|| String::with_capacity(100), check_one);

    let elapsed = started.elapsed();
    let executed = executed.into_inner();
    let failures = failures.into_inner();

    // Warn about bad estimates if relevant.
    let warning = if executed != total && res.is_ok() {
        let msg = format!("executed tests != estimated ({executed} != {total}) for {}", G::NAME);

        Some(msg.into())
    } else {
        None
    };

    let result = res.map(|()| FinishedAll);
    tx.send(Msg::new::<F, G>(Update::Completed(Completed {
        executed,
        failures,
        result,
        warning,
        elapsed,
    })))
    .unwrap();
}
