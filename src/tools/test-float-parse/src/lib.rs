#![feature(f16)]
#![feature(cfg_target_has_reliable_f16_f128)]
#![expect(internal_features)] // reliable_f16_f128

mod traits;
mod ui;
mod validate;

use std::any::type_name;
use std::cmp::min;
use std::ops::RangeInclusive;
use std::process::ExitCode;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::{fmt, time};

use rand::distr::{Distribution, StandardUniform};
use rayon::prelude::*;
use time::{Duration, Instant};
use traits::{Float, Generator, Int};
use validate::CheckError;

/// Test generators.
mod gen_ {
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

/// If there are more tests than this threshold, the test will be deferred until after all
/// others run (so as to avoid thread pool starvation). They also can be excluded with
/// `--skip-huge`.
const HUGE_TEST_CUTOFF: u64 = 5_000_000;

/// Seed for tests that use a deterministic RNG.
const SEED: [u8; 32] = *b"3.141592653589793238462643383279";

/// Global configuration.
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

    println!("Launching all");
    let elapsed = launch_tests(&mut tests, &cfg);
    ui::finish_all(&tests, elapsed, &cfg)
}

/// Enumerate tests to run but don't actually run them.
pub fn register_tests(cfg: &Config) -> Vec<TestInfo> {
    let mut tests = Vec::new();

    // Register normal generators for all floats.

    #[cfg(target_has_reliable_f16)]
    register_float::<f16>(&mut tests, cfg);
    register_float::<f32>(&mut tests, cfg);
    register_float::<f64>(&mut tests, cfg);

    tests.sort_unstable_by_key(|t| (t.float_name, t.gen_name));
    for i in 0..(tests.len() - 1) {
        if tests[i].gen_name == tests[i + 1].gen_name {
            panic!("duplicate test name {}", tests[i].gen_name);
        }
    }

    tests
}

/// Register all generators for a single float.
fn register_float<F: Float>(tests: &mut Vec<TestInfo>, cfg: &Config)
where
    RangeInclusive<F::Int>: Iterator<Item = F::Int>,
    <F::Int as TryFrom<u128>>::Error: std::fmt::Debug,
    StandardUniform: Distribution<<F as traits::Float>::Int>,
{
    if F::BITS <= MAX_BITS_FOR_EXHAUUSTIVE {
        // Only run exhaustive tests if there is a chance of completion.
        TestInfo::register::<F, gen_::exhaustive::Exhaustive<F>>(tests);
    }

    gen_::fuzz::Fuzz::<F>::set_iterations(cfg.fuzz_count);

    TestInfo::register::<F, gen_::exponents::LargeExponents<F>>(tests);
    TestInfo::register::<F, gen_::exponents::SmallExponents<F>>(tests);
    TestInfo::register::<F, gen_::fuzz::Fuzz<F>>(tests);
    TestInfo::register::<F, gen_::integers::LargeInt<F>>(tests);
    TestInfo::register::<F, gen_::integers::SmallInt>(tests);
    TestInfo::register::<F, gen_::long_fractions::RepeatingDecimal>(tests);
    TestInfo::register::<F, gen_::many_digits::RandDigits<F>>(tests);
    TestInfo::register::<F, gen_::sparse::FewOnesFloat<F>>(tests);
    TestInfo::register::<F, gen_::sparse::FewOnesInt<F>>(tests);
    TestInfo::register::<F, gen_::spot_checks::RegressionCheck>(tests);
    TestInfo::register::<F, gen_::spot_checks::Special>(tests);
    TestInfo::register::<F, gen_::subnorm::SubnormComplete<F>>(tests);
    TestInfo::register::<F, gen_::subnorm::SubnormEdgeCases<F>>(tests);
}

/// Configuration for a single test.
#[derive(Debug)]
pub struct TestInfo {
    pub name: String,
    float_name: &'static str,
    float_bits: u32,
    gen_name: &'static str,
    /// Name for display in the progress bar.
    short_name: String,
    /// Pad the short name to a common width for progress bar use.
    short_name_padded: String,
    total_tests: u64,
    /// Function to launch this test.
    launch: fn(&TestInfo, &Config),
    /// Progress bar to be updated.
    progress: Option<ui::Progress>,
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
        let name = format!("{f_name} {gen_name}");
        let short_name = format!("{f_name} {gen_short_name}");
        let short_name_padded = format!("{short_name:18}");

        let info = TestInfo {
            float_name: f_name,
            float_bits: F::BITS,
            gen_name,
            progress: None,
            name,
            short_name_padded,
            short_name,
            launch: test_runner::<F, G>,
            total_tests: G::total_tests(),
            completed: OnceLock::new(),
        };
        v.push(info);
    }

    /// True if this should be run after all others.
    fn is_huge_test(&self) -> bool {
        self.total_tests >= HUGE_TEST_CUTOFF
    }

    /// When the test is finished, update progress bar messages and finalize.
    fn complete(&self, c: Completed) {
        self.progress.as_ref().unwrap().complete(&c, 0);
        self.completed.set(c).unwrap();
    }
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
    /// String did not parse successfully.
    ParsingFailed(Box<str>),
    /// A panic was caught.
    Panic(Box<str>),
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
            CheckFailure::ParsingFailed(e) => write!(f, "parsing failed: {e}"),
            CheckFailure::Panic(e) => write!(f, "function panicked: {e}"),
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
    // Run shorter tests and smaller float types first.
    tests.sort_unstable_by_key(|test| (test.total_tests, test.float_bits));

    for test in tests.iter() {
        println!("Launching test '{}'", test.name);
    }

    let mut all_progress_bars = Vec::new();
    let start = Instant::now();

    for test in tests.iter_mut() {
        test.progress = Some(ui::Progress::new(test, &mut all_progress_bars));
        ui::set_panic_hook(&all_progress_bars);
        ((test.launch)(test, cfg));
    }

    start.elapsed()
}

/// Test runer for a single generator.
///
/// This calls the generator's iterator multiple times (in parallel) and validates each output.
fn test_runner<F: Float, G: Generator<F>>(test: &TestInfo, cfg: &Config) {
    let gen_ = G::new();
    let executed = AtomicU64::new(0);
    let failures = AtomicU64::new(0);

    let checks_per_update = min(test.total_tests, 1000);
    let started = Instant::now();

    // Function to execute for a single test iteration.
    let check_one = |buf: &mut String, ctx: G::WriteCtx| {
        let executed = executed.fetch_add(1, Ordering::Relaxed);
        buf.clear();
        G::write_string(buf, ctx);

        match validate::validate::<F>(buf) {
            Ok(()) => (),
            Err(e) => {
                let CheckError { fail, input, float_res } = e;
                test.progress.as_ref().unwrap().println(&format!(
                    "Failure in '{}': {fail}. parsing '{input}'. Parsed as: {float_res}",
                    test.name
                ));

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
            test.progress.as_ref().unwrap().update(executed, failures);
            if started.elapsed() > cfg.timeout {
                return Err(EarlyExit::Timeout);
            }
        }

        Ok(())
    };

    // Run the test iterations in parallel. Each thread gets a string buffer to write
    // its check values to.
    let res = gen_.par_bridge().try_for_each_init(String::new, check_one);

    let elapsed = started.elapsed();
    let executed = executed.into_inner();
    let failures = failures.into_inner();

    // Warn about bad estimates if relevant.
    let warning = if executed != test.total_tests && res.is_ok() {
        let msg = format!(
            "executed tests != estimated ({executed} != {}) for {}",
            test.total_tests,
            G::NAME
        );

        Some(msg.into())
    } else {
        None
    };

    let result = res.map(|()| FinishedAll);
    test.complete(Completed { executed, failures, result, warning, elapsed });
}
