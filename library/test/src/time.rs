//! Module `time` contains everything related to the time measurement of unit tests
//! execution.
//! The purposes of this module:
//! - Check whether test is timed out.
//! - Provide helpers for `report-time` and `measure-time` options.
//! - Provide newtypes for executions times.

use std::env;
use std::fmt;
use std::str::FromStr;
use std::time::{Duration, Instant};

use super::types::{TestDesc, TestType};

pub const TEST_WARN_TIMEOUT_S: u64 = 60;

/// This small module contains constants used by `report-time` option.
/// Those constants values will be used if corresponding environment variables are not set.
///
/// To override values for unit-tests, use a constant `RUST_TEST_TIME_UNIT`,
/// To override values for integration tests, use a constant `RUST_TEST_TIME_INTEGRATION`,
/// To override values for doctests, use a constant `RUST_TEST_TIME_DOCTEST`.
///
/// Example of the expected format is `RUST_TEST_TIME_xxx=100,200`, where 100 means
/// warn time, and 200 means critical time.
pub mod time_constants {
    use super::TEST_WARN_TIMEOUT_S;
    use std::time::Duration;

    /// Environment variable for overriding default threshold for unit-tests.
    pub const UNIT_ENV_NAME: &str = "RUST_TEST_TIME_UNIT";

    // Unit tests are supposed to be really quick.
    pub const UNIT_WARN: Duration = Duration::from_millis(50);
    pub const UNIT_CRITICAL: Duration = Duration::from_millis(100);

    /// Environment variable for overriding default threshold for unit-tests.
    pub const INTEGRATION_ENV_NAME: &str = "RUST_TEST_TIME_INTEGRATION";

    // Integration tests may have a lot of work, so they can take longer to execute.
    pub const INTEGRATION_WARN: Duration = Duration::from_millis(500);
    pub const INTEGRATION_CRITICAL: Duration = Duration::from_millis(1000);

    /// Environment variable for overriding default threshold for unit-tests.
    pub const DOCTEST_ENV_NAME: &str = "RUST_TEST_TIME_DOCTEST";

    // Doctests are similar to integration tests, because they can include a lot of
    // initialization code.
    pub const DOCTEST_WARN: Duration = INTEGRATION_WARN;
    pub const DOCTEST_CRITICAL: Duration = INTEGRATION_CRITICAL;

    // Do not suppose anything about unknown tests, base limits on the
    // `TEST_WARN_TIMEOUT_S` constant.
    pub const UNKNOWN_WARN: Duration = Duration::from_secs(TEST_WARN_TIMEOUT_S);
    pub const UNKNOWN_CRITICAL: Duration = Duration::from_secs(TEST_WARN_TIMEOUT_S * 2);
}

/// Returns an `Instance` object denoting when the test should be considered
/// timed out.
pub fn get_default_test_timeout() -> Instant {
    Instant::now() + Duration::from_secs(TEST_WARN_TIMEOUT_S)
}

/// The measured execution time of a unit test.
#[derive(Debug, Clone, PartialEq)]
pub struct TestExecTime(pub Duration);

impl fmt::Display for TestExecTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.3}s", self.0.as_secs_f64())
    }
}

/// The measured execution time of the whole test suite.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TestSuiteExecTime(pub Duration);

impl fmt::Display for TestSuiteExecTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}s", self.0.as_secs_f64())
    }
}

/// Structure denoting time limits for test execution.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct TimeThreshold {
    pub warn: Duration,
    pub critical: Duration,
}

impl TimeThreshold {
    /// Creates a new `TimeThreshold` instance with provided durations.
    pub fn new(warn: Duration, critical: Duration) -> Self {
        Self { warn, critical }
    }

    /// Attempts to create a `TimeThreshold` instance with values obtained
    /// from the environment variable, and returns `None` if the variable
    /// is not set.
    /// Environment variable format is expected to match `\d+,\d+`.
    ///
    /// # Panics
    ///
    /// Panics if variable with provided name is set but contains inappropriate
    /// value.
    pub fn from_env_var(env_var_name: &str) -> Option<Self> {
        let durations_str = env::var(env_var_name).ok()?;
        let (warn_str, critical_str) = durations_str.split_once(',').unwrap_or_else(|| {
            panic!(
                "Duration variable {} expected to have 2 numbers separated by comma, but got {}",
                env_var_name, durations_str
            )
        });

        let parse_u64 = |v| {
            u64::from_str(v).unwrap_or_else(|_| {
                panic!(
                    "Duration value in variable {} is expected to be a number, but got {}",
                    env_var_name, v
                )
            })
        };

        let warn = parse_u64(warn_str);
        let critical = parse_u64(critical_str);
        if warn > critical {
            panic!("Test execution warn time should be less or equal to the critical time");
        }

        Some(Self::new(Duration::from_millis(warn), Duration::from_millis(critical)))
    }
}

/// Structure with parameters for calculating test execution time.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct TestTimeOptions {
    /// Denotes if the test critical execution time limit excess should be considered
    /// a test failure.
    pub error_on_excess: bool,
    pub colored: bool,
    pub unit_threshold: TimeThreshold,
    pub integration_threshold: TimeThreshold,
    pub doctest_threshold: TimeThreshold,
}

impl TestTimeOptions {
    pub fn new_from_env(error_on_excess: bool, colored: bool) -> Self {
        let unit_threshold = TimeThreshold::from_env_var(time_constants::UNIT_ENV_NAME)
            .unwrap_or_else(Self::default_unit);

        let integration_threshold =
            TimeThreshold::from_env_var(time_constants::INTEGRATION_ENV_NAME)
                .unwrap_or_else(Self::default_integration);

        let doctest_threshold = TimeThreshold::from_env_var(time_constants::DOCTEST_ENV_NAME)
            .unwrap_or_else(Self::default_doctest);

        Self { error_on_excess, colored, unit_threshold, integration_threshold, doctest_threshold }
    }

    pub fn is_warn(&self, test: &TestDesc, exec_time: &TestExecTime) -> bool {
        exec_time.0 >= self.warn_time(test)
    }

    pub fn is_critical(&self, test: &TestDesc, exec_time: &TestExecTime) -> bool {
        exec_time.0 >= self.critical_time(test)
    }

    fn warn_time(&self, test: &TestDesc) -> Duration {
        match test.test_type {
            TestType::UnitTest => self.unit_threshold.warn,
            TestType::IntegrationTest => self.integration_threshold.warn,
            TestType::DocTest => self.doctest_threshold.warn,
            TestType::Unknown => time_constants::UNKNOWN_WARN,
        }
    }

    fn critical_time(&self, test: &TestDesc) -> Duration {
        match test.test_type {
            TestType::UnitTest => self.unit_threshold.critical,
            TestType::IntegrationTest => self.integration_threshold.critical,
            TestType::DocTest => self.doctest_threshold.critical,
            TestType::Unknown => time_constants::UNKNOWN_CRITICAL,
        }
    }

    fn default_unit() -> TimeThreshold {
        TimeThreshold::new(time_constants::UNIT_WARN, time_constants::UNIT_CRITICAL)
    }

    fn default_integration() -> TimeThreshold {
        TimeThreshold::new(time_constants::INTEGRATION_WARN, time_constants::INTEGRATION_CRITICAL)
    }

    fn default_doctest() -> TimeThreshold {
        TimeThreshold::new(time_constants::DOCTEST_WARN, time_constants::DOCTEST_CRITICAL)
    }
}
