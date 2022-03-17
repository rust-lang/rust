use std::any::Any;

use super::bench::BenchSamples;
use super::options::ShouldPanic;
use super::time;
use super::types::TestDesc;

pub use self::TestResult::*;

// Return codes for secondary process.
// Start somewhere other than 0 so we know the return code means what we think
// it means.
pub const TR_OK: i32 = 50;
pub const TR_FAILED: i32 = 51;

#[derive(Debug, Clone, PartialEq)]
pub enum TestResult {
    TrOk,
    TrFailed,
    TrFailedMsg(String),
    TrIgnored,
    TrBench(BenchSamples),
    TrTimedFail,
}

/// Creates a `TestResult` depending on the raw result of test execution
/// and associated data.
pub fn calc_result<'a>(
    desc: &TestDesc,
    task_result: Result<(), &'a (dyn Any + 'static + Send)>,
    time_opts: &Option<time::TestTimeOptions>,
    exec_time: &Option<time::TestExecTime>,
) -> TestResult {
    let result = match (&desc.should_panic, task_result) {
        (&ShouldPanic::No, Ok(())) | (&ShouldPanic::Yes, Err(_)) => TestResult::TrOk,
        (&ShouldPanic::YesWithMessage(msg), Err(ref err)) => {
            let maybe_panic_str = err
                .downcast_ref::<String>()
                .map(|e| &**e)
                .or_else(|| err.downcast_ref::<&'static str>().copied());

            if maybe_panic_str.map(|e| e.contains(msg)).unwrap_or(false) {
                TestResult::TrOk
            } else if let Some(panic_str) = maybe_panic_str {
                TestResult::TrFailedMsg(format!(
                    r#"panic did not contain expected string
      panic message: `{:?}`,
 expected substring: `{:?}`"#,
                    panic_str, msg
                ))
            } else {
                TestResult::TrFailedMsg(format!(
                    r#"expected panic with string value,
 found non-string value: `{:?}`
     expected substring: `{:?}`"#,
                    (**err).type_id(),
                    msg
                ))
            }
        }
        (&ShouldPanic::Yes, Ok(())) | (&ShouldPanic::YesWithMessage(_), Ok(())) => {
            TestResult::TrFailedMsg("test did not panic as expected".to_string())
        }
        _ => TestResult::TrFailed,
    };

    // If test is already failed (or allowed to fail), do not change the result.
    if result != TestResult::TrOk {
        return result;
    }

    // Check if test is failed due to timeout.
    if let (Some(opts), Some(time)) = (time_opts, exec_time) {
        if opts.error_on_excess && opts.is_critical(desc, time) {
            return TestResult::TrTimedFail;
        }
    }

    result
}

/// Creates a `TestResult` depending on the exit code of test subprocess.
pub fn get_result_from_exit_code(
    desc: &TestDesc,
    code: i32,
    time_opts: &Option<time::TestTimeOptions>,
    exec_time: &Option<time::TestExecTime>,
) -> TestResult {
    let result = match code {
        TR_OK => TestResult::TrOk,
        TR_FAILED => TestResult::TrFailed,
        _ => TestResult::TrFailedMsg(format!("got unexpected return code {}", code)),
    };

    // If test is already failed (or allowed to fail), do not change the result.
    if result != TestResult::TrOk {
        return result;
    }

    // Check if test is failed due to timeout.
    if let (Some(opts), Some(time)) = (time_opts, exec_time) {
        if opts.error_on_excess && opts.is_critical(desc, time) {
            return TestResult::TrTimedFail;
        }
    }

    result
}
