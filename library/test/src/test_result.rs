use std::any::Any;
use std::process::ExitStatus;

#[cfg(target_os = "fuchsia")]
use std::os::fuchsia::process::{ExitStatusExt as _, ZX_TASK_RETCODE_EXCEPTION_KILL};
#[cfg(unix)]
use std::os::unix::process::ExitStatusExt as _;

use super::bench::BenchSamples;
use super::options::ShouldPanic;
use super::time;
use super::types::TestDesc;

pub use self::TestResult::*;

// Return code for secondary process.
// Start somewhere other than 0 so we know the return code means what we think
// it means.
pub const TR_OK: i32 = 50;

// On Windows we use __fastfail to abort, which is documented to use this
// exception code.
#[cfg(windows)]
const STATUS_FAIL_FAST_EXCEPTION: i32 = 0xC0000409u32 as i32;

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
        (&ShouldPanic::YesWithMessage(msg), Err(err)) => {
            let maybe_panic_str = err
                .downcast_ref::<String>()
                .map(|e| &**e)
                .or_else(|| err.downcast_ref::<&'static str>().copied());

            if maybe_panic_str.map(|e| e.contains(msg)).unwrap_or(false) {
                TestResult::TrOk
            } else if let Some(panic_str) = maybe_panic_str {
                TestResult::TrFailedMsg(format!(
                    r#"panic did not contain expected string
      panic message: `{panic_str:?}`,
 expected substring: `{msg:?}`"#
                ))
            } else {
                TestResult::TrFailedMsg(format!(
                    r#"expected panic with string value,
 found non-string value: `{:?}`
     expected substring: `{:?}`"#,
                    (*err).type_id(),
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
    status: ExitStatus,
    time_opts: &Option<time::TestTimeOptions>,
    exec_time: &Option<time::TestExecTime>,
) -> TestResult {
    // Upon a panic, a Fuchsia process will trigger a kernel exception
    // that, if uncaught, will cause the kernel to kill the process with
    // ZX_TASK_RETCODE_EXCEPTION_KILL. Though unlikely, the same code could be
    // returned for other unhandled exceptions too. Even in those cases the test
    // should still fail and the printed stacktrace from the kernel should
    // sufficienly compensate for omitting this return code from test output.
    #[cfg(target_os = "fuchsia")]
    let result = match status.task_retcode() {
        Some(ZX_TASK_RETCODE_EXCEPTION_KILL) => Some(TestResult::TrFailed),
        _ => None,
    };
    #[cfg(not(target_os = "fuchsia"))]
    let result: Option<TestResult> = None;

    let result = result.unwrap_or_else(|| match status.code() {
        Some(TR_OK) => TestResult::TrOk,
        #[cfg(windows)]
        Some(STATUS_FAIL_FAST_EXCEPTION) => TestResult::TrFailed,
        #[cfg(any(windows, unix))]
        Some(code) => TestResult::TrFailedMsg(format!("got unexpected return code {code}")),
        #[cfg(not(any(windows, unix)))]
        Some(_) => TestResult::TrFailed,
        #[cfg(unix)]
        None => match status.signal() {
            Some(libc::SIGABRT) => TestResult::TrFailed,
            Some(signal) => {
                TestResult::TrFailedMsg(format!("child process exited with signal {signal}"))
            }
            None => unreachable!("status.code() returned None but status.signal() was None"),
        },
        #[cfg(not(unix))]
        None => TestResult::TrFailedMsg(format!("unknown return code")),
    });

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
