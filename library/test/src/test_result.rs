use std::any::Any;
#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;
use std::process::{ExitStatus, Output};
use std::{fmt, io};

pub use self::TestResult::*;
use super::bench::BenchSamples;
use super::options::ShouldPanic;
use super::time;
use super::types::TestDesc;

// Return code for secondary process.
// Start somewhere other than 0 so we know the return code means what we think
// it means.
pub(crate) const TR_OK: i32 = 50;

// On Windows we use __fastfail to abort, which is documented to use this
// exception code.
#[cfg(windows)]
const STATUS_FAIL_FAST_EXCEPTION: i32 = 0xC0000409u32 as i32;

// On Zircon (the Fuchsia kernel), an abort from userspace calls the
// LLVM implementation of __builtin_trap(), e.g., ud2 on x86, which
// raises a kernel exception. If a userspace process does not
// otherwise arrange exception handling, the kernel kills the process
// with this return code.
#[cfg(target_os = "fuchsia")]
const ZX_TASK_RETCODE_EXCEPTION_KILL: i32 = -1028;

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
pub(crate) fn calc_result(
    desc: &TestDesc,
    panic_payload: Option<&(dyn Any + Send)>,
    time_opts: Option<&time::TestTimeOptions>,
    exec_time: Option<&time::TestExecTime>,
) -> TestResult {
    let result = match (desc.should_panic, panic_payload) {
        // The test did or didn't panic, as expected.
        (ShouldPanic::No, None) | (ShouldPanic::Yes, Some(_)) => TestResult::TrOk,

        // Check the actual panic message against the expected message.
        (ShouldPanic::YesWithMessage(msg), Some(err)) => {
            let maybe_panic_str = err
                .downcast_ref::<String>()
                .map(|e| &**e)
                .or_else(|| err.downcast_ref::<&'static str>().copied());

            if maybe_panic_str.map(|e| e.contains(msg)).unwrap_or(false) {
                TestResult::TrOk
            } else if let Some(panic_str) = maybe_panic_str {
                TestResult::TrFailedMsg(format!(
                    r#"panic did not contain expected string
      panic message: {panic_str:?}
 expected substring: {msg:?}"#
                ))
            } else {
                TestResult::TrFailedMsg(format!(
                    r#"expected panic with string value,
 found non-string value: `{:?}`
     expected substring: {msg:?}"#,
                    (*err).type_id()
                ))
            }
        }

        // The test should have panicked, but didn't panic.
        (ShouldPanic::Yes, None) | (ShouldPanic::YesWithMessage(_), None) => {
            let fn_location = if !desc.source_file.is_empty() {
                &format!(" at {}:{}:{}", desc.source_file, desc.start_line, desc.start_col)
            } else {
                ""
            };
            TestResult::TrFailedMsg(format!("test did not panic as expected{}", fn_location))
        }

        // The test should not have panicked, but did panic.
        (ShouldPanic::No, Some(_)) => TestResult::TrFailed,
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

/// Creates a `TestResult` depending on the exit code of test subprocess
pub(crate) fn get_result_from_exit_code_inner(
    status: ExitStatus,
    success_error_code: i32,
) -> TestResult {
    match status.code() {
        Some(error_code) if error_code == success_error_code => TestResult::TrOk,
        Some(crate::ERROR_EXIT_CODE) => TestResult::TrFailed,
        #[cfg(windows)]
        Some(STATUS_FAIL_FAST_EXCEPTION) => TestResult::TrFailed,
        #[cfg(unix)]
        None => match status.signal() {
            Some(libc::SIGABRT) => TestResult::TrFailed,
            Some(signal) => {
                TestResult::TrFailedMsg(format!("child process exited with signal {signal}"))
            }
            None => unreachable!("status.code() returned None but status.signal() was None"),
        },
        // Upon an abort, Fuchsia returns the status code ZX_TASK_RETCODE_EXCEPTION_KILL.
        #[cfg(target_os = "fuchsia")]
        Some(ZX_TASK_RETCODE_EXCEPTION_KILL) => TestResult::TrFailed,
        #[cfg(not(unix))]
        None => TestResult::TrFailedMsg(format!("unknown return code")),
        #[cfg(any(windows, unix))]
        Some(code) => TestResult::TrFailedMsg(format!("got unexpected return code {code}")),
        #[cfg(not(any(windows, unix)))]
        Some(_) => TestResult::TrFailed,
    }
}

/// Creates a `TestResult` depending on the exit code of test subprocess and on its runtime.
pub(crate) fn get_result_from_exit_code(
    desc: &TestDesc,
    status: ExitStatus,
    time_opts: Option<&time::TestTimeOptions>,
    exec_time: Option<&time::TestExecTime>,
) -> TestResult {
    let result = get_result_from_exit_code_inner(status, TR_OK);

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

pub enum RustdocResult {
    /// The test failed to compile.
    CompileError,
    /// The test is marked `compile_fail` but compiled successfully.
    UnexpectedCompilePass,
    /// The test failed to compile (as expected) but the compiler output did not contain all
    /// expected error codes.
    MissingErrorCodes(Vec<String>),
    /// The test binary was unable to be executed.
    ExecutionError(io::Error),
    /// The test binary exited with a non-zero exit code.
    ///
    /// This typically means an assertion in the test failed or another form of panic occurred.
    ExecutionFailure(Output),
    /// The test is marked `should_panic` but the test binary executed successfully.
    NoPanic(Option<String>),
}

impl fmt::Display for RustdocResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CompileError => {
                write!(f, "Couldn't compile the test.")
            }
            Self::UnexpectedCompilePass => {
                write!(f, "Test compiled successfully, but it's marked `compile_fail`.")
            }
            Self::NoPanic(msg) => {
                write!(f, "Test didn't panic, but it's marked `should_panic`")?;
                if let Some(msg) = msg {
                    write!(f, " ({msg})")?;
                }
                f.write_str(".")
            }
            Self::MissingErrorCodes(codes) => {
                write!(f, "Some expected error codes were not found: {codes:?}")
            }
            Self::ExecutionError(err) => {
                write!(f, "Couldn't run the test: {err}")?;
                if err.kind() == io::ErrorKind::PermissionDenied {
                    f.write_str(" - maybe your tempdir is mounted with noexec?")?;
                }
                Ok(())
            }
            Self::ExecutionFailure(out) => {
                writeln!(f, "Test executable failed ({reason}).", reason = out.status)?;

                // FIXME(#12309): An unfortunate side-effect of capturing the test
                // executable's output is that the relative ordering between the test's
                // stdout and stderr is lost. However, this is better than the
                // alternative: if the test executable inherited the parent's I/O
                // handles the output wouldn't be captured at all, even on success.
                //
                // The ordering could be preserved if the test process' stderr was
                // redirected to stdout, but that functionality does not exist in the
                // standard library, so it may not be portable enough.
                let stdout = str::from_utf8(&out.stdout).unwrap_or_default();
                let stderr = str::from_utf8(&out.stderr).unwrap_or_default();

                if !stdout.is_empty() || !stderr.is_empty() {
                    writeln!(f)?;

                    if !stdout.is_empty() {
                        writeln!(f, "stdout:\n{stdout}")?;
                    }

                    if !stderr.is_empty() {
                        writeln!(f, "stderr:\n{stderr}")?;
                    }
                }
                Ok(())
            }
        }
    }
}

pub fn get_rustdoc_result(output: Output, should_panic: bool) -> Result<(), RustdocResult> {
    let result = get_result_from_exit_code_inner(output.status, 0);
    match (result, should_panic) {
        (TestResult::TrFailed, true) | (TestResult::TrOk, false) => Ok(()),
        (TestResult::TrOk, true) => Err(RustdocResult::NoPanic(None)),
        (TestResult::TrFailedMsg(msg), true) => Err(RustdocResult::NoPanic(Some(msg))),
        (TestResult::TrFailedMsg(_) | TestResult::TrFailed, false) => {
            Err(RustdocResult::ExecutionFailure(output))
        }
        _ => unreachable!("unexpected status for rustdoc test output"),
    }
}
