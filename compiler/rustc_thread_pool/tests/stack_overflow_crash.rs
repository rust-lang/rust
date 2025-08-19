#![allow(unused_crate_dependencies)]

use std::env;
#[cfg(target_os = "linux")]
use std::os::unix::process::ExitStatusExt;
use std::process::{Command, ExitStatus, Stdio};

use rustc_thread_pool::ThreadPoolBuilder;

fn force_stack_overflow(depth: u32) {
    let mut buffer = [0u8; 1024 * 1024];
    #[allow(clippy::incompatible_msrv)]
    std::hint::black_box(&mut buffer);
    if depth > 0 {
        force_stack_overflow(depth - 1);
    }
}

#[cfg(unix)]
fn disable_core() {
    unsafe {
        libc::setrlimit(libc::RLIMIT_CORE, &libc::rlimit { rlim_cur: 0, rlim_max: 0 });
    }
}

#[cfg(unix)]
fn overflow_code() -> Option<i32> {
    None
}

#[cfg(windows)]
fn overflow_code() -> Option<i32> {
    use std::os::windows::process::ExitStatusExt;

    ExitStatus::from_raw(0xc00000fd /*STATUS_STACK_OVERFLOW*/).code()
}

// FIXME: We should fix or remove this test on Windows.
#[test]
#[cfg_attr(not(any(unix)), ignore)]
fn stack_overflow_crash() {
    // First check that the recursive call actually causes a stack overflow,
    // and does not get optimized away.
    let status = run_ignored("run_with_small_stack");
    assert!(!status.success());
    #[cfg(any(unix, windows))]
    assert_eq!(status.code(), overflow_code());
    #[cfg(target_os = "linux")]
    assert!(matches!(status.signal(), Some(libc::SIGABRT | libc::SIGSEGV)));

    // Now run with a larger stack and verify correct operation.
    let status = run_ignored("run_with_large_stack");
    assert_eq!(status.code(), Some(0));
    #[cfg(target_os = "linux")]
    assert_eq!(status.signal(), None);
}

fn run_ignored(test: &str) -> ExitStatus {
    Command::new(env::current_exe().unwrap())
        .arg("--ignored")
        .arg("--exact")
        .arg(test)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap()
}

#[test]
#[ignore]
fn run_with_small_stack() {
    run_with_stack(8);
}

#[test]
#[ignore]
fn run_with_large_stack() {
    run_with_stack(48);
}

fn run_with_stack(stack_size_in_mb: usize) {
    let pool = ThreadPoolBuilder::new().stack_size(stack_size_in_mb * 1024 * 1024).build().unwrap();
    pool.install(|| {
        #[cfg(unix)]
        disable_core();
        force_stack_overflow(32);
    });
}
