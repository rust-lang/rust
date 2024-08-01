// libtest used to panic if it hit the thread limit. This often resulted in spurious test failures
// (thread 'main' panicked at 'called Result::unwrap() on an Err value: Os
// { code: 11, kind: WouldBlock, message: "Resource temporarily unavailable" }' ...
// error: test failed, to rerun pass '--lib').
// Since the fix in #81546, the test should continue to run synchronously
// if it runs out of threads. Therefore, this test's final execution step
// should succeed without an error.
// See https://github.com/rust-lang/rust/pull/81546

//@ only-linux
// Reason: thread limit modification

use run_make_support::{libc, run, rustc};

fn main() {
    rustc().input("test.rs").arg("--test").run();
    let rlimit = libc::rlimit {
        rlim_cur: 1, // or 1, needs testing
        rlim_max: 1, // or 1, needs testing
    };
    let ptr = &rlimit as *const libc::rlimit;
    unsafe {
        libc::setrlimit(0, ptr);
    }
    run("test");
}
