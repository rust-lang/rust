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
//@ ignore-cross-compile
// Reason: this test fails armhf-gnu, reasons unknown
//@ needs-unwind
// Reason: this should be ignored in cg_clif (Cranelift) CI and anywhere
// else that uses panic=abort.

use run_make_support::{libc, rustc};

fn main() {
    rustc().input("test.rs").arg("--test").run();

    // We need to emulate an environment for libtest where threads are exhausted and spawning
    // new threads are guaranteed to fail. This was previously achieved by ulimit shell builtin
    // that called out to prlimit64 underneath to set resource limits (specifically thread
    // number limits). Now that we don't have a shell, we need to implement that ourselves.
    // See https://linux.die.net/man/2/setrlimit

    // The fork + exec is required because we cannot first try to limit the number of
    // processes/threads to 1 and then try to spawn a new process to run the test. We need to
    // setrlimit and run the libtest test program in the same process.
    let pid = unsafe { libc::fork() };
    assert!(pid >= 0);

    // If the process ID is 0, this is the child process responsible for running the test
    // program.
    if pid == 0 {
        let test = c"test";
        // The argv array should be terminated with a NULL pointer.
        let argv = [test.as_ptr(), std::ptr::null()];
        // rlim_cur is soft limit, rlim_max is hard limit.
        // By setting the limit very low (max 1), we ensure that libtest is unable to create new
        // threads.
        let rlimit = libc::rlimit { rlim_cur: 1, rlim_max: 1 };
        // RLIMIT_NPROC: The maximum number of processes (or, more precisely on Linux,
        // threads) that can be created for the real user ID of the calling process. Upon
        // encountering this limit, fork(2) fails with the error EAGAIN.
        // Therefore, set the resource limit to RLIMIT_NPROC.
        let ret = unsafe { libc::setrlimit(libc::RLIMIT_NPROC, &rlimit as *const libc::rlimit) };
        assert_eq!(ret, 0);

        // Finally, execute the 2 tests in test.rs.
        let ret = unsafe { libc::execv(test.as_ptr(), argv.as_ptr()) };
        assert_eq!(ret, 0);
    } else {
        // Otherwise, other process IDs indicate that this is the parent process.

        let mut status: libc::c_int = 0;
        let ret = unsafe { libc::waitpid(pid, &mut status as *mut libc::c_int, 0) };
        assert_eq!(ret, pid);
        assert!(libc::WIFEXITED(status));
        assert_eq!(libc::WEXITSTATUS(status), 0);
    }
}
