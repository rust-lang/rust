//@only-target: linux
//@revisions: with_isolation without_isolation
//@[without_isolation] compile-flags: -Zmiri-disable-isolation

use std::thread;

use libc::{getpid, gettid};

fn main() {
    thread::spawn(|| {
        // Test that in isolation mode a deterministic value will be returned.
        // The value 1001 is not important, we only care that whatever the value
        // is, won't change from execution to execution.
        #[cfg(with_isolation)]
        assert_eq!(unsafe { gettid() }, 1001);

        assert_ne!(unsafe { gettid() }, unsafe { getpid() });
    });

    // Test that the thread ID of the main thread is the same as the process
    // ID.
    assert_eq!(unsafe { gettid() }, unsafe { getpid() });
}
