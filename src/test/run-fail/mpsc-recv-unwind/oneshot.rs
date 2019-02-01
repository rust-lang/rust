// Test that unwinding while an MPSC channel receiver is blocking doesn't panic
// or deadlock inside the MPSC implementation.
//
// Various platforms may trigger unwinding while a thread is blocked, due to an
// error condition. It can be tricky to trigger such unwinding. The test here
// uses pthread_cancel on Linux. If at some point in the future, pthread_cancel
// no longer unwinds through the MPSC code, that doesn't mean this test should
// be removed. Instead, another way should be found to trigger unwinding in a
// blocked MPSC channel receiver.

// only-linux
// error-pattern:FATAL: exception not rethrown
// failure-status:signal:6

#![feature(rustc_private)]

mod common;

fn main() {
    let (_s, r) = std::sync::mpsc::channel();
    common::panic_inside_mpsc_recv(r);
}
