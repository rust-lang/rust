use std::io::ErrorKind;
use std::sync::OnceLock;
use std::thread::{self, Builder, ThreadId};

static THREAD_ID: OnceLock<ThreadId> = OnceLock::new();

#[test]
fn spawn_thread_would_block() {
    assert_eq!(Builder::new().spawn(|| unreachable!()).unwrap_err().kind(), ErrorKind::WouldBlock);
    THREAD_ID.set(thread::current().id()).unwrap();
}

// Tests are run in alphabetical order, and the second test is dependent on the
// first to set THREAD_ID. Do not rename the tests in such a way that `test_run_in_same_thread`
// would run before `spawn_thread_would_block`.
// See https://doc.rust-lang.org/rustc/tests/index.html#--shuffle

#[test]
fn test_run_in_same_thread() {
    assert_eq!(*THREAD_ID.get().unwrap(), thread::current().id());
}
