//! Regression test for <https://github.com/rust-lang/rust/issues/123583>.
use std::thread;

pub(crate) fn with_thread_local() {
    thread_local! { static X: Box<u8> = Box::new(0); }
    X.with(|_x| {})
}

fn main() {
    let j2 = thread::spawn(with_thread_local);
    with_thread_local();
    j2.join().unwrap();
}
