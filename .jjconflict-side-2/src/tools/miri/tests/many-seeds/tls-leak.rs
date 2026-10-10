//! Regression test for <https://github.com/rust-lang/rust/issues/123583>.
use std::thread;

fn with_thread_local1() {
    thread_local! { static X: Box<u8> = Box::new(0); }
    X.with(|_x| {})
}

fn with_thread_local2() {
    thread_local! { static Y: Box<u8> = Box::new(0); }
    Y.with(|_y| {})
}

fn main() {
    // Here we have two threads racing on initializing the thread-local and adding it to the global
    // dtor list (on targets that have such a list, i.e., targets without target_thread_local).
    let t = thread::spawn(with_thread_local1);
    with_thread_local1();
    t.join().unwrap();

    // Here we have one thread running the destructors racing with another thread initializing a
    // thread-local. The second thread adds a destructor that could be picked up by the first.
    let t = thread::spawn(|| { /* immediately just run destructors */ });
    with_thread_local2(); // initialize thread-local
    t.join().unwrap();
}
