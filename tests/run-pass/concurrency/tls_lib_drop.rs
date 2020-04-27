// ignore-windows: Concurrency on Windows is not supported yet.

use std::cell::RefCell;
use std::thread;

struct TestCell {
    value: RefCell<u8>,
}

impl Drop for TestCell {
    fn drop(&mut self) {
        println!("Dropping: {}", self.value.borrow())
    }
}

thread_local! {
    static A: TestCell = TestCell { value: RefCell::new(0) };
}

/// Check that destructors of the library thread locals are executed immediately
/// after a thread terminates.
fn check_destructors() {
    thread::spawn(|| {
        A.with(|f| {
            assert_eq!(*f.value.borrow(), 0);
            *f.value.borrow_mut() = 5;
        });
    })
    .join()
    .unwrap();
    println!("Continue main.")
}

struct JoinCell {
    value: RefCell<Option<thread::JoinHandle<u8>>>,
}

impl Drop for JoinCell {
    fn drop(&mut self) {
        let join_handle = self.value.borrow_mut().take().unwrap();
        println!("Joining: {}", join_handle.join().unwrap());
    }
}

thread_local! {
    static B: JoinCell = JoinCell { value: RefCell::new(None) };
}

/// Check that the destructor can be blocked joining another thread.
fn check_blocking() {
    thread::spawn(|| {
        B.with(|f| {
            assert!(f.value.borrow().is_none());
            let handle = thread::spawn(|| 7);
            *f.value.borrow_mut() = Some(handle);
        });
    })
    .join()
    .unwrap();
    println!("Continue main 2.");
    // Preempt the main thread so that the destructor gets executed and can join
    // the thread.
    thread::yield_now();
    thread::yield_now();
}

fn main() {
    check_destructors();
    check_blocking();
}
