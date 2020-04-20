// ignore-windows: Concurrency on Windows is not supported yet.

//! Check that destructors of the library thread locals are executed immediately
//! after a thread terminates.

#![feature(thread_local_internals)]

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

static A: std::thread::LocalKey<TestCell> = {
    #[inline]
    fn __init() -> TestCell {
        TestCell { value: RefCell::new(0) }
    }

    unsafe fn __getit() -> Option<&'static TestCell> {
        static __KEY: std::thread::__OsLocalKeyInner<TestCell> =
            std::thread::__OsLocalKeyInner::new();
        __KEY.get(__init)
    }

    unsafe { std::thread::LocalKey::new(__getit) }
};

fn main() {
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
