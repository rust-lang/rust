// ignore-windows: Concurrency on Windows is not supported yet.

//! Check that destructors of the library thread locals are executed immediately
//! after a thread terminates.

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
