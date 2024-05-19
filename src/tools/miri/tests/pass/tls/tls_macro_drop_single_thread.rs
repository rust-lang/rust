//! Check that destructors of the thread locals are executed on all OSes
//! (even when we do not support concurrency, and cannot run the other test).

use std::cell::RefCell;

struct TestCell {
    value: RefCell<u8>,
}

impl Drop for TestCell {
    fn drop(&mut self) {
        eprintln!("Dropping: {}", *self.value.borrow())
    }
}

thread_local! {
    static A: TestCell = TestCell { value: RefCell::new(0) };
    static A_CONST: TestCell = const { TestCell { value: RefCell::new(10) } };
}

fn main() {
    A.with(|f| {
        assert_eq!(*f.value.borrow(), 0);
        *f.value.borrow_mut() = 5;
    });
    A_CONST.with(|f| {
        assert_eq!(*f.value.borrow(), 10);
        *f.value.borrow_mut() = 5; // Same value as above since the drop order is different on different platforms
    });
    eprintln!("Continue main.")
}
