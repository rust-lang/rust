//! Check that destructors of the thread locals are executed on all OSes.

use std::cell::RefCell;

struct TestCell {
    value: RefCell<u8>,
}

impl Drop for TestCell {
    fn drop(&mut self) {
        eprintln!("Dropping: {}", self.value.borrow())
    }
}

thread_local! {
    static A: TestCell = TestCell { value: RefCell::new(0) };
}

fn main() {
    A.with(|f| {
        assert_eq!(*f.value.borrow(), 0);
        *f.value.borrow_mut() = 5;
    });
    eprintln!("Continue main.")
}
