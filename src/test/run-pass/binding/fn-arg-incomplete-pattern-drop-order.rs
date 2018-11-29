// Check that partially moved from function parameters are dropped after the
// named bindings that move from them.

use std::{panic, cell::RefCell};

struct LogDrop<'a>(i32, Context<'a>);

#[derive(Copy, Clone)]
struct Context<'a> {
    panic_on: i32,
    drops: &'a RefCell<Vec<i32>>,
}

impl<'a> Context<'a> {
    fn record_drop(self, index: i32) {
        self.drops.borrow_mut().push(index);
        if index == self.panic_on {
            panic!();
        }
    }
}

impl<'a> Drop for LogDrop<'a> {
    fn drop(&mut self) {
        self.1.record_drop(self.0);
    }
}

fn foo((_x, _): (LogDrop, LogDrop), (_, _y): (LogDrop, LogDrop)) {}

fn test_drop_order(panic_on: i32) {
    let context = Context {
        panic_on,
        drops: &RefCell::new(Vec::new()),
    };
    let one = LogDrop(1, context);
    let two = LogDrop(2, context);
    let three = LogDrop(3, context);
    let four = LogDrop(4, context);

    let res = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        foo((three, four), (two, one));
    }));
    if panic_on == 0 {
        assert!(res.is_ok(), "should not have panicked");
    } else {
        assert!(res.is_err(), "should have panicked");
    }
    assert_eq!(*context.drops.borrow(), [1, 2, 3, 4], "incorrect drop order");
}

fn main() {
    (0..=4).for_each(test_drop_order);
}
