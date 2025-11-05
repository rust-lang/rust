// Test drop order for destructuring assignments against
// other expressions they should be consistent with.
//
// See:
//
// - https://github.com/rust-lang/rust/pull/145838
//
// Original author: TC
// Date: 2025-08-30
//@ edition: 2024
//@ run-pass

#![allow(unused_must_use)]

fn main() {
    assert_drop_order(1..=3, |e| {
        &({ &*&e.log(2) }, drop(e.log(1)));
        // &({ &raw const *&e.log(2) }, drop(e.log(1)));
        drop(e.log(3));
    });
    /*
    assert_drop_order(1..=3, |e| {
        { let _x; _x = &({ &raw const *&e.log(2) }, drop(e.log(1))); }
        drop(e.log(3));
    });
    assert_drop_order(1..=3, |e| {
        _ = &({ &raw const *&e.log(2) }, drop(e.log(1)));
        drop(e.log(3));
    });
    assert_drop_order(1..=3, |e| {
        { let _ = &({ &raw const *&e.log(2) }, drop(e.log(1))); }
        drop(e.log(3));
    });
    assert_drop_order(1..=3, |e| {
        let _x; let _y;
        (_x, _y) = ({ &raw const *&e.log(2) }, drop(e.log(1)));
        drop(e.log(3));
    });
    */
}

// # Test scaffolding...

use core::cell::RefCell;

struct DropOrder(RefCell<Vec<u64>>);
struct LogDrop<'o>(&'o DropOrder, u64);

impl DropOrder {
    fn log(&self, n: u64) -> LogDrop<'_> {
        LogDrop(self, n)
    }
}

impl<'o> Drop for LogDrop<'o> {
    fn drop(&mut self) {
        self.0 .0.borrow_mut().push(self.1);
    }
}

#[track_caller]
fn assert_drop_order(
    ex: impl IntoIterator<Item = u64>,
    f: impl Fn(&DropOrder),
) {
    let order = DropOrder(RefCell::new(Vec::new()));
    f(&order);
    let order = order.0.into_inner();
    let expected: Vec<u64> = ex.into_iter().collect();
    assert_eq!(order, expected);
}
