//! Regression test for <https://github.com/rust-lang/rust/issues/165322>: if a `super let` binding
//! had the scope of an `if let` guard's temporaries and the guard was lowered multiple times due
//! to or-patterns, drops weren't emitted for it if the first or-alternative didn't match.
//@ run-pass

#![feature(super_let)]
#![allow(irrefutable_let_patterns, unused)]

use std::{cell::RefCell, ops::Drop};
use std::pin::pin;

fn main() {
    // In each of these cases, the `super let` binding created in the guard should live until the
    // end of the match arm, then be dropped, regardless of which or-pattern alternative matches.

    for b in [false, true] {
        assert_drop_order(1..=2, |e| {
            match b {
                false | true if let p = pin!(e.log(2)) => drop(e.log(1)),
                _ => {}
            }
        });
    }

    for b in [false, true] {
        assert_drop_order(1..=2, |e| {
            match b {
                false | true
                    if let _ = {
                        super let a = e.log(2);
                        e.log(1);
                    } => {}
                _ => {}
            }
        });
    }
}

// # Test scaffolding...

struct DropOrder(RefCell<Vec<u64>>);
struct LogDrop<'o>(&'o DropOrder, u64);

impl DropOrder {
    fn log(&self, n: u64) -> LogDrop<'_> {
        LogDrop(self, n)
    }
}

impl<'o> LogDrop<'o> {
    fn as_true(&self) -> bool {
        true
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
