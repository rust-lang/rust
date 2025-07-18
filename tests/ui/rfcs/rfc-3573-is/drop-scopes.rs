//! Demonstrates where temporaries and bindings for `is` expressions are dropped.
//@ edition: 2024
//@ run-pass
//@ aux-crate: is_macro=is-macro.rs
#![feature(builtin_syntax)]
use is_macro::is;

use core::{cell::RefCell, ops::Drop};

#[expect(irrefutable_let_patterns)]
fn main() {
    // `is` used where a `let` expression is allowed desugars directly to `let`, using the scope of
    // the enclosing condition. As such, its temporaries and bindings behave the same as `let`'s.
    assert_drop_order(1..=6, |o| {
        if is!(o.log(5).ignore_temp().log(4) is _v)
            && is!(o.log(3).ignore_temp().log(2) is _v)
        {
            // The bindings and temporaries are all dropped after locals declared within the block.
            let _v = o.log(1);
        }
        // Everything is dropped before leaving the success block.
        o.log(6);
    });

    // `is` used elsewhere introduces a new scope.
    assert_drop_order(1..=9, |o| {
        // `let` isn't allowed through parentheses; the `is`'s bindings and temporaries only live to
        // the end of the `&&`-chain they're in.
        if (
            is!(o.log(4).ignore_temp().log(3) is _v)
                && is!(o.log(2).ignore_temp().log(1) is _v)
        ) && (
            is!(o.log(8).ignore_temp().log(7) is _v)
                && is!(o.log(6).ignore_temp().log(5) is _v)
        ) {
            o.log(9);
        }
    });
}

// # Test scaffolding...

struct DropOrder(RefCell<Vec<u64>>);
struct LogDrop<'o>(&'o DropOrder, u64);

impl DropOrder {
    fn log(&self, n: u64) -> LogDrop<'_> {
        LogDrop(self, n)
    }
    fn push(&self, n: u64) {
        self.0.borrow_mut().push(n);
    }
}

impl<'o> Drop for LogDrop<'o> {
    fn drop(&mut self) {
        self.0.push(self.1);
    }
}
impl<'o> LogDrop<'o> {
    fn ignore_temp(&self) -> &'o DropOrder {
        self.0
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
