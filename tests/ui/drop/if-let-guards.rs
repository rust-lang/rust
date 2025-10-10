//! Tests drop order for `if let` guard bindings and temporaries. This is for behavior specific to
//! `match` expressions, whereas `tests/ui/drop/drop-order-comparisons.rs` compares `let` chains in
//! guards to `let` chains in `if` expressions. Drop order for `let` chains in guards shouldn't
//! differ between Editions, so we test on both 2021 and 2024, expecting the same results.
//@ revisions: e2021 e2024
//@ [e2021] edition: 2021
//@ [e2024] edition: 2024
//@ run-pass

#![feature(if_let_guard)]
#![deny(rust_2024_compatibility)]

use core::{cell::RefCell, ops::Drop};

fn main() {
    // Test that `let` guard bindings and temps are dropped before the arm's pattern's bindings.
    assert_drop_order(1..=6, |o| {
        // We move out of the scrutinee, so the drop order of the array's elements are based on
        // binding declaration order, and they're dropped in the arm's scope.
        match [o.log(6), o.log(5)] {
            // Partially move from the guard temporary to test drops both for temps and the binding.
            [_x, _y] if let [_, _z, _] = [o.log(4), o.log(2), o.log(3)]
                && true => { let _a = o.log(1); }
            _ => unreachable!(),
        }
    });

    // Sanity check: we don't move out of the match scrutinee when the guard fails.
    assert_drop_order(1..=4, |o| {
        // The scrutinee uses the drop order for arrays since its elements aren't moved.
        match [o.log(3), o.log(4)] {
            [_x, _y] if let _z = o.log(1)
                && false => unreachable!(),
            _ => { let _a = o.log(2); }
        }
    });

    // Test `let` guards' temporaries are dropped immediately when a guard fails, even if the guard
    // is lowered and run multiple times on the same arm due to or-patterns.
    assert_drop_order(1..=8, |o| {
        let mut _x = 1;
        // The match's scrutinee isn't bound by-move, so it outlives the match.
        match o.log(8) {
            // Failing a guard breaks out of the arm's scope, dropping the `let` guard's scrutinee.
            _ | _ | _ if let _ = o.log(_x)
                && { _x += 1; false } => unreachable!(),
            // The temporaries from a failed guard are dropped before testing the next guard.
            _ if let _ = o.log(5)
                && { o.push(4); false } => unreachable!(),
            // If the guard succeeds, we stay in the arm's scope to execute its body.
            _ if let _ = o.log(7)
                && true => { o.log(6); }
            _ => unreachable!(),
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
