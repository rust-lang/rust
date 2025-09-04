//! Test for #145328: ensure the lifetime of a `super let` binding within an `if let` scrutinee is
//! at most the scope of the `if` condition's temporaries. Additionally, test `pin!` since it's
//! implemented in terms of `super let` and exposes this behavior.
//@ run-pass
//@ revisions: e2021 e2024
//@ [e2021] edition: 2021
//@ [e2024] edition: 2024

#![feature(if_let_guard)]
#![feature(super_let)]
#![expect(irrefutable_let_patterns)]

use std::cell::RefCell;
use std::pin::pin;

fn main() {
    // The `super let` bindings here should have the same scope as `if let` temporaries.
    // In Rust 2021, this means it lives past the end of the `if` expression.
    // In Rust 2024, this means it lives to the end of the `if`'s success block.
    assert_drop_order(0..=2, |o| {
        #[cfg(e2021)]
        (
            if let _ = { super let _x = o.log(2); } { o.push(0) },
            o.push(1),
        );
        #[cfg(e2024)]
        (
            if let _ = { super let _x = o.log(1); } { o.push(0) },
            o.push(2),
        );
    });
    assert_drop_order(0..=2, |o| {
        #[cfg(e2021)]
        (
            if let true = { super let _x = o.log(2); false } {} else { o.push(0) },
            o.push(1),
        );
        #[cfg(e2024)]
        (
            if let true = { super let _x = o.log(0); false } {} else { o.push(1) },
            o.push(2),
        );
    });

    // `pin!` should behave likewise.
    assert_drop_order(0..=2, |o| {
        #[cfg(e2021)] (if let _ = pin!(o.log(2)) { o.push(0) }, o.push(1));
        #[cfg(e2024)] (if let _ = pin!(o.log(1)) { o.push(0) }, o.push(2));
    });
    assert_drop_order(0..=2, |o| {
        #[cfg(e2021)]
        (
            if let None = Some(pin!(o.log(2))) {} else { o.push(0) },
            o.push(1),
        );
        #[cfg(e2024)]
        (
            if let None = Some(pin!(o.log(0))) {} else { o.push(1) },
            o.push(2),
        );
    });

    // `super let` bindings' scope should also be consistent with `if let` temporaries in guards.
    // Here, that means the `super let` binding in the second guard condition operand should be
    // dropped before the first operand's temporary. This is consistent across Editions.
    assert_drop_order(0..=1, |o| {
        match () {
            _ if let _ = o.log(1)
                && let _ = { super let _x = o.log(0); } => {}
            _ => unreachable!(),
        }
    });
    assert_drop_order(0..=1, |o| {
        match () {
            _ if let _ = o.log(1)
                && let _ = pin!(o.log(0)) => {}
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
