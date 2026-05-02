//! Test for #145784. This tests three things:
//!
//! - Since temporary lifetime extension applies to extending subexpressions in all contexts, it
//!   works through Rust 2024 block tail expressions: extending borrows and `super let`s in block
//!   tails are extended to outlive the result of the block.
//!
//! - Since `super let`'s initializer has the same temporary scope as the variable scope of its
//!   bindings, this means that lifetime extension can effectively see through `super let`.
//!
//! - In particular, the argument to `pin!` is an extending expression, and the argument of an
//!   extending `pin!` has an extended temporary scope. The lifetime of the argument, as well those
//!   of extending borrows and `super lets` within it, should match the result of the `pin!`,
//!   regardless of whether it itself is extended by a parent expression.
//!
//! For more information on temporary lifetime extension, see
//! https://doc.rust-lang.org/nightly/reference/destructors.html#temporary-lifetime-extension
//!
//! For tests that `super let` initializers aren't temporary drop scopes, and tests for
//! lifetime-extended `super let`, see tests/ui/borrowck/super-let-lifetime-and-drop.rs
//@ run-pass
//@ revisions: e2021 e2024
//@ [e2021] edition: 2021
//@ [e2024] edition: 2024

#![feature(super_let)]
#![allow(unused_braces)]

use std::cell::RefCell;
use std::pin::pin;

fn f<T>(_: LogDrop<'_>, x: T) -> T { x }

fn main() {
    // Test block arguments to non-extending `pin!`.
    // In Rust 2021, block tail expressions aren't temporary drop scopes, so their temporaries
    // should outlive the `pin!` invocation.
    // In Rust 2024, extending borrows within block tail expressions have extended lifetimes to
    // outlive result of the block, so the end result is the same in this case.
    // By nesting two `pin!` calls, this ensures extending borrows in the inner `pin!` outlive the
    // outer `pin!`.
    assert_drop_order(1..=3, |o| {
        (
            pin!((
                pin!({ &o.log(3) as *const LogDrop<'_> }),
                drop(o.log(1)),
            )),
            drop(o.log(2)),
        );
    });

    // The same holds for `super let` initializers in non-extending expressions.
    assert_drop_order(1..=4, |o| {
        (
            {
                super let _ = {
                    super let _ = { &o.log(4) as *const LogDrop<'_> };
                    drop(o.log(1))
                };
                drop(o.log(2))
            },
            drop(o.log(3)),
        );
    });

    // Within an extending expression, the argument to `pin!` is also an extending expression,
    // allowing borrow operands in block tail expressions to have extended lifetimes.
    assert_drop_order(1..=2, |o| {
        let _ = pin!({ &o.log(2) as *const LogDrop<'_> });
        drop(o.log(1));
    });

    // The same holds for `super let` initializers in extending expressions.
    assert_drop_order(1..=2, |o| {
        let _ =  { super let _ = { &o.log(2) as *const LogDrop<'_> }; };
        drop(o.log(1));
    });

    // We have extending borrow expressions within an extending block
    // expression (within an extending borrow expression) within a
    // non-extending expresion within the initializer expression.
    // These two should be the same.
    assert_drop_order(1..=3, |e| {
        let _v = f(e.log(1), &{ &raw const *&e.log(2) });
        drop(e.log(3));
    });
    assert_drop_order(1..=3, |e| {
        let _v = f(e.log(1), {
            super let v = &{ &raw const *&e.log(2) };
            v
        });
        drop(e.log(3));
    });

    // We have extending borrow expressions within a non-extending
    // expression within the initializer expression.
    //
    // These two should be the same.
    assert_drop_order(1..=3, |e| {
        let _v = f(e.log(1), &&raw const *&e.log(2));
        drop(e.log(3));
    });
    assert_drop_order(1..=3, |e| {
        let _v = f(e.log(1), {
            super let v = &&raw const *&e.log(2);
            v
        });
        drop(e.log(3));
    });

    // We have extending borrow expressions within an extending block
    // expression (within an extending borrow expression) within the
    // initializer expression.
    //
    // These two should be the same.
    assert_drop_order(1..=2, |e| {
        let _v = &{ &raw const *&e.log(2) };
        drop(e.log(1));
    });
    assert_drop_order(1..=2, |e| {
        let _v = {
            super let v = &{ &raw const *&e.log(2) };
            v
        };
        drop(e.log(1));
    });
}

// # Test scaffolding...

struct DropOrder(RefCell<Vec<u64>>);
struct LogDrop<'o>(&'o DropOrder, u64);

impl DropOrder {
    fn log(&self, n: u64) -> LogDrop<'_> {
        LogDrop(self, n)
    }
}

impl<'o> Drop for LogDrop<'o> {
    fn drop(&mut self) {
        self.0.0.borrow_mut().push(self.1);
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
