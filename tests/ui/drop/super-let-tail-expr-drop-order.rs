//! Test for #145784: the argument to `pin!` should be treated as an extending expression if and
//! only if the whole `pin!` invocation is an extending expression. Likewise, since `pin!` is
//! implemented in terms of `super let`, test the same for `super let` initializers. Since the
//! argument to `pin!` and the initializer of `super let` are not temporary drop scopes, this only
//! affects lifetimes in two cases:
//!
//! - Block tail expressions in Rust 2024, which are both extending expressions and temporary drop
//! scopes; treating them as extending expressions within a non-extending `pin!` resulted in borrow
//! expression operands living past the end of the block.
//!
//! - Nested `super let` statements, which can have their binding and temporary lifetimes extended
//! when the block they're in is an extending expression.
//!
//! For more information on extending expressions, see
//! https://doc.rust-lang.org/reference/destructors.html#extending-based-on-expressions
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
    // Test block arguments to `pin!` in non-extending expressions.
    // In Rust 2021, block tail expressions aren't temporary drop scopes, so their temporaries
    // should outlive the `pin!` invocation.
    // In Rust 2024, block tail expressions are temporary drop scopes, so their temporaries should
    // be dropped after evaluating the tail expression within the `pin!` invocation.
    // By nesting two `pin!` calls, this ensures non-extended `pin!` doesn't extend an inner `pin!`.
    assert_drop_order(1..=3, |o| {
        #[cfg(e2021)]
        (
            pin!((
                pin!({ &o.log(3) as *const LogDrop<'_> }),
                drop(o.log(1)),
            )),
            drop(o.log(2)),
        );
        #[cfg(e2024)]
        (
            pin!((
                pin!({ &o.log(1) as *const LogDrop<'_> }),
                drop(o.log(2)),
            )),
            drop(o.log(3)),
        );
    });

    // The same holds for `super let` initializers in non-extending expressions.
    assert_drop_order(1..=4, |o| {
        #[cfg(e2021)]
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
        #[cfg(e2024)]
        (
            {
                super let _ = {
                    super let _ = { &o.log(1) as *const LogDrop<'_> };
                    drop(o.log(2))
                };
                drop(o.log(3))
            },
            drop(o.log(4)),
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
    #[cfg(e2021)]
    {
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
    }
    #[cfg(e2024)]
    {
        // These two should be the same.
        assert_drop_order(1..=3, |e| {
            let _v = f(e.log(2), &{ &raw const *&e.log(1) });
            drop(e.log(3));
        });
        assert_drop_order(1..=3, |e| {
            let _v = f(e.log(2), {
                super let v = &{ &raw const *&e.log(1) };
                v
            });
            drop(e.log(3));
        });
    }

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
