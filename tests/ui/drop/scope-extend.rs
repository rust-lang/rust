// TODO: systematically add more tests, clean up tests, split up tests, reframe comments
//@ edition: 2024
//@ run-pass

#![feature(super_let)]
#![allow(unused, dropping_references)]

fn main() {
    // `scope!` sets the enclosing temporary scope to that of the labeled expression, affecting its
    // operand's temporary scope and potentially the temporary scopes of its subexpressions.
    assert_drop_order(1..=7, |e| {
        (
            'l: {
                &scope!('l => e.log(6));
                scope!('l => &e.log(5));
                match scope!('l => e.log(4)) {
                    _ => {}
                }
                scope!('l => match e.log(3) {
                    _ => {}
                });
                // Long-lived temporaries aren't created here.
                scope!('l => e.log(1));
            },
            drop(e.log(2)),
        );
        drop(e.log(7));
    });

    // `extend!` sets the temporary lifetime used by the `&` operator to that of a parent context.
    // Absent lifetime extension, it will be the temporary scope enclosing that parent.
    assert_drop_order(1..=6, |e| {
        {
            let x = 'l: {
                // Lifetime extension is interrupted for function arguments.
                let y = extend!('l => (&e.log(5), drop(&e.log(1))));
                e.log(2);
                // `extend!` can be used under `&`, like `scope!`. In general, I'd like re-scoping
                // operators to annotate the exact expression we want to modify the scope of.
                let z = &extend!('l => e.log(4));
                (y, z)
            };
            x;
            e.log(3);
        }
        e.log(6);
    });
    assert_drop_order(1..=6, |e| {
        (
            'l: {
                // The temporary scope used by the operand to a borrow operator at `'l` would be the
                // tuple expression. As such, `extend!('l => ...)` extends borrows to that scope.
                extend!('l => (&e.log(5), drop(&e.log(1))));
                e.log(2);
                &extend!('l => e.log(4));
            },
            drop(e.log(3)),
        );
        e.log(6);
    });

    // `scope!`'s operand is non-extending. Combine it with `extend!` for lifetime-extension.
    assert_drop_order(1..=7, |e| {
        {
            let x = 'l: {
                // These temporaries will be dropped at the end of the parent `let` statement.
                let _ = &scope!('l => e.log(3));
                let _ = scope!('l => &e.log(2));
                // These lifetime-extended temporaries will live to the end of the parent block.
                let _ = &scope!('l => extend!('l => e.log(6)));
                let _ = scope!('l => extend!('l => &e.log(5)));
                e.log(1);
            };
            e.log(4);
        }
        e.log(7);
    });

    // We currently lack a way to lifetime-extend without using `&` or `&mut`. This means we can't
    // lifetime-extend match scrutinees or method recievers. If that's something we want, we could
    // have an operator that extends an expression's temporary scope (to the same scope that would
    // be used by `&` or `&mut`). However, there's not yet a known use for this.
    assert_drop_order(1..=2, |e| {
        'l: {
            // This `extend!` does nothing. We'd need another operator to apply lifetime-extension.
            match extend!('l => e.log(1)) {
                _ => {}
            }
            e.log(2);
        }
    });
}

// Test scaffolding

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
        self.0.0.borrow_mut().push(self.1);
    }
}

#[track_caller]
fn assert_drop_order(ex: impl IntoIterator<Item = u64>, f: impl Fn(&DropOrder)) {
    let order = DropOrder(RefCell::new(Vec::new()));
    f(&order);
    let order = order.0.into_inner();
    let expected: Vec<u64> = ex.into_iter().collect();
    assert_eq!(order, expected);
}
