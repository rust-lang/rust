//@ edition: 2024
//@ run-pass

#![feature(temporary_rescoping)]
#![allow(unused, dropping_references)]

use std::temporary_rescoping::{scope, extend};

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
                // `extend!` can be used under `&`, like `scope!`, directly annotating the
                // expression it's changing the scope of. If we introduce an operator that makes an
                // expression use its lifetime-extended temporary scope, this would allow lifetime-
                // extension of temporaries created in other place expression contexts, such as the
                // scrutinee of a `match` expression or the receiver of a method call.
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

    // As mentioneed above, we currently lack a way to lifetime-extend without using `&` or `&mut`.
    // We could have an operator that extends an expression's temporary scope to the same scope that
    // would be used by `&` or `&mut`, but there's no known practical use for this yet.
    assert_drop_order(1..=2, |e| {
        'l: {
            // This `extend!` does nothing. We'd need another operator to apply lifetime-extension.
            match extend!('l => e.log(1)) {
                _ => {}
            }
            e.log(2);
        }
    });

    // `pin!` implemented in terms of `scope!` and `extend` doesn't fall victim to
    // <https://github.com/rust-lang/rust/issues/165322>, since it modifies the scopes of normal
    // temporaries, rather than of both temporaries and bindings. If we add an operator in the
    // future to give a binding a temporary's scope (or keep `super let` for this purpose), we'd
    // still need to be careful to treat them like temporaries for `if let` match guards.
    use std::pin::Pin;
    struct PinMacroHelper<T> { value: T }
    unsafe fn pin_new_unchecked_in_helper<'a, T>(
        pinned_ref: &'a mut PinMacroHelper<T>,
    ) -> Pin<&'a mut T> {
        unsafe { Pin::new_unchecked(&mut pinned_ref.value) }
    }
    macro_rules! pin {
        ($arg:expr) => {
            'l: {
                let pinned_ref: &mut PinMacroHelper<_> =
                    &mut scope!('l => extend!('l => PinMacroHelper { value: $arg }));
                unsafe { pin_new_unchecked_in_helper(pinned_ref) }
            }
        }
    }
    // make sure the pinned place is dropped in each sub-branch
    #[allow(irrefutable_let_patterns)]
    for b in [false, true] {
        assert_drop_order(1..=2, |e| {
            match b {
                false | true if let p = pin!(e.log(2)) => drop(e.log(1)),
                _ => {}
            }
        });
    }
    // make sure the re-scoped temporary is dropped in each sub-branch
    #[allow(irrefutable_let_patterns)]
    for b in [false, true] {
        assert_drop_order(1..=2, |e| {
            match b {
                false | true
                    if let _ = 'a: {
                        scope!('a => &e.log(2));
                        e.log(1);
                    } => {}
                _ => {}
            }
        });
    }
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
