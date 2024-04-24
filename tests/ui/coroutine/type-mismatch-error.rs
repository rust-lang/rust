//! Test that we get the expected type mismatch error instead of "closure is expected to take 0
//! arguments" (which got introduced after implementing resume arguments).

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::Coroutine;

fn f<G: Coroutine>(_: G, _: G::Return) {}

fn main() {
    f(
        #[coroutine]
        |a: u8| {
            if false {
                yield ();
            } else {
                a
                //~^ error: `if` and `else` have incompatible types
            }
        },
        0u8,
    );
}
