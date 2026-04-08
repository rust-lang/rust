//@ check-pass
#![feature(coroutines, coroutine_trait, stmt_expr_attributes, negative_impls)]

use std::ops::Coroutine;

struct Guard;
impl !Send for Guard {}
impl Drop for Guard {
    fn drop(&mut self) {}
}

fn lock() -> Guard {
    Guard
}

fn bar() -> impl Coroutine {
    #[coroutine]
    static || {
        let mut guard = lock();
        loop {
            drop(guard);
            yield;
            guard = lock();
        }
    }
}

fn main() {
    fn require_send<T: Send>(_: T) {}
    require_send(bar());
}
