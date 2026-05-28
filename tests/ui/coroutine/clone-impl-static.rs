//@compile-flags: --diagnostic-width=300
// gate-test-coroutine_clone
// Verifies that static coroutines cannot be cloned/copied.
// This is important: the cloned coroutine would reference state of the original
// coroutine, leading to semantic nonsense.

#![feature(coroutines, coroutine_clone, stmt_expr_attributes)]

fn main() {
    let generator = #[coroutine]
    static move || {
        yield;
    };
    check_copy(&generator);
    //~^ ERROR Copy` is not satisfied
    check_clone(&generator);
    //~^ ERROR Clone` is not satisfied
}

fn check_copy<T: Copy>(_x: &T) {}
fn check_clone<T: Clone>(_x: &T) {}
