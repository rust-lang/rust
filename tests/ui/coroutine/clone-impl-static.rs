//@compile-flags: --diagnostic-width=300
// gate-test-coroutine_clone
// Verifies that static coroutines cannot be cloned/copied.

#![feature(coroutines, coroutine_clone, stmt_expr_attributes)]

fn main() {
    let gen = #[coroutine]
    static move || {
        yield;
    };
    check_copy(&gen);
    //~^ ERROR Copy` is not satisfied
    check_clone(&gen);
    //~^ ERROR Clone` is not satisfied
}

fn check_copy<T: Copy>(_x: &T) {}
fn check_clone<T: Clone>(_x: &T) {}
