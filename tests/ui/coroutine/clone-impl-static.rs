// gate-test-coroutine_clone
// Verifies that static coroutines cannot be cloned/copied.

#![feature(coroutines, coroutine_clone)]

fn main() {
    let gen = static move || {
        yield;
    };
    check_copy(&gen);
    //~^ ERROR Copy` is not satisfied
    check_clone(&gen);
    //~^ ERROR Clone` is not satisfied
}

fn check_copy<T: Copy>(_x: &T) {}
fn check_clone<T: Clone>(_x: &T) {}
