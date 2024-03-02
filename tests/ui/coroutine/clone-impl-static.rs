// gate-test-coroutine_clone
// Verifies that static coroutines cannot be cloned/copied.

#![feature(coroutines, coroutine_clone)]

fn main() {
    let gen = static move || {
        yield;
    };
    check_copy(&gen);
    //~^ ERROR the trait `Copy` is not implemented for
    check_clone(&gen);
    //~^ ERROR the trait `Clone` is not implemented for
}

fn check_copy<T: Copy>(_x: &T) {}
fn check_clone<T: Clone>(_x: &T) {}
