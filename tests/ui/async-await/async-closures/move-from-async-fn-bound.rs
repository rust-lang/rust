//@ edition:2021
// Test that a by-ref `AsyncFn` closure gets an error when it tries to
// consume a value, with a helpful diagnostic pointing to the bound.

fn call<F>(_: F) where F: AsyncFn() {}

fn main() {
    let y = vec![format!("World")];
    call(async || {
        //~^ ERROR cannot move out of `y`, a captured variable in an `AsyncFn` closure
        y.into_iter();
    });
}
