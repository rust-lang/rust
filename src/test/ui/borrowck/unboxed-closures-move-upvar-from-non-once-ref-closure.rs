// Test that a by-ref `FnMut` closure gets an error when it tries to
// consume a value.

fn call<F>(f: F) where F : Fn() {
    f();
}

fn main() {
    let y = vec![format!("World")];
    call(|| {
        y.into_iter();
        //~^ ERROR cannot move out of `y`, a captured variable in an `Fn` closure
    });
}
