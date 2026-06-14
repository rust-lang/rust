// A regression test for https://github.com/rust-lang/rust/issues/151637.

struct Wrap<T: 'static>(T);

fn error1<T>(x: T) {
    Wrap(x);
    //~^ ERROR: the parameter type `T` may not live long enough
}

// We used to add implied bound `T: 'static` from the closure's return type
// when borrow checking the closure, which resulted in allowing non-wf return
// type. Implied bounds which contains only the external regions or type params
// from the parents should not be implied. That would make those bounds in needs
// propagated as closure requirements and either proved or make borrowck error
// from the parent's body.
fn error2<T>(x: T) {
    || Wrap(x);
    //~^ ERROR: the parameter type `T` may not live long enough
}

fn no_error1<T: 'static>(x: T) {
    || Wrap(x);
}

fn no_error2<T>(x: T, _: &'static T) {
    || Wrap(x);
}

fn main() {}
