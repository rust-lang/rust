//@ revisions: cpass1 cpass2
//@ edition: 2024
// regression test for https://github.com/rust-lang/rust/issues/158093

trait Super {
    type Assoc;
}

trait Sub: Super {}

impl<T: ?Sized> Super for T {
    type Assoc = i32;
}

fn illegal(x: &dyn Sub<Assoc = impl Sized>) -> &dyn Super<Assoc = impl Sized> {
    x
}

fn main() {}
