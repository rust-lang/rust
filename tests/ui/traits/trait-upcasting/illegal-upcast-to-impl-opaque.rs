//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ check-pass

trait Super {
    type Assoc;
}

trait Sub: Super {}

impl<T: ?Sized> Super for T {
    type Assoc = i32;
}

fn illegal(x: &dyn Sub<Assoc = i32>) -> &dyn Super<Assoc = impl Sized> {
    x
}

fn main() {}
