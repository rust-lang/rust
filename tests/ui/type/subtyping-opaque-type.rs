//@ check-pass
//@ compile-flags: -Zvalidate-mir
trait Duh {}

impl Duh for i32 {}

trait Trait {
    type Assoc: Duh;
}

impl<R: Duh, F: FnMut() -> R> Trait for F {
    type Assoc = R;
}

fn foo() -> impl Trait<Assoc = impl Send> {
    || 42
}

fn main() {}
