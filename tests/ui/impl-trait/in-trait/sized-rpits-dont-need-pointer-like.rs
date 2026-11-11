//@ check-pass

// Make sure that we don't enforce that an RPIT that has `where Self: Sized` is pointer-like.

trait Foo {
    fn foo() -> impl Sized where Self: Sized {}
}

impl Foo for () {}

fn main() {
    let x: &dyn Foo = &();
}
