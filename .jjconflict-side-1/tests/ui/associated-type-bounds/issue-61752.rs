//@ check-pass

trait Foo {
    type Bar;
}

impl Foo for () {
    type Bar = ();
}

fn a<F: Foo>() where F::Bar: Copy {}

fn b<F: Foo>() where <F as Foo>::Bar: Copy {}

// This used to complain about ambiguous associated types.
fn c<F: Foo<Bar: Foo>>() where F::Bar: Copy {}

fn main() {
    a::<()>();
    b::<()>();
    c::<()>();
}
