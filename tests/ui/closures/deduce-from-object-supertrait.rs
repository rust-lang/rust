//@ check-pass

// This test checks that we look at consider the super traits of trait objects
// when deducing closure signatures.

trait Foo: Fn(Bar) {}
impl<T> Foo for T where T: Fn(Bar) {}

struct Bar;
impl Bar {
    fn bar(&self) {}
}

fn main() {
    let x: &dyn Foo = &|x| {
        x.bar();
    };
}
