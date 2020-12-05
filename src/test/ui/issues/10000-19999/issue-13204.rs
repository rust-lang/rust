// run-pass
#![allow(unused_mut)]
// Test that when instantiating trait default methods, typeck handles
// lifetime parameters defined on the method bound correctly.


pub trait Foo {
    fn bar<'a, I: Iterator<Item=&'a ()>>(&self, it: I) -> usize {
        let mut xs = it.filter(|_| true);
        xs.count()
    }
}

pub struct Baz;

impl Foo for Baz {
    // When instantiating `Foo::bar` for `Baz` here, typeck used to
    // ICE due to the lifetime parameter of `bar`.
}

fn main() {
    let x = Baz;
    let y = vec![(), (), ()];
    assert_eq!(x.bar(y.iter()), 3);
}
