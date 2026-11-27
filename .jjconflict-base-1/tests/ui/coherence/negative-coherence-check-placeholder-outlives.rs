#![feature(negative_impls)]
#![feature(with_negative_coherence)]

struct Wrap<T>(T);

trait Foo {}
impl<T: 'static> !Foo for Box<T> {}

trait Bar {}
impl<T> Bar for T where T: Foo {}
impl<T> Bar for Box<T> {}
//~^ ERROR conflicting implementations of trait `Bar` for type `Box<_>`

fn main() {}
