//@ compile-flags: -Znext-solver=coherence

// Makes sure we don't ICE on associated const projection when the feature gate
// is not enabled, since we should avoid encountering ICEs on stable if possible.

trait Bar {
    const ASSOC: usize;
}
impl Bar for () {
    const ASSOC: usize = 1;
}

trait Foo {}
impl Foo for () {}
impl<T> Foo for T where T: Bar<ASSOC = 0> {}
//~^ ERROR associated const equality is incomplete
//~| ERROR conflicting implementations of trait `Foo` for type `()`

fn main() {}
