// Check that we rely on super trait bounds when computing implied bounds for ADTs.
//
// check-pass
struct Foo<U> {
    bar: Bar<U>,
}

trait Trait: 'static {}
impl<T: 'static> Trait for T {}
struct Bar<T: Trait> {
    x: T,
}

fn main() {}
