//@ check-pass

// Ensure that we skip uncaptured args from RPITITs when comptuing outlives.

struct Invariant<T>(*mut T);

trait Foo {
    fn hello<'s: 's>(&'s self) -> Invariant<impl Sized + use<Self>>;
}

fn outlives_static(_: impl Sized + 'static) {}

fn hello<'s, T: Foo + 'static>(x: &'s T) {
    outlives_static(x.hello());
}

fn main() {}
