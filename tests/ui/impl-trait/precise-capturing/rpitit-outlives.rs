//@ check-pass

// Ensure that we skip uncaptured args from RPITITs when collecting the regions
// to enforce member constraints in opaque type inference.

struct Invariant<T>(*mut T);

trait Foo {
    fn hello<'s: 's>(&'s self) -> Invariant<impl Sized + use<Self>>;
}

fn hello<'s, T: Foo>(x: &'s T) -> Invariant<impl Sized> {
    x.hello()
}

fn main() {}
