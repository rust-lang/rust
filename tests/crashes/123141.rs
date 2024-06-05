//@ known-bug: #123141

trait Trait {
    fn next(self) -> Self::Item;
    type Item;
}

struct Foo<T: ?Sized>(T);

impl<T: ?Sized, U> Trait for Foo<U> {
    type Item = Foo<T>;
    fn next(self) -> Self::Item {
        loop {}
    }
}

fn opaque() -> impl Trait {
    Foo::<_>(10_u32)
}

fn main() {
    opaque().next();
}
