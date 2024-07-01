//@ check-pass

struct Foo<T: Trait>(T);

trait Trait {
    type Assoc;
}

impl<T: Trait<Assoc = U>, U: ?Sized> Drop for Foo<T> {
    fn drop(&mut self) {}
}

fn main() {}
