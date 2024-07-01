struct Foo<T: Trait>(T);

trait Trait {
    type Assoc;
}

impl<T: Trait<Assoc = U>, U: ?Sized> Drop for Foo<T> {
    //~^ ERROR: `Drop` impl requires `<T as Trait>::Assoc == U`
    fn drop(&mut self) {}
}

fn main() {}
