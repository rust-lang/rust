trait Trait {
    type Assoc;
}

struct Foo<T: Trait, U: ?Sized>(T, U);

impl<T: Trait<Assoc = U>, U: ?Sized> Drop for Foo<T, U> {
    //~^ ERROR: `Drop` impl requires `<T as Trait>::Assoc == U`
    fn drop(&mut self) {}
}

fn main() {}
