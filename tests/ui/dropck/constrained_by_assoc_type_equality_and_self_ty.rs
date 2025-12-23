trait Trait {
    type Assoc;
    //~^ ERROR: `Drop` impl requires `U: Sized` but the struct it is implemented for does not [E0367]
}

struct Foo<T: Trait, U: ?Sized>(T, U);

impl<T: Trait<Assoc = U>, U: ?Sized> Drop for Foo<T, U> {
    //~^ ERROR: `Drop` impl requires `<T as Trait>::Assoc == U`
    fn drop(&mut self) {}
}

fn main() {}
