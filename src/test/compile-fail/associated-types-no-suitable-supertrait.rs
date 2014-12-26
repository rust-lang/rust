#![feature(associated_types)]

// Check that we get an error when you use `<Self as Get>::Value` in
// the trait definition but `Self` does not, in fact, implement `Get`.

trait Get {
    type Value;
}

trait Other {
    fn uhoh<U:Get>(&self, foo: U, bar: <Self as Get>::Value) {}
    //~^ ERROR the trait `Get` is not implemented for the type `Self`
}

impl<T:Get> Other for T {
    fn uhoh<U:Get>(&self, foo: U, bar: <(T, U) as Get>::Value) {}
    //~^ ERROR the trait `Get` is not implemented for the type `(T, U)`
    //~| ERROR the trait `Get` is not implemented for the type `(T, U)`
}

fn main() { }
