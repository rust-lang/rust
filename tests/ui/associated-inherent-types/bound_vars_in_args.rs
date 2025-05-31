#![feature(non_lifetime_binders, inherent_associated_types)]
#![expect(incomplete_features)]

// Test that we can resolve to the right IAT when the self type
// contains a bound type.

struct Foo<T: ?Sized>(T);

impl Foo<[u8]> {
    type IAT = u8;
}

impl<T: Sized> Foo<T> {
    type IAT = u8;
}

struct Bar
//~^ ERROR: the size for values of type `T` cannot be known at compilation time
//~| ERROR: the size for values of type `T` cannot be known at compilation time
where
    for<T> Foo<T>::IAT: Sized;

fn main() {}
