#![feature(non_lifetime_binders, inherent_associated_types)]
#![expect(incomplete_features)]

// Test whether we can resolve to the right IAT when the self type
// contains a bound type. This should ideally use the second impl.

struct Foo<T: ?Sized>(T);

impl Foo<[u8]> {
    type IAT = u8;
}

impl<T: Sized> Foo<T> {
    type IAT = u8;
}

struct Bar
where
    for<T> Foo<T>::IAT: Sized;
    //~^ ERROR: multiple applicable items in scope

fn main() {}
