#![feature(inherent_associated_types)]
#![expect(incomplete_features)]

trait Identity {
    type Assoc;
}
impl<T> Identity for T {
    type Assoc = T;
}

struct Foo<T>(T);
impl Foo<<u8 as Identity>::Assoc> {
    type Inherent = u8;
}
impl Foo<<u16 as Identity>::Assoc> {
    type Inherent = u32;
}

struct Bar {
    field: <Foo<u8>>::Inherent,
    //~^ ERROR: multiple applicable items in scope
}

fn main() {
    Bar { field: 10_u8 };
}
