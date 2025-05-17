//@ check-pass

#![feature(inherent_associated_types)]
#![expect(incomplete_features)]

struct Foo<T>(T);
impl Foo<u8> {
    type Inherent = u8;
}
impl Foo<u16> {
    type Inherent = u32;
}

struct Bar {
    field: <Foo<u16>>::Inherent,
}

fn main() {
    Bar { field: 10_u32 };
}
