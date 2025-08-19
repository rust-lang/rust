//@ check-pass

#![feature(inherent_associated_types, lazy_type_alias)]
#![expect(incomplete_features)]

// Test that we *do* normalize free aliases in order to resolve
// between multiple IAT candidates

type Free = u8;

struct Foo<T>(T);
impl Foo<u8> {
    type Assoc = u16;
}
impl Foo<u16> {
    type Assoc = u32;
}

struct Bar {
    field: <Foo<Free>>::Assoc,
}

fn main() {
    Bar {
        field: 1_u16,
    };
}
