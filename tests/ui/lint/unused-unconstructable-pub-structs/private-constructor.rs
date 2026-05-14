#![deny(unused_unconstructable_pub_structs)]

pub struct PrivateTuple(i32);
//~^ ERROR: struct `PrivateTuple` is never constructed

pub struct PrivateField {
//~^ ERROR: struct `PrivateField` is never constructed
    _field: i32,
}

pub struct MixedPhantom<T> {
//~^ ERROR: struct `MixedPhantom` is never constructed
    _marker: std::marker::PhantomData<T>,
    _field: i32,
}

fn main() {}
