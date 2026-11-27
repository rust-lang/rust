use std::marker::CoercePointee; //~ ERROR use of unstable library feature `derive_coerce_pointee`

#[derive(CoercePointee)] //~ ERROR use of unstable library feature `derive_coerce_pointee`
#[repr(transparent)]
struct MyPointer<'a, #[pointee] T: ?Sized> {
    ptr: &'a T,
}

fn main() {}
