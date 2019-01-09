#![deny(safe_packed_borrows)]

// check that derive on a packed struct with non-Copy fields
// correctly. This can't be made to work perfectly because
// we can't just use the field from the struct as it might
// not be aligned.

#[derive(Copy, Clone, PartialEq, Eq)]
//~^ ERROR #[derive] can't be used
//~| hard error
//~^^^ ERROR #[derive] can't be used
//~| hard error
#[repr(packed)]
pub struct Foo<T>(T, T, T);

#[derive(PartialEq, Eq)]
//~^ ERROR #[derive] can't be used
//~| hard error
#[repr(packed)]
pub struct Bar(u32, u32, u32);

#[derive(PartialEq)]
struct Y(usize);

#[derive(PartialEq)]
//~^ ERROR #[derive] can't be used
//~| hard error
#[repr(packed)]
struct X(Y);

fn main() {}
