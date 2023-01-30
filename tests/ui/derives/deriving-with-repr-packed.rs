#![deny(unaligned_references)]

// Check that deriving certain builtin traits on certain packed structs cause
// errors. To avoid potentially misaligned references, field copies must be
// used, which involves adding `T: Copy` bounds.

#[derive(Copy, Clone, Default, PartialEq, Eq)]
#[repr(packed)]
pub struct Foo<T>(T, T, T);

// This one is fine because the fields all impl `Copy`.
#[derive(Default, Hash)]
#[repr(packed)]
pub struct Bar(u32, u32, u32);

// This one is fine because it's not packed.
#[derive(Debug, Default)]
struct Y(usize);

// This one has an error because `Y` doesn't impl `Copy`.
// Note: there is room for improvement in the error message.
#[derive(Debug, Default)]
#[repr(packed)]
struct X(Y);
//~^ ERROR cannot move out of `self` which is behind a shared reference

// This is currently allowed, but will be phased out at some point. From
// `zerovec` within icu4x-0.9.0.
#[derive(Debug)]
#[repr(packed)]
struct FlexZeroSlice {
    width: u8,
    data: [u8],
    //~^ WARNING byte slice in a packed struct that derives a built-in trait
    //~^^ this was previously accepted
}

fn main() {}
