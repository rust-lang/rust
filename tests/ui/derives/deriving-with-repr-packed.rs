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
//~^ ERROR cannot move out of a shared reference [E0507]

#[derive(Debug)]
#[repr(packed)]
struct FlexZeroSlice {
    width: u8,
    data: [u8],
    //~^ ERROR cannot move
    //~| ERROR cannot move
}

#[derive(Debug)]
#[repr(packed)]
struct WithStr {
    width: u8,
    data: str,
    //~^ ERROR cannot move
    //~| ERROR cannot move
}

fn main() {}
