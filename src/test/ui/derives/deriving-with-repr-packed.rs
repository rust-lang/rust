#![deny(unaligned_references)]

// Check that deriving certain builtin traits on certain packed structs cause
// errors. This happens when the derived trait would need to use a potentially
// misaligned reference. But there are two cases that are allowed:
// - If all the fields within the struct meet the required alignment: 1 for
//   `repr(packed)`, or `N` for `repr(packed(N))`.
// - If `Default` is the only trait derived, because it doesn't involve any
//   references.

#[derive(Copy, Clone, Default, PartialEq, Eq)]
//~^ ERROR `Clone` can't be derived on this `#[repr(packed)]` struct with type or const parameters
//~| hard error
//~^^^ ERROR `PartialEq` can't be derived on this `#[repr(packed)]` struct with type or const parameters
//~| hard error
#[repr(packed)]
pub struct Foo<T>(T, T, T);

#[derive(Default, Hash)]
//~^ ERROR `Hash` can't be derived on this `#[repr(packed)]` struct that does not derive `Copy`
//~| hard error
#[repr(packed)]
pub struct Bar(u32, u32, u32);

// This one is fine because the field alignment is 1.
#[derive(Default, Hash)]
#[repr(packed)]
pub struct Bar2(u8, i8, bool);

// This one is fine because the field alignment is 2, matching `packed(2)`.
#[derive(Default, Hash)]
#[repr(packed(2))]
pub struct Bar3(u16, i16, bool);

// This one is fine because it's not packed.
#[derive(Debug, Default)]
struct Y(usize);

#[derive(Debug, Default)]
//~^ ERROR `Debug` can't be derived on this `#[repr(packed)]` struct that does not derive `Copy`
//~| hard error
#[repr(packed)]
struct X(Y);

fn main() {}
