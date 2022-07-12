#![deny(unaligned_references)]

// Check that derive on a packed struct with non-Copy fields
// correctly. This can't be made to work perfectly because
// we can't just use the field from the struct as it might
// not be aligned.

// One error for `Clone`, two (identical) errors for `PartialEq`, one error for `Eq`.
#[derive(Copy, Clone, PartialEq, Eq)]
//~^ ERROR `#[derive]` can't be used on a `#[repr(packed)]` struct with type or const parameters
//~^^ ERROR `#[derive]` can't be used on a `#[repr(packed)]` struct with type or const parameters
//~^^^ ERROR `#[derive]` can't be used on a `#[repr(packed)]` struct with type or const parameters
//~^^^^ ERROR `#[derive]` can't be used on a `#[repr(packed)]` struct with type or const parameters
#[repr(packed)]
pub struct Foo<T>(T, T, T);

// One error for `Debug`.
#[derive(Debug)]
//~^ ERROR `#[derive]` can't be used on a `#[repr(packed)]` struct that does not derive Copy
#[repr(packed)]
pub struct Bar(u32, u32, u32);

#[derive(Default)]
struct Y(usize);

// Two different errors for `Default`. This used to be allowed because
// `default` is a static method, but no longer.
#[derive(Default)]
//~^ ERROR `#[derive]` can't be used on a `#[repr(packed)]` struct that does not derive Copy
//~^^ ERROR `#[derive]` can't be used on a `#[repr(packed)]` struct with type or const parameters
#[repr(packed)]
struct X<T: Default>(Y, T);

// One error for `Hash`. This used to be allowed because the alignment of the
// fields is 1, but no longer.
#[derive(Hash)]
//~^ ERROR `#[derive]` can't be used on a `#[repr(packed)]` struct that does not derive Copy
#[repr(packed)]
pub struct Baz(u8, bool);

fn main() {}
