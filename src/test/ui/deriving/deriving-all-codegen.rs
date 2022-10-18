// check-pass
// compile-flags: -Zunpretty=expanded
// edition:2021
//
// This test checks the code generated for all[*] the builtin derivable traits
// on a variety of structs and enums. It protects against accidental changes to
// the generated code, and makes deliberate changes to the generated code
// easier to review.
//
// [*] It excludes `Copy` in some cases, because that changes the code
// generated for `Clone`.
//
// [*] It excludes `RustcEncodable` and `RustDecodable`, which are obsolete and
// also require the `rustc_serialize` crate.

#![crate_type = "lib"]
#![allow(dead_code)]
#![allow(deprecated)]

// Empty struct.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct Empty;

// A basic struct.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct Point {
    x: u32,
    y: u32,
}

// A large struct.
#[derive(Clone, Debug, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct Big {
    b1: u32, b2: u32, b3: u32, b4: u32, b5: u32, b6: u32, b7: u32, b8: u32,
}

// A struct with an unsized field. Some derives are not usable in this case.
#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct Unsized([u32]);

// A packed tuple struct that impls `Copy`.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[repr(packed)]
struct PackedCopy(u32);

// A packed tuple struct that does not impl `Copy`. Note that the alignment of
// the field must be 1 for this code to be valid. Otherwise it triggers an
// error "`#[derive]` can't be used on a `#[repr(packed)]` struct that does not
// derive Copy (error E0133)" at MIR building time. This is a weird case and
// it's possible that this struct is not supposed to work, but for now it does.
#[derive(Clone, Debug, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[repr(packed)]
struct PackedNonCopy(u8);

// An empty enum.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum Enum0 {}

// A single-variant enum.
#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum Enum1 {
    Single { x: u32 }
}

// A C-like, fieldless enum with a single variant.
#[derive(Clone, Debug, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum Fieldless1 {
    #[default]
    A,
}

// A C-like, fieldless enum.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum Fieldless {
    #[default]
    A,
    B,
    C,
}

// An enum with multiple fieldless and fielded variants.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum Mixed {
    #[default]
    P,
    Q,
    R(u32),
    S { d1: Option<u32>, d2: Option<i32> },
}

// An enum with no fieldless variants. Note that `Default` cannot be derived
// for this enum.
#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum Fielded {
    X(u32),
    Y(bool),
    Z(Option<i32>),
}

// A union. Most builtin traits are not derivable for unions.
#[derive(Clone, Copy)]
pub union Union {
    pub b: bool,
    pub u: u32,
    pub i: i32,
}
