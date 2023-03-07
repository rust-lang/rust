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

// A basic struct. Note: because this derives `Copy`, it gets the simple
// `clone` implemention that just does `*self`.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct Point {
    x: u32,
    y: u32,
}

// A basic packed struct. Note: because this derives `Copy`, it gets the simple
// `clone` implemention that just does `*self`.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[repr(packed)]
struct PackedPoint {
    x: u32,
    y: u32,
}

// A large struct. Note: because this derives `Copy`, it gets the simple
// `clone` implemention that just does `*self`.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct Big {
    b1: u32, b2: u32, b3: u32, b4: u32, b5: u32, b6: u32, b7: u32, b8: u32,
}

// A struct that doesn't impl `Copy`, which means it gets the non-simple
// `clone` implemention that clones the fields individually.
#[derive(Clone)]
struct NonCopy(u32);

// A packed struct that doesn't impl `Copy`, which means it gets the non-simple
// `clone` implemention that clones the fields individually.
#[derive(Clone)]
#[repr(packed)]
struct PackedNonCopy(u32);

// A struct that impls `Copy` manually, which means it gets the non-simple
// `clone` implemention that clones the fields individually.
#[derive(Clone)]
struct ManualCopy(u32);
impl Copy for ManualCopy {}

// A packed struct that impls `Copy` manually, which means it gets the
// non-simple `clone` implemention that clones the fields individually.
#[derive(Clone)]
#[repr(packed)]
struct PackedManualCopy(u32);
impl Copy for PackedManualCopy {}

// A struct with an unsized field. Some derives are not usable in this case.
#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct Unsized([u32]);

// A packed struct with an unsized `[u8]` field. This is currently allowed, but
// causes a warning and will be phased out at some point.
#[derive(Debug, Hash)]
#[repr(packed)]
struct PackedUnsizedU8([u8]);
//~^ WARNING byte slice in a packed struct that derives a built-in trait
//~^^ WARNING byte slice in a packed struct that derives a built-in trait
//~^^^ this was previously accepted
//~^^^^ this was previously accepted

trait Trait {
    type A;
}

// A generic struct involving an associated type.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct Generic<T: Trait, U> {
    t: T,
    ta: T::A,
    u: U,
}

// A packed, generic tuple struct involving an associated type. Because it is
// packed, a `T: Copy` bound is added to all impls (and where clauses within
// them) except for `Default`. This is because we must access fields using
// copies (e.g. `&{self.0}`), instead of using direct references (e.g.
// `&self.0`) which may be misaligned in a packed struct.
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[repr(packed)]
struct PackedGeneric<T: Trait, U>(T, T::A, U);

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

// A generic enum. Note that `Default` cannot be derived for this enum.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum EnumGeneric<T, U> {
    One(T),
    Two(U),
}

// A union. Most builtin traits are not derivable for unions.
#[derive(Clone, Copy)]
pub union Union {
    pub b: bool,
    pub u: u32,
    pub i: i32,
}
