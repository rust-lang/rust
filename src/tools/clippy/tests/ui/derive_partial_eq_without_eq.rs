// run-rustfix

#![allow(unused)]
#![warn(clippy::derive_partial_eq_without_eq)]

// Don't warn on structs that aren't PartialEq
struct NotPartialEq {
    foo: u32,
    bar: String,
}

// Eq can be derived but is missing
#[derive(Debug, PartialEq)]
struct MissingEq {
    foo: u32,
    bar: String,
}

// Eq is derived
#[derive(PartialEq, Eq)]
struct NotMissingEq {
    foo: u32,
    bar: String,
}

// Eq is manually implemented
#[derive(PartialEq)]
struct ManualEqImpl {
    foo: u32,
    bar: String,
}

impl Eq for ManualEqImpl {}

// Cannot be Eq because f32 isn't Eq
#[derive(PartialEq)]
struct CannotBeEq {
    foo: u32,
    bar: f32,
}

// Don't warn if PartialEq is manually implemented
struct ManualPartialEqImpl {
    foo: u32,
    bar: String,
}

impl PartialEq for ManualPartialEqImpl {
    fn eq(&self, other: &Self) -> bool {
        self.foo == other.foo && self.bar == other.bar
    }
}

// Generic fields should be properly checked for Eq-ness
#[derive(PartialEq)]
struct GenericNotEq<T: Eq, U: PartialEq> {
    foo: T,
    bar: U,
}

#[derive(PartialEq)]
struct GenericEq<T: Eq, U: Eq> {
    foo: T,
    bar: U,
}

#[derive(PartialEq)]
struct TupleStruct(u32);

#[derive(PartialEq)]
struct GenericTupleStruct<T: Eq>(T);

#[derive(PartialEq)]
struct TupleStructNotEq(f32);

#[derive(PartialEq)]
enum Enum {
    Foo(u32),
    Bar { a: String, b: () },
}

#[derive(PartialEq)]
enum GenericEnum<T: Eq, U: Eq, V: Eq> {
    Foo(T),
    Bar { a: U, b: V },
}

#[derive(PartialEq)]
enum EnumNotEq {
    Foo(u32),
    Bar { a: String, b: f32 },
}

// Ensure that rustfix works properly when `PartialEq` has other derives on either side
#[derive(Debug, PartialEq, Clone)]
struct RustFixWithOtherDerives;

#[derive(PartialEq)]
struct Generic<T>(T);

#[derive(PartialEq, Eq)]
struct GenericPhantom<T>(core::marker::PhantomData<T>);

fn main() {}
