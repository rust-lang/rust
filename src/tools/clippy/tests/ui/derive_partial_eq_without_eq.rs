#![allow(unused)]
#![warn(clippy::derive_partial_eq_without_eq)]

// Don't warn on structs that aren't PartialEq
pub struct NotPartialEq {
    foo: u32,
    bar: String,
}

// Eq can be derived but is missing
#[derive(Debug, PartialEq)]
pub struct MissingEq {
    foo: u32,
    bar: String,
}

// Check that we honor the `allow` attribute
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, PartialEq)]
pub struct AllowedMissingEq {
    foo: u32,
    bar: String,
}

// Check that we honor the `expect` attribute
#[expect(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, PartialEq)]
pub struct ExpectedMissingEq {
    foo: u32,
    bar: String,
}

// Eq is derived
#[derive(PartialEq, Eq)]
pub struct NotMissingEq {
    foo: u32,
    bar: String,
}

// Eq is manually implemented
#[derive(PartialEq)]
pub struct ManualEqImpl {
    foo: u32,
    bar: String,
}

impl Eq for ManualEqImpl {}

// Cannot be Eq because f32 isn't Eq
#[derive(PartialEq)]
pub struct CannotBeEq {
    foo: u32,
    bar: f32,
}

// Don't warn if PartialEq is manually implemented
pub struct ManualPartialEqImpl {
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
pub struct GenericNotEq<T: Eq, U: PartialEq> {
    foo: T,
    bar: U,
}

#[derive(PartialEq)]
pub struct GenericEq<T: Eq, U: Eq> {
    foo: T,
    bar: U,
}

#[derive(PartialEq)]
pub struct TupleStruct(u32);

#[derive(PartialEq)]
pub struct GenericTupleStruct<T: Eq>(T);

#[derive(PartialEq)]
pub struct TupleStructNotEq(f32);

#[derive(PartialEq)]
pub enum Enum {
    Foo(u32),
    Bar { a: String, b: () },
}

#[derive(PartialEq)]
pub enum GenericEnum<T: Eq, U: Eq, V: Eq> {
    Foo(T),
    Bar { a: U, b: V },
}

#[derive(PartialEq)]
pub enum EnumNotEq {
    Foo(u32),
    Bar { a: String, b: f32 },
}

// Ensure that rustfix works properly when `PartialEq` has other derives on either side
#[derive(Debug, PartialEq, Clone)]
pub struct RustFixWithOtherDerives;

#[derive(PartialEq)]
pub struct Generic<T>(T);

#[derive(PartialEq, Eq)]
pub struct GenericPhantom<T>(core::marker::PhantomData<T>);

mod _hidden {
    #[derive(PartialEq)]
    pub struct Reexported;

    #[derive(PartialEq)]
    pub struct InPubFn;

    #[derive(PartialEq)]
    pub(crate) struct PubCrate;

    #[derive(PartialEq)]
    pub(super) struct PubSuper;
}

pub use _hidden::Reexported;
pub fn _from_mod() -> _hidden::InPubFn {
    _hidden::InPubFn
}

#[derive(PartialEq)]
struct InternalTy;

// This is a `non_exhaustive` type so should not warn.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub struct MissingEqNonExhaustive {
    foo: u32,
    bar: String,
}

// This is a `non_exhaustive` type so should not warn.
#[derive(Debug, PartialEq)]
pub struct MissingEqNonExhaustive1 {
    foo: u32,
    #[non_exhaustive]
    bar: String,
}

// This is a `non_exhaustive` type so should not warn.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum MissingEqNonExhaustive2 {
    Foo,
    Bar,
}

// This is a `non_exhaustive` type so should not warn.
#[derive(Debug, PartialEq)]
pub enum MissingEqNonExhaustive3 {
    Foo,
    #[non_exhaustive]
    Bar,
}

mod struct_gen {
    // issue 9413
    pub trait Group {
        type Element: Eq + PartialEq;
    }

    pub trait Suite {
        type Group: Group;
    }

    #[derive(PartialEq)]
    //~^ ERROR: you are deriving `PartialEq` and can implement `Eq`
    pub struct Foo<C: Suite>(<C::Group as Group>::Element);

    #[derive(PartialEq, Eq)]
    pub struct Bar<C: Suite>(i32, <C::Group as Group>::Element);

    // issue 9319
    #[derive(PartialEq)]
    //~^ ERROR: you are deriving `PartialEq` and can implement `Eq`
    pub struct Oof<T: Fn()>(T);

    #[derive(PartialEq, Eq)]
    pub struct Rab<T: Fn()>(T);
}

fn main() {}
