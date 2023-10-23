// Test that the `non_exhaustive_omitted_patterns` lint is triggered correctly.

#![feature(non_exhaustive_omitted_patterns_lint, unstable_test_feature)]

// aux-build:enums.rs
extern crate enums;
// aux-build:unstable.rs
extern crate unstable;
// aux-build:structs.rs
extern crate structs;

use enums::{
    EmptyNonExhaustiveEnum, NestedNonExhaustive, NonExhaustiveEnum, NonExhaustiveSingleVariant,
    VariantNonExhaustive,
};
use structs::{FunctionalRecord, MixedVisFields, NestedStruct, NormalStruct};
use unstable::{OnlyUnstableEnum, OnlyUnstableStruct, UnstableEnum, UnstableStruct};

#[non_exhaustive]
#[derive(Default)]
pub struct Foo {
    a: u8,
    b: usize,
    c: String,
}

#[non_exhaustive]
pub enum Bar {
    A,
    B,
    C,
}

fn main() {
    let enumeration = Bar::A;

    // Ok: this is a crate local non_exhaustive enum
    #[deny(non_exhaustive_omitted_patterns)]
    match enumeration {
        Bar::A => {}
        Bar::B => {}
        _ => {}
    }

    let non_enum = NonExhaustiveEnum::Unit;

    // Ok: without the attribute
    match non_enum {
        NonExhaustiveEnum::Unit => {}
        NonExhaustiveEnum::Tuple(_) => {}
        _ => {}
    }

    #[deny(non_exhaustive_omitted_patterns)]
    match non_enum {
        //~^ some variants are not matched explicitly
        NonExhaustiveEnum::Unit => {}
        NonExhaustiveEnum::Tuple(_) => {}
        _ => {}
    }

    #[deny(non_exhaustive_omitted_patterns)]
    match non_enum {
        //~^ some variants are not matched explicitly
        NonExhaustiveEnum::Unit | NonExhaustiveEnum::Struct { .. } => {}
        _ => {}
    }

    let x = 5;
    // We ignore the guard.
    #[deny(non_exhaustive_omitted_patterns)]
    match non_enum {
        NonExhaustiveEnum::Unit if x > 10 => {}
        NonExhaustiveEnum::Tuple(_) => {}
        NonExhaustiveEnum::Struct { .. } => {}
        _ => {}
    }

    #[deny(non_exhaustive_omitted_patterns)]
    match (non_enum, true) {
        (NonExhaustiveEnum::Unit, true) => {}
        (NonExhaustiveEnum::Tuple(_), false) => {}
        (NonExhaustiveEnum::Struct { .. }, false) => {}
        _ => {}
    }
    #[deny(non_exhaustive_omitted_patterns)]
    match (non_enum, true) {
        //~^ some variants are not matched explicitly
        (NonExhaustiveEnum::Unit, true) => {}
        (NonExhaustiveEnum::Tuple(_), false) => {}
        _ => {}
    }

    #[deny(non_exhaustive_omitted_patterns)]
    match (true, non_enum) {
        (true, NonExhaustiveEnum::Unit) => {}
        (false, NonExhaustiveEnum::Tuple(_)) => {}
        (false, NonExhaustiveEnum::Struct { .. }) => {}
        _ => {}
    }
    #[deny(non_exhaustive_omitted_patterns)]
    match (true, non_enum) {
        //~^ some variants are not matched explicitly
        (true, NonExhaustiveEnum::Unit) => {}
        (false, NonExhaustiveEnum::Tuple(_)) => {}
        _ => {}
    }

    #[deny(non_exhaustive_omitted_patterns)]
    match Some(non_enum) {
        //~^ some variants are not matched explicitly
        Some(NonExhaustiveEnum::Unit) => {}
        Some(NonExhaustiveEnum::Tuple(_)) => {}
        _ => {}
    }

    // Ok: all covered and not `unreachable-patterns`
    #[deny(unreachable_patterns)]
    #[deny(non_exhaustive_omitted_patterns)]
    match non_enum {
        NonExhaustiveEnum::Unit => {}
        NonExhaustiveEnum::Tuple(_) => {}
        NonExhaustiveEnum::Struct { .. } => {}
        _ => {}
    }

    #[deny(non_exhaustive_omitted_patterns)]
    match NestedNonExhaustive::B {
        //~^ some variants are not matched explicitly
        NestedNonExhaustive::A(NonExhaustiveEnum::Unit) => {}
        NestedNonExhaustive::A(_) => {}
        NestedNonExhaustive::B => {}
        _ => {}
    }

    #[warn(non_exhaustive_omitted_patterns)]
    match VariantNonExhaustive::Baz(1, 2) {
        VariantNonExhaustive::Baz(_, _) => {}
        VariantNonExhaustive::Bar { x, .. } => {}
    }
    //~^^ some fields are not explicitly listed

    #[warn(non_exhaustive_omitted_patterns)]
    let FunctionalRecord { first_field, second_field, .. } = FunctionalRecord::default();
    //~^ some fields are not explicitly listed

    // Ok: this is local
    #[warn(non_exhaustive_omitted_patterns)]
    let Foo { a, b, .. } = Foo::default();

    #[warn(non_exhaustive_omitted_patterns)]
    let NestedStruct { bar: NormalStruct { first_field, .. }, .. } = NestedStruct::default();
    //~^ some fields are not explicitly listed
    //~^^ some fields are not explicitly listed

    // Ok: this tests https://github.com/rust-lang/rust/issues/89382
    #[warn(non_exhaustive_omitted_patterns)]
    let MixedVisFields { a, b, .. } = MixedVisFields::default();

    // Ok: this only has 1 variant
    #[deny(non_exhaustive_omitted_patterns)]
    match NonExhaustiveSingleVariant::A(true) {
        NonExhaustiveSingleVariant::A(true) => {}
        _ => {}
    }

    // We can't catch the case below, so for consistency we don't catch this one either.
    #[deny(non_exhaustive_omitted_patterns)]
    match NonExhaustiveSingleVariant::A(true) {
        _ => {}
    }
    // We can't catch this case, because this would require digging fully through all the values of
    // any type we encounter. We need to be able to only consider present constructors.
    #[deny(non_exhaustive_omitted_patterns)]
    match &NonExhaustiveSingleVariant::A(true) {
        _ => {}
    }

    // Ok: we don't lint on `if let` expressions
    #[deny(non_exhaustive_omitted_patterns)]
    if let NonExhaustiveEnum::Tuple(_) = non_enum {}

    #[deny(non_exhaustive_omitted_patterns)]
    match UnstableEnum::Stable {
        //~^ some variants are not matched explicitly
        UnstableEnum::Stable => {}
        UnstableEnum::Stable2 => {}
        _ => {}
    }

    // Ok: the feature is on and all variants are matched
    #[deny(non_exhaustive_omitted_patterns)]
    match UnstableEnum::Stable {
        UnstableEnum::Stable => {}
        UnstableEnum::Stable2 => {}
        UnstableEnum::Unstable => {}
        _ => {}
    }

    // Ok: the feature is on and both variants are matched
    #[deny(non_exhaustive_omitted_patterns)]
    match OnlyUnstableEnum::Unstable {
        OnlyUnstableEnum::Unstable => {}
        OnlyUnstableEnum::Unstable2 => {}
        _ => {}
    }

    #[deny(non_exhaustive_omitted_patterns)]
    match OnlyUnstableEnum::Unstable {
        //~^ some variants are not matched explicitly
        OnlyUnstableEnum::Unstable => {}
        _ => {}
    }

    #[warn(non_exhaustive_omitted_patterns)]
    let OnlyUnstableStruct { unstable, .. } = OnlyUnstableStruct::new();
    //~^ some fields are not explicitly listed

    // OK: both unstable fields are matched with feature on
    #[warn(non_exhaustive_omitted_patterns)]
    let OnlyUnstableStruct { unstable, unstable2, .. } = OnlyUnstableStruct::new();

    #[warn(non_exhaustive_omitted_patterns)]
    let UnstableStruct { stable, stable2, .. } = UnstableStruct::default();
    //~^ some fields are not explicitly listed

    // OK: both unstable and stable fields are matched with feature on
    #[warn(non_exhaustive_omitted_patterns)]
    let UnstableStruct { stable, stable2, unstable, .. } = UnstableStruct::default();

    // Ok: local bindings are allowed
    #[deny(non_exhaustive_omitted_patterns)]
    let local = NonExhaustiveEnum::Unit;

    // Ok: missing patterns will be blocked by the pattern being refutable
    #[deny(non_exhaustive_omitted_patterns)]
    let local_refutable @ NonExhaustiveEnum::Unit = NonExhaustiveEnum::Unit;
    //~^ refutable pattern in local binding

    #[deny(non_exhaustive_omitted_patterns)]
    match &non_enum {
        //~^ some variants are not matched explicitly
        NonExhaustiveEnum::Unit => {}
        NonExhaustiveEnum::Tuple(_) => {}
        _ => {}
    }
}

#[deny(non_exhaustive_omitted_patterns)]
// Ok: Pattern in a param is always wildcard
pub fn takes_non_exhaustive(_: NonExhaustiveEnum) {
    let _closure = |_: NonExhaustiveEnum| {};
}

// ICE #117033
enum Void {}
#[deny(non_exhaustive_omitted_patterns)]
pub fn void(v: Void) -> ! {
    match v {}
}
