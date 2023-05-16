// Test that the `non_exhaustive_omitted_patterns` lint is triggered correctly.

#![feature(non_exhaustive_omitted_patterns_lint, unstable_test_feature)]
#![deny(unreachable_patterns)]

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

fn no_lint() {
    let non_enum = NonExhaustiveEnum::Unit;
    // Ok: without the attribute
    match non_enum {
        NonExhaustiveEnum::Unit => {}
        NonExhaustiveEnum::Tuple(_) => {}
        _ => {}
    }
}

#[warn(non_exhaustive_omitted_patterns)]
fn main() {
    let enumeration = Bar::A;

    // Ok: this is a crate local non_exhaustive enum
    match enumeration {
        Bar::A => {}
        Bar::B => {}
        _ => {}
    }

    let non_enum = NonExhaustiveEnum::Unit;

    match non_enum {
        NonExhaustiveEnum::Unit => {}
        NonExhaustiveEnum::Tuple(_) => {}
        _ => {}
    }
    //~^^ some variants are not matched explicitly

    match non_enum {
        NonExhaustiveEnum::Unit | NonExhaustiveEnum::Struct { .. } => {}
        _ => {}
    }
    //~^^ some variants are not matched explicitly

    let x = 5;
    match non_enum {
        NonExhaustiveEnum::Unit if x > 10 => {}
        NonExhaustiveEnum::Tuple(_) => {}
        NonExhaustiveEnum::Struct { .. } => {}
        _ => {}
    }
    //~^^ some variants are not matched explicitly

    // Ok: all covered and not `unreachable-patterns`
    match non_enum {
        NonExhaustiveEnum::Unit => {}
        NonExhaustiveEnum::Tuple(_) => {}
        NonExhaustiveEnum::Struct { .. } => {}
        _ => {}
    }

    match NestedNonExhaustive::B {
        NestedNonExhaustive::A(NonExhaustiveEnum::Unit) => {}
        NestedNonExhaustive::A(_) => {}
        NestedNonExhaustive::B => {}
        _ => {}
    }
    //~^^ some variants are not matched explicitly
    //~^^^^^ some variants are not matched explicitly

    match VariantNonExhaustive::Baz(1, 2) {
        VariantNonExhaustive::Baz(_, _) => {}
        VariantNonExhaustive::Bar { x, .. } => {}
    }
    //~^^ some fields are not explicitly listed

    let FunctionalRecord { first_field, second_field, .. } = FunctionalRecord::default();
    //~^ some fields are not explicitly listed

    // Ok: this is local
    let Foo { a, b, .. } = Foo::default();

    let NestedStruct { bar: NormalStruct { first_field, .. }, .. } = NestedStruct::default();
    //~^ some fields are not explicitly listed
    //~^^ some fields are not explicitly listed

    // Ok: this tests https://github.com/rust-lang/rust/issues/89382
    let MixedVisFields { a, b, .. } = MixedVisFields::default();

    // Ok: because this only has 1 variant
    match NonExhaustiveSingleVariant::A(true) {
        NonExhaustiveSingleVariant::A(true) => {}
        _ => {}
    }

    // No variants are mentioned
    match NonExhaustiveSingleVariant::A(true) {
        _ => {}
    }
    //~^^ some variants are not matched explicitly
    match Some(NonExhaustiveSingleVariant::A(true)) {
        Some(_) => {}
        None => {}
    }
    match Some(&NonExhaustiveSingleVariant::A(true)) {
        Some(_) => {}
        None => {}
    }

    // Ok: we don't lint on `if let` expressions
    if let NonExhaustiveEnum::Tuple(_) = non_enum {}

    match UnstableEnum::Stable {
        UnstableEnum::Stable => {}
        UnstableEnum::Stable2 => {}
        _ => {}
    }
    //~^^ some variants are not matched explicitly

    // Ok: the feature is on and all variants are matched
    match UnstableEnum::Stable {
        UnstableEnum::Stable => {}
        UnstableEnum::Stable2 => {}
        UnstableEnum::Unstable => {}
        _ => {}
    }

    // Ok: the feature is on and both variants are matched
    match OnlyUnstableEnum::Unstable {
        OnlyUnstableEnum::Unstable => {}
        OnlyUnstableEnum::Unstable2 => {}
        _ => {}
    }

    match OnlyUnstableEnum::Unstable {
        OnlyUnstableEnum::Unstable => {}
        _ => {}
    }
    //~^^ some variants are not matched explicitly

    let OnlyUnstableStruct { unstable, .. } = OnlyUnstableStruct::new();
    //~^ some fields are not explicitly listed

    // OK: both unstable fields are matched with feature on
    let OnlyUnstableStruct { unstable, unstable2, .. } = OnlyUnstableStruct::new();

    let UnstableStruct { stable, stable2, .. } = UnstableStruct::default();
    //~^ some fields are not explicitly listed

    // OK: both unstable and stable fields are matched with feature on
    let UnstableStruct { stable, stable2, unstable, .. } = UnstableStruct::default();

    // Ok: local bindings are allowed
    let local = NonExhaustiveEnum::Unit;

    // Ok: missing patterns will be blocked by the pattern being refutable
    let local_refutable @ NonExhaustiveEnum::Unit = NonExhaustiveEnum::Unit;
    //~^ refutable pattern in local binding

    // Check that matching on a reference results in a correctly spanned diagnostic
    match &non_enum {
        NonExhaustiveEnum::Unit => {}
        NonExhaustiveEnum::Tuple(_) => {}
        _ => {}
    }
    //~^^ some variants are not matched explicitly

    match (true, &non_enum) {
        (true, NonExhaustiveEnum::Unit) => {}
        _ => {}
    }

    match (&non_enum, true) {
        (NonExhaustiveEnum::Unit, true) => {}
        _ => {}
    }
    //~^^ some variants are not matched explicitly

    match Some(&non_enum) {
        Some(NonExhaustiveEnum::Unit | NonExhaustiveEnum::Tuple(_)) => {}
        _ => {}
    }
}

#[deny(non_exhaustive_omitted_patterns)]
// Ok: Pattern in a param is always wildcard
pub fn takes_non_exhaustive(_: NonExhaustiveEnum) {
    let _closure = |_: NonExhaustiveEnum| {};
}
