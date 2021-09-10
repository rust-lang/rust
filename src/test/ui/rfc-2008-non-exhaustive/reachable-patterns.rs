// Test that the `non_exhaustive_omitted_patterns` lint is triggered correctly.

// aux-build:enums.rs
extern crate enums;

// aux-build:structs.rs
extern crate structs;

use enums::{
    EmptyNonExhaustiveEnum, NestedNonExhaustive, NonExhaustiveEnum, NonExhaustiveSingleVariant,
    VariantNonExhaustive,
};
use structs::{FunctionalRecord, NestedStruct, NormalStruct};

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
    match enumeration {
        Bar::A => {}
        Bar::B => {}
        #[deny(non_exhaustive_omitted_patterns)]
        _ => {}
    }

    let non_enum = NonExhaustiveEnum::Unit;

    // Ok: without the attribute
    match non_enum {
        NonExhaustiveEnum::Unit => {}
        NonExhaustiveEnum::Tuple(_) => {}
        _ => {}
    }

    match non_enum {
        NonExhaustiveEnum::Unit => {}
        NonExhaustiveEnum::Tuple(_) => {}
        #[deny(non_exhaustive_omitted_patterns)]
        _ => {}
    }
    //~^^ some variants are not matched explicitly

    match non_enum {
        NonExhaustiveEnum::Unit | NonExhaustiveEnum::Struct { .. } => {}
        #[deny(non_exhaustive_omitted_patterns)]
        _ => {}
    }
    //~^^ some variants are not matched explicitly

    let x = 5;
    match non_enum {
        NonExhaustiveEnum::Unit if x > 10 => {}
        NonExhaustiveEnum::Tuple(_) => {}
        NonExhaustiveEnum::Struct { .. } => {}
        #[deny(non_exhaustive_omitted_patterns)]
        _ => {}
    }
    //~^^ some variants are not matched explicitly

    // Ok: all covered and not `unreachable-patterns`
    #[deny(unreachable_patterns)]
    match non_enum {
        NonExhaustiveEnum::Unit => {}
        NonExhaustiveEnum::Tuple(_) => {}
        NonExhaustiveEnum::Struct { .. } => {}
        #[deny(non_exhaustive_omitted_patterns)]
        _ => {}
    }

    #[deny(non_exhaustive_omitted_patterns)]
    match NestedNonExhaustive::B {
        NestedNonExhaustive::A(NonExhaustiveEnum::Unit) => {}
        NestedNonExhaustive::A(_) => {}
        NestedNonExhaustive::B => {}
        _ => {}
    }
    //~^^ some variants are not matched explicitly
    //~^^^^^ some variants are not matched explicitly

    // The io::ErrorKind has many `unstable` fields how do they interact with this
    // lint
    #[deny(non_exhaustive_omitted_patterns)]
    match std::io::ErrorKind::Other {
        std::io::ErrorKind::NotFound => {}
        std::io::ErrorKind::PermissionDenied => {}
        std::io::ErrorKind::ConnectionRefused => {}
        std::io::ErrorKind::ConnectionReset => {}
        std::io::ErrorKind::ConnectionAborted => {}
        std::io::ErrorKind::NotConnected => {}
        std::io::ErrorKind::AddrInUse => {}
        std::io::ErrorKind::AddrNotAvailable => {}
        std::io::ErrorKind::BrokenPipe => {}
        std::io::ErrorKind::AlreadyExists => {}
        std::io::ErrorKind::WouldBlock => {}
        std::io::ErrorKind::InvalidInput => {}
        std::io::ErrorKind::InvalidData => {}
        std::io::ErrorKind::TimedOut => {}
        std::io::ErrorKind::WriteZero => {}
        std::io::ErrorKind::Interrupted => {}
        std::io::ErrorKind::Other => {}
        std::io::ErrorKind::UnexpectedEof => {}
        std::io::ErrorKind::Unsupported => {}
        std::io::ErrorKind::OutOfMemory => {}
        // All stable variants are above and unstable in `_`
        _ => {}
    }
    //~^^ some variants are not matched explicitly

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

    // Ok: because this only has 1 variant
    #[deny(non_exhaustive_omitted_patterns)]
    match NonExhaustiveSingleVariant::A(true) {
        NonExhaustiveSingleVariant::A(true) => {}
        _ => {}
    }

    #[deny(non_exhaustive_omitted_patterns)]
    match NonExhaustiveSingleVariant::A(true) {
        _ => {}
    }
    //~^^ some variants are not matched explicitly

    // Ok: we don't lint on `if let` expressions
    #[deny(non_exhaustive_omitted_patterns)]
    if let NonExhaustiveEnum::Tuple(_) = non_enum {}
}
