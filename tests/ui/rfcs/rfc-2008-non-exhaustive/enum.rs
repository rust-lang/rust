//@ aux-build:enums.rs
extern crate enums;

use enums::{EmptyNonExhaustiveEnum, NonExhaustiveEnum};

fn empty(x: EmptyNonExhaustiveEnum) {
    match x {} //~ ERROR type `EmptyNonExhaustiveEnum` is non-empty
    match x {
        _ => {}, // ok
    }
}

fn main() {
    let enum_unit = NonExhaustiveEnum::Unit;

    match enum_unit {
        //~^ ERROR non-exhaustive patterns: `_` not covered [E0004]
        NonExhaustiveEnum::Unit => "first",
        NonExhaustiveEnum::Tuple(_) => "second",
        NonExhaustiveEnum::Struct { .. } => "third"
    };

    match enum_unit {};
    //~^ ERROR non-exhaustive patterns: `_` not covered [E0004]

    // Everything below this is expected to compile successfully.

    let enum_unit = NonExhaustiveEnum::Unit;

    match enum_unit {
        NonExhaustiveEnum::Unit => 1,
        NonExhaustiveEnum::Tuple(_) => 2,
        // This particular arm tests that an enum marked as non-exhaustive
        // will not error if its variants are matched exhaustively.
        NonExhaustiveEnum::Struct { field } => field,
        _ => 0 // no error with wildcard
    };

    match enum_unit {
        _ => "no error with only wildcard"
    };

    // #53549: Check that variant constructors can still be called normally.
    match NonExhaustiveEnum::Unit {
        NonExhaustiveEnum::Unit => {},
        _ => {}
    };

    match NonExhaustiveEnum::Tuple(2) {
        NonExhaustiveEnum::Tuple(2) => {},
        _ => {}
    };

    match (NonExhaustiveEnum::Unit {}) {
        NonExhaustiveEnum::Unit {} => {},
        _ => {}
    };

    match (NonExhaustiveEnum::Tuple { 0: 2 }) {
        NonExhaustiveEnum::Tuple { 0: 2 } => {},
        _ => {}
    };

    match (NonExhaustiveEnum::Struct { field: 2 }) {
        NonExhaustiveEnum::Struct { field: 2 } => {},
        _ => {}
    };

}
