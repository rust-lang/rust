// aux-build:enums.rs
extern crate enums;

// ignore-pretty issue #37199

use enums::NonExhaustiveEnum;

fn main() {
    let enum_unit = NonExhaustiveEnum::Unit;

    match enum_unit {
        NonExhaustiveEnum::Unit => 1,
        NonExhaustiveEnum::Tuple(_) => 2,
        // This particular arm tests that a enum marked as non-exhaustive
        // will not error if its variants are matched exhaustively.
        NonExhaustiveEnum::Struct { field } => field,
        _ => 0 // no error with wildcard
    };

    match enum_unit {
        _ => "no error with only wildcard"
    };
}
