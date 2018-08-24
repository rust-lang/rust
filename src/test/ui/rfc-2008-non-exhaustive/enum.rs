// aux-build:enums.rs
extern crate enums;

use enums::NonExhaustiveEnum;

fn main() {
    let enum_unit = NonExhaustiveEnum::Unit;

    match enum_unit {
        //~^ ERROR non-exhaustive patterns: `_` not covered [E0004]
        NonExhaustiveEnum::Unit => "first",
        NonExhaustiveEnum::Tuple(_) => "second",
        NonExhaustiveEnum::Struct { .. } => "third"
    };
}
