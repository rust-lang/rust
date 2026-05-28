//@ revisions: normal lint
// Test that putting the lint level on a match arm emits a warning, as this was previously
// meaningful and is no longer.
#![feature(non_exhaustive_omitted_patterns_lint)]

//@ aux-build:enums.rs
extern crate enums;

use enums::NonExhaustiveEnum;

fn main() {
    let val = NonExhaustiveEnum::Unit;

    #[deny(non_exhaustive_omitted_patterns)]
    match val {
        //~^ ERROR some variants are not matched explicitly
        NonExhaustiveEnum::Unit => {}
        NonExhaustiveEnum::Tuple(_) => {}
        _ => {}
    }

    #[cfg_attr(lint, deny(non_exhaustive_omitted_patterns))]
    match val {
        //[lint]~^ ERROR some variants are not matched explicitly
        NonExhaustiveEnum::Unit => {}
        NonExhaustiveEnum::Tuple(_) => {}
        _ => {}
    }

    match val {
        NonExhaustiveEnum::Unit => {}
        NonExhaustiveEnum::Tuple(_) => {}
        #[deny(non_exhaustive_omitted_patterns)]
        _ => {}
    }
    //~^^ WARN lint level must be set on the whole match

    match val {
        NonExhaustiveEnum::Unit => {}
        NonExhaustiveEnum::Tuple(_) => {}
        #[cfg_attr(lint, deny(non_exhaustive_omitted_patterns))]
        _ => {}
    }
    //[lint]~^^ WARN lint level must be set on the whole match

    match val {
        NonExhaustiveEnum::Unit => {}
        NonExhaustiveEnum::Tuple(_) => {}
        #[cfg_attr(lint, warn(non_exhaustive_omitted_patterns))]
        _ => {}
    }
    //[lint]~^^ WARN lint level must be set on the whole match
}
