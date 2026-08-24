//@ run-rustfix

#![allow(dead_code)]

// Extra semicolons after semicolon-terminated items should have a removal suggestion.

trait Factory {
    fn create() -> u32;;
    //~^ ERROR non-item in item list
    fn second() -> u32;
}

struct Local;

impl Local {
    const VALUE: u32 = 0;;
    //~^ ERROR non-item in item list
}

unsafe extern "C" {
    fn foreign();;
    //~^ ERROR non-item in item list
}

fn main() {}
