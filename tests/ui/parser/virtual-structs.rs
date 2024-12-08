// Test diagnostics for the removed struct inheritance feature.

virtual struct SuperStruct {
//~^ ERROR expected item, found reserved keyword `virtual`
    f1: isize,
}

struct Struct : SuperStruct;

pub fn main() {}
