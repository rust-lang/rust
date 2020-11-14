#![derive(Debug, PartialEq, Eq)] // should be an outer attribute!
//~^ ERROR cannot determine resolution for the attribute macro `derive`
struct DerivedOn;

fn main() {}
