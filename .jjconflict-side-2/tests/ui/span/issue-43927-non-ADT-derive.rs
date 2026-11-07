#![derive(Debug, PartialEq, Eq)] // should be an outer attribute!
//~^ ERROR `derive` attribute cannot be used at crate level
struct DerivedOn;

fn main() {}
