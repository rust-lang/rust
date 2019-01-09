#![allow(dead_code)]

#![derive(Debug, PartialEq, Eq)] // should be an outer attribute!
//~^ ERROR `derive` may only be applied to structs, enums and unions
struct DerivedOn;

fn main() {}
