#![allow(dead_code)]

#![derive(Debug, PartialEq, Eq)] // should be an outer attribute!
//~^ ERROR `derive` may only be applied to structs, enums and unions
//~| ERROR cannot determine resolution for the derive macro `Debug`
//~| ERROR cannot determine resolution for the derive macro `PartialEq`
//~| ERROR cannot determine resolution for the derive macro `Eq`
struct DerivedOn;

fn main() {}
