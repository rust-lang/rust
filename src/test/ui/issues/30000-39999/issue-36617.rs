#![derive(Copy)] //~ ERROR `derive` may only be applied to structs, enums and unions
                 //~| ERROR cannot determine resolution for the derive macro `Copy`

fn main() {}
