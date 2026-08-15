impl<const A: i32 = { || [0; B] }> Tr {}
//~^ ERROR cannot find type `Tr`
//~| ERROR cannot find value `B`
//~| ERROR defaults for generic parameters are not allowed here

fn main() {}
