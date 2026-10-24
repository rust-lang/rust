const ONE: [u16] = [1];
//~^ ERROR the size for values of type `[u16]` cannot be known at compilation time
//~| ERROR mismatched types

const TWO: &'static u16 = &ONE[0];
//~^ ERROR cannot move a value of type `[u16]`

fn main() {}
