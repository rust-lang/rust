#![deny(warnings)]

#![doc(test(""))]
//~^ ERROR `#![doc(test(...)]` does not take a literal
//~^^ WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

fn main() {}
