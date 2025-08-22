struct Foo<const N: [u8]>;
//~^ ERROR: `[u8]` is forbidden as the type of a const generic parameter

fn main() {}
