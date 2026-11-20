struct Bar(u8);

struct Foo<const N: Bar>;
//~^ ERROR: `Bar` is forbidden as the type of a const generic parameter

fn main() {}
