struct Struct<const N: usize>;

impl Struct<{ N }> {}
//~^ ERROR cannot find value `N` in this scope
//~| HELP you might be missing a const parameter

fn func0(_: Struct<{ N }>) {}
//~^ ERROR cannot find value `N` in this scope
//~| HELP you might be missing a const parameter

fn func1(_: [u8; N]) {}
//~^ ERROR cannot find value `N` in this scope
//~| HELP you might be missing a const parameter

fn func2<T>(_: [T; N]) {}
//~^ ERROR cannot find value `N` in this scope
//~| HELP you might be missing a const parameter

struct Image<const R: usize>([[u32; C]; R]);
//~^ ERROR cannot find value `C` in this scope
//~| HELP a const parameter with a similar name exists
//~| HELP you might be missing a const parameter

fn main() {}
