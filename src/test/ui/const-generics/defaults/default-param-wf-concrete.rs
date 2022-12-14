struct Foo<const N: u8 = { 255 + 1 }>;
//~^ ERROR evaluation of constant value failed
fn main() {}
