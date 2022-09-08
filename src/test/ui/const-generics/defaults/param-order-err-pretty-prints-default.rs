struct Foo<const M: usize = 10, 'a>(&'a u32);
//~^ ERROR lifetime parameters must be declared prior to type or const parameters

fn main() {}
