trait Foo<'a, T, 'b> {}
//~^ ERROR lifetime parameters must be declared prior to type or const parameters

fn main() {}
