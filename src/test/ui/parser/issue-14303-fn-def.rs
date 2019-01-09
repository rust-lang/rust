fn foo<'a, T, 'b>(x: &'a T) {}
//~^ ERROR lifetime parameters must be declared prior to type parameters

fn main() {}
