struct Foo<T, 'a>(&'a ());
//~^ ERROR lifetime parameters must be declared prior to

fn main() {}
