struct Foo<T, 'a>(&'a ());
//~^ ERROR lifetime parameters must be declared prior to
//~| ERROR parameter `T` is never used

fn main() {}
