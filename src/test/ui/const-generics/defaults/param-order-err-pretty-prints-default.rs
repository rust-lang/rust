struct Foo<const M: usize = 10, 'a>(&'a u32);
//~^ Error lifetime parameters must be declared prior to const parameters

fn main() {}
