pub fn foo<T>() where T == u32 {}
//~^ ERROR type equality

fn main() {}
