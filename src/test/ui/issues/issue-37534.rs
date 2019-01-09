struct Foo<T: ?Hash> { }
//~^ ERROR cannot find trait `Hash` in this scope
//~^^ ERROR parameter `T` is never used
//~^^^ WARN default bound relaxed for a type parameter, but this does nothing

fn main() { }
