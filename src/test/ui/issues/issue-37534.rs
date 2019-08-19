struct Foo<T: ?Hash> { }
//~^ ERROR expected trait, found derive macro `Hash`
//~^^ ERROR parameter `T` is never used
//~^^^ WARN default bound relaxed for a type parameter, but this does nothing

fn main() { }
