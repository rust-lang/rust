struct Foo<T: ?Hash> {}
//~^ ERROR expected trait, found derive macro `Hash`
//~^^ ERROR parameter `T` is never used
//~^^^ WARN relaxing a default bound only does something for `?Sized`

fn main() {}
