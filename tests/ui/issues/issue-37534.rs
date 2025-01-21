struct Foo<T: ?Hash> {}
//~^ ERROR expected trait, found derive macro `Hash`
//~| ERROR relaxing a default bound only does something for `?Sized`

fn main() {}
