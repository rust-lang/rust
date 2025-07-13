struct Foo<T: ?Hash> {}
//~^ ERROR expected trait, found derive macro `Hash`
//~| ERROR bound modifier `?` can only be applied to `Sized`

fn main() {}

// https://github.com/rust-lang/rust/issues/37534
