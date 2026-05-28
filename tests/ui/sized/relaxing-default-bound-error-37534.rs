// issue: <https://github.com/rust-lang/rust/issues/37534>

struct Foo<T: ?Hash> {} //~ ERROR expected trait, found derive macro `Hash`

fn main() {}
