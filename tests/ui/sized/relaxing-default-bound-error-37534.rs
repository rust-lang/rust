// issue: <https://github.com/rust-lang/rust/issues/37534>

struct Foo<T: ?Hash> {} //~ ERROR cannot find trait `Hash` in this scope

fn main() {}
