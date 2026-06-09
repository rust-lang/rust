//@ pp-exact

// Check that the visibility is printed on an enum variant.

fn main() {}

#[cfg(false)]
enum Foo { pub V, }
