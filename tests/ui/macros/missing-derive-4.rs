// aux-build:serde.rs
// compile-flags:--extern serde --crate-type bin --edition 2021

// derive macros not imported, but namespace imported
use serde;

#[serde(untagged)] //~ ERROR cannot find attribute `serde`
enum A {
    A,
    B,
}

enum B {
    A,
    #[serde(untagged)] //~ ERROR cannot find attribute `serde`
    B,
}

enum C {
    A,
    #[sede(untagged)] //~ ERROR cannot find attribute `sede`
    B,
}

fn main() {}
