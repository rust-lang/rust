//@aux-build:serde.rs

// derive macros not imported, but namespace imported. Not yet handled.
extern crate serde;

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
