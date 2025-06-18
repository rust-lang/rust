//@aux-build:serde.rs

// derive macros imported but unused

extern crate serde;
use serde::{Serialize, Deserialize};

#[serde(untagged)] //~ ERROR cannot find attribute `serde`
enum A { //~ HELP `serde` is an attribute that can be used by the derive macros `Deserialize` and `Serialize`
    A,
    B,
}

enum B { //~ HELP `serde` is an attribute that can be used by the derive macros `Deserialize` and `Serialize`
    A,
    #[serde(untagged)] //~ ERROR cannot find attribute `serde`
    B,
}

enum C {
    A,
    #[sede(untagged)] //~ ERROR cannot find attribute `sede`
    B, //~^ HELP the derive macros `Deserialize` and `Serialize` accept the similarly named `serde` attribute
}

fn main() {}
