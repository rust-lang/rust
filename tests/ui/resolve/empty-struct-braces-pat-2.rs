// Can't use empty braced struct as enum pattern

//@ aux-build:empty-struct.rs

extern crate empty_struct;
use empty_struct::*;

struct Empty1 {}

fn main() {
    let e1 = Empty1 {};
    let xe1 = XEmpty1 {};

    match e1 {
        Empty1() => () //~ ERROR cannot find tuple struct or tuple variant `Empty1` in this scope
    }
    match xe1 {
        XEmpty1() => () //~ ERROR cannot find tuple struct or tuple variant `XEmpty1` in this scope
    }
    match e1 {
        Empty1(..) => () //~ ERROR cannot find tuple struct or tuple variant `Empty1` in this scope
    }
    match xe1 {
        XEmpty1(..) => () //~ ERROR cannot find tuple struct or tuple variant `XEmpty1` in this scope
    }
}
