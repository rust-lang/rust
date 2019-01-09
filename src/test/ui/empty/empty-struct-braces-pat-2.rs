// Can't use empty braced struct as enum pattern

// aux-build:empty-struct.rs

extern crate empty_struct;
use empty_struct::*;

struct Empty1 {}

fn main() {
    let e1 = Empty1 {};
    let xe1 = XEmpty1 {};

    match e1 {
        Empty1() => () //~ ERROR expected tuple struct/variant, found struct `Empty1`
    }
    match xe1 {
        XEmpty1() => () //~ ERROR expected tuple struct/variant, found struct `XEmpty1`
    }
    match e1 {
        Empty1(..) => () //~ ERROR expected tuple struct/variant, found struct `Empty1`
    }
    match xe1 {
        XEmpty1(..) => () //~ ERROR expected tuple struct/variant, found struct `XEmpty1`
    }
}
