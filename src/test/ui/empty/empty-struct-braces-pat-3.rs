// Can't use empty braced struct as enum pattern

// aux-build:empty-struct.rs

extern crate empty_struct;
use empty_struct::*;

enum E {
    Empty3 {}
}

fn main() {
    let e3 = E::Empty3 {};
    let xe3 = XE::XEmpty3 {};

    match e3 {
        E::Empty3() => ()
        //~^ ERROR expected tuple struct/variant, found struct variant `E::Empty3`
    }
    match xe3 {
        XE::XEmpty3() => ()
        //~^ ERROR expected tuple struct/variant, found struct variant `XE::XEmpty3`
    }
    match e3 {
        E::Empty3(..) => ()
        //~^ ERROR expected tuple struct/variant, found struct variant `E::Empty3`
    }
    match xe3 {
        XE::XEmpty3(..) => ()
        //~^ ERROR expected tuple struct/variant, found struct variant `XE::XEmpty3
    }
}
