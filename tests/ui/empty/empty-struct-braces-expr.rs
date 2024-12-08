// Can't use empty braced struct as constant or constructor function

//@ aux-build:empty-struct.rs

extern crate empty_struct;
use empty_struct::*;

struct Empty1 {}

enum E {
    Empty3 {}
}

fn main() {
    let e1 = Empty1; //~ ERROR expected value, found struct `Empty1`
    let e1 = Empty1();
    //~^ ERROR expected function, tuple struct or tuple variant, found struct `Empty1`
    let e3 = E::Empty3; //~ ERROR expected value, found struct variant `E::Empty3`
    let e3 = E::Empty3();
    //~^ ERROR expected value, found struct variant `E::Empty3`

    let xe1 = XEmpty1; //~ ERROR expected value, found struct `XEmpty1`
    let xe1 = XEmpty1();
    //~^ ERROR expected function, tuple struct or tuple variant, found struct `XEmpty1`
    let xe3 = XE::Empty3; //~ ERROR no variant or associated item named `Empty3` found for enum
    let xe3 = XE::Empty3(); //~ ERROR no variant or associated item named `Empty3` found for enum

    XE::Empty1 {}; //~ ERROR no variant named `Empty1` found for enum `empty_struct::XE`
}
