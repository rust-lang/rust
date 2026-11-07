// Can't use empty braced struct as constant or constructor function

//@ aux-build:empty-struct.rs

extern crate empty_struct;
use empty_struct::*;

struct Empty1 {}

enum E {
    Empty3 {}
}

fn main() {
    let e1 = Empty1; //~ ERROR cannot find value `Empty1` in this scope
    let e1 = Empty1();
    //~^ ERROR cannot find function, tuple struct or tuple variant `Empty1` in this scope
    let e3 = E::Empty3; //~ ERROR expected value, found struct variant `E::Empty3`
    let e3 = E::Empty3();
    //~^ ERROR expected value, found struct variant `E::Empty3`

    let xe1 = XEmpty1; //~ ERROR cannot find value `XEmpty1` in this scope
    let xe1 = XEmpty1();
    //~^ ERROR cannot find function, tuple struct or tuple variant `XEmpty1` in this scope
    let xe3 = XE::Empty3; //~ ERROR no variant, associated function, or constant named `Empty3` found for enum
    let xe3 = XE::Empty3(); //~ ERROR no variant, associated function, or constant named `Empty3` found for enum

    XE::Empty1 {}; //~ ERROR no variant named `Empty1` found for enum `empty_struct::XE`
}
