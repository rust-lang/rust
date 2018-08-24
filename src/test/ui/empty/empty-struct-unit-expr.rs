// Can't use unit struct as constructor function

// aux-build:empty-struct.rs

extern crate empty_struct;
use empty_struct::*;

struct Empty2;

enum E {
    Empty4
}

fn main() {
    let e2 = Empty2(); //~ ERROR expected function, found `Empty2`
    let e4 = E::Empty4();
    //~^ ERROR expected function, found enum variant `E::Empty4` [E0618]
    let xe2 = XEmpty2(); //~ ERROR expected function, found `empty_struct::XEmpty2`
    let xe4 = XE::XEmpty4();
    //~^ ERROR expected function, found enum variant `XE::XEmpty4` [E0618]
}
