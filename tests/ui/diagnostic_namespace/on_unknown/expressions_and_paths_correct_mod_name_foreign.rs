//@ aux-crate:foreign=foreign.rs

extern crate foreign;

use foreign::c::d::x; //~ ERROR this=b, unresolved=x

fn main() {
    let _ = foreign::c::d::X(); //~ ERROR this=foreign::c::d, unresolved=X

    let _: foreign::c::d::X; //~ ERROR this=foreign::c::d, unresolved=X
}
