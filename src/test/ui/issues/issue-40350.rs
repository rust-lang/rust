#![feature(rustc_attrs)]
#![allow(warnings)]

enum E {
    A = {
        enum F { B }
        0
    }
}

#[rustc_error]
fn main() {}
//~^ ERROR compilation successful

