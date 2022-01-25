// run-rustfix

#![feature(rust_2018_preview, crate_visibility_modifier)]
#![deny(absolute_paths_not_starting_with_crate)]

use foo::{a, b};
//~^ ERROR absolute paths must start with
//~| this is accepted in the current edition
//~| ERROR absolute paths must start with
//~| this is accepted in the current edition
//~| ERROR absolute paths must start with
//~| this is accepted in the current edition
//~| ERROR absolute paths must start with
//~| this is accepted in the current edition

mod foo {
    crate fn a() {}
    crate fn b() {}
    crate fn c() {}
}

fn main() {
    a();
    b();

    {
        use foo::{self as x, c};
        //~^ ERROR absolute paths must start with
        //~| this is accepted in the current edition
        //~| ERROR absolute paths must start with
        //~| this is accepted in the current edition
        //~| ERROR absolute paths must start with
        //~| this is accepted in the current edition
        x::a();
        c();
    }
}
