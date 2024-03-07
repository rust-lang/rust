#![warn(clippy::mixed_attributes_style)]

#[allow(unused)] //~ ERROR: item has both inner and outer attributes
fn foo1() {
    #![allow(unused)]
}

#[allow(unused)]
#[allow(unused)]
fn foo2() {}

fn foo3() {
    #![allow(unused)]
    #![allow(unused)]
}

/// linux
//~^ ERROR: item has both inner and outer attributes
fn foo4() {
    //! windows
}

/// linux
/// windows
fn foo5() {}

fn foo6() {
    //! linux
    //! windows
}

#[allow(unused)] //~ ERROR: item has both inner and outer attributes
mod bar {
    #![allow(unused)]
}

fn main() {
    // test code goes here
}
