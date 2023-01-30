// edition:2018

#![no_std]
#![crate_type = "lib"]

use alloc::vec;
//~^ ERROR unresolved import `alloc`

pub fn foo() {
    let mut xs = vec![];
    //~^ ERROR cannot determine resolution for the macro `vec`
    xs.push(0);
}
