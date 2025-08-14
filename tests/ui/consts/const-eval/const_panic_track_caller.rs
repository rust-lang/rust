#![allow(non_fmt_panics)]
#![crate_type = "lib"]

#[track_caller]
const fn a() -> u32 {
    panic!("hey")
}

#[track_caller]
const fn b() -> u32 {
    a()
}

const fn c() -> u32 {
    b() //~ NOTE inside `c`
    //~^ NOTE the failure occurred here
}

const X: u32 = c();
//~^ ERROR evaluation of constant value failed
//~| NOTE hey
