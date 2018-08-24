// Test that we can't call random fns in a const fn or do other bad things.

#![feature(const_fn)]

use std::mem::transmute;

fn random() -> u32 { 0 }

const fn sub(x: &u32) -> usize {
    unsafe { transmute(x) } //~ ERROR E0015
}

const fn sub1() -> u32 {
    random() //~ ERROR E0015
}

static Y: u32 = 0;

const fn get_Y() -> u32 {
    Y
        //~^ ERROR E0013
}

const fn get_Y_addr() -> &'static u32 {
    &Y
        //~^ ERROR E0013
}

const fn get() -> u32 {
    let x = 22;
    //~^ ERROR let bindings in constant functions are unstable
    //~| ERROR statements in constant functions are unstable
    let y = 44;
    //~^ ERROR let bindings in constant functions are unstable
    //~| ERROR statements in constant functions are unstable
    x + y
    //~^ ERROR let bindings in constant functions are unstable
    //~| ERROR let bindings in constant functions are unstable
}

fn main() {}
