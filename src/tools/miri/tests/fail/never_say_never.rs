// This should fail even without validation
//@compile-flags: -Zmiri-disable-validation

#![feature(never_type)]
#![allow(unreachable_code)]

fn main() {
    let y = &5;
    let x: ! = unsafe { *(y as *const _ as *const !) };
    f(x) //~ ERROR: entering unreachable code
}

fn f(x: !) -> ! {
    x
}
