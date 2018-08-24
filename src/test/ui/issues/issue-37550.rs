#![feature(const_fn)]

const fn x() {
    let t = true;
    //~^ ERROR let bindings in constant functions are unstable
    //~| ERROR statements in constant functions are unstable
    let x = || t;
    //~^ ERROR let bindings in constant functions are unstable
    //~| ERROR statements in constant functions are unstable
}

fn main() {}
