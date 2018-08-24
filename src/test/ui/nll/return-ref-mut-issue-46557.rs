// Regression test for issue #46557

#![feature(nll)]
#![allow(dead_code)]

fn gimme_static_mut() -> &'static mut u32 {
    let ref mut x = 1234543; //~ ERROR borrowed value does not live long enough [E0597]
    x
}

fn main() {}
