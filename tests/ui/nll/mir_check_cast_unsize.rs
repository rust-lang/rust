#![allow(dead_code)]

use std::fmt::Debug;

fn bar<'a>(x: &'a u32) -> &'static dyn Debug {
    x
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
