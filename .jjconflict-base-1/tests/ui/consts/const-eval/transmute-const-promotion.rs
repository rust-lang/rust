#![allow(unnecessary_transmutes)]
use std::mem;

fn main() {
    let x: &'static u32 = unsafe { &mem::transmute(3.0f32) };
    //~^ ERROR temporary value dropped while borrowed
}
