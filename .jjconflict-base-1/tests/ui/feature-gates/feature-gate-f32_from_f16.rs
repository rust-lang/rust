#![feature(f16)]
#![allow(unused)]

pub fn main() {
    let _: f32 = 1.0f16.into();
    //~^ ERROR use of unstable library feature `f32_from_f16`
}
