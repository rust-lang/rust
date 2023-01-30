#![allow(warnings)]
#![feature(type_ascription)]

fn main() {
    let x = 22_u32;
    let y: &u32 = type_ascribe!(&x, &'static u32); //~ ERROR E0597
}
