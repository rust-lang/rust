#![allow(warnings)]

fn main() {
    let x = 22_u32;
    let y: &u32 = (&x) as &'static u32; //~ ERROR `x` does not live long enough
}
