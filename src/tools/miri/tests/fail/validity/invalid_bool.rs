#![allow(unnecessary_transmutes)]
fn main() {
    let _b = unsafe { std::mem::transmute::<u8, bool>(2) }; //~ ERROR: expected a boolean
}
