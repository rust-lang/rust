// compile-pass

#![feature(nll)]

pub fn main() {
    let y: &'static mut [u8; 0] = &mut [];
}
