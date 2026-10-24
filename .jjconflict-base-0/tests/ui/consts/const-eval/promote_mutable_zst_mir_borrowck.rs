//@ check-pass

pub fn main() {
    let y: &'static mut [u8; 0] = &mut [];
}
