// build-pass (FIXME(62277): could be check-pass?)

pub fn main() {
    let y: &'static mut [u8; 0] = &mut [];
}
