#[rustfmt::skip] // fails with "left behind trailing whitespace"
fn main() {
    let x = Box::into_raw(Box::new(0u32));
    let x = x.wrapping_offset(8); // ok, this has no inbounds tag
    let _x = unsafe { x.offset(0) }; // UB despite offset 0, the pointer is not inbounds of the only object it can point to
    //~^ERROR: pointer at offset 32 is out-of-bounds
}
