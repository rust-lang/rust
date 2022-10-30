fn main() {
    let start_ptr = &4 as *const _ as *const u8;
    let length = 10;
    let end_ptr = start_ptr.wrapping_add(length);
    // Even if the offset is 0, a dangling OOB pointer is not allowed.
    unsafe { end_ptr.offset_from(end_ptr) }; //~ERROR: pointer at offset 10 is out-of-bounds
}
