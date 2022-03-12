// error-pattern: pointer arithmetic failed: null pointer is not a valid pointer

fn main() {
    let x = 0 as *mut i32;
    let _x = x.wrapping_offset(8); // ok, this has no inbounds tag
    let _x = unsafe { x.offset(0) }; // UB despite offset 0, NULL is never inbounds
}
