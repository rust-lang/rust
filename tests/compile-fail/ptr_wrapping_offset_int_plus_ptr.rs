// error-pattern: pointer value as raw bytes

fn main() {
    let ptr = Box::into_raw(Box::new(0u32));
    // Can't start with an integer pointer and get to something usable
    let ptr = (1 as *mut u8).wrapping_offset(ptr as isize);
    let _val = unsafe { *ptr };
}
