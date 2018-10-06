// error-pattern: pointer value as raw bytes

fn main() {
    let ptr = &mut *Box::new(0u32) as *mut u32;
    // Can't start with an integer pointer and get to something usable
    unsafe {
        let _ = (1 as *mut u8).offset(ptr as isize);
    }
}
