//@error-pattern: pointer to 4 bytes starting at offset 0 is out-of-bounds

fn main() {
    unsafe {
        let ptr = Box::into_raw(Box::new(0u16));
        drop(Box::from_raw(ptr as *mut u32));
    }
}
