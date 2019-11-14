// error-pattern: pointer must be in-bounds

fn main() { unsafe {
    let ptr = Box::into_raw(Box::new(0u16));
    Box::from_raw(ptr as *mut u32);
} }
