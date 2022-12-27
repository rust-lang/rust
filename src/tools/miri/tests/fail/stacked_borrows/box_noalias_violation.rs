unsafe fn test(mut x: Box<i32>, y: *const i32) -> i32 {
    // We will call this in a way that x and y alias.
    *x = 5;
    std::mem::forget(x);
    *y //~ERROR: does not exist in the borrow stack
}

fn main() {
    unsafe {
        let mut v = 42;
        let ptr = &mut v as *mut i32;
        let b = Box::from_raw(ptr);
        let ptr = &*b as *const i32;
        test(b, ptr);
    }
}
