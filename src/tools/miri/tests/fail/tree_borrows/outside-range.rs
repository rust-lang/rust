//@compile-flags: -Zmiri-tree-borrows

// Check that in the case of locations outside the range of a pointer,
// protectors trigger if and only if the location has already been accessed
fn main() {
    unsafe {
        let data = &mut [0u8, 1, 2, 3];
        let raw = data.as_mut_ptr();
        stuff(&mut *raw, raw);
    }
}

unsafe fn stuff(x: &mut u8, y: *mut u8) {
    let xraw = x as *mut u8;
    // No issue here: location 1 is not accessed
    *y.add(1) = 42;
    // Still no issue: location 2 is not invalidated
    let _val = *xraw.add(2);
    // However protector triggers if location is both accessed and invalidated
    let _val = *xraw.add(3);
    *y.add(3) = 42; //~ ERROR: /write access through .* is forbidden/
}
