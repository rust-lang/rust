const C: () = {
    let value = [1, 2];
    let ptr = value.as_ptr().wrapping_add(2);
    let fat = std::ptr::slice_from_raw_parts(ptr, usize::MAX);
    unsafe {
        // This used to ICE, but it should just report UB.
        let _ice = (*fat)[usize::MAX - 1];
        //~^ERROR: overflow
    }
};

fn main() {}
