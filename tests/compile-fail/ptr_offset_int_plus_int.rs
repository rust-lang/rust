// error-pattern: tried to interpret some bytes as a pointer

fn main() {
    // Can't offset an integer pointer by non-zero offset.
    unsafe {
        let _val = (1 as *mut u8).offset(1);
    }
}
