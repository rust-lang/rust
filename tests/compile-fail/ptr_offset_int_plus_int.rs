// error-pattern: invalid arithmetic on pointers

fn main() {
    // Can't offset an integer pointer by non-zero offset.
    unsafe {
        let _val = (1 as *mut u8).offset(1);
    }
}
