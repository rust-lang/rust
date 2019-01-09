// Test bounds checking for DST raw slices
// error-pattern:index out of bounds

fn main() {
    let a: *const [_] = &[1, 2, 3];
    unsafe {
        let _b = (*a)[3];
    }
}
