//error-pattern: overflowing math on a pointer
fn main() {
    let v = [1i8, 2];
    let x = &v[1] as *const i8;
    // One of them is guaranteed to overflow
    let _ = unsafe { x.offset(isize::max_value()) };
    let _ = unsafe { x.offset(isize::min_value()) };
}
