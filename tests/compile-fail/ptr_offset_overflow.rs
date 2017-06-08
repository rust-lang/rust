//error-pattern: overflowing math
fn main() {
    let v = [1i8, 2];
    let x = &v[1] as *const i8;
    let _ = unsafe { x.offset(isize::min_value()) };
}
