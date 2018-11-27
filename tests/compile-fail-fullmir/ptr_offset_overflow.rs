//error-pattern: attempt to add with overflow
fn main() {
    let v = [1i8, 2];
    let x = &v[1] as *const i8;
    let _val = unsafe { x.offset(isize::min_value()) };
}
