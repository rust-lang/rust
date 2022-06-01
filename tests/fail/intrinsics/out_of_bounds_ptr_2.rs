// error-pattern: overflowing in-bounds pointer arithmetic
fn main() {
    let v = [0i8; 4];
    let x = &v as *const i8;
    let x = unsafe { x.offset(isize::MIN) };
    panic!("this should never print: {:?}", x);
}
