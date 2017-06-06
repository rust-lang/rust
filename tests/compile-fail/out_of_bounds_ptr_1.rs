// error-pattern: pointer computed at offset 5, outside bounds of allocation
fn main() {
    let v = [0i8; 4];
    let x = &v as *const i8;
    // The error is inside another function, so we cannot match it by line
    let x = unsafe { x.offset(5) };
    panic!("this should never print: {:?}", x);
}
