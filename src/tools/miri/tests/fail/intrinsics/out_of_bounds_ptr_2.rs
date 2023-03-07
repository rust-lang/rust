fn main() {
    let v = [0i8; 4];
    let x = &v as *const i8;
    let x = unsafe { x.offset(isize::MIN) }; //~ERROR: overflowing in-bounds pointer arithmetic
    panic!("this should never print: {:?}", x);
}
