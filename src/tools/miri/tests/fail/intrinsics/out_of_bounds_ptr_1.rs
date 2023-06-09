fn main() {
    let v = [0i8; 4];
    let x = &v as *const i8;
    // The error is inside another function, so we cannot match it by line
    let x = unsafe { x.offset(5) }; //~ERROR: pointer to 5 bytes starting at offset 0 is out-of-bounds
    panic!("this should never print: {:?}", x);
}
