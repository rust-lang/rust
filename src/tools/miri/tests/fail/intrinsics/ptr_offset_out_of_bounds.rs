fn main() {
    let v = [0i8; 4];
    let x = &v as *const i8;
    let x = unsafe { x.offset(5) }; //~ERROR: is only 4 bytes from the end of the allocation
    panic!("this should never print: {:?}", x);
}
