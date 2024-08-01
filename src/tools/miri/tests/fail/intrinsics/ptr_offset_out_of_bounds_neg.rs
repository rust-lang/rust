fn main() {
    let v = [0i8; 4];
    let x = &v as *const i8;
    let x = unsafe { x.offset(-1) }; //~ERROR: expected a pointer to the end of 1 byte of memory
    panic!("this should never print: {:?}", x);
}
