fn main() {
    let v = [0i8; 4];
    let x = &v as *const i8;
    let x = unsafe { x.offset(5) }; //~ERROR: expected a pointer to 5 bytes of memory
    panic!("this should never print: {:?}", x);
}
