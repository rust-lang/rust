fn main() {
    let v = [0i8; 4];
    let x = &v as *const i8;
    let x = unsafe { x.offset(-1) }; //~ERROR: is at the beginning of the allocation
    panic!("this should never print: {:?}", x);
}
