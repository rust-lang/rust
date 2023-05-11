fn main() {
    let v = [0i8; 4];
    let x = &v as *const i8;
    let x = unsafe { x.offset(-1) }; //~ERROR: pointer to 1 byte starting at offset -1 is out-of-bounds
    panic!("this should never print: {:?}", x);
}
