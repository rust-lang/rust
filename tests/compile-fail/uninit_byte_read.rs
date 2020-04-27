fn main() {
    let v: Vec<u8> = Vec::with_capacity(10);
    let undef = unsafe { *v.get_unchecked(5) };
    let x = undef + 1; //~ ERROR this operation requires initialized memory
    panic!("this should never print: {}", x);
}
