fn main() {
    let v: Vec<u8> = vec![1, 2];
    let x = unsafe { *v.as_ptr().wrapping_offset(5) }; //~ ERROR: which has size 2
    panic!("this should never print: {}", x);
}
