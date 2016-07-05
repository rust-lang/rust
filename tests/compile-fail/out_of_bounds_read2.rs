fn main() {
    let v: Vec<u8> = vec![1, 2];
    let x = unsafe { *v.get_unchecked(5) }; //~ ERROR: memory access of 6..7 outside bounds of allocation
    panic!("this should never print: {}", x);
}
