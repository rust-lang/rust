fn main() {
    let v: Vec<u8> = vec![1, 2];
    let x = unsafe { *v.as_ptr().wrapping_offset(5) }; //~ ERROR: memory access at offset 6, outside bounds of allocation
    panic!("this should never print: {}", x);
}
