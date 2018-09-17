fn main() {
    let bytes = [0i8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let one = bytes.as_ptr().wrapping_offset(1);
    let three = bytes.as_ptr().wrapping_offset(3);
    let res = (one as usize) | (three as usize); //~ ERROR invalid arithmetic on pointers that would leak base addresses
    println!("{}", res);
}
