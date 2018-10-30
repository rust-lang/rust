fn main() {
    let v = [1i16, 2];
    let x = v.as_ptr().wrapping_offset(1); // ptr to the 2nd element
    // Adding 2*isize::max and then 1 is like substracting 1
    let x = x.wrapping_offset(isize::max_value());
    let x = x.wrapping_offset(isize::max_value());
    let x = x.wrapping_offset(1);
    assert_eq!(unsafe { *x }, 1);
}
