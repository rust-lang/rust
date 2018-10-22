fn main() {
    let v = [1i16, 2];
    let x = &v[1] as *const i16;
    let _ = &v[0] as *const i16; // we need to leak the 2nd thing too or it cannot be accessed through a raw ptr
    // Adding 2*isize::max and then 1 is like substracting 1
    let x = x.wrapping_offset(isize::max_value());
    let x = x.wrapping_offset(isize::max_value());
    let x = x.wrapping_offset(1);
    assert_eq!(unsafe { *x }, 1);
}
