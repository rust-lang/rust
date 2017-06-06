fn main() {
    let v = [1i16, 2];
    let x = &v as *const i16;
    let x = x.wrapping_offset(1);
    assert_eq!(unsafe { *x }, 2);
}
