fn main() {
    let v = [1i16, 2];
    let x = &v[1] as *const i16 as usize;
    // arithmetic
    let _y = x + 4;
    let _y = 4 + x;
    let _y = x - 2;
    // bit-operations, covered by alignment
    assert_eq!(x & 1, 0);
    assert_eq!(x & 0, 0);
    assert_eq!(1 & (x+1), 1);
    let _y = !1 & x;
    let _y = !0 & x;
    let _y = x & !1;
    // remainder, covered by alignment
    assert_eq!(x % 2, 0);
    assert_eq!((x+1) % 2, 1);
    // remainder with 1 is always 0
    assert_eq!(x % 1, 0);
}
