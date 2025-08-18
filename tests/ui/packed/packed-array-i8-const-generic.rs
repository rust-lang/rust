//@ check-pass
#![allow(dead_code)]

#[repr(C, packed)]
struct PascalStringI8<const CAP: usize> {
    len: u8,
    buf: [i8; CAP],
}

fn bar<const CAP: usize>(s: &PascalStringI8<CAP>) -> &[i8] {
    // Goal: this line should not trigger E0793 for i8 arrays
    &s.buf[0..s.len as usize]
}

fn main() {
    let p = PascalStringI8::<10> { len: 3, buf: [1, 2, 3, 0, 0, 0, 0, 0, 0, 0] };
    let s = bar(&p);
    assert_eq!(s, &[1, 2, 3]);
}
