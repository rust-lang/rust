// run-pass
#![allow(dead_code)]

enum E { V16(u16), V32(u32) }
struct S { a: E, b: u16, c: u16 }
static C: S = S { a: E::V16(0xDEAD), b: 0x600D, c: 0xBAD };

pub fn main() {
    let n = C.b;
    assert!(n != 0xBAD);
    assert_eq!(n, 0x600D);
}
