//@ run-pass
#![allow(dead_code)]

enum E { V16(u16), V32(u32) }
static C: (E, u16, u16) = (E::V16(0xDEAD), 0x600D, 0xBAD);

pub fn main() {
    let (_, n, _) = C;
    assert!(n != 0xBAD);
    assert_eq!(n, 0x600D);
}
