// run-pass
#![allow(dead_code)]
// A quick test of 'unsafe const fn' functionality

const unsafe fn dummy(v: u32) -> u32 {
    !v
}

struct Type;
impl Type {
    const unsafe fn new() -> Type {
        Type
    }
}

const VAL: u32 = unsafe { dummy(0xFFFF) };
const TYPE_INST: Type = unsafe { Type::new() };

fn main() {
    assert_eq!(VAL, 0xFFFF0000);
}
