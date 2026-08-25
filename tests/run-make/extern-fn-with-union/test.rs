extern crate testcrate;

use std::mem;

extern "C" {
    fn give_back(tu: testcrate::TestUnion) -> u64;
}

fn main() {
    let magic: u64 = 0xDEADBEEF;

    // Let's test calling it cross crate
    let back = unsafe { testcrate::give_back(mem::transmute(magic)) };
    assert_eq!(magic, back);

    // And just within this crate
    let back = unsafe { give_back(mem::transmute(magic)) };
    assert_eq!(magic, back);
}
