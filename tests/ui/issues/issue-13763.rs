//@ run-pass
#![allow(dead_code)]

mod u8 {
    pub const BITS: usize = 8;
}

const NUM: usize = u8::BITS;

struct MyStruct { nums: [usize; 8] }

fn main() {
    let _s = MyStruct { nums: [0; NUM] };
}
