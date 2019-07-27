// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

mod u8 {
    pub const BITS: usize = 8;
}

const NUM: usize = u8::BITS;

struct MyStruct { nums: [usize; 8] }

fn main() {
    let _s = MyStruct { nums: [0; NUM] };
}
