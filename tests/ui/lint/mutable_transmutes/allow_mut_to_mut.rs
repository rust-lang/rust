// check-pass

use std::mem::transmute;

fn main() {
    let _a: &mut u8 = unsafe { transmute(&mut 0u8) };
}
