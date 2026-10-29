//@ run-pass

use std::mem;

#[repr(transparent)]
struct Foo(u32);

const TRANSMUTED_U32: u32 = unsafe { mem::transmute(Foo(3)) };

fn main() {
    assert_eq!(TRANSMUTED_U32, 3);
}
