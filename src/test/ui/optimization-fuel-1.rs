// run-pass

#![crate_name="foo"]

use std::mem::size_of;

// compile-flags: -Z fuel=foo=1

#[allow(unused_tuple_struct_fields)]
struct S1(u8, u16, u8);
#[allow(unused_tuple_struct_fields)]
struct S2(u8, u16, u8);

fn main() {
    let optimized = (size_of::<S1>() == 4) as usize
        +(size_of::<S2>() == 4) as usize;
    assert_eq!(optimized, 1);
}
