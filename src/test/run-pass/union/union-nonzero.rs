// run-pass
#![allow(dead_code)]

use std::mem::{size_of, transmute};

union U1<A: Copy> {
    a: A,
}

union U2<A: Copy, B: Copy> {
    a: A,
    b: B,
}

fn main() {
    // Unions do not participate in niche-filling/non-zero optimization...
    assert!(size_of::<Option<U2<&u8, u8>>>() > size_of::<U2<&u8, u8>>());
    assert!(size_of::<Option<U2<&u8, ()>>>() > size_of::<U2<&u8, ()>>());

    // ...even when theoretically possible:
    assert!(size_of::<Option<U1<&u8>>>() > size_of::<U1<&u8>>());
    assert!(size_of::<Option<U2<&u8, &u8>>>() > size_of::<U2<&u8, &u8>>());

    // The unused bits of the () variant can have any value.
    let zeroed: U2<&u8, ()> = unsafe { transmute(std::ptr::null::<u8>()) };

    if let None = Some(zeroed) {
        panic!()
    }
}
