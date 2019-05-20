// run-pass
#![allow(dead_code)]

// Tests that unions aren't subject to unsafe non-zero/niche-filling optimizations.
//
// For example, if a union `U` can contain both a `&T` and a `*const T`, there's definitely no
// bit-value that an `Option<U>` could reuse as `None`; this test makes sure that isn't done.
//
// Secondly, this tests the status quo (not a guarantee; subject to change!) to not apply such
// optimizations to types containing unions even if they're theoretically possible. (discussion:
// https://github.com/rust-lang/rust/issues/36394)
//
// Notably this nails down part of the behavior that `MaybeUninit` assumes: that a
// `Option<MaybeUninit<&u8>>` does not take advantage of non-zero optimization, and thus is a safe
// construct.

use std::mem::{size_of, transmute};

union U1<A: Copy> {
    a: A,
}

union U2<A: Copy, B: Copy> {
    a: A,
    b: B,
}

// Option<E> uses a value other than 0 and 1 as None
#[derive(Clone,Copy)]
enum E {
    A = 0,
    B = 1,
}

fn main() {
    // Unions do not participate in niche-filling/non-zero optimization...
    assert!(size_of::<Option<U2<&u8, u8>>>() > size_of::<U2<&u8, u8>>());
    assert!(size_of::<Option<U2<&u8, ()>>>() > size_of::<U2<&u8, ()>>());
    assert!(size_of::<Option<U2<u8, E>>>() > size_of::<U2<u8, E>>());

    // ...even when theoretically possible:
    assert!(size_of::<Option<U1<&u8>>>() > size_of::<U1<&u8>>());
    assert!(size_of::<Option<U2<&u8, &u8>>>() > size_of::<U2<&u8, &u8>>());

    // The unused bits of the () variant can have any value.
    let zeroed: U2<&u8, ()> = unsafe { transmute(std::ptr::null::<u8>()) };

    if let None = Some(zeroed) {
        panic!()
    }
}
