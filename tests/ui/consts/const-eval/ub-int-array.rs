//! Test the "array of int" fast path in validity checking, and in particular whether it
//! points at the right array element.

//@ dont-require-annotations: NOTE

use std::mem;

#[repr(C)]
union MaybeUninit<T: Copy> {
    uninit: (),
    init: T,
}

impl<T: Copy> MaybeUninit<T> {
    const fn new(t: T) -> Self {
        MaybeUninit { init: t }
    }
}

const UNINIT_INT_0: [u32; 3] = unsafe {
    //~^ ERROR invalid value at [0]
    mem::transmute([
        MaybeUninit { uninit: () },
        // Constants chosen to achieve endianness-independent hex dump.
        MaybeUninit::new(0x11111111),
        MaybeUninit::new(0x22222222),
    ])
};
const UNINIT_INT_1: [u32; 3] = unsafe {
    //~^ ERROR invalid value at [1]
    mem::transmute([
        MaybeUninit::new(0u8),
        MaybeUninit::new(0u8),
        MaybeUninit::new(0u8),
        MaybeUninit::new(0u8),
        MaybeUninit::new(1u8),
        MaybeUninit { uninit: () },
        MaybeUninit::new(1u8),
        MaybeUninit::new(1u8),
        MaybeUninit::new(2u8),
        MaybeUninit::new(2u8),
        MaybeUninit { uninit: () },
        MaybeUninit::new(2u8),
    ])
};
const UNINIT_INT_2: [u32; 3] = unsafe {
    //~^ ERROR invalid value at [2]
    mem::transmute([
        MaybeUninit::new(0u8),
        MaybeUninit::new(0u8),
        MaybeUninit::new(0u8),
        MaybeUninit::new(0u8),
        MaybeUninit::new(1u8),
        MaybeUninit::new(1u8),
        MaybeUninit::new(1u8),
        MaybeUninit::new(1u8),
        MaybeUninit::new(2u8),
        MaybeUninit::new(2u8),
        MaybeUninit::new(2u8),
        MaybeUninit { uninit: () },
    ])
};

fn main() {}
