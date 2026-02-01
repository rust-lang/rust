//! Test that we properly error when there is a pointer fragment in the final value.

use std::{mem::{self, MaybeUninit}, ptr};

type Byte = MaybeUninit<u8>;

const unsafe fn memcpy(dst: *mut Byte, src: *const Byte, n: usize) {
    let mut i = 0;
    while i < n {
        dst.add(i).write(src.add(i).read());
        i += 1;
    }
}

const MEMCPY_RET: MaybeUninit<*const i32> = unsafe { //~ERROR: partial pointer in final value
    let ptr = &42;
    let mut ptr2 = MaybeUninit::new(ptr::null::<i32>());
    memcpy(&mut ptr2 as *mut _ as *mut _, &ptr as *const _ as *const _, mem::size_of::<&i32>() / 2);
    // Return in a MaybeUninit so it does not get treated as a scalar.
    ptr2
};

// Mixing two different pointers that have the same provenance.
const MIXED_PTR: MaybeUninit<*const u8> = { //~ERROR: partial pointer in final value
    static A: u8 = 123;
    const HALF_PTR: usize = std::mem::size_of::<*const ()>() / 2;

    unsafe {
        let x: *const u8 = &raw const A;
        let mut y = MaybeUninit::new(x.wrapping_add(usize::MAX / 4));
        core::ptr::copy_nonoverlapping(
            (&raw const x).cast::<u8>(),
            (&raw mut y).cast::<u8>(),
            HALF_PTR,
        );
        y
    }
};

/// This has pointer bytes in the padding of the memory that the final value is read from.
/// To ensure consistent behavior, we want to *always* copy that padding, even if the value
/// could be represented as a more efficient ScalarPair. Hence this must fail to compile.
fn fragment_in_padding() -> impl Copy {
    // We can't use `repr(align)` here as that would make this not a `ScalarPair` any more.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Thing {
        x: u128,
        y: usize,
        // at least one pointer worth of padding
    }
    // Ensure there is indeed padding.
    const _: () = assert!(mem::size_of::<Thing>() > 16 + mem::size_of::<usize>());

    #[derive(Clone, Copy)]
    union PreservePad {
        thing: Thing,
        bytes: [u8; mem::size_of::<Thing>()],
    }

    const A: Thing = unsafe { //~ERROR: partial pointer in final value
        let mut buffer = [PreservePad { bytes: [0u8; mem::size_of::<Thing>()] }; 2];
        // The offset half a pointer from the end, so that copying a `Thing` copies exactly
        // half the pointer.
        let offset = mem::size_of::<Thing>() - mem::size_of::<usize>()/2;
        // Ensure this is inside the padding.
        assert!(offset >= std::mem::offset_of!(Thing, y) + mem::size_of::<usize>());

        (&raw mut buffer).cast::<&i32>().byte_add(offset).write_unaligned(&1);
        buffer[0].thing
    };

    A
}

fn main() {}
