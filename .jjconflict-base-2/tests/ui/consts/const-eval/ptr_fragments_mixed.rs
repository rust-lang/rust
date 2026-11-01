//! This mixes fragments from different pointers, in a way that we should not accept.
//! See <https://github.com/rust-lang/rust/issues/146291>.

static A: u8 = 123;
static B: u8 = 123;

const HALF_PTR: usize = std::mem::size_of::<*const ()>() / 2;

// All fragments have the same provenance, but they did not all come from the same pointer.
const APTR: *const u8 = {
    unsafe {
        let x: *const u8 = &raw const A;
        let mut y = x.wrapping_add(usize::MAX / 4);
        core::ptr::copy_nonoverlapping(
            (&raw const x).cast::<u8>(),
            (&raw mut y).cast::<u8>(),
            HALF_PTR,
        );
        y //~ERROR: unable to read parts of a pointer
    }
};

// All fragments have the same relative offset, but not all the same provenance.
const BPTR: *const u8 = {
    unsafe {
        let x: *const u8 = &raw const A;
        let mut y = &raw const B;
        core::ptr::copy_nonoverlapping(
            (&raw const x).cast::<u8>(),
            (&raw mut y).cast::<u8>(),
            HALF_PTR,
        );
        y //~ERROR: unable to read parts of a pointer
    }
};

fn main() {}
