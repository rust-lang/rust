// Regression test for #63952, shouldn't hang.
//@ stderr-per-bitwidth

#[repr(C)]
#[derive(Copy, Clone)]
struct SliceRepr {
    ptr: *const u8,
    len: usize,
}

union SliceTransmute {
    repr: SliceRepr,
    slice: &'static [u8],
}

// bad slice: length too big to even exist anywhere
const SLICE_WAY_TOO_LONG: &[u8] = unsafe { //~ ERROR: slice is bigger than largest supported object
    SliceTransmute {
        repr: SliceRepr {
            ptr: &42,
            len: usize::MAX,
        },
    }
    .slice
};

fn main() {}
