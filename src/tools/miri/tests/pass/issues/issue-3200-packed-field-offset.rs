#![feature(layout_for_ptr)]
use std::mem;

#[repr(packed, C)]
struct PackedSized {
    f: u8,
    d: [u32; 4],
}

#[repr(packed, C)]
struct PackedUnsized {
    f: u8,
    d: [u32],
}

impl PackedSized {
    fn unsize(&self) -> &PackedUnsized {
        // We can't unsize via a generic type since then we get the error
        // that packed structs with unsized tail don't work if the tail
        // might need dropping.
        let len = 4usize;
        unsafe { mem::transmute((self, len)) }
    }
}

fn main() {
    unsafe {
        let p = PackedSized { f: 0, d: [1, 2, 3, 4] };
        let p = p.unsize() as *const PackedUnsized;
        // Make sure the size computation does *not* think there is
        // any padding in front of the `d` field.
        assert_eq!(mem::size_of_val_raw(p), 1 + 4 * 4);
        // And likewise for the offset computation.
        let d = std::ptr::addr_of!((*p).d);
        assert_eq!(d.cast::<u32>().read_unaligned(), 1);
    }
}
