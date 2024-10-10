use std::mem;

#[repr(C)]
struct RefAndLen {
    ptr: &'static u8,
    len: usize,
}

// If this is `None`, the len becomes padding.
type T = Option<RefAndLen>;

fn main() {
    unsafe {
        let mut p: mem::MaybeUninit<T> = mem::MaybeUninit::zeroed();
        // The copy when `T` is returned from `transmute` should destroy padding
        // (even when we use `write_unaligned`, which under the hood uses an untyped copy).
        p.as_mut_ptr().write_unaligned(mem::transmute((0usize, 0usize)));
        // Null epresents `None`.
        assert!(matches!(*p.as_ptr(), None));

        // The second part, with the length, becomes padding.
        let c = &p as *const _ as *const u8;
        // Read a padding byte.
        let _val = *c.add(mem::size_of::<*const u8>());
        //~^ERROR: uninitialized
    }
}
