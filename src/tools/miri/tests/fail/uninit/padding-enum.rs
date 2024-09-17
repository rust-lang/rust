use std::mem;

// We have three fields to avoid the ScalarPair optimization.
#[allow(unused)]
enum E {
    None,
    Some(&'static (), &'static (), usize),
}

fn main() {
    unsafe {
        let mut p: mem::MaybeUninit<E> = mem::MaybeUninit::zeroed();
        // The copy when `E` is returned from `transmute` should destroy padding
        // (even when we use `write_unaligned`, which under the hood uses an untyped copy).
        p.as_mut_ptr().write_unaligned(mem::transmute((0usize, 0usize, 0usize)));
        // This is a `None`, so everything but the discriminant is padding.
        assert!(matches!(*p.as_ptr(), E::None));

        // Turns out the discriminant is (currently) stored
        // in the 2nd pointer, so the first half is padding.
        let c = &p as *const _ as *const u8;
        // Read a padding byte.
        let _val = *c.add(0);
        //~^ERROR: uninitialized
    }
}
