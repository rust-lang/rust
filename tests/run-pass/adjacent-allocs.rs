fn main() {
    // The slack between allocations is random.
    // Loop a few times to hit the zero-slack case.
    for _ in 0..1024 {
        let n = 0u64;
        let ptr: *const u64 = &n;

        // Allocate a new stack variable whose lifetime quickly ends.
        // If there's a chance that &m == ptr.add(1), then an int-to-ptr cast of
        // that value will have ambiguous provenance between n and m.
        // See https://github.com/rust-lang/miri/issues/1866#issuecomment-985770125
        {
            let m = 0u64;
            let _ = &m as *const u64;
        }

        let iptr = ptr as usize;
        let zst = (iptr + 8) as *const ();
        // This is a ZST ptr just at the end of `n`, so it should be valid to deref.
        unsafe { *zst }
    }
}
