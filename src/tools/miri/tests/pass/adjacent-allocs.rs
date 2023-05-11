//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
//@compile-flags: -Zmiri-permissive-provenance

fn ensure_allocs_can_be_adjacent() {
    for _ in 0..512 {
        let n = 0u64;
        let ptr: *const u64 = &n;
        let ptr2 = {
            let m = 0u64;
            &m as *const u64
        };
        if ptr.wrapping_add(1) == ptr2 {
            return;
        }
    }
    panic!("never saw adjacent stack variables?");
}

fn test1() {
    // The slack between allocations is random.
    // Loop a few times to hit the zero-slack case.
    for _ in 0..512 {
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

fn test2() {
    fn foo() -> u64 {
        0
    }

    for _ in 0..512 {
        let n = 0u64;
        let ptr: *const u64 = &n;
        foo();
        let iptr = ptr as usize;
        unsafe {
            let start = &*std::ptr::slice_from_raw_parts(iptr as *const (), 1);
            let end = &*std::ptr::slice_from_raw_parts((iptr + 8) as *const (), 1);
            assert_eq!(start.len(), end.len());
        }
    }
}

fn main() {
    ensure_allocs_can_be_adjacent();
    test1();
    test2();
}
