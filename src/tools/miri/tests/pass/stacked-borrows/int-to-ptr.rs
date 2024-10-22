//@compile-flags: -Zmiri-permissive-provenance
use std::ptr;

// Just to make sure that casting a ref to raw, to int and back to raw
// and only then using it works. This rules out ideas like "do escape-to-raw lazily";
// after casting to int and back, we lost the tag that could have let us do that.
fn ref_raw_int_raw() {
    let mut x = 3;
    let xref = &mut x;
    let xraw = xref as *mut i32 as usize as *mut i32;
    assert_eq!(unsafe { *xraw }, 3);
}

/// Ensure that we do not just pick the topmost possible item on int2ptr casts.
fn example(variant: bool) {
    unsafe {
        fn not_so_innocent(x: &mut u32) -> usize {
            let x_raw4 = x as *mut u32;
            x_raw4.expose_provenance()
        }

        let mut c = 42u32;

        let x_unique1 = &mut c;
        // stack: [..., Unique(1)]

        let x_raw2 = x_unique1 as *mut u32;
        let x_raw2_addr = x_raw2.expose_provenance();
        // stack: [..., Unique(1), SharedRW(2)]

        let x_unique3 = &mut *x_raw2;
        // stack: [.., Unique(1), SharedRW(2), Unique(3)]

        assert_eq!(not_so_innocent(x_unique3), x_raw2_addr);
        // stack: [.., Unique(1), SharedRW(2), Unique(3), ..., SharedRW(4)]

        // Do an int2ptr cast. This can pick tag 2 or 4 (the two previously exposed tags).
        // 4 is the "obvious" choice (topmost tag, what we used to do with untagged pointers).
        // And indeed if `variant == true` it is the only possible choice.
        // But if `variant == false` then 2 is the only possible choice!
        let x_wildcard = ptr::with_exposed_provenance_mut::<i32>(x_raw2_addr);

        if variant {
            // If we picked 2, this will invalidate 3.
            *x_wildcard = 10;
            // Now we use 3. Only possible if above we picked 4.
            *x_unique3 = 12;
        } else {
            // This invalidates tag 4.
            *x_unique3 = 10;
            // Now try to write with the "guessed" tag; it must be 2.
            *x_wildcard = 12;
        }
    }
}

fn test() {
    unsafe {
        let root = &mut 42;
        let root_raw = root as *mut i32;
        let addr1 = root_raw as usize;
        let child = &mut *root_raw;
        let child_raw = child as *mut i32;
        let addr2 = child_raw as usize;
        assert_eq!(addr1, addr2);
        // First use child.
        *(addr2 as *mut i32) -= 2; // picks child_raw
        *child -= 2;
        // Then use root.
        *(addr1 as *mut i32) += 2; // picks root_raw
        *root += 2;
        // Value should be unchanged.
        assert_eq!(*root, 42);
    }
}

fn main() {
    ref_raw_int_raw();
    example(false);
    example(true);
    test();
}
