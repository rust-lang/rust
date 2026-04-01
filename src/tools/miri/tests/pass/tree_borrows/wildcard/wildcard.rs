//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance

pub fn main() {
    multiple_exposed_siblings();
    multiple_exposed_child();
    dealloc();
    protector();
    protector_conflicted_release();
    returned_mut_is_usable();
}

/// Checks that an access through a wildcard reference
/// doesn't disable any exposed references.
/// It tests this with exposed references that are siblings of each other.
pub fn multiple_exposed_siblings() {
    let mut x: u32 = 42;

    let ptr_base = &mut x as *mut u32;
    let ref1 = unsafe { &mut *ptr_base };
    let ref2 = unsafe { &mut *ptr_base };

    // Both references get exposed.
    let int1 = ref1 as *mut u32 as usize;
    let _int2 = ref2 as *mut u32 as usize;

    let wild = int1 as *mut u32;

    //   ┌────────────┐
    //   │            │
    //   │  ptr_base  ├────────────┐
    //   │            │            │
    //   └──────┬─────┘            │
    //          │                  │
    //          │                  │
    //          ▼                  ▼
    //   ┌────────────┐     ┌────────────┐
    //   │            │     │            │
    //   │ ref1(Res)* │     │ ref2(Res)* │
    //   │            │     │            │
    //   └────────────┘     └────────────┘

    // Writes through either of the two exposed references.
    // We do not know which so we cannot disable the other.
    unsafe { wild.write(13) };

    // Reading through either of these references should be valid.
    assert_eq!(*ref2, 13);
}

/// Checks that an access through a wildcard reference
/// doesn't disable any exposed references.
/// It tests this with exposed references where one is the ancestor of the other.
pub fn multiple_exposed_child() {
    let mut x: u32 = 42;

    let ref1 = &mut x;
    let int1 = ref1 as *mut u32 as usize;

    let ref2 = &mut *ref1;

    let ref3 = &mut *ref2;
    let _int3 = ref3 as *mut u32 as usize;

    let wild = int1 as *mut u32;

    //     ┌────────────┐
    //     │            │
    //     │ ref1(Res)* │
    //     │            │
    //     └──────┬─────┘
    //            │
    //            │
    //            ▼
    //     ┌────────────┐
    //     │            │
    //     │ ref2(Res)  │
    //     │            │
    //     └──────┬─────┘
    //            │
    //            │
    //            ▼
    //     ┌────────────┐
    //     │            │
    //     │ ref3(Res)* │
    //     │            │
    //     └────────────┘

    // This writes either through ref1 or ref3, which is either a child or foreign access to ref2.
    unsafe { wild.write(42) };

    // Reading from ref2 still works, since the previous access could have been through its child.
    // This also freezes ref3.
    let _x = *ref2;

    // We can still write through wild, as there is still the exposed ref1 with write permissions.
    unsafe { wild.write(43) };
}

/// Checks that we can deallocate through a wildcard reference.
fn dealloc() {
    use std::alloc::Layout;
    let x = unsafe { std::alloc::alloc_zeroed(Layout::new::<u32>()) as *mut u32 };
    let ref1 = unsafe { &mut *x };
    let int = ref1 as *mut u32 as usize;
    let wild = int as *mut u32;

    unsafe { std::alloc::dealloc(wild as *mut u8, Layout::new::<u32>()) };
}

/// Checks that we can pass a wildcard reference to a function.
fn protector() {
    fn protect(arg: &mut u32) {
        *arg = 4;
    }
    let mut x: u32 = 32;
    let ref1 = &mut x;
    let int = ref1 as *mut u32 as usize;
    let wild = int as *mut u32;
    let wild_ref = unsafe { &mut *wild };

    protect(wild_ref);

    assert_eq!(*ref1, 4);
}

/// Checks whether we correctly handle the protector being released on
/// a conflicted exposed reference.
fn protector_conflicted_release() {
    let mut x: u32 = 42;

    let ptr_base = &mut x as *mut u32;
    let ref1 = unsafe { &mut *ptr_base };
    let ref2 = unsafe { &mut *ptr_base };

    let protect = |arg: &mut u32| {
        // Expose arg.
        let int = arg as *mut u32 as usize;
        let wild = int as *mut u32;

        // Do a foreign read to arg marking it as conflicted and making child_writes UB while its protected.
        let _x = *ref2;

        return wild;
    };

    let wild = protect(ref1);

    // The protector on arg got released so writes through arg should work again.
    unsafe { *wild = 4 };
}

/// Analogous to same test in `../tree-borrows.rs` but with a protected wildcard reference.
fn returned_mut_is_usable() {
    fn reborrow(x: &mut u8) -> &mut u8 {
        let y = &mut *x;
        // Activate the reference so that it is vulnerable to foreign reads.
        *y = *y;
        y
        // An implicit read through `x` is inserted here.
    }
    let mut x: u8 = 0;
    let ref1 = &mut x;
    let int = ref1 as *mut u8 as usize;
    let wild = int as *mut u8;
    let wild_ref = unsafe { &mut *wild };

    let y = reborrow(wild_ref);

    *y = 1;
}
