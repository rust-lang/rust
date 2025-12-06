//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance

pub fn main() {
    multiple_exposed_siblings1();
    multiple_exposed_siblings2();
    reborrow3();
    returned_mut_is_usable();
    only_foreign_is_temporary();
}

/// Checks that accessing through a reborrowed wildcard doesn't
/// disable any exposed reference.
fn multiple_exposed_siblings1() {
    let mut x: u32 = 42;

    let ptr_base = &mut x as *mut u32;

    let ref1 = unsafe { &mut *ptr_base };
    let int1 = ref1 as *mut u32 as usize;

    let ref2 = unsafe { &mut *ptr_base };
    let _int2 = ref2 as *mut u32 as usize;

    let wild = int1 as *mut u32;

    let reb = unsafe { &mut *wild };

    //   ┌────────────┐
    //   │            │
    //   │  ptr_base  ├────────────┐                 *
    //   │            │            │                 │
    //   └──────┬─────┘            │                 │
    //          │                  │                 │
    //          │                  │                 │
    //          ▼                  ▼                 ▼
    //   ┌────────────┐     ┌────────────┐    ┌────────────┐
    //   │            │     │            │    │            │
    //   │ ref1(Res)* │     │ ref2(Res)* │    │  reb(Res)  │
    //   │            │     │            │    │            │
    //   └────────────┘     └────────────┘    └────────────┘

    // Could either have as a parent ref1 or ref2.
    // So we can't disable either of them.
    *reb = 13;

    // We can still access either ref1 or ref2.
    // Although it is actually UB to access both of them.
    assert_eq!(*ref2, 13);
    assert_eq!(*ref1, 13);
}

/// Checks that wildcard accesses do not invalidate any exposed
/// nodes through which the access could have happened.
/// It checks this for the case where some reborrowed wildcard
/// pointers are exposed as well.
fn multiple_exposed_siblings2() {
    let mut x: u32 = 42;

    let ptr_base = &mut x as *mut u32;
    let int = ptr_base as usize;

    let wild = int as *mut u32;

    let reb_ptr = unsafe { &mut *wild } as *mut u32;

    let ref1 = unsafe { &mut *reb_ptr };
    let _int1 = ref1 as *mut u32 as usize;

    let ref2 = unsafe { &mut *reb_ptr };
    let _int2 = ref2 as *mut u32 as usize;

    //   ┌────────────┐
    //   │            │
    //   │ ptr_base*  │            *
    //   │            │            │
    //   └────────────┘            │
    //                             │
    //                             │
    //                             ▼
    //                      ┌────────────┐
    //                      │            │
    //                      │    reb     ├────────────┐
    //                      │            │            │
    //                      └──────┬─────┘            │
    //                             │                  │
    //                             │                  │
    //                             ▼                  ▼
    //                      ┌────────────┐     ┌────────────┐
    //                      │            │     │            │
    //                      │ ref1(Res)* │     │ ref2(Res)* │
    //                      │            │     │            │
    //                      └────────────┘     └────────────┘

    // Writes either through ref1, ref2 or ptr_base, which are all exposed.
    // Since we don't know which we do not apply any transitions to any of
    // the references.
    unsafe { wild.write(13) };

    // We should be able to access either ref1 or ref2.
    // Although it is actually UB to access ref1 and ref2 together.
    assert_eq!(*ref2, 13);
    assert_eq!(*ref1, 13);
}

/// Checks that accessing a reborrowed wildcard reference doesn't
/// invalidate other reborrowed wildcard references, if they
/// are also exposed.
fn reborrow3() {
    let mut x: u32 = 42;

    let ptr_base = &mut x as *mut u32;
    let int = ptr_base as usize;

    let wild = int as *mut u32;

    let reb1 = unsafe { &mut *wild };
    let ref2 = &mut *reb1;
    let _int = ref2 as *mut u32 as usize;

    let reb3 = unsafe { &mut *wild };

    //   ┌────────────┐
    //   │            │
    //   │ ptr_base*  │            *                  *
    //   │            │            │                  │
    //   └────────────┘            │                  │
    //                             │                  │
    //                             │                  │
    //                             ▼                  ▼
    //                      ┌────────────┐     ┌────────────┐
    //                      │            │     │            │
    //                      │ reb1(Res)  |     │ reb3(Res)  |
    //                      │            │     │            │
    //                      └──────┬─────┘     └────────────┘
    //                             │
    //                             │
    //                             ▼
    //                      ┌────────────┐
    //                      │            │
    //                      │ ref2(Res)* │
    //                      │            │
    //                      └────────────┘

    // This is the only valid ordering these accesses can happen in.

    // reb3 could be a child of ref2 so we don't disable ref2, reb1.
    *reb3 = 1;
    // Disables reb3 as it cannot be an ancestor of ref2.
    *ref2 = 2;
    // Disables ref2 (and reb3 if it wasn't already).
    *reb1 = 3;
}

/// Analogous to same test in `../tree-borrows.rs` but with returning a
/// reborrowed wildcard reference.
fn returned_mut_is_usable() {
    let mut x: u32 = 32;
    let ref1 = &mut x;

    let y = protect(ref1);

    fn protect(arg: &mut u32) -> &mut u32 {
        // Reborrow `arg` through a wildcard.
        let int = arg as *mut u32 as usize;
        let wild = int as *mut u32;
        let ref2 = unsafe { &mut *wild };

        // Activate the reference so that it is vulnerable to foreign reads.
        *ref2 = 42;

        ref2
        // An implicit read through `arg` is inserted here.
    }

    *y = 4;
}

/// When accessing an allocation through a tag that was created from wildcard reference
/// we treat nodes with a larger tag as if the access could only have been foreign to them.
/// This change in access relatedness should not be visible in later accesses.
fn only_foreign_is_temporary() {
    let mut x = 0u32;
    let wild = &mut x as *mut u32 as usize as *mut u32;

    let reb1 = unsafe { &mut *wild };
    let reb2 = unsafe { &mut *wild };
    let ref3 = &mut *reb1;
    let _int = ref3 as *mut u32 as usize;

    let reb4 = unsafe { &mut *wild };

    //
    //
    //             *                 *                 *
    //             │                 │                 │
    //             │                 │                 │
    //             │                 │                 │
    //             │                 │                 │
    //             ▼                 ▼                 ▼
    //       ┌────────────┐    ┌────────────┐    ┌────────────┐
    //       │            │    │            │    │            │
    //       │  reb1(Res) │    │  reb2(Res) │    │  reb4(Res) │
    //       │            │    │            │    │            │
    //       └──────┬─────┘    └────────────┘    └────────────┘
    //              │
    //              │
    //              ▼
    //       ┌────────────┐
    //       │            │
    //       │ ref3(Res)* │
    //       │            │
    //       └────────────┘

    // Performs a foreign read on ref3 and doesn't update reb1.
    // This temporarily treats ref3 as if only foreign accesses are possible to
    // it. This is because the accessed tag reb2 has a larger tag than ref3.
    let _x = *reb2;
    // Should not update ref3, reb1 as we don't know if the access is local or foreign.
    // This should stop treating ref3 as only foreign because the accessed tag reb4
    // has a larger tag than ref3.
    *reb4 = 32;
    // The previous write could have been local to ref3, so this access should still work.
    *ref3 = 4;
}
