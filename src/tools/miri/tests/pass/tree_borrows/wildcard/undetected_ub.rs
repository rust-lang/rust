//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance
// NOTE: This file documents UB that is not detected by wildcard provenance.

pub fn main() {
    uncertain_provenance();
    protected_exposed();
    cross_tree_update_older_invalid_exposed();
}

/// Currently, if we do not know for a tag if an access is local or foreign,
/// then we do not do any state transitions on that tag. However, to implement
/// proper provenance we would have to instead pick the correct transition
/// non-deterministically.
///
/// This test contains such UB that will not be detectable with wildcard provenance,
/// but would be detectable if we implemented this behavior correctly.      
pub fn uncertain_provenance() {
    let mut x: u32 = 42;

    let ptr_base = &mut x as *mut u32;
    // We create 2 mutable references, each with a unique tag.
    let ref1 = unsafe { &mut *ptr_base };
    let ref2 = unsafe { &mut *ptr_base };

    // Both references get exposed.
    let int1 = ref1 as *mut u32 as usize;
    let _int2 = ref2 as *mut u32 as usize;
    //ref1 :        Reserved
    //ref2 :        Reserved

    // We need to pick the "correct" tag for wild from the exposed tags.
    let wild = int1 as *mut u32;
    //              wild=ref1   wild=ref2
    //ref1 :        Reserved    Reserved
    //ref2 :        Reserved    Reserved

    // We write to wild, disabling the other tag.
    unsafe { wild.write(13) };
    //              wild=ref1   wild=ref2
    //ref1 :        Unique      Disabled
    //ref2 :        Disabled    Unique

    // We access both references, even though one of them should be
    // disabled under proper exposed provenance.
    // This is UB, however, wildcard provenance cannot detect this.
    assert_eq!(*ref1, 13);
    //              wild=ref1   wild=ref2
    //ref1 :        Unique      UB
    //ref2 :        Disabled    Frozen
    assert_eq!(*ref2, 13);
    //              wild=ref1   wild=ref2
    //ref1 :        Frozen      UB
    //ref2 :        UB          Frozen
}

/// If a reference is protected, then all foreign writes to it cause UB.
/// This effectively means any write needs to happen through a child of
/// the protected reference.
/// With this information we could further narrow the possible candidates
/// for a wildcard write.
/// However, currently tree borrows doesn't do this, so this test has UB
/// that isn't detected.
pub fn protected_exposed() {
    let mut x: u32 = 42;

    let ptr_base = &mut x as *mut u32;
    let ref1 = unsafe { &mut *ptr_base };
    let ref2 = unsafe { &mut *ptr_base };

    let _int2 = ref2 as *mut u32 as usize;

    fn protect(ref3: &mut u32) {
        let int3 = ref3 as *mut u32 as usize;

        //    ┌────────────┐
        //    │            │
        //    │  ptr_base  ├──────────────┐
        //    │            │              │
        //    └──────┬─────┘              │
        //           │                    │
        //           │                    │
        //           ▼                    ▼
        //    ┌────────────┐       ┌────────────┐
        //    │            │       │            │
        //    │ ref1(Res)  │       │ ref2(Res)* │
        //    │            │       │            │
        //    └──────┬─────┘       └────────────┘
        //           │
        //           │
        //           ▼
        //    ┌────────────┐
        //    │            │
        //    │ ref3(Res)* │
        //    │            │
        //    └────────────┘

        // Since ref3 is protected, we could know that every write from outside it will be UB.
        // This means we know that the access is through ref3, disabling ref2.
        let wild = int3 as *mut u32;
        unsafe { wild.write(13) }
    }
    protect(ref1);

    // ref2 should be disabled, so this read causes UB, but we currently don't detect this.
    let _fail = *ref2;
}

/// Checks how accesses from one subtree affect other subtrees.
/// This test shows an example where we don't update a node whose exposed
/// children are greater than `max_local_tag`.
pub fn cross_tree_update_older_invalid_exposed() {
    let mut x: [u32; 2] = [42, 43];

    let ref_base = &mut x;

    let int0 = ref_base as *mut u32 as usize;
    let wild = int0 as *mut u32;

    let reb1 = unsafe { &mut *wild };
    *reb1 = 44;

    // We need this reference to be at a different location,
    // so that creating it doesn't freeze reb1.
    let reb2 = unsafe { &mut *wild.wrapping_add(1) };
    let reb2_ptr = (reb2 as *mut u32).wrapping_sub(1);

    let ref3 = &mut *reb1;
    let _int3 = ref3 as *mut u32 as usize;

    //    ┌──────────────┐
    //    │              │
    //    │ptr_base(Res)*│        *                *
    //    │              │        │                │
    //    └──────────────┘        │                │
    //                            │                │
    //                            │                │
    //                            ▼                ▼
    //                      ┌────────────┐   ┌────────────┐
    //                      │            │   │            │
    //                      │ reb1(Act)  │   │ reb2(Res)  │
    //                      │            │   │            │
    //                      └──────┬─────┘   └────────────┘
    //                             │
    //                             │
    //                             ▼
    //                      ┌────────────┐
    //                      │            │
    //                      │ ref3(Res)* │
    //                      │            │
    //                      └────────────┘

    // This access doesn't freeze reb1 even though no access could have come from its
    // child ref3 (since ref3>reb2). This is because ref3 doesnt get disabled during this
    // access.
    //
    // ref3 doesn't get frozen because it's still reserved.
    let _y = unsafe { *reb2_ptr };

    // reb1 should be frozen so a write should be UB. But we currently don't detect this.
    *reb1 = 4;
}
