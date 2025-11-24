//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance
// NOTE: This file documents UB that is not detected by wildcard provenance.

pub fn main() {
    uncertain_provenance();
    protected_exposed();
    protected_wildcard();
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

    // ref2 is disabled, so this read causes UB, but we currently don't protect this.
    let _fail = *ref2;
}

/// Currently, we do not assign protectors to wildcard references.
/// This test has UB because it does a foreign write to a protected reference.
/// However, that reference is a wildcard, so this doesn't get detected.
#[allow(unused_variables)]
pub fn protected_wildcard() {
    let mut x: u32 = 32;
    let ref1 = &mut x;
    let ref2 = &mut *ref1;

    let int = ref2 as *mut u32 as usize;
    let wild = int as *mut u32;
    let wild_ref = unsafe { &mut *wild };

    let mut protect = |arg: &mut u32| {
        // arg is a protected pointer with wildcard provenance.

        //    ┌────────────┐
        //    │            │
        //    │ ref1(Res)  │
        //    │            │
        //    └──────┬─────┘
        //           │
        //           │
        //           ▼
        //    ┌────────────┐
        //    │            │
        //    │ ref2(Res)* │
        //    │            │
        //    └────────────┘

        // Writes to ref1, disabling ref2, i.e. disabling all exposed references.
        // Since a wildcard reference is protected, this is UB. But we currently don't detect this.
        *ref1 = 13;
    };

    // We pass a pointer with wildcard provenance to the function.
    protect(wild_ref);
}
