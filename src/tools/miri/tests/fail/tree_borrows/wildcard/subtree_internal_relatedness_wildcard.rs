//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance

// Checks if we correctly infer the relatedness of nodes that are
// part of the same wildcard root during a wildcard access.
pub fn main() {
    let mut x: u32 = 42;

    let ref_base = &mut x;

    let int = ref_base as *mut u32 as usize;
    let wild = int as *mut u32;

    let reb = unsafe { &mut *wild };
    let ptr_reb = reb as *mut u32;
    let ref1 = unsafe { &mut *ptr_reb };
    let _int1 = ref1 as *mut u32 as usize;
    let ref2 = unsafe { &mut *ptr_reb };

    //    ┌──────────────┐
    //    │              │
    //    │ptr_base(Res)*│         *
    //    │              │         │
    //    └──────────────┘         │
    //                             │
    //                             │
    //                             ▼
    //                       ┌────────────┐
    //                       │            │
    //                       │  reb(Res)  ├───────────┐
    //                       │            │           │
    //                       └──────┬─────┘           │
    //                              │                 │
    //                              │                 │
    //                              ▼                 ▼
    //                       ┌────────────┐     ┌───────────┐
    //                       │            │     │           │
    //                       │ ref1(Res)* │     │ ref2(Res) │
    //                       │            │     │           │
    //                       └────────────┘     └───────────┘

    // Writes either through ref1 or ptr_base.
    // This disables ref2 as the access is foreign to it in either case.
    unsafe { *wild = 13 };

    // This is fine because the earlier write could have come from ref1.
    let _succ = *ref1;
    // ref2 is disabled so this fails.
    let _fail = *ref2; //~ ERROR: /read access through .* is forbidden/
}
