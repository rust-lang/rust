//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance

// Checks if we correctly infer the relatedness of nodes that are
// part of the same wildcard root.
pub fn main() {
    let mut x: u32 = 42;

    let ref_base = &mut x;

    let int1 = ref_base as *mut u32 as usize;
    let wild = int1 as *mut u32;

    let reb = unsafe { &mut *wild };
    let ptr_reb = reb as *mut u32;
    let ref1 = unsafe { &mut *ptr_reb };
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
    //                       │ ref1(Res)  │     │ ref2(Res) │
    //                       │            │     │           │
    //                       └────────────┘     └───────────┘

    // ref1 is foreign to ref2, so this should disable ref2.
    *ref1 = 13;

    let _fail = *ref2; //~ ERROR: /read access through .* is forbidden/
}
