//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance

/// Checks how accesses from one subtree affect other subtrees.
/// This test checks that an access from a newer created subtree
/// performs a wildcard access on all earlier trees.
pub fn main() {
    let mut x: u32 = 42;

    let ref_base = &mut x;

    let int0 = ref_base as *mut u32 as usize;
    let wild = int0 as *mut u32;

    let reb1 = unsafe { &mut *wild };

    let reb2 = unsafe { &mut *wild };

    let ref3 = &mut *reb2;
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
    //                      │ reb1(Res)  ├   │ reb2(Res)  ├
    //                      │            │   │            │
    //                      └────────────┘   └──────┬─────┘
    //                                              │
    //                                              │
    //                                              ▼
    //                                       ┌────────────┐
    //                                       │            │
    //                                       │ ref3(Res)* │
    //                                       │            │
    //                                       └────────────┘

    // this access disables reb2 because ref3 cannot be a child of it
    // as reb1 does not have any exposed children.
    *ref3 = 13;

    let _fail = *reb1; //~ ERROR: /read access through .* is forbidden/
}
