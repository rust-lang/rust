//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance

/// Checks how accesses from one subtree affect other subtrees.
/// This test checks that an access from a newer created subtree
/// performs a wildcard access on all earlier trees, and that
/// either accesses are treated as foreign for tags that are
/// larger than the root of the accessed subtree.
/// This tests the special case where these updates get propagated
/// up the tree.
pub fn main() {
    let mut x: u32 = 42;

    let ref_base = &mut x;

    let int0 = ref_base as *mut u32 as usize;
    let wild = int0 as *mut u32;

    let reb1 = unsafe { &mut *wild };

    let reb2 = unsafe { &mut *wild };

    let ref3 = &mut *reb1;
    let _int3 = ref3 as *mut u32 as usize;

    let ref4 = &mut *reb2;
    let _int4 = ref4 as *mut u32 as usize;
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
    //                      └──────┬─────┘   └──────┬─────┘
    //                             │                │
    //                             │                │
    //                             ▼                ▼
    //                      ┌────────────┐   ┌────────────┐
    //                      │            │   │            │
    //                      │ ref3(Res)* │   │ ref4(Res)* │
    //                      │            │   │            │
    //                      └────────────┘   └────────────┘

    // This access disables ref3 and reb1 because ref4 cannot be a child of it
    // as reb2 has a smaller tag than ref3.
    //
    // Because of the update order during a wildcard access (child before parent)
    // ref3 gets disabled before we update reb1. So reb1 has no exposed children
    // with write access at the time it gets updated so it also gets disabled.
    *ref4 = 13;

    // Fails because reb1 is disabled.
    let _fail = *reb1; //~ ERROR: /read access through .* is forbidden/
}
