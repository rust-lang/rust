//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance

/// Checks how accesses from one subtree affect other subtrees.
/// This test checks that an access from an earlier created subtree
/// is foreign to a later created one.
pub fn main() {
    let mut x: u32 = 42;

    let ref_base = &mut x;

    let int0 = ref_base as *mut u32 as usize;
    let wild = int0 as *mut u32;

    let reb1 = unsafe { &mut *wild };

    let reb2 = unsafe { &mut *wild };

    let ref3 = &mut *reb1;
    let _int3 = ref3 as *mut u32 as usize;
    //    ┌──────────────┐
    //    │              │
    //    │ptr_base(Res)*│         *                 *
    //    │              │         │                 │
    //    └──────────────┘         │                 │
    //                             │                 │
    //                             │                 │
    //                             ▼                 ▼
    //                       ┌────────────┐    ┌────────────┐
    //                       │            │    │            │
    //                       │ reb1(Res)  ├    │ reb2(Res)  ├
    //                       │            │    │            │
    //                       └──────┬─────┘    └────────────┘
    //                              │
    //                              │
    //                              ▼
    //                       ┌────────────┐
    //                       │            │
    //                       │ ref3(Res)* │
    //                       │            │
    //                       └────────────┘

    // This access disables reb2 because ref3 cannot be a child of it
    // as reb2 both has a higher tag and doesn't have any exposed children.
    *ref3 = 13;

    let _fail = *reb2; //~ ERROR: /read access through .* is forbidden/
}
