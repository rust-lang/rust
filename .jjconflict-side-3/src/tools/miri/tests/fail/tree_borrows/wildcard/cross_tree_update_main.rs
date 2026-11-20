//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance

/// Checks how accesses from one subtree affect other subtrees.
/// This tests how main is effected by an access through a subtree.
pub fn main() {
    let mut x: u32 = 42;

    let ptr_base = &mut x as *mut u32;
    let ref1 = unsafe { &mut *ptr_base };
    let ref2 = unsafe { &mut *ptr_base };

    let int1 = ref1 as *mut u32 as usize;
    let wild = int1 as *mut u32;

    let reb = unsafe { &mut *wild };

    //    ┌────────────┐
    //    │            │
    //    │  ptr_base  ├───────────┐                 *
    //    │            │           │                 │
    //    └──────┬─────┘           │                 │
    //           │                 │                 │
    //           │                 │                 │
    //           ▼                 ▼                 ▼
    //    ┌────────────┐     ┌───────────┐     ┌───────────┐
    //    │            │     │           │     │           │
    //    │ ref1(Res)* │     │ ref2(Res) │     │ reb(Res)  │
    //    │            │     │           │     │           │
    //    └────────────┘     └───────────┘     └───────────┘

    // Writes through the reborrowed reference causing a wildcard
    // write on the main tree. This disables ref2 as it doesn't
    // have any exposed children.
    *reb = 13;

    let _fail = *ref2; //~ ERROR: /read access through .* is forbidden/
}
