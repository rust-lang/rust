//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance

/// Checks how accesses from one subtree affect other subtrees.
/// This test checks the case where the access is to the main tree.
pub fn main() {
    let mut x: u32 = 42;

    let ptr_base = &mut x as *mut u32;
    let ref1 = unsafe { &mut *ptr_base };
    let ref2 = unsafe { &mut *ptr_base };

    let int1 = ref1 as *mut u32 as usize;
    let wild = int1 as *mut u32;

    let reb3 = unsafe { &mut *wild };

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
    //    │ ref1(Res)* │     │ ref2(Res) │     │ reb3(Res) │
    //    │            │     │           │     │           │
    //    └────────────┘     └───────────┘     └───────────┘

    // ref2 is part of the main tree and therefore foreign to all subtrees.
    // Therefore, this disables reb3.
    *ref2 = 13;

    let _fail = *reb3; //~ ERROR: /read access through .* is forbidden/
}
