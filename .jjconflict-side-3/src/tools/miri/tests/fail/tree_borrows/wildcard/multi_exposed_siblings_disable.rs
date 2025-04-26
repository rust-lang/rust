//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance

/// Checks with multiple exposed nodes, that if they are all disabled
/// then no wildcard accesses are possible.
pub fn main() {
    let mut x: u32 = 42;

    let ptr_base = &mut x as *mut u32;
    let ref1 = unsafe { &mut *ptr_base };
    let ref2 = unsafe { &mut *ptr_base };
    let ref3 = unsafe { &mut *ptr_base };

    // Both references get exposed.
    let int1 = ref1 as *mut u32 as usize;
    let _int2 = ref2 as *mut u32 as usize;

    let wild = int1 as *mut u32;

    //    ┌────────────┐
    //    │            │
    //    │  ptr_base  ├──────────────┬───────────────────┐
    //    │            │              │                   │
    //    └──────┬─────┘              │                   │
    //           │                    │                   │
    //           │                    │                   │
    //           ▼                    ▼                   ▼
    //    ┌────────────┐       ┌────────────┐       ┌───────────┐
    //    │            │       │            │       │           │
    //    │ ref1(Res)* │       │ ref2(Res)* │       │ ref3(Res) │
    //    │            │       │            │       │           │
    //    └────────────┘       └────────────┘       └───────────┘

    // Disables ref1,ref2.
    *ref3 = 13;

    // Both exposed references are disabled so this fails.
    let _fail = unsafe { *wild }; //~ ERROR: /read access through .* is forbidden/
}
