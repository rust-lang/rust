//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance

/// Checks that with only one exposed reference, that if this reference gets
/// disabled no wildcard accesses are possible.
pub fn main() {
    let mut x: u32 = 42;

    let ptr_base = &mut x as *mut u32;
    let ref1 = unsafe { &mut *ptr_base };
    let ref2 = unsafe { &mut *ptr_base };

    let int1 = ref1 as *mut u32 as usize;
    let wild = int1 as *mut u32;

    //    ┌────────────┐
    //    │            │
    //    │  ptr_base  ├───────────┐
    //    │            │           │
    //    └──────┬─────┘           │
    //           │                 │
    //           │                 │
    //           ▼                 ▼
    //    ┌────────────┐     ┌───────────┐
    //    │            │     │           │
    //    │ ref1(Res)* │     │ ref2(Res) │
    //    │            │     │           │
    //    └────────────┘     └───────────┘

    // Disables ref1.
    *ref2 = 13;

    // Tries to do a wildcard access through the only exposed reference ref1,
    // which is disabled.
    let _fail = unsafe { *wild }; //~ ERROR: /read access through .* is forbidden/
}
