//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance

/// Checks if we correctly determine the correct exposed reference a write
/// access could happen through, if there are also exposed reference
/// through which only a read access could happen.
pub fn main() {
    let mut x: u32 = 42;

    let ptr_base = &mut x as *mut u32;
    let ref1 = unsafe { &mut *ptr_base };
    let ref2 = unsafe { &*ptr_base };

    // Both references get exposed.
    let int1 = ref1 as *mut u32 as usize;
    let _int2 = ref2 as *const u32 as usize;

    let wild = int1 as *mut u32;

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
    //    │ ref1(Res)* │       │ ref2(Frz)* │
    //    │            │       │            │
    //    └────────────┘       └────────────┘

    // Disables ref2 as the only write could happen through ref1.
    unsafe { wild.write(13) };

    // Fails because ref2 is disabled.
    let _fail = *ref2; //~ ERROR: /read access through .* is forbidden/
}
