//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance

/// Checks if we correctly determine the correct exposed reference a write
/// access could happen through,
/// if there are also exposed reference through which only a read could happen.
pub fn main() {
    let mut x: u32 = 42;

    let ref1 = &mut x;
    let int1 = ref1 as *mut u32 as usize;

    let ref2 = &mut *ref1;

    let ref3 = &*ref2;
    let _int3 = ref3 as *const u32 as usize;

    let wild = int1 as *mut u32;

    //     ┌────────────┐
    //     │            │
    //     │ ref1(Res)* │
    //     │            │
    //     └──────┬─────┘
    //            │
    //            │
    //            ▼
    //     ┌────────────┐
    //     │            │
    //     │ ref2(Res)  │
    //     │            │
    //     └──────┬─────┘
    //            │
    //            │
    //            ▼
    //     ┌────────────┐
    //     │            │
    //     │ ref3(Frz)* │
    //     │            │
    //     └────────────┘

    // Writes through ref1 as we cannot write through ref3 since it's frozen.
    // Disables ref2, ref3.
    unsafe { wild.write(42) };

    // ref2 is disabled.
    let _fail = *ref2; //~ ERROR: /read access through .* is forbidden/
}
