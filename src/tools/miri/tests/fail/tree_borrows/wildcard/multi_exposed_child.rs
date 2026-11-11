//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance

/// Checks that disabling an exposed reference correctly narrows the
/// possible locations a wildcard access could happen from.
/// Also checks that an access is treated as foreign, if all exposed
/// (non-disabled) references are ancestors.
pub fn main() {
    let mut x: u32 = 42;

    let ref1 = &mut x;
    let int1 = ref1 as *mut u32 as usize;

    let ref2 = &mut *ref1;

    let ref3 = &mut *ref2;
    let _int3 = ref3 as *mut u32 as usize;

    // Write through ref3 so that all references are active.
    *ref3 = 43;

    let wild = int1 as *mut u32;

    //     ┌────────────┐
    //     │            │
    //     │ ref1(Act)* │
    //     │            │
    //     └──────┬─────┘
    //            │
    //            │
    //            ▼
    //     ┌────────────┐
    //     │            │
    //     │ ref2(Act)  │
    //     │            │
    //     └──────┬─────┘
    //            │
    //            │
    //            ▼
    //     ┌────────────┐
    //     │            │
    //     │ ref3(Act)* │
    //     │            │
    //     └────────────┘

    // Writes through either ref1 or ref3, which is either a child or foreign
    // access to ref2.
    unsafe { wild.write(42) };

    // Reading from ref2 still works, since the previous access could have been
    // through its child.
    // This also freezes ref3.
    let _x = *ref2;

    // We can still write through wild, as there is still the exposed ref1 with
    // write permissions under proper exposed provenance, this would be UB as the
    // only tag wild can assume to not invalidate ref2 is ref3, which we just
    // invalidated.
    //
    // This disables ref2, ref3.
    unsafe { wild.write(43) };

    // Fails because ref2 is disabled.
    let _fail = *ref2; //~ ERROR: /read access through .* is forbidden/
}
