//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance
use std::cell::UnsafeCell;

/// This is a variant of the test in `../protector-write-lazy.rs`, but with
/// wildcard references.
/// Checks that a protector release access correctly determines that certain tags
/// cannot be children of the protected tag and that it updates them accordingly.
///
/// For this version we know the tag is not a child because its wildcard root has
/// a smaller tag then the released reference.
pub fn main() {
    // We need two locations so that we can create a new reference
    // that is foreign to an already active tag.
    let mut x: UnsafeCell<[u32; 2]> = UnsafeCell::new([32, 33]);
    let ref1 = &mut x;
    let cell_ptr = ref1.get() as *mut u32;

    let int = ref1 as *mut UnsafeCell<[u32; 2]> as usize;
    let wild = int as *mut UnsafeCell<u32>;

    let ref2 = unsafe { &mut *cell_ptr };

    // `ref3` gets created before the protected ref `arg4`.
    let ref3 = unsafe { &mut *wild.wrapping_add(1) };

    let protect = |arg4: &mut u32| {
        // Activates arg4. This would disable ref3  at [0] if it wasn't a cell.
        *arg4 = 41;

        // Creates an exposed child of arg4.
        let ref5 = &mut *arg4;
        let _int = ref5 as *mut u32 as usize;

        // This creates ref6 from ref3 at [1], so that it doesn't disable arg4 at [0].
        let ref6 = unsafe { &mut *ref3.get() };

        //    ┌───────────┐
        //    │    ref1*  │
        //    │ Cel │ Cel │           *
        //    └─────┬─────┘           │
        //          │                 │
        //          │                 │
        //          ▼                 ▼
        //    ┌───────────┐     ┌───────────┐
        //    │ ref2│     │     │     │ ref3│
        //    │ Act │ Res │     │ Cel │ Cel │
        //    └─────┬─────┘     └─────┬─────┘
        //          │                 │
        //          │                 │
        //          ▼                 ▼
        //    ┌───────────┐     ┌───────────┐
        //    │ arg4│     │     │     │ ref6│
        //    │ Act │ Res │     │ Res │ Res │
        //    └─────┬─────┘     └───────────┘
        //          │
        //          │
        //          ▼
        //    ┌───────────┐
        //    │ref5*|     │
        //    │ Res │ Res │
        //    └───────────┘

        // Creates a pointer to [0] with the provenance of ref6.
        return (ref6 as *mut u32).wrapping_sub(1);

        // Protector release on arg4 happens here.
        // This should cause a foreign write on all foreign nodes,
        // unless they could be a child of arg4.
        // arg4 has an exposed child, so some tags with a wildcard
        // ancestor could be its children.
        // However, the root of ref6 was created before arg4, so it
        // cannot be a child of it. So it should get disabled
        // (at location [0]).
    };
    // ptr has provenance of ref6
    let ptr = protect(ref2);
    // ref6 is disabled at [0].
    let _fail = unsafe { *ptr }; //~ ERROR: /read access through .* is forbidden/
}
