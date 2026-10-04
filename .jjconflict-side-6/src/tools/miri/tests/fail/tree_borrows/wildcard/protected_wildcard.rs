//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance

/// Checks if we pass a reference derived from a wildcard pointer
/// that it gets correctly protected.
pub fn main() {
    let mut x: u32 = 32;
    let ref1 = &mut x;

    let ref2 = &mut *ref1;
    let int2 = ref2 as *mut u32 as usize;

    let wild = int2 as *mut u32;
    let wild_ref = unsafe { &mut *wild };

    let mut protect = |_arg: &mut u32| {
        // _arg is a protected pointer with wildcard parent.

        //    ┌────────────┐
        //    │            │
        //    │ ref1(Res)  │          *
        //    │            │          │
        //    └──────┬─────┘          │
        //           │                │
        //           │                │
        //           ▼                ▼
        //    ┌────────────┐   ┌────────────┐
        //    │            │   │            │
        //    │ ref2(Res)* │   │  _arg(Res) │
        //    │            │   │            │
        //    └────────────┘   └────────────┘

        // Writes to ref1, causing a foreign write to ref2 and _arg.
        // Since _arg is protected this is UB.
        *ref1 = 13; //~ ERROR: /write access through .* is forbidden/
    };

    // We pass a pointer with wildcard provenance to the function.
    protect(wild_ref);
}
