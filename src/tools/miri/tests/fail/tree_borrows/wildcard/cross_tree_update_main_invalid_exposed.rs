//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance

/// Checks how accesses from one subtree affect other subtrees.
/// This test checks that an access from a subtree performs a
/// wildcard access on all earlier trees, and that local
/// accesses are treated as access errors for tags that are
/// larger than the root of the accessed subtree.
pub fn main() {
    let mut x: u32 = 42;

    let ptr_base = &mut x as *mut u32;
    let ref1 = unsafe { &mut *ptr_base };

    // Activates ref1.
    *ref1 = 4;

    let int1 = ref1 as *mut u32 as usize;
    let wild = int1 as *mut u32;

    let ref2 = unsafe { &mut *wild };

    // Freezes ref1.
    let ref3 = unsafe { &mut *ptr_base };
    let _int3 = ref3 as *mut u32 as usize;

    //    ┌──────────────┐
    //    │              │
    //    │ptr_base(Act) ├───────────┐                  *
    //    │              │           │                  │
    //    └──────┬───────┘           │                  │
    //           │                   │                  │
    //           │                   │                  │
    //           ▼                   ▼                  ▼
    //     ┌─────────────┐     ┌────────────┐     ┌───────────┐
    //     │             │     │            │     │           │
    //     │ ref1(Frz)*  │     │ ref3(Res)* │     │ ref2(Res) │
    //     │             │     │            │     │           │
    //     └─────────────┘     └────────────┘     └───────────┘

    // Performs a wildcard access on the main root. However, as there are
    // no exposed tags with write permissions and a tag smaller than ref2
    // this access fails.
    *ref2 = 13; //~ ERROR: /write access through .* is forbidden/
}
