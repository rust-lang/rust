//@compile-flags: -Zmiri-tree-borrows -Zmiri-permissive-provenance

/// Checks if a local access gets correctly triggered, if we know that
/// all exposed references are local to a node.
pub fn main() {
    let mut x: u32 = 42;

    let ptr_base = &mut x as *mut u32;
    let ref1 = unsafe { &mut *ptr_base };
    let ref2 = unsafe { &mut *ptr_base };

    // Both references get exposed.
    let int1 = ref1 as *mut u32 as usize;
    let _int2 = ref2 as *mut u32 as usize;

    let wild = int1 as *mut u32;

    // Activates ptr_base.
    unsafe { wild.write(41) };

    //    ┌─────────────┐
    //    │             │
    //    │    x (Act)  │
    //    │             │
    //    └──────┬──────┘
    //           │
    //           │
    //           ▼
    //    ┌────────────────┐
    //    │                │
    //    │ ptr_base (Act) ├──────────┐
    //    │                │          │
    //    └──────┬─────────┘          │
    //           │                    │
    //           │                    │
    //           ▼                    ▼
    //    ┌────────────┐       ┌────────────┐
    //    │            │       │            │
    //    │ ref1(Res)* │       │ ref2(Res)* │
    //    │            │       │            │
    //    └────────────┘       └────────────┘

    // We read from x causing a foreign access to ptr_base, freezing it
    // (as the previous wildcard access has made it active).
    let _y = x;

    // While both exposed references are still enabled for writes, any write
    // through them would cause UB at ptr_base.
    unsafe { wild.write(0) }; //~ ERROR: /write access through <wildcard> at .* is forbidden/
}
