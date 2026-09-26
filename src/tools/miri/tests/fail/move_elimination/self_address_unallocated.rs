//@revisions: normal raw reference
//@[normal]check-pass
//@[raw]compile-flags: -Zmir-move-elimination
//@[reference]compile-flags: -Zmir-move-elimination

#![feature(core_intrinsics, custom_mir)]

use std::intrinsics::mir::*;

// Evaluating the destination must not allocate the source before taking its address.
#[cfg(any(normal, raw))]
struct Node {
    ptr: *const Node,
}

#[cfg(reference)]
struct Node {
    ptr: &'static Node,
}

#[cfg(any(normal, raw))]
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let value: Node;
        {
            value.ptr = &raw const value; //~[raw] ERROR: live but unallocated
            Return()
        }
    }
}

#[cfg(reference)]
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let value: Node;
        {
            value.ptr = &value; //~[reference] ERROR: live but unallocated
            Return()
        }
    }
}
