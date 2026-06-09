//@ compile-flags:-g -Copt-level=0 -C panic=abort

// Check that debug information exists for structures containing loops (cyclic references).
// Previously it may incorrectly prune member information during recursive type inference check.

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "Arc<debuginfo_cyclic_structure::Inner<alloc::sync::Arc<debuginfo_cyclic_structure::Handle{{.*}}elements: ![[FIELDS:[0-9]+]]
// CHECK: ![[FIELDS]] = !{!{{.*}}}
// CHECK-NOT: ![[FIELDS]] = !{}

#![crate_type = "lib"]

use std::mem::MaybeUninit;
use std::sync::Arc;

struct Inner<T> {
    buffer: Box<MaybeUninit<T>>,
}
struct Shared {
    shared: Arc<Inner<Arc<Handle>>>,
}
struct Handle {
    shared: Shared,
}
struct Core {
    inner: Arc<Inner<Arc<Handle>>>,
}

#[no_mangle]
extern "C" fn test() {
    let с = Core { inner: Arc::new(Inner { buffer: Box::new(MaybeUninit::uninit()) }) };
    std::hint::black_box(с);
}
