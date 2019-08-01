#![feature(core_intrinsics, rustc_attrs)]
use std::intrinsics::rustc_peek;

struct NoSideEffectsInDrop<'a>(&'a mut u32);
struct SideEffectsInDrop<'a>(&'a mut u32);

impl Drop for SideEffectsInDrop<'_> {
    fn drop(&mut self) {
        *self.0 = 42
    }
}

#[rustc_mir(rustc_peek_use_def_chain, stop_after_dataflow, borrowck_graphviz_postflow="flow.dot")]
fn main() {
    let mut x;
    x=0;

    NoSideEffectsInDrop(&mut x);
    SideEffectsInDrop(&mut x);

    // The ";" on line 19 is the point at which `SideEffectsInDrop` is dropped.
    unsafe { rustc_peek(x); }
    //~^ ERROR rustc_peek: [16: "x=0", 19: ";"]

    assert_eq!(x, 42);
}
