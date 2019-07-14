#![feature(core_intrinsics, rustc_attrs)]
use std::intrinsics::rustc_peek;

#[rustc_mir(rustc_peek_use_def_chain, stop_after_dataflow, borrowck_graphviz_postflow="flow.dot")]
fn main() {
    let mut x;
    x=0;

    while x != 10 {
        x+=1;
    }

    unsafe { rustc_peek(x); }
    //~^ ERROR rustc_peek: [7: "x=0", 10: "x+=1"]
}
