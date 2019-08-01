// General test of reaching_defs state computed by MIR dataflow.

#![feature(core_intrinsics, rustc_attrs)]
use std::intrinsics::rustc_peek;

#[rustc_mir(rustc_peek_use_def_chain, stop_after_dataflow, borrowck_graphviz_postflow="flow.dot")]
fn foo(test: bool) -> (i32, i32) {
    // Splitting declarations and assignment gives us nicer spans
    let mut x;
    let mut y;

    x=0;
    y=1;

    if test {
        x=2;
        unsafe { rustc_peek(&x); }
        //~^ ERROR rustc_peek: [16: "x=2"]
    } else {
        x=3;
        y=4;
    }

    unsafe { rustc_peek(&x); }
    //~^ ERROR rustc_peek: [16: "x=2", 20: "x=3"]

    unsafe { rustc_peek(&y); }
    //~^ ERROR rustc_peek: [13: "y=1", 21: "y=4"]

    (x, y)
}

fn main() {
    foo(true);
    foo(false);
}
