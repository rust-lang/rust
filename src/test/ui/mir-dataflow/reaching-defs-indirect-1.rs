#![feature(core_intrinsics, rustc_attrs)]
use std::intrinsics::rustc_peek;

#[rustc_mir(rustc_peek_use_def_chain, stop_after_dataflow, borrowck_graphviz_postflow="flow.dot")]
fn foo(test: bool) -> (i32, i32) {
    let mut x;
    let mut y;
    let mut p;

    x=0;
    y=1;

    unsafe { rustc_peek(x); }
    //~^ ERROR rustc_peek: [10: "x=0"]

    if test {
        p = &mut x;
    } else {
        p = &mut y;
    }

    *p=2;

    unsafe { rustc_peek(x); }
    //~^ ERROR rustc_peek: [10: "x=0", 22: "*p=2"]

    (x, y)
}

fn main() {
    foo(true);
    foo(false);
}
