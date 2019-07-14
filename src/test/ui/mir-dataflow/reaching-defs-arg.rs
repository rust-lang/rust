#![feature(core_intrinsics, rustc_attrs)]
use std::intrinsics::rustc_peek;

#[rustc_mir(rustc_peek_use_def_chain, stop_after_dataflow, borrowck_graphviz_postflow="flow.dot")]
fn foo(test: bool, mut x: i32) -> i32 {
    if test {
        x=42;
    }

    unsafe { rustc_peek(&x); }
    //~^ ERROR rustc_peek: [5: "mut x", 7: "x=42"]

    x
}

fn main() {
    foo(true, 32);
    foo(false, 56);
}
