// skip-filecheck
// Basic test for liveness constraints: the region (`R1`) that appears
// in the type of `p` includes the points after `&v[0]` up to (but not
// including) the call to `use_x`. The `else` branch is not included.

//@ compile-flags:-Zverbose-internals
//                ^^^^^^^^^^^^^^^^^^^ force compiler to dump more region information

#![allow(warnings)]

fn use_x(_: usize) -> bool {
    true
}

// EMIT_MIR_FOR_EACH_BIT_WIDTH
// EMIT_MIR region_subtyping_basic.main.nll.0.mir
fn main() {
    let mut v = [1, 2, 3];
    let p = &v[0];
    let q = p;
    if true {
        use_x(*q);
    } else {
        use_x(22);
    }
}
