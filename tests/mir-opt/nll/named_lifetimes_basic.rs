// skip-filecheck
// Basic test for named lifetime translation. Check that we
// instantiate the types that appear in function arguments with
// suitable variables and that we setup the outlives relationship
// between R0 and R1 properly.

//@ compile-flags: -Zverbose-internals
//                ^^^^^^^^^^^^^^^^^^^ force compiler to dump more region information

#![allow(warnings)]

// EMIT_MIR named_lifetimes_basic.use_x.nll.0.mir
fn use_x<'a, 'b: 'a, 'c>(w: &'a mut i32, x: &'b u32, y: &'a u32, z: &'c u32) -> bool {
    true
}

fn main() {}
