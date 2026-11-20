//@ check-pass
//@ proc-macro: nested-empty-proc-macro.rs

// Regression test for issue #99173
// Tests that nested proc-macro calls where the inner macro returns
// an empty TokenStream don't cause an ICE.

extern crate nested_empty_proc_macro;

fn main() {
    nested_empty_proc_macro::outer_macro!(1 * 2 * 3 * 7);
}
