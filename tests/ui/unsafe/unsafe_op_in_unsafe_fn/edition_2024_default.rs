//@ edition: 2024
//@ check-pass

// Tests that `unsafe_op_in_unsafe_fn` is warn-by-default in edition 2024 and that the
// `unused_unsafe` lint does not consider the inner unsafe block to be unused.
#![crate_type = "lib"]
#![deny(unused_unsafe)]

unsafe fn unsf() {}

unsafe fn foo() {
    unsf();
    //~^ WARN

    // no unused_unsafe
    unsafe {
        unsf();
    }
}
