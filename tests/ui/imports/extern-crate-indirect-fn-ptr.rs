// issue: <https://github.com/rust-lang/rust/issues/13620>
// Test cross crate resolution of an indirect function pointer
//@ run-pass
//@ aux-build:extern-crate-indirect-fn-ptr-aux-1.rs
//@ aux-build:extern-crate-indirect-fn-ptr-aux-2.rs

extern crate extern_crate_indirect_fn_ptr_aux_2 as crate2;

fn main() {
    (crate2::FOO2.foo)();
}
