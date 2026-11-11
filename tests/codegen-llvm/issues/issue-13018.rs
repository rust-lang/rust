//@ compile-flags: -Copt-level=3

// A drop([...].clone()) sequence on an Rc should be a no-op
// In particular, no call to __rust_dealloc should be emitted
//
// We use a cdylib since it's a leaf unit for Rust purposes, so doesn't codegen -Zshare-generics
// code.
#![crate_type = "cdylib"]
use std::rc::Rc;

pub fn foo(t: &Rc<Vec<usize>>) {
    // CHECK-NOT: __rust_dealloc
    drop(t.clone());
}
