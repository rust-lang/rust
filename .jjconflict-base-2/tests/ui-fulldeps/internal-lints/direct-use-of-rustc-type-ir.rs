//@ compile-flags: -Z unstable-options
//@ ignore-stage1

#![feature(rustc_private)]
#![deny(rustc::direct_use_of_rustc_type_ir)]

extern crate rustc_middle;
extern crate rustc_type_ir;

use rustc_middle::ty::*; // OK, we have to accept rustc_middle::ty::*

// We have to deny direct import of type_ir
use rustc_type_ir::*;
//~^ ERROR: do not use `rustc_type_ir` unless you are implementing type system internals

// We have to deny direct types usages which resolves to type_ir
fn foo<I: rustc_type_ir::Interner>(cx: I, did: I::DefId) {
//~^ ERROR: do not use `rustc_type_ir` unless you are implementing type system internals
}

fn main() {
    let _ = rustc_type_ir::InferConst::Fresh(42);
//~^ ERROR: do not use `rustc_type_ir` unless you are implementing type system internals
    let _: rustc_type_ir::InferConst;
//~^ ERROR: do not use `rustc_type_ir` unless you are implementing type system internals
}
