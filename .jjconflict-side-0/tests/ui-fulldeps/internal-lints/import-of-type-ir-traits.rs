//@ compile-flags: -Z unstable-options
//@ ignore-stage1

#![feature(rustc_private)]
#![deny(rustc::usage_of_type_ir_traits)]

extern crate rustc_type_ir;

use rustc_type_ir::Interner;

fn foo<I: Interner>(cx: I, did: I::TraitId) {
    let _ = cx.trait_is_unsafe(did);
    //~^ ERROR do not use `rustc_type_ir::Interner` or `rustc_type_ir::InferCtxtLike` unless you're inside of the trait solver
}

fn main() {}
