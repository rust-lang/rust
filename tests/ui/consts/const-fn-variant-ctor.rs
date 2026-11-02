//@ check-pass
// @ needs-rustc-debug-assertions
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![feature(const_destruct, const_trait_impl)]

use std::marker::Destruct;

const fn impls_fn_once<F: [const] FnOnce(u32) -> Enum + [const] Destruct>(_: F) {}

enum Enum {
    Variant(u32),
}

const fn test() {
    impls_fn_once(Enum::Variant);
}

fn main() {}
