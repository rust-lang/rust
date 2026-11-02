//@ revisions: old next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
//@[old] failure-status: 101
//@[old] dont-check-compiler-stderr
//@[old] known-bug: #134838
#![feature(type_ascription)]
#![allow(dead_code)]

// Regression test for #134838. This ICEd during MIR borrowck with the
// old solver. Fixed by eagerly normalizing opaque types with the new
// solver.

struct Ty(());

fn mk() -> impl Sized {
    if false {
         let _ = type_ascribe!(mk(), Ty).0;
    }
    Ty(())
}

fn main() {}
