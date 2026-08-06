// Regression test for failure #4 in <https://github.com/rust-lang/rust/issues/155924>.
//
// This checks that when we are checking the sized-ness of an inference variable
// (here coming from the never-to-any coercion) we consider subtyping. That is,
// `?0` is sized if `(?0 <: ?1 || ?0 :> ?1) && ?1: Sized`).
//
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![feature(never_type)]

fn blah(e: !) {
    let source = Box::new(e);
    let _: Box<dyn Send> = source;
}

fn main() {}
