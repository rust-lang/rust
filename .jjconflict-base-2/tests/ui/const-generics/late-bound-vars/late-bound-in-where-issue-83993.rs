//@ known-bug: unknown
// see comment on `tests/ui/const-generics/late-bound-vars/simple.rs`

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn bug<'a>()
where
    for<'b> [(); {
        let x: &'b ();
        0
    }]:
{}

fn main() {}
