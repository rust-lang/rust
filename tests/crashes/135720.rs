//@ known-bug: #135720
#![feature(generic_const_exprs)]
type S<'l> = [i32; A];
fn lint_me(x: S<()>) {}
