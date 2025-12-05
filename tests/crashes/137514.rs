//@ known-bug: #137514
//@ needs-rustc-debug-assertions
#![feature(generic_const_exprs)]

trait Bar<const N: usize> {}

trait BB = Bar<{ 1i32 + 1 }>;

fn foo(x: &dyn BB) {}
