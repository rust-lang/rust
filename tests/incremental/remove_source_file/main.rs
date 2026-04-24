// This test case makes sure that the compiler doesn't crash due to a failing
// table lookup when a source file is removed.

//@ revisions: bpass1 bpass2

// Note that we specify -g so that the SourceFiles actually get referenced by the
// incr. comp. cache:
//@ compile-flags: -Z query-dep-graph -g
// FIXME(#62277): could be check-pass?

#![crate_type= "rlib"]

#[cfg(bpass1)]
mod auxiliary;

#[cfg(bpass1)]
pub fn foo() {
    auxiliary::print_hello();
}

#[cfg(bpass2)]
pub fn foo() {
    println!("hello");
}
