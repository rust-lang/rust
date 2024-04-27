// This test case makes sure that the compiler doesn't crash due to a failing
// table lookup when a source file is removed.

//@ revisions:cfail1 cfail2

// Note that we specify -g so that the SourceFiles actually get referenced by the
// incr. comp. cache:
//@ compile-flags: -Z query-dep-graph -g
//@ build-pass (FIXME(62277): could be check-pass?)

#![crate_type= "rlib"]

#[cfg(cfail1)]
mod auxiliary;

#[cfg(cfail1)]
pub fn foo() {
    auxiliary::print_hello();
}

#[cfg(cfail2)]
pub fn foo() {
    println!("hello");
}
