// This test has been spuriously failing a lot recently (#92000).
// Ignore it until the underlying issue is fixed.
//@ ignore-test (#92000)

// Regression test for #87481: short backtrace formatting cut off the entire stack trace.

// Codegen-units is specified here so that we can replicate a typical rustc invocation which
// is not normally limited to 1 CGU. This is important so that the `__rust_begin_short_backtrace`
// and `__rust_end_short_backtrace` symbols are not marked internal to the CGU and thus will be
// named in the symbol table.
//@ compile-flags: -O -Ccodegen-units=8

//@ run-fail
//@ check-run-results
//@ exec-env:RUST_BACKTRACE=1

// We need to normalize out frame 5 because without debug info, dbghelp.dll doesn't know where CGU
// internal functions like `main` start or end and so it will return whatever symbol happens
// to be located near the address.
//@ normalize-stderr: "5: .*" -> "5: some Rust fn"

// Backtraces are pretty broken in general on i686-pc-windows-msvc (#62897).
//@ only-x86_64-pc-windows-msvc

fn main() {
    a();
}

// Make these no_mangle so dbghelp.dll can figure out the symbol names.

#[no_mangle]
#[inline(never)]
fn a() {
    b();
}

#[no_mangle]
#[inline(never)]
fn b() {
    c();
}

#[no_mangle]
#[inline(never)]
fn c() {
    d();
}

#[no_mangle]
#[inline(never)]
fn d() {
    panic!("d was called");
}
