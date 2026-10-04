//@ proc-macro: sub-error-diag.rs

// Regression test for issue https://github.com/rust-lang/rust/issues/145305, which used to cause an ICE
// due to an assertion in the compiler that errors could not be subdiagnostics.

extern crate sub_error_diag;

//~? ERROR: Parent message
#[sub_error_diag::proc_emit_err]
//~^ ERROR: Child message
fn foo() {}

fn main() {}
