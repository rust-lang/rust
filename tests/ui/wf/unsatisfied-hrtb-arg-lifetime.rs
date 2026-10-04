//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Arguments of HRTBs must satisfy the lifetime bounds declared
// on the argument types. We currently emit a FCW for such
// code, as emitting a hard error would cause substantial
// crater breakage.
// See <https://github.com/rust-lang/rust/issues/162200>.

#![allow(dead_code)]

struct W<'a, 'b, T: 'b>(&'a (), &'b T);

fn warns<'b, T>(_: &'b ()) where for<'a> W<'a, 'b, T>: Sized {}
//~^ WARN the type parameter `T` may not live long enough
//~| WARN this was previously accepted by the compiler
//[next]~| WARN the type parameter `T` may not live long enough
//[next]~| WARN this was previously accepted by the compiler

// Adding the required bound silences the lint.
fn fixed<'b, T: 'b>(_: &'b ()) where for<'a> W<'a, 'b, T>: Sized {}

fn main() {}
