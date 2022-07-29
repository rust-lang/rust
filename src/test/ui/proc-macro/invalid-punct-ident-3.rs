// aux-build:invalid-punct-ident.rs
// ignore-stage1
// only-linux
//
// FIXME: This should be a normal (stage1, all platforms) test in
// src/test/ui/proc-macro once issue #59998 is fixed.

#[macro_use]
extern crate invalid_punct_ident;

invalid_raw_ident!(); //~ ERROR proc macro panicked

fn main() {}
