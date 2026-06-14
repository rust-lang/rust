// Regression test for #151025: derive-generated `#[allow(non_ascii_idents)]`
// should not be rejected as an unused attribute.

//@ check-pass
//@ proc-macro: non-ascii-idents-derive.rs

#![allow(dead_code)]
#![deny(non_ascii_idents, unused_attributes)]

extern crate non_ascii_idents_derive;

use non_ascii_idents_derive::NonAsciiIdent;

#[derive(NonAsciiIdent)]
struct S;

fn main() {}
