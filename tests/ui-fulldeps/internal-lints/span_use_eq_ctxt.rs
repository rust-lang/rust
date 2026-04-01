// Test the `rustc::span_use_eq_ctxt` internal lint
//@ compile-flags: -Z unstable-options

#![feature(rustc_private)]
#![deny(rustc::span_use_eq_ctxt)]
#![crate_type = "lib"]

extern crate rustc_span;
use rustc_span::Span;

pub fn f(s: Span, t: Span) -> bool {
    s.ctxt() == t.ctxt() //~ ERROR use `.eq_ctxt()` instead of `.ctxt() == .ctxt()`
}
