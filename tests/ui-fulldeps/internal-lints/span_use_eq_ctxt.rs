// compile-flags: -Z unstable-options

#![feature(rustc_private)]
#![deny(rustc::span_use_eq_ctxt)]

extern crate rustc_span;
use rustc_span::Span;

pub fn f(s: Span, t: Span) -> bool {
    s.ctxt() == t.ctxt() //~ ERROR use of span ctxt
}

fn main() {}
