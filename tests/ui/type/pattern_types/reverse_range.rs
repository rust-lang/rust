//! Check that the range start must be smaller than the range end
//@ known-bug: unknown
//@ failure-status: 101
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "thread 'rustc' panicked.*\n" -> ""
//@ normalize-stderr: "(error: internal compiler error: [^:]+):\d+:\d+: " -> "$1:LL:CC: "
//@ rustc-env:RUST_BACKTRACE=0

#![feature(pattern_types)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

const NONE: pattern_type!(u8 is 1..0) = unsafe { std::mem::transmute(3_u8) };
