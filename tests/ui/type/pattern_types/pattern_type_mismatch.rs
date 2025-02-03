//! Check that pattern types patterns must be of the type of the base type

//@ known-bug: unknown
//@ failure-status: 101
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "thread 'rustc' panicked.*\n" -> ""
//@ normalize-stderr: "(error: internal compiler error: [^:]+):\d+:\d+: " -> "$1:LL:CC: "
//@ normalize-stderr: "(delayed at compiler/rustc_mir_transform/src/lib.rs:)\d+:\d+" -> "$1:LL:CC"
//@ rustc-env:RUST_BACKTRACE=0

#![feature(pattern_types)]
#![feature(pattern_type_macro)]

use std::pat::pattern_type;

const BAD_NESTING4: pattern_type!(u8 is 'a'..='a') = todo!();

const BAD_NESTING5: pattern_type!(char is 1..=1) = todo!();

fn main() {}
