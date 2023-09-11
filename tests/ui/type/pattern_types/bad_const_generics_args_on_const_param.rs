//@known-bug: #127972
//@ failure-status: 101
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "thread 'rustc'.*panicked.*\n" -> ""
//@ normalize-stderr: "(error: internal compiler error: [^:]+):\d+:\d+: " -> "$1:LL:CC: "
//@ rustc-env:RUST_BACKTRACE=0

#![feature(pattern_types, pattern_type_macro, generic_const_exprs)]
#![allow(internal_features)]

type Pat<const START: u32, const END: u32> =
    std::pat::pattern_type!(u32 is START::<(), i32, 2>..=END::<_, Assoc = ()>);

fn main() {}
