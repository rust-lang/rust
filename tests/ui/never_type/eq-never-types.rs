//@ known-bug: #120600
//
// issue: rust-lang/rust#120600

//@ failure-status: 101
//@ normalize-stderr-test: "DefId\(.*?\]::" -> "DefId("
//@ normalize-stderr-test: "(?m)note: we would appreciate a bug report.*\n\n" -> ""
//@ normalize-stderr-test: "(?m)note: rustc.*running on.*\n\n" -> ""
//@ normalize-stderr-test: "(?m)note: compiler flags.*\n\n" -> ""
//@ normalize-stderr-test: "(?m)note: delayed at.*$" -> ""
//@ normalize-stderr-test: "(?m)^ *\d+: .*\n" -> ""
//@ normalize-stderr-test: "(?m)^ *at .*\n" -> ""

#![allow(internal_features)]
#![feature(never_type, rustc_attrs)]
#![rustc_never_type_options(fallback = "never")]

fn ice(a: !) {
    a == a;
}

fn main() {}
