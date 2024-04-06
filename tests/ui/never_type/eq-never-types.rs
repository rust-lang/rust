//@ check-pass
//
// issue: rust-lang/rust#120600

#![allow(internal_features)]
#![feature(never_type, rustc_attrs)]
#![rustc_never_type_options(fallback = "never")]

fn ice(a: !) {
    a == a;
}

fn main() {}
