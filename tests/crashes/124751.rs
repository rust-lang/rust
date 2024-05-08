//@ known-bug: rust-lang/rust#124751
//@ compile-flags: -Zunstable-options --edition=2024

#![feature(gen_blocks)]

fn main() {
    let _ = async gen || {};
}
