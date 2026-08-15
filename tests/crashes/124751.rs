//@ known-bug: rust-lang/rust#124751
//@ edition: 2024

#![feature(gen_blocks)]

fn main() {
    let _ = async gen || {};
}
