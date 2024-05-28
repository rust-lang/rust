//@ known-bug: rust-lang/rust#125059
#![feature(deref_patterns)]
#![allow(incomplete_features)]

fn simple_vec(vec: Vec<u32>) -> u32 {
   (|| match Vec::<u32>::new() {
        deref!([]) => 100,
        _ => 2000,
    })()
}

fn main() {}
