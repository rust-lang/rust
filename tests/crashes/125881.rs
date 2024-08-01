//@ known-bug: rust-lang/rust#125881
#![crate_type = "lib"]
#![feature(transmutability)]
#![feature(unboxed_closures,effects)]

const fn test() -> impl std::mem::BikeshedIntrinsicFrom() {
    || {}
}
