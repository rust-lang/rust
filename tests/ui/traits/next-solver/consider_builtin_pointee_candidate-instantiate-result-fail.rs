//@ compile-flags: -Znext-solver

//! This test is extremely similar to `nested-rerun-not-erased-in-normalizes-to.rs`, however,
//! instead of the eager normalization failing due to an anon const when in ErasedNotCoherence, this
//! just straight up fails normalization because u32 is not Iterator

#![feature(ptr_metadata)]

struct ThisStructAintValid(<u32 as Iterator>::Item);
//~^ ERROR `u32` is not an iterator

fn main() {
    let y: <ThisStructAintValid as std::ptr::Pointee>::Metadata;
    //~^ ERROR type mismatch resolving `<ThisStructAintValid as Pointee>::Metadata == _`
}
