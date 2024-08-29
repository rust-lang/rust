//@ known-bug: rust-lang/rust#125680
//@ edition:2021

#![feature(generic_const_exprs)]

use core::fmt::Debug;

struct Inline<T>
where
    [(); std::mem::offset_of!((T,), 0)]:,
{}

fn main() {
    let dst = Inline::<dyn Debug>::new(0); // BANG!
}
