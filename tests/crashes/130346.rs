//@ known-bug: rust-lang/rust#130346

#![feature(non_lifetime_binders)]
#![allow(unused)]

trait A<T>: Iterator<Item = T> {}

fn demo(x: &mut impl for<U> A<U>) {
    let _: Option<u32> = x.next(); // Removing this line stops the ICE
}
