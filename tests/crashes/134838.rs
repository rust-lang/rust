//@ known-bug: #134838
#![feature(type_ascription)]
#![allow(dead_code)]

struct Ty(());

fn mk() -> impl Sized {
    if false {
         let _ = type_ascribe!(mk(), Ty).0;
    }
    Ty(())
}

fn main() {}
