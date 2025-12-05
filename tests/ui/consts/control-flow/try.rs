//@ check-pass

#![allow(dead_code)]
#![feature(const_trait_impl)]
#![feature(const_try)]

const fn opt() -> Option<i32> {
    let x = Some(2);
    x?;
    None
}

fn main() {}
