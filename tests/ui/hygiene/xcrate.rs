//@ run-pass

//@ aux-build:xcrate.rs

#![feature(decl_macro)]

extern crate xcrate;

fn main() {
    xcrate::test!();
}
