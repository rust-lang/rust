//@ edition:2015
//@ run-pass

//@ aux-build:xcrate.rs


extern crate xcrate;

fn main() {
    xcrate::test!();
}
