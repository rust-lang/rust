//@ run-pass
//@ aux-build:issue-2380.rs


extern crate a;

pub fn main() {
    a::f::<()>();
}
