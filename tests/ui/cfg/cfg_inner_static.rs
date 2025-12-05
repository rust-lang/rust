//@ run-pass
//@ aux-build:cfg_inner_static.rs


extern crate cfg_inner_static;

pub fn main() {
    cfg_inner_static::foo();
}
