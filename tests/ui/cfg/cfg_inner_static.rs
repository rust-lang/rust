// run-pass
// aux-build:cfg_inner_static.rs

// pretty-expanded FIXME #23616

extern crate cfg_inner_static;

pub fn main() {
    cfg_inner_static::foo();
}
