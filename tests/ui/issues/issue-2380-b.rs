// run-pass
// aux-build:issue-2380.rs

// pretty-expanded FIXME #23616

extern crate a;

pub fn main() {
    a::f::<()>();
}
