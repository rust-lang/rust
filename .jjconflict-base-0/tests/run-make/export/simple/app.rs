extern crate libr;
use libr::*;

fn main() {
    let s = m::S { x: 42 };
    assert_eq!(m::foo1(s), 42);
    assert_eq!(m::S::foo2(1), 1);
}
