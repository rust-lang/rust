//@ run-pass

#![deny(unused_mut)]
#![allow(unreachable_code)]

fn main() {
    for _ in { return (); 0..3 } {} // ok
}
