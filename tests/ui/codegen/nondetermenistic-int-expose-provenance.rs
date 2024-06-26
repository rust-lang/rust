//@ run-pass
//@ known-bug: #107975

#![feature(exposed_provenance)]

use std::ptr::addr_of;

fn main() {
    let a: usize = {
        let v = 0u8;
        addr_of!(v).expose_provenance()
    };
    let b: usize = {
        let v = 0u8;
        addr_of!(v).expose_provenance()
    };
    let i: usize = a - b;
    assert_ne!(i, 0);
    println!("{}", i);
    assert_eq!(i, 0);
}
