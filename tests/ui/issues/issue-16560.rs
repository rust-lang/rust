//@ run-pass
#![allow(unused_variables)]
//@ needs-threads

use std::thread;
use std::mem;

fn main() {
    let y = 0u8;
    let closure = move |x: u8| y + x;

    // Check that both closures are capturing by value
    assert_eq!(1, mem::size_of_val(&closure));

    thread::spawn(move|| {
        let ok = closure;
    }).join().ok().unwrap();
}
