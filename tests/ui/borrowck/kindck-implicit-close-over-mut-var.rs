//@ run-pass

#![allow(unused_must_use)]
#![allow(dead_code)]
use std::thread;

fn user(_i: isize) {}

fn foo() {
    // Here, i is *copied* into the proc (heap closure).
    // Requires allocation.  The proc's copy is not mutable.
    let mut i = 0;
    let t = thread::spawn(move|| {
        user(i);
        println!("spawned {}", i)
    });
    i += 1;
    println!("original {}", i);
    t.join();
}

fn bar() {
    // Here, the original i has not been moved, only copied, so is still
    // mutable outside of the proc.
    let mut i = 0;
    while i < 10 {
        let t = thread::spawn(move|| {
            user(i);
        });
        i += 1;
        t.join();
    }
}

fn car() {
    // Here, i must be shadowed in the proc to be mutable.
    let mut i = 0;
    while i < 10 {
        let t = thread::spawn(move|| {
            let mut i = i;
            i += 1;
            user(i);
        });
        i += 1;
        t.join();
    }
}

pub fn main() {}
