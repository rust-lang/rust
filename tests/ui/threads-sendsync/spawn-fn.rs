//@ run-pass
#![allow(unused_must_use)]
//@ needs-threads

use std::thread;

fn x(s: String, n: isize) {
    println!("{}", s);
    println!("{}", n);
}

pub fn main() {
    let t1 = thread::spawn(|| x("hello from first spawned fn".to_string(), 65));
    let t2 = thread::spawn(|| x("hello from second spawned fn".to_string(), 66));
    let t3 = thread::spawn(|| x("hello from third spawned fn".to_string(), 67));
    let mut i = 30;
    while i > 0 {
        i = i - 1;
        println!("parent sleeping");
        thread::yield_now();
    }
    t1.join();
    t2.join();
    t3.join();
}
