//@ run-pass
#![allow(unused_must_use)]
//@ needs-threads

use std::thread;

pub fn main() {
    let mut i = 10;
    while i > 0 {
        thread::spawn({
            let i = i;
            move || child(i)
        })
        .join();
        i = i - 1;
    }
    println!("main thread exiting");
}

fn child(x: isize) {
    println!("{}", x);
}
