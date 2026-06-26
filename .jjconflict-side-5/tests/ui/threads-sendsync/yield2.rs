//@ run-pass

use std::thread;

pub fn main() {
    let mut i: isize = 0;
    while i < 100 {
        i = i + 1;
        println!("{}", i);
        thread::yield_now();
    }
}
