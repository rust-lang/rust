// run-pass

#![allow(stable_features)]
#![allow(unused_must_use)]

// ignore-emscripten no threads support

#![feature(std_panic)]

use std::panic;
use std::thread;

fn a() {
    panic::set_hook(Box::new(|_| println!("hello yes this is a")));
    panic::take_hook();
    panic::set_hook(Box::new(|_| println!("hello yes this is a part 2")));
    panic::take_hook();
}

fn b() {
    panic::take_hook();
    panic::take_hook();
    panic::take_hook();
    panic::take_hook();
    panic::take_hook();
    panic!();
}

fn c() {
    panic::set_hook(Box::new(|_| ()));
    panic::set_hook(Box::new(|_| ()));
    panic::set_hook(Box::new(|_| ()));
    panic::set_hook(Box::new(|_| ()));
    panic::set_hook(Box::new(|_| ()));
    panic::set_hook(Box::new(|_| ()));
    panic!();
}

fn main() {
    for _ in 0..10 {
        let mut handles = vec![];
        for _ in 0..10 {
            handles.push(thread::spawn(a));
        }
        for _ in 0..10 {
            handles.push(thread::spawn(b));
        }
        for _ in 0..10 {
            handles.push(thread::spawn(c));
        }
        for handle in handles {
            let _ = handle.join();
        }
    }
}
