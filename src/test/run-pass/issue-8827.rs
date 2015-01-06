// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::thread::Thread;
use std::sync::mpsc::{channel, Receiver};

fn periodical(n: int) -> Receiver<bool> {
    let (chan, port) = channel();
    Thread::spawn(move|| {
        loop {
            for _ in range(1, n) {
                match chan.send(false) {
                    Ok(()) => {}
                    Err(..) => break,
                }
            }
            match chan.send(true) {
                Ok(()) => {}
                Err(..) => break
            }
        }
    });
    return port;
}

fn integers() -> Receiver<int> {
    let (chan, port) = channel();
    Thread::spawn(move|| {
        let mut i = 1;
        loop {
            match chan.send(i) {
                Ok(()) => {}
                Err(..) => break,
            }
            i = i + 1;
        }
    });
    return port;
}

fn main() {
    let ints = integers();
    let threes = periodical(3);
    let fives = periodical(5);
    for _ in range(1i, 100i) {
        match (ints.recv().unwrap(), threes.recv().unwrap(), fives.recv().unwrap()) {
            (_, true, true) => println!("FizzBuzz"),
            (_, true, false) => println!("Fizz"),
            (_, false, true) => println!("Buzz"),
            (i, false, false) => println!("{}", i)
        }
    }
}
