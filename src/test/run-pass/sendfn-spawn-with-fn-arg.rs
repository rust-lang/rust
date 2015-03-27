// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_features)]
#![feature(box_syntax)]

use std::thread;

pub fn main() { test05(); }

fn test05_start<F:FnOnce(isize)>(f: F) {
    f(22);
}

fn test05() {
    let three: Box<_> = box 3;
    let fn_to_send = move|n:isize| {
        println!("{}", *three + n); // will copy x into the closure
        assert_eq!(*three, 3);
    };
    thread::spawn(move|| {
        test05_start(fn_to_send);
    }).join().ok().unwrap();
}
