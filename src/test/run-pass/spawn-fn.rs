// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::task;

fn x(s: String, n: int) {
    println!("{:?}", s);
    println!("{:?}", n);
}

pub fn main() {
    task::spawn(proc() x("hello from first spawned fn".to_string(), 65) );
    task::spawn(proc() x("hello from second spawned fn".to_string(), 66) );
    task::spawn(proc() x("hello from third spawned fn".to_string(), 67) );
    let mut i: int = 30;
    while i > 0 {
        i = i - 1;
        println!("parent sleeping");
        task::deschedule();
    }
}
