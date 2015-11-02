// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::{env, fmt, process, sync, thread};

struct SlowFmt(u32);
impl fmt::Debug for SlowFmt {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        thread::sleep_ms(3);
        self.0.fmt(f)
    }
}

fn do_print(x: u32) {
    let x = SlowFmt(x);
    println!("{:?}{:?}{:?}{:?}{:?}", x, x, x, x, x);
}

fn main(){
    if env::args().count() == 2 {
        let barrier = sync::Arc::new(sync::Barrier::new(2));
        let tbarrier = barrier.clone();
        let t = thread::spawn(move || {
            tbarrier.wait();
            do_print(1);
        });
        barrier.wait();
        do_print(2);
        t.join();
    } else {
        let this = env::args().next().unwrap();
        let output = process::Command::new(this).arg("-").output().unwrap();
        for line in String::from_utf8(output.stdout).unwrap().lines() {
            match line.chars().next().unwrap() {
                '1' => assert_eq!(line, "11111"),
                '2' => assert_eq!(line, "22222"),
                chr => panic!("unexpected character {:?}", chr)
            }
        }
    }
}
