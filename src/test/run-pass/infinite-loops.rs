// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*
  A simple way to make sure threading works. This should use all the
  CPU cycles an any machines that we're likely to see for a while.
*/
// xfail-test

extern mod std;
use task::join;

fn loop(n: int) {
    let t1: task;
    let t2: task;

    if n > 0 { t1 = spawn loop(n - 1); t2 = spawn loop(n - 1); }


    loop { }
}

fn main() { let t: task = spawn loop(5); join(t); }