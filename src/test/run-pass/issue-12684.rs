// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast

extern crate native;
extern crate green;
extern crate rustuv;

#[start]
fn start(argc: int, argv: **u8) -> int { green::start(argc, argv, main) }

fn main() {
    native::task::spawn(proc() customtask());
}

fn customtask() {
    let mut timer = std::io::timer::Timer::new().unwrap();
    let periodic = timer.periodic(10);
    periodic.recv();
}
