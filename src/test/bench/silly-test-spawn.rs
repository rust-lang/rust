// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is (hopefully) a quick test to get a good idea about spawning
// performance in libgreen. Note that this uses the rustuv event loop rather
// than the basic event loop in order to get a better real world idea about the
// performance of a task spawn.

extern crate green;
extern crate rustuv;

#[start]
fn start(argc: int, argv: **u8) -> int {
    green::start(argc, argv, rustuv::event_loop, main)
}

fn main() {
    for _ in range(1u32, 100_000) {
        spawn(proc() {})
    }
}
