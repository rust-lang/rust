// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(thread_local)]

#[thread_local]
static FOO: u8 = 3;

fn main() {
    let a = &FOO;
    //~^ ERROR borrowed value does not live long enough
    //~| does not live long enough
    //~| NOTE borrowed value must be valid for the static lifetime

    std::thread::spawn(move || {
        println!("{}", a);
    });
}
//~^ NOTE temporary value only lives until here
