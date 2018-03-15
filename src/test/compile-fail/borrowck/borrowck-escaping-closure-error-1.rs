// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::thread::spawn;

// Test that we give a custom error (E0373) for the case where a
// closure is escaping current frame, and offer a suggested code edit.
// I refrained from including the precise message here, but the
// original text as of the time of this writing is:
//
//    closure may outlive the current function, but it borrows `books`,
//    which is owned by the current function

fn main() {
    let mut books = vec![1,2,3];
    spawn(|| books.push(4));
    //~^ ERROR E0373
    //~| NOTE `books` is borrowed here
    //~| NOTE may outlive borrowed value `books`
}
