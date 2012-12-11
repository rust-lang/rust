// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn send<T: Owned>(ch: _chan<T>, -data: T) {
    log(debug, ch);
    log(debug, data);
    fail;
}

enum _chan<T> = int;

// Tests that "log(debug, message);" is flagged as using
// message after the send deinitializes it
fn test00_start(ch: _chan<int>, message: int, _count: int) {
    send(ch, move message); //~ NOTE move of variable occurred here
    log(debug, message); //~ ERROR use of moved variable: `message`
}

fn main() { fail; }
