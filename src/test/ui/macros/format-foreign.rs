// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    println!("%.*3$s %s!\n", "Hello,", "World", 4);
    println!("%1$*2$.*3$f", 123.456);

    // This should *not* produce hints, on the basis that there's equally as
    // many "correct" format specifiers.  It's *probably* just an actual typo.
    println!("{} %f", "one", 2.0);

    println!("Hi there, $NAME.", NAME="Tim");
}
