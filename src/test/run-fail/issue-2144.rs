// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:ran out of stack

// Don't leak when the landing pads need to request more stack
// than is allowed during normal execution

fn useBlock(f: fn~() -> uint) { useBlock(|| 22u ) }
fn main() {
    useBlock(|| 22u );
}
