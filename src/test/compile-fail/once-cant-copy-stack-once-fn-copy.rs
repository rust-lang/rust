// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Though it should be legal to copy a heap-allocated "once fn:Copy",
// stack closures are not deep-copied, so (counterintuitively) it should be
// illegal to copy them.

fn foo<'r>(blk: &'r once fn:Copy()) -> (&'r once fn:Copy(), &'r once fn:Copy()) {
    (copy blk, blk) //~ ERROR copying a value of non-copyable type
}

fn main() {
}
