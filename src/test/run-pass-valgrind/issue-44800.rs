// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::alloc::System;
use std::collections::VecDeque;

#[global_allocator]
static ALLOCATOR: System = System;

fn main() {
    let mut deque = VecDeque::with_capacity(32);
    deque.push_front(0);
    deque.reserve(31);
    deque.push_back(0);
}
