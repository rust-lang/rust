// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #5192

#![allow(unknown_features)]
#![feature(box_syntax)]

pub trait EventLoop { fn foo(&self) {} }

pub struct UvEventLoop {
    uvio: int
}

impl EventLoop for UvEventLoop { }

pub fn main() {
    let loop_: Box<EventLoop> = box UvEventLoop { uvio: 0 } as Box<EventLoop>;
    let _loop2_ = loop_;
}
