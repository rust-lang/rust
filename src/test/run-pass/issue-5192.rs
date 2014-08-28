// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


pub trait EventLoop {
}

pub struct UvEventLoop {
    uvio: int
}

impl UvEventLoop {
    pub fn new() -> UvEventLoop {
        UvEventLoop {
            uvio: 0
        }
    }
}

impl EventLoop for UvEventLoop {
}

pub struct Scheduler {
    event_loop: Box<EventLoop+'static>,
}

impl Scheduler {

    pub fn new(event_loop: Box<EventLoop+'static>) -> Scheduler {
        Scheduler {
            event_loop: event_loop,
        }
    }
}

pub fn main() {
    let _sched = Scheduler::new(box UvEventLoop::new() as Box<EventLoop>);
}
