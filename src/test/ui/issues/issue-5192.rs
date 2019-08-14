// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

#![feature(box_syntax)]

pub trait EventLoop {
    fn dummy(&self) { }
}

pub struct UvEventLoop {
    uvio: isize
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
    event_loop: Box<dyn EventLoop+'static>,
}

impl Scheduler {

    pub fn new(event_loop: Box<dyn EventLoop+'static>) -> Scheduler {
        Scheduler {
            event_loop: event_loop,
        }
    }
}

pub fn main() {
    let _sched = Scheduler::new(box UvEventLoop::new() as Box<dyn EventLoop>);
}
