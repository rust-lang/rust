//@ run-pass
#![allow(dead_code)]

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
    let _sched = Scheduler::new(Box::new(UvEventLoop::new()) as Box<dyn EventLoop>);
}
