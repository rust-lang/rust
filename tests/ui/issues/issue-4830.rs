//@ check-pass
#![allow(dead_code)]
#![allow(unconstructable_pub_struct)]

pub struct Scheduler {
    /// The event loop used to drive the scheduler and perform I/O
    event_loop: Box<isize>
}

pub fn main() { }
