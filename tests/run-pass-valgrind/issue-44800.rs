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
