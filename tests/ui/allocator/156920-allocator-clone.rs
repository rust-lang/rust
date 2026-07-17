#![feature(allocator_api)]

use std::{
    alloc::{Allocator, Global, System},
    sync::Arc,
};

pub trait MyAlloc: Allocator {}
impl<A: Allocator> MyAlloc for A {}

impl Clone for Box<dyn MyAlloc> {
    fn clone(&self) -> Self {
        Box::new(System)
    }
}

fn main() {
    let evil_arc: Arc<i32, Box<dyn MyAlloc>> = Arc::new_in(69420, Box::new(Global));
    // clone() returns `System` instead of `Global`, but `Arc` requires equivalent allocators
    // Make sure to not accidentally call <&Arc>::clone here.
    let _ = <Arc<_, _> as Clone>::clone(&evil_arc);
    //~^ ERROR: the trait bound `dyn MyAlloc: AllocatorClone` is not satisfied [E0277]
}
