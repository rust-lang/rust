//@ check-pass
//@ edition: 2021

// This test case is meant to demonstrate how close we can get to async
// functions in dyn traits with the current level of dyn* support.

#![feature(dyn_star)]
#![allow(incomplete_features)]

use std::future::Future;

trait DynAsyncCounter {
    fn increment<'a>(&'a mut self) -> dyn* Future<Output = usize> + 'a;
}

struct MyCounter {
    count: usize,
}

impl DynAsyncCounter for MyCounter {
    fn increment<'a>(&'a mut self) -> dyn* Future<Output = usize> + 'a {
        Box::pin(async {
            self.count += 1;
            self.count
        })
    }
}

async fn do_counter(counter: &mut dyn DynAsyncCounter) -> usize {
    counter.increment().await
}

fn main() {
    let mut counter = MyCounter { count: 0 };
    let _ = do_counter(&mut counter);
}
