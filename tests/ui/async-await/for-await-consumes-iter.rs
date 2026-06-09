//@ edition: 2021
#![feature(async_iterator, async_iter_from_iter, async_for_loop)]

use std::future::Future;

// a test to make sure `for await` consumes the iterator

async fn real_main() {
    let iter = core::async_iter::from_iter(0..3);
    let mut count = 0;
    for await i in iter {
    }
    // make sure iter has been moved and we can't iterate over it again.
    for await i in iter {
        //~^ ERROR: use of moved value: `iter`
    }
}

fn main() {
}
