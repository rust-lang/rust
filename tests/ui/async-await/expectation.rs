// check-pass
// edition: 2021

use std::fmt::Debug;
use std::future::Future;

fn needs_future(_: impl Future<Output = Box<dyn Debug>>) {}

fn main() {
    needs_future(async { Box::new(()) })
}
