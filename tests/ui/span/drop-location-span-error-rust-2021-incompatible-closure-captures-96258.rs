//@ edition: 2015

#![warn(rust_2021_incompatible_closure_captures)]

fn main() {}

pub(crate) struct Numberer {}

impl Numberer {
    pub(crate) async fn new(
    //~^ ERROR `async fn` is not permitted in Rust 2015
        interval: Duration,
        //~^ ERROR cannot find type `Duration` in this scope
    ) -> Numberer {
        Numberer {}
    }
}
