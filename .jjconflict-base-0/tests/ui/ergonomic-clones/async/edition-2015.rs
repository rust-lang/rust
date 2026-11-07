//@ edition:2015
#![feature(ergonomic_clones)]
#![allow(incomplete_features)]

fn main() {
    async use {};
    //~^ ERROR `async use` blocks are only allowed in Rust 2018 or later
}
