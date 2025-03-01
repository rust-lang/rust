#![feature(ergonomic_clones)]

fn main() {
    async use {};
    //~^ ERROR `async use` blocks are only allowed in Rust 2018 or later
}
