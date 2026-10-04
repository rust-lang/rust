//@ edition: 2015

fn main() {
    async move {};
    //~^ ERROR `async move` blocks are only allowed in Rust 2018 or later
}
