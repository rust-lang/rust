//@ edition: 2018

#![feature(try_blocks)]

fn main() {
    while try { false } {}
    //~^ ERROR a `try` block must
}
